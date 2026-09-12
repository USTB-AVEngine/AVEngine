"""Bounded CPU/GPU resource allocation for V1 production stages.

A stage name does not decide what a work item occupies; the backend it
actually runs on does. Habitat and SPEAR capture open a visual world on a
graphics device, so they hold a GPU reservation. RLR audio propagation creates
a native context and can run entirely on CPU cores, so it holds CPU capacity
and no GPU quota at all. Delivery is ordinary CPU work. That is why this
module carries its own small vocabulary instead of a stage-to-device table:
the caller states the backend, and the backend states the device.

Three counts are kept apart on purpose, because collapsing them is how a
scheduler silently oversubscribes a machine: a *native context* is an
initialised runtime inside one process, a *visual world* is a loaded scene on
a graphics device, and a *render job* is one frame transaction against that
world. A CPU audio worker has a native context, no visual world and no render
job.

Concurrency is one budget with two layers, never a product. Every granted
lease occupies exactly one worker slot out of cpu.max_workers; a lease
whose backend needs a graphics device *additionally* occupies one of
gpu.max_workers. gpu.max_workers may therefore never exceed
cpu.max_workers, and the two numbers cannot multiply into more processes
than the machine was given.

Sharing a GPU with other people is allowed and expected on this host. The
allocator reads free VRAM, never utilisation, never process ownership; it
never stops, signals or waits for a foreign process. A device is a candidate
when the memory this run needs still fits beside whatever is already there.

Reservations are coordinated inside one run so two pending work items cannot
each spend the same free megabytes. A reservation holds back its full estimate
until the worker's real usage is observed through the driver, after which only
the unrealised remainder is held back. Between grant and launch the driver is
read again, because a foreign process may have taken the memory in between.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from avengine.rooms.room_package import (
    RENDERER_RUNTIME_KEYS,
    RENDERERS_REQUIRING_FRESH_INTERPRETER,
    RUNTIME_ISOLATION_KEYS,
    RUNTIME_KEY_ALIASES,
)

SCHEMA = "avengine_v1_production_resources_v1"

# The two capacity layers. A lease is on exactly one lane; a "gpu" lease still
# occupies a worker slot, so the layers add up instead of multiplying.
LANES = ("cpu", "gpu")

# Runtime parameters that bind native shared libraries into an interpreter.
# They cannot be changed after import, so two work items that disagree on any
# of them need different processes. The room route already owns this table, so
# it is read from there rather than restated: one place decides what makes two
# rooms incompatible. Room family is deliberately absent, because it is
# routing data: two families on the same prefix share a worker fine, and one
# family can span two prefixes.
PROCESS_IDENTITY_RUNTIME_KEYS = RUNTIME_ISOLATION_KEYS
# Runtime parameters that vary per instance and never force a new process.
PER_INSTANCE_RUNTIME_KEYS = {
    "habitat": ("mp3d_root", "graphics_adapter"),
    "ue_spear": ("graphics_adapter", "rpc_port", "streaming_warmup_frames",
                 "ddc_profile", "ddc_directory"),
}

# Native runtimes that abort or silently keep the first library once loaded.
# A parent holding any of these must not fork a worker for another prefix.
NATIVE_RUNTIME_MODULES = (
    "habitat_sim",
    "magnum",
    "corrade",
    "rlr_audio_propagation",
    "spear",
)

DECISION_STATUSES = ("granted", "wait", "blocked")
TERMINAL_DEVICE_REASON_CODES = frozenset({
    "gpu_inventory_empty",
    "pinned_device_not_available",
    "no_configured_device_can_ever_fit",
})


class ResourcePolicyError(ValueError):
    """The resource configuration cannot be executed as written."""


class ResourceUnavailable(RuntimeError):
    """A bounded wait ended without the resource; carries the exact reason."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _text(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourcePolicyError(f"{owner} must be a nonempty string")
    return value.strip()


def _positive_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResourcePolicyError(f"{owner} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourcePolicyError(f"{owner} must be a nonnegative integer")
    return int(value)


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ResourcePolicyError(f"{owner} must be a mapping")
    return dict(value)


# ---------------------------------------------------------------------------
# Which process can run this work item
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerCompatibility:
    """The real inputs that decide whether two work items can share a process.

    Built from the same runtime mapping the room route already resolves, so a
    new backend key is declared once in ``RENDERER_RUNTIME_KEYS`` rather than
    twice. ``routing`` (family, room_id, task family) is carried for logs and
    is explicitly not part of the key.
    """

    python_executable: str
    renderer: str
    identity: tuple[tuple[str, str], ...] = ()
    # Which native runtime the process initialises. RLR must be imported
    # before Habitat, so an acoustic worker and a geometry worker on the same
    # prefix are still two processes, not one.
    runtime_context: str = ""

    def __post_init__(self) -> None:
        _text(self.python_executable, "worker python_executable")
        renderer = _text(self.renderer, "worker renderer")
        if renderer not in PROCESS_IDENTITY_RUNTIME_KEYS:
            raise ResourcePolicyError(
                f"renderer {renderer!r} has no declared process identity keys; "
                f"declared renderers are {sorted(PROCESS_IDENTITY_RUNTIME_KEYS)}"
            )

    @classmethod
    def from_runtime(
        cls,
        runtime: Mapping[str, Any],
        *,
        renderer: str,
        python_executable: str | None = None,
        runtime_context: str | None = None,
    ) -> "WorkerCompatibility":
        """Read the identity-bearing runtime values for one renderer."""

        data = _mapping(runtime, "runtime")
        canonical: dict[str, Any] = {}
        for key, value in data.items():
            canonical[RUNTIME_KEY_ALIASES.get(str(key), str(key))] = value
        renderer = _text(renderer, "renderer")
        if renderer not in PROCESS_IDENTITY_RUNTIME_KEYS:
            raise ResourcePolicyError(
                f"renderer {renderer!r} has no declared process identity keys"
            )
        identity: list[tuple[str, str]] = []
        for name in PROCESS_IDENTITY_RUNTIME_KEYS[renderer]:
            value = canonical.get(name)
            if value in (None, ""):
                continue
            identity.append((name, str(Path(str(value)).expanduser())))
        return cls(
            python_executable=str(python_executable or sys.executable),
            renderer=renderer,
            identity=tuple(sorted(identity)),
            runtime_context="" if runtime_context is None else str(runtime_context),
        )

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.python_executable, self.renderer, self.runtime_context,
                self.identity)

    def compatible_with(self, other: "WorkerCompatibility") -> bool:
        return self.key == other.key

    @classmethod
    def from_runtime_report(
        cls,
        runtime_report: Mapping[str, Any],
        *,
        python_executable: str | None = None,
        runtime_context: str | None = None,
    ) -> "WorkerCompatibility":
        """Build the process identity straight from a resolved room runtime.

        ``resolve_room_runtime`` already reports the renderer, the effective
        values and which of them isolate a process, so a caller that resolved
        a room does not restate any of it here.
        """

        report = _mapping(runtime_report, "runtime_report")
        renderer = _text(report.get("renderer"), "runtime_report.renderer")
        return cls.from_runtime(
            _mapping(report.get("effective"), "runtime_report.effective"),
            renderer=renderer,
            python_executable=python_executable,
            runtime_context=runtime_context,
        )

    @property
    def requires_fresh_interpreter(self) -> bool:
        """Whether this renderer may never be reached by forking a parent."""

        return self.renderer in RENDERERS_REQUIRING_FRESH_INTERPRETER

    @classmethod
    def from_dict(cls, value: Any) -> "WorkerCompatibility":
        data = _mapping(value, "compatibility")
        identity = _mapping(data.get("identity"), "compatibility.identity")
        return cls(
            python_executable=_text(
                data.get("python_executable"), "compatibility.python_executable"),
            renderer=_text(data.get("renderer"), "compatibility.renderer"),
            identity=tuple(sorted((str(k), str(v)) for k, v in identity.items())),
            runtime_context=str(data.get("runtime_context") or ""),
        )

    def missing_identity_keys(self) -> tuple[str, ...]:
        present = {name for name, _ in self.identity}
        required = RENDERER_RUNTIME_KEYS[self.renderer]["required"]
        declared = PROCESS_IDENTITY_RUNTIME_KEYS[self.renderer]
        return tuple(
            name for name in declared if name in required and name not in present
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "renderer": self.renderer,
            "runtime_context": self.runtime_context,
            "identity": {name: value for name, value in self.identity},
        }


def classified_runtime_keys(renderer: str) -> dict[str, tuple[str, ...]]:
    """Report how one renderer's runtime keys are classified, plus leftovers.

    ``unclassified`` is the honest part: if a renderer grows a runtime key,
    this says so instead of quietly treating it as per-instance.
    """

    contract = RENDERER_RUNTIME_KEYS[renderer]
    known = tuple(contract["required"]) + tuple(contract["optional"])
    identity = PROCESS_IDENTITY_RUNTIME_KEYS.get(renderer, ())
    per_instance = PER_INSTANCE_RUNTIME_KEYS.get(renderer, ())
    unclassified = tuple(
        key for key in known if key not in identity and key not in per_instance
    )
    return {
        "renderer": renderer,
        "process_identity": tuple(identity),
        "per_instance": tuple(per_instance),
        "unclassified": unclassified,
    }


def native_runtime_modules_loaded(modules: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Name the native runtimes already imported into this interpreter."""

    loaded = sys.modules if modules is None else modules
    found = []
    for name in loaded:
        root = str(name).split(".", 1)[0]
        if root in NATIVE_RUNTIME_MODULES and root not in found:
            found.append(root)
    return tuple(sorted(found))


def assert_clean_worker_parent(modules: Mapping[str, Any] | None = None) -> None:
    """Refuse to hand out a fork-based worker from a native-loaded parent.

    Habitat keeps the first prefix it loaded, so a forked child cannot switch
    to another one and reporting isolation from such a fork would be false.
    Workers are launched as fresh interpreters instead.
    """

    loaded = native_runtime_modules_loaded(modules)
    if loaded:
        raise ResourceUnavailable(
            "native_runtime_already_loaded",
            "this interpreter already imported "
            f"{', '.join(loaded)}; a worker for another runtime prefix must be "
            "a fresh interpreter, not a fork of this process",
            loaded_modules=list(loaded),
        )


# ---------------------------------------------------------------------------
# What a work item occupies, decided by its backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendRequirement:
    """One backend's real occupancy, kept separate from the stage that uses it."""

    backend_id: str
    lane: str
    # How much this backend is expected to use at its peak. This is what the
    # run holds back from other pending workers, so lowering it silently is
    # how a shared device gets oversubscribed.
    estimated_peak_vram_mb: int | None = None
    # How much a device must already report free before this backend is even
    # considered. A start-up floor, never a substitute for the peak: a small
    # floor does not make a large worker small.
    min_free_vram_mb: int | None = None
    native_context: bool = False
    visual_worlds: int = 0
    render_jobs: int = 0
    needs_rpc_port: bool = False
    cpu_threads: int = 1
    memory_mb: int = 0
    io_weight: float = 1.0

    def __post_init__(self) -> None:
        _text(self.backend_id, "backend_id")
        if self.lane not in LANES:
            raise ResourcePolicyError(
                f"backend {self.backend_id!r} lane must be one of {LANES}"
            )
        _positive_int(self.cpu_threads, f"backend {self.backend_id} cpu_threads")
        _nonnegative_int(self.memory_mb, f"backend {self.backend_id} memory_mb")
        _nonnegative_int(self.visual_worlds, f"backend {self.backend_id} visual_worlds")
        _nonnegative_int(self.render_jobs, f"backend {self.backend_id} render_jobs")
        if self.io_weight < 0.0:
            raise ResourcePolicyError(
                f"backend {self.backend_id} io_weight must not be negative"
            )
        if self.lane == "gpu":
            if self.estimated_peak_vram_mb is None:
                raise ResourcePolicyError(
                    f"backend {self.backend_id!r} occupies a graphics device, so "
                    "it must declare estimated_peak_vram_mb"
                )
            _positive_int(
                self.estimated_peak_vram_mb,
                f"backend {self.backend_id} estimated_peak_vram_mb",
            )
            if self.min_free_vram_mb is not None:
                _positive_int(
                    self.min_free_vram_mb,
                    f"backend {self.backend_id} min_free_vram_mb",
                )
        else:
            for name in ("estimated_peak_vram_mb", "min_free_vram_mb"):
                if getattr(self, name) is not None:
                    raise ResourcePolicyError(
                        f"backend {self.backend_id!r} runs on CPU, so it must not "
                        f"declare {name}. Creating a native context is "
                        "not by itself a graphics-device requirement"
                    )
        if self.visual_worlds and self.lane != "gpu":
            raise ResourcePolicyError(
                f"backend {self.backend_id!r} declares a visual world but no "
                "graphics device"
            )
        if self.render_jobs and not self.visual_worlds:
            raise ResourcePolicyError(
                f"backend {self.backend_id!r} declares render jobs without a "
                "visual world to render"
            )

    @property
    def holds_gpu(self) -> bool:
        return self.lane == "gpu"

    @classmethod
    def from_mapping(cls, value: Any, *, backend_id: str) -> "BackendRequirement":
        data = _mapping(value, f"backends.{backend_id}")
        vram = data.get("estimated_peak_vram_mb")
        floor = data.get("min_free_vram_mb")
        return cls(
            backend_id=backend_id,
            lane=_text(data.get("lane"), f"backends.{backend_id}.lane"),
            estimated_peak_vram_mb=None if vram is None else int(vram),
            min_free_vram_mb=None if floor is None else int(floor),
            native_context=bool(data.get("native_context", False)),
            visual_worlds=int(data.get("visual_worlds", 0) or 0),
            render_jobs=int(data.get("render_jobs", 0) or 0),
            needs_rpc_port=bool(data.get("needs_rpc_port", False)),
            cpu_threads=int(data.get("cpu_threads", 1) or 1),
            memory_mb=int(data.get("memory_mb", 0) or 0),
            io_weight=float(data.get("io_weight", 1.0)),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "BackendRequirement":
        data = _mapping(value, "requirement")
        return cls.from_mapping(
            data, backend_id=_text(data.get("backend_id"), "requirement.backend_id")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "lane": self.lane,
            "estimated_peak_vram_mb": self.estimated_peak_vram_mb,
            "min_free_vram_mb": self.min_free_vram_mb,
            "native_context": self.native_context,
            "visual_worlds": self.visual_worlds,
            "render_jobs": self.render_jobs,
            "needs_rpc_port": self.needs_rpc_port,
            "cpu_threads": self.cpu_threads,
            "memory_mb": self.memory_mb,
            "io_weight": self.io_weight,
        }


# Starting values only. Every field is overridable from configuration, and a
# run that measures a different peak should record the measured number rather
# than keep an estimate that no longer matches.
DEFAULT_BACKEND_PROFILES: dict[str, dict[str, Any]] = {
    "habitat_native_capture": {
        "lane": "gpu",
        "estimated_peak_vram_mb": 6144,
        "native_context": True,
        "visual_worlds": 1,
        "render_jobs": 1,
        "cpu_threads": 2,
    },
    "ue_spear_capture": {
        "lane": "gpu",
        "estimated_peak_vram_mb": 12288,
        "native_context": True,
        "visual_worlds": 1,
        "render_jobs": 1,
        "needs_rpc_port": True,
        "cpu_threads": 4,
    },
    # RLR builds a native propagation context and runs its rays on CPU cores.
    # A native context is not a graphics-device requirement.
    "rlr_audio_cpu": {
        "lane": "cpu",
        "native_context": True,
        "cpu_threads": 1,
    },
    "rlr_audio_gpu": {
        "lane": "gpu",
        "estimated_peak_vram_mb": 4096,
        "native_context": True,
        "cpu_threads": 1,
    },
    # Navmesh, pathfinder and floor work inside a Habitat process. It holds a
    # native context and no visual world, so it takes no graphics quota.
    "habitat_native_cpu": {
        "lane": "cpu",
        "native_context": True,
        "cpu_threads": 1,
    },
    "cpu_only": {
        "lane": "cpu",
        "cpu_threads": 1,
    },
}

# How a stage protocol's declared (execution slot, runtime context) resolves to
# a backend profile. The renderer only matters where two renderers share the
# same pair, which is the graphics capture case.
DEFAULT_BACKEND_BY_RUNTIME_PROFILE: dict[tuple[str, ...], str] = {
    ("cpu", "pure_python"): "cpu_only",
    ("cpu", "habitat_native"): "habitat_native_cpu",
    ("cpu", "rlr_native"): "rlr_audio_cpu",
    ("gpu", "rlr_native"): "rlr_audio_gpu",
    ("gpu", "renderer_native", "habitat"): "habitat_native_capture",
    ("gpu", "renderer_native", "ue_spear"): "ue_spear_capture",
    ("gpu", "habitat_native"): "habitat_native_capture",
}


def backend_for_runtime_profile(
    execution: str,
    runtime_context: str,
    *,
    renderer: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """Name the backend a declared execution slot and runtime context resolve to.

    Configuration keys are written ``"gpu/renderer_native/ue_spear"`` or
    ``"cpu/rlr_native"``, so a run can retarget one pair without restating the
    table.
    """

    table = dict(DEFAULT_BACKEND_BY_RUNTIME_PROFILE)
    for key, value in _mapping(overrides, "backend_by_runtime_profile").items():
        table[tuple(str(key).split("/"))] = str(value)
    candidates = []
    if renderer:
        candidates.append((execution, runtime_context, renderer))
    candidates.append((execution, runtime_context))
    for candidate in candidates:
        if candidate in table:
            return table[candidate]
    raise ResourcePolicyError(
        f"no backend is declared for execution {execution!r} in runtime context "
        f"{runtime_context!r}"
        + (f" on renderer {renderer!r}" if renderer else "")
        + "; declare one under backend_by_runtime_profile"
    )


def backend_profiles(overrides: Mapping[str, Any] | None = None) -> dict[str, BackendRequirement]:
    """Resolve the backend occupancy table, caller values winning per backend."""

    merged: dict[str, dict[str, Any]] = {
        name: dict(value) for name, value in DEFAULT_BACKEND_PROFILES.items()
    }
    for name, value in _mapping(overrides, "backends").items():
        base = dict(merged.get(str(name), {}))
        base.update(_mapping(value, f"backends.{name}"))
        merged[str(name)] = base
    return {
        name: BackendRequirement.from_mapping(value, backend_id=name)
        for name, value in merged.items()
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CpuPolicy:
    """The outer process budget. RLR's own thread count is separate."""

    max_workers: int = 4
    max_threads: int | None = None
    max_memory_mb: int | None = None
    max_io_weight: float | None = None
    # RLR's internal thread count stays 1 by default: raising it changes the
    # rendered waveform, so it is a measurement decision, not a speed knob.
    rlr_threads: int = 1

    @classmethod
    def from_mapping(cls, value: Any) -> "CpuPolicy":
        data = _mapping(value, "cpu")
        threads = data.get("max_threads")
        memory = data.get("max_memory_mb")
        io_weight = data.get("max_io_weight")
        return cls(
            max_workers=_positive_int(data.get("max_workers", 4), "cpu.max_workers"),
            max_threads=None if threads is None else _positive_int(threads, "cpu.max_threads"),
            max_memory_mb=None if memory is None else _positive_int(memory, "cpu.max_memory_mb"),
            max_io_weight=None if io_weight is None else float(io_weight),
            rlr_threads=_positive_int(data.get("rlr_threads", 1), "cpu.rlr_threads"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "max_threads": self.max_threads,
            "max_memory_mb": self.max_memory_mb,
            "max_io_weight": self.max_io_weight,
            "rlr_threads": self.rlr_threads,
        }


@dataclass(frozen=True)
class GpuDisplayProcessRule:
    """One exact, opt-in display process that may share an exclusive GPU.

    The rule is deliberately narrow. Missing or unreadable process identity
    never matches it, and the process memory remains part of the driver's
    free-memory number used by the allocator.
    """

    executable: str
    uid: int
    process_type: str = "G"
    max_memory_mb: int = 16
    effective_uid: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "executable", _text(self.executable, "gpu.display.executable"))
        object.__setattr__(self, "uid", _nonnegative_int(self.uid, "gpu.display.uid"))
        process_type = _text(self.process_type, "gpu.display.process_type")
        if process_type != "G":
            raise ResourcePolicyError(
                "gpu.nonblocking_display_processes process_type must be exactly 'G'"
            )
        object.__setattr__(self, "process_type", process_type)
        object.__setattr__(
            self,
            "max_memory_mb",
            _nonnegative_int(self.max_memory_mb, "gpu.display.max_memory_mb"),
        )
        if self.effective_uid is not None:
            object.__setattr__(
                self,
                "effective_uid",
                _nonnegative_int(self.effective_uid, "gpu.display.effective_uid"),
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "GpuDisplayProcessRule":
        data = _mapping(value, "gpu.nonblocking_display_processes[]")
        return cls(
            executable=data.get("executable"),
            uid=data.get("uid"),
            process_type=data.get("process_type", "G"),
            max_memory_mb=data.get("max_memory_mb", 16),
            effective_uid=data.get("effective_uid"),
        )

    def matches(self, process: "GpuProcess") -> bool:
        """Every configured condition holds, and nothing was left unread.

        ``effective_uid`` is checked only when the rule states one, so an
        existing two-uid-blind configuration keeps its meaning; when it is
        stated, the process must actually report that effective uid rather
        than merely fail to report a different one.
        """

        if self.effective_uid is not None and process.effective_uid != self.effective_uid:
            return False
        return (
            process.executable == self.executable
            and process.uid == self.uid
            and process.process_type == self.process_type
            and process.used_memory_mb <= self.max_memory_mb
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "uid": self.uid,
            "process_type": self.process_type,
            "max_memory_mb": self.max_memory_mb,
            "effective_uid": self.effective_uid,
        }


@dataclass(frozen=True)
class GpuPolicy:
    """Graphics-device budget. Device numbers come from here, never from code."""

    max_workers: int = 2
    devices: tuple[int, ...] | None = None
    min_free_vram_mb: int = 8192
    headroom_mb: int = 2048
    max_workers_per_device: int = 1
    allow_shared_device: bool = True
    selection: str = "most_free"
    nonblocking_display_processes: tuple[GpuDisplayProcessRule, ...] = ()

    def __post_init__(self) -> None:
        if self.selection not in ("most_free", "first_fit"):
            raise ResourcePolicyError(
                "gpu.selection must be 'most_free' or 'first_fit'"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "GpuPolicy":
        data = _mapping(value, "gpu")
        devices = data.get("devices")
        if devices is not None:
            if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes)):
                raise ResourcePolicyError("gpu.devices must be a list of indices")
            devices = tuple(
                _nonnegative_int(item, "gpu.devices[]") for item in devices
            )
            if not devices:
                raise ResourcePolicyError("gpu.devices must not be empty when given")
        display_processes = data.get("nonblocking_display_processes", ())
        if display_processes is None:
            display_processes = ()
        if not isinstance(display_processes, Sequence) or isinstance(
            display_processes, (str, bytes)
        ):
            raise ResourcePolicyError(
                "gpu.nonblocking_display_processes must be a list"
            )
        return cls(
            max_workers=_positive_int(data.get("max_workers", 2), "gpu.max_workers"),
            devices=devices,
            min_free_vram_mb=_nonnegative_int(
                data.get("min_free_vram_mb", 8192), "gpu.min_free_vram_mb"
            ),
            headroom_mb=_nonnegative_int(data.get("headroom_mb", 2048), "gpu.headroom_mb"),
            max_workers_per_device=_positive_int(
                data.get("max_workers_per_device", 1), "gpu.max_workers_per_device"
            ),
            allow_shared_device=bool(data.get("allow_shared_device", True)),
            selection=str(data.get("selection", "most_free")),
            nonblocking_display_processes=tuple(
                GpuDisplayProcessRule.from_mapping(item)
                for item in display_processes
            ),
        )

    def display_exception_for(
        self, process: "GpuProcess"
    ) -> GpuDisplayProcessRule | None:
        for rule in self.nonblocking_display_processes:
            if rule.matches(process):
                return rule
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "devices": None if self.devices is None else list(self.devices),
            "min_free_vram_mb": self.min_free_vram_mb,
            "headroom_mb": self.headroom_mb,
            "max_workers_per_device": self.max_workers_per_device,
            "allow_shared_device": self.allow_shared_device,
            "selection": self.selection,
            "nonblocking_display_processes": [
                rule.to_dict() for rule in self.nonblocking_display_processes
            ],
        }


@dataclass(frozen=True)
class PortPolicy:
    """The RPC ports one run may bind; also the key SPEAR isolates files by."""

    start: int = 30000
    count: int = 32
    host: str = "127.0.0.1"

    def __post_init__(self) -> None:
        _positive_int(self.start, "ports.start")
        _positive_int(self.count, "ports.count")
        if self.start < 1024 or self.start + self.count - 1 > 65535:
            raise ResourcePolicyError(
                "ports.start..start+count-1 must stay inside 1024..65535"
            )

    @property
    def candidates(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.start + self.count))

    @classmethod
    def from_mapping(cls, value: Any) -> "PortPolicy":
        data = _mapping(value, "ports")
        return cls(
            start=int(data.get("start", 30000)),
            count=int(data.get("count", 32)),
            host=str(data.get("host", "127.0.0.1")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "count": self.count, "host": self.host}


@dataclass(frozen=True)
class QueuePolicy:
    max_backlog: int = 256

    @classmethod
    def from_mapping(cls, value: Any) -> "QueuePolicy":
        data = _mapping(value, "queue")
        return cls(max_backlog=_positive_int(data.get("max_backlog", 256), "queue.max_backlog"))

    def to_dict(self) -> dict[str, Any]:
        return {"max_backlog": self.max_backlog}


@dataclass(frozen=True)
class WaitPolicy:
    """Waiting is bounded. A run that cannot start says so and stops waiting."""

    poll_interval_s: float = 5.0
    max_wait_s: float = 900.0

    def __post_init__(self) -> None:
        if not self.poll_interval_s > 0.0:
            raise ResourcePolicyError("wait.poll_interval_s must be positive")
        if not self.max_wait_s >= 0.0:
            raise ResourcePolicyError("wait.max_wait_s must not be negative")

    @classmethod
    def from_mapping(cls, value: Any) -> "WaitPolicy":
        data = _mapping(value, "wait")
        return cls(
            poll_interval_s=float(data.get("poll_interval_s", 5.0)),
            max_wait_s=float(data.get("max_wait_s", 900.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"poll_interval_s": self.poll_interval_s, "max_wait_s": self.max_wait_s}


@dataclass(frozen=True)
class ResourcePolicy:
    cpu: CpuPolicy = field(default_factory=CpuPolicy)
    gpu: GpuPolicy = field(default_factory=GpuPolicy)
    ports: PortPolicy = field(default_factory=PortPolicy)
    queue: QueuePolicy = field(default_factory=QueuePolicy)
    wait: WaitPolicy = field(default_factory=WaitPolicy)
    backends: Mapping[str, BackendRequirement] = field(default_factory=dict)
    backend_by_runtime_profile: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gpu.max_workers > self.cpu.max_workers:
            raise ResourcePolicyError(
                f"gpu.max_workers ({self.gpu.max_workers}) exceeds cpu.max_workers "
                f"({self.cpu.max_workers}); every graphics worker is also a "
                "process, so the two limits add up rather than multiply"
            )
        if self.gpu.max_workers_per_device > self.gpu.max_workers:
            raise ResourcePolicyError(
                "gpu.max_workers_per_device exceeds gpu.max_workers"
            )
        if self.cpu.max_threads is not None and self.cpu.max_threads < self.cpu.max_workers:
            raise ResourcePolicyError(
                "cpu.max_threads is below cpu.max_workers, so no full worker set "
                "could ever run"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "ResourcePolicy":
        data = _mapping(value, "resources")
        schema = data.get("schema")
        if schema is not None and str(schema) != SCHEMA:
            raise ResourcePolicyError(
                f"resource policy schema must be {SCHEMA}, got {schema!r}"
            )
        return cls(
            cpu=CpuPolicy.from_mapping(data.get("cpu")),
            gpu=GpuPolicy.from_mapping(data.get("gpu")),
            ports=PortPolicy.from_mapping(data.get("ports")),
            queue=QueuePolicy.from_mapping(data.get("queue")),
            wait=WaitPolicy.from_mapping(data.get("wait")),
            backends=backend_profiles(data.get("backends")),
            backend_by_runtime_profile={
                str(key): str(value)
                for key, value in _mapping(
                    data.get("backend_by_runtime_profile"),
                    "backend_by_runtime_profile",
                ).items()
            },
        )

    def requirement_for_runtime_profile(
        self, execution: str, runtime_context: str, *, renderer: str | None = None
    ) -> BackendRequirement:
        """Resolve what a stage protocol's declared axes actually occupy."""

        return self.requirement(
            backend_for_runtime_profile(
                execution, runtime_context, renderer=renderer,
                overrides=self.backend_by_runtime_profile,
            )
        )

    def requirement(self, backend_id: str) -> BackendRequirement:
        table = self.backends or backend_profiles(None)
        try:
            return table[str(backend_id)]
        except KeyError as error:
            raise ResourcePolicyError(
                f"no resource profile for backend {backend_id!r}; declared "
                f"backends are {sorted(table)}"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "cpu": self.cpu.to_dict(),
            "gpu": self.gpu.to_dict(),
            "ports": self.ports.to_dict(),
            "queue": self.queue.to_dict(),
            "wait": self.wait.to_dict(),
            "backends": {
                name: value.to_dict() for name, value in sorted((self.backends or {}).items())
            },
            "backend_by_runtime_profile": dict(self.backend_by_runtime_profile),
        }


# ---------------------------------------------------------------------------
# Reading the machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuDevice:
    index: int
    uuid: str
    name: str
    total_memory_mb: int
    free_memory_mb: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "total_memory_mb": self.total_memory_mb,
            "free_memory_mb": self.free_memory_mb,
        }


@dataclass(frozen=True)
class GpuProcess:
    """One process holding memory on a device, whoever started it.

    ``process_type`` is the driver's own C / G / C+G. A headless Habitat or
    SPEAR worker renders through EGL and is reported as a graphics process, so
    a compute-only query does not see it at all and would report no usage for
    a worker that is plainly running.
    """

    pid: int
    gpu_uuid: str
    used_memory_mb: int
    process_type: str = ""
    executable: str | None = None
    uid: int | None = None
    effective_uid: int | None = None
    identity_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "gpu_uuid": self.gpu_uuid,
            "used_memory_mb": self.used_memory_mb,
            "process_type": self.process_type,
            "executable": self.executable,
            "uid": self.uid,
            "effective_uid": self.effective_uid,
            "identity_source": self.identity_source,
        }


def _run_nvidia_smi(arguments, timeout_s: float) -> str:
    command = ["nvidia-smi", *arguments]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout_s
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ResourceUnavailable(
            "gpu_inventory_unavailable", f"cannot run nvidia-smi: {error}"
        ) from error
    if result.returncode != 0:
        raise ResourceUnavailable(
            "gpu_inventory_unavailable",
            f"nvidia-smi exited {result.returncode}",
            stderr=(result.stderr or "").strip()[-1000:],
        )
    return result.stdout or ""


def _first_int(text):
    if text is None:
        return None
    found = re.search(r"-?\d+", str(text))
    return None if found is None else int(found.group(0))


def _read_process_identity(
    pid: int,
) -> tuple[str | None, int | None, int | None, str]:
    """Read the identity an exact display exception rule is allowed to test.

    Both uids are returned because they disagree on a real desktop host: the
    display server this machine runs is started by the display manager and
    reports ``Uid: 128 0 0 0`` -- real uid gdm, effective uid root. ``ps``
    prints the effective one, ``/proc`` lists the real one first, and a rule
    written against only one of them either fails to match the process it was
    written for or matches more processes than intended.

    /proc/<pid>/exe is the identity that cannot be forged, but the link is
    unreadable for a process whose effective uid is root, which is exactly the
    display server. The first argv from cmdline is the documented fallback and
    is recorded as such, because a caller can set its own argv: a rule that
    rests on a cmdline identity is only as strong as the uid and memory
    conditions beside it. A field that could not be read stays unknown and can
    never satisfy a rule.
    """

    proc = Path("/proc") / str(int(pid))
    uid: int | None = None
    effective_uid: int | None = None
    try:
        for line in (proc / "status").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if not line.startswith("Uid:"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                try:
                    uid = int(fields[1])
                except ValueError:
                    pass
            if len(fields) >= 3:
                try:
                    effective_uid = int(fields[2])
                except ValueError:
                    pass
            break
    except OSError:
        pass

    executable: str | None = None
    identity_source = "unknown"
    try:
        executable = os.readlink(proc / "exe")
        if executable:
            identity_source = "exe"
    except OSError:
        pass
    if not executable:
        try:
            raw = (proc / "cmdline").read_bytes().split(b"\0", 1)[0]
            if raw:
                executable = raw.decode("utf-8", errors="replace")
                identity_source = "cmdline"
        except OSError:
            pass
    return executable, uid, effective_uid, identity_source


def read_gpu_state(timeout_s: float = 30.0):
    """Read devices and every process on them in one driver query.

    One query rather than two, because the free-memory number and the process
    attribution are compared against each other; reading them a second apart
    would compare two different moments. Utilisation is deliberately not read:
    a busy device with room is a valid co-tenant.
    """

    import xml.etree.ElementTree as ElementTree

    text = _run_nvidia_smi(["-q", "-x"], timeout_s)
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise ResourceUnavailable(
            "gpu_inventory_unavailable", f"cannot parse nvidia-smi output: {error}"
        ) from error
    devices = []
    processes = []
    for index, gpu in enumerate(root.findall("gpu")):
        uuid = gpu.findtext("uuid") or ""
        minor = _first_int(gpu.findtext("minor_number"))
        memory = gpu.find("fb_memory_usage")
        total = _first_int(None if memory is None else memory.findtext("total"))
        free = _first_int(None if memory is None else memory.findtext("free"))
        if uuid == "" or total is None or free is None:
            continue
        devices.append(
            GpuDevice(
                index=index if minor is None else minor,
                uuid=uuid,
                name=(gpu.findtext("product_name") or "").strip(),
                total_memory_mb=total,
                free_memory_mb=free,
            )
        )
        block = gpu.find("processes")
        for entry in [] if block is None else block.findall("process_info"):
            pid = _first_int(entry.findtext("pid"))
            used = _first_int(entry.findtext("used_memory"))
            if pid is None or used is None:
                continue
            executable, uid, effective_uid, identity_source = (
                _read_process_identity(pid)
            )
            processes.append(
                GpuProcess(
                    pid=pid,
                    gpu_uuid=uuid,
                    used_memory_mb=used,
                    process_type=(entry.findtext("type") or "").strip(),
                    executable=executable,
                    uid=uid,
                    effective_uid=effective_uid,
                    identity_source=identity_source,
                )
            )
    if not devices:
        raise ResourceUnavailable(
            "gpu_inventory_unavailable", "nvidia-smi reported no devices"
        )
    return tuple(devices), tuple(processes)


def query_gpu_inventory(timeout_s: float = 30.0):
    """Index, uuid, name and free VRAM. Utilisation is deliberately absent."""

    return read_gpu_state(timeout_s)[0]


def query_gpu_processes(timeout_s: float = 30.0):
    """List every process holding memory, graphics as well as compute.

    Used to attribute this run's own workers and to measure their real peak.
    Other people's processes are read, never signalled.
    """

    return read_gpu_state(timeout_s)[1]


def descendant_pids(pid: int) -> tuple[int, ...]:
    """Every live descendant of one process, for graphics-memory attribution.

    A launcher's own pid is often not the pid the driver charges: SPEAR starts
    the packaged game as a child process. Returning an empty tuple when the
    process is gone or unreadable is correct here -- attribution simply finds
    nothing extra -- but it is not proof that no child exists.
    """

    try:
        import psutil
    except ImportError:
        return ()
    try:
        children = psutil.Process(int(pid)).children(recursive=True)
    except Exception:
        return ()
    found = []
    for child in children:
        try:
            found.append(int(child.pid))
        except Exception:
            continue
    return tuple(sorted(found))


def port_is_bindable(port: int, host: str = "127.0.0.1") -> bool:
    """Bind then release one loopback port. Never connect, never scan."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaseRequest:
    """One schedulable unit's claim on the machine."""

    lease_id: str
    compatibility: WorkerCompatibility
    requirement: BackendRequirement
    output_relative: str
    routing: Mapping[str, Any] = field(default_factory=dict)
    # A saved request may already name the device or port it was written for.
    pinned_device_index: int | None = None
    pinned_rpc_port: int | None = None
    # A cap on how many of this backend may run at once, under the layer caps.
    max_parallel: int | None = None
    rlr_threads: int | None = None
    # A start-up floor for this one item, raising the policy and backend
    # floors. It never changes the peak estimate the reservation is built on.
    min_free_vram_mb: int | None = None

    def __post_init__(self) -> None:
        _text(self.lease_id, "lease_id")
        _text(self.output_relative, "output_relative")
        if Path(self.output_relative).is_absolute():
            raise ResourcePolicyError(
                "output_relative must be a repository-relative path so a run "
                "stays inside its own fresh output root"
            )

    @property
    def effective_min_free_vram_mb(self) -> int | None:
        """The strictest start-up floor the backend and this item ask for.

        The policy floor is applied separately, by the allocator that owns it.
        """

        floors = [
            value for value in
            (self.requirement.min_free_vram_mb, self.min_free_vram_mb)
            if value is not None
        ]
        return max(floors) if floors else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "compatibility": self.compatibility.to_dict(),
            "requirement": self.requirement.to_dict(),
            "output_relative": self.output_relative,
            "routing": deepcopy(dict(self.routing)),
            "pinned_device_index": self.pinned_device_index,
            "pinned_rpc_port": self.pinned_rpc_port,
            "max_parallel": self.max_parallel,
            "rlr_threads": self.rlr_threads,
            "min_free_vram_mb": self.min_free_vram_mb,
            "effective_min_free_vram_mb": self.effective_min_free_vram_mb,
        }


@dataclass(frozen=True)
class Lease:
    """A granted claim. Releasing it returns every part in one step."""

    lease_id: str
    compatibility: WorkerCompatibility
    requirement: BackendRequirement
    output_relative: str
    worker_slot: int
    device_index: int | None = None
    device_uuid: str | None = None
    reserved_vram_mb: int | None = None
    rpc_port: int | None = None
    granted_at: float = 0.0
    routing: Mapping[str, Any] = field(default_factory=dict)

    @property
    def holds_gpu(self) -> bool:
        return self.device_index is not None

    @classmethod
    def from_dict(cls, value: Any) -> "Lease":
        """Read back a lease a previous run wrote, for recovery."""

        data = _mapping(value, "lease")
        return cls(
            lease_id=_text(data.get("lease_id"), "lease.lease_id"),
            compatibility=WorkerCompatibility.from_dict(data.get("compatibility")),
            requirement=BackendRequirement.from_dict(data.get("requirement")),
            output_relative=_text(data.get("output_relative"), "lease.output_relative"),
            worker_slot=_positive_int(data.get("worker_slot"), "lease.worker_slot"),
            device_index=data.get("device_index"),
            device_uuid=data.get("device_uuid"),
            reserved_vram_mb=data.get("reserved_vram_mb"),
            rpc_port=data.get("rpc_port"),
            granted_at=float(data.get("granted_at") or 0.0),
            routing=deepcopy(_mapping(data.get("routing"), "lease.routing")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "worker_slot": self.worker_slot,
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "reserved_vram_mb": self.reserved_vram_mb,
            "rpc_port": self.rpc_port,
            "output_relative": self.output_relative,
            "granted_at": self.granted_at,
            "compatibility": self.compatibility.to_dict(),
            "requirement": self.requirement.to_dict(),
            "routing": deepcopy(dict(self.routing)),
        }


@dataclass(frozen=True)
class LeaseDecision:
    """Granted, worth waiting for, or never going to fit. Always with a reason."""

    status: str
    reason_code: str
    reason: str
    lease: Lease | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in DECISION_STATUSES:
            raise ResourcePolicyError(f"decision status must be one of {DECISION_STATUSES}")
        if (self.lease is None) == (self.status == "granted"):
            raise ResourcePolicyError(
                "a granted decision carries a lease and no other decision does"
            )

    @property
    def granted(self) -> bool:
        return self.status == "granted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "lease": None if self.lease is None else self.lease.to_dict(),
            "detail": deepcopy(dict(self.detail)),
        }


# ---------------------------------------------------------------------------
# The allocator
# ---------------------------------------------------------------------------


class ResourceAllocator:
    """Hand out bounded, non-overlapping claims on this machine.

    Callers are ordinary worker programs: they submit, pump, launch, report
    the real usage they observe, and release. Nothing here waits for a person
    or an agent, and nothing here touches a process it did not start.
    """

    def __init__(
        self,
        policy: ResourcePolicy | None = None,
        *,
        inventory_reader: Callable[[], Sequence[GpuDevice]] | None = None,
        process_reader: Callable[[], Sequence[GpuProcess]] | None = None,
        port_probe: Callable[[int, str], bool] | None = None,
        descendants_reader: Callable[[int], Sequence[int]] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.policy = policy or ResourcePolicy(backends=backend_profiles(None))
        self._inventory_reader = inventory_reader or query_gpu_inventory
        self._process_reader = process_reader or query_gpu_processes
        self._port_probe = port_probe or (lambda port, host: port_is_bindable(port, host))
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._lock = threading.RLock()
        self._queue: list[LeaseRequest] = []
        self._live: dict[str, Lease] = {}
        self._observed_vram_mb: dict[str, int] = {}
        self._worker_pid: dict[str, int] = {}
        self._extra_pids: dict[str, set[int]] = {}
        self._follow_descendants: dict[str, bool] = {}
        self._descendants_reader = descendants_reader or descendant_pids
        self._next_slot = 0
        self._history: list[dict[str, Any]] = []
        # Display processes an exclusive device actually tolerated, per device
        # index, as they were last read. Written so a run report can show what
        # was let past rather than only that something was.
        self._display_exclusions: dict[int, list[dict[str, Any]]] = {}

    # -- inventory -------------------------------------------------------

    def gpu_inventory(self) -> tuple[GpuDevice, ...]:
        return tuple(self._inventory_reader())

    def candidate_devices(self, inventory: Sequence[GpuDevice]) -> tuple[GpuDevice, ...]:
        inventory = tuple(inventory)
        if not inventory:
            raise ResourceUnavailable(
                "gpu_inventory_empty",
                "GPU inventory is empty; no graphics device can be selected",
                configured=(
                    None if self.policy.gpu.devices is None
                    else list(self.policy.gpu.devices)
                ),
                reported=[],
            )
        wanted = self.policy.gpu.devices
        if wanted is None:
            return inventory
        by_index = {device.index: device for device in inventory}
        missing = [index for index in wanted if index not in by_index]
        if missing:
            raise ResourceUnavailable(
                "configured_device_not_present",
                f"gpu.devices names indices this machine does not report: {missing}",
                configured=list(wanted),
                reported=sorted(by_index),
            )
        return tuple(by_index[index] for index in wanted)

    def outstanding_reservation_mb(self, device_index: int) -> int:
        """VRAM this run has promised but cannot yet see in the driver's number.

        A worker that has not started holds its whole estimate. Once its real
        usage is observed, only the unrealised remainder is held back, so two
        pending items never spend the same megabytes and a running worker is
        not counted twice.
        """

        total = 0
        for lease in self._live.values():
            if lease.device_index != device_index or lease.reserved_vram_mb is None:
                continue
            observed = self._observed_vram_mb.get(lease.lease_id, 0)
            total += max(0, lease.reserved_vram_mb - observed)
        return total

    def effective_free_mb(self, device: GpuDevice) -> int:
        with self._lock:
            return device.free_memory_mb - self.outstanding_reservation_mb(device.index)

    def _device_worker_count(self, device_index: int) -> int:
        return sum(1 for lease in self._live.values() if lease.device_index == device_index)

    def _attributed_pid_set(self) -> set[int]:
        """Every process this run owns: launchers, registered helpers, children.

        A SPEAR worker charges its graphics memory to a child process. Reading
        only the launcher pid would make this run's own child look like a
        foreign tenant and refuse the device it is already sitting on.
        """

        mine: set[int] = set(int(pid) for pid in self._worker_pid.values())
        for pids in self._extra_pids.values():
            mine.update(int(pid) for pid in pids)
        for lease_id in list(self._live):
            mine.update(self.attributed_pids(lease_id))
        return mine

    def _partition_device_processes(
        self, device: GpuDevice, apps: Sequence[GpuProcess]
    ) -> tuple[list[GpuProcess], list[dict[str, Any]]]:
        """Split a device's other tenants into foreign work and allowed display.

        An exception is granted only to a process whose real identity was read
        and matches a configured rule exactly; unknown identity is foreign, so
        an unreadable process can never talk its way onto an exclusive device.
        An excluded process is not adopted: its memory stays inside the
        driver's free number and it is never attributed to this run.
        """

        mine = self._attributed_pid_set()
        foreign: list[GpuProcess] = []
        excluded: list[dict[str, Any]] = []
        for app in apps:
            if app.gpu_uuid != device.uuid or app.pid in mine:
                continue
            rule = self.policy.gpu.display_exception_for(app)
            if rule is None:
                foreign.append(app)
                continue
            excluded.append(
                {
                    "device_index": device.index,
                    "process": app.to_dict(),
                    "rule": rule.to_dict(),
                }
            )
        return foreign, excluded

    def _record_display_exclusions(
        self, device_index: int, excluded: Sequence[Mapping[str, Any]]
    ) -> None:
        if excluded:
            self._display_exclusions[device_index] = [
                deepcopy(dict(item)) for item in excluded
            ]
        else:
            self._display_exclusions.pop(device_index, None)

    def display_exclusions(self) -> dict[int, list[dict[str, Any]]]:
        """What an exclusive device last tolerated, for a report to print."""

        with self._lock:
            return {
                index: deepcopy(items)
                for index, items in sorted(self._display_exclusions.items())
            }

    def _foreign_apps_on(self, device: GpuDevice, apps: Sequence[GpuProcess]) -> list[int]:
        return [app.pid for app in self._partition_device_processes(device, apps)[0]]

    # -- observation -----------------------------------------------------

    def bind_worker_pid(
        self, lease_id: str, pid: int, *, include_descendants: bool = True
    ) -> None:
        """Bind the process this run started, and by default its children.

        The pid a launcher gets back is not always the pid the driver charges.
        A SPEAR worker starts the packaged game as a separate process, so the
        Python pid can show no graphics memory at all while its child holds
        gigabytes. Attributing only the parent would read that as zero usage,
        keep the whole reservation held back, and report a peak of nothing.
        A Habitat worker renders in its own process, where parent and charged
        process are the same, and this costs nothing.
        """

        with self._lock:
            if lease_id not in self._live:
                raise ResourcePolicyError(f"no live lease {lease_id!r} to bind a pid to")
            self._worker_pid[lease_id] = int(pid)
            self._follow_descendants[lease_id] = bool(include_descendants)
            self._extra_pids.setdefault(lease_id, set())

    def register_worker_pid(self, lease_id: str, pid: int) -> None:
        """Name another process this lease owns, when the launcher reports one.

        Use it for a process the parent does not own in the tree sense -- a
        game the client started under a helper, or one adopted on recovery.
        """

        with self._lock:
            if lease_id not in self._live:
                raise ResourcePolicyError(f"no live lease {lease_id!r} to register a pid to")
            self._extra_pids.setdefault(lease_id, set()).add(int(pid))

    def attributed_pids(self, lease_id: str) -> tuple[int, ...]:
        """Every process whose graphics memory counts against this lease."""

        with self._lock:
            pids: set[int] = set(self._extra_pids.get(lease_id, set()))
            root = self._worker_pid.get(lease_id)
            if root is not None:
                pids.add(root)
                if self._follow_descendants.get(lease_id, True):
                    pids.update(self._descendants_reader(root))
            return tuple(sorted(pids))

    def report_observed_vram_mb(self, lease_id: str, used_memory_mb: int) -> None:
        """Record a worker's measured usage so its reservation stops double-counting."""

        with self._lock:
            if lease_id not in self._live:
                raise ResourcePolicyError(f"no live lease {lease_id!r} to observe")
            current = self._observed_vram_mb.get(lease_id, 0)
            self._observed_vram_mb[lease_id] = max(current, _nonnegative_int(
                used_memory_mb, "used_memory_mb"
            ))

    def refresh_observations(self) -> dict[str, int]:
        """Attribute live driver usage to this run's leases, process tree included.

        A pid is counted for one lease only. If two leases somehow claim the
        same process, the one that bound it as its own root wins, so a shared
        helper cannot be charged twice and inflate both peaks.
        """

        apps = tuple(self._process_reader())
        with self._lock:
            by_pid: dict[int, int] = {}
            for app in apps:
                by_pid[app.pid] = by_pid.get(app.pid, 0) + app.used_memory_mb
            owner: dict[int, str] = {}
            for lease_id in self._live:
                root = self._worker_pid.get(lease_id)
                if root is not None:
                    owner[root] = lease_id
            for lease_id in self._live:
                for pid in self.attributed_pids(lease_id):
                    owner.setdefault(pid, lease_id)
            totals: dict[str, int] = {}
            for pid, used in by_pid.items():
                lease_id = owner.get(pid)
                if lease_id is None:
                    continue
                totals[lease_id] = totals.get(lease_id, 0) + used
            seen: dict[str, int] = {}
            for lease_id, used in totals.items():
                previous = self._observed_vram_mb.get(lease_id, 0)
                self._observed_vram_mb[lease_id] = max(previous, used)
                seen[lease_id] = self._observed_vram_mb[lease_id]
            return seen

    def observed_peak_vram_mb(self, lease_id: str) -> int | None:
        with self._lock:
            return self._observed_vram_mb.get(lease_id)

    # -- granting --------------------------------------------------------

    def _cpu_capacity_reason(self, requirement: BackendRequirement) -> tuple[str, str] | None:
        cpu = self.policy.cpu
        if len(self._live) >= cpu.max_workers:
            return ("worker_slots_busy",
                    f"all {cpu.max_workers} worker slots are in use")
        if cpu.max_threads is not None:
            used = sum(lease.requirement.cpu_threads for lease in self._live.values())
            if used + requirement.cpu_threads > cpu.max_threads:
                return ("cpu_threads_busy",
                        f"{used} of {cpu.max_threads} CPU threads are in use and "
                        f"this item needs {requirement.cpu_threads}")
        if cpu.max_memory_mb is not None:
            used = sum(lease.requirement.memory_mb for lease in self._live.values())
            if used + requirement.memory_mb > cpu.max_memory_mb:
                return ("memory_budget_busy",
                        f"{used} MiB of {cpu.max_memory_mb} MiB host memory is in "
                        f"use and this item needs {requirement.memory_mb} MiB")
        if cpu.max_io_weight is not None:
            used = sum(lease.requirement.io_weight for lease in self._live.values())
            if used + requirement.io_weight > cpu.max_io_weight:
                return ("io_budget_busy",
                        f"{used} of {cpu.max_io_weight} I/O budget is in use and "
                        f"this item needs {requirement.io_weight}")
        return None

    def effective_min_free_vram_mb(self, request: LeaseRequest) -> int:
        """The start-up floor in force: policy, backend and item, strictest wins.

        This is a separate question from how much the worker will use. A small
        floor never shrinks the reservation, and a large floor never grows it.
        """

        floors = [self.policy.gpu.min_free_vram_mb]
        declared = request.effective_min_free_vram_mb
        if declared is not None:
            floors.append(declared)
        return max(floors)

    def _select_device(
        self,
        requirement: BackendRequirement,
        inventory: Sequence[GpuDevice],
        apps: Sequence[GpuProcess],
        pinned_device_index: int | None = None,
        min_free_vram_mb: int | None = None,
    ) -> tuple[GpuDevice | None, str, str]:
        gpu = self.policy.gpu
        floor = gpu.min_free_vram_mb if min_free_vram_mb is None else int(min_free_vram_mb)
        needed = int(requirement.estimated_peak_vram_mb or 0) + gpu.headroom_mb
        candidates = self.candidate_devices(inventory)
        if not candidates:
            return (
                None,
                "gpu_inventory_empty",
                "GPU inventory is empty; no graphics device can be selected",
            )
        if pinned_device_index is not None:
            candidates = tuple(
                item for item in candidates if item.index == pinned_device_index
            )
            if not candidates:
                return (None, "pinned_device_not_available",
                        f"this item pins gpu{pinned_device_index}, which is not "
                        "among the configured devices this machine reports")
        fits_at_all = [
            device for device in candidates if device.total_memory_mb >= needed
        ]
        if not fits_at_all:
            return (None, "no_configured_device_can_ever_fit",
                    f"no configured device has {needed} MiB of total memory; "
                    f"largest is {max(d.total_memory_mb for d in candidates)} MiB")
        usable: list[tuple[int, GpuDevice]] = []
        rejections: list[str] = []
        for device in fits_at_all:
            if self._device_worker_count(device.index) >= gpu.max_workers_per_device:
                rejections.append(
                    f"gpu{device.index}: already running "
                    f"{gpu.max_workers_per_device} of this run's workers"
                )
                continue
            if not gpu.allow_shared_device:
                foreign, excluded = self._partition_device_processes(device, apps)
                self._record_display_exclusions(device.index, excluded)
                if foreign:
                    rejections.append(
                        f"gpu{device.index}: gpu.allow_shared_device is false and "
                        f"pids {[app.pid for app in foreign]} are already using it"
                    )
                    continue
            if device.free_memory_mb < floor:
                rejections.append(
                    f"gpu{device.index}: {device.free_memory_mb} MiB free is below "
                    f"the min_free_vram_mb floor of {floor} MiB"
                )
                continue
            effective = device.free_memory_mb - self.outstanding_reservation_mb(device.index)
            if effective < needed:
                rejections.append(
                    f"gpu{device.index}: {effective} MiB effectively free "
                    f"(driver {device.free_memory_mb} MiB minus this run's "
                    f"{self.outstanding_reservation_mb(device.index)} MiB "
                    f"outstanding) is below the {needed} MiB needed"
                )
                continue
            usable.append((effective, device))
        if not usable:
            return (None, "insufficient_free_vram", "; ".join(rejections))
        if gpu.selection == "first_fit":
            chosen = min(usable, key=lambda pair: pair[1].index)[1]
        else:
            chosen = max(usable, key=lambda pair: (pair[0], -pair[1].index))[1]
        return (chosen, "granted", "")

    def _take_port(self, pinned: int | None = None) -> int | None:
        taken = {lease.rpc_port for lease in self._live.values() if lease.rpc_port}
        candidates = self.policy.ports.candidates if pinned is None else (pinned,)
        for port in candidates:
            if port in taken:
                continue
            if self._port_probe(port, self.policy.ports.host):
                return port
        return None

    def try_acquire(self, request: LeaseRequest) -> LeaseDecision:
        """One non-blocking attempt. Never raises for an ordinary shortage."""

        requirement = request.requirement
        with self._lock:
            if request.lease_id in self._live:
                return LeaseDecision(
                    status="blocked",
                    reason_code="lease_already_live",
                    reason=f"lease {request.lease_id!r} is already granted",
                )
            collision = [
                lease.lease_id
                for lease in self._live.values()
                if lease.output_relative == request.output_relative
            ]
            if collision:
                return LeaseDecision(
                    status="blocked",
                    reason_code="output_collision",
                    reason=(
                        f"output {request.output_relative!r} is already held by "
                        f"{collision[0]!r}; every worker writes its own fresh root"
                    ),
                    detail={"held_by": collision},
                )
            if request.max_parallel is not None:
                same = sum(
                    1 for lease in self._live.values()
                    if lease.requirement.backend_id == requirement.backend_id
                )
                if same >= request.max_parallel:
                    return LeaseDecision(
                        status="wait",
                        reason_code="backend_max_parallel_reached",
                        reason=(
                            f"{same} of a permitted {request.max_parallel} "
                            f"{requirement.backend_id!r} items are already running"
                        ),
                    )
            cpu_reason = self._cpu_capacity_reason(requirement)
            if cpu_reason is not None:
                return LeaseDecision(
                    status="wait", reason_code=cpu_reason[0], reason=cpu_reason[1]
                )
            device: GpuDevice | None = None
            if requirement.holds_gpu:
                in_use = sum(1 for lease in self._live.values() if lease.holds_gpu)
                if in_use >= self.policy.gpu.max_workers:
                    return LeaseDecision(
                        status="wait",
                        reason_code="gpu_slots_busy",
                        reason=f"all {self.policy.gpu.max_workers} graphics workers are in use",
                    )
                try:
                    inventory = self.gpu_inventory()
                    apps: tuple[GpuProcess, ...] = ()
                    if not self.policy.gpu.allow_shared_device:
                        apps = tuple(self._process_reader())
                except ResourceUnavailable as error:
                    return LeaseDecision(
                        status="blocked",
                        reason_code=error.code,
                        reason=str(error),
                        detail=dict(error.details),
                    )
                try:
                    device, code, reason = self._select_device(
                        requirement, inventory, apps,
                        pinned_device_index=request.pinned_device_index,
                        min_free_vram_mb=self.effective_min_free_vram_mb(request),
                    )
                except ResourceUnavailable as error:
                    return LeaseDecision(
                        status="blocked",
                        reason_code=error.code,
                        reason=str(error),
                        detail=dict(error.details),
                    )
                if device is None:
                    terminal = code in TERMINAL_DEVICE_REASON_CODES
                    # Keep the one-shot try_acquire compatibility for callers
                    # that historically saw a pinned mismatch as "wait", but
                    # mark it terminal so pump/acquire never busy-wait it.
                    status = "wait" if code == "pinned_device_not_available" else (
                        "blocked" if terminal else "wait"
                    )
                    return LeaseDecision(
                        status=status,
                        reason_code=code,
                        reason=reason,
                        detail={"terminal": terminal},
                    )
            port: int | None = None
            if requirement.needs_rpc_port:
                port = self._take_port(request.pinned_rpc_port)
                if port is None:
                    where = (
                        f"port {request.pinned_rpc_port}, which this item pins,"
                        if request.pinned_rpc_port is not None
                        else f"no port in {self.policy.ports.start}.."
                             f"{self.policy.ports.start + self.policy.ports.count - 1}"
                    )
                    return LeaseDecision(
                        status="wait",
                        reason_code="no_free_rpc_port",
                        reason=f"{where} is free for a new instance",
                    )
            self._next_slot += 1
            lease = Lease(
                lease_id=request.lease_id,
                compatibility=request.compatibility,
                requirement=requirement,
                output_relative=request.output_relative,
                worker_slot=self._next_slot,
                device_index=None if device is None else device.index,
                device_uuid=None if device is None else device.uuid,
                reserved_vram_mb=(
                    None if device is None else int(requirement.estimated_peak_vram_mb or 0)
                ),
                rpc_port=port,
                granted_at=self._clock(),
                routing=deepcopy(dict(request.routing)),
            )
            self._live[lease.lease_id] = lease
            self._history.append({"event": "granted", "lease": lease.to_dict()})
            return LeaseDecision(
                status="granted",
                reason_code="granted",
                reason="",
                lease=lease,
                detail={
                    "effective_free_mb": (
                        None if device is None else self.effective_free_mb(device)
                    ),
                    "excluded_display_processes": (
                        []
                        if device is None
                        else deepcopy(self._display_exclusions.get(device.index, []))
                    ),
                },
            )

    def recheck_before_launch(self, lease: Lease) -> LeaseDecision:
        """Read the driver again just before starting the process.

        A foreign process may have taken the memory between the grant and the
        launch. This check excludes the lease's own reservation, so it asks
        the honest question: is the memory still there for me?
        """

        if not lease.holds_gpu:
            return LeaseDecision(
                status="granted", reason_code="granted", reason="", lease=lease
            )
        try:
            inventory = self.gpu_inventory()
        except ResourceUnavailable as error:
            return LeaseDecision(
                status="blocked", reason_code=error.code, reason=str(error),
                detail=dict(error.details),
            )
        device = next((item for item in inventory if item.index == lease.device_index), None)
        if device is None:
            return LeaseDecision(
                status="blocked",
                reason_code="configured_device_not_present",
                reason=f"gpu{lease.device_index} is no longer reported by the driver",
            )
        if not self.policy.gpu.allow_shared_device:
            try:
                apps = tuple(self._process_reader())
            except ResourceUnavailable as error:
                return LeaseDecision(
                    status="blocked", reason_code=error.code, reason=str(error),
                    detail=dict(error.details),
                )
            with self._lock:
                foreign, excluded = self._partition_device_processes(device, apps)
                self._record_display_exclusions(device.index, excluded)
            if foreign:
                pids = [app.pid for app in foreign]
                return LeaseDecision(
                    status="wait",
                    reason_code="foreign_workload_appeared",
                    reason=(
                        f"gpu{device.index} is exclusive for this run but pids "
                        f"{pids} are on it now; they arrived between the grant "
                        "and the launch"
                    ),
                    detail={
                        "device": device.to_dict(),
                        "foreign_pids": pids,
                        "foreign_processes": [app.to_dict() for app in foreign],
                        "excluded_display_processes": excluded,
                    },
                )
        with self._lock:
            others = self.outstanding_reservation_mb(device.index)
            own = max(
                0,
                (lease.reserved_vram_mb or 0)
                - self._observed_vram_mb.get(lease.lease_id, 0),
            )
        available = device.free_memory_mb - (others - own)
        needed = int(lease.requirement.estimated_peak_vram_mb or 0) + self.policy.gpu.headroom_mb
        if available < needed:
            return LeaseDecision(
                status="wait",
                reason_code="insufficient_free_vram",
                reason=(
                    f"gpu{device.index} now has {available} MiB available for this "
                    f"lease but {needed} MiB is needed; another process took it "
                    "between the grant and the launch"
                ),
                detail={
                    "device": device.to_dict(),
                    "available_mb": available,
                    "needed_mb": needed,
                },
            )
        return LeaseDecision(
            status="granted", reason_code="granted", reason="", lease=lease,
            detail={
                "device": device.to_dict(),
                "available_mb": available,
                "excluded_display_processes": deepcopy(
                    self._display_exclusions.get(device.index, [])
                ),
            },
        )

    def release(self, lease: Lease | str) -> None:
        """Return the worker slot, the graphics reservation and the port at once."""

        lease_id = lease if isinstance(lease, str) else lease.lease_id
        with self._lock:
            released = self._live.pop(lease_id, None)
            self._observed_vram_mb.pop(lease_id, None)
            self._worker_pid.pop(lease_id, None)
            self._extra_pids.pop(lease_id, None)
            self._follow_descendants.pop(lease_id, None)
            if released is not None:
                self._history.append({"event": "released", "lease_id": lease_id})

    # -- recovery --------------------------------------------------------

    def adopt_lease(
        self,
        record: Mapping[str, Any] | Lease,
        *,
        pid: int | None = None,
        extra_pids: Sequence[int] = (),
        observed_vram_mb: int | None = None,
        include_descendants: bool = True,
    ) -> Lease:
        """Take an already-running worker back onto the books after a restart.

        Capacity is not re-tested, because the process exists whether or not
        the budget likes it. What matters is that its slot, its device
        reservation and its port are accounted for again, so the next grant
        does not hand the same memory or the same port to somebody else. A
        conflict with a lease that is already live is an error rather than a
        silent overwrite.
        """

        lease = record if isinstance(record, Lease) else Lease.from_dict(record)
        with self._lock:
            if lease.lease_id in self._live:
                raise ResourcePolicyError(
                    f"lease {lease.lease_id!r} is already live; adopting it again "
                    "would count its slot and its memory twice"
                )
            for other in self._live.values():
                if other.output_relative == lease.output_relative:
                    raise ResourcePolicyError(
                        f"cannot adopt {lease.lease_id!r}: {other.lease_id!r} is "
                        f"already writing {lease.output_relative!r}"
                    )
                if lease.rpc_port is not None and other.rpc_port == lease.rpc_port:
                    raise ResourcePolicyError(
                        f"cannot adopt {lease.lease_id!r}: {other.lease_id!r} is "
                        f"already holding port {lease.rpc_port}"
                    )
            self._live[lease.lease_id] = lease
            self._next_slot = max(self._next_slot, lease.worker_slot)
            if pid is not None:
                self._worker_pid[lease.lease_id] = int(pid)
                self._follow_descendants[lease.lease_id] = bool(include_descendants)
            self._extra_pids.setdefault(lease.lease_id, set()).update(
                int(value) for value in extra_pids
            )
            if observed_vram_mb is not None:
                self._observed_vram_mb[lease.lease_id] = _nonnegative_int(
                    observed_vram_mb, "observed_vram_mb"
                )
            self._history.append({"event": "adopted", "lease": lease.to_dict()})
            return lease

    def restore_from_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        is_running: Callable[[int], bool] | None = None,
    ) -> dict[str, Any]:
        """Rebuild the live table from a snapshot a previous run wrote.

        ``is_running`` decides which recorded workers are still alive; the
        caller owns that check because only it knows how it started them. A
        lease whose process is gone is reported as dropped rather than
        adopted, so its memory and its port come back into circulation.
        """

        adopted: list[str] = []
        dropped: list[dict[str, Any]] = []
        pids = _mapping(snapshot.get("worker_pids"), "snapshot.worker_pids")
        extra = _mapping(snapshot.get("extra_pids"), "snapshot.extra_pids")
        observed = _mapping(
            snapshot.get("observed_peak_vram_mb"), "snapshot.observed_peak_vram_mb")
        for entry in snapshot.get("live_leases") or []:
            lease = Lease.from_dict(entry)
            pid = pids.get(lease.lease_id)
            if is_running is not None and pid is not None and not is_running(int(pid)):
                dropped.append({"lease_id": lease.lease_id, "pid": int(pid),
                                "reason": "the recorded process is no longer running"})
                continue
            self.adopt_lease(
                lease,
                pid=None if pid is None else int(pid),
                extra_pids=[int(value) for value in (extra.get(lease.lease_id) or [])],
                observed_vram_mb=observed.get(lease.lease_id),
            )
            adopted.append(lease.lease_id)
        return {"adopted": adopted, "dropped": dropped,
                "snapshot": self.snapshot()}

    # -- queue -----------------------------------------------------------

    def submit(self, request: LeaseRequest) -> None:
        """Queue one claim. A full backlog is refused with its reason, not dropped."""

        with self._lock:
            if len(self._queue) >= self.policy.queue.max_backlog:
                raise ResourceUnavailable(
                    "queue_backlog_full",
                    f"the resource queue already holds {len(self._queue)} items, "
                    f"which is its configured maximum",
                    max_backlog=self.policy.queue.max_backlog,
                    lease_id=request.lease_id,
                )
            if any(item.lease_id == request.lease_id for item in self._queue):
                raise ResourcePolicyError(
                    f"lease {request.lease_id!r} is already queued"
                )
            self._queue.append(request)

    @property
    def backlog(self) -> int:
        with self._lock:
            return len(self._queue)

    def queued_lease_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(item.lease_id for item in self._queue)

    def pump(self) -> tuple[list[Lease], list[LeaseDecision]]:
        """Grant what fits right now and keep the rest queued, in order.

        Head-of-line order is preserved inside a lane: once a lane cannot
        start its oldest item, later items in that lane wait too, so a stream
        of small CPU work cannot starve a graphics job. The other lane keeps
        moving, which is the whole point of separating them.
        """

        granted: list[Lease] = []
        deferred: list[LeaseDecision] = []
        with self._lock:
            blocked_lanes: set[str] = set()
            remaining: list[LeaseRequest] = []
            for request in self._queue:
                lane = request.requirement.lane
                if lane in blocked_lanes:
                    remaining.append(request)
                    continue
                decision = self.try_acquire(request)
                if decision.granted and decision.lease is not None:
                    granted.append(decision.lease)
                    continue
                deferred.append(decision)
                if decision.status == "wait" and not decision.detail.get("terminal"):
                    blocked_lanes.add(lane)
                    remaining.append(request)
                # A blocked item leaves the queue: it will never fit as written.
            self._queue = remaining
        return granted, deferred

    def acquire(
        self,
        request: LeaseRequest,
        *,
        max_wait_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> LeaseDecision:
        """Wait a bounded time for a transient shortage, then report the reason.

        A permanent mismatch returns immediately. A bounded wait that runs out
        returns ``wait_deadline_exceeded`` with the last real shortage, which
        is what the caller re-queues or exits on; it never waits forever.
        """

        wait = self.policy.wait
        limit = wait.max_wait_s if max_wait_s is None else float(max_wait_s)
        interval = wait.poll_interval_s if poll_interval_s is None else float(poll_interval_s)
        started = self._clock()
        last = self.try_acquire(request)
        while not last.granted:
            if last.status == "blocked":
                return last
            if last.detail.get("terminal"):
                return LeaseDecision(
                    status="blocked",
                    reason_code=last.reason_code,
                    reason=last.reason,
                    detail=last.detail,
                )
            waited = self._clock() - started
            if waited + interval > limit:
                return LeaseDecision(
                    status="blocked",
                    reason_code="wait_deadline_exceeded",
                    reason=(
                        f"waited {waited:.1f}s of an allowed {limit:.1f}s for "
                        f"{request.lease_id!r}; last shortage was "
                        f"{last.reason_code}: {last.reason}"
                    ),
                    detail={
                        "waited_s": waited,
                        "max_wait_s": limit,
                        "last_reason_code": last.reason_code,
                        "last_reason": last.reason,
                    },
                )
            self._sleep(interval)
            last = self.try_acquire(request)
        return last

    # -- reporting -------------------------------------------------------

    def live_leases(self) -> tuple[Lease, ...]:
        with self._lock:
            return tuple(self._live.values())

    def snapshot(self) -> dict[str, Any]:
        """A plain record a worker log or a run report can write straight out."""

        with self._lock:
            live = list(self._live.values())
            per_device: dict[str, Any] = {}
            for lease in live:
                if lease.device_index is None:
                    continue
                bucket = per_device.setdefault(
                    str(lease.device_index),
                    {"workers": 0, "reserved_vram_mb": 0, "outstanding_vram_mb": 0},
                )
                bucket["workers"] += 1
                bucket["reserved_vram_mb"] += lease.reserved_vram_mb or 0
            for index in list(per_device):
                per_device[index]["outstanding_vram_mb"] = (
                    self.outstanding_reservation_mb(int(index))
                )
            return {
                "schema": SCHEMA,
                "policy": self.policy.to_dict(),
                "worker_slots_used": len(live),
                "worker_slots_total": self.policy.cpu.max_workers,
                "gpu_workers_used": sum(1 for lease in live if lease.holds_gpu),
                "gpu_workers_total": self.policy.gpu.max_workers,
                "cpu_threads_used": sum(lease.requirement.cpu_threads for lease in live),
                "backlog": len(self._queue),
                "backlog_total": self.policy.queue.max_backlog,
                "ports_in_use": sorted(
                    lease.rpc_port for lease in live if lease.rpc_port is not None
                ),
                "per_device": per_device,
                "gpu_display_exclusions": {
                    str(index): deepcopy(items)
                    for index, items in sorted(self._display_exclusions.items())
                },
                "observed_peak_vram_mb": dict(self._observed_vram_mb),
                "worker_pids": dict(self._worker_pid),
                "extra_pids": {
                    lease_id: sorted(pids)
                    for lease_id, pids in self._extra_pids.items() if pids
                },
                "attributed_pids": {
                    lease.lease_id: list(self.attributed_pids(lease.lease_id))
                    for lease in live
                },
                "live_leases": [lease.to_dict() for lease in live],
            }

    def history(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(deepcopy(item) for item in self._history)


# ---------------------------------------------------------------------------
# Consuming a stage work item
# ---------------------------------------------------------------------------

# Where a work item may state the backend it runs on. The stage name is not
# one of them: the same stage runs on different backends per room route, and
# a CPU acoustic run and a GPU acoustic run are the same stage.
BACKEND_ID_FIELDS = ("backend_id", "resource_backend_id", "executor_backend_id")


def _work_item_attribute(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def backend_id_for_work_item(item: Any) -> str | None:
    """Find the backend a work item declares, in payload or inputs."""

    for source_name in ("payload", "inputs"):
        source = _work_item_attribute(item, source_name)
        if not isinstance(source, Mapping):
            continue
        for field_name in BACKEND_ID_FIELDS:
            value = source.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for field_name in BACKEND_ID_FIELDS:
        value = _work_item_attribute(item, field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def lease_request_for_work_item(
    item: Any,
    *,
    policy: ResourcePolicy,
    compatibility: WorkerCompatibility,
    backend_id: str | None = None,
    output_relative: str | None = None,
    estimated_peak_vram_mb: int | None = None,
) -> LeaseRequest:
    """Turn one stage work item into a claim, without a stage-to-device table.

    The backend is resolved in the order a caller would expect: an explicit
    argument, then a backend the item names itself, then the execution slot
    and runtime context its own resource request declares. That last route is
    the normal one, because the stage protocol already separates *which
    scheduler slot* a work item occupies from *which process* it needs.

    Per-item values the request already carries are honoured, each as the
    thing it actually is. ``min_free_vram_mb`` is a start-up floor and is
    applied as a floor; it does not touch the peak estimate, because a worker
    that may start on a device with 1 GiB free is not a worker that will only
    use 1 GiB, and treating them as one number under-reserves on a shared
    device. A better peak estimate is passed as ``estimated_peak_vram_mb``,
    under its own name. A declared adapter or port pins the lease.
    """

    lease_id = _work_item_attribute(item, "work_item_id")
    if not isinstance(lease_id, str) or not lease_id.strip():
        raise ResourcePolicyError(
            "a stage work item must carry a work_item_id to be scheduled"
        )
    resource = _work_item_attribute(item, "resource")
    execution = _work_item_attribute(resource, "execution")
    runtime_context = _work_item_attribute(resource, "runtime_context")
    resolved_backend = backend_id or backend_id_for_work_item(item)
    if resolved_backend is None and execution and runtime_context:
        resolved_backend = backend_for_runtime_profile(
            str(execution), str(runtime_context),
            renderer=compatibility.renderer,
            overrides=policy.backend_by_runtime_profile,
        )
    if resolved_backend is None:
        raise ResourcePolicyError(
            f"work item {lease_id!r} does not say which backend it runs on; pass "
            f"backend_id, put one of {BACKEND_ID_FIELDS} in its payload, or give "
            "its resource request an execution slot and runtime context. The "
            "stage name is not a device."
        )
    requirement = policy.requirement(resolved_backend)
    # A peak estimate only ever comes from something that means "peak": an
    # explicit argument, or a field the request declares under that name.
    declared_peak = (
        estimated_peak_vram_mb
        if estimated_peak_vram_mb is not None
        else _work_item_attribute(resource, "estimated_peak_vram_mb")
    )
    if declared_peak is not None:
        if not requirement.holds_gpu:
            raise ResourcePolicyError(
                f"work item {lease_id!r} declares estimated_peak_vram_mb but its "
                f"backend {requirement.backend_id!r} runs on the CPU"
            )
        requirement = replace(
            requirement,
            estimated_peak_vram_mb=_positive_int(
                declared_peak, f"{lease_id} estimated_peak_vram_mb"
            ),
        )
    declared_floor = _work_item_attribute(resource, "min_free_vram_mb")
    if declared_floor is not None:
        _positive_int(declared_floor, f"{lease_id} min_free_vram_mb")
    output = output_relative or _work_item_attribute(item, "fresh_output_relative")
    if not isinstance(output, str) or not output.strip():
        raise ResourcePolicyError(
            f"work item {lease_id!r} has no fresh output path to isolate"
        )
    routing = {
        "stage": _work_item_attribute(item, "stage"),
        "request_id": _work_item_attribute(item, "request_id"),
        "group_id": _work_item_attribute(item, "group_id"),
        "task_family": _work_item_attribute(item, "task_family"),
        "attempt": _work_item_attribute(item, "attempt"),
        "resource_kind": _work_item_attribute(resource, "kind"),
        "execution": execution,
        "runtime_context": runtime_context,
    }
    max_parallel = _work_item_attribute(resource, "max_parallel")
    rlr_threads = _work_item_attribute(resource, "rlr_threads")
    return LeaseRequest(
        lease_id=lease_id.strip(),
        compatibility=compatibility,
        requirement=requirement,
        output_relative=output.strip(),
        routing={key: value for key, value in routing.items() if value is not None},
        pinned_device_index=_work_item_attribute(resource, "graphics_adapter"),
        pinned_rpc_port=(
            _work_item_attribute(resource, "rpc_port")
            if requirement.needs_rpc_port else None
        ),
        max_parallel=None if max_parallel is None else int(max_parallel),
        rlr_threads=None if rlr_threads is None else int(rlr_threads),
        min_free_vram_mb=(
            None if declared_floor is None or not requirement.holds_gpu
            else int(declared_floor)
        ),
    )


def worker_compatibility_for_work_item(
    item: Any,
    runtime: Mapping[str, Any],
    *,
    renderer: str,
    python_executable: str | None = None,
) -> WorkerCompatibility:
    """Build the process identity for one work item on one room's runtime.

    The runtime context comes from the work item, so an acoustic worker and a
    geometry worker on the same Habitat prefix stay two processes.
    """

    resource = _work_item_attribute(item, "resource")
    return WorkerCompatibility.from_runtime(
        runtime,
        renderer=renderer,
        python_executable=python_executable,
        runtime_context=_work_item_attribute(resource, "runtime_context"),
    )


# ---------------------------------------------------------------------------
# Launching the worker
# ---------------------------------------------------------------------------


def worker_launch_plan(
    lease: Lease,
    *,
    entry: Sequence[str],
    repository_root: str | Path = ".",
    python_path: Sequence[str] = ("src",),
    environment: Mapping[str, str] | None = None,
    rlr_threads: int | None = None,
) -> dict[str, Any]:
    """Describe the fresh interpreter one lease should run in.

    A fresh interpreter, never a fork: the parent may already hold one Habitat
    prefix, and a forked child keeps it. ``CUDA_VISIBLE_DEVICES`` is
    deliberately not set, because renumbering the devices would break the
    ``graphics_adapter`` and ``gpu_device_id`` the backends are given; the
    selected index is passed through instead.
    """

    if not entry:
        raise ResourcePolicyError("worker_launch_plan needs an entry to run")
    root = Path(repository_root)
    env: dict[str, str] = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(str(item) for item in python_path),
    }
    identity = dict(lease.compatibility.identity)
    if identity.get("runtime_prefix"):
        env["AVENGINE_HABITAT_RUNTIME_PREFIX"] = identity["runtime_prefix"]
    if identity.get("magnum_python_site"):
        env["AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"] = identity["magnum_python_site"]
    if identity.get("rlr_sdk_root"):
        env["AVENGINE_RLR_SDK_ROOT"] = identity["rlr_sdk_root"]
    env.update({str(key): str(value) for key, value in (environment or {}).items()})
    plan_extra: dict[str, Any] = {}
    if lease.compatibility.renderer == "ue_spear" and lease.rpc_port is not None:
        # The port the allocator reserved is the same port SPEAR isolates its
        # temp directory, log file and shared-memory id by.
        from avengine.backends.spear_ue.launch import (
            describe_instance_isolation,
            instance_settings_for_lease,
            launch_arguments_for_lease,
        )

        settings = instance_settings_for_lease(lease)
        plan_extra["spear_instance"] = settings
        plan_extra["spear_launch_arguments"] = launch_arguments_for_lease(lease)
        plan_extra["spear_isolation"] = describe_instance_isolation(
            settings,
            uproject=identity.get("uproject"),
            ddc_directory=(environment or {}).get("ddc_directory"),
        )
    return {
        **plan_extra,
        "schema": SCHEMA,
        "lease_id": lease.lease_id,
        "argv": [lease.compatibility.python_executable, "-B", *[str(item) for item in entry]],
        "cwd": str(root),
        "env": env,
        "start_method": "fresh_interpreter_subprocess",
        "worker_key": list(lease.compatibility.key[:2]) + [
            [name, value] for name, value in lease.compatibility.identity
        ],
        "graphics_device_index": lease.device_index,
        "rpc_port": lease.rpc_port,
        "output_relative": lease.output_relative,
        # Passed to the acoustic simulation config rather than an environment
        # variable, because RLR takes its thread count as a simulation field.
        "acoustic_thread_count": 1 if rlr_threads is None else int(rlr_threads),
        # What the runner must bind so graphics memory is attributed. A SPEAR
        # worker's packaged game is a separate process, so the parent's own
        # pid can read as zero while its child holds the memory.
        "pid_binding": {
            "bind": "the pid returned by the launcher",
            "include_descendants": True,
            "call": "bind_worker_pid(lease_id, pid, include_descendants=True)",
            "note": (
                "add register_worker_pid(lease_id, pid) for a process the tree "
                "does not show, such as a game started under a helper"
            ),
        },
    }


def group_by_worker(requests: Sequence[LeaseRequest]) -> dict[tuple[Any, ...], list[str]]:
    """Bucket claims by the process that could actually run them."""

    buckets: dict[tuple[Any, ...], list[str]] = {}
    for request in requests:
        buckets.setdefault(request.compatibility.key, []).append(request.lease_id)
    return buckets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_policy_file(path: str | Path) -> ResourcePolicy:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ResourcePolicyError(f"resource policy file must be an object: {path}")
    block = raw.get("resources") if "resources" in raw else raw
    return ResourcePolicy.from_mapping(block)


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m avengine.dataset.production_resources",
        description=(
            "Inspect the resource policy and this machine's current capacity. "
            "Nothing here starts, stops or signals a process."
        ),
    )
    parser.add_argument(
        "action",
        choices=("inventory", "policy", "plan"),
        help=(
            "inventory: what the driver reports now. policy: the effective "
            "configuration. plan: which device a named backend would get now."
        ),
    )
    parser.add_argument("--config", help="JSON file with a resources block")
    parser.add_argument("--backend", help="backend id for the plan action")
    parser.add_argument("--output-relative", default="tmp/resource_plan_probe")
    parser.add_argument("--renderer", default="habitat")
    parser.add_argument("--runtime-prefix")
    parser.add_argument("--magnum-python-site")
    parser.add_argument("--rlr-sdk-root")
    parser.add_argument("--uproject")
    parser.add_argument("--unreal-editor")
    parser.add_argument("--spear-ext-dir")
    args = parser.parse_args(argv)

    policy = load_policy_file(args.config) if args.config else ResourcePolicy(
        backends=backend_profiles(None)
    )
    if args.action == "policy":
        print(json.dumps(policy.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.action == "inventory":
        try:
            devices, apps = read_gpu_state()
        except ResourceUnavailable as error:
            print(json.dumps(
                {"status": "blocked", "reason_code": error.code, "reason": str(error)},
                indent=2, sort_keys=True,
            ))
            return 2
        print(json.dumps(
            {
                "schema": SCHEMA,
                "devices": [device.to_dict() for device in devices],
                "processes": [app.to_dict() for app in apps],
            },
            indent=2, sort_keys=True,
        ))
        return 0
    if not args.backend:
        parser.error("plan needs --backend")
    runtime = {
        "runtime_prefix": args.runtime_prefix,
        "magnum_python_site": args.magnum_python_site,
        "rlr_sdk_root": args.rlr_sdk_root,
        "uproject": args.uproject,
        "unreal_editor": args.unreal_editor,
        "spear_ext_dir": args.spear_ext_dir,
    }
    compatibility = WorkerCompatibility.from_runtime(
        {key: value for key, value in runtime.items() if value},
        renderer=args.renderer,
    )
    allocator = ResourceAllocator(policy)
    request = LeaseRequest(
        lease_id="cli_plan_probe",
        compatibility=compatibility,
        requirement=policy.requirement(args.backend),
        output_relative=args.output_relative,
    )
    decision = allocator.try_acquire(request)
    report = {"decision": decision.to_dict(), "snapshot": allocator.snapshot()}
    if decision.lease is not None:
        report["recheck"] = allocator.recheck_before_launch(decision.lease).to_dict()
        allocator.release(decision.lease)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if decision.granted else 3


__all__ = [
    "SCHEMA",
    "LANES",
    "PROCESS_IDENTITY_RUNTIME_KEYS",
    "RENDERERS_REQUIRING_FRESH_INTERPRETER",
    "PER_INSTANCE_RUNTIME_KEYS",
    "NATIVE_RUNTIME_MODULES",
    "TERMINAL_DEVICE_REASON_CODES",
    "DEFAULT_BACKEND_PROFILES",
    "ResourcePolicyError",
    "ResourceUnavailable",
    "WorkerCompatibility",
    "BackendRequirement",
    "CpuPolicy",
    "GpuPolicy",
    "PortPolicy",
    "QueuePolicy",
    "WaitPolicy",
    "ResourcePolicy",
    "GpuDevice",
    "GpuProcess",
    "LeaseRequest",
    "Lease",
    "LeaseDecision",
    "ResourceAllocator",
    "backend_profiles",
    "backend_id_for_work_item",
    "backend_for_runtime_profile",
    "DEFAULT_BACKEND_BY_RUNTIME_PROFILE",
    "classified_runtime_keys",
    "group_by_worker",
    "lease_request_for_work_item",
    "load_policy_file",
    "native_runtime_modules_loaded",
    "assert_clean_worker_parent",
    "descendant_pids",
    "port_is_bindable",
    "query_gpu_processes",
    "read_gpu_state",
    "query_gpu_inventory",
    "worker_compatibility_for_work_item",
    "worker_launch_plan",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
