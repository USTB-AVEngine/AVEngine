"""V1 production request parsing and the minimal stage protocol.

One configuration describes both ordinary Episodes and the four core group
tasks. Parsing resolves room, entity instances, sound pool, clock, rig, QA
identifiers, real QA targets, audio layouts, resource and retry policy once;
after that a saved legacy request still executes on its own.

The stage protocol is deliberately small. It is not a workflow framework: a
work item is an identity plus its inputs, dependencies, resource kind and fresh
output, and a stage result carries the facts the next stage needs. Stages are
handed out as their dependencies actually pass, because a cross-time-state
group has to measure its real reverberation tail before the path and capture
work for that group can be described at all.

Entity instances are the unit of identity here. Two instances may resolve the
same registered asset, so the number of registry entries is never the number of
entities in an Episode.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from numbers import Integral, Real
from typing import Any

from avengine.episode_clock import EpisodeClock, EpisodeClockError
from avengine.qa.binding_questions import TASK_FAMILIES as CORE_TASK_FAMILIES
from avengine.qa.unified_catalog import QA_IDS, get_requirements

SCHEMA = "avengine_v1_production_spec_v1"
LEGACY_REQUEST_SCHEMA = "avengine_native_qa_room_request_v1"
SAMPLING_POLICY = "conditioned_static_v2"

REQUEST_KINDS = ("episode", "core_group_member")
SOURCE_CLASSES = ("articulated_human", "articulated_animal", "rigid_static_object")

STAGES = ("plan", "capture", "audio", "late_plan", "assembly", "delivery")

# Two independent axes, because they were conflated before. `execution` is the
# scheduler slot a work item occupies; `runtime_context` is the process a worker
# has to be started in. RLR runs its propagation on the CPU inside a native
# context, so an acoustic stage is not a GPU job -- but a GPU acoustic backend
# stays declarable rather than being ruled out forever.
RESOURCE_KINDS = (
    "cpu",
    "cpu_native_geometry",
    "cpu_native_acoustic",
    "gpu_native_acoustic",
    "gpu_native_visual",
)
EXECUTION_SLOTS = ("cpu", "gpu")
RUNTIME_CONTEXTS = ("pure_python", "habitat_native", "renderer_native", "rlr_native")
RESOURCE_KIND_PROFILE = {
    "cpu": ("cpu", "pure_python"),
    "cpu_native_geometry": ("cpu", "habitat_native"),
    "cpu_native_acoustic": ("cpu", "rlr_native"),
    "gpu_native_acoustic": ("gpu", "rlr_native"),
    "gpu_native_visual": ("gpu", "renderer_native"),
}
# What each stage may legally occupy, default first. A configuration picks from
# this set; no stage forces one value.
STAGE_RESOURCE_KINDS = {
    "plan": ("cpu", "cpu_native_geometry"),
    "capture": ("gpu_native_visual",),
    "audio": ("cpu_native_acoustic", "gpu_native_acoustic"),
    "late_plan": ("cpu_native_geometry", "cpu"),
    "assembly": ("cpu",),
    "delivery": ("cpu",),
}
STAGE_RESOURCE_KIND = {stage: kinds[0] for stage, kinds in STAGE_RESOURCE_KINDS.items()}

# Facts a passing stage must publish, with real values, before its dependents
# can be described. `wet_tail_intervals` is the measured start_s/end_s list of
# one native binaural readback, which is what the real cross-time recipe reads;
# a single declared tail length is not a substitute.
STAGE_PUBLISHED_FACTS = {
    "plan": ("episode_plan_path", "renderer", "clock"),
    "capture": ("capture_receipt_path", "captured_frame_count"),
    "audio": ("facts_path", "audio_report_path", "wet_tail_intervals"),
    "late_plan": ("episode_plan_path", "first_motion_frame", "last_motion_frame",
                  "measured_wet_end_s"),
    "assembly": ("group_spec_path", "assembled_path", "validation"),
    "delivery": ("questions_path", "facts_path"),
}
STAGE_STATUSES = ("pass", "fail", "blocked", "not_run")

# `audio_tail_probe` is gone on purpose. dataset/binding_group_motion.py
# measures the reverberation tail from the two early official audio columns.
# There is no wet audio to probe before the first native readback exists, so a
# separate probe stage would have been a second render with nothing to measure.
REMOVED_STAGES = {
    "audio_tail_probe": "the tail is measured from the early audio columns instead",
}

MOTION_TIMINGS = ("none", "during_audible_window", "after_wet_tail")
# Recipes own both their visual intervention and their equivalence rule.  The
# rule must not be inferred from motion_timing alone: cross-event identity may
# change a controlled path while keeping the source slot order and using
# motion_timing="none".
PLAN_EQUIVALENCE_MODES = ("controlled_slots", "world")
VISUAL_INTERVENTION_MODES = (
    "source_slot_permutation",
    "identity_path_topology",
    "after_wet_tail_motion",
)
QUERY_IDENTITY_POLICIES = ("exact", "slot")
AUDIO_CONTENT_SCOPES = ("shared_audio_pairs", "member_scoped")

# These fields affect which clips may be selected.  Actual selected clip IDs
# and event schedules remain member-scoped until the declared shared audio
# pairs are checked against their assignment plans.
SOUND_SELECTION_POLICY_DEFAULTS: dict[str, Any] = {
    "clip_span_fit_policy": "filter_to_remaining_budget_then_uniform",
    "max_clip_s": None,
    "min_audible_s": 1.5,
    "unique_first_utterance_transcripts": True,
}
SOUND_SELECTION_POLICY_FIELDS = tuple(SOUND_SELECTION_POLICY_DEFAULTS)
# These are two different conditions and must stay that way. cross_time_state
# moves the entity only after the measured wet tail ends; motion while a source
# is audible is a separate condition carried by other recipes and by ordinary
# Episodes. Raising dynamic coverage must not turn one into the other.
MOTION_TIMING_BY_TASK_FAMILY = {
    "visible_binding": "none",
    "visual_conditioned_relation": "none",
    "cross_event_identity": "none",
    "cross_time_state": "after_wet_tail",
}
FORCED_MOTION_TIMING = {"cross_time_state": "after_wet_tail"}
MOVING_SPEECH_MOTIONS = ("speaker_moving", "competitor_moving")

EVENT_SELECTOR_KINDS = (
    "target_audible_window",
    "event_ordinal",
    "event_id",
    "clip_tail",
    "whole_clip",
)
AUDIO_LAYOUT_ROLES = ("primary", "attached_view")
FIXED_CHANNEL_COUNTS = {"mono": 1, "binaural": 2}
CAMERA_MOTION = "static"
QUESTION_FORMS = ("mcq", "open")
QA_INTENT_SOURCES = ("qa_ids_only", "explicit_conditions")


class ProductionSpecError(ValueError):
    """The configuration cannot be executed without changing its meaning."""


def _text(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionSpecError(f"{owner} must be nonempty text")
    return value


def _positive_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ProductionSpecError(f"{owner} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ProductionSpecError(f"{owner} must be a nonnegative integer")
    return int(value)


def _finite(value: Any, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ProductionSpecError(f"{owner} must be a finite number")
    return float(value)


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProductionSpecError(f"{owner} must be an object")
    return dict(value)


def _sequence(value: Any, owner: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProductionSpecError(f"{owner} must be a list")
    return list(value)


def normalize_sound_selection_policy(value: Any) -> dict[str, Any]:
    """Return only sound-selection rules shared by a controlled world."""
    data = value if isinstance(value, Mapping) else {}
    return {
        key: deepcopy(data[key] if key in data else default)
        for key, default in SOUND_SELECTION_POLICY_DEFAULTS.items()
    }


def normalize_sound_selection_content(
    value: Any, *, asset_by_actor: Mapping[str, Any] | None = None,
    include_candidate_allowlist: bool = False,
) -> dict[str, Any]:
    """Keep declared dry selections, canonicalized by physical asset when known."""
    data = value if isinstance(value, Mapping) else {}
    actor_assets = {
        str(actor): str(asset)
        for actor, asset in (asset_by_actor or {}).items()
        if asset is not None
    }

    def normalize_actor_map(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return deepcopy(value)
        normalized: dict[str, Any] = {}
        for actor, selected in value.items():
            actor_key = str(actor)
            key = actor_assets.get(actor_key, f"actor:{actor_key}")
            if key in normalized:
                key = f"{key}::{actor_key}"
            normalized[key] = deepcopy(selected)
        return normalized

    content: dict[str, Any] = {}
    for key, selected in data.items():
        if key in SOUND_SELECTION_POLICY_FIELDS:
            continue
        if key == "preallocated_sound_asset_ids_by_actor" and not include_candidate_allowlist:
            continue
        if key in {
            "preallocated_sound_asset_ids_by_actor",
            "selected_sound_asset_ids_by_actor",
        }:
            content[str(key)] = normalize_actor_map(selected)
        else:
            content[str(key)] = deepcopy(selected)
    return content


def deep_merge_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested objects by key; lists and scalars replace as a whole."""
    result = deepcopy(dict(base))
    for key, value in overrides.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge_mappings(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def _conflict(owner: str, first: Any, second: Any) -> None:
    raise ProductionSpecError(
        f"{owner} is declared twice with different values: {first!r} and {second!r}"
    )


@dataclass(frozen=True)
class RigSpec:
    """The one fixed camera/listener rig of an Episode."""

    resolution_hw: tuple[int, int]
    fov_deg: float
    height_above_floor_m: float | None = None
    motion: str = CAMERA_MOTION

    @classmethod
    def from_mapping(cls, value: Any, *, owner: str = "rig") -> "RigSpec":
        data = _mapping(value, owner)
        resolution = _sequence(data.get("resolution_hw"), owner + ".resolution_hw")
        if len(resolution) != 2:
            raise ProductionSpecError(owner + ".resolution_hw must be [height, width]")
        motion = str(data.get("motion", CAMERA_MOTION))
        if motion != CAMERA_MOTION:
            raise ProductionSpecError(
                f"{owner}.motion must be {CAMERA_MOTION!r}; this batch fixes the camera"
            )
        height = data.get("height_above_floor_m")
        result = cls(
            resolution_hw=(
                _positive_int(resolution[0], owner + ".resolution_hw[0]"),
                _positive_int(resolution[1], owner + ".resolution_hw[1]"),
            ),
            fov_deg=_finite(data.get("fov_deg"), owner + ".fov_deg"),
            height_above_floor_m=None if height is None else _finite(height, owner + ".height_above_floor_m"),
            motion=motion,
        )
        if not 0.0 < result.fov_deg < 180.0:
            raise ProductionSpecError(owner + ".fov_deg must be within (0, 180)")
        return result

    def to_request_camera(self) -> dict[str, Any]:
        camera: dict[str, Any] = {
            "fov_deg": self.fov_deg,
            "resolution_hw": [self.resolution_hw[0], self.resolution_hw[1]],
            "motion": self.motion,
        }
        if self.height_above_floor_m is not None:
            camera["height_above_floor_m"] = self.height_above_floor_m
        return camera

    def to_dict(self) -> dict[str, Any]:
        return self.to_request_camera()


@dataclass(frozen=True)
class AudioLayoutSpec:
    """One delivered channel layout of the shared listener rig."""

    layout_type: str
    channel_count: int
    role: str = "primary"
    ambisonic_order: int | None = None
    indirect_sh_order: int | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, owner: str = "audio_layout") -> "AudioLayoutSpec":
        data = _mapping(value, owner)
        layout_type = _text(data.get("type", data.get("layout_type")), owner + ".type")
        role = str(data.get("role", "primary"))
        if role not in AUDIO_LAYOUT_ROLES:
            raise ProductionSpecError(f"{owner}.role must be one of {AUDIO_LAYOUT_ROLES}")
        order = data.get("ambisonic_order")
        indirect = data.get("indirect_sh_order")
        if layout_type in FIXED_CHANNEL_COUNTS:
            expected = FIXED_CHANNEL_COUNTS[layout_type]
            if order is not None:
                raise ProductionSpecError(f"{owner}.ambisonic_order does not apply to {layout_type}")
        elif layout_type == "ambisonics":
            expected = (_nonnegative_int(1 if order is None else order, owner + ".ambisonic_order") + 1) ** 2
            order = 1 if order is None else int(order)
        else:
            raise ProductionSpecError(
                owner + ".type must be mono, binaural or ambisonics"
            )
        declared = data.get("channel_count")
        channel_count = expected if declared is None else _positive_int(declared, owner + ".channel_count")
        if channel_count != expected:
            raise ProductionSpecError(
                f"{owner} {layout_type} requires channel_count={expected}, got {channel_count}"
            )
        indirect = None if indirect is None else _nonnegative_int(indirect, owner + ".indirect_sh_order")
        if layout_type == "ambisonics" and indirect is not None and indirect < int(order):
            raise ProductionSpecError(
                f"{owner}.indirect_sh_order={indirect} cannot carry order {order} ambisonics"
            )
        return cls(
            layout_type=layout_type,
            channel_count=channel_count,
            role=role,
            ambisonic_order=order,
            indirect_sh_order=indirect,
        )

    def to_channel_layout(self) -> dict[str, Any]:
        """The exact shape avengine.acoustics.runtime.RLRChannelLayout reads."""
        return {"type": self.layout_type, "channel_count": self.channel_count}

    def to_dict(self) -> dict[str, Any]:
        value = {**self.to_channel_layout(), "role": self.role}
        if self.ambisonic_order is not None:
            value["ambisonic_order"] = self.ambisonic_order
        if self.indirect_sh_order is not None:
            value["indirect_sh_order"] = self.indirect_sh_order
        return value


@dataclass(frozen=True)
class EntityInstanceSpec:
    """One entity instance. Its identity is the instance, never its asset."""

    instance_id: str
    source_class: str
    asset_id: str | None = None
    role: str | None = None
    speaking: bool | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, index: int, owner: str = "instances") -> "EntityInstanceSpec":
        data = _mapping(value, f"{owner}[{index}]")
        source_class = _text(data.get("source_class"), f"{owner}[{index}].source_class")
        if source_class not in SOURCE_CLASSES:
            raise ProductionSpecError(f"{owner}[{index}].source_class must be one of {SOURCE_CLASSES}")
        speaking = data.get("speaking")
        if speaking is not None and not isinstance(speaking, bool):
            raise ProductionSpecError(f"{owner}[{index}].speaking must be true or false")
        asset_id = data.get("asset_id")
        return cls(
            instance_id=_text(data.get("instance_id", f"source{index + 1}"), f"{owner}[{index}].instance_id"),
            source_class=source_class,
            asset_id=None if asset_id is None else _text(asset_id, f"{owner}[{index}].asset_id"),
            role=None if data.get("role") is None else _text(data["role"], f"{owner}[{index}].role"),
            speaking=speaking,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"instance_id": self.instance_id, "source_class": self.source_class}
        if self.asset_id is not None:
            value["asset_id"] = self.asset_id
        if self.role is not None:
            value["role"] = self.role
        if self.speaking is not None:
            value["speaking"] = self.speaking
        return value


@dataclass(frozen=True)
class EventSelector:
    """Which sound event a question is about, stated rather than assumed.

    ``target_audible_window`` means the audible window of the named target
    instance. It is not the first event of the program, and an ordinal has to be
    written down when a specific ordinal is what the question means.
    """

    kind: str
    ordinal: int | None = None
    event_id: str | None = None
    window_s: tuple[float, float] | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, owner: str = "event") -> "EventSelector":
        if value is None:
            return cls(kind="target_audible_window")
        if isinstance(value, str):
            data: dict[str, Any] = {"kind": value}
        else:
            data = _mapping(value, owner)
        kind = _text(data.get("kind"), owner + ".kind")
        if kind not in EVENT_SELECTOR_KINDS:
            raise ProductionSpecError(f"{owner}.kind must be one of {EVENT_SELECTOR_KINDS}")
        ordinal = data.get("ordinal")
        event_id = data.get("event_id")
        window = data.get("window_s")
        if kind == "event_ordinal":
            if ordinal is None:
                raise ProductionSpecError(owner + ".ordinal is required for event_ordinal")
            ordinal = _positive_int(ordinal, owner + ".ordinal")
        elif ordinal is not None:
            raise ProductionSpecError(f"{owner}.ordinal does not apply to {kind}")
        if kind == "event_id":
            event_id = _text(event_id, owner + ".event_id")
        elif event_id is not None:
            raise ProductionSpecError(f"{owner}.event_id does not apply to {kind}")
        if window is not None:
            pair = _sequence(window, owner + ".window_s")
            if len(pair) != 2:
                raise ProductionSpecError(owner + ".window_s must be [start_s, end_s]")
            start, end = _finite(pair[0], owner + ".window_s[0]"), _finite(pair[1], owner + ".window_s[1]")
            if not 0.0 <= start < end:
                raise ProductionSpecError(owner + ".window_s must satisfy 0 <= start < end")
            window = (start, end)
        return cls(kind=kind, ordinal=ordinal, event_id=event_id, window_s=window)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind}
        if self.ordinal is not None:
            value["ordinal"] = self.ordinal
        if self.event_id is not None:
            value["event_id"] = self.event_id
        if self.window_s is not None:
            value["window_s"] = [self.window_s[0], self.window_s[1]]
        return value


@dataclass(frozen=True)
class QaTargetSpec:
    """One QA type asked about named instances and a named event selection."""

    qa_id: str
    target_instance_ids: tuple[str, ...]
    event: EventSelector
    items: int = 1
    forms: tuple[str, ...] = QUESTION_FORMS
    query_time_policy: str = "uniform_in_legal_window"
    target_source: str = "config"
    branch: str | None = None

    def __post_init__(self) -> None:
        if self.branch is not None:
            _text(self.branch, "qa_target.branch")
        if self.qa_id not in QA_IDS:
            raise ProductionSpecError(f"qa_target.qa_id is not in the unified catalog: {self.qa_id}")
        if not self.target_instance_ids:
            raise ProductionSpecError(
                f"qa_target {self.qa_id} must name the entity instances it asks about"
            )
        if len(set(self.target_instance_ids)) != len(self.target_instance_ids):
            raise ProductionSpecError(f"qa_target {self.qa_id} target_instance_ids must be distinct")
        _positive_int(self.items, f"qa_target[{self.qa_id}].items")
        catalog_forms = tuple(get_requirements(self.qa_id)["forms"])
        if not self.forms:
            raise ProductionSpecError(f"qa_target {self.qa_id} must keep at least one question form")
        illegal = [item for item in self.forms if item not in catalog_forms]
        if illegal:
            raise ProductionSpecError(f"qa_target {self.qa_id} forms are not offered: {illegal}")

    @classmethod
    def from_mapping(
        cls, value: Any, *, instance_ids: Sequence[str], owner: str = "qa_targets"
    ) -> "QaTargetSpec":
        data = _mapping(value, owner)
        qa_id = _text(data.get("qa_id"), owner + ".qa_id")
        if qa_id not in QA_IDS:
            raise ProductionSpecError(f"{owner}.qa_id is not in the unified catalog: {qa_id}")
        targets = tuple(
            _text(item, owner + ".target_instance_ids[]")
            for item in _sequence(
                data.get("target_instance_ids", data.get("target_instances")),
                owner + ".target_instance_ids",
            )
        )
        if not targets:
            raise ProductionSpecError(owner + ".target_instance_ids must name at least one instance")
        if len(set(targets)) != len(targets):
            raise ProductionSpecError(owner + ".target_instance_ids must be distinct")
        unknown = [value for value in targets if value not in set(instance_ids)]
        if unknown:
            raise ProductionSpecError(
                f"{owner}.target_instance_ids names entities absent from this request: {unknown}"
            )
        catalog_forms = tuple(get_requirements(qa_id)["forms"])
        declared_forms = _sequence(data.get("forms"), owner + ".forms")
        forms = tuple(str(item) for item in declared_forms) if declared_forms else catalog_forms
        illegal = [item for item in forms if item not in catalog_forms]
        if illegal:
            raise ProductionSpecError(f"{owner}.forms are not offered by {qa_id}: {illegal}")
        return cls(
            qa_id=qa_id,
            target_instance_ids=targets,
            event=EventSelector.from_mapping(data.get("event"), owner=owner + ".event"),
            items=_positive_int(data.get("items", 1), owner + ".items"),
            forms=forms,
            query_time_policy=_text(
                data.get("query_time_policy", "uniform_in_legal_window"),
                owner + ".query_time_policy",
            ),
            target_source=str(data.get("target_source", "config")),
            branch=None if data.get("branch") is None else _text(data["branch"], owner + ".branch"),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "qa_id": self.qa_id,
            "target_instance_ids": list(self.target_instance_ids),
            "event": self.event.to_dict(),
            "items": self.items,
            "forms": list(self.forms),
            "query_time_policy": self.query_time_policy,
            "target_source": self.target_source,
        }
        if self.branch is not None:
            value["branch"] = self.branch
        return value


@dataclass(frozen=True)
class RetryPolicy:
    """How often a stage and a within-profile sampler attempt may be repeated."""

    attempts_per_stage: int = 1
    attempts_within_profile: int = 200
    resume_from_stage: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, owner: str = "retry") -> "RetryPolicy":
        data = _mapping(value, owner)
        resume = data.get("resume_from_stage")
        if resume is not None and resume not in STAGES:
            raise ProductionSpecError(f"{owner}.resume_from_stage must be one of {STAGES}")
        result = cls(
            attempts_per_stage=_positive_int(data.get("attempts_per_stage", 1), owner + ".attempts_per_stage"),
            attempts_within_profile=_positive_int(
                data.get("attempts_within_profile", 200), owner + ".attempts_within_profile"
            ),
            resume_from_stage=resume,
        )
        if result.attempts_within_profile > 200:
            raise ProductionSpecError(owner + ".attempts_within_profile must be within 1..200")
        return result

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "attempts_per_stage": self.attempts_per_stage,
            "attempts_within_profile": self.attempts_within_profile,
        }
        if self.resume_from_stage is not None:
            value["resume_from_stage"] = self.resume_from_stage
        return value


@dataclass(frozen=True)
class ResourceRequest:
    """What one work item occupies while it runs."""

    kind: str
    graphics_adapter: int | None = None
    min_free_vram_mb: int | None = None
    rpc_port: int | None = None
    rlr_threads: int | None = None
    max_parallel: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in RESOURCE_KINDS:
            raise ProductionSpecError(f"resource kind must be one of {RESOURCE_KINDS}")
        if self.execution == "cpu" and self.min_free_vram_mb is not None:
            raise ProductionSpecError(
                f"{self.kind} runs on the CPU, so min_free_vram_mb does not apply"
            )

    @property
    def execution(self) -> str:
        """The scheduler slot: a CPU acoustic render must not hold a GPU slot."""
        return RESOURCE_KIND_PROFILE[self.kind][0]

    @property
    def runtime_context(self) -> str:
        """Which process a worker needs; one loaded Habitat prefix per process."""
        return RESOURCE_KIND_PROFILE[self.kind][1]

    @classmethod
    def from_mapping(cls, value: Any, *, kind: str, owner: str = "resources") -> "ResourceRequest":
        if kind not in RESOURCE_KINDS:
            raise ProductionSpecError(f"{owner}.kind must be one of {RESOURCE_KINDS}")
        data = _mapping(value, owner)
        port = data.get("rpc_port")
        if port is not None:
            port = _positive_int(port, owner + ".rpc_port")
            if port > 65535:
                raise ProductionSpecError(owner + ".rpc_port must be within 1..65535")
        adapter = data.get("graphics_adapter")
        vram = data.get("min_free_vram_mb")
        threads = data.get("rlr_threads")
        parallel = data.get("max_parallel")
        return cls(
            kind=kind,
            graphics_adapter=None if adapter is None else _nonnegative_int(adapter, owner + ".graphics_adapter"),
            min_free_vram_mb=None if vram is None else _positive_int(vram, owner + ".min_free_vram_mb"),
            rpc_port=port,
            rlr_threads=None if threads is None else _positive_int(threads, owner + ".rlr_threads"),
            max_parallel=None if parallel is None else _positive_int(parallel, owner + ".max_parallel"),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind, "execution": self.execution,
                                 "runtime_context": self.runtime_context}
        for name in ("graphics_adapter", "min_free_vram_mb", "rpc_port", "rlr_threads", "max_parallel"):
            current = getattr(self, name)
            if current is not None:
                value[name] = current
        return value


@dataclass(frozen=True)
class ProductionRequest:
    """A parsed production request: everything a stage needs, resolved once."""

    request_id: str
    kind: str
    room_id: str
    instances: tuple[EntityInstanceSpec, ...]
    clock: EpisodeClock
    rig: RigSpec
    audio_layouts: tuple[AudioLayoutSpec, ...]
    qa_ids: tuple[str, ...]
    qa_targets: tuple[QaTargetSpec, ...]
    quota_by_qa: dict[str, int]
    items_per_type: int
    profile: dict[str, Any]
    reserve_tail_s: float
    post_assembly_convolution_gain: float
    seed: int
    resources: dict[str, ResourceRequest]
    retry: RetryPolicy
    sound_pool_path: str | None = None
    sound_selection: dict[str, Any] = field(default_factory=dict)
    task_family: str | None = None
    group_id: str | None = None
    member_role: str | None = None
    condition_group: str | None = None
    motion_timing: str = "none"
    quota_source: str = "config"
    request_extras: dict[str, Any] = field(default_factory=dict)
    qa_targets_declared: bool = False
    qa_intent_source: str = "qa_ids_only"

    def __post_init__(self) -> None:
        if self.kind not in REQUEST_KINDS:
            raise ProductionSpecError(f"request kind must be one of {REQUEST_KINDS}")
        if not 2 <= len(self.instances) <= 4:
            raise ProductionSpecError("a production request carries 2..4 entity instances")
        ids = [instance.instance_id for instance in self.instances]
        if len(set(ids)) != len(ids):
            raise ProductionSpecError(f"entity instance_id values must be distinct: {ids}")
        if not any(instance.speaking is not False for instance in self.instances):
            raise ProductionSpecError("a production request must keep a speaking entity instance")
        if self.task_family is not None and self.task_family not in CORE_TASK_FAMILIES:
            raise ProductionSpecError(f"task_family must be one of {CORE_TASK_FAMILIES}")
        if self.kind == "core_group_member" and self.task_family is None:
            raise ProductionSpecError("a core group member must declare its task_family")
        unknown = [value for value in self.qa_ids if value not in QA_IDS]
        if unknown:
            raise ProductionSpecError(f"qa_ids are not in the unified catalog: {unknown}")
        if not self.qa_ids:
            raise ProductionSpecError("qa_ids must name at least one catalog type")
        if self.qa_intent_source not in QA_INTENT_SOURCES:
            raise ProductionSpecError(
                f"qa_intent_source must be one of {QA_INTENT_SOURCES}"
            )
        if self.qa_targets_declared and self.qa_intent_source != "explicit_conditions":
            raise ProductionSpecError(
                "declared qa_targets require qa_intent_source=explicit_conditions"
            )
        outside = sorted(set(self.quota_by_qa) - set(self.qa_ids))
        if outside:
            raise ProductionSpecError(f"quota_by_qa names QA types outside qa_ids: {outside}")
        for qa_id, quota in self.quota_by_qa.items():
            _positive_int(quota, f"quota_by_qa[{qa_id}]")
        target_outside = sorted({target.qa_id for target in self.qa_targets} - set(self.qa_ids))
        if target_outside:
            raise ProductionSpecError(f"qa_targets name QA types outside qa_ids: {target_outside}")
        if self.reserve_tail_s < 0 or not math.isfinite(self.reserve_tail_s):
            raise ProductionSpecError("reserve_tail_s must be finite and nonnegative")
        if self.reserve_tail_s >= float(self.clock.clip_seconds):
            raise ProductionSpecError(
                f"reserve_tail_s={self.reserve_tail_s} leaves no program time in a "
                f"{float(self.clock.clip_seconds):g}s clip"
            )
        if not math.isfinite(self.post_assembly_convolution_gain) or self.post_assembly_convolution_gain <= 0:
            raise ProductionSpecError("post_assembly_convolution_gain must be a positive finite number")
        primary = [layout for layout in self.audio_layouts if layout.role == "primary"]
        if len(primary) != 1:
            raise ProductionSpecError("exactly one audio layout must carry role=primary")
        seen_layouts = [(layout.layout_type, layout.role) for layout in self.audio_layouts]
        if len(set(seen_layouts)) != len(seen_layouts):
            raise ProductionSpecError("audio layouts must be distinct in type and role")
        for stage, resource in self.resources.items():
            if stage not in STAGES:
                raise ProductionSpecError(f"resources names an unknown stage: {stage}")
            if resource.kind not in STAGE_RESOURCE_KINDS[stage]:
                raise ProductionSpecError(
                    f"resources[{stage}].kind must be one of {STAGE_RESOURCE_KINDS[stage]}"
                )
        if self.motion_timing not in MOTION_TIMINGS:
            raise ProductionSpecError(f"motion_timing must be one of {MOTION_TIMINGS}")
        forced = FORCED_MOTION_TIMING.get(self.task_family)
        if forced is not None and self.motion_timing != forced:
            raise ProductionSpecError(
                f"{self.task_family} moves the entity {forced}, not {self.motion_timing!r}; "
                "motion while a source is audible is a separate condition"
            )
        if self.motion_timing == "after_wet_tail" and self.moving_speech_motion is not None:
            raise ProductionSpecError(
                f"after_wet_tail motion cannot also request speech_motion="
                f"{self.moving_speech_motion!r}; the early columns are stationary"
            )

    @property
    def moving_speech_motion(self) -> str | None:
        """The during-sound motion condition, read from the existing profile."""
        value = self.profile.get("speech_motion")
        return str(value) if value in MOVING_SPEECH_MOTIONS else None

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def distinct_asset_ids(self) -> tuple[str, ...]:
        return tuple(sorted({i.asset_id for i in self.instances if i.asset_id is not None}))

    @property
    def shares_assets_across_instances(self) -> bool:
        """True when two instances resolve one asset, so counting assets is wrong."""
        bound = [i.asset_id for i in self.instances if i.asset_id is not None]
        return len(set(bound)) != len(bound)

    @property
    def silent_count(self) -> int:
        return sum(1 for instance in self.instances if instance.speaking is False)

    @property
    def duration_seconds(self) -> float:
        return float(self.clock.clip_seconds)

    @property
    def available_program_seconds(self) -> float:
        return self.duration_seconds - self.reserve_tail_s

    @property
    def primary_audio_layout(self) -> AudioLayoutSpec:
        return next(layout for layout in self.audio_layouts if layout.role == "primary")

    @property
    def sound_selection_policy(self) -> dict[str, Any]:
        """The shared selection policy, without member-scoped clip IDs."""
        return normalize_sound_selection_policy(self.sound_selection)

    @property
    def sound_selection_content(self) -> dict[str, Any]:
        """Declared dry selections, without policy or actor-slot identity."""
        asset_by_actor = {
            instance.instance_id: instance.asset_id
            for instance in self.instances
        }
        return normalize_sound_selection_content(
            self.sound_selection, asset_by_actor=asset_by_actor
        )

    @property
    def qa_targets_source(self) -> str:
        """Whether targets are explicit planning intent or catalog-only derivation."""
        return "explicit" if self.qa_targets_declared else "derived_for_catalog"

    @property
    def binding_motion(self) -> dict[str, Any]:
        """The declared motion timing/speed block the real recipe reads."""
        return dict(_mapping(self.request_extras.get("binding_motion"), "binding_motion"))

    @property
    def has_own_stage_plan(self) -> bool:
        """False when the stages live on a group instead of this request."""
        return self.kind == "episode" and self.motion_timing != "after_wet_tail"

    def stage_plan(self) -> tuple[str, ...]:
        """The stages one ordinary Episode needs, in dependency order.

        A core group member has no plan of its own: its visual and audio units
        are shared with the other three members, and the after-tail motion of a
        cross-time group cannot be described from a single request.
        """
        if self.kind == "core_group_member":
            raise ProductionSpecError(
                f"{self.request_id} is a core group member; its stages are the shared units "
                f"of group {self.group_id}. Use recipe_for_task_family / group_stage_units."
            )
        if self.motion_timing == "after_wet_tail":
            raise ProductionSpecError(
                f"{self.request_id} declares after_wet_tail motion, which needs the early "
                "audio columns and a late plan; that shape only exists as a core group"
            )
        return ("plan", "capture", "audio", "delivery")

    def resource_for(self, stage: str) -> ResourceRequest:
        if stage not in STAGES:
            raise ProductionSpecError(f"unknown stage: {stage}")
        return self.resources.get(stage, ResourceRequest(kind=STAGE_RESOURCE_KIND[stage]))

    def to_legacy_request(self) -> dict[str, Any]:
        """Render the saved ``avengine_native_qa_room_request_v1`` payload.

        Declared extras stay readable: the parsed fields are written on top of
        whatever the configuration already supplied for this request.
        """
        profile = deepcopy(self.profile)
        profile["reserve_tail_s"] = self.reserve_tail_s
        request = deepcopy(self.request_extras)
        request = deep_merge_mappings(
            request,
            {
                "schema": LEGACY_REQUEST_SCHEMA,
                "episode_id": self.request_id,
                "room_id": self.room_id,
                "seed": self.seed,
                "sampling_policy": SAMPLING_POLICY,
                "camera": self.rig.to_request_camera(),
                "frame_count": self.clock.frame_count,
                "frame_rate_hz": self.clock.frame_rate_float,
                "sample_rate_hz": self.clock.sample_rate_hz,
                "entities": {
                    "total_count": self.instance_count,
                    "silent_count": self.silent_count,
                },
                "entity_instances": [instance.to_dict() for instance in self.instances],
                "profile": profile,
                "qa_ids": list(self.qa_ids),
                "qa_sampling": {"items_per_type": self.items_per_type},
                "quota_by_qa": dict(self.quota_by_qa),
                "audio_layouts": [layout.to_dict() for layout in self.audio_layouts],
                "post_assembly_convolution_gain": self.post_assembly_convolution_gain,
            },
        )
        if self.qa_targets_declared:
            request["qa_targets"] = [target.to_dict() for target in self.qa_targets]
        else:
            # Derived targets remain available to catalog generation through
            # ProductionRequest.qa_targets, but are not planning intent for the
            # legacy sampler request.
            request.pop("qa_targets", None)
        request["entities"].setdefault("min_articulated_count", 0)
        bound = [instance.asset_id for instance in self.instances]
        if all(value is not None for value in bound):
            request["source_asset_ids"] = list(bound)
        if self.sound_pool_path is not None:
            request["sound_pool"] = self.sound_pool_path
        if self.sound_selection:
            request["sound_selection"] = deep_merge_mappings(
                _mapping(request.get("sound_selection"), "sound_selection"), self.sound_selection
            )
        indirect = self.primary_audio_layout.indirect_sh_order
        if indirect is not None:
            request["simulation"] = deep_merge_mappings(
                _mapping(request.get("simulation"), "simulation"), {"indirect_sh_order": indirect}
            )
        if self.task_family is not None:
            request["task_family"] = self.task_family
        if self.group_id is not None:
            request["group_id"] = self.group_id
        if self.member_role is not None:
            request["member_role"] = self.member_role
        if self.condition_group is not None:
            request["condition_group"] = self.condition_group
        request["motion_timing"] = self.motion_timing
        request["production"] = deep_merge_mappings(
            _mapping(request.get("production"), "production"),
            {"stage_resources": {stage: value.to_dict()
                                 for stage, value in sorted(self.resources.items())},
             "retry": self.retry.to_dict()},
        )
        return request

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "kind": self.kind,
            "room_id": self.room_id,
            "task_family": self.task_family,
            "group_id": self.group_id,
            "member_role": self.member_role,
            "condition_group": self.condition_group,
            "motion_timing": self.motion_timing,
            "moving_speech_motion": self.moving_speech_motion,
            "instances": [instance.to_dict() for instance in self.instances],
            "instance_count": self.instance_count,
            "distinct_asset_ids": list(self.distinct_asset_ids),
            "shares_assets_across_instances": self.shares_assets_across_instances,
            "clock": self.clock.to_dict(),
            "rig": self.rig.to_dict(),
            "audio_layouts": [layout.to_dict() for layout in self.audio_layouts],
            "qa_ids": list(self.qa_ids),
            "qa_targets": [target.to_dict() for target in self.qa_targets],
            "qa_targets_source": self.qa_targets_source,
            "qa_intent_source": self.qa_intent_source,
            "quota_by_qa": dict(self.quota_by_qa),
            "quota_source": self.quota_source,
            "items_per_type": self.items_per_type,
            "reserve_tail_s": self.reserve_tail_s,
            "post_assembly_convolution_gain": self.post_assembly_convolution_gain,
            "duration_seconds": self.duration_seconds,
            "available_program_seconds": self.available_program_seconds,
            "seed": self.seed,
            "sound_pool": self.sound_pool_path,
            "sound_selection": deepcopy(self.sound_selection),
            "resources": {stage: value.to_dict() for stage, value in sorted(self.resources.items())},
            "retry": self.retry.to_dict(),
            "stage_scope": "episode" if self.has_own_stage_plan else "core_group",
            "stage_plan": list(self.stage_plan()) if self.has_own_stage_plan else None,
        }


@dataclass(frozen=True)
class CoreGroupRequest:
    """One core-task group and the members whose answers must actually differ."""

    group_id: str
    task_family: str
    room_id: str
    members: tuple[ProductionRequest, ...]
    shared_audio_member_ids: tuple[tuple[str, ...], ...] = ()
    shared_visual_member_ids: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.task_family not in CORE_TASK_FAMILIES:
            raise ProductionSpecError(f"task_family must be one of {CORE_TASK_FAMILIES}")
        if len(self.members) != 4:
            raise ProductionSpecError(
                f"core group {self.group_id} needs exactly 4 members, got {len(self.members)}"
            )
        ids = [member.request_id for member in self.members]
        if len(set(ids)) != len(ids):
            raise ProductionSpecError(f"core group member request_id values must be distinct: {ids}")
        for member in self.members:
            if member.room_id != self.room_id:
                raise ProductionSpecError(
                    f"core group {self.group_id} member {member.request_id} is in another room"
                )
            if member.task_family != self.task_family:
                raise ProductionSpecError(
                    f"core group {self.group_id} member {member.request_id} declares another task_family"
                )
        first = self.members[0]
        for member in self.members[1:]:
            for label, left, right in (
                ("seed", first.seed, member.seed),
                ("clock", first.clock.to_dict(), member.clock.to_dict()),
                ("rig", first.rig, member.rig),
                ("reserve_tail_s", first.reserve_tail_s, member.reserve_tail_s),
                ("post_assembly_convolution_gain", first.post_assembly_convolution_gain,
                 member.post_assembly_convolution_gain),
                ("audio_layouts", first.audio_layouts, member.audio_layouts),
                ("motion_timing", first.motion_timing, member.motion_timing),
                ("sound_selection_policy", first.sound_selection_policy,
                 member.sound_selection_policy),
            ):
                if left != right:
                    raise ProductionSpecError(
                        f"core group {self.group_id} member {member.request_id} differs in "
                        f"{label}; members share media and cannot disagree there"
                    )
            if sorted(instance.source_class for instance in first.instances) != sorted(
                instance.source_class for instance in member.instances
            ):
                raise ProductionSpecError(
                    f"core group {self.group_id} member {member.request_id} has another "
                    "source-class composition"
                )
        recipe = GROUP_RECIPES.get(self.task_family)
        member_ids = list(ids)
        member_by_id = {member.request_id: member for member in self.members}
        audio_pairs = self.shared_audio_member_ids
        if not audio_pairs and recipe is not None:
            by_column: dict[str, list[str]] = {}
            for unit in recipe.units:
                if unit.unit_kind != "audio" or unit.member_index is None:
                    continue
                column = unit.unit_id.rsplit("_a", 1)[-1]
                by_column.setdefault(column, []).append(
                    member_ids[unit.member_index]
                )
            audio_pairs = tuple(
                tuple(values) for values in by_column.values() if len(values) == 2
            )
        for pair in audio_pairs:
            if (
                len(pair) != 2
                or pair[0] not in member_by_id
                or pair[1] not in member_by_id
            ):
                # The structural validation below keeps the established
                # diagnostic for malformed shared_audio_member_ids.
                continue
            left_member = member_by_id[pair[0]]
            right_member = member_by_id[pair[1]]
            left = left_member.sound_selection_content
            right = right_member.sound_selection_content
            if left != right:
                raise ProductionSpecError(
                    f"core group {self.group_id} shared audio pair {list(pair)} "
                    "declares conflicting dry sound content"
                )
            left_assets = {
                instance.instance_id: instance.asset_id
                for instance in left_member.instances
            }
            right_assets = {
                instance.instance_id: instance.asset_id
                for instance in right_member.instances
            }
            left_declared = normalize_sound_selection_content(
                left_member.sound_selection,
                asset_by_actor=left_assets,
                include_candidate_allowlist=True,
            ).get("preallocated_sound_asset_ids_by_actor")
            right_declared = normalize_sound_selection_content(
                right_member.sound_selection,
                asset_by_actor=right_assets,
                include_candidate_allowlist=True,
            ).get("preallocated_sound_asset_ids_by_actor")
            if isinstance(left_declared, Mapping) and isinstance(right_declared, Mapping):
                for asset_id in sorted(set(left_declared) | set(right_declared)):
                    common = set(left_declared.get(asset_id, ())) & set(
                        right_declared.get(asset_id, ())
                    )
                    if not common:
                        raise ProductionSpecError(
                            f"core group {self.group_id} shared audio pair {list(pair)} "
                            f"has no common legal sound candidate for asset {asset_id}"
                        )
        if recipe is not None:
            member_ids = [member.request_id for member in self.members]
            for unit in recipe.units:
                if unit.unit_kind != "visual_capture":
                    continue
                served = _members_served_by(recipe, unit, member_ids)
                orders = {tuple(
                    instance.asset_id for instance in
                    self.members[member_ids.index(value)].instances
                ) for value in served}
                if len(orders) > 1:
                    raise ProductionSpecError(
                        f"core group {self.group_id} unit {unit.unit_id} serves members with "
                        f"different asset orders {sorted(orders)}; one capture is one video"
                    )
        known = set(ids)
        for name, pairs in (
            ("shared_audio_member_ids", self.shared_audio_member_ids),
            ("shared_visual_member_ids", self.shared_visual_member_ids),
        ):
            for pair in pairs:
                if len(pair) != 2 or len(set(pair)) != 2:
                    raise ProductionSpecError(f"{name} entries must name two different members")
                missing = [value for value in pair if value not in known]
                if missing:
                    raise ProductionSpecError(f"{name} names members outside this group: {missing}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "task_family": self.task_family,
            "room_id": self.room_id,
            "member_request_ids": [member.request_id for member in self.members],
            "shared_audio_member_ids": [list(pair) for pair in self.shared_audio_member_ids],
            "shared_visual_member_ids": [list(pair) for pair in self.shared_visual_member_ids],
            "recipe": recipe_for_task_family(self.task_family).to_dict(),
            "stage_units": group_stage_units(self),
        }


@dataclass(frozen=True)
class StageWorkItem:
    """One schedulable unit: identity, inputs, dependencies, resource, output."""

    work_item_id: str
    stage: str
    request_id: str
    attempt: int
    resource: ResourceRequest
    fresh_output_relative: str
    depends_on: tuple[str, ...] = ()
    inputs: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None
    task_family: str | None = None
    unit_id: str | None = None
    member_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            hint = REMOVED_STAGES.get(self.stage)
            raise ProductionSpecError(
                f"unknown stage: {self.stage}" + (f" ({hint})" if hint else "")
            )
        if self.resource.kind not in STAGE_RESOURCE_KINDS[self.stage]:
            raise ProductionSpecError(
                f"a {self.stage} work item must occupy one of "
                f"{STAGE_RESOURCE_KINDS[self.stage]}, not {self.resource.kind!r}"
            )
        _positive_int(self.attempt, "work item attempt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_item_id": self.work_item_id,
            "stage": self.stage,
            "request_id": self.request_id,
            "scope_id": self.request_id,
            "unit_id": self.unit_id,
            "member_request_ids": list(self.member_request_ids),
            "group_id": self.group_id,
            "task_family": self.task_family,
            "attempt": self.attempt,
            "resource": self.resource.to_dict(),
            "fresh_output_relative": self.fresh_output_relative,
            "depends_on": list(self.depends_on),
            "inputs": deepcopy(self.inputs),
            "payload": deepcopy(self.payload),
        }


@dataclass(frozen=True)
class StageResult:
    """What a finished work item reports, plus the facts its dependents read.

    Identity is checked here. `work_item_id` has to be exactly the id this
    scope, stage and attempt would produce, so a result cannot be filed against
    another unit, and `attempt` is read back out of it rather than trusted
    separately. `depends_on` names the upstream work items this result was
    actually produced from; that is what lets a newer upstream attempt
    invalidate the downstream results of an older one.

    Whether the named artifacts exist and agree with the media is P17's
    readback. Nothing here walks the filesystem.
    """

    work_item_id: str
    stage: str
    request_id: str
    status: str
    facts: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            hint = REMOVED_STAGES.get(self.stage)
            raise ProductionSpecError(
                f"unknown stage: {self.stage}" + (f" ({hint})" if hint else "")
            )
        if self.status not in STAGE_STATUSES:
            raise ProductionSpecError(f"stage status must be one of {STAGE_STATUSES}")
        if self.status != "pass" and not self.reason:
            raise ProductionSpecError(f"a {self.status} stage result must carry a reason")
        _text(self.request_id, "stage_result.request_id")
        self.attempt
        for value in self.depends_on:
            _text(value, "stage_result.depends_on[]")
        if self.status == "pass":
            missing = self.missing_published_facts()
            if missing:
                raise ProductionSpecError(
                    f"{self.work_item_id} reports pass without real values for {missing}"
                )

    @property
    def scope_id(self) -> str:
        """Whose stage sequence this belongs to: an Episode, or one group unit."""
        return self.request_id

    @property
    def attempt(self) -> int:
        """Read the round out of the identity instead of trusting a second field."""
        prefix = f"{self.request_id}:{self.stage}:"
        tail = self.work_item_id[len(prefix):] if self.work_item_id.startswith(prefix) else ""
        if not tail.isdigit() or work_item_id(self.request_id, self.stage, int(tail)) != self.work_item_id:
            raise ProductionSpecError(
                f"work_item_id {self.work_item_id!r} is not the id this scope and stage "
                f"produce; expected {work_item_id(self.request_id, self.stage, 1)!r} shape"
            )
        return _positive_int(int(tail), "stage_result attempt")

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def missing_published_facts(self) -> list[str]:
        """A required fact that is absent, None or empty is not published."""
        missing = []
        for key in STAGE_PUBLISHED_FACTS[self.stage]:
            if key not in self.facts:
                missing.append(key)
                continue
            value = self.facts[key]
            if value is None or (isinstance(value, (str, list, tuple, dict, set)) and not value):
                missing.append(key)
        return missing

    def wet_tail_end_seconds(self) -> list[float]:
        """The measured end of every wet-tail interval this audio column read back."""
        if self.stage != "audio":
            raise ProductionSpecError(f"{self.stage} does not publish wet_tail_intervals")
        intervals = self.facts.get("wet_tail_intervals")
        rows = _sequence(intervals, f"{self.work_item_id}.facts.wet_tail_intervals")
        if not rows:
            raise ProductionSpecError(
                f"{self.work_item_id} has no measured wet-tail interval"
            )
        ends = []
        for index, row in enumerate(rows):
            entry = _mapping(row, f"{self.work_item_id}.facts.wet_tail_intervals[{index}]")
            if "end_s" not in entry:
                raise ProductionSpecError(
                    f"{self.work_item_id}.facts.wet_tail_intervals[{index}] has no end_s"
                )
            ends.append(_finite(entry["end_s"], f"wet_tail_intervals[{index}].end_s"))
        return ends

    @classmethod
    def from_mapping(cls, value: Any, *, owner: str = "stage_result") -> "StageResult":
        data = _mapping(value, owner)
        scope = data.get("scope_id", data.get("request_id"))
        return cls(
            work_item_id=_text(data.get("work_item_id"), owner + ".work_item_id"),
            stage=_text(data.get("stage"), owner + ".stage"),
            request_id=_text(scope, owner + ".scope_id"),
            status=_text(data.get("status"), owner + ".status"),
            facts=_mapping(data.get("facts"), owner + ".facts"),
            outputs=_mapping(data.get("outputs"), owner + ".outputs"),
            reason=None if data.get("reason") is None else str(data["reason"]),
            depends_on=tuple(
                _text(item, owner + ".depends_on[]")
                for item in _sequence(data.get("depends_on"), owner + ".depends_on")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_item_id": self.work_item_id,
            "stage": self.stage,
            "request_id": self.request_id,
            "scope_id": self.request_id,
            "attempt": self.attempt,
            "status": self.status,
            "facts": deepcopy(self.facts),
            "outputs": deepcopy(self.outputs),
            "reason": self.reason,
            "depends_on": list(self.depends_on),
        }


def work_item_id(scope_id: str, stage: str, attempt: int) -> str:
    """One work item identity. `scope_id` is an Episode id or `group_id/unit_id`."""
    if ":" in scope_id:
        raise ProductionSpecError(f"scope_id must not contain ':': {scope_id!r}")
    return f"{scope_id}:{stage}:{int(attempt):02d}"


def fresh_output_relative(scope_id: str, stage: str, attempt: int) -> str:
    return f"{scope_id}/{stage}/attempt_{int(attempt):02d}"


def group_unit_scope_id(group_id: str, unit_id: str) -> str:
    """A group unit is its own scope, so two members cannot claim one capture."""
    _text(group_id, "group_id")
    _text(unit_id, "unit_id")
    if "/" in unit_id or ":" in unit_id or ":" in group_id:
        raise ProductionSpecError(f"unit scope cannot contain ':' or '/': {group_id}/{unit_id}")
    return f"{group_id}/{unit_id}"


UNIT_KINDS = ("visual_plan", "visual_capture", "audio", "assembly")


@dataclass(frozen=True)
class SharedUnitSpec:
    """One shared visual, audio or assembly unit of a four-member core group.

    A unit is produced once. `member_index` says which group member this unit
    delivers, and `visual_unit_id` says which captured visual an audio column
    consumes, so two members that share a video do not each re-plan one.
    """

    unit_id: str
    stage: str
    unit_kind: str
    depends_on_units: tuple[str, ...] = ()
    member_index: int | None = None
    visual_unit_id: str | None = None
    consumes_wet_tails_of: tuple[str, ...] = ()
    internal_only: bool = False

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ProductionSpecError(f"unit {self.unit_id} names an unknown stage: {self.stage}")
        if self.unit_kind not in UNIT_KINDS:
            raise ProductionSpecError(f"unit_kind must be one of {UNIT_KINDS}")
        if self.internal_only and self.member_index is not None:
            raise ProductionSpecError(
                f"internal unit {self.unit_id} cannot deliver a public member"
            )
        if self.internal_only and self.unit_kind == "assembly":
            raise ProductionSpecError(
                f"internal unit {self.unit_id} cannot be an assembly unit"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "stage": self.stage,
            "unit_kind": self.unit_kind,
            "depends_on_units": list(self.depends_on_units),
            "member_index": self.member_index,
            "visual_unit_id": self.visual_unit_id,
            "consumes_wet_tails_of": list(self.consumes_wet_tails_of),
            "internal_only": self.internal_only,
            "default_resource_kind": STAGE_RESOURCE_KIND[self.stage],
        }


def _audio_unit(unit_id: str, visual_unit_id: str, member_index: int) -> SharedUnitSpec:
    return SharedUnitSpec(unit_id=unit_id, stage="audio", unit_kind="audio",
                          depends_on_units=(visual_unit_id,), member_index=member_index,
                          visual_unit_id=visual_unit_id)


# Two visuals planned and captured independently, then one audio column per
# (visual, assignment) pair. This is the shape of
# dataset/binding_group_native.py prepare_visible_binding_group and
# prepare_visual_conditioned_relation_group.
PARALLEL_TWO_VISUAL_UNITS = (
    SharedUnitSpec("v0", "plan", "visual_plan"),
    SharedUnitSpec("v1", "plan", "visual_plan"),
    SharedUnitSpec("v0_capture", "capture", "visual_capture", depends_on_units=("v0",)),
    SharedUnitSpec("v1_capture", "capture", "visual_capture", depends_on_units=("v1",)),
    _audio_unit("v0_a0", "v0_capture", 0),
    _audio_unit("v0_a1", "v0_capture", 1),
    _audio_unit("v1_a0", "v1_capture", 2),
    _audio_unit("v1_a1", "v1_capture", 3),
    SharedUnitSpec("group", "assembly", "assembly",
                   depends_on_units=("v0_capture", "v1_capture",
                                     "v0_a0", "v0_a1", "v1_a0", "v1_a1")),
)

# The real cross-time shape from dataset/binding_group_motion.py
# prepare_fixed_camera_state_group: one static visual serves both early audio
# columns; the late plan reads the measured wet tails of BOTH of them before a
# second visual can be planned at all.
CROSS_TIME_STATE_UNITS = (
    SharedUnitSpec("v0", "plan", "visual_plan"),
    SharedUnitSpec("v0_capture", "capture", "visual_capture", depends_on_units=("v0",)),
    _audio_unit("v0_a0", "v0_capture", 0),
    _audio_unit("v0_a1", "v0_capture", 1),
    SharedUnitSpec("v1", "late_plan", "visual_plan",
                   depends_on_units=("v0", "v0_a0", "v0_a1"),
                   consumes_wet_tails_of=("v0_a0", "v0_a1")),
    SharedUnitSpec("v1_capture", "capture", "visual_capture", depends_on_units=("v1",)),
    _audio_unit("v1_a0", "v1_capture", 2),
    _audio_unit("v1_a1", "v1_capture", 3),
    SharedUnitSpec("group", "assembly", "assembly",
                   depends_on_units=("v0_capture", "v1_capture",
                                     "v0_a0", "v0_a1", "v1_a0", "v1_a1")),
)

# Identity must first measure the first event on a static probe. The topology
# plan consumes that measured tail and is an internal planning artifact; only
# v0/v1 and their four audio columns are public group members.
CROSS_EVENT_IDENTITY_UNITS = (
    SharedUnitSpec(
        "identity_probe_plan", "plan", "visual_plan", internal_only=True
    ),
    SharedUnitSpec(
        "identity_probe_capture", "capture", "visual_capture",
        depends_on_units=("identity_probe_plan",), internal_only=True
    ),
    SharedUnitSpec(
        "identity_probe_audio", "audio", "audio",
        depends_on_units=("identity_probe_capture",),
        visual_unit_id="identity_probe_capture",
        internal_only=True,
    ),
    SharedUnitSpec(
        "identity_topology", "plan", "visual_plan",
        depends_on_units=("identity_probe_audio",), internal_only=True
    ),
    SharedUnitSpec(
        "v0", "plan", "visual_plan",
        depends_on_units=("identity_topology",)
    ),
    SharedUnitSpec(
        "v1", "plan", "visual_plan",
        depends_on_units=("identity_topology",)
    ),
    SharedUnitSpec(
        "v0_capture", "capture", "visual_capture",
        depends_on_units=("v0",)
    ),
    SharedUnitSpec(
        "v1_capture", "capture", "visual_capture",
        depends_on_units=("v1",)
    ),
    _audio_unit("v0_a0", "v0_capture", 0),
    _audio_unit("v0_a1", "v0_capture", 1),
    _audio_unit("v1_a0", "v1_capture", 2),
    _audio_unit("v1_a1", "v1_capture", 3),
    SharedUnitSpec(
        "group", "assembly", "assembly",
        depends_on_units=(
            "v0_capture", "v1_capture", "v0_a0", "v0_a1", "v1_a0", "v1_a1"
        ),
    ),
)


@dataclass(frozen=True)
class GroupRecipe:
    """How one task family actually shares its visual and audio units."""

    task_family: str
    units: tuple[SharedUnitSpec, ...]
    motion_timing: str
    source: str
    plan_equivalence: str = "controlled_slots"
    visual_intervention: str = "source_slot_permutation"
    query_identity_policy: str = "exact"
    audio_content_scope: str = "shared_audio_pairs"

    def __post_init__(self) -> None:
        if self.plan_equivalence not in PLAN_EQUIVALENCE_MODES:
            raise ProductionSpecError(
                f"{self.task_family} plan_equivalence must be one of {PLAN_EQUIVALENCE_MODES}"
            )
        if self.visual_intervention not in VISUAL_INTERVENTION_MODES:
            raise ProductionSpecError(
                f"{self.task_family} visual_intervention must be one of "
                f"{VISUAL_INTERVENTION_MODES}"
            )
        if self.query_identity_policy not in QUERY_IDENTITY_POLICIES:
            raise ProductionSpecError(
                f"{self.task_family} query_identity_policy must be one of "
                f"{QUERY_IDENTITY_POLICIES}"
            )
        if self.audio_content_scope not in AUDIO_CONTENT_SCOPES:
            raise ProductionSpecError(
                f"{self.task_family} audio_content_scope must be one of "
                f"{AUDIO_CONTENT_SCOPES}"
            )
        ids = [unit.unit_id for unit in self.units]
        if len(set(ids)) != len(ids):
            raise ProductionSpecError(f"{self.task_family} repeats a unit_id: {ids}")
        known = set(ids)
        for unit in self.units:
            missing = [name for name in unit.depends_on_units if name not in known]
            if missing:
                raise ProductionSpecError(f"{unit.unit_id} depends on unknown units {missing}")
        members = [unit.member_index for unit in self.units if unit.member_index is not None]
        if sorted(members) != [0, 1, 2, 3]:
            raise ProductionSpecError(
                f"{self.task_family} must deliver exactly four members, got {sorted(members)}"
            )

    def unit(self, unit_id: str) -> SharedUnitSpec:
        for unit in self.units:
            if unit.unit_id == unit_id:
                return unit
        raise ProductionSpecError(f"{self.task_family} has no unit {unit_id!r}")

    @property
    def member_unit_ids(self) -> tuple[str, ...]:
        """The delivering unit of member 0..3, in member order."""
        by_index = {unit.member_index: unit.unit_id for unit in self.units
                    if unit.member_index is not None}
        return tuple(by_index[index] for index in range(4))

    def units_in_dependency_order(self) -> tuple[SharedUnitSpec, ...]:
        ordered: list[SharedUnitSpec] = []
        placed: set[str] = set()
        remaining = list(self.units)
        while remaining:
            ready = [unit for unit in remaining
                     if all(name in placed for name in unit.depends_on_units)]
            if not ready:
                raise ProductionSpecError(f"{self.task_family} unit graph has a cycle")
            for unit in ready:
                ordered.append(unit)
                placed.add(unit.unit_id)
                remaining.remove(unit)
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family,
            "motion_timing": self.motion_timing,
            "source": self.source,
            "plan_equivalence": self.plan_equivalence,
            "visual_intervention": self.visual_intervention,
            "query_identity_policy": self.query_identity_policy,
            "audio_content_scope": self.audio_content_scope,
            "unit_count": len(self.units),
            "member_unit_ids": list(self.member_unit_ids),
            "units": [unit.to_dict() for unit in self.units_in_dependency_order()],
        }


GROUP_RECIPES = {
    "visible_binding": GroupRecipe(
        "visible_binding", PARALLEL_TWO_VISUAL_UNITS, "none",
        "dataset/binding_group_native.py prepare_visible_binding_group",
        plan_equivalence="controlled_slots",
        visual_intervention="source_slot_permutation",
        query_identity_policy="slot",
        audio_content_scope="shared_audio_pairs"),
    "visual_conditioned_relation": GroupRecipe(
        "visual_conditioned_relation", PARALLEL_TWO_VISUAL_UNITS, "none",
        "dataset/binding_group_native.py prepare_visual_conditioned_relation_group",
        plan_equivalence="controlled_slots",
        visual_intervention="source_slot_permutation",
        query_identity_policy="slot",
        audio_content_scope="shared_audio_pairs"),
    "cross_event_identity": GroupRecipe(
        "cross_event_identity", CROSS_EVENT_IDENTITY_UNITS, "none",
        "dataset/binding_group_identity.py probe/topology/public identity stages",
        plan_equivalence="world",
        visual_intervention="identity_path_topology",
        query_identity_policy="exact",
        audio_content_scope="shared_audio_pairs"),
    "cross_time_state": GroupRecipe(
        "cross_time_state", CROSS_TIME_STATE_UNITS, "after_wet_tail",
        "dataset/binding_group_motion.py prepare_fixed_camera_state_group",
        plan_equivalence="world",
        visual_intervention="after_wet_tail_motion",
        query_identity_policy="exact",
        audio_content_scope="shared_audio_pairs"),
}


def recipe_for_task_family(task_family: str) -> GroupRecipe:
    recipe = GROUP_RECIPES.get(task_family)
    if recipe is None:
        raise ProductionSpecError(
            f"no shared-unit recipe for task_family {task_family!r}; "
            f"known: {sorted(GROUP_RECIPES)}"
        )
    return recipe


def measured_motion_window(
    *,
    wet_tail_end_seconds_by_unit: Mapping[str, Sequence[float]],
    clock: EpisodeClock,
    minimum_motion_s: float,
    end_hold_s: float,
    reserve_tail_s: float,
    source_work_item_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """The real recipe boundary, with tail length and absolute time kept apart.

    `measured_wet_end_s` is when the wet audio actually stops, not how long the
    tail lasted. The first movement frame is `ceil(measured_wet_end_s * fps) + 1`
    exactly as dataset/binding_group_motion.py computes it, and the declared
    terminal hold is subtracted from the far end.
    """
    ends: list[float] = []
    for unit_id, values in sorted(wet_tail_end_seconds_by_unit.items()):
        if not values:
            raise ProductionSpecError(f"{unit_id} published no measured wet-tail end")
        ends.extend(_finite(value, f"{unit_id} wet tail end_s") for value in values)
    if not ends:
        raise ProductionSpecError("a motion window needs at least one measured wet tail")
    fps = clock.frame_rate_float
    measured_wet_end_s = max(ends)
    first_motion_frame = math.ceil(measured_wet_end_s * fps) + 1
    last_motion_frame = clock.frame_count - 1 - math.ceil(_finite(end_hold_s, "end_hold_s") * fps)
    minimum_motion_frames = math.ceil(_finite(minimum_motion_s, "minimum_motion_s") * fps)
    available = last_motion_frame - first_motion_frame + 1
    return {
        "authority": "actual early binaural readbacks",
        "measured_wet_end_s": measured_wet_end_s,
        "measured_wet_end_sample": int(math.ceil(measured_wet_end_s * clock.sample_rate_hz)),
        "measured_tail_seconds_note": (
            "measured_wet_end_s is an absolute clip time, not a tail duration"
        ),
        "first_motion_frame": first_motion_frame,
        "last_motion_frame": last_motion_frame,
        "minimum_motion_frames": minimum_motion_frames,
        "available_motion_frames": available,
        "sufficient": bool(available >= minimum_motion_frames),
        "frame_rate_hz": fps,
        "requested_terminal_tail_s": _finite(reserve_tail_s, "reserve_tail_s"),
        "end_hold_s": float(end_hold_s),
        "minimum_motion_s": float(minimum_motion_s),
        "wet_tail_source_work_item_ids": sorted(source_work_item_ids),
        "boundary_formula": "ceil(max(wet_tail end_s) * fps) + 1",
    }


def _latest_by_key(
    results: Sequence[StageResult],
    *,
    key_of,
    allowed: Mapping[str, str],
    owner: str,
) -> dict[str, StageResult]:
    """One authoritative attempt per key: the newest, never the first to pass."""
    latest: dict[str, StageResult] = {}
    for result in results:
        key = key_of(result)
        if key not in allowed:
            raise ProductionSpecError(
                f"{owner} has no unit for stage result {result.work_item_id}"
            )
        if allowed[key] != result.stage:
            raise ProductionSpecError(
                f"{result.work_item_id} is a {result.stage} result filed against a "
                f"{allowed[key]} unit"
            )
        current = latest.get(key)
        if current is None or result.attempt > current.attempt:
            latest[key] = result
        elif result.attempt == current.attempt and result.to_dict() != current.to_dict():
            raise ProductionSpecError(
                f"two different results are filed as {result.work_item_id}"
            )
    return latest


@dataclass(frozen=True)
class RoundState:
    """The one valid round: what is done, what may run now, what is stopped."""

    done: dict[str, StageResult]
    ready: tuple[str, ...]
    stale: tuple[str, ...]
    blocked: tuple[tuple[str, str], ...] = ()

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked)

    def attempt_for(self, key: str, latest: Mapping[str, StageResult]) -> int:
        previous = latest.get(key)
        return 1 if previous is None else previous.attempt + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "done": {key: result.work_item_id for key, result in sorted(self.done.items())},
            "ready": list(self.ready),
            "stale": list(self.stale),
            "blocked": [list(entry) for entry in self.blocked],
        }


def _resolve_round(
    order: Sequence[str],
    latest: Mapping[str, StageResult],
    depends_of,
) -> RoundState:
    """Walk the graph once. A superseded upstream makes a result stale, not done."""
    done: dict[str, StageResult] = {}
    ready: list[str] = []
    stale: list[str] = []
    blocked: list[tuple[str, str]] = []
    for key in order:
        prerequisites = tuple(depends_of(key))
        if any(name not in done for name in prerequisites):
            continue
        result = latest.get(key)
        if result is None:
            ready.append(key)
            continue
        if not result.passed:
            blocked.append((key, result.reason or result.status))
            continue
        authoritative = {done[name].work_item_id for name in prerequisites}
        superseded = [dep for dep in result.depends_on if dep not in authoritative]
        if superseded:
            stale.append(key)
            ready.append(key)
            continue
        done[key] = result
    return RoundState(done=done, ready=tuple(ready), stale=tuple(stale), blocked=tuple(blocked))


def _stage_payload(
    request: ProductionRequest,
    stage: str,
    upstream: Mapping[str, StageResult],
    *,
    unit: SharedUnitSpec | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Carry the explicit clock, rig, layout and budget values into the item."""
    payload: dict[str, Any] = {
        "clock": request.clock.to_dict(),
        "duration_seconds": request.duration_seconds,
        "reserve_tail_s": request.reserve_tail_s,
        "available_program_seconds": request.available_program_seconds,
        "rig": request.rig.to_dict(),
        "audio_layouts": [layout.to_dict() for layout in request.audio_layouts],
        "post_assembly_convolution_gain": request.post_assembly_convolution_gain,
        "instance_ids": [instance.instance_id for instance in request.instances],
        "retry": request.retry.to_dict(),
        "motion_timing": request.motion_timing,
    }
    if stage in ("plan", "late_plan"):
        payload["qa_ids"] = list(request.qa_ids)
        payload["qa_intent_source"] = request.qa_intent_source
        if request.qa_targets_declared:
            payload["qa_targets"] = [target.to_dict() for target in request.qa_targets]
        payload["quota_by_qa"] = dict(request.quota_by_qa)
    if unit is not None:
        payload["unit_id"] = unit.unit_id
        payload["unit_kind"] = unit.unit_kind
        if unit.visual_unit_id is not None:
            payload["visual_unit_id"] = unit.visual_unit_id
    if extra:
        payload.update(deepcopy(dict(extra)))
    return payload


def _stage_inputs(
    request: ProductionRequest,
    stage: str,
    upstream: Mapping[str, StageResult],
) -> dict[str, Any]:
    if stage == "plan" and not upstream:
        return {"request": request.to_legacy_request()}
    inputs: dict[str, Any] = {"request_id": request.request_id}
    if stage in ("plan", "late_plan"):
        inputs["request"] = request.to_legacy_request()
    for name, result in sorted(upstream.items()):
        inputs[name] = {"work_item_id": result.work_item_id, "facts": deepcopy(result.facts),
                        "outputs": deepcopy(result.outputs)}
    return inputs


def _make_work_item(
    request: ProductionRequest,
    stage: str,
    *,
    attempt: int,
    upstream: Mapping[str, StageResult],
    scope_id: str | None = None,
    unit: SharedUnitSpec | None = None,
    member_request_ids: Sequence[str] = (),
    extra_payload: Mapping[str, Any] | None = None,
) -> StageWorkItem:
    scope = scope_id or request.request_id
    return StageWorkItem(
        work_item_id=work_item_id(scope, stage, attempt),
        stage=stage,
        request_id=scope,
        attempt=attempt,
        resource=request.resource_for(stage),
        fresh_output_relative=fresh_output_relative(scope, stage, attempt),
        depends_on=tuple(result.work_item_id for _, result in sorted(upstream.items())),
        inputs=_stage_inputs(request, stage, upstream),
        payload=_stage_payload(request, stage, upstream, unit=unit, extra=extra_payload),
        group_id=request.group_id,
        task_family=request.task_family,
        unit_id=None if unit is None else unit.unit_id,
        member_request_ids=tuple(member_request_ids),
    )


def initial_stage_work_items(request: ProductionRequest) -> list[StageWorkItem]:
    """Only the work that is already describable: this Episode's planning stage.

    Later stages are not expanded here; the schedule grows from actual results.
    A core group member has no independent schedule -- its visual and audio
    units are shared, so use `initial_group_work_items` on its group.
    """
    if request.kind == "core_group_member":
        raise ProductionSpecError(
            f"{request.request_id} is a core group member; its stages are group units. "
            "Call initial_group_work_items(group) so one visual is not planned twice."
        )
    return [_make_work_item(request, "plan", attempt=1, upstream={})]


def request_round_state(
    request: ProductionRequest, results: Sequence[StageResult]
) -> RoundState:
    """The valid round of one ordinary Episode."""
    order = request.stage_plan()
    allowed = {stage: stage for stage in order}
    for result in results:
        if result.scope_id != request.request_id:
            raise ProductionSpecError(
                f"stage result {result.work_item_id} belongs to {result.scope_id}, "
                f"not {request.request_id}"
            )
    latest = _latest_by_key(results, key_of=lambda item: item.stage, allowed=allowed,
                            owner=request.request_id)
    index = {stage: position for position, stage in enumerate(order)}
    return _resolve_round(
        order, latest,
        lambda stage: order[: index[stage]],
    )


def next_stage_work_items(
    request: ProductionRequest, results: Sequence[StageResult]
) -> list[StageWorkItem]:
    """Hand out the stages whose dependencies have actually passed this round."""
    order = request.stage_plan()
    state = request_round_state(request, results)
    latest = _latest_by_key(results, key_of=lambda item: item.stage,
                            allowed={stage: stage for stage in order},
                            owner=request.request_id)
    if state.is_blocked or not state.ready:
        return []
    index = {stage: position for position, stage in enumerate(order)}
    stage = state.ready[0]
    upstream = {name: state.done[name] for name in order[: index[stage]]}
    return [_make_work_item(request, stage, attempt=state.attempt_for(stage, latest),
                            upstream=upstream)]


def group_round_state(
    group: "CoreGroupRequest", results: Sequence[StageResult]
) -> RoundState:
    """The valid round of one four-member core group, keyed by shared unit."""
    recipe = recipe_for_task_family(group.task_family)
    order = [unit.unit_id for unit in recipe.units_in_dependency_order()]
    allowed = {unit.unit_id: unit.stage for unit in recipe.units}
    scope_to_unit = {group_unit_scope_id(group.group_id, unit_id): unit_id
                     for unit_id in allowed}

    def key_of(result: StageResult) -> str:
        unit_id = scope_to_unit.get(result.scope_id)
        if unit_id is None:
            raise ProductionSpecError(
                f"stage result {result.work_item_id} is not a unit of {group.group_id}"
            )
        return unit_id

    latest = _latest_by_key(results, key_of=key_of, allowed=allowed, owner=group.group_id)
    return _resolve_round(order, latest, lambda unit_id: recipe.unit(unit_id).depends_on_units)


def _group_unit_extra_payload(
    group: "CoreGroupRequest",
    unit: SharedUnitSpec,
    done: Mapping[str, StageResult],
) -> dict[str, Any]:
    if not unit.consumes_wet_tails_of:
        return {}
    member = group.members[0]
    motion = member.binding_motion
    missing = [key for key in ("minimum_motion_s", "end_hold_s") if key not in motion]
    if missing:
        raise ProductionSpecError(
            f"{group.group_id} needs binding_motion {missing} before a late plan can be "
            "described; the motion window comes from configuration, not a default"
        )
    ends = {name: done[name].wet_tail_end_seconds() for name in unit.consumes_wet_tails_of}
    window = measured_motion_window(
        wet_tail_end_seconds_by_unit=ends,
        clock=member.clock,
        minimum_motion_s=motion["minimum_motion_s"],
        end_hold_s=motion["end_hold_s"],
        reserve_tail_s=member.reserve_tail_s,
        source_work_item_ids=[done[name].work_item_id for name in unit.consumes_wet_tails_of],
    )
    return {"measured_motion_window": window}


def group_stage_units(group: "CoreGroupRequest") -> list[dict[str, Any]]:
    """The shared unit graph with the members each unit actually delivers."""
    recipe = recipe_for_task_family(group.task_family)
    member_ids = [member.request_id for member in group.members]
    rows = []
    for unit in recipe.units_in_dependency_order():
        served = _members_served_by(recipe, unit, member_ids)
        rows.append({
            **unit.to_dict(),
            "scope_id": group_unit_scope_id(group.group_id, unit.unit_id),
            "member_request_ids": served,
        })
    return rows


def _members_served_by(
    recipe: GroupRecipe, unit: SharedUnitSpec, member_ids: Sequence[str]
) -> list[str]:
    """Which members this unit is produced for, by direct consumption.

    A transitive walk would be wrong: the late plan of a cross-time group is
    downstream of the early audio, which would make the static capture look
    like it delivered all four members instead of the two that share it.
    """
    if unit.internal_only:
        return []
    if unit.member_index is not None:
        return [member_ids[unit.member_index]]
    if unit.unit_kind == "assembly":
        return list(member_ids)
    if unit.unit_kind == "visual_capture":
        return [member_ids[column.member_index] for column in recipe.units
                if column.member_index is not None
                and column.visual_unit_id == unit.unit_id]
    if unit.unit_kind == "visual_plan":
        served: list[str] = []
        for capture in recipe.units:
            if capture.unit_kind != "visual_capture":
                continue
            if unit.unit_id in capture.depends_on_units:
                served.extend(_members_served_by(recipe, capture, member_ids))
        return served
    return []


def _group_unit_work_item(
    group: "CoreGroupRequest",
    unit: SharedUnitSpec,
    *,
    attempt: int,
    done: Mapping[str, StageResult],
) -> StageWorkItem:
    recipe = recipe_for_task_family(group.task_family)
    member_ids = [member.request_id for member in group.members]
    served = _members_served_by(recipe, unit, member_ids)
    owner = group.members[unit.member_index] if unit.member_index is not None else group.members[0]
    upstream = {name: done[name] for name in unit.depends_on_units if name in done}
    return _make_work_item(
        owner, unit.stage, attempt=attempt, upstream=upstream,
        scope_id=group_unit_scope_id(group.group_id, unit.unit_id),
        unit=unit, member_request_ids=served,
        extra_payload=_group_unit_extra_payload(group, unit, done),
    )


def initial_group_work_items(group: "CoreGroupRequest") -> list[StageWorkItem]:
    """The units of one group that are already describable, produced once each."""
    return next_group_work_items(group, [])


def next_group_work_items(
    group: "CoreGroupRequest", results: Sequence[StageResult]
) -> list[StageWorkItem]:
    """Every shared unit whose real dependencies have passed, and no others.

    A late plan appears only after both early audio columns published their
    measured wet tails, and only if the declared movement time actually fits.
    """
    recipe = recipe_for_task_family(group.task_family)
    state = group_round_state(group, results)
    allowed = {unit.unit_id: unit.stage for unit in recipe.units}
    scope_to_unit = {group_unit_scope_id(group.group_id, unit_id): unit_id
                     for unit_id in allowed}
    latest = _latest_by_key(results, key_of=lambda item: scope_to_unit[item.scope_id],
                            allowed=allowed, owner=group.group_id)
    if state.is_blocked:
        return []
    items = []
    for unit_id in state.ready:
        unit = recipe.unit(unit_id)
        extra = _group_unit_extra_payload(group, unit, state.done)
        window = extra.get("measured_motion_window")
        if window is not None and not window["sufficient"]:
            continue
        items.append(_group_unit_work_item(
            group, unit, attempt=state.attempt_for(unit_id, latest), done=state.done))
    return items


def group_blockers(
    group: "CoreGroupRequest", results: Sequence[StageResult]
) -> list[dict[str, Any]]:
    """Why a group cannot advance: a failed round, or no legal movement time."""
    recipe = recipe_for_task_family(group.task_family)
    state = group_round_state(group, results)
    rows = [{"unit_id": unit_id, "code": "stage_not_passed", "reason": reason}
            for unit_id, reason in state.blocked]
    for unit_id in state.ready:
        unit = recipe.unit(unit_id)
        if not unit.consumes_wet_tails_of:
            continue
        window = _group_unit_extra_payload(group, unit, state.done).get("measured_motion_window")
        if window is not None and not window["sufficient"]:
            rows.append({
                "unit_id": unit_id,
                "code": "measured_reverberation_leaves_insufficient_movement_time",
                "reason": (
                    f"measured wet audio ends at {window['measured_wet_end_s']:g}s, leaving "
                    f"{window['available_motion_frames']} frames for a declared minimum of "
                    f"{window['minimum_motion_frames']}"
                ),
                "measured_motion_window": window,
            })
    return rows


def retry_stage_work_item(
    item: StageWorkItem, *, retry: RetryPolicy, reason: str
) -> StageWorkItem | None:
    """Repeat one stage under the same conditions, or report the budget is gone."""
    _text(reason, "retry reason")
    if item.attempt >= retry.attempts_per_stage:
        return None
    attempt = item.attempt + 1
    return replace(
        item,
        work_item_id=work_item_id(item.request_id, item.stage, attempt),
        attempt=attempt,
        fresh_output_relative=fresh_output_relative(item.request_id, item.stage, attempt),
        payload={**deepcopy(item.payload), "retry_reason": reason, "retry_of": item.work_item_id},
    )


def stage_protocol_summary() -> dict[str, Any]:
    """A compact description of the protocol for manifests and reports."""
    return {
        "schema": SCHEMA,
        "stages": list(STAGES),
        "removed_stages": dict(REMOVED_STAGES),
        "resource_kinds": list(RESOURCE_KINDS),
        "resource_kind_by_stage": dict(STAGE_RESOURCE_KIND),
        "legal_resource_kinds_by_stage": {stage: list(kinds)
                                          for stage, kinds in STAGE_RESOURCE_KINDS.items()},
        "resource_kind_profile": {kind: {"execution": profile[0], "runtime_context": profile[1]}
                                  for kind, profile in RESOURCE_KIND_PROFILE.items()},
        "published_facts_by_stage": {stage: list(keys) for stage, keys in STAGE_PUBLISHED_FACTS.items()},
        "stage_statuses": list(STAGE_STATUSES),
        "motion_timings": list(MOTION_TIMINGS),
        "motion_timing_by_task_family": dict(MOTION_TIMING_BY_TASK_FAMILY),
        "group_recipes": {family: recipe.to_dict() for family, recipe in sorted(GROUP_RECIPES.items())},
        "round_policy": (
            "the newest attempt of a unit is authoritative; a failed newest attempt stops the "
            "round, and a result produced from a superseded upstream attempt is stale"
        ),
        "expansion": "units are emitted as their dependencies pass; the graph is not static",
        "claim_boundary": "A described work item is not an executed stage or an admission claim.",
    }


def _resolve_clock(value: Any, *, owner: str) -> EpisodeClock:
    data = _mapping(value, owner)
    try:
        return EpisodeClock.from_mapping(data)
    except EpisodeClockError as exc:
        raise ProductionSpecError(f"{owner}: {exc}") from exc


def legal_resource_kinds(stage: str) -> tuple[str, ...]:
    """What a stage may occupy. The first entry is the default, not the only one."""
    if stage not in STAGES:
        raise ProductionSpecError(f"unknown stage: {stage}")
    return STAGE_RESOURCE_KINDS[stage]


def _resolve_resources(value: Any, *, owner: str) -> dict[str, ResourceRequest]:
    data = _mapping(value, owner)
    if "kind" in data:
        raise ProductionSpecError(
            f"{owner}.kind is per stage; declare it under {owner}.<stage>.kind"
        )
    resources: dict[str, ResourceRequest] = {}
    shared = {key: data[key] for key in data if key not in STAGES}
    for stage in STAGES:
        block = deep_merge_mappings(shared, _mapping(data.get(stage), f"{owner}.{stage}"))
        block.pop("kind_source", None)
        declared = block.pop("kind", None)
        legal = legal_resource_kinds(stage)
        kind = legal[0] if declared is None else str(declared)
        if kind not in legal:
            raise ProductionSpecError(
                f"{owner}.{stage}.kind must be one of {legal}, got {kind!r}"
            )
        resources[stage] = ResourceRequest.from_mapping(
            block, kind=kind, owner=f"{owner}.{stage}"
        )
    return resources


def _resolve_instances(data: Mapping[str, Any], *, owner: str) -> tuple[EntityInstanceSpec, ...]:
    declared = _sequence(data.get("instances"), owner + ".instances")
    assets = _sequence(data.get("source_asset_ids"), owner + ".source_asset_ids")
    if declared and assets:
        bound = [instance.get("asset_id") if isinstance(instance, Mapping) else None for instance in declared]
        if list(assets) != bound:
            _conflict(owner + " entity binding", assets, bound)
    if declared:
        return _checked_instances(
            tuple(
                EntityInstanceSpec.from_mapping(item, index=index, owner=owner + ".instances")
                for index, item in enumerate(declared)
            ),
            owner=owner,
        )
    classes = _sequence(data.get("source_classes"), owner + ".source_classes")
    if assets and classes and len(assets) != len(classes):
        _conflict(owner + " entity count", len(assets), len(classes))
    count = len(assets) or len(classes)
    if not count:
        raise ProductionSpecError(
            owner + " must declare instances, source_asset_ids or source_classes"
        )
    silent = _nonnegative_int(data.get("silent_count", 0), owner + ".silent_count")
    if silent >= count:
        raise ProductionSpecError(owner + ".silent_count must leave a speaking entity")
    instances = []
    for index in range(count):
        instances.append(
            EntityInstanceSpec(
                instance_id=f"source{index + 1}",
                source_class=str(classes[index]) if classes else "articulated_human",
                asset_id=str(assets[index]) if assets else None,
                speaking=None if silent == 0 else index < count - silent,
            )
        )
    return _checked_instances(tuple(instances), owner=owner)


def _checked_instances(
    instances: tuple[EntityInstanceSpec, ...], *, owner: str
) -> tuple[EntityInstanceSpec, ...]:
    ids = [instance.instance_id for instance in instances]
    if len(set(ids)) != len(ids):
        raise ProductionSpecError(f"{owner} entity instance_id values must be distinct: {ids}")
    if not any(instance.speaking is not False for instance in instances):
        raise ProductionSpecError(owner + " must keep a speaking entity instance")
    return instances


def _derived_qa_targets(
    instances: Sequence[EntityInstanceSpec], qa_ids: Sequence[str]
) -> tuple[QaTargetSpec, ...]:
    """Name the speaking instances and their own audible window, not event one."""
    speaking = [i.instance_id for i in instances if i.speaking is not False]
    targets = tuple(
        QaTargetSpec(
            qa_id=qa_id,
            target_instance_ids=tuple(speaking),
            event=EventSelector(kind="target_audible_window"),
            items=1,
            forms=tuple(get_requirements(qa_id)["forms"]),
            target_source="derived_from_speaking_instances",
        )
        for qa_id in qa_ids
    )
    return targets


def _qa_intent_source(
    data: Mapping[str, Any], *, qa_targets_declared: bool,
    request_extras: Mapping[str, Any] | None = None,
) -> str:
    """Separate request-level QA intent from per-target entity provenance."""
    extras = request_extras if isinstance(request_extras, Mapping) else {}
    branches = data.get("question_branches", extras.get("question_branches"))
    drive = data.get(
        "drive_sampler_from_questions",
        extras.get("drive_sampler_from_questions"),
    )
    if qa_targets_declared or bool(branches) or bool(drive):
        return "explicit_conditions"
    return "qa_ids_only"


def _resolve_qa(
    data: Mapping[str, Any], *, instances: Sequence[EntityInstanceSpec], owner: str
) -> tuple[
    tuple[str, ...], tuple[QaTargetSpec, ...], dict[str, int], int, str, bool
]:
    declared_ids = _sequence(data.get("qa_ids"), owner + ".qa_ids")
    qa_ids = tuple(_text(value, owner + ".qa_ids[]") for value in declared_ids) or tuple(QA_IDS)
    if len(set(qa_ids)) != len(qa_ids):
        raise ProductionSpecError(owner + ".qa_ids must be distinct")
    unknown = [value for value in qa_ids if value not in QA_IDS]
    if unknown:
        raise ProductionSpecError(f"{owner}.qa_ids are not in the unified catalog: {unknown}")
    instance_ids = [instance.instance_id for instance in instances]
    declared_targets = _sequence(data.get("qa_targets"), owner + ".qa_targets")
    qa_targets_declared = bool(declared_targets)
    if declared_targets:
        targets = tuple(
            QaTargetSpec.from_mapping(item, instance_ids=instance_ids, owner=owner + ".qa_targets")
            for item in declared_targets
        )
    else:
        targets = _derived_qa_targets(instances, qa_ids)
    quota_raw = _mapping(data.get("quota_by_qa"), owner + ".quota_by_qa")
    if quota_raw:
        quota = {_text(key, owner + ".quota_by_qa key"): _positive_int(value, f"{owner}.quota_by_qa[{key}]")
                 for key, value in quota_raw.items()}
        quota_source = "config"
    else:
        quota = {qa_id: sum(t.items for t in targets if t.qa_id == qa_id) or 1 for qa_id in qa_ids}
        quota_source = "derived_from_qa_targets" if declared_targets else "unit_default"
    items_per_type = _positive_int(data.get("items_per_type", 1), owner + ".items_per_type")
    return qa_ids, targets, quota, items_per_type, quota_source, qa_targets_declared


def _resolve_motion_timing(data: Mapping[str, Any], task_family: Any, *, owner: str) -> str:
    """Default from the task family; a forced family refuses any other value."""
    declared = data.get("motion_timing")
    forced = FORCED_MOTION_TIMING.get(task_family)
    if declared is None:
        if forced is not None:
            return forced
        return MOTION_TIMING_BY_TASK_FAMILY.get(task_family, "none")
    timing = _text(declared, owner + ".motion_timing")
    if timing not in MOTION_TIMINGS:
        raise ProductionSpecError(f"{owner}.motion_timing must be one of {MOTION_TIMINGS}")
    if forced is not None and timing != forced:
        raise ProductionSpecError(
            f"{owner}: {task_family} moves the entity {forced}, not {timing!r}; "
            "motion while a source is audible is a separate condition"
        )
    return timing


def _resolve_audio_layouts(value: Any, *, owner: str) -> tuple[AudioLayoutSpec, ...]:
    declared = _sequence(value, owner)
    if not declared:
        return (AudioLayoutSpec(layout_type="binaural", channel_count=2, role="primary"),)
    return tuple(
        AudioLayoutSpec.from_mapping(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(declared)
    )


def _build_request(
    data: Mapping[str, Any],
    *,
    kind: str,
    owner: str,
    group_id: str | None = None,
    task_family: str | None = None,
) -> ProductionRequest:
    request_id = _text(data.get("request_id", data.get("episode_id")), owner + ".request_id")
    declared_family = data.get("task_family", task_family)
    if task_family is not None and declared_family != task_family:
        _conflict(owner + ".task_family", task_family, declared_family)
    request_extras = _mapping(data.get("request_extras"), owner + ".request_extras")
    for key in ("question_branches", "drive_sampler_from_questions"):
        if key in data:
            request_extras.setdefault(key, deepcopy(data[key]))
    qa_input = dict(data)
    if "qa_targets" not in qa_input and "qa_targets" in request_extras:
        qa_input["qa_targets"] = deepcopy(request_extras["qa_targets"])
    instances = _resolve_instances(data, owner=owner)
    (
        qa_ids, targets, quota, items_per_type, quota_source,
        qa_targets_declared,
    ) = _resolve_qa(qa_input, instances=instances, owner=owner)
    qa_intent_source = _qa_intent_source(
        qa_input,
        qa_targets_declared=qa_targets_declared,
        request_extras=request_extras,
    )
    clock = _resolve_clock(data.get("clock"), owner=owner + ".clock")
    profile = _mapping(data.get("profile"), owner + ".profile")
    reserve_tail = data.get("reserve_tail_s", profile.get("reserve_tail_s"))
    if reserve_tail is None:
        raise ProductionSpecError(owner + ".reserve_tail_s is required")
    reserve_tail = _finite(reserve_tail, owner + ".reserve_tail_s")
    if "reserve_tail_s" in profile and _finite(profile["reserve_tail_s"], owner + ".profile.reserve_tail_s") != reserve_tail:
        _conflict(owner + ".reserve_tail_s", reserve_tail, profile["reserve_tail_s"])
    sound = _mapping(data.get("sound"), owner + ".sound")
    return ProductionRequest(
        request_id=request_id,
        kind=kind,
        room_id=_text(data.get("room_id"), owner + ".room_id"),
        instances=instances,
        clock=clock,
        rig=RigSpec.from_mapping(data.get("rig"), owner=owner + ".rig"),
        audio_layouts=_resolve_audio_layouts(data.get("audio_layouts"), owner=owner + ".audio_layouts"),
        qa_ids=qa_ids,
        qa_targets=targets,
        quota_by_qa=quota,
        items_per_type=items_per_type,
        profile=profile,
        reserve_tail_s=reserve_tail,
        post_assembly_convolution_gain=_finite(
            data.get("post_assembly_convolution_gain", 0.5),
            owner + ".post_assembly_convolution_gain",
        ),
        seed=_nonnegative_int(data.get("seed", 0), owner + ".seed"),
        resources=_resolve_resources(data.get("resources"), owner=owner + ".resources"),
        retry=RetryPolicy.from_mapping(data.get("retry"), owner=owner + ".retry"),
        sound_pool_path=None if sound.get("pool") is None else _text(sound["pool"], owner + ".sound.pool"),
        sound_selection=_mapping(sound.get("selection"), owner + ".sound.selection"),
        task_family=None if declared_family is None else _text(declared_family, owner + ".task_family"),
        group_id=group_id,
        member_role=None if data.get("member_role") is None else _text(data["member_role"], owner + ".member_role"),
        condition_group=None if data.get("condition_group") is None
        else _text(data["condition_group"], owner + ".condition_group"),
        motion_timing=_resolve_motion_timing(data, declared_family, owner=owner),
        quota_source=quota_source,
        request_extras=request_extras,
        qa_targets_declared=qa_targets_declared,
        qa_intent_source=qa_intent_source,
    )


@dataclass(frozen=True)
class ParsedProductionConfig:
    """One configuration resolved into ordinary Episodes and core groups."""

    schema: str
    batch_id: str
    seed: int
    episodes: tuple[ProductionRequest, ...]
    core_groups: tuple[CoreGroupRequest, ...]
    coverage_quota: dict[str, Any] = field(default_factory=dict)

    def all_requests(self) -> tuple[ProductionRequest, ...]:
        members = tuple(member for group in self.core_groups for member in group.members)
        return self.episodes + members

    def request_by_id(self) -> dict[str, ProductionRequest]:
        return {request.request_id: request for request in self.all_requests()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "batch_id": self.batch_id,
            "seed": self.seed,
            "episode_count": len(self.episodes),
            "core_group_count": len(self.core_groups),
            "core_member_count": sum(len(group.members) for group in self.core_groups),
            "episodes": [request.to_dict() for request in self.episodes],
            "core_groups": [group.to_dict() for group in self.core_groups],
            "coverage_quota": deepcopy(self.coverage_quota),
            "stage_protocol": stage_protocol_summary(),
        }


def parse_production_config(config: Mapping[str, Any]) -> ParsedProductionConfig:
    """Resolve one configuration into ordinary Episodes and four-member groups.

    ``defaults`` supplies clock, rig, audio layouts, QA selection, resources and
    retry once; each Episode or member overrides what it actually changes.
    """
    data = _mapping(config, "config")
    schema = str(data.get("schema", SCHEMA))
    if schema != SCHEMA:
        raise ProductionSpecError(f"config.schema must be {SCHEMA!r}, got {schema!r}")
    batch_id = _text(data.get("batch_id"), "config.batch_id")
    seed = _nonnegative_int(data.get("seed", 0), "config.seed")
    defaults = _mapping(data.get("defaults"), "config.defaults")
    episodes = []
    for index, raw in enumerate(_sequence(data.get("episodes"), "config.episodes")):
        merged = deep_merge_mappings(defaults, _mapping(raw, f"config.episodes[{index}]"))
        merged.setdefault("seed", seed + index)
        episodes.append(_build_request(merged, kind="episode", owner=f"config.episodes[{index}]"))
    groups = []
    for index, raw in enumerate(_sequence(data.get("core_groups"), "config.core_groups")):
        block = _mapping(raw, f"config.core_groups[{index}]")
        group_id = _text(block.get("group_id"), f"config.core_groups[{index}].group_id")
        task_family = _text(block.get("task_family"), f"config.core_groups[{index}].task_family")
        group_defaults = deep_merge_mappings(defaults, _mapping(block.get("member_defaults"),
                                                                f"config.core_groups[{index}].member_defaults"))
        group_defaults.setdefault("seed", seed + 1000 * (index + 1))
        if block.get("room_id") is not None:
            group_defaults["room_id"] = block["room_id"]
        members = []
        raw_members = _sequence(block.get("members"), f"config.core_groups[{index}].members")
        for position, raw_member in enumerate(raw_members):
            owner = f"config.core_groups[{index}].members[{position}]"
            merged = deep_merge_mappings(group_defaults, _mapping(raw_member, owner))
            # All four rows are crossed conditions of one sampled world.  A
            # member may vary its declared intervention and audio content, but
            # an omitted seed inherits the group seed rather than sampling
            # another geometry world.
            merged.setdefault("seed", group_defaults["seed"])
            members.append(
                _build_request(
                    merged, kind="core_group_member", owner=owner,
                    group_id=group_id, task_family=task_family,
                )
            )
        groups.append(
            CoreGroupRequest(
                group_id=group_id,
                task_family=task_family,
                room_id=_text(block.get("room_id", members[0].room_id if members else None),
                              f"config.core_groups[{index}].room_id"),
                members=tuple(members),
                shared_audio_member_ids=tuple(
                    tuple(str(value) for value in _sequence(pair, "shared_audio_member_ids[]"))
                    for pair in _sequence(block.get("shared_audio_member_ids"),
                                          f"config.core_groups[{index}].shared_audio_member_ids")
                ),
                shared_visual_member_ids=tuple(
                    tuple(str(value) for value in _sequence(pair, "shared_visual_member_ids[]"))
                    for pair in _sequence(block.get("shared_visual_member_ids"),
                                          f"config.core_groups[{index}].shared_visual_member_ids")
                ),
            )
        )
    if not episodes and not groups:
        raise ProductionSpecError("config must declare episodes, core_groups or both")
    request_ids = [request.request_id for request in (tuple(episodes) + tuple(
        member for group in groups for member in group.members))]
    duplicates = sorted({value for value in request_ids if request_ids.count(value) > 1})
    if duplicates:
        raise ProductionSpecError(f"request_id values must be unique across the config: {duplicates}")
    return ParsedProductionConfig(
        schema=schema,
        batch_id=batch_id,
        seed=seed,
        episodes=tuple(episodes),
        core_groups=tuple(groups),
        coverage_quota=_mapping(data.get("coverage_quota"), "config.coverage_quota"),
    )


def _legacy_stage_resource(stage: str, block: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a declared kind, drop one that an earlier build forced in code.

    A saved block written before the execution/runtime_context split carries no
    `execution` key. Its `kind` came from a table that forced every acoustic
    stage onto a GPU slot, so restoring it would keep a CPU render holding a GPU
    slot. Such a block falls back to the current stage default; everything else
    in it, including the adapter and thread count, is preserved.
    """
    value = {key: item for key, item in block.items() if key != "runtime_context"}
    if "execution" not in value:
        value.pop("kind", None)
        value["kind_source"] = "stage_default_after_pre_two_axis_block"
    value.pop("execution", None)
    return value


def production_request_from_legacy(
    request: Mapping[str, Any], *, request_id: str | None = None, kind: str = "episode"
) -> ProductionRequest:
    """Read a saved ``avengine_native_qa_room_request_v1`` back into a spec.

    Old entry points keep working: an existing request stays executable and can
    still be described as stage work items without being rewritten by hand.
    """
    data = _mapping(request, "legacy_request")
    profile = _mapping(data.get("profile"), "legacy_request.profile")
    entities = _mapping(data.get("entities"), "legacy_request.entities")
    camera = _mapping(data.get("camera"), "legacy_request.camera")
    sound_selection = _mapping(data.get("sound_selection"), "legacy_request.sound_selection")
    sampling = _mapping(data.get("qa_sampling"), "legacy_request.qa_sampling")
    production = _mapping(data.get("production"), "legacy_request.production")
    stage_resources = _mapping(production.get("stage_resources"),
                               "legacy_request.production.stage_resources")
    extras = {
        key: deepcopy(value)
        for key, value in data.items()
        if key not in {
            "schema", "episode_id", "room_id", "seed", "sampling_policy", "camera",
            "frame_count", "frame_rate_hz", "sample_rate_hz", "entities", "entity_instances",
            "profile", "qa_ids", "qa_sampling", "qa_targets", "quota_by_qa", "audio_layouts",
            "post_assembly_convolution_gain", "source_asset_ids", "sound_pool", "sound_selection",
            "task_family", "group_id", "member_role", "condition_group", "production",
            "motion_timing",
        }
    }
    spec_input = {
        "request_id": request_id or data.get("episode_id"),
        "room_id": data.get("room_id"),
        "seed": data.get("seed", 0),
        "clock": {
            "frame_count": data.get("frame_count", 150),
            "frame_rate_hz": data.get("frame_rate_hz", 15),
            "sample_rate_hz": data.get("sample_rate_hz", 16000),
        },
        "rig": {
            "resolution_hw": camera.get("resolution_hw", [720, 1280]),
            "fov_deg": camera.get("fov_deg", 85.0),
            "height_above_floor_m": camera.get("height_above_floor_m"),
            "motion": camera.get("motion", CAMERA_MOTION),
        },
        "profile": profile,
        "reserve_tail_s": profile.get("reserve_tail_s", 3.0),
        "post_assembly_convolution_gain": data.get("post_assembly_convolution_gain", 0.5),
        "qa_ids": data.get("qa_ids"),
        "qa_targets": data.get("qa_targets"),
        "quota_by_qa": data.get("quota_by_qa"),
        "items_per_type": sampling.get("items_per_type", 1),
        "audio_layouts": data.get("audio_layouts"),
        "sound": {"pool": data.get("sound_pool"), "selection": sound_selection},
        "task_family": data.get("task_family"),
        "member_role": data.get("member_role"),
        "condition_group": data.get("condition_group"),
        "motion_timing": data.get("motion_timing"),
        "resources": {stage: _legacy_stage_resource(stage, block)
                      for stage, block in stage_resources.items()
                      if isinstance(block, Mapping)},
        "retry": production.get("retry"),
        "request_extras": extras,
    }
    # Keep solver and instance declarations not represented by the compact rig.
    # They must survive a normal legacy -> staged -> legacy request round trip.
    extras["camera"] = deepcopy(camera)
    extras["entities"] = deepcopy(entities)
    residual_sampling = {key: deepcopy(value) for key, value in sampling.items()
                         if key != "items_per_type"}
    if residual_sampling:
        extras["qa_sampling"] = deep_merge_mappings(
            _mapping(extras.get("qa_sampling"), "legacy_request.qa_sampling"), residual_sampling)
    declared_instances = data.get("entity_instances")
    if declared_instances:
        spec_input["instances"] = declared_instances
    else:
        spec_input["source_asset_ids"] = data.get("source_asset_ids")
        spec_input["source_classes"] = entities.get("source_classes")
        spec_input["silent_count"] = entities.get("silent_count", 0)
    if not spec_input.get("source_asset_ids") and not declared_instances:
        total = entities.get("total_count")
        if isinstance(total, Integral) and not isinstance(total, bool):
            spec_input["source_classes"] = ["articulated_human"] * int(total)
    return _build_request(
        spec_input, kind=kind, owner="legacy_request", group_id=data.get("group_id")
    )


__all__ = [
    "AUDIO_LAYOUT_ROLES",
    "AudioLayoutSpec",
    "CORE_TASK_FAMILIES",
    "CoreGroupRequest",
    "EVENT_SELECTOR_KINDS",
    "EntityInstanceSpec",
    "EventSelector",
    "LEGACY_REQUEST_SCHEMA",
    "ParsedProductionConfig",
    "ProductionRequest",
    "ProductionSpecError",
    "QA_INTENT_SOURCES",
    "QaTargetSpec",
    "RESOURCE_KINDS",
    "ResourceRequest",
    "RetryPolicy",
    "RigSpec",
    "SCHEMA",
    "STAGES",
    "STAGE_PUBLISHED_FACTS",
    "STAGE_RESOURCE_KIND",
    "STAGE_RESOURCE_KINDS",
    "STAGE_STATUSES",
    "EXECUTION_SLOTS",
    "RUNTIME_CONTEXTS",
    "RESOURCE_KIND_PROFILE",
    "REMOVED_STAGES",
    "MOTION_TIMINGS",
    "MOTION_TIMING_BY_TASK_FAMILY",
    "FORCED_MOTION_TIMING",
    "PLAN_EQUIVALENCE_MODES",
    "VISUAL_INTERVENTION_MODES",
    "QUERY_IDENTITY_POLICIES",
    "AUDIO_CONTENT_SCOPES",
    "SOUND_SELECTION_POLICY_DEFAULTS",
    "SOUND_SELECTION_POLICY_FIELDS",
    "GROUP_RECIPES",
    "GroupRecipe",
    "RoundState",
    "SharedUnitSpec",
    "UNIT_KINDS",
    "group_blockers",
    "group_round_state",
    "group_stage_units",
    "group_unit_scope_id",
    "initial_group_work_items",
    "legal_resource_kinds",
    "measured_motion_window",
    "next_group_work_items",
    "recipe_for_task_family",
    "request_round_state",
    "SOURCE_CLASSES",
    "StageResult",
    "StageWorkItem",
    "deep_merge_mappings",
    "normalize_sound_selection_policy",
    "normalize_sound_selection_content",
    "fresh_output_relative",
    "initial_stage_work_items",
    "next_stage_work_items",
    "parse_production_config",
    "production_request_from_legacy",
    "retry_stage_work_item",
    "stage_protocol_summary",
    "work_item_id",
]
