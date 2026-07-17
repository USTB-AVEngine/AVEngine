"""Strict multi-source bridge to the AVEngine Habitat/RLR fork.

M4 treats source identity as data, not as a native list position.  Callers may
provide sources in any order; this module realizes them in a canonical ID
order and indexes every returned impulse response by the explicit
``(listener_id, source_id)`` pair.

The module intentionally reuses M3's hash-checked acoustic package loader and
native upload verification.  M4 changes endpoint cardinality and output
layouts; it does not weaken or duplicate the M3 geometry/material contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import resource
import statistics
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import sha256_file
from avengine.m3.runtime import (
    CompiledAcousticScene,
    RLRChannelLayout,
    RLRSimulationConfig,
    RuntimeAnchor,
    RuntimeContractError,
    RuntimeExecutionError,
    RuntimeUnavailableError,
    _native_configuration,
    _upload_report,
    _verify_upload_report,
    load_habitat_runtime,
)


FOA_LAYOUT_ID = "rlr_foa_acn_n3d_world_v1"
BINAURAL_LAYOUT_ID = "rlr_binaural_lr_v1"


@dataclass(frozen=True)
class M4SimulationConfig:
    """RLR configuration with M4's explicit reset/temporal semantics.

    M3's configuration deliberately requires full indirect propagation and
    rejects temporal coherence because every material repeat is independent.
    M4 additionally needs direct-only direction probes and a tested persistent
    temporal sequence, so it owns a separate policy while retaining the same
    native fields.
    """

    frequency_bands: int
    direct_sh_order: int
    indirect_sh_order: int
    direct_ray_count: int
    indirect_ray_count: int
    indirect_ray_depth: int
    source_ray_count: int
    source_ray_depth: int
    max_diffraction_order: int
    thread_count: int
    sample_rate_hz: float
    max_ir_seconds: float
    unit_scale: float
    global_volume: float
    speed_of_sound_m_s: float
    direct: bool
    indirect: bool
    diffraction: bool
    transmission: bool
    mesh_simplification: bool
    temporal_coherence: bool
    channel_layout: RLRChannelLayout

    _INTEGER_FIELDS = RLRSimulationConfig._INTEGER_FIELDS
    _FLOAT_FIELDS = RLRSimulationConfig._FLOAT_FIELDS
    _BOOLEAN_FIELDS = RLRSimulationConfig._BOOLEAN_FIELDS

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "M4SimulationConfig":
        if not isinstance(value, Mapping):
            raise RuntimeContractError("simulation must be an object")
        expected = {
            *cls._INTEGER_FIELDS,
            *cls._FLOAT_FIELDS,
            *cls._BOOLEAN_FIELDS,
            "channel_layout",
        }
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing or unknown:
            raise RuntimeContractError(
                "simulation fields differ from the explicit RLR contract; "
                f"missing={missing}, unknown={unknown}"
            )
        arguments = {field: value[field] for field in expected if field != "channel_layout"}
        arguments["channel_layout"] = RLRChannelLayout.from_mapping(
            value["channel_layout"]
        )
        result = cls(**arguments)
        result.validate()
        return result

    @classmethod
    def from_m3(cls, value: RLRSimulationConfig) -> "M4SimulationConfig":
        return cls.from_mapping(value.to_dict())

    def validate(self) -> None:
        for field in self._INTEGER_FIELDS:
            value = getattr(self, field)
            minimum = 0 if field in {
                "direct_sh_order",
                "indirect_sh_order",
                "max_diffraction_order",
            } else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise RuntimeContractError(
                    f"simulation.{field} must be an integer >= {minimum}"
                )
        for field in self._FLOAT_FIELDS:
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise RuntimeContractError(
                    f"simulation.{field} must be a finite positive number"
                )
        for field in self._BOOLEAN_FIELDS:
            if not isinstance(getattr(self, field), bool):
                raise RuntimeContractError(f"simulation.{field} must be boolean")
        if self.frequency_bands not in {4, 8}:
            raise RuntimeContractError("simulation.frequency_bands must be 4 or 8")
        if self.direct_sh_order > 5 or self.indirect_sh_order > 5:
            raise RuntimeContractError("simulation SH orders must be in [0, 5]")
        if not math.isclose(self.unit_scale, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeContractError("M4 requires meter-native unit_scale=1.0")
        if not math.isclose(
            self.speed_of_sound_m_s, 343.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeContractError("M4 pins speed_of_sound_m_s=343.0")
        if self.mesh_simplification:
            raise RuntimeContractError("M4 forbids native mesh simplification")
        if not self.direct:
            raise RuntimeContractError("M4 requires direct propagation")
        if not self.indirect and any(
            (self.diffraction, self.transmission, self.temporal_coherence)
        ):
            raise RuntimeContractError(
                "direct-only probes must disable diffraction, transmission and temporal coherence"
            )
        self.channel_layout.validate()
        if self.channel_layout.layout_type == "ambisonics":
            root = math.isqrt(self.channel_layout.channel_count)
            if root * root != self.channel_layout.channel_count or root - 1 > 5:
                raise RuntimeContractError(
                    "ambisonic channel_count must encode an order in [0, 5]"
                )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["channel_layout"] = self.channel_layout.to_dict()
        return value


@dataclass(frozen=True)
class NamedPairIR:
    """One copied, independently addressable listener/source response."""

    listener_id: str
    source_id: str
    sample_rate_hz: float
    samples: np.ndarray
    layout_type: str
    layout_id: str
    channel_labels: tuple[str, ...]
    normalization: str
    coordinate_frame: str


@dataclass(frozen=True)
class MultiSourceRenderResult:
    """All pair responses and the receipts for one native simulation."""

    pairs: tuple[NamedPairIR, ...]
    requested_source_order: tuple[str, ...]
    canonical_native_source_order: tuple[str, ...]
    listener_id: str
    sample_rate_hz: float
    layout_type: str
    layout_id: str
    runtime: dict[str, Any]
    upload_report: dict[str, Any]
    endpoint_receipts: dict[str, Any]
    indirect_ray_efficiency: float
    timing: dict[str, Any]

    def by_pair(self) -> dict[tuple[str, str], NamedPairIR]:
        result = {(pair.listener_id, pair.source_id): pair for pair in self.pairs}
        if len(result) != len(self.pairs):
            raise RuntimeContractError("duplicate named pair in render result")
        return result


def _stable_endpoint_id(value: str, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeContractError(f"{owner} must be a non-empty string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeContractError(
            f"M4 {owner} must be ASCII so native ordering is portable"
        ) from exc
    if len(encoded) > 128:
        raise RuntimeContractError(f"M4 {owner} must contain at most 128 ASCII bytes")
    first = encoded[0]
    if not (
        ord("a") <= first <= ord("z")
        or ord("A") <= first <= ord("Z")
        or ord("0") <= first <= ord("9")
    ):
        raise RuntimeContractError(
            f"M4 {owner} must start with an ASCII letter or digit"
        )
    if any(
        not (
            ord("a") <= character <= ord("z")
            or ord("A") <= character <= ord("Z")
            or ord("0") <= character <= ord("9")
            or character in b"._-"
        )
        for character in encoded
    ):
        raise RuntimeContractError(
            f"M4 {owner} may contain only ASCII letters, digits, '.', '_' and '-'"
        )
    return value


def _stable_source_id(value: str) -> str:
    return _stable_endpoint_id(value, owner="source IDs")


def canonical_source_order(sources: Sequence[RuntimeAnchor]) -> tuple[str, ...]:
    """Return the bytewise, cross-language native realization order."""

    ids = tuple(_stable_source_id(source.anchor_id) for source in sources)
    if len(set(ids)) != len(ids):
        raise RuntimeContractError("source IDs must be unique")
    return tuple(sorted(ids, key=lambda value: value.encode("ascii")))


def _layout_contract(layout_type: str, channel_count: int) -> dict[str, Any]:
    if layout_type == "ambisonics" and channel_count == 4:
        return {
            "layout_id": FOA_LAYOUT_ID,
            "channel_labels": ("W", "Y", "Z", "X"),
            "normalization": "N3D",
            "coordinate_frame": "avengine_world",
        }
    if layout_type == "binaural" and channel_count == 2:
        return {
            "layout_id": BINAURAL_LAYOUT_ID,
            "channel_labels": ("left", "right"),
            "normalization": "not_applicable",
            "coordinate_frame": "listener_local",
        }
    if layout_type == "mono" and channel_count == 1:
        return {
            "layout_id": "rlr_mono_v1",
            "channel_labels": ("mono",),
            "normalization": "not_applicable",
            "coordinate_frame": "not_directional",
        }
    raise RuntimeContractError(
        f"unsupported M4 output layout {layout_type!r}/{channel_count} channels"
    )


def simulation_with_layout(
    simulation: M4SimulationConfig | RLRSimulationConfig,
    *,
    layout_type: str,
    channel_count: int,
) -> M4SimulationConfig:
    """Clone a validated simulation with one explicit output layout."""

    value = simulation.to_dict()
    value["channel_layout"] = {
        "type": layout_type,
        "channel_count": channel_count,
    }
    return M4SimulationConfig.from_mapping(value)


def direct_only_simulation(
    simulation: M4SimulationConfig | RLRSimulationConfig,
    *,
    layout_type: str,
    channel_count: int,
) -> M4SimulationConfig:
    """Derive the deterministic direct-path configuration used by axis probes."""

    value = simulation.to_dict()
    value.update(
        {
            "indirect": False,
            "diffraction": False,
            "transmission": False,
            "temporal_coherence": False,
            "channel_layout": {
                "type": layout_type,
                "channel_count": channel_count,
            },
        }
    )
    return M4SimulationConfig.from_mapping(value)


def _native_layout(habitat_module: Any, layout_type: str) -> Any:
    member = {
        "mono": "Mono",
        "binaural": "Binaural",
        "ambisonics": "Ambisonics",
    }.get(layout_type)
    if member is None or not hasattr(habitat_module.RLRChannelLayoutType, member):
        raise RuntimeContractError(f"native RLR layout {layout_type!r} is unavailable")
    return getattr(habitat_module.RLRChannelLayoutType, member)


def _float32_receipt_vector(
    value: Any,
    requested: Sequence[float],
    *,
    owner: str,
) -> list[float]:
    try:
        observed = np.asarray(value, dtype=np.float64)
        expected = np.asarray(requested, dtype=np.float32).astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError(f"{owner} is not a numeric vector: {exc}") from exc
    if (
        observed.shape != expected.shape
        or observed.ndim != 1
        or not np.all(np.isfinite(observed))
        or not np.array_equal(observed, expected)
    ):
        raise RuntimeContractError(f"{owner} differs from the float32 native request")
    return [float(item) for item in observed]


def _float32_receipt_scalar(value: Any, requested: float, *, owner: str) -> float:
    if isinstance(value, bool):
        raise RuntimeContractError(f"{owner} must be a native float readback")
    try:
        observed = float(value)
        expected = float(np.float32(requested))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError(f"{owner} is not a native float readback: {exc}") from exc
    if not math.isfinite(observed) or observed != expected:
        raise RuntimeContractError(f"{owner} differs from the float32 native request")
    return observed


def _native_index(value: Any, expected: int, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeContractError(f"{owner} must be an integer")
    observed = int(value)
    if observed != expected:
        raise RuntimeContractError(f"{owner} differs from canonical native order")
    return observed


def _native_registration_receipts(
    context: Any,
    habitat_module: Any,
    sources: Sequence[RuntimeAnchor],
    listener: RuntimeAnchor,
    *,
    canonical_order: Sequence[str],
    layout_type: str,
    channel_count: int,
    hrtf_file_path: str,
) -> dict[str, Any]:
    by_id = {source.anchor_id: source for source in sources}
    required_methods = (
        "source_registration_receipts",
        "listener_registration_receipts",
        "source_index",
        "listener_index",
    )
    missing = [name for name in required_methods if not hasattr(context, name)]
    if missing:
        raise RuntimeContractError(
            "native M4 binding lacks endpoint receipt methods: " + ", ".join(missing)
        )
    try:
        source_receipts = list(context.source_registration_receipts())
        listener_receipts = list(context.listener_registration_receipts())
    except Exception as exc:
        raise RuntimeContractError(f"native endpoint receipts are unreadable: {exc}") from exc
    if len(source_receipts) != len(canonical_order):
        raise RuntimeContractError("native source receipt count differs from request")
    if len(listener_receipts) != 1:
        raise RuntimeContractError("native listener receipt count must be exactly one")

    portable_sources: list[dict[str, Any]] = []
    for index, (source_id, receipt) in enumerate(
        zip(canonical_order, source_receipts, strict=True)
    ):
        try:
            observed_id = receipt.source_id
        except AttributeError as exc:
            raise RuntimeContractError(
                f"native source receipt {index} lacks a stable ID: {exc}"
            ) from exc
        if not isinstance(observed_id, str):
            raise RuntimeContractError(
                f"native source receipt {index} ID must be a string"
            )
        _stable_source_id(observed_id)
        if observed_id != source_id:
            raise RuntimeContractError("native source receipt order/ID differs from request")
        source = by_id[source_id]
        try:
            receipt_index = _native_index(
                receipt.native_index,
                index,
                owner=f"source receipt {source_id!r} native_index",
            )
            lookup_index = _native_index(
                context.source_index(source_id),
                index,
                owner=f"source lookup {source_id!r} native_index",
            )
            position = _float32_receipt_vector(
                receipt.position,
                source.position_m,
                owner=f"source receipt {source_id!r} position",
            )
            radius = _float32_receipt_scalar(
                receipt.radius,
                source.radius_m,
                owner=f"source receipt {source_id!r} radius",
            )
            native_realized = receipt.native_realized
        except AttributeError as exc:
            raise RuntimeContractError(
                f"native source receipt {source_id!r} is incomplete: {exc}"
            ) from exc
        if receipt_index != lookup_index:
            raise RuntimeContractError("native source receipt/index lookup disagrees")
        if native_realized is not True:
            raise RuntimeContractError(
                f"native source receipt {source_id!r} was not realized by simulation"
            )
        portable_sources.append(
            {
                "source_id": observed_id,
                "canonical_native_index": receipt_index,
                "position_m": position,
                "radius_m": radius,
                "native_realized": True,
            }
        )

    receipt = listener_receipts[0]
    try:
        observed_listener_id = receipt.listener_id
    except AttributeError as exc:
        raise RuntimeContractError(f"native listener receipt lacks a stable ID: {exc}") from exc
    if not isinstance(observed_listener_id, str):
        raise RuntimeContractError("native listener receipt ID must be a string")
    _stable_endpoint_id(observed_listener_id, owner="listener ID")
    if observed_listener_id != listener.anchor_id:
        raise RuntimeContractError("native listener receipt ID differs from request")
    expected_layout = _native_layout(habitat_module, layout_type)
    try:
        listener_index = _native_index(
            receipt.native_index, 0, owner="listener receipt native_index"
        )
        lookup_index = _native_index(
            context.listener_index(listener.anchor_id),
            0,
            owner="listener lookup native_index",
        )
        position = _float32_receipt_vector(
            receipt.position,
            listener.position_m,
            owner="listener receipt position",
        )
        orientation = _float32_receipt_vector(
            receipt.orientation_wxyz,
            listener.orientation_wxyz,
            owner="listener receipt orientation_wxyz",
        )
        radius = _float32_receipt_scalar(
            receipt.radius,
            listener.radius_m,
            owner="listener receipt radius",
        )
        observed_layout = receipt.layout_type
        observed_channel_count = receipt.channel_count
        observed_hrtf = receipt.hrtf_file_path
        native_realized = receipt.native_realized
    except AttributeError as exc:
        raise RuntimeContractError(f"native listener receipt is incomplete: {exc}") from exc
    if listener_index != lookup_index:
        raise RuntimeContractError("native listener receipt/index lookup disagrees")
    if observed_layout != expected_layout:
        raise RuntimeContractError("native listener receipt layout differs from request")
    if (
        isinstance(observed_channel_count, bool)
        or not isinstance(observed_channel_count, (int, np.integer))
        or int(observed_channel_count) != channel_count
    ):
        raise RuntimeContractError("native listener receipt channel count differs from request")
    if not isinstance(observed_hrtf, str) or observed_hrtf != hrtf_file_path:
        raise RuntimeContractError("native listener receipt HRTF path differs from request")
    if native_realized is not True:
        raise RuntimeContractError("native listener receipt was not realized by simulation")

    return {
        "authority": "native_registration_readback",
        "sources": portable_sources,
        "listener": {
            "listener_id": observed_listener_id,
            "canonical_native_index": listener_index,
            "position_m": position,
            "orientation_wxyz": orientation,
            "radius_m": radius,
            "layout_type": layout_type,
            "channel_count": int(observed_channel_count),
            "hrtf_mode": (
                "external_file" if hrtf_file_path else "rlr_builtin_default"
            ),
            "hrtf_file_path": observed_hrtf,
            "native_realized": True,
        },
    }


def _rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; this repository's supported runtime is
    # Linux.  The platform is recorded in the surrounding runtime receipt.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def render_named_sources(
    scene: CompiledAcousticScene,
    simulation: M4SimulationConfig | RLRSimulationConfig,
    *,
    sources: Sequence[RuntimeAnchor],
    listener: RuntimeAnchor,
    layout_type: str | None = None,
    channel_count: int | None = None,
    hrtf_file_path: str = "",
) -> MultiSourceRenderResult:
    """Render all named source pairs through one fresh native RLR context.

    The caller's source order is preserved as evidence only.  Native endpoint
    realization is canonical by ID, which removes RLR's native-index-bound RNG
    stream from the public ordering contract.
    """

    if not isinstance(scene, CompiledAcousticScene):
        raise RuntimeContractError("scene must be a validated CompiledAcousticScene")
    if not isinstance(simulation, (M4SimulationConfig, RLRSimulationConfig)):
        raise RuntimeContractError("simulation must be an explicit RLR configuration")
    if not isinstance(listener, RuntimeAnchor):
        raise RuntimeContractError("listener must be RuntimeAnchor")
    if not sources:
        raise RuntimeContractError("at least one named source is required")
    if any(not isinstance(source, RuntimeAnchor) for source in sources):
        raise RuntimeContractError("every source must be RuntimeAnchor")
    _stable_endpoint_id(listener.anchor_id, owner="listener ID")
    requested_order = tuple(source.anchor_id for source in sources)
    canonical_order = canonical_source_order(sources)
    by_id = {source.anchor_id: source for source in sources}

    selected_layout = (
        simulation.channel_layout.layout_type
        if layout_type is None
        else layout_type
    )
    selected_channel_count = (
        simulation.channel_layout.channel_count
        if channel_count is None
        else channel_count
    )
    selected = simulation_with_layout(
        simulation,
        layout_type=selected_layout,
        channel_count=selected_channel_count,
    )
    contract = _layout_contract(selected_layout, selected_channel_count)
    if not isinstance(hrtf_file_path, str):
        raise RuntimeContractError("hrtf_file_path must be a string")
    if hrtf_file_path and selected_layout != "binaural":
        raise RuntimeContractError("an HRTF file is valid only for binaural output")
    if hrtf_file_path and not Path(hrtf_file_path).resolve().is_file():
        raise RuntimeContractError("declared HRTF file does not exist")

    habitat_module, runtime_report = load_habitat_runtime()
    native_configuration, config_readback = _native_configuration(
        habitat_module, selected
    )
    runtime_report["configuration_readback"] = config_readback
    runtime_report["output_contract"] = {
        **contract,
        "layout_type": selected_layout,
        "channel_count": selected_channel_count,
    }
    wall_start = time.perf_counter_ns()
    process_start = time.process_time_ns()
    rss_before = _rss_bytes()
    try:
        context = habitat_module.RLRAcousticContext(native_configuration)
        with tempfile.TemporaryDirectory(prefix="avengine-m4-rlr-db-") as temp_dir:
            private_database = Path(temp_dir) / "material_database.json"
            private_database.write_bytes(scene.material_database_bytes)
            if sha256_file(private_database) != scene.material_database_sha256:
                raise RuntimeContractError(
                    "private RLR material database snapshot hash differs"
                )
            raw_upload = context.load_acoustic_scene(
                str(private_database),
                list(scene.material_categories),
                list(scene.objects),
            )
        upload = _upload_report(raw_upload)
        _verify_upload_report(scene, upload)
        for source_id in canonical_order:
            source = by_id[source_id]
            context.add_source(source_id, source.position_m, source.radius_m)
        context.add_listener(
            listener.anchor_id,
            listener.position_m,
            listener.orientation_wxyz,
            _native_layout(habitat_module, selected_layout),
            selected_channel_count,
            listener.radius_m,
            hrtf_file_path,
        )
        raw_irs = context.simulate_owned()
        endpoint_receipts = _native_registration_receipts(
            context,
            habitat_module,
            sources,
            listener,
            canonical_order=canonical_order,
            layout_type=selected_layout,
            channel_count=selected_channel_count,
            hrtf_file_path=hrtf_file_path,
        )
        ray_efficiency = float(context.indirect_ray_efficiency())
    except (RuntimeContractError, RuntimeUnavailableError):
        raise
    except Exception as exc:
        raise RuntimeExecutionError(f"modern multi-source RLR simulation failed: {exc}") from exc
    process_end = time.process_time_ns()
    wall_end = time.perf_counter_ns()
    rss_after = _rss_bytes()

    expected_pairs = {(listener.anchor_id, source_id) for source_id in canonical_order}
    observed: dict[tuple[str, str], NamedPairIR] = {}
    for raw_ir in raw_irs:
        pair_key = (
            str(getattr(raw_ir, "listener_id", "")),
            str(getattr(raw_ir, "source_id", "")),
        )
        if pair_key in observed:
            raise RuntimeContractError(f"RLR returned duplicate pair {pair_key!r}")
        try:
            sample_rate = float(raw_ir.sample_rate)
            native_channel_count = int(raw_ir.channel_count)
            sample_count = int(raw_ir.sample_count)
            samples = np.array(raw_ir.samples, dtype="<f4", order="C", copy=True)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeContractError(f"RLR returned malformed owned IR: {exc}") from exc
        if pair_key not in expected_pairs:
            raise RuntimeContractError(f"RLR returned undeclared pair {pair_key!r}")
        if not math.isclose(
            sample_rate,
            selected.sample_rate_hz,
            rel_tol=1.0e-6,
            abs_tol=1.0e-3,
        ):
            raise RuntimeContractError("RLR IR sample rate differs from request")
        if (
            native_channel_count != selected_channel_count
            or samples.shape != (selected_channel_count, sample_count)
            or sample_count < 2
            or not samples.flags.c_contiguous
        ):
            raise RuntimeContractError("RLR IR shape/count metadata is inconsistent")
        if not np.all(np.isfinite(samples)) or not np.any(samples != 0.0):
            raise RuntimeContractError("RLR IR must be finite and non-silent")

        # Newer M4 fork builds expose these fields on every IR.  Reject a
        # disagreement but retain compatibility with the M3 build so pure unit
        # tests and migration diagnostics remain importable.
        optional_metadata = {
            "layout_id": contract["layout_id"],
            "normalization": contract["normalization"],
            "coordinate_frame": contract["coordinate_frame"],
        }
        for field, expected in optional_metadata.items():
            if hasattr(raw_ir, field) and str(getattr(raw_ir, field)) != expected:
                raise RuntimeContractError(
                    f"native IR {field} differs from the M4 output contract"
                )
        if hasattr(raw_ir, "channel_labels") and tuple(raw_ir.channel_labels) != tuple(
            contract["channel_labels"]
        ):
            raise RuntimeContractError(
                "native IR channel labels differ from the M4 output contract"
            )
        observed[pair_key] = NamedPairIR(
            listener_id=pair_key[0],
            source_id=pair_key[1],
            sample_rate_hz=sample_rate,
            samples=samples,
            layout_type=selected_layout,
            layout_id=contract["layout_id"],
            channel_labels=tuple(contract["channel_labels"]),
            normalization=contract["normalization"],
            coordinate_frame=contract["coordinate_frame"],
        )
    if set(observed) != expected_pairs:
        missing = sorted(expected_pairs - set(observed))
        extra = sorted(set(observed) - expected_pairs)
        raise RuntimeContractError(f"RLR pair set mismatch; missing={missing}, extra={extra}")
    if not math.isfinite(ray_efficiency) or not 0.0 <= ray_efficiency <= 1.0:
        raise RuntimeContractError("RLR indirect ray efficiency is outside [0, 1]")

    ordered_pairs = tuple(
        observed[(listener.anchor_id, source_id)] for source_id in canonical_order
    )
    timing = {
        "wall_seconds": (wall_end - wall_start) / 1.0e9,
        "process_cpu_seconds": (process_end - process_start) / 1.0e9,
        "peak_rss_before_bytes": rss_before,
        "peak_rss_after_bytes": rss_after,
        "ir_payload_bytes": sum(pair.samples.nbytes for pair in ordered_pairs),
        "pair_count": len(ordered_pairs),
    }
    if not all(
        math.isfinite(float(timing[key])) and float(timing[key]) >= 0.0
        for key in ("wall_seconds", "process_cpu_seconds")
    ) or timing["wall_seconds"] <= 0.0:
        raise RuntimeContractError("runtime timing receipt is invalid")
    return MultiSourceRenderResult(
        pairs=ordered_pairs,
        requested_source_order=requested_order,
        canonical_native_source_order=canonical_order,
        listener_id=listener.anchor_id,
        sample_rate_hz=selected.sample_rate_hz,
        layout_type=selected_layout,
        layout_id=contract["layout_id"],
        runtime=runtime_report,
        upload_report=upload,
        endpoint_receipts=endpoint_receipts,
        indirect_ray_efficiency=ray_efficiency,
        timing=timing,
    )


def benchmark_source_scaling(
    scene: CompiledAcousticScene,
    simulation: M4SimulationConfig | RLRSimulationConfig,
    *,
    sources: Sequence[RuntimeAnchor],
    listener: RuntimeAnchor,
    repeat_count: int = 3,
) -> dict[str, Any]:
    """Measure fresh-context one-source versus N-source FOA execution.

    M4 records performance rather than inventing a platform-independent speed
    threshold.  Every repeat still validates the full named-pair contract.
    """

    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 1:
        raise RuntimeContractError("performance repeat_count must be a positive integer")
    if len(sources) < 2:
        raise RuntimeContractError("M4 performance comparison requires at least two sources")
    canonical = canonical_source_order(sources)
    by_id = {source.anchor_id: source for source in sources}
    conditions = {
        "one_source": (by_id[canonical[0]],),
        "multi_source": tuple(by_id[source_id] for source_id in canonical),
    }
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in conditions}
    for repeat_index in range(repeat_count):
        order = tuple(conditions) if repeat_index % 2 == 0 else tuple(reversed(conditions))
        for name in order:
            result = render_named_sources(
                scene,
                simulation,
                sources=conditions[name],
                listener=listener,
                layout_type="ambisonics",
                channel_count=4,
            )
            records[name].append({"repeat_index": repeat_index, **result.timing})
    summary: dict[str, Any] = {}
    for name, runs in records.items():
        wall = sorted(float(run["wall_seconds"]) for run in runs)
        cpu = sorted(float(run["process_cpu_seconds"]) for run in runs)
        summary[name] = {
            "source_count": len(conditions[name]),
            "pair_count": len(conditions[name]),
            "repeat_count": repeat_count,
            "median_wall_seconds": statistics.median(wall),
            "p95_wall_seconds": wall[min(len(wall) - 1, math.ceil(0.95 * len(wall)) - 1)],
            "median_process_cpu_seconds": statistics.median(cpu),
            "maximum_peak_rss_bytes": max(int(run["peak_rss_after_bytes"]) for run in runs),
            "median_ir_payload_bytes": int(
                statistics.median(int(run["ir_payload_bytes"]) for run in runs)
            ),
            "runs": runs,
        }
    one = float(summary["one_source"]["median_wall_seconds"])
    multi = float(summary["multi_source"]["median_wall_seconds"])
    summary["comparison"] = {
        "multi_to_one_median_wall_ratio": multi / one,
        "multi_pair_throughput_pairs_per_second": len(sources) / multi,
        "hard_speed_gate": None,
        "interpretation": "measurement_only_platform_dependent",
    }
    return summary


def exercise_endpoint_lifecycle(
    scene: CompiledAcousticScene,
    simulation: M4SimulationConfig | RLRSimulationConfig,
    *,
    sources: Sequence[RuntimeAnchor],
    listener: RuntimeAnchor,
    moved_distance_m: float = 0.25,
) -> dict[str, Any]:
    """Execute the M4 temporal update and reset boundary in one context.

    The first canonical source is moved directly away from the listener while
    its stable ID is retained.  A reset then destroys all state, reloads the
    same scene/endpoints and must reproduce the original first temporal frame
    sample-for-sample.
    """

    if len(sources) < 2:
        raise RuntimeContractError("lifecycle canary requires at least two sources")
    if (
        isinstance(moved_distance_m, bool)
        or not isinstance(moved_distance_m, (int, float))
        or not math.isfinite(float(moved_distance_m))
        or float(moved_distance_m) <= 0.0
    ):
        raise RuntimeContractError("moved_distance_m must be finite and positive")
    value = simulation.to_dict()
    value["temporal_coherence"] = True
    value["channel_layout"] = {"type": "ambisonics", "channel_count": 4}
    temporal = M4SimulationConfig.from_mapping(value)
    canonical = canonical_source_order(sources)
    by_id = {source.anchor_id: source for source in sources}
    habitat_module, runtime_report = load_habitat_runtime()
    native_configuration, config_readback = _native_configuration(
        habitat_module, temporal
    )
    runtime_report["configuration_readback"] = config_readback

    def upload(context: Any) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="avengine-m4-lifecycle-db-") as temp_dir:
            database = Path(temp_dir) / "material_database.json"
            database.write_bytes(scene.material_database_bytes)
            raw = context.load_acoustic_scene(
                str(database), list(scene.material_categories), list(scene.objects)
            )
        report = _upload_report(raw)
        _verify_upload_report(scene, report)
        return report

    def add_endpoints(context: Any) -> None:
        for source_id in canonical:
            source = by_id[source_id]
            context.add_source(source_id, source.position_m, source.radius_m)
        context.add_listener(
            listener.anchor_id,
            listener.position_m,
            listener.orientation_wxyz,
            _native_layout(habitat_module, "ambisonics"),
            4,
            listener.radius_m,
            "",
        )

    def copy_pairs(context: Any) -> dict[str, np.ndarray]:
        raw = context.simulate_owned()
        result: dict[str, np.ndarray] = {}
        for item in raw:
            listener_id = str(getattr(item, "listener_id", ""))
            source_id = str(getattr(item, "source_id", ""))
            if listener_id != listener.anchor_id or source_id not in canonical:
                raise RuntimeContractError("lifecycle returned an undeclared named pair")
            samples = np.array(item.samples, dtype="<f4", order="C", copy=True)
            if samples.ndim != 2 or samples.shape[0] != 4 or not np.all(np.isfinite(samples)):
                raise RuntimeContractError("lifecycle returned a malformed FOA IR")
            if source_id in result:
                raise RuntimeContractError("lifecycle returned a duplicate source pair")
            result[source_id] = samples
        if set(result) != set(canonical):
            raise RuntimeContractError("lifecycle did not return the complete source set")
        return result

    try:
        context = habitat_module.RLRAcousticContext(native_configuration)
        upload_report = upload(context)
        add_endpoints(context)
        fresh_first = copy_pairs(context)

        moved_source_id = canonical[0]
        original = np.asarray(by_id[moved_source_id].position_m, dtype=np.float64)
        listener_position = np.asarray(listener.position_m, dtype=np.float64)
        direction = original - listener_position
        distance = float(np.linalg.norm(direction))
        if distance <= 0.0:
            raise RuntimeContractError("lifecycle source cannot be co-located with listener")
        moved = original + direction / distance * float(moved_distance_m)
        if not hasattr(context, "set_source_position"):
            raise RuntimeContractError(
                "native M4 binding lacks set_source_position; rebuild the fork"
            )
        context.set_source_position(moved_source_id, moved.tolist())
        updated = copy_pairs(context)
        source_receipts = (
            list(context.source_registration_receipts())
            if hasattr(context, "source_registration_receipts")
            else []
        )

        context.reset()
        counts_after_reset = {
            "object_count": int(context.object_count),
            "source_count": int(context.source_count),
            "listener_count": int(context.listener_count),
        }
        if counts_after_reset != {
            "object_count": 0,
            "source_count": 0,
            "listener_count": 0,
        }:
            raise RuntimeContractError("reset did not destroy all native acoustic state")
        upload(context)
        add_endpoints(context)
        reset_first = copy_pairs(context)
    except (RuntimeContractError, RuntimeUnavailableError):
        raise
    except Exception as exc:
        raise RuntimeExecutionError(f"M4 endpoint lifecycle canary failed: {exc}") from exc

    reset_matches = all(
        np.array_equal(fresh_first[source_id], reset_first[source_id])
        for source_id in canonical
    )
    update_preserves_identity = (
        set(updated) == set(fresh_first) == set(canonical)
        and not np.array_equal(updated[moved_source_id], fresh_first[moved_source_id])
    )
    return {
        "fresh_first": fresh_first,
        "updated": updated,
        "reset_first": reset_first,
        "moved_source_id": moved_source_id,
        "moved_distance_m": float(moved_distance_m),
        "original_position_m": original.tolist(),
        "updated_position_m": moved.tolist(),
        "source_registration_receipts_after_update": [
            {
                "source_id": str(receipt.source_id),
                "canonical_native_index": int(receipt.native_index),
                "position_m": [float(value) for value in receipt.position],
                "radius_m": float(receipt.radius),
                "native_realized": bool(receipt.native_realized),
            }
            for receipt in source_receipts
        ],
        "reset_matches_fresh_first": reset_matches,
        "source_update_preserves_identity": update_preserves_identity,
        "temporal_sequence_executed": True,
        "reset_boundary_policy": "reset_reload_before_independent_episode",
        "counts_after_reset": counts_after_reset,
        "runtime": runtime_report,
        "upload_report": upload_report,
    }


__all__ = [
    "BINAURAL_LAYOUT_ID",
    "FOA_LAYOUT_ID",
    "M4SimulationConfig",
    "MultiSourceRenderResult",
    "NamedPairIR",
    "benchmark_source_scaling",
    "canonical_source_order",
    "direct_only_simulation",
    "exercise_endpoint_lifecycle",
    "render_named_sources",
    "simulation_with_layout",
]
