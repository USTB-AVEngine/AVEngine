"""Strict bridge from a compiled M3 package to ``RLRAcousticContext``.

The module deliberately has no import-time dependency on Habitat.  The pinned
runtime currently requires ``quaternion`` to be imported before
``habitat_sim``; importing in the opposite order can abort the interpreter
instead of raising a Python exception.  :func:`load_habitat_runtime` owns and
records that workaround so every M3 invocation follows the same order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from avengine.backends.rlr.sdk import (
    ExternalRlrSdkError,
    discover_external_rlr_sdk,
    preload_external_rlr_sdk,
    require_outside_git_checkout,
    validate_loaded_external_rlr_sdk,
)
from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m3.contracts import (
    AcousticSceneContractError,
    ImmutableFileSnapshot,
    ValidatedAcousticScenePackage,
    load_and_validate_acoustic_scene_package,
)


RUNTIME_IMPORT_WORKAROUND = {
    "workaround_id": "habitat_import_quaternion_first_v1",
    "required_import_order": ["quaternion", "habitat_sim"],
    "reason": (
        "the pinned baseline can abort with an invalid free when habitat_sim is "
        "imported before numpy-quaternion"
    ),
}

# Historical is the existing lock-bound evidence format. Current-installed is
# deliberately separate: it records the user-provided runtime identity for one
# fresh run and never treats its bytes as a replacement for the historical lock.
RUNTIME_MODE_HISTORICAL = "historical"
RUNTIME_MODE_CURRENT_INSTALLED = "current-installed"
_RUNTIME_MODES = {RUNTIME_MODE_HISTORICAL, RUNTIME_MODE_CURRENT_INSTALLED}


def require_runtime_mode(value: str) -> str:
    """Validate the public M3/M4 native-runtime mode selector."""

    if value not in _RUNTIME_MODES:
        choices = ", ".join(sorted(_RUNTIME_MODES))
        raise RuntimeContractError(f"runtime_mode must be one of: {choices}")
    return value



class RuntimeUnavailableError(RuntimeError):
    """The pinned Habitat/RLR runtime cannot be imported."""


class RuntimeContractError(ValueError):
    """The package or Python binding violates the declared runtime contract."""


class RuntimeExecutionError(RuntimeError):
    """The native RLR runtime rejected an otherwise valid invocation."""


@dataclass(frozen=True)
class RLRChannelLayout:
    layout_type: str
    channel_count: int

    @classmethod
    def from_mapping(cls, value: Any) -> "RLRChannelLayout":
        if not isinstance(value, Mapping) or set(value) != {"type", "channel_count"}:
            raise RuntimeContractError(
                "simulation.channel_layout must contain only type and channel_count"
            )
        result = cls(
            layout_type=value["type"], channel_count=value["channel_count"]
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected_counts = {"mono": 1, "binaural": 2}
        if self.layout_type not in {"mono", "binaural", "ambisonics"}:
            raise RuntimeContractError("simulation.channel_layout.type is invalid")
        if (
            isinstance(self.channel_count, bool)
            or not isinstance(self.channel_count, int)
            or self.channel_count < 1
        ):
            raise RuntimeContractError(
                "simulation.channel_layout.channel_count must be a positive integer"
            )
        expected = expected_counts.get(self.layout_type)
        if expected is not None and self.channel_count != expected:
            raise RuntimeContractError(
                f"{self.layout_type} channel layout requires channel_count={expected}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.layout_type, "channel_count": self.channel_count}


@dataclass(frozen=True)
class RLRSimulationConfig:
    """AVEngine-owned values copied into a default-initialized RLR config."""

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

    _INTEGER_FIELDS = (
        "frequency_bands",
        "direct_sh_order",
        "indirect_sh_order",
        "direct_ray_count",
        "indirect_ray_count",
        "indirect_ray_depth",
        "source_ray_count",
        "source_ray_depth",
        "max_diffraction_order",
        "thread_count",
    )
    _FLOAT_FIELDS = (
        "sample_rate_hz",
        "max_ir_seconds",
        "unit_scale",
        "global_volume",
        "speed_of_sound_m_s",
    )
    _BOOLEAN_FIELDS = (
        "direct",
        "indirect",
        "diffraction",
        "transmission",
        "mesh_simplification",
        "temporal_coherence",
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RLRSimulationConfig":
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
        if missing:
            raise RuntimeContractError(
                "simulation is missing explicit RLR fields: " + ", ".join(missing)
            )
        if unknown:
            raise RuntimeContractError(
                "simulation has unknown RLR fields: " + ", ".join(unknown)
            )
        arguments = {
            field: value[field] for field in expected if field != "channel_layout"
        }
        arguments["channel_layout"] = RLRChannelLayout.from_mapping(
            value["channel_layout"]
        )
        result = cls(**arguments)
        result.validate()
        return result

    def validate(self) -> None:
        for field in self._INTEGER_FIELDS:
            value = getattr(self, field)
            minimum = 1
            if field in {
                "direct_sh_order",
                "indirect_sh_order",
                "max_diffraction_order",
            }:
                minimum = 0
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
                raise RuntimeContractError(f"simulation.{field} must be a boolean")
        if self.mesh_simplification:
            raise RuntimeContractError(
                "simulation.mesh_simplification must be false for M3 geometry evidence"
            )
        if self.temporal_coherence:
            raise RuntimeContractError(
                "simulation.temporal_coherence must be false for independent M3 repeats"
            )
        if not math.isclose(
            float(self.unit_scale), 1.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeContractError(
                "simulation.unit_scale must be exactly 1.0 for meter-native M3 packages"
            )
        if not math.isclose(
            float(self.speed_of_sound_m_s), 343.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeContractError(
                "simulation.speed_of_sound_m_s must equal pinned RLR value 343.0"
            )
        if not self.direct or not self.indirect:
            raise RuntimeContractError(
                "M3 material activation requires direct and indirect propagation"
            )
        self.channel_layout.validate()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["channel_layout"] = self.channel_layout.to_dict()
        return value


@dataclass(frozen=True)
class RuntimeAnchor:
    anchor_id: str
    position_m: tuple[float, float, float]
    radius_m: float = 0.0
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, listener: bool
    ) -> "RuntimeAnchor":
        if not isinstance(value, Mapping):
            raise RuntimeContractError("runtime anchor must be an object")
        allowed = {"id", "position_m", "radius_m"}
        if listener:
            allowed.add("orientation_wxyz")
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise RuntimeContractError(
                "runtime anchor has unknown fields: " + ", ".join(unknown)
            )
        anchor_id = value.get("id")
        if not isinstance(anchor_id, str) or not anchor_id:
            raise RuntimeContractError("runtime anchor id must be non-empty")
        position = _finite_vector(value.get("position_m"), 3, "anchor.position_m")
        radius = value.get("radius_m", 0.0)
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(float(radius))
            or float(radius) < 0.0
        ):
            raise RuntimeContractError("anchor.radius_m must be finite and non-negative")
        orientation = (
            _finite_vector(
                value.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0)),
                4,
                "anchor.orientation_wxyz",
            )
            if listener
            else (1.0, 0.0, 0.0, 0.0)
        )
        if listener:
            norm = math.sqrt(sum(component * component for component in orientation))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
                raise RuntimeContractError(
                    "listener orientation_wxyz must already be unit normalized"
                )
        return cls(
            anchor_id=anchor_id,
            position_m=position,
            radius_m=float(radius),
            orientation_wxyz=orientation,
        )


@dataclass(frozen=True)
class CompiledAcousticScene:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    package_id: str
    package_content_sha256: str
    material_database_path: Path
    material_database_bytes: bytes
    material_database_sha256: str
    material_categories_document: dict[str, Any]
    rlr_material_database: dict[str, Any]
    material_categories: tuple[str, ...]
    objects: tuple[dict[str, Any], ...]
    geometry_records: dict[str, dict[str, Any]]
    triangle_count_by_material: dict[str, int]
    qa_reports: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RuntimeIRResult:
    listener_id: str
    source_id: str
    sample_rate_hz: float
    samples: np.ndarray
    package_manifest_sha256: str
    package_content_sha256: str
    runtime: dict[str, Any]
    upload_report: dict[str, Any]
    indirect_ray_efficiency: float
    ray_checks: tuple[dict[str, Any], ...]


def _finite_vector(value: Any, length: int, owner: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise RuntimeContractError(f"{owner} must contain {length} finite numbers")
    return tuple(float(item) for item in value)


def _load_installed_habitat_runtime(
    prefix: Path, *, magnum_python_site: Path | None = None
) -> tuple[ModuleType, ModuleType, Path, Path]:
    """Import only the M1-installed runtime and return validated native paths."""

    # M1 owns the common installed-prefix/Magnum activation sequence.  It
    # imports quaternion before habitat_sim, validates the external Magnum
    # Python site, and rejects modules or the physics config outside *prefix*.
    from avengine.m1.habitat_capture import (
        _import_installed_habitat,
        _installed_runtime_paths,
    )

    _, habitat_module, _, _ = _import_installed_habitat(
        prefix, magnum_python_site=magnum_python_site
    )
    binding_module = importlib.import_module(
        "habitat_sim._ext.habitat_sim_bindings"
    )
    module_path, binding_path, _ = _installed_runtime_paths(
        prefix, habitat_module, binding_module
    )
    return habitat_module, binding_module, module_path, binding_path


def load_habitat_runtime(
    *,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    runtime_mode: str = RUNTIME_MODE_HISTORICAL,
) -> tuple[ModuleType, dict[str, Any]]:
    """Load the installed Habitat adapter against an explicit external RLR SDK.

    The legacy AVENGINE_HABITAT_RUNTIME_ROOT checkout interface is never
    consulted here. The modern adapter is independently optional from the
    legacy AudioSensor, so habitat_sim.audio_enabled is intentionally not a
    capability check for M3/M4.
    """

    runtime_mode = require_runtime_mode(runtime_mode)
    if runtime_mode == RUNTIME_MODE_CURRENT_INSTALLED:
        missing = [
            name
            for name, value in (
                ("runtime_prefix", runtime_prefix),
                ("rlr_sdk_root", rlr_sdk_root),
                ("magnum_python_site", magnum_python_site),
            )
            if value is None or not str(value).strip()
        ]
        if missing:
            raise RuntimeUnavailableError(
                "current-installed Habitat/RLR runtime requires explicit "
                + ", ".join(missing)
            )
    try:
        from avengine.m1.habitat_capture import discover_runtime_prefix

        prefix = discover_runtime_prefix(runtime_prefix)
        prefix = require_outside_git_checkout(
            prefix, owner="AVENGINE_HABITAT_RUNTIME_PREFIX"
        )
        sdk = discover_external_rlr_sdk(rlr_sdk_root)
        if runtime_mode == RUNTIME_MODE_CURRENT_INSTALLED:
            from avengine.m1.habitat_capture import discover_magnum_python_site

            magnum_site = require_outside_git_checkout(
                discover_magnum_python_site(magnum_python_site),
                owner="AVENGINE_HABITAT_MAGNUM_PYTHON_SITE",
            )
            habitat_module, _binding_module, module_path, binding_path = (
                _load_installed_habitat_runtime(
                    prefix, magnum_python_site=magnum_site
                )
            )
            # M1 removes editable Habitat finders and verifies prefix-only
            # imports. Run that isolation before global SDK preloading: a
            # preload first can let an editable extension win native linkage.
            preload_external_rlr_sdk(sdk)
        else:
            # Historical v1 keeps its retained import/preload sequence.
            preload_external_rlr_sdk(sdk)
            habitat_module, _binding_module, module_path, binding_path = (
                _load_installed_habitat_runtime(prefix)
            )
        validate_loaded_external_rlr_sdk(sdk)
    except (
        ExternalRlrSdkError,
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
    ) as error:
        raise RuntimeUnavailableError(
            "Installed Habitat/RLR runtime is unavailable: " + str(error)
        ) from error

    required = (
        "RLRContextConfiguration",
        "RLRAcousticContext",
        "RLRChannelLayoutType",
    )
    missing = [name for name in required if not hasattr(habitat_module, name)]
    if missing:
        raise RuntimeUnavailableError(
            "habitat_sim lacks the AVEngine modern RLR binding: " + ", ".join(missing)
        )
    if getattr(habitat_module, "RLR_ADAPTER_ENABLED", None) is not True:
        raise RuntimeUnavailableError(
            "Installed Habitat prefix has AVENGINE_HABITAT_BUILD_RLR_ADAPTER=OFF; "
            "rebuild that prefix with the modern adapter and the declared external "
            "RLR SDK. Legacy habitat_sim.audio_enabled does not satisfy M3/M4."
        )
    if any(getattr(habitat_module, name, None) is None for name in required):
        raise RuntimeUnavailableError(
            "habitat_sim exposes placeholder None values instead of the modern RLR binding"
        )
    quaternion_module = sys.modules.get("quaternion")
    report: dict[str, Any] = {
        "import_workaround": dict(RUNTIME_IMPORT_WORKAROUND),
        "quaternion_module": {
            "path": str(getattr(quaternion_module, "__file__", "")),
            "version": str(getattr(quaternion_module, "__version__", "unknown")),
        },
        "habitat_sim_module": {
            "path": str(module_path),
            "version": str(getattr(habitat_module, "__version__", "unknown")),
        },
        "binding_api": "habitat_sim.RLRAcousticContext_v1",
        "installed_habitat_runtime": {
            "prefix": str(prefix),
            "binding_path": str(binding_path),
        },
        "external_rlr_sdk": {
            "root": str(sdk.root),
            "header": str(sdk.header),
            "library": str(sdk.library),
        },
    }
    if runtime_mode == RUNTIME_MODE_CURRENT_INSTALLED:
        # This is an observed one-run identity, not a byte pin. The M3/M4 v2
        # readers require it to agree across fresh contexts in one output.
        report["runtime_mode"] = runtime_mode
        report["runtime_identity"] = {
            "identity_schema": "avengine_current_installed_rlr_runtime_v1",
            "mode": runtime_mode,
            "habitat_runtime_prefix": str(prefix),
            "habitat_sim_module": str(module_path),
            "habitat_sim_binding": str(binding_path),
            "magnum_python_site": str(magnum_site),
            "rlr_sdk_root": str(sdk.root),
            "rlr_sdk_header": str(sdk.header),
            "rlr_sdk_library": str(sdk.library),
            "rlr_adapter_enabled": True,
            "binding_api": "habitat_sim.RLRAcousticContext_v1",
        }
        return habitat_module, report
    try:
        binding_payload = binding_path.read_bytes()
        rlr_payload = sdk.library.read_bytes()
    except OSError as exc:
        raise RuntimeUnavailableError(
            f"unable to snapshot native Habitat/RLR binaries: {exc}"
        ) from exc
    report["native_binaries"] = {
        "habitat_sim_bindings": {
            "path": str(binding_path),
            "byte_size": len(binding_payload),
            "sha256": hashlib.sha256(binding_payload).hexdigest(),
        },
        "rlr_audio_propagation": {
            "path": str(sdk.library),
            "byte_size": len(rlr_payload),
            "sha256": hashlib.sha256(rlr_payload).hexdigest(),
        },
    }
    return habitat_module, report


def load_compiled_acoustic_scene(
    manifest_path: str | Path,
    *,
    manifest_snapshot: ImmutableFileSnapshot | None = None,
    snapshot_cache: dict[Path, ImmutableFileSnapshot] | None = None,
    validated_package: ValidatedAcousticScenePackage | None = None,
    allow_nonpassing_research_qa: bool = False,
) -> CompiledAcousticScene:
    """Read and hash-check the exact arrays consumed by the native runtime.

    The default remains the M3 admission boundary: every required QA report
    must be ``pass``.  A review-only caller may explicitly load a
    ``research_candidate`` whose material semantics are
    ``research_placeholder`` by setting ``allow_nonpassing_research_qa``.
    That narrow escape hatch does not accept production packages, missing QA
    reports, or any physical-material claim; it exists so downstream review
    media can expose the behavior of unqualified real-room geometry without
    misrepresenting it as an admitted acoustic scene.
    """

    if not isinstance(allow_nonpassing_research_qa, bool):
        raise RuntimeContractError(
            "allow_nonpassing_research_qa must be an explicit boolean"
        )

    path = Path(manifest_path).resolve()
    if validated_package is not None:
        if validated_package.manifest_path.resolve() != path:
            raise RuntimeContractError(
                "validated package path differs from requested manifest path"
            )
        validated = validated_package
    else:
        try:
            validated = load_and_validate_acoustic_scene_package(
                path,
                manifest_snapshot=manifest_snapshot,
                snapshot_cache=snapshot_cache,
            )
        except AcousticSceneContractError as exc:
            raise RuntimeContractError(
                "compiled acoustic package failed its formal contract: " + str(exc)
            ) from exc
        except (OSError, ValueError) as exc:
            raise RuntimeContractError(
                f"unable to validate compiled acoustic package: {exc}"
            ) from exc
    manifest = validated.manifest
    package_id = manifest.get("package_id")
    package_hash = manifest.get("package_content_sha256")
    if not isinstance(package_id, str) or not package_id:
        raise RuntimeContractError("manifest.package_id must be non-empty")
    if (
        not isinstance(package_hash, str)
        or len(package_hash) != 64
        or any(character not in "0123456789abcdef" for character in package_hash)
    ):
        raise RuntimeContractError("manifest.package_content_sha256 is invalid")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping):
        raise RuntimeContractError("manifest.arrays must be an object")
    # The formal loader parsed every package file from one immutable byte
    # snapshot.  Copy those exact arrays into runtime-owned mutable storage;
    # never reopen the package paths between validation and native upload.
    vertices = np.array(
        validated.vertices,
        dtype="<f4",
        order="C",
        copy=True,
    )
    triangles = np.array(
        validated.triangles,
        dtype="<u4",
        order="C",
        copy=True,
    )
    material_ids = np.array(
        validated.triangle_material_ids,
        dtype="<u4",
        order="C",
        copy=True,
    )
    if not np.all(np.isfinite(vertices)):
        raise RuntimeContractError("arrays.vertices contains non-finite values")
    if len(triangles) != len(material_ids):
        raise RuntimeContractError(
            "triangle_material_ids must contain exactly one ID per triangle"
        )

    materials = manifest.get("materials")
    if not isinstance(materials, Mapping):
        raise RuntimeContractError("manifest.materials must be an object")
    database_path = validated.rlr_material_database_path
    category_bytes = bytes(validated.material_categories_bytes)
    if (
        len(category_bytes) != materials["categories"]["byte_size"]
        or hashlib.sha256(category_bytes).hexdigest()
        != materials["categories"]["sha256"]
    ):
        raise RuntimeContractError("material categories changed during snapshot read")
    database_bytes = bytes(validated.rlr_material_database_bytes)
    if (
        len(database_bytes) != materials["rlr_database"]["byte_size"]
        or hashlib.sha256(database_bytes).hexdigest()
        != materials["rlr_database"]["sha256"]
    ):
        raise RuntimeContractError("RLR material database changed during snapshot read")
    try:
        categories_document = dict(validated.material_categories)
        database = dict(validated.rlr_material_database)
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(f"material snapshot is invalid: {exc}") from exc
    raw_categories = categories_document.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise RuntimeContractError("material categories document is empty")
    ordered_categories: list[str] = []
    for expected_id, category in enumerate(raw_categories):
        if not isinstance(category, Mapping) or category.get("material_id") != expected_id:
            raise RuntimeContractError(
                "material categories must be ordered and contiguous from ID zero"
            )
        name = category.get("category_name")
        if not isinstance(name, str) or not name:
            raise RuntimeContractError("material category_name must be non-empty")
        if category.get("fallback") is not False:
            raise RuntimeContractError("fallback acoustic material is forbidden")
        ordered_categories.append(name)
    if len(set(ordered_categories)) != len(ordered_categories):
        raise RuntimeContractError("material category names must be unique")
    if int(material_ids.max(initial=0)) >= len(ordered_categories):
        raise RuntimeContractError("triangle material ID is out of range")
    if set(int(value) for value in np.unique(material_ids)) != set(
        range(len(ordered_categories))
    ):
        raise RuntimeContractError("every declared material category must be used")
    if not isinstance(database.get("materials"), list) or not database["materials"]:
        raise RuntimeContractError("RLR material database contains no materials")
    required_qa = {
        "geometry_report",
        "material_coverage",
        "ray_leakage",
        "compiler_source_to_package_parity",
    }
    if set(validated.qa_reports) != required_qa:
        raise RuntimeContractError("compiled package is missing required M3 QA reports")
    failed_qa = sorted(
        name
        for name, report in validated.qa_reports.items()
        if report.get("status") != "pass"
    )
    if failed_qa:
        materials_claim = manifest.get("materials")
        research_override_allowed = (
            allow_nonpassing_research_qa
            and manifest.get("package_mode") == "research_candidate"
            and isinstance(materials_claim, Mapping)
            and materials_claim.get("material_semantics")
            == "research_placeholder"
            and materials_claim.get("qualification_claim")
            == "unqualified_research_placeholder"
        )
        if not research_override_allowed:
            raise RuntimeContractError(
                "compiled package QA is not pass: " + ", ".join(failed_qa)
            )

    triangle_count_by_material = {
        category: int(np.count_nonzero(material_ids == material_id))
        for material_id, category in enumerate(ordered_categories)
    }

    raw_objects = manifest.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise RuntimeContractError("manifest.objects must be non-empty")
    native_objects: list[dict[str, Any]] = []
    next_vertex = 0
    next_triangle = 0
    for index, item in enumerate(raw_objects):
        if not isinstance(item, Mapping):
            raise RuntimeContractError(f"objects[{index}] must be an object")
        object_id = item.get("object_id")
        vertex_offset = item.get("vertex_offset")
        vertex_count = item.get("vertex_count")
        triangle_offset = item.get("triangle_offset")
        triangle_count = item.get("triangle_count")
        if not isinstance(object_id, str) or not object_id:
            raise RuntimeContractError(f"objects[{index}].object_id must be non-empty")
        ranges = (vertex_offset, vertex_count, triangle_offset, triangle_count)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ranges
        ) or vertex_count < 3 or triangle_count < 1:
            raise RuntimeContractError(f"objects[{index}] has invalid array ranges")
        if vertex_offset != next_vertex or triangle_offset != next_triangle:
            raise RuntimeContractError(
                "object array ranges must form ordered, gap-free partitions"
            )
        vertex_end = vertex_offset + vertex_count
        triangle_end = triangle_offset + triangle_count
        if vertex_end > len(vertices) or triangle_end > len(triangles):
            raise RuntimeContractError(f"objects[{index}] range exceeds package arrays")
        object_triangles_u64 = triangles[triangle_offset:triangle_end].astype(
            np.uint64, copy=False
        )
        if (
            int(object_triangles_u64.min(initial=vertex_offset)) < vertex_offset
            or int(object_triangles_u64.max(initial=vertex_offset)) >= vertex_end
        ):
            raise RuntimeContractError(
                f"objects[{index}] triangles escape their local vertex partition"
            )
        local_triangles = np.ascontiguousarray(
            object_triangles_u64 - vertex_offset, dtype="<u4"
        )
        native_objects.append(
            {
                "object_id": object_id,
                "vertices": np.ascontiguousarray(
                    vertices[vertex_offset:vertex_end], dtype="<f4"
                ),
                "triangles": local_triangles,
                "triangle_material_ids": np.ascontiguousarray(
                    material_ids[triangle_offset:triangle_end], dtype="<u4"
                ),
                "position": (0.0, 0.0, 0.0),
                "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            }
        )
        next_vertex = vertex_end
        next_triangle = triangle_end
    if next_vertex != len(vertices) or next_triangle != len(triangles):
        raise RuntimeContractError("object partitions do not cover all package arrays")

    return CompiledAcousticScene(
        manifest_path=path,
        manifest_sha256=validated.manifest_file_sha256,
        manifest=dict(manifest),
        package_id=package_id,
        package_content_sha256=package_hash,
        material_database_path=database_path,
        material_database_bytes=database_bytes,
        material_database_sha256=hashlib.sha256(database_bytes).hexdigest(),
        material_categories_document=dict(categories_document),
        rlr_material_database=dict(database),
        material_categories=tuple(ordered_categories),
        objects=tuple(native_objects),
        geometry_records={
            "vertices": dict(arrays["vertices"]),
            "triangles": dict(arrays["triangles"]),
            "triangle_material_ids": dict(arrays["triangle_material_ids"]),
        },
        triangle_count_by_material=triangle_count_by_material,
        qa_reports={name: dict(report) for name, report in validated.qa_reports.items()},
    )


_NATIVE_CONFIG_FIELDS = {
    "frequency_bands": "frequency_bands",
    "direct_sh_order": "direct_sh_order",
    "indirect_sh_order": "indirect_sh_order",
    "direct_ray_count": "direct_ray_count",
    "indirect_ray_count": "indirect_ray_count",
    "indirect_ray_depth": "indirect_ray_depth",
    "source_ray_count": "source_ray_count",
    "source_ray_depth": "source_ray_depth",
    "max_diffraction_order": "max_diffraction_order",
    "thread_count": "thread_count",
    "sample_rate_hz": "sample_rate",
    "max_ir_seconds": "max_ir_length",
    "unit_scale": "unit_scale",
    "global_volume": "global_volume",
    "direct": "direct",
    "indirect": "indirect",
    "diffraction": "diffraction",
    "transmission": "transmission",
    "mesh_simplification": "mesh_simplification",
    "temporal_coherence": "temporal_coherence",
}


def _native_configuration(
    habitat_module: ModuleType, simulation: RLRSimulationConfig
) -> tuple[Any, dict[str, Any]]:
    simulation.validate()
    try:
        native = habitat_module.RLRContextConfiguration()
    except Exception as exc:  # pybind exception type depends on the runtime build
        raise RuntimeExecutionError(f"unable to create RLR configuration: {exc}") from exc
    readback: dict[str, Any] = {}
    for public_name, native_name in _NATIVE_CONFIG_FIELDS.items():
        if not hasattr(native, native_name):
            raise RuntimeContractError(
                f"RLRContextConfiguration lacks required field {native_name!r}"
            )
        requested = getattr(simulation, public_name)
        try:
            setattr(native, native_name, requested)
            observed = getattr(native, native_name)
        except Exception as exc:
            raise RuntimeContractError(
                f"unable to set/read RLR configuration field {native_name!r}: {exc}"
            ) from exc
        if isinstance(requested, float):
            if not math.isclose(
                float(observed), float(requested), rel_tol=1.0e-6, abs_tol=1.0e-6
            ):
                raise RuntimeContractError(
                    f"RLR configuration readback differs for {native_name!r}"
                )
            readback[public_name] = float(observed)
        else:
            if observed != requested:
                raise RuntimeContractError(
                    f"RLR configuration readback differs for {native_name!r}"
                )
            readback[public_name] = observed
    return native, readback


def _upload_report(value: Any) -> dict[str, Any]:
    fields = (
        "object_count",
        "vertex_count",
        "triangle_count",
        "material_category_count",
        "object_ids",
        "triangle_count_by_material",
        "material_upload_call_count",
        "resolved_material_name_by_category",
        "resolved_material_index_by_category",
        "material_upload_receipts",
        "expected_material_block_count",
        "material_database_sha1",
        "expected_world_geometry_sha1",
        "expected_canonical_byte_count",
        "expected_material_coefficient_sha1",
        "expected_material_coefficient_byte_count",
    )
    missing = [field for field in fields if not hasattr(value, field)]
    if missing:
        raise RuntimeContractError(
            "RLR upload report lacks fields: " + ", ".join(missing)
        )
    receipts: list[dict[str, Any]] = []
    for index, receipt in enumerate(value.material_upload_receipts):
        receipt_fields = (
            "object_id",
            "material_category",
            "triangle_count",
            "index_count",
            "canonical_payload_byte_count",
            "canonical_payload_sha1",
        )
        receipt_missing = [
            field for field in receipt_fields if not hasattr(receipt, field)
        ]
        if receipt_missing:
            raise RuntimeContractError(
                f"RLR upload receipt {index} lacks fields: "
                + ", ".join(receipt_missing)
            )
        receipts.append(
            {
                "object_id": str(receipt.object_id),
                "material_category": str(receipt.material_category),
                "triangle_count": int(receipt.triangle_count),
                "index_count": int(receipt.index_count),
                "canonical_payload_byte_count": int(
                    receipt.canonical_payload_byte_count
                ),
                "canonical_payload_sha1": str(receipt.canonical_payload_sha1),
            }
        )
    return {
        "object_count": int(value.object_count),
        "vertex_count": int(value.vertex_count),
        "triangle_count": int(value.triangle_count),
        "material_category_count": int(value.material_category_count),
        "object_ids": [str(item) for item in value.object_ids],
        "triangle_count_by_material": {
            str(key): int(count)
            for key, count in dict(value.triangle_count_by_material).items()
        },
        "material_upload_call_count": {
            str(key): int(count)
            for key, count in dict(value.material_upload_call_count).items()
        },
        "resolved_material_name_by_category": {
            str(key): str(name)
            for key, name in dict(
                value.resolved_material_name_by_category
            ).items()
        },
        "resolved_material_index_by_category": {
            str(key): int(material_index)
            for key, material_index in dict(
                value.resolved_material_index_by_category
            ).items()
        },
        "material_upload_receipts": receipts,
        "expected_material_block_count": int(value.expected_material_block_count),
        "material_database_sha1": str(value.material_database_sha1),
        "expected_world_geometry_sha1": str(value.expected_world_geometry_sha1),
        "expected_canonical_byte_count": int(value.expected_canonical_byte_count),
        "expected_material_coefficient_sha1": str(
            value.expected_material_coefficient_sha1
        ),
        "expected_material_coefficient_byte_count": int(
            value.expected_material_coefficient_byte_count
        ),
    }


def _length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded


def _expected_material_upload_receipts(
    scene: CompiledAcousticScene,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    header = b"AVENGINE_RLR_ADD_MESH_INDICES_V1\x00"
    for item in scene.objects:
        triangles = np.asarray(item["triangles"], dtype=np.uint32)
        material_ids = np.asarray(item["triangle_material_ids"], dtype=np.uint32)
        for material_id, category in enumerate(scene.material_categories):
            selected = np.ascontiguousarray(
                triangles[material_ids == material_id].reshape(-1),
                dtype="<u4",
            )
            if not len(selected):
                continue
            payload = bytearray(header)
            payload.extend(_length_prefixed(str(item["object_id"])))
            payload.extend(_length_prefixed(category))
            payload.extend((3).to_bytes(8, "little"))
            payload.extend(len(selected).to_bytes(8, "little"))
            payload.extend(selected.tobytes(order="C"))
            receipts.append(
                {
                    "object_id": str(item["object_id"]),
                    "material_category": category,
                    "triangle_count": int(len(selected) // 3),
                    "index_count": int(len(selected)),
                    "canonical_payload_byte_count": len(payload),
                    "canonical_payload_sha1": hashlib.sha1(payload).hexdigest(),
                }
            )
    return receipts


def _canonical_world_geometry(scene: CompiledAcousticScene) -> bytes:
    def coordinate(value: float) -> str:
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeContractError("scene geometry contains a non-finite value")
        if abs(number) < 0.5e-6:
            number = 0.0
        return f"{number:.6f}"

    vertex_tokens: list[str] = []
    triangle_tokens: list[str] = []
    for item in scene.objects:
        position = np.asarray(item["position"], dtype=np.float64)
        quaternion = np.asarray(item["orientation_wxyz"], dtype=np.float64)
        w = float(quaternion[0])
        q = quaternion[1:]
        object_tokens: list[str] = []
        for raw_vertex in np.asarray(item["vertices"], dtype=np.float64):
            twice_cross = 2.0 * np.cross(q, raw_vertex)
            transformed = raw_vertex + w * twice_cross + np.cross(q, twice_cross)
            transformed = transformed + position
            token = " ".join(coordinate(value) for value in transformed)
            object_tokens.append(token)
            vertex_tokens.append(token)
        for face in np.asarray(item["triangles"], dtype=np.int64):
            values = [object_tokens[int(index)] for index in face]
            rotations = [
                "|".join(values),
                "|".join(values[1:] + values[:1]),
                "|".join(values[2:] + values[:2]),
            ]
            triangle_tokens.append(min(rotations))
    lines = ["AVENGINE_RLR_WORLD_GEOMETRY_V1"]
    lines.extend(f"v {value}" for value in sorted(vertex_tokens))
    lines.extend(f"f {value}" for value in sorted(triangle_tokens))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _canonical_material_coefficients(scene: CompiledAcousticScene) -> bytes:
    def coordinate(value: float) -> str:
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise RuntimeContractError(
                "material coefficient must be finite and in [0, 1]"
            )
        if abs(number) < 0.5e-6:
            number = 0.0
        return f"{number:.6f}"

    database_materials = scene.rlr_material_database["materials"]
    resolved: list[Mapping[str, Any]] = []
    for category in scene.material_categories:
        candidates = [
            material
            for material in database_materials
            if category.casefold()
            in {
                str(label).casefold() for label in material.get("labels", [])
            }
        ]
        if len(candidates) != 1:
            raise RuntimeContractError(
                f"category {category!r} does not resolve to exactly one material"
            )
        resolved.append(candidates[0])
    lines = ["AVENGINE_RLR_MATERIAL_COEFFICIENTS_V1"]
    for object_index, item in enumerate(scene.objects):
        material_ids = np.asarray(item["triangle_material_ids"], dtype=np.int64)
        local_material_index = 0
        for category_index, material in enumerate(resolved):
            if not np.any(material_ids == category_index):
                continue
            lines.append(
                f"object {object_index} material {local_material_index}"
            )
            for field in ("absorption", "scattering", "transmission"):
                interleaved = material[field]
                if len(interleaved) < 2 or len(interleaved) % 2:
                    raise RuntimeContractError(
                        f"RLR material {field} is not frequency/value paired"
                    )
                for coefficient_index, value in enumerate(interleaved[1::2]):
                    lines.append(
                        f"{field} {coefficient_index} {coordinate(float(value))}"
                    )
            local_material_index += 1
    return ("\n".join(lines) + "\n").encode("utf-8")


def _expected_upload_report(scene: CompiledAcousticScene) -> dict[str, Any]:
    receipts = _expected_material_upload_receipts(scene)
    upload_calls = {category: 0 for category in scene.material_categories}
    for receipt in receipts:
        upload_calls[receipt["material_category"]] += 1
    database_materials = scene.rlr_material_database["materials"]
    resolved_names: dict[str, str] = {}
    resolved_indices: dict[str, int] = {}
    for category in scene.material_categories:
        candidates = [
            index
            for index, material in enumerate(database_materials)
            if category.casefold()
            in {
                str(label).casefold() for label in material.get("labels", [])
            }
        ]
        if len(candidates) != 1:
            raise RuntimeContractError(
                f"category {category!r} does not resolve to exactly one material"
            )
        resolved_indices[category] = candidates[0]
        resolved_names[category] = str(database_materials[candidates[0]]["name"])
    canonical_geometry = _canonical_world_geometry(scene)
    canonical_coefficients = _canonical_material_coefficients(scene)
    return {
        "object_count": len(scene.objects),
        "vertex_count": sum(len(item["vertices"]) for item in scene.objects),
        "triangle_count": sum(len(item["triangles"]) for item in scene.objects),
        "material_category_count": len(scene.material_categories),
        "object_ids": [str(item["object_id"]) for item in scene.objects],
        "triangle_count_by_material": dict(scene.triangle_count_by_material),
        "material_upload_call_count": upload_calls,
        "resolved_material_name_by_category": resolved_names,
        "resolved_material_index_by_category": resolved_indices,
        "material_upload_receipts": receipts,
        "expected_material_block_count": len(receipts),
        "material_database_sha1": hashlib.sha1(
            scene.material_database_bytes
        ).hexdigest(),
        "expected_world_geometry_sha1": hashlib.sha1(
            canonical_geometry
        ).hexdigest(),
        "expected_canonical_byte_count": len(canonical_geometry),
        "expected_material_coefficient_sha1": hashlib.sha1(
            canonical_coefficients
        ).hexdigest(),
        "expected_material_coefficient_byte_count": len(
            canonical_coefficients
        ),
    }


def _verify_upload_report(
    scene: CompiledAcousticScene, report: Mapping[str, Any]
) -> None:
    expected = _expected_upload_report(scene)
    comparisons = {
        name: (report.get(name), value) for name, value in expected.items()
    }
    mismatches = [
        name for name, (observed, expected) in comparisons.items() if observed != expected
    ]
    if mismatches:
        raise RuntimeContractError(
            "RLR upload report differs from the hash-checked package: "
            + ", ".join(mismatches)
        )


_READBACK_COORDINATE_DECIMAL_PLACES = 6
_RLR_COEFFICIENT_PATTERN = re.compile(
    r"# (Absorption|Scattering|Transmission) - Index:(\d+), "
    r"Value: (-?\d+\.\d{6})"
)
_RLR_COORDINATE_PATTERN = re.compile(r"-?\d+\.\d{6}")


def _quantized_coordinate(value: Any) -> tuple[float, float, float]:
    if isinstance(value, np.ndarray):
        if value.shape != (3,) or not np.issubdtype(value.dtype, np.number):
            raise RuntimeContractError(
                "scene readback coordinate must have numeric shape [3]"
            )
        raw_value: Any = value.tolist()
    else:
        raw_value = value
    raw = _finite_vector(raw_value, 3, "scene readback coordinate")
    rounded = tuple(
        round(float(component), _READBACK_COORDINATE_DECIMAL_PLACES)
        for component in raw
    )
    # Canonical JSON distinguishes -0.0 textually, while it has no geometric
    # meaning.  Normalize it before constructing the evidence fingerprint.
    return tuple(0.0 if component == 0.0 else component for component in rounded)


def _geometry_fingerprint(
    vertices: np.ndarray, triangles: np.ndarray
) -> dict[str, Any]:
    coordinates = [_quantized_coordinate(row) for row in vertices]
    vertex_multiset = sorted(coordinates)
    triangle_multiset: list[tuple[tuple[float, float, float], ...]] = []
    for triangle in np.asarray(triangles, dtype=np.int64):
        if len(triangle) != 3 or any(
            int(index) < 0 or int(index) >= len(coordinates) for index in triangle
        ):
            raise RuntimeContractError("scene readback triangle index is out of range")
        ordered = tuple(coordinates[int(index)] for index in triangle)
        rotations = (
            ordered,
            ordered[1:] + ordered[:1],
            ordered[2:] + ordered[:2],
        )
        # Canonicalize only the starting vertex.  Reversed winding remains a
        # different fingerprint because it changes acoustic surface normals.
        triangle_multiset.append(min(rotations))
    triangle_multiset.sort()
    return {
        "coordinate_quantization_decimal_places": (
            _READBACK_COORDINATE_DECIMAL_PLACES
        ),
        "vertex_coordinate_multiset_sha256": canonical_json_sha256(
            vertex_multiset
        ),
        "triangle_coordinate_multiset_sha256": canonical_json_sha256(
            triangle_multiset
        ),
    }


def _expected_scene_readback(scene: CompiledAcousticScene) -> dict[str, Any]:
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    block_layout: list[dict[str, int]] = []
    coefficient_layout: list[list[dict[str, Any]]] = []
    vertex_offset = 0
    for item in scene.objects:
        object_vertices = np.asarray(item["vertices"], dtype=np.float64)
        object_triangles = np.asarray(item["triangles"], dtype=np.int64)
        object_material_ids = np.asarray(
            item["triangle_material_ids"], dtype=np.int64
        )
        material_block_count = sum(
            bool(np.any(object_material_ids == material_id))
            for material_id in range(len(scene.material_categories))
        )
        object_coefficients: list[dict[str, Any]] = []
        for material_id, category in enumerate(scene.material_categories):
            if not np.any(object_material_ids == material_id):
                continue
            candidates = [
                material
                for material in scene.rlr_material_database["materials"]
                if category.casefold()
                in {
                    str(label).casefold()
                    for label in material.get("labels", [])
                }
            ]
            if len(candidates) != 1:
                raise RuntimeContractError(
                    f"category {category!r} does not resolve to one RLR material"
                )
            material = candidates[0]
            coefficients: dict[str, list[str]] = {}
            for key in ("absorption", "scattering", "transmission"):
                interleaved = material[key]
                if len(interleaved) < 2 or len(interleaved) % 2:
                    raise RuntimeContractError(
                        f"RLR material {key} coefficients are not frequency/value pairs"
                    )
                coefficients[key] = [
                    f"{float(value):.6f}" for value in interleaved[1::2]
                ]
            object_coefficients.append(coefficients)
        coefficient_layout.append(object_coefficients)
        vertices.append(object_vertices)
        triangles.append(object_triangles + vertex_offset)
        block_layout.append(
            {
                "material_block_count": int(material_block_count),
                "vertex_count": int(len(object_vertices)),
                "triangle_count": int(len(object_triangles)),
                "material_assignment_count": int(len(object_triangles)),
            }
        )
        vertex_offset += len(object_vertices)
    combined_vertices = np.ascontiguousarray(np.concatenate(vertices, axis=0))
    combined_triangles = np.ascontiguousarray(np.concatenate(triangles, axis=0))
    return {
        "vertex_count": int(len(combined_vertices)),
        "triangle_count": int(len(combined_triangles)),
        "material_block_count": sum(
            block["material_block_count"] for block in block_layout
        ),
        "material_assignment_count": int(len(combined_triangles)),
        "material_block_layout_sha256": canonical_json_sha256(block_layout),
        "material_coefficient_sha256": canonical_json_sha256(
            coefficient_layout
        ),
        **_geometry_fingerprint(combined_vertices, combined_triangles),
    }


def _parse_scene_obj(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeContractError(f"unable to read RLR scene OBJ: {exc}") from exc
    return _parse_scene_obj_bytes(payload)


def _parse_scene_obj_bytes(payload: bytes) -> dict[str, Any]:
    vertices: list[tuple[float, float, float]] = []
    face_indices: list[tuple[int, int, int]] = []
    object_layout: list[dict[str, int]] = []
    coefficient_layout: list[list[dict[str, list[str]]]] = []
    current_object: dict[str, Any] | None = None

    def finish_object() -> None:
        nonlocal current_object
        if current_object is None:
            return
        observed_vertex_count = len(vertices) - current_object["vertex_offset"]
        observed_triangle_count = len(face_indices) - current_object["face_offset"]
        declared = {
            "material_block_count": len(current_object["material_indices"]),
            "vertex_count": current_object.get("vertex_count"),
            "triangle_count": current_object.get("triangle_count"),
            "material_assignment_count": current_object.get(
                "material_assignment_count"
            ),
        }
        expected = {
            "material_block_count": len(current_object["material_indices"]),
            "vertex_count": observed_vertex_count,
            "triangle_count": observed_triangle_count,
            "material_assignment_count": observed_triangle_count,
        }
        if declared != expected:
            raise RuntimeContractError(
                "RLR scene OBJ object-block headers differ from their payload"
            )
        expected_material_indices = list(
            range(len(current_object["material_indices"]))
        )
        if current_object["material_indices"] != expected_material_indices:
            raise RuntimeContractError(
                "RLR scene OBJ object material indices are not contiguous from zero"
            )
        object_coefficients: list[dict[str, list[str]]] = []
        for section in current_object["material_sections"]:
            coefficients = {
                key: list(section[key])
                for key in ("absorption", "scattering", "transmission")
            }
            lengths = {len(values) for values in coefficients.values()}
            if lengths == {0} or len(lengths) != 1:
                raise RuntimeContractError(
                    "RLR scene OBJ material coefficients are empty or misaligned"
                )
            object_coefficients.append(coefficients)
        coefficient_layout.append(object_coefficients)
        first_index = current_object["vertex_offset"] + 1
        last_index = len(vertices)
        for face in face_indices[current_object["face_offset"] :]:
            if any(index < first_index or index > last_index for index in face):
                raise RuntimeContractError(
                    "RLR scene OBJ face indices escape their object block"
                )
        object_layout.append(expected)
        current_object = None

    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeContractError(f"unable to parse RLR scene OBJ: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("# Material Index :"):
                try:
                    material_index = int(line.split(":", 1)[1].strip())
                except ValueError as exc:
                    raise RuntimeContractError(
                        "RLR scene OBJ has a malformed material index"
                    ) from exc
                if material_index < 0:
                    raise RuntimeContractError(
                        "RLR scene OBJ material index must be non-negative"
                    )
                if current_object is None:
                    current_object = {
                        "material_indices": [],
                        "material_sections": [],
                        "vertex_offset": len(vertices),
                        "face_offset": len(face_indices),
                    }
                elif (
                    len(vertices) > current_object["vertex_offset"]
                    or len(face_indices) > current_object["face_offset"]
                ):
                    finish_object()
                    current_object = {
                        "material_indices": [],
                        "material_sections": [],
                        "vertex_offset": len(vertices),
                        "face_offset": len(face_indices),
                    }
                elif any(
                    name in current_object
                    for name in (
                        "vertex_count",
                        "triangle_count",
                        "material_assignment_count",
                    )
                ):
                    raise RuntimeContractError(
                        "RLR scene OBJ material section appears after object headers"
                    )
                current_object["material_indices"].append(material_index)
                current_object["material_sections"].append(
                    {
                        "absorption": [],
                        "scattering": [],
                        "transmission": [],
                    }
                )
            elif _RLR_COEFFICIENT_PATTERN.fullmatch(line):
                if current_object is None or not current_object["material_sections"]:
                    raise RuntimeContractError(
                        "RLR scene OBJ coefficient appears outside a material section"
                    )
                match = _RLR_COEFFICIENT_PATTERN.fullmatch(line)
                assert match is not None
                coefficient_name = match.group(1).lower()
                coefficient_index = int(match.group(2))
                coefficient_value = match.group(3)
                values = current_object["material_sections"][-1][
                    coefficient_name
                ]
                if coefficient_index != len(values):
                    raise RuntimeContractError(
                        "RLR scene OBJ coefficient indices are not contiguous"
                    )
                if not math.isfinite(float(coefficient_value)):
                    raise RuntimeContractError(
                        "RLR scene OBJ material coefficient is non-finite"
                    )
                if not 0.0 <= float(coefficient_value) <= 1.0:
                    raise RuntimeContractError(
                        "RLR scene OBJ material coefficient is outside [0, 1]"
                    )
                values.append(coefficient_value)
            elif line.startswith(
                ("# Absorption", "# Scattering", "# Transmission")
            ):
                raise RuntimeContractError(
                    "RLR scene OBJ has a malformed material coefficient"
                )
            elif line.startswith("# Vertex Count:"):
                if current_object is None or "vertex_count" in current_object:
                    raise RuntimeContractError(
                        "RLR scene OBJ vertex-count header is misplaced or duplicated"
                    )
                try:
                    current_object["vertex_count"] = int(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError as exc:
                    raise RuntimeContractError(
                        "RLR scene OBJ has a malformed vertex-count header"
                    ) from exc
            elif line.startswith("# Triangle Count:"):
                if current_object is None or "triangle_count" in current_object:
                    raise RuntimeContractError(
                        "RLR scene OBJ triangle-count header is misplaced or duplicated"
                    )
                try:
                    current_object["triangle_count"] = int(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError as exc:
                    raise RuntimeContractError(
                        "RLR scene OBJ has a malformed triangle-count header"
                    ) from exc
            elif line.startswith("# Material Count:"):
                if (
                    current_object is None
                    or "material_assignment_count" in current_object
                ):
                    raise RuntimeContractError(
                        "RLR scene OBJ material-count header is misplaced or duplicated"
                    )
                try:
                    current_object["material_assignment_count"] = int(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError as exc:
                    raise RuntimeContractError(
                        "RLR scene OBJ has a malformed material-count header"
                    ) from exc
            continue
        fields = line.split()
        if fields[0] == "v":
            if current_object is None:
                raise RuntimeContractError(
                    "RLR scene OBJ vertex appears outside an object block"
                )
            if len(fields) < 4:
                raise RuntimeContractError(
                    f"RLR scene OBJ line {line_number} has a malformed vertex"
                )
            if any(
                _RLR_COORDINATE_PATTERN.fullmatch(value) is None
                for value in fields[1:4]
            ):
                raise RuntimeContractError(
                    f"RLR scene OBJ line {line_number} coordinates must use "
                    "fixed six-decimal format"
                )
            try:
                coordinates = [float(value) for value in fields[1:4]]
            except ValueError as exc:
                raise RuntimeContractError(
                    f"RLR scene OBJ line {line_number} has a malformed vertex"
                ) from exc
            if not all(math.isfinite(value) for value in coordinates):
                raise RuntimeContractError("RLR scene OBJ contains a non-finite vertex")
            vertices.append(tuple(coordinates))
        elif fields[0] == "f":
            if current_object is None:
                raise RuntimeContractError(
                    "RLR scene OBJ face appears outside an object block"
                )
            if len(fields) != 4:
                raise RuntimeContractError(
                    "RLR scene OBJ readback contains a non-triangle face"
                )
            try:
                indices = tuple(int(value.split("/", 1)[0]) for value in fields[1:])
            except ValueError as exc:
                raise RuntimeContractError(
                    f"RLR scene OBJ line {line_number} has malformed face indices"
                ) from exc
            if any(index <= 0 for index in indices):
                raise RuntimeContractError(
                    "RLR scene OBJ must use positive one-based vertex indices"
                )
            face_indices.append(indices)  # type: ignore[arg-type]
    finish_object()
    vertex_count = len(vertices)
    triangle_count = len(face_indices)
    if vertex_count == 0 or triangle_count == 0:
        raise RuntimeContractError("RLR scene OBJ contains no triangle geometry")
    if any(index > vertex_count for face in face_indices for index in face):
        raise RuntimeContractError("RLR scene OBJ face index is out of range")
    if not object_layout:
        raise RuntimeContractError("RLR scene OBJ contains no material blocks")
    zero_based_faces = np.asarray(face_indices, dtype=np.int64) - 1
    return {
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
        "material_block_count": sum(
            block["material_block_count"] for block in object_layout
        ),
        "material_assignment_count": sum(
            block["material_assignment_count"] for block in object_layout
        ),
        "material_block_layout_sha256": canonical_json_sha256(object_layout),
        "material_coefficient_sha256": canonical_json_sha256(
            coefficient_layout
        ),
        **_geometry_fingerprint(
            np.asarray(vertices, dtype=np.float64), zero_based_faces
        ),
    }


def _native_scene_readback_report(
    value: Any,
    *,
    output_path: Path,
) -> dict[str, Any]:
    fields = (
        "output_path",
        "vertex_count",
        "triangle_count",
        "material_block_count",
        "material_assignment_count",
        "expected_vertex_count",
        "expected_triangle_count",
        "expected_material_block_count",
        "canonical_byte_count",
        "expected_canonical_byte_count",
        "world_geometry_sha1",
        "expected_world_geometry_sha1",
        "material_coefficient_byte_count",
        "expected_material_coefficient_byte_count",
        "material_coefficient_sha1",
        "expected_material_coefficient_sha1",
        "world_geometry_matches",
        "material_coefficients_match",
        "material_evidence_matches",
        "verification_passed",
        "per_face_material_ids_available",
        "material_evidence_mode",
    )
    missing = [field for field in fields if not hasattr(value, field)]
    if missing:
        raise RuntimeContractError(
            "RLR scene readback report lacks fields: " + ", ".join(missing)
        )
    if Path(str(value.output_path)).resolve() != output_path.resolve():
        raise RuntimeContractError("RLR scene readback report output path differs")
    result = {
        name: int(getattr(value, name))
        for name in (
            "vertex_count",
            "triangle_count",
            "material_block_count",
            "material_assignment_count",
            "expected_vertex_count",
            "expected_triangle_count",
            "expected_material_block_count",
            "canonical_byte_count",
            "expected_canonical_byte_count",
            "material_coefficient_byte_count",
            "expected_material_coefficient_byte_count",
        )
    }
    result.update(
        {
            "world_geometry_sha1": str(value.world_geometry_sha1),
            "expected_world_geometry_sha1": str(
                value.expected_world_geometry_sha1
            ),
            "material_coefficient_sha1": str(
                value.material_coefficient_sha1
            ),
            "expected_material_coefficient_sha1": str(
                value.expected_material_coefficient_sha1
            ),
            "world_geometry_matches": bool(value.world_geometry_matches),
            "material_coefficients_match": bool(
                value.material_coefficients_match
            ),
            "material_evidence_matches": bool(value.material_evidence_matches),
            "verification_passed": bool(value.verification_passed),
            "per_face_material_ids_available": bool(
                value.per_face_material_ids_available
            ),
            "material_evidence_mode": str(value.material_evidence_mode),
        }
    )
    return result


def _expected_native_scene_readback_report(
    parsed: Mapping[str, Any], upload: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "vertex_count": parsed["vertex_count"],
        "triangle_count": parsed["triangle_count"],
        "material_block_count": parsed["material_block_count"],
        "material_assignment_count": parsed["material_assignment_count"],
        "expected_vertex_count": upload["vertex_count"],
        "expected_triangle_count": upload["triangle_count"],
        "expected_material_block_count": upload["expected_material_block_count"],
        "canonical_byte_count": upload["expected_canonical_byte_count"],
        "expected_canonical_byte_count": upload["expected_canonical_byte_count"],
        "world_geometry_sha1": upload["expected_world_geometry_sha1"],
        "expected_world_geometry_sha1": upload["expected_world_geometry_sha1"],
        "material_coefficient_byte_count": upload[
            "expected_material_coefficient_byte_count"
        ],
        "expected_material_coefficient_byte_count": upload[
            "expected_material_coefficient_byte_count"
        ],
        "material_coefficient_sha1": upload[
            "expected_material_coefficient_sha1"
        ],
        "expected_material_coefficient_sha1": upload[
            "expected_material_coefficient_sha1"
        ],
        "world_geometry_matches": True,
        "material_coefficients_match": True,
        "material_evidence_matches": True,
        "verification_passed": True,
        "per_face_material_ids_available": False,
        "material_evidence_mode": (
            "exact_upload_receipt_plus_resolved_coefficient_blocks"
        ),
    }


def _cpu_first_hit_distance(
    objects: tuple[dict[str, Any], ...],
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    minimum_distance_m: float,
    maximum_distance_m: float,
) -> float | None:
    nearest: float | None = None
    chunk_size = 100_000
    for item in objects:
        vertices = np.asarray(item["vertices"], dtype=np.float64)
        triangles = np.asarray(item["triangles"], dtype=np.int64)
        for start in range(0, len(triangles), chunk_size):
            indices = triangles[start : start + chunk_size]
            v0 = vertices[indices[:, 0]]
            v1 = vertices[indices[:, 1]]
            v2 = vertices[indices[:, 2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            p_vector = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
            determinant = np.einsum("ij,ij->i", edge1, p_vector)
            nonparallel = np.abs(determinant) > 1.0e-10
            inverse = np.zeros_like(determinant)
            inverse[nonparallel] = 1.0 / determinant[nonparallel]
            t_vector = origin - v0
            u = np.einsum("ij,ij->i", t_vector, p_vector) * inverse
            q_vector = np.cross(t_vector, edge1)
            v = np.einsum("j,ij->i", direction, q_vector) * inverse
            distance = np.einsum("ij,ij->i", edge2, q_vector) * inverse
            valid = (
                nonparallel
                & (u >= -1.0e-9)
                & (v >= -1.0e-9)
                & (u + v <= 1.0 + 1.0e-9)
                & (distance >= minimum_distance_m)
                & (distance <= maximum_distance_m)
            )
            if np.any(valid):
                candidate = float(np.min(distance[valid]))
                nearest = candidate if nearest is None else min(nearest, candidate)
    return nearest


def _native_ray_result(value: Any, *, first_hit: bool) -> dict[str, Any]:
    fields = ("hit", "has_hit_details", "distance", "normal")
    missing = [field for field in fields if not hasattr(value, field)]
    if missing:
        raise RuntimeContractError(
            "RLR ray result lacks fields: " + ", ".join(missing)
        )
    hit = value.hit
    has_details = value.has_hit_details
    if not isinstance(hit, (bool, np.bool_)) or not isinstance(
        has_details, (bool, np.bool_)
    ):
        raise RuntimeContractError("RLR ray hit fields must be boolean")
    distance = float(value.distance)
    normal = _finite_vector(value.normal, 3, "RLR ray normal")
    if not math.isfinite(distance) or distance < 0.0:
        raise RuntimeContractError("RLR ray distance must be finite and non-negative")
    if first_hit and bool(hit) != bool(has_details):
        raise RuntimeContractError(
            "RLR first-hit detail flag does not agree with its hit flag"
        )
    if not first_hit and bool(has_details):
        raise RuntimeContractError("RLR any-hit query unexpectedly returned hit details")
    normal_is_zero = all(component == 0.0 for component in normal)
    if first_hit and bool(hit):
        # RLR returns the triangle normal without normalization. Its magnitude
        # therefore depends on the native triangle geometry and is not a
        # direction-vector invariant.
        if normal_is_zero:
            raise RuntimeContractError(
                "RLR first-hit normal must be finite and non-zero"
            )
    elif distance != 0.0 or not normal_is_zero:
        owner = "RLR first-hit miss" if first_hit else "RLR any-hit query"
        raise RuntimeContractError(
            f"{owner} must return finite zero distance/normal sentinels"
        )
    return {
        "hit": bool(hit),
        "has_hit_details": bool(has_details),
        "distance_m": distance,
        "normal": list(normal),
    }


def _run_ray_checks(
    context: Any,
    scene: CompiledAcousticScene,
    declarations: tuple[Mapping[str, Any], ...],
    *,
    distance_tolerance_m: float,
) -> tuple[dict[str, Any], ...]:
    if not math.isfinite(distance_tolerance_m) or distance_tolerance_m < 0.0:
        raise RuntimeContractError("ray distance tolerance must be finite and non-negative")
    reports: list[dict[str, Any]] = []
    for index, declared in enumerate(declarations):
        check_id = declared.get("check_id")
        expectation = declared.get("expectation")
        distance = declared.get("distance_m")
        if not isinstance(check_id, str) or not check_id:
            raise RuntimeContractError(f"ray_checks[{index}].check_id must be non-empty")
        if expectation not in {"clear_until_m", "hit_within_m"}:
            raise RuntimeContractError(f"ray_checks[{index}].expectation is invalid")
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(float(distance))
            or float(distance) <= 0.0
        ):
            raise RuntimeContractError(f"ray_checks[{index}].distance_m is invalid")
        origin = np.asarray(
            _finite_vector(declared.get("origin_m"), 3, f"ray_checks[{index}].origin_m"),
            dtype=np.float64,
        )
        direction = np.asarray(
            _finite_vector(
                declared.get("direction"), 3, f"ray_checks[{index}].direction"
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(direction))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeContractError(f"ray_checks[{index}].direction must be unit length")
        direction /= norm
        maximum_distance = float(distance)
        cpu_distance = _cpu_first_hit_distance(
            scene.objects,
            origin=origin,
            direction=direction,
            minimum_distance_m=0.0,
            maximum_distance_m=maximum_distance,
        )
        any_hit = _native_ray_result(
            context.trace_ray_any_hit(
                origin.tolist(), direction.tolist(), 0.0, maximum_distance
            ),
            first_hit=False,
        )
        first_hit = _native_ray_result(
            context.trace_ray_first_hit(
                origin.tolist(), direction.tolist(), 0.0, maximum_distance
            ),
            first_hit=True,
        )
        cpu_hit = cpu_distance is not None
        expected_hit = expectation == "hit_within_m"
        hit_consistent = cpu_hit == any_hit["hit"] == first_hit["hit"]
        distance_consistent = (
            not cpu_hit
            or abs(float(first_hit["distance_m"]) - float(cpu_distance))
            <= distance_tolerance_m
        )
        passed = hit_consistent and distance_consistent and cpu_hit == expected_hit
        reports.append(
            {
                "check_id": check_id,
                "expectation": expectation,
                "maximum_distance_m": maximum_distance,
                "cpu_first_hit_distance_m": cpu_distance,
                "rlr_any_hit": any_hit,
                "rlr_first_hit": first_hit,
                "cpu_rlr_hit_consistent": hit_consistent,
                "cpu_rlr_distance_consistent": distance_consistent,
                "distance_tolerance_m": distance_tolerance_m,
                "passed": passed,
            }
        )
    return tuple(reports)


def simulate_compiled_acoustic_scene(
    scene: CompiledAcousticScene,
    simulation: RLRSimulationConfig,
    *,
    source: RuntimeAnchor,
    listener: RuntimeAnchor,
    scene_readback_obj: str | Path | None = None,
    ray_checks: tuple[Mapping[str, Any], ...] = (),
    ray_distance_tolerance_m: float = 1.0e-4,
    runtime_mode: str = RUNTIME_MODE_HISTORICAL,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
) -> RuntimeIRResult:
    """Upload, simulate and copy one named mono source/listener pair.

    Callers must invoke this function independently for every repeat.  A new
    native context is constructed on each call, and temporal coherence is
    rejected by :class:`RLRSimulationConfig`.
    """

    runtime_mode = require_runtime_mode(runtime_mode)
    runtime_loader_kwargs: dict[str, str | Path] = {}
    if runtime_mode == RUNTIME_MODE_CURRENT_INSTALLED:
        runtime_loader_kwargs = {
            "runtime_mode": runtime_mode,
            "runtime_prefix": runtime_prefix,
            "rlr_sdk_root": rlr_sdk_root,
            "magnum_python_site": magnum_python_site,
        }
    habitat_module, runtime_report = load_habitat_runtime(**runtime_loader_kwargs)
    native_config, config_readback = _native_configuration(habitat_module, simulation)
    runtime_report["configuration_readback"] = config_readback
    try:
        context = habitat_module.RLRAcousticContext(native_config)
        with tempfile.TemporaryDirectory(prefix="avengine-m3-rlr-db-") as temp_dir:
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
        if scene_readback_obj is not None:
            readback_path = Path(scene_readback_obj).resolve()
            readback_path.parent.mkdir(parents=True, exist_ok=True)
            raw_native_readback = context.write_scene_mesh_obj(str(readback_path))
            try:
                readback_payload = readback_path.read_bytes()
            except OSError as exc:
                raise RuntimeContractError(
                    f"RLR scene mesh readback is missing or unreadable: {exc}"
                ) from exc
            if not readback_payload:
                raise RuntimeContractError("RLR scene mesh readback is empty")
            parsed_obj = _parse_scene_obj_bytes(readback_payload)
            expected_obj = _expected_scene_readback(scene)
            if parsed_obj != expected_obj:
                raise RuntimeContractError(
                    "RLR scene OBJ geometry/material fingerprint differs from the "
                    "hash-checked package"
                )
            native_readback = _native_scene_readback_report(
                raw_native_readback,
                output_path=readback_path,
            )
            expected_native_readback = _expected_native_scene_readback_report(
                parsed_obj, upload
            )
            if native_readback != expected_native_readback:
                raise RuntimeContractError(
                    "native RLR scene readback report differs from independent "
                    "package/OBJ verification"
                )
            runtime_report["scene_mesh_readback"] = {
                "path": str(readback_path),
                "byte_size": len(readback_payload),
                "sha256": hashlib.sha256(readback_payload).hexdigest(),
                "native_report": native_readback,
                **parsed_obj,
            }
        context.add_source(source.anchor_id, source.position_m, source.radius_m)
        layout_name = {
            "mono": "Mono",
            "binaural": "Binaural",
            "ambisonics": "Ambisonics",
        }[simulation.channel_layout.layout_type]
        if not hasattr(habitat_module.RLRChannelLayoutType, layout_name):
            raise RuntimeContractError(
                f"RLRChannelLayoutType lacks required member {layout_name}"
            )
        context.add_listener(
            listener.anchor_id,
            listener.position_m,
            listener.orientation_wxyz,
            getattr(habitat_module.RLRChannelLayoutType, layout_name),
            simulation.channel_layout.channel_count,
            listener.radius_m,
            "",
        )
        raw_irs = context.simulate_owned()
        ray_efficiency = float(context.indirect_ray_efficiency())
        ray_reports = _run_ray_checks(
            context,
            scene,
            ray_checks,
            distance_tolerance_m=ray_distance_tolerance_m,
        )
    except (RuntimeContractError, RuntimeUnavailableError):
        raise
    except Exception as exc:  # native binding maps RLRA errors to RuntimeError
        raise RuntimeExecutionError(f"modern RLR simulation failed: {exc}") from exc

    matching = [
        value
        for value in raw_irs
        if getattr(value, "listener_id", None) == listener.anchor_id
        and getattr(value, "source_id", None) == source.anchor_id
    ]
    if len(raw_irs) != 1 or len(matching) != 1:
        raise RuntimeContractError(
            "RLR did not return exactly the declared named source/listener IR"
        )
    raw_ir = matching[0]
    try:
        sample_rate = float(raw_ir.sample_rate)
        channel_count = int(raw_ir.channel_count)
        sample_count = int(raw_ir.sample_count)
        samples = np.array(raw_ir.samples, dtype="<f4", order="C", copy=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeContractError(f"RLR returned a malformed owned IR: {exc}") from exc
    if (
        not math.isfinite(sample_rate)
        or not math.isclose(
            sample_rate, simulation.sample_rate_hz, rel_tol=1.0e-6, abs_tol=1.0e-3
        )
    ):
        raise RuntimeContractError("RLR IR sample rate differs from requested config")
    if (
        channel_count != simulation.channel_layout.channel_count
        or samples.shape != (channel_count, sample_count)
    ):
        raise RuntimeContractError("RLR owned IR shape/count metadata is inconsistent")
    if sample_count < 2 or not samples.flags.c_contiguous:
        raise RuntimeContractError("RLR owned IR is empty or not C-contiguous")
    if not np.all(np.isfinite(samples)):
        raise RuntimeContractError("RLR owned IR contains non-finite samples")
    if not math.isfinite(ray_efficiency) or not 0.0 <= ray_efficiency <= 1.0:
        raise RuntimeContractError("RLR indirect ray efficiency is outside [0, 1]")
    return RuntimeIRResult(
        listener_id=listener.anchor_id,
        source_id=source.anchor_id,
        sample_rate_hz=sample_rate,
        samples=samples,
        package_manifest_sha256=scene.manifest_sha256,
        package_content_sha256=scene.package_content_sha256,
        runtime=runtime_report,
        upload_report=upload,
        indirect_ray_efficiency=ray_efficiency,
        ray_checks=ray_reports,
    )


__all__ = [
    "CompiledAcousticScene",
    "RLRSimulationConfig",
    "RUNTIME_IMPORT_WORKAROUND",
    "RUNTIME_MODE_CURRENT_INSTALLED",
    "RUNTIME_MODE_HISTORICAL",
    "RuntimeAnchor",
    "RuntimeContractError",
    "RuntimeExecutionError",
    "RuntimeIRResult",
    "RuntimeUnavailableError",
    "load_compiled_acoustic_scene",
    "load_habitat_runtime",
    "require_runtime_mode",
    "simulate_compiled_acoustic_scene",
]
