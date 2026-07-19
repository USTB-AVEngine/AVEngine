"""Compile a ReplicaCAD scene plan into a UE import/runtime request.

The ReplicaCAD source has three different counts that must not be conflated:

* logical imports (templates),
* source GLB containers and the StaticMesh assets they contain, and
* logical scene instances versus the mesh actors needed to draw them.

For ``apt_0`` these are respectively 87 logical imports, 101 GLBs expanding
to 127 StaticMesh assets, and 120 logical instances expanding to 171 runtime
mesh actors.  The final count includes every URDF visual occurrence; repeated
door/drawer meshes are imported once but drawn once per link.  This module keeps
those closures explicit without importing
SPEAR, Unreal, or Habitat-Sim.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

from avengine.optional_backends.spear_replicacad import (
    ReplicaCADPlanError,
    ReplicaCADScenePlan,
    habitat_position_to_unreal_cm,
)
from avengine.optional_backends.spear_apartment import (
    BEAGLE_ASSET_ID,
    DEFAULT_ACTOR_BINDINGS,
    HUMAN_ASSET_ID,
    component_frame_delta_for_asset,
)


EXECUTION_REQUEST_SCHEMA = "avengine_optional_spear_replicacad_execution_v1"
EDITOR_RESULT_SCHEMA = "avengine_optional_spear_replicacad_editor_result_v1"
M5_1_RUNTIME_SCHEMA = "avengine_optional_spear_replicacad_m5_1_runtime_v1"
M5_1_ROUTE_SCHEMA = "avengine_m5_1_replicacad_center_route_v1"
M5_1_CAPTURE_SCHEMA = "avengine_m5_1_human_beagle_capture_v1"
M5_1_SOURCE_PROGRAM_SCHEMA = "avengine_m5_1_habitat_native_source_program_reuse_v1"
M5_1_EMITTER_SCHEMA = "avengine_m5_1_actual_emitter_trajectories_v1"
M5_1_SOURCE_BINDING_SCHEMA = "avengine_m5_1_source_actor_binding_v1"
M5_1_SOURCE_GATE_SCHEMA = "avengine_m6x_source_center_obstacle_gate_v2"
M5_1_ROOM_ID = "replicacad_apt_0"
M5_1_ROUTE_ID = "m5_1_replicacad_apt_0_human_beagle_parallel_18s_v2"
M5_1_FRAME_COUNT = 270
M5_1_FPS = 15
M5_1_TIME_BASE_HZ = 48_000
M5_1_TICKS_PER_FRAME = 3_200
M5_1_MAP_PATH = "/Game/AVEngine/Optional/ReplicaCAD/apt_0/Maps/apt_0_comparison"
APT0_EXPECTED_LOGICAL_COUNTS = {"stage": 1, "rigid": 113, "articulated": 6}


class ReplicaCADExecutionError(ValueError):
    """The UE execution request/result is incomplete or inconsistent."""


def _matrix_multiply(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [
            sum(float(left[row][inner]) * float(right[inner][column]) for inner in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _habitat_transform_matrix(transform: Any) -> list[list[float]]:
    translation = transform.translation_m
    w, x, y, z = transform.rotation_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ReplicaCADExecutionError("Habitat transform quaternion is singular")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    sx, sy, sz = transform.scale_xyz
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    scale = (sx, sy, sz)
    return [
        [
            rotation[row][column] * scale[column]
            for column in range(3)
        ]
        + [float(translation[row])]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _rotation_matrix_to_xyzw(rotation: Sequence[Sequence[float]]) -> list[float]:
    trace = sum(float(rotation[index][index]) for index in range(3))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * root
        x = (rotation[2][1] - rotation[1][2]) / root
        y = (rotation[0][2] - rotation[2][0]) / root
        z = (rotation[1][0] - rotation[0][1]) / root
    else:
        index = max(range(3), key=lambda value: float(rotation[value][value]))
        if index == 0:
            root = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
            w = (rotation[2][1] - rotation[1][2]) / root
            x = 0.25 * root
            y = (rotation[0][1] + rotation[1][0]) / root
            z = (rotation[0][2] + rotation[2][0]) / root
        elif index == 1:
            root = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
            w = (rotation[0][2] - rotation[2][0]) / root
            x = (rotation[0][1] + rotation[1][0]) / root
            y = 0.25 * root
            z = (rotation[1][2] + rotation[2][1]) / root
        else:
            root = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
            w = (rotation[1][0] - rotation[0][1]) / root
            x = (rotation[0][2] + rotation[2][0]) / root
            y = (rotation[1][2] + rotation[2][1]) / root
            z = 0.25 * root
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return [x / norm, y / norm, z / norm, w / norm]


def _habitat_matrix_to_unreal_transform(
    matrix: Sequence[Sequence[float]],
) -> dict[str, list[float]]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ReplicaCADExecutionError("Habitat affine matrix must be 4x4")
    # P swaps Habitat Y/Z.  P is its own inverse, so A_ue=P A_h P.
    permutation = (0, 2, 1)
    linear = [
        [float(matrix[permutation[row]][permutation[column]]) for column in range(3)]
        for row in range(3)
    ]
    translation = [
        100.0 * float(matrix[permutation[row]][3]) for row in range(3)
    ]
    scales = [
        math.sqrt(sum(linear[row][column] ** 2 for row in range(3)))
        for column in range(3)
    ]
    if min(scales) <= 1.0e-12:
        raise ReplicaCADExecutionError("Habitat affine matrix has singular scale")
    rotation = [
        [linear[row][column] / scales[column] for column in range(3)]
        for row in range(3)
    ]
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    orthogonality_error = max(
        abs(
            sum(rotation[row][left] * rotation[row][right] for row in range(3))
            - (1.0 if left == right else 0.0)
        )
        for left in range(3)
        for right in range(3)
    )
    if determinant < 0.999999 or determinant > 1.000001 or orthogonality_error > 1.0e-6:
        raise ReplicaCADExecutionError(
            "Habitat affine matrix contains reflection or shear and cannot be a UE Transform"
        )
    return {
        "translation_cm": translation,
        "rotation_xyzw": _rotation_matrix_to_xyzw(rotation),
        "scale_xyz": scales,
    }


def _content_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not token:
        raise ReplicaCADExecutionError("content identity cannot normalize to empty")
    return token


def _glb_document(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReplicaCADExecutionError(f"cannot read ReplicaCAD GLB: {path}") from exc
    if len(payload) < 20:
        raise ReplicaCADExecutionError(f"ReplicaCAD GLB is truncated: {path}")
    magic, version, declared_size = struct.unpack_from("<4sII", payload, 0)
    json_length, json_kind = struct.unpack_from("<II", payload, 12)
    if (
        magic != b"glTF"
        or version != 2
        or declared_size != len(payload)
        or json_kind != 0x4E4F534A
        or 20 + json_length > len(payload)
    ):
        raise ReplicaCADExecutionError(f"ReplicaCAD asset is not a complete GLB 2.0: {path}")
    try:
        document = json.loads(
            payload[20 : 20 + json_length].rstrip(b" \t\r\n\x00").decode("utf-8")
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplicaCADExecutionError(f"ReplicaCAD GLB JSON is invalid: {path}") from exc
    if not isinstance(document, Mapping):
        raise ReplicaCADExecutionError(f"ReplicaCAD GLB JSON root is invalid: {path}")
    return document


def _glb_inventory(path: Path) -> dict[str, int]:
    document = _glb_document(path)
    mesh_count = len(document.get("meshes", []))
    if mesh_count <= 0:
        raise ReplicaCADExecutionError(f"ReplicaCAD GLB contains no meshes: {path}")
    return {
        "mesh_count": mesh_count,
        "material_count": len(document.get("materials", [])),
        "texture_count": len(document.get("textures", [])),
        "image_count": len(document.get("images", [])),
    }


def _lighting_config_path(plan: ReplicaCADScenePlan) -> Path | None:
    if plan.default_lighting is None:
        return None
    relative = Path(plan.default_lighting)
    candidates = (
        plan.dataset_config_path.parent
        / "configs"
        / relative.parent
        / f"{relative.name}.lighting_config.json",
        plan.dataset_config_path.parent / f"{plan.default_lighting}.lighting_config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ReplicaCADExecutionError(
        f"default ReplicaCAD lighting config cannot be resolved: {plan.default_lighting}"
    )


def _finite_vector(value: Any, *, owner: str, length: int) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise ReplicaCADExecutionError(f"{owner} must contain {length} finite numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ReplicaCADExecutionError(f"{owner}[{index}] must be finite")
        number = float(item)
        if not math.isfinite(number):
            raise ReplicaCADExecutionError(f"{owner}[{index}] must be finite")
        result.append(number)
    return result


def _load_lighting(plan: ReplicaCADScenePlan) -> dict[str, Any]:
    path = _lighting_config_path(plan)
    if path is None:
        return {
            "default_lighting": None,
            "source_config_path": None,
            "lights": [],
            "ue_realization": "no_dataset_lights_declared",
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplicaCADExecutionError(f"cannot read ReplicaCAD lighting config: {path}") from exc
    lights = value.get("lights") if isinstance(value, Mapping) else None
    if not isinstance(lights, Mapping):
        raise ReplicaCADExecutionError("ReplicaCAD lighting config lacks lights")

    dataset = json.loads(plan.dataset_config_path.read_text(encoding="utf-8"))
    defaults = (
        dataset.get("light_setups", {}).get("default_attributes", {})
        if isinstance(dataset, Mapping)
        else {}
    )
    positive_scale = float(defaults.get("positive_intensity_scale", 1.0))
    negative_scale = float(defaults.get("negative_intensity_scale", 1.0))
    records: list[dict[str, Any]] = []
    for light_id, item in sorted(lights.items(), key=lambda pair: str(pair[0])):
        if not isinstance(item, Mapping) or item.get("type") != "point":
            raise ReplicaCADExecutionError(
                f"ReplicaCAD light {light_id!r} is not a supported point light"
            )
        position = _finite_vector(
            item.get("position"), owner=f"ReplicaCAD light {light_id} position", length=3
        )
        color = _finite_vector(
            item.get("color"), owner=f"ReplicaCAD light {light_id} color", length=3
        )
        raw_intensity = item.get("intensity")
        if isinstance(raw_intensity, bool) or not isinstance(raw_intensity, (int, float)):
            raise ReplicaCADExecutionError(f"ReplicaCAD light {light_id} intensity is invalid")
        raw = float(raw_intensity)
        if not math.isfinite(raw):
            raise ReplicaCADExecutionError(f"ReplicaCAD light {light_id} intensity is invalid")
        scale = positive_scale if raw >= 0.0 else negative_scale
        records.append(
            {
                "light_id": str(light_id),
                "type": "point",
                "habitat_position_m": position,
                "ue_position_cm": list(habitat_position_to_unreal_cm(position)),
                "color_rgb": color,
                "source_intensity": raw,
                "dataset_scaled_intensity": raw * scale,
                "ue_realization": (
                    "native_positive_point_light"
                    if raw >= 0.0
                    else "recorded_negative_fill_not_representable_by_ue_point_light"
                ),
            }
        )
    return {
        "default_lighting": plan.default_lighting,
        "source_config_path": str(path),
        "positive_intensity_scale": positive_scale,
        "negative_intensity_scale": negative_scale,
        "lights": records,
        "ue_realization": (
            "positive dataset lights are instantiated; signed negative fills remain "
            "recorded because UE point lights cannot subtract radiance"
        ),
    }


def build_replicacad_execution_request(
    plan: ReplicaCADScenePlan,
    *,
    content_root: str = "/Game/AVEngine/Optional/ReplicaCAD/apt_0",
) -> dict[str, Any]:
    """Create the complete editor/runtime request for a ReplicaCAD plan."""

    if not isinstance(plan, ReplicaCADScenePlan):
        raise ReplicaCADExecutionError("plan must be a ReplicaCADScenePlan")
    try:
        plan.assert_closed()
    except ReplicaCADPlanError as exc:
        raise ReplicaCADExecutionError(str(exc)) from exc
    if not isinstance(content_root, str) or not content_root.startswith("/Game/"):
        raise ReplicaCADExecutionError("content_root must be a /Game/ path")
    if ".." in content_root or content_root.endswith("/"):
        raise ReplicaCADExecutionError("content_root is unsafe")

    sources: dict[Path, dict[str, Any]] = {}
    import_mesh_ids: dict[str, list[str]] = {}
    for logical_import in plan.imports:
        mesh_ids: list[str] = []
        for path in logical_import.pbr_mesh_paths:
            resolved = path.resolve()
            record = sources.get(resolved)
            if record is None:
                mesh_id = f"mesh_source_{len(sources):03d}"
                inventory = _glb_inventory(resolved)
                record = {
                    "mesh_source_id": mesh_id,
                    "source_glb_path": str(resolved),
                    "destination_content_path": (
                        f"{content_root}/Meshes/{mesh_id}_{_content_token(resolved.stem)}"
                    ),
                    "source_inventory": inventory,
                    "import_policy": {
                        "backend": "UE_Interchange_glTF",
                        "import_materials": True,
                        "import_textures": True,
                        "replace_materials": False,
                    },
                }
                sources[resolved] = record
            mesh_ids.append(record["mesh_source_id"])
        import_mesh_ids[logical_import.import_id] = mesh_ids

    source_records = list(sources.values())
    mesh_count_by_source = {
        item["mesh_source_id"]: item["source_inventory"]["mesh_count"]
        for item in source_records
    }
    logical_imports = [
        {
            "import_id": item.import_id,
            "asset_kind": item.asset_kind,
            "template_name": item.template_name,
            "mesh_source_ids": import_mesh_ids[item.import_id],
            "urdf_path": str(item.urdf_path) if item.urdf_path else None,
        }
        for item in plan.imports
    ]
    source_id_by_path = {
        path: record["mesh_source_id"] for path, record in sources.items()
    }
    spawns = []
    runtime_mesh_actor_count = 0
    articulated_visual_occurrence_count = 0
    for item in plan.spawns:
        source_ids = import_mesh_ids[item.import_id]
        world_from_root_habitat = _habitat_transform_matrix(item.habitat_transform)
        visual_instances: list[dict[str, Any]] = []
        if item.asset_kind == "articulated":
            if not item.articulated_visuals:
                raise ReplicaCADExecutionError(
                    f"articulated spawn {item.spawn_id} has no URDF visual occurrences"
                )
            for visual in item.articulated_visuals:
                try:
                    source_id = source_id_by_path[visual.mesh_path.resolve()]
                except KeyError as exc:
                    raise ReplicaCADExecutionError(
                        f"articulated visual {visual.visual_id} references an "
                        "unplanned GLB"
                    ) from exc
                local_matrix = [list(row) for row in visual.root_from_visual_matrix]
                visual_instances.append(
                    {
                        "visual_id": visual.visual_id,
                        "link_name": visual.link_name,
                        "mesh_source_id": source_id,
                        "root_from_visual_habitat_matrix": local_matrix,
                        "world_transform_ue": _habitat_matrix_to_unreal_transform(
                            _matrix_multiply(world_from_root_habitat, local_matrix)
                        ),
                        "expected_mesh_actor_count": mesh_count_by_source[source_id],
                    }
                )
            articulated_visual_occurrence_count += len(visual_instances)
        else:
            identity = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            visual_instances = [
                {
                    "visual_id": f"{item.spawn_id}:visual:{index:03d}",
                    "link_name": None,
                    "mesh_source_id": source_id,
                    "root_from_visual_habitat_matrix": identity,
                    "world_transform_ue": _habitat_matrix_to_unreal_transform(
                        world_from_root_habitat
                    ),
                    "expected_mesh_actor_count": mesh_count_by_source[source_id],
                }
                for index, source_id in enumerate(source_ids)
            ]
        actor_count = sum(
            visual["expected_mesh_actor_count"] for visual in visual_instances
        )
        runtime_mesh_actor_count += actor_count
        spawns.append(
            {
                "spawn_id": item.spawn_id,
                "asset_kind": item.asset_kind,
                "source_index": item.source_index,
                "import_id": item.import_id,
                "template_name": item.template_name,
                "mesh_source_ids": source_ids,
                "visual_instances": visual_instances,
                "expected_mesh_actor_count": actor_count,
                "habitat_transform": asdict(item.habitat_transform),
                "unreal_transform": asdict(item.unreal_transform),
                "motion_type": item.motion_type,
                "translation_origin": item.translation_origin,
                "fixed_base": item.fixed_base,
                "joint_defaults": [asdict(value) for value in item.joint_defaults],
            }
        )

    counts = {
        "logical_import_count": len(plan.imports),
        "source_glb_count": len(source_records),
        "expected_imported_static_mesh_asset_count": sum(
            item["source_inventory"]["mesh_count"] for item in source_records
        ),
        "logical_instance_count": len(spawns),
        "logical_instances_by_kind": {
            "stage": len(plan.stage_spawns),
            "rigid": len(plan.rigid_spawns),
            "articulated": len(plan.articulated_spawns),
        },
        "expected_runtime_mesh_actor_count": runtime_mesh_actor_count,
        "articulated_visual_occurrence_count": articulated_visual_occurrence_count,
    }
    return {
        "schema": EXECUTION_REQUEST_SCHEMA,
        "backend_role": "comparison_visual",
        "authority": {
            "scene_layout": "ReplicaCAD_scene_instance",
            "navigation_and_source_centers": "Habitat_native_AVEngine",
            "backend_may_replan": False,
        },
        "scene": {
            "dataset_config_path": str(plan.dataset_config_path),
            "scene_instance_path": str(plan.scene_instance_path),
            "content_root": content_root,
        },
        "pbr_import": {
            "source_meshes": source_records,
            "logical_imports": logical_imports,
            "material_override_allowed": False,
        },
        "lighting": _load_lighting(plan),
        "spawns": spawns,
        "counts": counts,
        "claim_boundary": (
            "All source GLBs, PBR material/texture imports, and all logical scene "
            "instances and URDF visual occurrences are count-closed at the declared "
            "joint pose. Actual UE import/spawn counts still require an editor/runtime "
            "result validated against this request."
        ),
    }


def assert_apt0_execution_request(request: Mapping[str, Any]) -> None:
    """Freeze only the meaningful apt_0 completeness counts."""

    if request.get("schema") != EXECUTION_REQUEST_SCHEMA:
        raise ReplicaCADExecutionError("ReplicaCAD execution request schema differs")
    counts = request.get("counts")
    if not isinstance(counts, Mapping):
        raise ReplicaCADExecutionError("ReplicaCAD execution request lacks counts")
    expected = {
        "logical_import_count": 87,
        "source_glb_count": 101,
        "expected_imported_static_mesh_asset_count": 127,
        "logical_instance_count": 120,
        "logical_instances_by_kind": APT0_EXPECTED_LOGICAL_COUNTS,
        "expected_runtime_mesh_actor_count": 171,
        "articulated_visual_occurrence_count": 31,
    }
    if dict(counts) != expected:
        raise ReplicaCADExecutionError(
            f"ReplicaCAD apt_0 closure differs: observed={dict(counts)} expected={expected}"
        )
    if request.get("lighting", {}).get("default_lighting") != (
        "lighting/frl_apartment_stage"
    ):
        raise ReplicaCADExecutionError("ReplicaCAD apt_0 default lighting differs")


def validate_replicacad_editor_result(
    request: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate actual UE import/spawn readback against the prepared request."""

    if result.get("schema") != EDITOR_RESULT_SCHEMA or result.get("status") != "pass":
        raise ReplicaCADExecutionError("ReplicaCAD UE editor result did not pass")
    counts = request.get("counts")
    observed = result.get("counts")
    if not isinstance(counts, Mapping) or not isinstance(observed, Mapping):
        raise ReplicaCADExecutionError("ReplicaCAD request/result count block is missing")
    required_pairs = {
        "imported_source_glb_count": "source_glb_count",
        "imported_static_mesh_asset_count": "expected_imported_static_mesh_asset_count",
        "logical_instance_count": "logical_instance_count",
        "spawned_static_mesh_actor_count": "expected_runtime_mesh_actor_count",
        "articulated_visual_occurrence_count": "articulated_visual_occurrence_count",
    }
    for result_key, request_key in required_pairs.items():
        if observed.get(result_key) != counts.get(request_key):
            raise ReplicaCADExecutionError(
                f"ReplicaCAD UE {result_key}={observed.get(result_key)!r} does not "
                f"close over request {request_key}={counts.get(request_key)!r}"
            )
    if observed.get("logical_instances_by_kind") != counts.get(
        "logical_instances_by_kind"
    ):
        raise ReplicaCADExecutionError("ReplicaCAD UE logical kind counts differ")
    requested_ids = [item.get("spawn_id") for item in request.get("spawns", [])]
    observed_ids = result.get("logical_spawn_ids")
    if observed_ids != requested_ids or len(requested_ids) != len(set(requested_ids)):
        raise ReplicaCADExecutionError("ReplicaCAD UE logical spawn ID closure differs")
    pbr = result.get("pbr_readback")
    if (
        not isinstance(pbr, Mapping)
        or int(pbr.get("material_asset_count", 0)) <= 0
        or int(pbr.get("texture_asset_count", 0)) <= 0
        or pbr.get("material_overrides_applied") is not False
    ):
        raise ReplicaCADExecutionError("ReplicaCAD UE PBR readback is incomplete")
    return dict(result)


def replicacad_fixed_exposure_profile(*, output_gain: float = 1.0) -> dict[str, Any]:
    """Return the deterministic review exposure for imported ReplicaCAD PBR.

    ReplicaCAD declares seven signed Habitat lights.  The editor importer
    instantiates the five positive point lights and records the two negative
    fills, which UE cannot represent as subtractive lights.  This profile does
    not silently add a skylight or key light: it disables temporal eye
    adaptation and measures the resulting dataset-light-only pixels.
    """

    if isinstance(output_gain, bool) or not isinstance(output_gain, (int, float)):
        raise ReplicaCADExecutionError("ReplicaCAD output_gain must be numeric")
    gain = float(output_gain)
    if not math.isfinite(gain) or not 0.25 <= gain <= 2.0:
        raise ReplicaCADExecutionError("ReplicaCAD output_gain must be in [0.25,2.0]")
    return {
        "profile_id": "replicacad_dataset_point_lights_fixed_exposure_v1",
        "eye_adaptation": "disabled",
        "console_commands": [
            "r.DefaultFeature.AutoExposure 0",
            "r.EyeAdaptationQuality 0",
        ],
        "fixed_output_gain": gain,
        "dataset_declared_light_count": 7,
        "runtime_positive_point_light_count": 5,
        "recorded_negative_fill_count": 2,
        "review_light_added": False,
        "qa": {
            "luminance_saturation_threshold": 0.98,
            "nonblack_luminance_threshold": 0.01,
            "minimum_nonblack_fraction": 0.05,
            "minimum_mean_luminance": 0.01,
            "minimum_p95_luminance": 0.025,
            "maximum_saturated_fraction": 0.05,
            "maximum_mean_luminance": 0.80,
            "maximum_p95_luminance": 0.98,
        },
        "claim_boundary": (
            "Five positive lights from ReplicaCAD's declared lighting setup are "
            "rendered with shadows; the two signed negative fills remain recorded "
            "but are not representable by UE point lights. No review light is added."
        ),
    }


def _close_vector(
    left: Sequence[float], right: Sequence[float], *, tolerance: float = 1.0e-6
) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right)
    )


def _m5_1_routes(route_manifest: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    if (
        route_manifest.get("schema") != M5_1_ROUTE_SCHEMA
        or route_manifest.get("route_id") != M5_1_ROUTE_ID
        or route_manifest.get("room_id") != M5_1_ROOM_ID
        or route_manifest.get("frame_count") != M5_1_FRAME_COUNT
        or route_manifest.get("frame_rate_hz") != M5_1_FPS
        or route_manifest.get("center_navigation_semantics")
        != "actor_root_center_only"
    ):
        raise ReplicaCADExecutionError("retained ReplicaCAD route authority changed")
    routes = route_manifest.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != {"human0", "dog0"}:
        raise ReplicaCADExecutionError("ReplicaCAD route actor closure differs")
    result: dict[str, list[list[float]]] = {}
    for actor_id in ("human0", "dog0"):
        value = routes[actor_id]
        if not isinstance(value, Mapping):
            raise ReplicaCADExecutionError(f"ReplicaCAD route {actor_id} is invalid")
        start = _finite_vector(
            value.get("start_m"), owner=f"ReplicaCAD {actor_id} route start", length=3
        )
        end = _finite_vector(
            value.get("end_m"), owner=f"ReplicaCAD {actor_id} route end", length=3
        )
        result[actor_id] = [
            [
                start[axis]
                + (end[axis] - start[axis]) * frame / (M5_1_FRAME_COUNT - 1)
                for axis in range(3)
            ]
            for frame in range(M5_1_FRAME_COUNT)
        ]
    return result


def _validate_m5_1_source_authority(
    *,
    source_center_gate: Mapping[str, Any],
    source_program: Mapping[str, Any],
    emitter_trajectories: Mapping[str, Any],
    source_actor_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    expected_ids = ["source0", "source1"]
    sources = source_center_gate.get("sources")
    if (
        source_center_gate.get("schema") != M5_1_SOURCE_GATE_SCHEMA
        or source_center_gate.get("status") != "pass"
        or source_center_gate.get("pathfinder_snapshot_match") is not True
        or source_center_gate.get("full_body_collision_claim") is not False
        or source_center_gate.get("failed_source_frame_indices") != {}
        or not isinstance(sources, Mapping)
        or sorted(sources) != expected_ids
    ):
        raise ReplicaCADExecutionError("ReplicaCAD source-center gate changed")
    source_gate_summary: dict[str, Any] = {}
    for source_id in expected_ids:
        value = sources[source_id]
        if (
            not isinstance(value, Mapping)
            or value.get("status") != "pass"
            or value.get("frame_count") != M5_1_FRAME_COUNT
            or value.get("failed_frame_indices") != []
        ):
            raise ReplicaCADExecutionError(
                f"ReplicaCAD source-center gate failed for {source_id}"
            )
        source_gate_summary[source_id] = {
            key: deepcopy(value.get(key))
            for key in (
                "status",
                "frame_count",
                "failed_frame_indices",
                "minimum_navmesh_clearance_m",
                "minimum_loaded_rigid_clearance_m",
                "minimum_blocking_loaded_rigid_clearance_m",
            )
        }

    program_sources = source_program.get("sources")
    if (
        source_program.get("schema") != M5_1_SOURCE_PROGRAM_SCHEMA
        or source_program.get("room_family") != "replicacad"
        or not isinstance(program_sources, list)
        or sorted(item.get("source_id") for item in program_sources) != expected_ids
    ):
        raise ReplicaCADExecutionError("ReplicaCAD source program changed")

    trajectories = emitter_trajectories.get("sources")
    if (
        emitter_trajectories.get("schema") != M5_1_EMITTER_SCHEMA
        or emitter_trajectories.get("source_ids") != expected_ids
        or not isinstance(trajectories, Mapping)
        or sorted(trajectories) != expected_ids
    ):
        raise ReplicaCADExecutionError("ReplicaCAD emitter trajectory closure differs")
    trajectory_summary: dict[str, Any] = {}
    for source_id in expected_ids:
        value = trajectories[source_id]
        positions = value.get("positions_m") if isinstance(value, Mapping) else None
        if (
            not isinstance(value, Mapping)
            or value.get("source_id") != source_id
            or value.get("frame_count") != M5_1_FRAME_COUNT
            or not isinstance(positions, list)
            or len(positions) != M5_1_FRAME_COUNT
        ):
            raise ReplicaCADExecutionError(
                f"ReplicaCAD emitter trajectory failed for {source_id}"
            )
        trajectory_summary[source_id] = {
            "source_id": source_id,
            "actor_id": value.get("actor_id"),
            "emitter_anchor_id": value.get("emitter_anchor_id"),
            "frame_count": M5_1_FRAME_COUNT,
            "trajectory_content_sha256": value.get("trajectory_content_sha256"),
        }

    bindings = source_actor_bindings.get("bindings")
    if (
        source_actor_bindings.get("schema") != M5_1_SOURCE_BINDING_SCHEMA
        or source_actor_bindings.get("room_id") != M5_1_ROOM_ID
        or source_actor_bindings.get("route_id") != M5_1_ROUTE_ID
        or source_actor_bindings.get("source_ids") != expected_ids
        or not isinstance(bindings, Mapping)
        or sorted(bindings) != expected_ids
        or bindings["source0"].get("actor_id") != "human0"
        or bindings["source1"].get("actor_id") != "dog0"
    ):
        raise ReplicaCADExecutionError("ReplicaCAD source/actor bindings changed")

    return {
        "source_ids": expected_ids,
        "source_actor_bindings": deepcopy(bindings),
        "event_windows": {
            item["source_id"]: deepcopy(item.get("event_windows", []))
            for item in program_sources
        },
        "emitter_trajectories": trajectory_summary,
        "source_center_gate": {
            "status": "pass",
            "semantics": source_center_gate.get("semantics"),
            "pathfinder_snapshot_match": True,
            "full_body_collision_claim": False,
            "sources": source_gate_summary,
        },
    }


def build_m5_1_replicacad_runtime_plan(
    *,
    route_manifest: Mapping[str, Any],
    capture_evidence: Mapping[str, Any],
    frame_readback: Sequence[Mapping[str, Any]],
    source_center_gate: Mapping[str, Any],
    source_program: Mapping[str, Any],
    emitter_trajectories: Mapping[str, Any],
    source_actor_bindings: Mapping[str, Any],
    execution_request: Mapping[str, Any],
    editor_import_result: Mapping[str, Any],
    editor_reload_result: Mapping[str, Any],
    output_gain: float = 1.0,
) -> dict[str, Any]:
    """Compile the retained 270-frame ReplicaCAD route for actual SPEAR pixels."""

    assert_apt0_execution_request(execution_request)
    imported = validate_replicacad_editor_result(
        execution_request, editor_import_result
    )
    reloaded = validate_replicacad_editor_result(
        execution_request, editor_reload_result
    )
    if (
        imported.get("map", {}).get("object_path") != M5_1_MAP_PATH
        or reloaded.get("map", {}).get("object_path") != M5_1_MAP_PATH
        or reloaded.get("map", {}).get("reloaded") is not True
        or reloaded.get("reload_verification") != "pass"
        or imported.get("lighting", {}).get("positive_dataset_light_count") != 5
        or reloaded.get("lighting", {}).get("positive_dataset_light_count") != 5
    ):
        raise ReplicaCADExecutionError("ReplicaCAD editor import/reload evidence differs")

    routes = _m5_1_routes(route_manifest)
    camera = capture_evidence.get("camera")
    if (
        capture_evidence.get("schema") != M5_1_CAPTURE_SCHEMA
        or capture_evidence.get("status") != "pass"
        or capture_evidence.get("room_id") != M5_1_ROOM_ID
        or capture_evidence.get("route_id") != M5_1_ROUTE_ID
        or capture_evidence.get("frame_count") != M5_1_FRAME_COUNT
        or capture_evidence.get("frame_rate_hz") != M5_1_FPS
        or capture_evidence.get("time_base_hz") != M5_1_TIME_BASE_HZ
        or capture_evidence.get("qualification_claim") is not False
        or capture_evidence.get("research_only") is not True
        or not isinstance(camera, Mapping)
        or camera.get("position_m") != [2.6, 1.47, 3.4]
        or camera.get("rotation_xyzw") != [0, 1, 0, 0]
        or camera.get("horizontal_fov_deg") != 90
    ):
        raise ReplicaCADExecutionError("ReplicaCAD retained capture authority changed")
    if isinstance(frame_readback, (str, bytes)) or len(frame_readback) != M5_1_FRAME_COUNT:
        raise ReplicaCADExecutionError("ReplicaCAD capture readback must contain 270 frames")

    source_logic = _validate_m5_1_source_authority(
        source_center_gate=source_center_gate,
        source_program=source_program,
        emitter_trajectories=emitter_trajectories,
        source_actor_bindings=source_actor_bindings,
    )
    bindings = DEFAULT_ACTOR_BINDINGS
    actors = {
        "human0": {
            "actor_id": "human0",
            "asset_id": HUMAN_ASSET_ID,
            "source_id": "source0",
            "actor_class": "human",
            "blueprint_class_path": bindings[HUMAN_ASSET_ID]["blueprint_class_path"],
            "walking_animation": bindings[HUMAN_ASSET_ID]["walking_animation"],
            "action_sample_count": 16,
            "animation_clip_start_seconds": 1.0 / 30.0,
            "actor_yaw_ue_deg": 0.0,
            "ue_component_frame_delta": component_frame_delta_for_asset(HUMAN_ASSET_ID),
        },
        "dog0": {
            "actor_id": "dog0",
            "asset_id": BEAGLE_ASSET_ID,
            "source_id": "source1",
            "actor_class": "dog",
            "blueprint_class_path": bindings[BEAGLE_ASSET_ID]["blueprint_class_path"],
            "walking_animation": bindings[BEAGLE_ASSET_ID]["walking_animation"],
            "action_sample_count": 25,
            "validated_walk_state_count": 45,
            "animation_clip_start_seconds": 0.0,
            "actor_yaw_ue_deg": 90.0,
            "ue_component_frame_delta": component_frame_delta_for_asset(BEAGLE_ASSET_ID),
        },
    }

    frames: list[dict[str, Any]] = []
    for frame_index, record in enumerate(frame_readback):
        if (
            not isinstance(record, Mapping)
            or record.get("frame_index") != frame_index
            or record.get("pts_ticks") != frame_index * M5_1_TICKS_PER_FRAME
        ):
            raise ReplicaCADExecutionError(
                f"ReplicaCAD capture clock differs at frame {frame_index}"
            )
        states: list[dict[str, Any]] = []
        for actor_id, record_key in (("human0", "human"), ("dog0", "beagle")):
            actor_record = record.get(record_key)
            if not isinstance(actor_record, Mapping):
                raise ReplicaCADExecutionError(
                    f"ReplicaCAD frame {frame_index} lacks {actor_id}"
                )
            position = _finite_vector(
                actor_record.get("actor_root_position_m"),
                owner=f"ReplicaCAD frame {frame_index} {actor_id} root",
                length=3,
            )
            if not _close_vector(position, routes[actor_id][frame_index]):
                raise ReplicaCADExecutionError(
                    f"ReplicaCAD frame {frame_index} {actor_id} root differs from route"
                )
            sample_count = int(actors[actor_id]["action_sample_count"])
            expected_sample = frame_index % sample_count
            if actor_id == "dog0":
                expected_sample = (
                    frame_index % int(actors[actor_id]["validated_walk_state_count"])
                ) % sample_count
            sample_index = actor_record.get("action_sample_index")
            if sample_index != expected_sample:
                raise ReplicaCADExecutionError(
                    f"ReplicaCAD frame {frame_index} {actor_id} action sample differs"
                )
            states.append(
                {
                    "actor_id": actor_id,
                    "source_id": actors[actor_id]["source_id"],
                    "action_id": "walk",
                    "action_sample_index": sample_index,
                    "action_sample_count": sample_count,
                    "action_phase": sample_index / sample_count,
                    "animation_position_seconds": (
                        float(actors[actor_id]["animation_clip_start_seconds"])
                        + sample_index / M5_1_FPS
                    ),
                    "ue_animation": actors[actor_id]["walking_animation"],
                    "translation_m": position,
                    "translation_ue_cm": list(habitat_position_to_unreal_cm(position)),
                    "actor_yaw_ue_deg": actors[actor_id]["actor_yaw_ue_deg"],
                }
            )
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * M5_1_TICKS_PER_FRAME,
                "actor_states": states,
            }
        )

    return {
        "schema": M5_1_RUNTIME_SCHEMA,
        "backend_role": "comparison_visual",
        "room_id": M5_1_ROOM_ID,
        "route_id": M5_1_ROUTE_ID,
        "authority": {
            "actor_state": "retained_M5.1_route_and_capture_frame_readback",
            "navigation": "Habitat_PathFinder_and_live_furniture_source_center_gate",
            "source_logic": "retained_Habitat_native_source_program",
            "source_positions": "retained_articulated_emitter_link_readback",
            "audio_and_topdown": "retained_Habitat_native_delivery",
            "backend_may_replan": False,
        },
        "route_characterization": {
            "retained_legacy_compatibility_route": True,
            "distance_m_by_actor": {"human0": 1.2, "dog0": 1.2},
            "nominal_duration_seconds": 18.0,
            "nominal_speed_m_per_s_by_actor": {
                "human0": 1.2 / 18.0,
                "dog0": 1.2 / 18.0,
            },
            "normal_speed_issue_resolved": False,
            "note": (
                "This retained Habitat-authoritative route is intentionally reused "
                "for visual-engine compatibility. Its approximately 0.067 m/s speed "
                "is too slow for a normal walk and must not be presented as the "
                "later normal-speed route fix."
            ),
        },
        "clock": {
            "timeline_v2_applicable": False,
            "timeline_v2_non_applicability_reason": (
                "this retained compatibility route is 270 frames/18 seconds; "
                "the frozen Timeline-v2 episode is 75 frames/5 seconds"
            ),
            "compatibility_authority_schema": M5_1_ROUTE_SCHEMA,
            "time_base_hz": M5_1_TIME_BASE_HZ,
            "ticks_per_frame": M5_1_TICKS_PER_FRAME,
            "duration_ticks": M5_1_FRAME_COUNT * M5_1_TICKS_PER_FRAME,
            "frame_count": M5_1_FRAME_COUNT,
            "fps_num": M5_1_FPS,
            "fps_den": 1,
            "sample_rate_hz": 16_000,
            "sample_count": 288_000,
        },
        "scene": {
            "map_path": M5_1_MAP_PATH,
            "logical_instance_count": 120,
            "static_mesh_actor_count": 171,
            "imported_static_mesh_asset_count": 127,
            "pbr_material_count": imported["pbr_readback"]["material_asset_count"],
            "pbr_texture_count": imported["pbr_readback"]["texture_asset_count"],
            "runtime_positive_point_light_count": 5,
            "declared_dataset_light_count": 7,
            "review_light_added": False,
            "collision_authority": "Habitat_native_source_center_gate",
        },
        "render": {
            "width": 1280,
            "height": 720,
            "frame_count": M5_1_FRAME_COUNT,
            "fps_num": M5_1_FPS,
            "fps_den": 1,
            "streaming_warmup_frames": 120,
            "camera_warmup_frames": 40,
        },
        "exposure_and_lighting": replicacad_fixed_exposure_profile(
            output_gain=output_gain
        ),
        "camera": {
            "habitat_position_m": [2.6, 1.47, 3.4],
            "ue_position_cm": [260.0, 340.0, 147.0],
            "habitat_rotation_xyzw": [0.0, 1.0, 0.0, 0.0],
            "ue_yaw_deg": 90.0,
            "horizontal_fov_deg": 90.0,
        },
        "actors": [actors["human0"], actors["dog0"]],
        "frames": frames,
        "source_logic": source_logic,
        "qualification": {
            "visual_runtime_status": "pending_actual_pixel_run",
            "navigation_status": "pass",
            "source_center_status": "pass",
            "source_center_semantics": "source_center_only",
            "full_body_clearance_claim": False,
            "ue_collision_authority": False,
            "dataset_admission": False,
        },
        "claim_boundary": (
            "SPEAR/UE renders the imported ReplicaCAD PBR scene and retained "
            "human/Beagle root route only. Habitat-native AVEngine remains the "
            "navigation, emitter trajectory, source-center, audio and Topdown "
            "authority. Both roots travel only 1.2 m in 18 seconds (about 0.067 "
            "m/s), so this retained slow route does not resolve the normal-speed "
            "requirement. This is an 18-second research compatibility canary, not "
            "Timeline-v2 conformance, full-body clearance, acoustic parity or "
            "dataset admission."
        ),
    }


__all__ = [
    "APT0_EXPECTED_LOGICAL_COUNTS",
    "EDITOR_RESULT_SCHEMA",
    "EXECUTION_REQUEST_SCHEMA",
    "M5_1_CAPTURE_SCHEMA",
    "M5_1_FPS",
    "M5_1_FRAME_COUNT",
    "M5_1_MAP_PATH",
    "M5_1_ROOM_ID",
    "M5_1_ROUTE_ID",
    "M5_1_RUNTIME_SCHEMA",
    "ReplicaCADExecutionError",
    "assert_apt0_execution_request",
    "build_replicacad_execution_request",
    "build_m5_1_replicacad_runtime_plan",
    "replicacad_fixed_exposure_profile",
    "validate_replicacad_editor_result",
]
