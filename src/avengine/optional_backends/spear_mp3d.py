"""Execution planning for the optional MP3D UE comparison renderer."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.optional_backends.spear_visual import BACKEND_ROLE, PLAN_SCHEMA
from avengine.optional_backends.spear_apartment import (
    BEAGLE_ASSET_ID,
    HUMAN_ASSET_ID,
    component_frame_delta_for_asset,
)


EXECUTION_SCHEMA = "avengine_optional_spear_mp3d_execution_v1"
IMPORT_SCHEMA = "avengine_mp3d_ue_import_result_v1"
MATERIAL_COLOR_SCHEMA = "avengine_optional_spear_mp3d_material_color_v1"
MP3D_ROOM_ID = "habitat_mp3d_example_17DRP5sb8fy"
EXPECTED_SCENE_MESH_COUNT = 71
EXPECTED_SCENE_MATERIAL_COUNT = 23
EXPECTED_SCENE_TEXTURE_COUNT = 23

# M5.1 predates the frozen five-second Timeline-v2 schema.  Its retained MP3D
# review is an 18-second compatibility authority and must never be relabelled
# as the 75-frame Timeline-v2 contract.
M5_1_EXECUTION_SCHEMA = "avengine_optional_spear_mp3d_m5_1_execution_v1"
M5_1_ROUTE_SCHEMA = "avengine_m5_1_mp3d_center_route_v1"
M5_1_CAPTURE_SCHEMA = "avengine_m5_1_human_beagle_capture_v1"
M5_1_CAPTURE_INSTALLED_SCHEMA_V2 = "avengine_m5_1_human_beagle_capture_v2"
M5_1_CAPTURE_SCHEMAS = (
    M5_1_CAPTURE_SCHEMA,
    M5_1_CAPTURE_INSTALLED_SCHEMA_V2,
)
M5_1_GATE_SCHEMA = "avengine_m5_1_mp3d_mixed_visual_gate_v1"
M5_1_SOURCE_PROGRAM_SCHEMA = "avengine_m5_1_mp3d_source_program_reuse_v1"
M5_1_EMITTER_SCHEMA = "avengine_m5_1_actual_emitter_trajectories_v1"
M5_1_FRAME_COUNT = 270
M5_1_FPS = 15
M5_1_TIME_BASE_HZ = 48_000
M5_1_TICKS_PER_FRAME = 3_200
M5_1_DURATION_TICKS = M5_1_FRAME_COUNT * M5_1_TICKS_PER_FRAME
M5_1_SAMPLE_RATE_HZ = 16_000
M5_1_SAMPLE_COUNT = 288_000
M5_1_ROUTE_ID = "m5_1_mp3d_human_beagle_parallel_18s_v1"
M5_1_CAMERA_HABITAT_M = (-4.1499128342, 1.572447, -1.2454376221)
M5_1_CAMERA_UE_CM = (-414.99128342, -124.54376221, 157.2447)
M5_1_CAMERA_UE_YAW_DEG = -90.0
M5_1_CAMERA_HFOV_DEG = 90.0
M5_1_ACTOR_ORDER = ("human0", "dog0")
M5_1_SOURCE_TO_ACTOR = {"source0": "human0", "source1": "dog0"}

HUMAN_BP_CLASS_PATH = (
    "/Game/MyAssets/Audioset/Blueprints/"
    "gate_rocketbox_male_adult_01_original_ue_v3/"
    "BP_gate_rocketbox_male_adult_01_original_ue_v3."
    "BP_gate_rocketbox_male_adult_01_original_ue_v3_C"
)
DOG_BP_CLASS_PATH = (
    "/Game/MyAssets/Audioset/Blueprints/gate_m2_beagle_v7_world_contact_r5/"
    "BP_gate_m2_beagle_v7_world_contact_r5."
    "BP_gate_m2_beagle_v7_world_contact_r5_C"
)


class MP3DExecutionError(ValueError):
    """The MP3D UE plan cannot preserve the authoritative visual contract."""


def fixed_exposure_profile(*, output_gain: float = 1.0) -> dict[str, Any]:
    """Return a deterministic anti-overexposure profile for scanned textures.

    MP3D base-color textures already contain illumination from the Matterport
    scan.  The UE pass therefore uses a weak shadow key, disables temporal eye
    adaptation, and begins at a neutral display-domain gain of one.  It does
    not claim to reconstruct the original Matterport lights.  Color-space
    correctness is enforced separately: display-domain gain must never be
    used to hide an sRGB/base-color import error.
    """

    if isinstance(output_gain, bool) or not isinstance(output_gain, (int, float)):
        raise MP3DExecutionError("output_gain must be finite and positive")
    gain = float(output_gain)
    if not math.isfinite(gain) or not 0.1 <= gain <= 1.0:
        raise MP3DExecutionError("output_gain must be in [0.1,1.0]")
    return {
        "profile_id": "mp3d_srgb_basecolor_fixed_exposure_v2",
        "eye_adaptation": "disabled",
        "console_commands": [
            "r.DefaultFeature.AutoExposure 0",
            "r.EyeAdaptationQuality 0",
        ],
        "directional_key": {
            "yaw_deg": -45.0,
            "pitch_deg": -50.0,
            "intensity_lux": 3.0,
            "cast_dynamic_shadows": True,
        },
        "skylight_intensity": 0.15,
        "fixed_output_gain": gain,
        "qa": {
            "luminance_saturation_threshold": 0.98,
            "nonblack_luminance_threshold": 0.01,
            "minimum_nonblack_fraction": 0.05,
            "minimum_mean_luminance": 0.015,
            "minimum_p95_luminance": 0.04,
            "maximum_saturated_fraction": 0.02,
            "maximum_mean_luminance": 0.65,
            "maximum_p95_luminance": 0.90,
            "lower_bound_status": (
                "calibrated against packaged SPEAR smoke; rejects black frames"
            ),
        },
        "color_qa": {
            "chroma_threshold": 0.04,
            "minimum_reference_mean_chroma": 0.03,
            "minimum_ue_mean_chroma": 0.03,
            "minimum_mean_chroma_ratio": 0.80,
            "maximum_mean_chroma_ratio": 1.80,
            "minimum_chromatic_fraction_ratio": 0.75,
            "maximum_chromatic_fraction_ratio": 1.80,
        },
        "claim_boundary": (
            "Review lighting for shadow readability over illumination-baked scan "
            "textures; not a reconstruction of Matterport capture lighting."
        ),
    }


def _validated_material_color_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require a fresh-editor readback of MP3D base-color bindings.

    UE 5.5 Interchange imported the MP3D JPEG atlases with ``sRGB=False`` in
    the first comparison.  Merely counting Texture2D and material assets did
    not detect that semantic error.  This validator therefore closes the
    exact 23-texture/23-material base-color relation after a second editor
    process has reloaded the saved assets.
    """

    base_color_textures = value.get("base_color_textures")
    occlusion_textures = value.get("occlusion_textures")
    bindings = value.get("material_bindings")
    counts = value.get("counts")
    if (
        value.get("schema") != MATERIAL_COLOR_SCHEMA
        or value.get("status") != "pass"
        or value.get("operation") != "verify_only"
        or value.get("fresh_editor_reload") is not True
        or value.get("content_root")
        != "/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy"
        or not isinstance(counts, Mapping)
        or counts.get("source_texture_count") != EXPECTED_SCENE_TEXTURE_COUNT
        or counts.get("material_count") != EXPECTED_SCENE_MATERIAL_COUNT
        or counts.get("base_color_binding_count") != EXPECTED_SCENE_MATERIAL_COUNT
        or counts.get("base_color_texture_count") != EXPECTED_SCENE_TEXTURE_COUNT
        or counts.get("base_color_srgb_true_count") != EXPECTED_SCENE_TEXTURE_COUNT
        or counts.get("occlusion_binding_count") != EXPECTED_SCENE_MATERIAL_COUNT
        or counts.get("occlusion_texture_count") != EXPECTED_SCENE_TEXTURE_COUNT
        or counts.get("occlusion_srgb_false_count") != EXPECTED_SCENE_TEXTURE_COUNT
        or counts.get("unexpected_texture_binding_count") != 0
        or not isinstance(base_color_textures, list)
        or len(base_color_textures) != EXPECTED_SCENE_TEXTURE_COUNT
        or not isinstance(occlusion_textures, list)
        or len(occlusion_textures) != EXPECTED_SCENE_TEXTURE_COUNT
        or not isinstance(bindings, list)
        or len(bindings) != EXPECTED_SCENE_MATERIAL_COUNT
    ):
        raise MP3DExecutionError(
            "MP3D UE material-color result is not a fresh 23/23 sRGB readback"
        )

    base_color_paths: list[str] = []
    for record in base_color_textures:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("texture_path"), str)
            or record.get("srgb") is not True
            or record.get("semantic") != "base_color_srgb"
        ):
            raise MP3DExecutionError("MP3D UE base-color Texture2D is not sRGB")
        base_color_paths.append(record["texture_path"])
    if len(set(base_color_paths)) != EXPECTED_SCENE_TEXTURE_COUNT:
        raise MP3DExecutionError("MP3D UE base-color Texture2D paths are not unique")

    occlusion_paths: list[str] = []
    for record in occlusion_textures:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("texture_path"), str)
            or record.get("srgb") is not False
            or record.get("semantic") != "occlusion_linear_red_channel"
        ):
            raise MP3DExecutionError("MP3D UE occlusion Texture2D is not linear")
        occlusion_paths.append(record["texture_path"])
    if len(set(occlusion_paths)) != EXPECTED_SCENE_TEXTURE_COUNT or set(
        occlusion_paths
    ) & set(base_color_paths):
        raise MP3DExecutionError("MP3D UE base-color/occlusion texture views overlap")

    material_paths: list[str] = []
    bound_base_color_paths: list[str] = []
    bound_occlusion_paths: list[str] = []
    for record in bindings:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("material_path"), str)
            or record.get("base_color_parameter_name") != "BaseColorTexture"
            or not isinstance(record.get("base_color_texture_path"), str)
            or record.get("occlusion_parameter_name") != "OcclusionTexture"
            or not isinstance(record.get("occlusion_texture_path"), str)
            or record.get("unexpected_bound_texture_parameters") != []
        ):
            raise MP3DExecutionError(
                "MP3D UE BaseColorTexture/OcclusionTexture binding is incomplete"
            )
        material_paths.append(record["material_path"])
        bound_base_color_paths.append(record["base_color_texture_path"])
        bound_occlusion_paths.append(record["occlusion_texture_path"])
    if (
        len(set(material_paths)) != EXPECTED_SCENE_MATERIAL_COUNT
        or set(bound_base_color_paths) != set(base_color_paths)
        or set(bound_occlusion_paths) != set(occlusion_paths)
    ):
        raise MP3DExecutionError("MP3D UE material color-slot closure differs")

    source = value.get("source_gltf_contract")
    if (
        not isinstance(source, Mapping)
        or source.get("material_count") != EXPECTED_SCENE_MATERIAL_COUNT
        or source.get("base_color_reference_count") != EXPECTED_SCENE_MATERIAL_COUNT
        or source.get("occlusion_reference_count") != EXPECTED_SCENE_MATERIAL_COUNT
        or source.get("shared_base_color_and_occlusion_texture_count")
        != EXPECTED_SCENE_MATERIAL_COUNT
        or source.get("other_texture_reference_count") != 0
    ):
        raise MP3DExecutionError("MP3D source glTF color-slot contract differs")

    return {
        "status": "pass",
        "schema": MATERIAL_COLOR_SCHEMA,
        "fresh_editor_reload": True,
        "texture_count": EXPECTED_SCENE_TEXTURE_COUNT,
        "material_count": EXPECTED_SCENE_MATERIAL_COUNT,
        "base_color_binding_count": EXPECTED_SCENE_MATERIAL_COUNT,
        "base_color_srgb_true_count": EXPECTED_SCENE_TEXTURE_COUNT,
        "occlusion_binding_count": EXPECTED_SCENE_MATERIAL_COUNT,
        "occlusion_srgb_false_count": EXPECTED_SCENE_TEXTURE_COUNT,
        "source_slots_share_one_texture": True,
        "ue_uses_distinct_color_space_views": True,
    }


def _validated_import_manifest(value: Mapping[str, Any]) -> list[str]:
    scene = value.get("scene_content")
    reload_verification = value.get("reload_verification")
    meshes = scene.get("static_meshes") if isinstance(scene, Mapping) else None
    if (
        value.get("schema") != IMPORT_SCHEMA
        or value.get("status") != "passed"
        or not isinstance(reload_verification, Mapping)
        or reload_verification.get("status") != "passed"
        or not isinstance(meshes, list)
        or len(meshes) != EXPECTED_SCENE_MESH_COUNT
        or len(set(meshes)) != EXPECTED_SCENE_MESH_COUNT
        or any(
            not isinstance(path, str) or not path.startswith("/Game/")
            for path in meshes
        )
        or int(scene.get("material_count", 0)) <= 0
        or int(scene.get("texture_count", 0)) <= 0
    ):
        raise MP3DExecutionError("MP3D UE import manifest is not reload/PBR complete")
    return list(meshes)


def build_mp3d_execution_plan(
    *,
    visual_plan: Mapping[str, Any],
    ue_import_manifest: Mapping[str, Any],
    ue_material_color_result: Mapping[str, Any],
    output_gain: float = 1.0,
) -> dict[str, Any]:
    """Bind a current Timeline-v2 visual plan to the imported MP3D scene."""

    if visual_plan.get("schema") != PLAN_SCHEMA:
        raise MP3DExecutionError("MP3D execution requires a compiled SPEAR visual plan")
    if visual_plan.get("backend_role") != BACKEND_ROLE:
        raise MP3DExecutionError("MP3D backend role must remain comparison_visual")
    authority = visual_plan.get("authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("backend_may_replan") is not False
    ):
        raise MP3DExecutionError("MP3D execution cannot grant UE replanning authority")
    room = visual_plan.get("room")
    if not isinstance(room, Mapping) or room.get("room_id") != MP3D_ROOM_ID:
        raise MP3DExecutionError(f"MP3D visual plan room_id must be {MP3D_ROOM_ID!r}")
    render = visual_plan.get("render")
    frames = visual_plan.get("frames")
    actors = visual_plan.get("actors")
    if (
        not isinstance(render, Mapping)
        or render.get("frame_count") != 75
        or render.get("fps_num") != 15
        or render.get("fps_den") != 1
        or not isinstance(frames, list)
        or len(frames) != 75
        or not isinstance(actors, list)
        or not actors
    ):
        raise MP3DExecutionError(
            "MP3D visual plan must retain the current 75-frame clock"
        )
    mesh_paths = _validated_import_manifest(ue_import_manifest)
    material_color = _validated_material_color_result(ue_material_color_result)

    return {
        "schema": EXECUTION_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "room_id": MP3D_ROOM_ID,
        "authority": deepcopy(authority),
        "scene": {
            "static_mesh_object_paths": mesh_paths,
            "spawned_scene_mesh_actor_count": EXPECTED_SCENE_MESH_COUNT,
            "collision": "NoCollision",
            "pbr_material_policy": (
                "split each shared glTF image into an sRGB base-color view and "
                "a linear occlusion view; preserve both source material slots"
            ),
            "material_color_contract": material_color,
        },
        "render": {
            "width": 1280,
            "height": 720,
            "frame_count": 75,
            "fps_num": 15,
            "fps_den": 1,
            "streaming_warmup_frames": 120,
        },
        "exposure_and_lighting": fixed_exposure_profile(output_gain=output_gain),
        "camera": deepcopy(visual_plan.get("camera")),
        "actors": deepcopy(actors),
        "frames": deepcopy(frames),
        "source_logic": deepcopy(visual_plan.get("source_logic")),
        "qualification": deepcopy(visual_plan.get("qualification")),
        "claim_boundary": (
            "Timeline v2 owns actor state and Habitat-native AVEngine owns source "
            "centers/audio. UE renders the imported MP3D PBR scene only."
        ),
    }


def luminance_exposure_qa(
    frames_rgb: np.ndarray, profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure highlight saturation after the fixed output transform."""

    array = np.asarray(frames_rgb)
    if array.ndim not in (3, 4) or array.shape[-1] < 3 or array.size == 0:
        raise MP3DExecutionError("frames_rgb must be one or more RGB frames")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise MP3DExecutionError("frames_rgb must contain finite numeric pixels")
    rgb = array[..., :3].astype(np.float64)
    if np.issubdtype(array.dtype, np.integer):
        rgb /= float(np.iinfo(array.dtype).max)
    if np.min(rgb) < 0.0 or np.max(rgb) > 1.0 + 1.0e-9:
        raise MP3DExecutionError("frames_rgb must be normalized or an integer image")
    gain = float(profile.get("fixed_output_gain", math.nan))
    qa = profile.get("qa")
    if not isinstance(qa, Mapping) or not math.isfinite(gain):
        raise MP3DExecutionError("fixed exposure profile is incomplete")
    graded = np.clip(rgb * gain, 0.0, 1.0)
    luminance = (
        0.2126 * graded[..., 0] + 0.7152 * graded[..., 1] + 0.0722 * graded[..., 2]
    )
    threshold = float(qa["luminance_saturation_threshold"])
    nonblack_threshold = float(qa["nonblack_luminance_threshold"])
    maximum = float(qa["maximum_saturated_fraction"])
    saturated = float(np.mean(luminance >= threshold))
    nonblack = float(np.mean(luminance >= nonblack_threshold))
    mean_luminance = float(np.mean(luminance))
    p95_luminance = float(np.percentile(luminance, 95.0))
    minimum_nonblack = float(qa["minimum_nonblack_fraction"])
    minimum_mean = float(qa["minimum_mean_luminance"])
    minimum_p95 = float(qa["minimum_p95_luminance"])
    maximum_mean = float(qa["maximum_mean_luminance"])
    maximum_p95 = float(qa["maximum_p95_luminance"])
    passed = (
        saturated <= maximum
        and nonblack >= minimum_nonblack
        and mean_luminance >= minimum_mean
        and p95_luminance >= minimum_p95
        and mean_luminance <= maximum_mean
        and p95_luminance <= maximum_p95
    )
    return {
        "status": "pass" if passed else "fail",
        "fixed_output_gain": gain,
        "mean_luminance": mean_luminance,
        "p95_luminance": p95_luminance,
        "saturated_fraction": saturated,
        "nonblack_fraction": nonblack,
        "nonblack_luminance_threshold": nonblack_threshold,
        "minimum_nonblack_fraction": minimum_nonblack,
        "minimum_mean_luminance": minimum_mean,
        "minimum_p95_luminance": minimum_p95,
        "maximum_saturated_fraction": maximum,
        "maximum_mean_luminance": maximum_mean,
        "maximum_p95_luminance": maximum_p95,
        "threshold": threshold,
    }


def render_color_fidelity_qa(
    ue_frames_rgb: np.ndarray,
    habitat_frames_rgb: np.ndarray,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare rendered channel chroma with Habitat's authoritative RGB.

    The two renderers need not have the same spatial resolution.  This metric
    intentionally compares aggregate color retention rather than pixelwise
    appearance, because UE adds review shadows and renders different actor
    assets.  It catches the observed sRGB-as-linear failure, while the direct
    23/23 editor readback remains the primary material semantic gate.
    """

    def normalized(value: np.ndarray, *, owner: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 3:
            array = array[np.newaxis, ...]
        if (
            array.ndim != 4
            or array.shape[-1] < 3
            or array.shape[0] <= 0
            or not np.issubdtype(array.dtype, np.number)
            or not np.all(np.isfinite(array))
        ):
            raise MP3DExecutionError(f"{owner} must contain finite RGB frames")
        rgb = array[..., :3].astype(np.float64)
        if np.issubdtype(array.dtype, np.integer):
            rgb /= float(np.iinfo(array.dtype).max)
        if np.min(rgb) < 0.0 or np.max(rgb) > 1.0 + 1.0e-9:
            raise MP3DExecutionError(f"{owner} must be normalized or integer RGB")
        return np.clip(rgb, 0.0, 1.0)

    ue = normalized(ue_frames_rgb, owner="ue_frames_rgb")
    habitat = normalized(habitat_frames_rgb, owner="habitat_frames_rgb")
    if ue.shape[0] != habitat.shape[0]:
        raise MP3DExecutionError("UE and Habitat color QA frame counts differ")
    config = profile.get("color_qa")
    if not isinstance(config, Mapping):
        raise MP3DExecutionError("fixed exposure profile lacks color_qa")

    threshold = float(config["chroma_threshold"])

    def metrics(array: np.ndarray) -> dict[str, float]:
        chroma = np.max(array, axis=-1) - np.min(array, axis=-1)
        return {
            "mean_channel_chroma": float(np.mean(chroma)),
            "chromatic_pixel_fraction": float(np.mean(chroma >= threshold)),
        }

    ue_metrics = metrics(ue)
    habitat_metrics = metrics(habitat)
    reference_chroma = habitat_metrics["mean_channel_chroma"]
    reference_fraction = habitat_metrics["chromatic_pixel_fraction"]
    mean_ratio = (
        ue_metrics["mean_channel_chroma"] / reference_chroma
        if reference_chroma > 0.0
        else math.inf
    )
    fraction_ratio = (
        ue_metrics["chromatic_pixel_fraction"] / reference_fraction
        if reference_fraction > 0.0
        else math.inf
    )
    passed = (
        reference_chroma >= float(config["minimum_reference_mean_chroma"])
        and ue_metrics["mean_channel_chroma"] >= float(config["minimum_ue_mean_chroma"])
        and float(config["minimum_mean_chroma_ratio"])
        <= mean_ratio
        <= float(config["maximum_mean_chroma_ratio"])
        and float(config["minimum_chromatic_fraction_ratio"])
        <= fraction_ratio
        <= float(config["maximum_chromatic_fraction_ratio"])
    )
    return {
        "status": "pass" if passed else "fail",
        "frame_count": int(ue.shape[0]),
        "chroma_threshold": threshold,
        "ue": ue_metrics,
        "habitat_reference": habitat_metrics,
        "mean_chroma_ratio_ue_to_habitat": mean_ratio,
        "chromatic_fraction_ratio_ue_to_habitat": fraction_ratio,
        "thresholds": dict(config),
        "claim_boundary": (
            "aggregate rendered color-retention QA; direct UE Texture2D sRGB and "
            "BaseColorTexture reload readback is the material-semantic authority"
        ),
    }


def _vector3(value: Any, *, owner: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise MP3DExecutionError(f"{owner} must contain three finite numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise MP3DExecutionError(f"{owner}[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise MP3DExecutionError(f"{owner}[{index}] must be a finite number")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


def _close_vector(
    actual: Sequence[float], expected: Sequence[float], *, tolerance: float = 1.0e-6
) -> bool:
    return len(actual) == len(expected) and all(
        abs(float(left) - float(right)) <= tolerance
        for left, right in zip(actual, expected)
    )


def _linear_route(
    start: Sequence[float], end: Sequence[float], frame_count: int
) -> list[list[float]]:
    start_value = _vector3(start, owner="route start")
    end_value = _vector3(end, owner="route end")
    if frame_count < 2:
        raise MP3DExecutionError("route frame_count must be at least two")
    return [
        [
            start_value[axis]
            + (end_value[axis] - start_value[axis]) * frame_index / (frame_count - 1)
            for axis in range(3)
        ]
        for frame_index in range(frame_count)
    ]


def _habitat_to_ue_cm(point: Sequence[float]) -> list[float]:
    x, y, z = _vector3(point, owner="Habitat point")
    return [100.0 * x, 100.0 * z, 100.0 * y]


def _room_registry_record(room_registry: Mapping[str, Any]) -> Mapping[str, Any]:
    records = room_registry.get("records")
    if not isinstance(records, list):
        raise MP3DExecutionError("room registry records are missing")
    matches = [
        item
        for item in records
        if isinstance(item, Mapping) and item.get("room_id") == MP3D_ROOM_ID
    ]
    if len(matches) != 1:
        raise MP3DExecutionError("room registry must contain exactly one MP3D record")
    record = matches[0]
    if (
        record.get("tier") != "visual_research_only"
        or record.get("admission_state") != "not_admitted"
        or record.get("lineage", {}).get("episode_layout_id")
        != "m5_1_human_beagle_route_v1"
    ):
        raise MP3DExecutionError("MP3D room registry claim boundary changed")
    return record


def _validate_m5_1_route(
    route_manifest: Mapping[str, Any], navmesh_gate: Mapping[str, Any]
) -> dict[str, list[list[float]]]:
    if (
        route_manifest.get("schema") != M5_1_ROUTE_SCHEMA
        or route_manifest.get("room_id") != MP3D_ROOM_ID
        or route_manifest.get("route_id") != M5_1_ROUTE_ID
        or route_manifest.get("frame_count") != M5_1_FRAME_COUNT
        or route_manifest.get("frame_rate_hz") != M5_1_FPS
        or route_manifest.get("center_navigation_semantics") != "actor_root_center_only"
        or route_manifest.get("path_generation") != "linear_endpoint_interpolation_v1"
    ):
        raise MP3DExecutionError("M5.1 MP3D route contract changed")
    routes = route_manifest.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != set(M5_1_ACTOR_ORDER):
        raise MP3DExecutionError("M5.1 MP3D actor-route closure changed")

    if (
        navmesh_gate.get("schema") != M5_1_GATE_SCHEMA
        or navmesh_gate.get("status") != "pass"
        or navmesh_gate.get("route_id") != M5_1_ROUTE_ID
        or navmesh_gate.get("frame_count") != M5_1_FRAME_COUNT
        or navmesh_gate.get("frame_rate_hz") != M5_1_FPS
    ):
        raise MP3DExecutionError("M5.1 MP3D navmesh gate is not a retained pass")
    gate_pathfinder = navmesh_gate.get("pathfinder")
    gate_routes = (
        gate_pathfinder.get("routes") if isinstance(gate_pathfinder, Mapping) else None
    )
    if (
        not isinstance(gate_pathfinder, Mapping)
        or gate_pathfinder.get("declared_navmesh_loaded") is not True
        or gate_pathfinder.get("center_navigation_semantics")
        != "actor_root_center_only"
        or not isinstance(gate_routes, Mapping)
    ):
        raise MP3DExecutionError("M5.1 MP3D Pathfinder authority is incomplete")
    gates = navmesh_gate.get("gates")
    if (
        not isinstance(gates, list)
        or len(gates) != 14
        or any(
            not isinstance(item, Mapping) or item.get("status") != "pass"
            for item in gates
        )
    ):
        raise MP3DExecutionError("M5.1 MP3D 14/14 gate closure changed")

    result: dict[str, list[list[float]]] = {}
    for actor_id in M5_1_ACTOR_ORDER:
        route = routes[actor_id]
        gate = gate_routes.get(actor_id)
        if not isinstance(route, Mapping) or not isinstance(gate, Mapping):
            raise MP3DExecutionError(f"MP3D route {actor_id!r} is incomplete")
        start = _vector3(route.get("start_m"), owner=f"{actor_id} start")
        end = _vector3(route.get("end_m"), owner=f"{actor_id} end")
        if (
            gate.get("all_frames_navigable") is not True
            or gate.get("navigable_frame_count") != M5_1_FRAME_COUNT
            or gate.get("frame_count") != M5_1_FRAME_COUNT
            or gate.get("no_sliding_passed_segment_count") != M5_1_FRAME_COUNT - 1
            or not _close_vector(gate.get("start_m", ()), start)
            or not _close_vector(gate.get("end_m", ()), end)
        ):
            raise MP3DExecutionError(
                f"MP3D route {actor_id!r} is not 270/270 Pathfinder-qualified"
            )
        result[actor_id] = _linear_route(start, end, M5_1_FRAME_COUNT)
    return result


def _validate_source_authority(
    source_program: Mapping[str, Any],
    emitter_trajectories: Mapping[str, Any],
) -> dict[str, Any]:
    clip = source_program.get("clip_time_and_audio_contract")
    sources = source_program.get("sources")
    if (
        source_program.get("schema") != M5_1_SOURCE_PROGRAM_SCHEMA
        or source_program.get("applicability")
        != "taxonomy_event_timing_and_audio_program_only"
        or not isinstance(clip, Mapping)
        or clip.get("frame_count") != M5_1_FRAME_COUNT
        or clip.get("fps_num") != M5_1_FPS
        or clip.get("fps_den") != 1
        or clip.get("sample_rate_hz") != M5_1_SAMPLE_RATE_HZ
        or clip.get("sample_count") != M5_1_SAMPLE_COUNT
        or source_program.get("legacy_spatial_trajectory_applicable") is not False
        or not isinstance(sources, list)
    ):
        raise MP3DExecutionError("MP3D source-program compatibility contract changed")
    source_ids = [
        item.get("source_id") if isinstance(item, Mapping) else None for item in sources
    ]
    if source_ids != list(M5_1_SOURCE_TO_ACTOR):
        raise MP3DExecutionError("MP3D source-program source order changed")
    for source in sources:
        assert isinstance(source, Mapping)
        windows = source.get("event_windows")
        if not isinstance(windows, list):
            raise MP3DExecutionError("MP3D source event windows are missing")
        for window in windows:
            if (
                not isinstance(window, Mapping)
                or not 0
                <= int(window.get("start_frame", -1))
                < int(window.get("end_frame_exclusive", -1))
                <= M5_1_FRAME_COUNT
                or not 0
                <= int(window.get("start_sample", -1))
                < int(window.get("end_sample_exclusive", -1))
                <= M5_1_SAMPLE_COUNT
            ):
                raise MP3DExecutionError("MP3D source event window is invalid")

    emitter_sources = emitter_trajectories.get("sources")
    if (
        emitter_trajectories.get("schema") != M5_1_EMITTER_SCHEMA
        or emitter_trajectories.get("source_ids") != list(M5_1_SOURCE_TO_ACTOR)
        or not isinstance(emitter_sources, Mapping)
        or set(emitter_sources) != set(M5_1_SOURCE_TO_ACTOR)
    ):
        raise MP3DExecutionError("MP3D animated-emitter authority changed")
    emitter_summary = []
    for source_id, actor_id in M5_1_SOURCE_TO_ACTOR.items():
        record = emitter_sources[source_id]
        positions = record.get("positions_m") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or record.get("frame_count") != M5_1_FRAME_COUNT
            or record.get("position_authority")
            != "animated_articulated_link_world_transform_readback"
            or not isinstance(positions, list)
            or len(positions) != M5_1_FRAME_COUNT
        ):
            raise MP3DExecutionError(
                f"MP3D emitter trajectory {source_id!r} is incomplete"
            )
        for frame_index, point in enumerate(positions):
            _vector3(point, owner=f"{source_id} emitter frame {frame_index}")
        emitter_summary.append(
            {
                "source_id": source_id,
                "actor_id": actor_id,
                "link_name": record.get("link_name"),
                "frame_count": M5_1_FRAME_COUNT,
                "position_authority": record.get("position_authority"),
            }
        )
    return {
        "source_program": deepcopy(source_program),
        "animated_emitters": emitter_summary,
        "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
    }


def build_m5_1_mp3d_execution_plan(
    *,
    route_manifest: Mapping[str, Any],
    capture_evidence: Mapping[str, Any],
    frame_readback: Sequence[Mapping[str, Any]],
    navmesh_gate: Mapping[str, Any],
    source_program: Mapping[str, Any],
    emitter_trajectories: Mapping[str, Any],
    room_registry: Mapping[str, Any],
    raw_room_qualification: Mapping[str, Any],
    ue_import_manifest: Mapping[str, Any],
    ue_material_color_result: Mapping[str, Any],
    human_ue_manifest: Mapping[str, Any],
    output_gain: float = 1.0,
) -> dict[str, Any]:
    """Compile the retained 270-frame M5.1 MP3D comparison authority.

    This is deliberately *not* ``avengine_authoritative_timeline_v2``.  That
    schema freezes a 75-frame/5-second episode.  The retained MP3D review is
    instead driven frame-for-frame by its M5.1 route and capture readback while
    retaining the same 48 kHz integer tick cadence.
    """

    routes = _validate_m5_1_route(route_manifest, navmesh_gate)
    room = _room_registry_record(room_registry)
    if (
        raw_room_qualification.get("schema")
        != "avengine_m6_room_qualification_report_v1"
        or raw_room_qualification.get("subject", {}).get("room_id") != MP3D_ROOM_ID
        or raw_room_qualification.get("dataset_admission") is not False
        or raw_room_qualification.get("dimensions", {})
        .get("visual_runtime_status", {})
        .get("status")
        != "pass"
        or raw_room_qualification.get("dimensions", {})
        .get("navigation_status", {})
        .get("status")
        != "pass"
    ):
        raise MP3DExecutionError("MP3D room qualification boundary changed")
    capture_schema = capture_evidence.get("schema")
    if capture_schema not in M5_1_CAPTURE_SCHEMAS:
        raise MP3DExecutionError(
            "M5.1 MP3D capture evidence schema is unsupported; expected one of "
            f"{sorted(M5_1_CAPTURE_SCHEMAS)!r}"
        )
    # v1 and installed-prefix v2 share the visual authority fields below.
    # v2 runtime provenance remains opaque to this UE comparison reader: it
    # neither resolves nor reinterprets the Habitat runtime.
    if (
        capture_evidence.get("status") != "pass"
        or capture_evidence.get("frame_count") != M5_1_FRAME_COUNT
        or capture_evidence.get("frame_rate_hz") != M5_1_FPS
        or capture_evidence.get("time_base_hz") != M5_1_TIME_BASE_HZ
        or capture_evidence.get("qualification_claim") is not False
        or capture_evidence.get("research_only") is not True
    ):
        raise MP3DExecutionError("M5.1 MP3D capture authority changed")
    camera = capture_evidence.get("camera")
    if (
        not isinstance(camera, Mapping)
        or camera.get("horizontal_fov_deg") != M5_1_CAMERA_HFOV_DEG
        or camera.get("rotation_xyzw") != [0, 0, 0, 1]
        or not _close_vector(camera.get("position_m", ()), M5_1_CAMERA_HABITAT_M)
    ):
        raise MP3DExecutionError("M5.1 MP3D camera authority changed")

    mesh_paths = _validated_import_manifest(ue_import_manifest)
    material_color = _validated_material_color_result(ue_material_color_result)
    beagle = ue_import_manifest.get("m2_beagle")
    beagle_content = beagle.get("content") if isinstance(beagle, Mapping) else None
    dog_animations = (
        beagle_content.get("animations")
        if isinstance(beagle_content, Mapping)
        else None
    )
    human_content = human_ue_manifest.get("content")
    human_animations = (
        human_content.get("animations") if isinstance(human_content, Mapping) else None
    )
    if (
        not isinstance(beagle_content, Mapping)
        or beagle_content.get("blueprint_class_path") != DOG_BP_CLASS_PATH
        or not isinstance(dog_animations, Mapping)
        or not isinstance(dog_animations.get("Walking"), str)
        or not isinstance(human_content, Mapping)
        or human_content.get("blueprint") != HUMAN_BP_CLASS_PATH.rsplit(".", 1)[0]
        or not isinstance(human_animations, Mapping)
        or not isinstance(human_animations.get("Walking"), str)
    ):
        raise MP3DExecutionError("MP3D human/Beagle UE binding is incomplete")

    source_logic = _validate_source_authority(source_program, emitter_trajectories)
    if (
        isinstance(frame_readback, (str, bytes))
        or len(frame_readback) != M5_1_FRAME_COUNT
    ):
        raise MP3DExecutionError("MP3D capture readback must contain 270 frames")
    actors = {
        "human0": {
            "actor_id": "human0",
            "asset_id": HUMAN_ASSET_ID,
            "source_id": "source0",
            "actor_class": "human",
            "blueprint_class_path": HUMAN_BP_CLASS_PATH,
            "walking_animation": human_animations["Walking"],
            "action_sample_count": 16,
            "animation_clip_start_seconds": 1.0 / 30.0,
            "habitat_local_anatomical_forward_axis": [0.0, 0.0, 1.0],
            "ue_asset_local_anatomical_forward_axis": "+Y",
            "actor_yaw_ue_deg": -180.0,
            "ue_component_frame_delta": component_frame_delta_for_asset(HUMAN_ASSET_ID),
        },
        "dog0": {
            "actor_id": "dog0",
            "asset_id": BEAGLE_ASSET_ID,
            "source_id": "source1",
            "actor_class": "dog",
            "blueprint_class_path": DOG_BP_CLASS_PATH,
            "walking_animation": dog_animations["Walking"],
            "action_sample_count": 25,
            "validated_walk_state_count": 45,
            "animation_clip_start_seconds": 0.0,
            "habitat_local_anatomical_forward_axis": [1.0, 0.0, 0.0],
            "ue_asset_local_anatomical_forward_axis": "+X",
            "actor_yaw_ue_deg": -90.0,
            "ue_component_frame_delta": component_frame_delta_for_asset(
                BEAGLE_ASSET_ID
            ),
        },
    }
    frames: list[dict[str, Any]] = []
    for frame_index, record in enumerate(frame_readback):
        if (
            not isinstance(record, Mapping)
            or record.get("frame_index") != frame_index
            or record.get("pts_ticks") != frame_index * M5_1_TICKS_PER_FRAME
        ):
            raise MP3DExecutionError(
                f"MP3D capture readback clock differs at frame {frame_index}"
            )
        states = []
        for actor_id, record_key in (("human0", "human"), ("dog0", "beagle")):
            actor_record = record.get(record_key)
            if not isinstance(actor_record, Mapping):
                raise MP3DExecutionError(
                    f"MP3D frame {frame_index} lacks {actor_id} readback"
                )
            position = _vector3(
                actor_record.get("actor_root_position_m"),
                owner=f"MP3D frame {frame_index} {actor_id} root",
            )
            if not _close_vector(position, routes[actor_id][frame_index]):
                raise MP3DExecutionError(
                    f"MP3D frame {frame_index} {actor_id} root differs from route"
                )
            sample_index = actor_record.get("action_sample_index")
            sample_count = actors[actor_id]["action_sample_count"]
            expected_sample_index = frame_index % sample_count
            if actor_id == "dog0":
                # The validated Habitat Beagle authority repeats its only
                # continuous 45-state walk block.  The underlying Walking
                # animation contains 25 samples, so the action sample resets
                # when either the animation or the retained state block wraps.
                walk_state_index = (
                    frame_index % actors[actor_id]["validated_walk_state_count"]
                )
                expected_sample_index = walk_state_index % sample_count
            if (
                isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or sample_index != expected_sample_index
            ):
                raise MP3DExecutionError(
                    f"MP3D frame {frame_index} {actor_id} action sample differs"
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
                        actors[actor_id]["animation_clip_start_seconds"]
                        + sample_index / M5_1_FPS
                    ),
                    "ue_animation": actors[actor_id]["walking_animation"],
                    "translation_m": list(position),
                    "translation_ue_cm": _habitat_to_ue_cm(position),
                    "actor_yaw_ue_deg": actors[actor_id]["actor_yaw_ue_deg"],
                }
            )
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": record["pts_ticks"],
                "actor_states": states,
            }
        )

    exposure = fixed_exposure_profile(output_gain=output_gain)
    duration_seconds = M5_1_SAMPLE_COUNT / M5_1_SAMPLE_RATE_HZ
    route_distances = {
        actor_id: sum(math.dist(start, end) for start, end in zip(points, points[1:]))
        for actor_id, points in routes.items()
    }
    return {
        "schema": M5_1_EXECUTION_SCHEMA,
        "backend_role": BACKEND_ROLE,
        "room_id": MP3D_ROOM_ID,
        "route_id": M5_1_ROUTE_ID,
        "authority": {
            "actor_state": "M5.1_route_and_capture_frame_readback",
            "navigation": "Habitat_PathFinder_270_frame_gate",
            "room_identity_and_lineage": "M6_room_registry",
            "source_logic": "M5.1_MP3D_source_program_reuse",
            "source_positions": "M5.1_animated_emitter_link_readback",
            "audio_and_topdown": "Habitat_native_retained_delivery",
            "backend_may_replan": False,
        },
        "clock": {
            "timeline_v2_applicable": False,
            "timeline_v2_non_applicability_reason": (
                "avengine_authoritative_timeline_v2 freezes a 75-frame/5-second "
                "episode; this retained compatibility route is 270 frames/18 seconds"
            ),
            "compatibility_authority_schema": M5_1_ROUTE_SCHEMA,
            "time_base_hz": M5_1_TIME_BASE_HZ,
            "ticks_per_frame": M5_1_TICKS_PER_FRAME,
            "duration_ticks": M5_1_DURATION_TICKS,
            "frame_count": M5_1_FRAME_COUNT,
            "fps_num": M5_1_FPS,
            "fps_den": 1,
            "sample_rate_hz": M5_1_SAMPLE_RATE_HZ,
            "sample_count": M5_1_SAMPLE_COUNT,
        },
        "route_characterization": {
            "retained_compatibility_route": True,
            "duration_seconds": duration_seconds,
            "distance_m_by_actor": route_distances,
            "average_speed_m_s_by_actor": {
                actor_id: distance / duration_seconds
                for actor_id, distance in route_distances.items()
            },
            "normal_speed_requirement_resolved": False,
        },
        "room": {
            "room_id": room["room_id"],
            "revision": room.get("revision"),
            "provider_id": room.get("provider_id"),
            "tier": room.get("tier"),
            "admission_state": room.get("admission_state"),
            "visual_runtime_status": "pass",
            "navigation_status": "pass",
            "acoustic_and_material_qualification": "not_promoted_by_visual_backend",
        },
        "scene": {
            "static_mesh_object_paths": mesh_paths,
            "spawned_scene_mesh_actor_count": EXPECTED_SCENE_MESH_COUNT,
            "collision": "NoCollision",
            "pbr_material_policy": (
                "split each shared glTF image into an sRGB base-color view and "
                "a linear occlusion view; preserve both source material slots"
            ),
            "material_color_contract": material_color,
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
        "exposure_and_lighting": exposure,
        "camera": {
            "habitat_position_m": list(M5_1_CAMERA_HABITAT_M),
            "ue_position_cm": list(M5_1_CAMERA_UE_CM),
            "habitat_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "ue_yaw_deg": M5_1_CAMERA_UE_YAW_DEG,
            "horizontal_fov_deg": M5_1_CAMERA_HFOV_DEG,
        },
        "actors": [actors[actor_id] for actor_id in M5_1_ACTOR_ORDER],
        "frames": frames,
        "source_logic": source_logic,
        "qualification": {
            "visual_runtime_status": "pass",
            "navigation_status": "pass",
            "source_center_semantics": "actor_root_center_only",
            "dataset_admission": False,
            "full_body_clearance_claim": False,
            "ue_collision_authority": False,
        },
        "claim_boundary": (
            "SPEAR renders a comparison visual from the retained M5.1 270-frame "
            "MP3D route. Habitat remains navigation, source, audio and Topdown "
            "authority. Each root travels only 1.1 m in 18 seconds (about "
            "0.061 m/s), so this retained slow route does not resolve the "
            "normal-speed requirement. This does not claim Timeline-v2 "
            "conformance, full-body clearance, Matterport light reconstruction, "
            "acoustic parity or dataset admission."
        ),
    }


__all__ = [
    "EXECUTION_SCHEMA",
    "EXPECTED_SCENE_MATERIAL_COUNT",
    "EXPECTED_SCENE_MESH_COUNT",
    "EXPECTED_SCENE_TEXTURE_COUNT",
    "M5_1_EXECUTION_SCHEMA",
    "M5_1_FRAME_COUNT",
    "IMPORT_SCHEMA",
    "MATERIAL_COLOR_SCHEMA",
    "MP3DExecutionError",
    "MP3D_ROOM_ID",
    "build_m5_1_mp3d_execution_plan",
    "build_mp3d_execution_plan",
    "fixed_exposure_profile",
    "luminance_exposure_qa",
    "render_color_fidelity_qa",
]
