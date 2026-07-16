from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from avengine.m1.contracts import (
    ContractError,
    load_and_validate_inputs,
    validate_capture_request,
    validate_room_manifest,
    validate_scene_asset_graph,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_ROOM_PATH = (
    REPOSITORY_ROOT / "examples/m1/rooms/blender_custom/room_manifest.json"
)
CUSTOM_REQUEST_PATH = REPOSITORY_ROOT / "examples/m1/requests/blender_custom.json"
NATIVE_ROOM_PATH = (
    REPOSITORY_ROOT / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"
)
NATIVE_REQUEST_PATH = REPOSITORY_ROOT / "examples/m1/requests/habitat_mp3d_example.json"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _native_graph_fixture(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    dataset_root = runtime_root / "data/scene_datasets/mp3d_example"
    scene_root = dataset_root / "17DRP5sb8fy"
    scene_root.mkdir(parents=True)
    for name, payload in (
        ("17DRP5sb8fy.glb", b"glb"),
        ("17DRP5sb8fy.navmesh", b"nav"),
        ("17DRP5sb8fy_semantic.ply", b"ply"),
        ("17DRP5sb8fy.house", b"house"),
    ):
        (scene_root / name).write_bytes(payload)
    dataset = {
        "stages": {
            "paths": {".glb": ["*/*.glb"]},
            "default_attributes": {
                "nav_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.navmesh",
                "semantic_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%_semantic.ply",
                "semantic_descriptor_filename": (
                    "%%CONFIG_NAME_AS_ASSET_FILENAME%%.house"
                ),
            },
        }
    }
    dataset_path = dataset_root / "mp3d.scene_dataset_config.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    room_path = tmp_path / "native_room.json"
    request_path = tmp_path / "native_request.json"
    room_path.write_text(json.dumps(_load(NATIVE_ROOM_PATH)), encoding="utf-8")
    request_path.write_text(json.dumps(_load(NATIVE_REQUEST_PATH)), encoding="utf-8")
    return (
        load_and_validate_inputs(room_path, request_path),
        runtime_root,
        dataset_path,
        scene_root,
    )


@pytest.mark.parametrize(
    ("room_path", "request_path"),
    [
        (CUSTOM_ROOM_PATH, CUSTOM_REQUEST_PATH),
        (NATIVE_ROOM_PATH, NATIVE_REQUEST_PATH),
    ],
)
def test_checked_in_m1_room_and_request_examples_are_valid(
    room_path: Path, request_path: Path
) -> None:
    validated = load_and_validate_inputs(room_path, request_path)

    assert validated.room["room_id"] == validated.request["room_id"]
    assert validated.request["primary_camera_rig"]["view_id"] == "view0"


def test_request_rejects_multiple_primary_camera_rigs() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    first_rig = request["primary_camera_rig"]
    second_rig = copy.deepcopy(first_rig)
    second_rig["rig_id"] = "camera_rig_1"
    second_rig["view_id"] = "view1"
    request["primary_camera_rig"] = [first_rig, second_rig]

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert "primary_camera_rig must be one object" in errors


def test_request_rejects_noncanonical_view_id() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["primary_camera_rig"]["view_id"] = "view9"

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert "primary_camera_rig.view_id must be 'view0'" in errors


def test_request_rejects_secondary_camera_rig_field() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["secondary_camera_rig"] = copy.deepcopy(request["primary_camera_rig"])

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert any(
        "capture request has unsupported fields" in error
        and "secondary_camera_rig" in error
        for error in errors
    )


def test_topdown_qa_view_cannot_become_a_second_formal_view() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["qa_views"][0]["view_id"] = "view1"

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert any("QA-only" in error and "formal view_id" in error for error in errors)


@pytest.mark.parametrize("field", ["position", "orientation", "resolution", "hfov"])
def test_modality_cannot_override_shared_calibration(field: str) -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["primary_camera_rig"]["modalities"][0][field] = [0.0, 0.0, 0.0]

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert any(
        "modality-specific calibration fields" in error and field in error
        for error in errors
    )


def test_request_rejects_non_pinhole_360_degree_hfov() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["primary_camera_rig"]["shared_calibration"]["hfov_degrees"] = 360.0

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert "shared_calibration.hfov_degrees must be smaller than 180" in errors


def test_request_requires_at_least_two_sources() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["sources"] = []

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert "M1 capture requires at least two independently named sources" in errors


def test_request_requires_pairwise_distinct_source_transforms() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["sources"][1]["world_from_source"] = copy.deepcopy(
        request["sources"][0]["world_from_source"]
    )

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert "M1 source world transforms must be pairwise distinct" in errors


def test_request_rejects_negative_topdown_scale() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["qa_views"][0]["meters_per_pixel"] = -0.01

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert "qa_views[0].meters_per_pixel must be positive" in errors


@pytest.mark.parametrize(
    "room_kind", ["blender_custom", "legacy_ue_real_surface_export"]
)
def test_authored_and_legacy_rooms_reject_debug_aabb_proxy(room_kind: str) -> None:
    room = _load(CUSTOM_ROOM_PATH)
    room["room_kind"] = room_kind
    room["geometry_representation"] = "debug_aabb_proxy"

    errors = validate_room_manifest(room)

    assert f"{room_kind} cannot use a debug AABB proxy" in errors


def test_surface_audit_cannot_claim_aabb_for_real_surface_mesh() -> None:
    room = _load(CUSTOM_ROOM_PATH)
    room["surface_audit"]["aabb_proxy"] = True

    errors = validate_room_manifest(room)

    assert "surface_audit.aabb_proxy must be false" in errors


def test_authored_room_requires_surface_audit() -> None:
    room = _load(CUSTOM_ROOM_PATH)
    del room["surface_audit"]

    errors = validate_room_manifest(room)

    assert "blender_custom requires surface_audit evidence" in errors


@pytest.mark.parametrize(
    ("missing_field", "expected_error"),
    [
        ("navigation", "navigation must be an object"),
        ("openings", "openings must be an array"),
        ("provenance", "provenance must be an object"),
    ],
)
def test_room_rejects_missing_required_package_sections(
    missing_field: str, expected_error: str
) -> None:
    room = _load(CUSTOM_ROOM_PATH)
    del room[missing_field]

    errors = validate_room_manifest(room)

    assert expected_error in errors


def test_request_requires_exactly_rgb_depth_and_semantic_modalities() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["primary_camera_rig"]["modalities"] = request["primary_camera_rig"][
        "modalities"
    ][:2]

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert any("modalities must be exactly" in error for error in errors)


def test_listener_must_share_visual_sensor_pose() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["listener"]["rig_from_listener"]["translation_m"][0] = 0.01

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert any(
        "listener must be co-located and co-oriented" in error for error in errors
    )


def test_request_room_id_must_match_room_manifest() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["room_id"] = "a_different_room"

    errors = validate_capture_request(request, room_id="blender_custom_two_zone_v1")

    assert any("does not match" in error for error in errors)


def test_load_and_validate_inputs_reports_room_and_request_context(
    tmp_path: Path,
) -> None:
    room = _load(CUSTOM_ROOM_PATH)
    request = _load(CUSTOM_REQUEST_PATH)
    room["geometry_representation"] = "debug_aabb_proxy"
    request["room_id"] = "a_different_room"
    room_path = tmp_path / "room.json"
    request_path = tmp_path / "request.json"
    room_path.write_text(json.dumps(room), encoding="utf-8")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ContractError) as captured:
        load_and_validate_inputs(room_path, request_path)

    assert any(error.startswith("room: ") for error in captured.value.errors)
    assert any(error.startswith("request: ") for error in captured.value.errors)


def test_request_resolution_rejects_booleans_as_integers() -> None:
    request = _load(CUSTOM_REQUEST_PATH)
    request["primary_camera_rig"]["shared_calibration"]["resolution_hw"] = [
        True,
        True,
    ]

    errors = validate_capture_request(request, room_id=request["room_id"])

    assert any("resolution_hw" in error for error in errors)


def test_room_semantic_label_map_must_be_an_object() -> None:
    room = _load(CUSTOM_ROOM_PATH)
    room["semantics"]["id_to_label"] = []

    errors = validate_room_manifest(room)

    assert "semantics.id_to_label must be an object" in errors


@pytest.mark.parametrize(
    ("scene_field", "expected_error"),
    [
        (
            "dataset_config_path",
            "scene.dataset_config_path must equal the scene_dataset_config asset path",
        ),
        ("navmesh_path", "scene.navmesh_path must equal the navmesh asset path"),
    ],
)
def test_room_scene_support_paths_must_match_hashed_asset_roles(
    scene_field: str, expected_error: str
) -> None:
    room = _load(CUSTOM_ROOM_PATH)
    room["scene"][scene_field] = f"visual/alternate-{scene_field}"

    errors = validate_room_manifest(room)

    assert expected_error in errors


def test_path_scene_id_must_match_hashed_render_surface_role() -> None:
    room = _load(NATIVE_ROOM_PATH)
    room["scene"]["scene_id"] = (
        "${AVENGINE_HABITAT_RUNTIME_ROOT}/data/scene_datasets/alternate.glb"
    )

    errors = validate_room_manifest(room)

    assert "path scene.scene_id must equal the render_surface_mesh asset path" in errors


def test_handle_scene_graph_rejects_alternate_stage_render_asset(
    tmp_path: Path,
) -> None:
    room_root = tmp_path / "blender_custom"
    shutil.copytree(CUSTOM_ROOM_PATH.parent, room_root)
    stage_path = room_root / "visual/stages/m1_custom_room.stage_config.json"
    alternate_path = room_root / "visual/stages/alternate.glb"
    shutil.copyfile(room_root / "visual/stages/m1_custom_room.glb", alternate_path)
    stage = _load(stage_path)
    stage["render_asset"] = alternate_path.name
    stage["collision_asset"] = alternate_path.name
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    request_path = tmp_path / "request.json"
    shutil.copyfile(CUSTOM_REQUEST_PATH, request_path)
    inputs = load_and_validate_inputs(room_root / "room_manifest.json", request_path)

    errors = validate_scene_asset_graph(inputs, tmp_path / "runtime")

    assert "stage render_asset does not resolve to render_surface_mesh" in errors
    assert "stage collision_asset does not resolve to render_surface_mesh" in errors


@pytest.mark.parametrize(
    ("default_key", "alternate_name", "expected_error"),
    [
        (
            "nav_asset",
            "alternate.navmesh",
            "path stage nav_asset does not resolve to the declared navmesh",
        ),
        (
            "semantic_asset",
            "alternate_semantic.ply",
            "path stage semantic_asset does not resolve to semantic_surface_mesh",
        ),
        (
            "semantic_descriptor_filename",
            "alternate.house",
            "path stage semantic descriptor does not resolve to semantic_descriptor",
        ),
    ],
)
def test_path_scene_graph_rejects_same_bytes_from_alternate_support_asset(
    tmp_path: Path,
    default_key: str,
    alternate_name: str,
    expected_error: str,
) -> None:
    inputs, runtime_root, dataset_path, scene_root = _native_graph_fixture(tmp_path)
    dataset = _load(dataset_path)
    original_name = {
        "nav_asset": "17DRP5sb8fy.navmesh",
        "semantic_asset": "17DRP5sb8fy_semantic.ply",
        "semantic_descriptor_filename": "17DRP5sb8fy.house",
    }[default_key]
    shutil.copyfile(scene_root / original_name, scene_root / alternate_name)
    dataset["stages"]["default_attributes"][default_key] = alternate_name
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    errors = validate_scene_asset_graph(inputs, runtime_root)

    assert expected_error in errors


@pytest.mark.parametrize(
    ("section", "relative_source"),
    [
        ("stages", "stages/m1_custom_room.stage_config.json"),
        ("objects", "objects/source_marker_0.object_config.json"),
        ("light_setups", "lighting/m1_custom_room.lighting_config.json"),
        ("scene_instances", "scenes/m1_custom_room.scene_instance.json"),
    ],
)
def test_handle_scene_graph_rejects_duplicate_named_config_search_result(
    tmp_path: Path, section: str, relative_source: str
) -> None:
    room_root = tmp_path / "blender_custom"
    shutil.copytree(CUSTOM_ROOM_PATH.parent, room_root)
    visual = room_root / "visual"
    source = visual / relative_source
    duplicate_dir = visual / f"duplicate_{section}"
    duplicate_dir.mkdir()
    shutil.copyfile(source, duplicate_dir / source.name)
    dataset_path = visual / "m1_custom_room.scene_dataset_config.json"
    dataset = _load(dataset_path)
    dataset[section]["paths"][".json"].append(duplicate_dir.name)
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    request_path = tmp_path / "request.json"
    shutil.copyfile(CUSTOM_REQUEST_PATH, request_path)
    inputs = load_and_validate_inputs(room_root / "room_manifest.json", request_path)

    errors = validate_scene_asset_graph(inputs, tmp_path / "runtime")

    assert any(
        f"dataset {section} search paths must resolve exactly one" in error
        for error in errors
    )


def test_handle_scene_graph_rejects_substring_ambiguous_scene_instance(
    tmp_path: Path,
) -> None:
    room_root = tmp_path / "blender_custom"
    shutil.copytree(CUSTOM_ROOM_PATH.parent, room_root)
    scenes_dir = room_root / "visual/scenes"
    source = scenes_dir / "m1_custom_room.scene_instance.json"
    shutil.copyfile(
        source,
        scenes_dir / "archived_m1_custom_room.scene_instance.json",
    )
    request_path = tmp_path / "request.json"
    shutil.copyfile(CUSTOM_REQUEST_PATH, request_path)
    inputs = load_and_validate_inputs(room_root / "room_manifest.json", request_path)

    errors = validate_scene_asset_graph(inputs, tmp_path / "runtime")

    assert (
        "dataset scene_instances handle query 'm1_custom_room' must have exactly "
        "one declared candidate"
    ) in errors


def test_handle_scene_graph_rejects_object_config_alternate_same_bytes_mesh(
    tmp_path: Path,
) -> None:
    room_root = tmp_path / "blender_custom"
    shutil.copytree(CUSTOM_ROOM_PATH.parent, room_root)
    object_dir = room_root / "visual/objects"
    alternate = object_dir / "alternate_source_marker_0.glb"
    shutil.copyfile(object_dir / "source_marker_0.glb", alternate)
    config_path = object_dir / "source_marker_0.object_config.json"
    config = _load(config_path)
    config["render_asset"] = alternate.name
    config["collision_asset"] = alternate.name
    config_path.write_text(json.dumps(config), encoding="utf-8")
    request_path = tmp_path / "request.json"
    shutil.copyfile(CUSTOM_REQUEST_PATH, request_path)
    inputs = load_and_validate_inputs(room_root / "room_manifest.json", request_path)

    errors = validate_scene_asset_graph(inputs, tmp_path / "runtime")

    assert "source0 object render_asset does not resolve to its declared mesh" in errors
    assert (
        "source0 object collision_asset does not resolve to its declared mesh" in errors
    )


def test_handle_scene_graph_rejects_alternate_default_lighting(
    tmp_path: Path,
) -> None:
    room_root = tmp_path / "blender_custom"
    shutil.copytree(CUSTOM_ROOM_PATH.parent, room_root)
    lighting_dir = room_root / "visual/lighting"
    shutil.copyfile(
        lighting_dir / "m1_custom_room.lighting_config.json",
        lighting_dir / "alternate.lighting_config.json",
    )
    instance_path = room_root / "visual/scenes/m1_custom_room.scene_instance.json"
    instance = _load(instance_path)
    instance["default_lighting"] = "alternate"
    instance_path.write_text(json.dumps(instance), encoding="utf-8")
    request_path = tmp_path / "request.json"
    shutil.copyfile(CUSTOM_REQUEST_PATH, request_path)
    inputs = load_and_validate_inputs(room_root / "room_manifest.json", request_path)

    errors = validate_scene_asset_graph(inputs, tmp_path / "runtime")

    assert "scene_instance default_lighting does not select lighting_config" in errors
