from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IS_STAGING_LAYOUT = (ROOT / "config/rooms/skokloster_castle").is_dir()
ROOM = (
    ROOT / "config/rooms/skokloster_castle"
    if IS_STAGING_LAYOUT
    else ROOT / "examples/acoustics/skokloster_castle"
)
ACOUSTIC_PROFILE = (
    ROOT / "config/acoustics/skokloster_acoustic_profile.json"
    if IS_STAGING_LAYOUT
    else ROOM / "skokloster_acoustic_profile.json"
)
RUNTIME_PROFILE = (
    ROOT / "config/runtime/skokloster_room_runtime_profile.json"
    if IS_STAGING_LAYOUT
    else ROOM / "skokloster_room_runtime_profile.json"
)
EDITOR_PLAN = (
    ROOT / "config/runtime/editor_import_cook_plan.json"
    if IS_STAGING_LAYOUT
    else ROOM / "editor_import_cook_plan.json"
)
ADAPTER = (
    ROOT / "tools/spear_skokloster_room_adapter.py"
    if IS_STAGING_LAYOUT
    else ROOT / "tools/qa/spear_skokloster_room_adapter.py"
)
PREPARER = (
    ROOT / "tools/prepare_skokloster_interchange_glb.py"
    if IS_STAGING_LAYOUT
    else ROOT / "tools/rooms/prepare_skokloster_interchange_glb.py"
)
IMPORTER = (
    ROOT / "tools/editor/import_spear_skokloster_editor.py"
    if IS_STAGING_LAYOUT
    else ROOT / "tools/ue/import_spear_skokloster_editor.py"
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_room_manifest_binds_exact_source_transform_and_cc_by() -> None:
    room = _json(ROOM / "room_manifest.json")
    mapping = _json(ROOM / "material_mapping.json")
    matrix = mapping["source_to_canonical"]["matrix_row_major"]
    assert room["room_id"] == "habitat_test_skokloster_castle"
    assert room["room_kind"] == "habitat_native"
    assert room["scene"]["load_semantic_mesh"] is False
    assert matrix == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert "H=(S.x,S.z,-S.y)" in mapping["source_to_canonical"]["source"]
    assert room["provenance"]["asset_title"] == "The King's Hall"
    assert room["provenance"]["asset_license"].startswith(
        "Creative Commons Attribution 4.0"
    )
    assert room["provenance"]["asset_source_url"].startswith("https://sketchfab.com/")


def test_rlr48_cleanup_profile_is_exact_and_research_only() -> None:
    room = _json(ROOM / "room_manifest.json")
    profile = _json(ACOUSTIC_PROFILE)
    geometry = profile["geometry"]
    assert room["surface_audit"]["source_degenerate_triangle_count"] == 2
    assert room["surface_audit"]["cleanup_policy"] == (
        "remove_only_RLR_incompatible_faces_no_hole_filling"
    )
    assert profile["status"] == "acoustic_research_ready"
    assert geometry["source_triangle_count"] == 999983
    assert geometry["derived_triangle_count"] == 999935
    assert geometry["removed_triangle_count"] == 48
    assert geometry["removed_geometry_qa_face_count"] == 2
    assert geometry["removed_native_rlr_only_face_count"] == 46
    assert geometry["hole_filling"] is False
    assert geometry["vertex_repositioning"] is False
    assert geometry["vertex_removal"] is False
    assert profile["qualification_claim"] is False


def test_enclosure_and_native_rlr_evidence_close_the_cpu_gate() -> None:
    profile = _json(ACOUSTIC_PROFILE)
    enclosure = profile["enclosure"]
    assert enclosure["origin_count"] == 3
    assert enclosure["directions_per_origin"] == 48
    assert enclosure["hit_ray_count"] == 144
    assert enclosure["escaped_ray_count"] == 0
    assert enclosure["probe_clearance_status"] == "pass"
    assert profile["status"] == "acoustic_research_ready"
    assert profile["native_rlr_test"]["status"] == "pass"
    assert profile["native_rlr_test"]["finite"] is True
    assert profile["native_rlr_test"]["nonzero_sample_count"] > 0
    assert profile["qualification_claim"] is False
    assert profile["formal_dataset_count"] == 0


def test_runtime_adapter_reuses_shared_camera_contract() -> None:
    adapter = _module(
        "spear_skokloster_room_adapter",
        ADAPTER,
    )
    object_path = (
        "/Game/MyAssets/Audioset/Scenes/skokloster_castle/Imported/"
        "skokloster_castle.skokloster_castle"
    )
    import_result = {
        "schema": adapter.IMPORT_SCHEMA,
        "status": "pass",
        "mode": "fresh_editor_verify_only",
        "reload_verification": "pass",
        "content_root": "/Game/MyAssets/Audioset/Scenes/skokloster_castle",
        "scene_content": {
            "static_mesh_count": 1,
            "static_meshes": [object_path],
            "class_counts": {"static_mesh_assets": 1},
        },
    }
    record = adapter.build_room_adapter_record(
        import_result, import_result_path="/tmp/fresh_reload_result.json"
    )
    adapter.validate_room_adapter(record)
    components = record["camera_contract"]["components"]
    assert record["static_mesh_object_paths"] == [object_path]
    assert components["normal_metric_depth"] == adapter.DEPTH_COMPONENT
    assert components["source1_target_only_metric_depth"] == adapter.DEPTH_COMPONENT
    assert components["source2_target_only_metric_depth"] == adapter.DEPTH_COMPONENT


def test_interchange_preparation_bakes_x_z_negative_y(tmp_path: Path) -> None:
    preparer = _module(
        "prepare_skokloster_interchange_glb",
        PREPARER,
    )
    positions = np.zeros((800936, 3), dtype="<f4")
    positions[0] = [1.25, 2.5, 3.75]
    positions[-1] = [-4.0, -5.0, -6.0]
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "root", "children": [1]}, {"name": "mesh", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 800936,
                "type": "VEC3",
                "min": positions.min(axis=0).astype(float).tolist(),
                "max": positions.max(axis=0).astype(float).tolist(),
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": positions.nbytes,
                "target": 34962,
            }
        ],
        "buffers": [{"byteLength": positions.nbytes}],
    }
    source = tmp_path / "source.glb"
    output = tmp_path / "prepared.glb"
    preparer._write_glb(source, document, bytearray(positions.tobytes()))
    result = preparer.prepare(source, output)
    prepared_document, prepared_binary = preparer._read_glb(output)
    prepared = np.ndarray(
        shape=(800936, 3), dtype="<f4", buffer=prepared_binary, offset=0
    )
    assert prepared[0].tolist() == [1.25, 3.75, -2.5]
    assert prepared[-1].tolist() == [-4.0, -6.0, 5.0]
    assert (
        prepared_document["extras"]["avengine_coordinate_preparation"][
            "source_to_habitat"
        ]
        == "H=(S.x,S.z,-S.y)"
    )
    assert result["material_and_texture_payload_preserved"] is True


def test_editor_cook_and_packaged_readback_are_isolated_and_pass() -> None:
    plan = _json(EDITOR_PLAN)
    runtime = _json(RUNTIME_PROFILE)
    importer = IMPORTER.read_text(encoding="utf-8")
    assert plan["status"] == (
        "packaged_room_object_readback_pass_visual_sparse_pending"
    )
    assert plan["preflight"]["managed_content_filesystem_target_absent"] is True
    assert plan["preflight"]["snapshot_phase"] == "before_editor_import"
    assert plan["preflight"]["actual_import_authorized_in_this_atom"] is True
    assert plan["preflight"]["cook_authorized_in_this_atom"] is True
    assert plan["unreal_engine_root"] == "/data/UE_5.5"
    assert plan["managed_content_root"] == (
        "/Game/MyAssets/Audioset/Scenes/skokloster_castle"
    )
    assert all(
        "skokloster" in " ".join(step["argv"]).casefold() for step in plan["steps"]
    )
    cook = next(
        step
        for step in plan["steps"]
        if step["step_id"] == "cook_isolated_content_and_map"
    )
    assert cook["authorization"] == "historical_test_v1_authorized_and_rejected"
    assert cook["execution_status"] == "rejected_before_cook_in_build_phase"
    assert "--cook-dirs" not in cook["argv"]
    assert [
        flag
        for flag in ("-build", "-cook", "-stage", "-package", "-archive", "-pak")
        if flag not in cook["argv"]
    ] == []
    assert "AVENGINE_SKOKLOSTER_VERIFY_ONLY" in importer
    assert "static_mesh_geometry_readback" in importer
    assert "delete_directory" not in importer
    assert "AVENGINE_SKOKLOSTER_REPLACE_EXISTING" not in importer
    unreal_stub = types.SimpleNamespace(Name=lambda value: value)
    with mock.patch.dict(sys.modules, {"unreal": unreal_stub}):
        editor_module = _module(
            "import_spear_skokloster_editor_no_clobber_test",
            IMPORTER,
        )
    with mock.patch.dict(os.environ, {"AVENGINE_SKOKLOSTER_REPLACE_EXISTING": "1"}):
        try:
            editor_module._assert_content_root_absent(True, False)
        except RuntimeError as error:
            assert "no-clobber import refused" in str(error)
        else:
            raise AssertionError("existing content root must always be rejected")
        try:
            editor_module._assert_content_root_absent(False, True)
        except RuntimeError as error:
            assert "no-clobber import refused" in str(error)
        else:
            raise AssertionError("existing filesystem root must always be rejected")
    assert runtime["camera"]["shared_actor_for_all_passes"] is True
    assert plan["execution_history"]["editor_import"]["status"] == "pass"
    assert plan["execution_history"]["fresh_editor_reload"]["status"] == "pass"
    assert plan["execution_history"]["uat_test_v1"]["status"] == "rejected"
    assert plan["execution_history"]["uat_development_v3"]["status"] == "pass"
    assert plan["execution_history"]["packaged_object_readback_v1"]["status"] == (
        "rejected"
    )
    assert (
        plan["execution_history"]["packaged_object_readback_v1"]["asset_or_map_failure"]
        is False
    )
    assert plan["execution_history"]["packaged_object_readback_v2"]["status"] == (
        "pass"
    )
    assert runtime["readiness"]["ue_editor_import"] == "pass"
    assert runtime["readiness"]["fresh_editor_reload"] == "pass"
    assert runtime["readiness"]["cook"] == "pass"
    assert runtime["readiness"]["packaged_mesh_readback"] == "pass"
    assert runtime["visual"]["packaged_readback"] == {
        "result": "artifacts/packaged_object_readback_v2/RESULT.json",
        "nullrhi": True,
        "actor_count": 1,
        "mesh_handle_match": True,
        "material_handle_match": True,
        "material_slot_count": 1,
        "bounds_match_editor_cm": True,
        "identity_transform": True,
    }
    assert runtime["formal_dataset_count"] == 0


def _run_without_pytest() -> None:
    tests = [
        test_room_manifest_binds_exact_source_transform_and_cc_by,
        test_rlr48_cleanup_profile_is_exact_and_research_only,
        test_enclosure_and_native_rlr_evidence_close_the_cpu_gate,
        test_runtime_adapter_reuses_shared_camera_contract,
        test_editor_cook_and_packaged_readback_are_isolated_and_pass,
    ]
    for test in tests:
        test()
    with tempfile.TemporaryDirectory(prefix="skokloster_glb_test_") as directory:
        test_interchange_preparation_bakes_x_z_negative_y(Path(directory))
    print(f"SKOKLOSTER_TESTS_OK count={len(tests) + 1}", flush=True)


if __name__ == "__main__":
    _run_without_pytest()
