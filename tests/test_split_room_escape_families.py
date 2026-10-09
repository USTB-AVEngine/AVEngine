"""Family geometry and house-level parallel result contracts."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.acoustics.check_split_room_escape import partition_houses, merge_worker_results, write_csv
from tools.acoustics.prepare_split_house_inputs import prepare
from tools.acoustics.split_house_inputs import (
    package_matrix, placement_mesh, require_houses, validate_house_input,
)

pytestmark = pytest.mark.fast_unit


def test_parallel_assignments_keep_every_house_whole_and_are_deterministic():
    houses = {"mp3d_b": ["r1", "r2"], "hm3d_a": ["r3"], "kujiale_c": ["r4"]}
    assert partition_houses(houses, 1) == [["hm3d_a", "kujiale_c", "mp3d_b"]]
    assert partition_houses(houses, 4) == [["hm3d_a"], ["kujiale_c"], ["mp3d_b"]]
    assert partition_houses(houses, 2) == [["hm3d_a", "mp3d_b"], ["kujiale_c"]]
    assert partition_houses({}, 4) == []
    with pytest.raises(ValueError, match="positive"):
        partition_houses(houses, 0)


def test_parallel_completion_order_does_not_change_csv_row_order(tmp_path):
    rows = []
    for room, value in [("b", .125), ("a", .05)]:
        rows.append(dict(room=room, house="mp3d_h", family="mp3d", source_region="R0",
                         max_escape_fraction=value, band="只训练", sources_inside_filled_exterior=True,
                         all_ray_origins_inside_filled_exterior=True, placement_seconds=0., ray_seconds=1.,
                         wall_seconds=1., house_setup_seconds_per_room=2., wall_seconds_including_amortized_setup=3.,
                         input_room_json="input.json"))
    chunks = [{"results": [row], "setups": [{"house": row["room"]}]} for row in rows]
    forward, setups = merge_worker_results(chunks)
    backward, _ = merge_worker_results(chunks[::-1])
    assert forward == backward and [s["house"] for s in setups] == ["a", "b"]
    write_csv(tmp_path, forward)
    first = (tmp_path / "room_results.csv").read_bytes()
    write_csv(tmp_path, backward)
    assert first == (tmp_path / "room_results.csv").read_bytes()
    assert [r["room"] for r in csv.DictReader((tmp_path / "room_results.csv").open())] == ["a", "b"]
    with pytest.raises(ValueError, match="duplicate"):
        merge_worker_results(chunks + chunks[:1])


def test_missing_house_error_lists_every_id_and_how_to_regenerate():
    with pytest.raises(ValueError) as error:
        require_houses({"hm3d_a": {}}, ["mp3d_new", "hm3d_a", "kujiale_new"])
    assert "mp3d_new" in str(error.value) and "kujiale_new" in str(error.value)
    assert "--house-inputs" in str(error.value) and "prepare_split_house_inputs.py" in str(error.value)


def kujiale_fixture(tmp_path):
    coordinate = dict(up_axis="+Y", forward_axis="-Z", handedness="right", linear_unit="meter", quaternion_order="xyzw")
    transform = dict(matrix_row_major=[1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1], reviewed=True, source="fixture USD package matrix")
    vertices = np.array([[1,2,3], [4,5,6], [7,9,8]], dtype="<f4")
    triangles = np.array([[0,1,2]], dtype="<u4")
    arrays = {}
    for name, array in [("vertices", vertices), ("triangles", triangles)]:
        path = tmp_path / (name + ".npy")
        np.save(path, array)
        arrays[name] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "byte_size": path.stat().st_size}
    manifest = dict(source_room={"room_id": "kujiale_test"}, coordinate_system=coordinate, unit_scale_to_m=1,
                    geometry=dict(source_to_canonical=transform, transform_policy="baked_to_canonical_world"), arrays=arrays)
    ref = tmp_path / "original_manifest.json"
    ref.write_text(json.dumps(manifest))
    package = tmp_path / "package_rlr"
    package.mkdir()
    (package / "manifest.json").write_text(json.dumps(manifest))
    for name in arrays:
        (package / (name + ".npy")).write_bytes((tmp_path / (name + ".npy")).read_bytes())
    geo = tmp_path / "geometry.json"
    geo.write_text(json.dumps(dict(room_id="kujiale_test", coordinate_system=coordinate)))
    for name in ("navmesh", "annotations"):
        (tmp_path / name).write_text("fixture")
    h = dict(house="kujiale_test", family="kujiale", acoustic_package_manifest=str(package / "manifest.json"),
             placement_reference_package_manifest=str(ref), placement_geometry_manifest=str(geo),
             placement_mesh_vertices=str(tmp_path / "vertices.npy"), placement_mesh_triangles=str(tmp_path / "triangles.npy"),
             navmesh_source=str(tmp_path / "navmesh"), annotation_source=str(tmp_path / "annotations"),
             semantic_source=str(tmp_path / "archived_render.usdc"))
    return h, manifest, vertices


def test_kujiale_uses_original_canonical_cache_without_a_second_axis_transform(tmp_path):
    h, manifest, vertices = kujiale_fixture(tmp_path)
    mesh, receipt = placement_mesh(h)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    assert receipt["coordinate_transform"] == manifest["geometry"]["source_to_canonical"]
    assert receipt["canonical_arrays_equal_reference"] and not receipt["transform_applied_this_run"]
    check = validate_house_input(h)
    assert all(x["exists"] for x in check["required_files"])
    assert check["historical_provenance_files"][0]["exists"] is False
    assert check["historical_provenance_files"][0]["required_for_cpu_measurement"] is False


def test_kujiale_requires_original_package_and_reviewed_coordinate_matrix(tmp_path):
    h, manifest, _ = kujiale_fixture(tmp_path)
    manifest["geometry"]["source_to_canonical"]["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        package_matrix(manifest)
    h["acoustic_package_manifest"] = h["placement_reference_package_manifest"]
    with pytest.raises(ValueError, match="package_rlr"):
        validate_house_input(h)


def test_kujiale_rejects_changed_original_placement_geometry(tmp_path):
    h, _, vertices = kujiale_fixture(tmp_path)
    np.save(h["placement_mesh_vertices"], vertices + 1)
    with pytest.raises(ValueError, match="differs from the original package"):
        placement_mesh(h)


def test_hm3d_records_copied_without_altering_any_fields(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    for name in ("s.glb", "semantic", "annotation", "navmesh"):
        (scene / name).write_text("fixture")
    h = dict(house="hm3d_original", family="hm3d", scan_id="s", scene_directory=str(scene),
             semantic_source=str(scene / "semantic"), annotation_source=str(scene / "annotation"),
             navmesh_source=str(scene / "navmesh"), acoustic_package_manifest=None, user_field=["preserve", 163])
    original = dict(houses={h["house"]: h}, source_plan="unchanged source")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(original))
    mp = tmp_path / "mp.csv"
    mp.write_text("house,family\n")
    k = tmp_path / "kujiale"
    k.mkdir()
    (k / "inventory.json").write_text("[]")
    args = SimpleNamespace(hm3d_inputs=inputs, mp3d_rooms=mp, mp3d_data_root=tmp_path,
                           measurement_root=tmp_path, kujiale_root=k, output=tmp_path / "all.json",
                           validation_output=tmp_path / "validation.json")
    prepare(args)
    result = json.loads(args.output.read_text())
    assert result["houses"] == original["houses"] and result["source_plan"] == original["source_plan"]
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare(args)


def test_historical_compatible_package_alias_keeps_geometry_and_material_identity(tmp_path):
    from tools.acoustics.prepare_split_house_inputs import rlr_package_alias
    h, _, _ = kujiale_fixture(tmp_path)
    original = Path(h["placement_reference_package_manifest"])
    before = original.read_bytes()
    alias, receipt = rlr_package_alias(original, tmp_path / "aliases", h["house"])
    assert alias.read_bytes() == before and original.read_bytes() == before
    record = json.loads(receipt.read_text())
    assert record["zero_area_triangles"] == 0 and record["native_incompatible_triangles"] == 0
    assert alias.parent.is_symlink()


def test_historical_alias_cannot_hide_zero_area_geometry(tmp_path):
    from tools.acoustics.prepare_split_house_inputs import rlr_package_alias
    h, _, _ = kujiale_fixture(tmp_path)
    np.save(h["placement_mesh_triangles"], np.array([[0,0,1]], dtype="<u4"))
    with pytest.raises(ValueError, match="zero-area"):
        rlr_package_alias(Path(h["placement_reference_package_manifest"]), tmp_path / "aliases", h["house"])
    assert not (tmp_path / "aliases").exists()
