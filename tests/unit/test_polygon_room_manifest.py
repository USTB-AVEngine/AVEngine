import json

import pytest

from tools.rooms.build_polygon_room_manifest import build_manifest, read_records

pytestmark = pytest.mark.fast_unit
POLYGON = {"type": "Polygon", "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]],
    [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]]}


def inputs(tmp_path):
    catalog = tmp_path / "original.json"
    original = {"schema": "avengine_qa_room_package_catalog_v1", "rooms": [
        {"room_id": "old", "house": "h", "floor_polygon_xz_m": POLYGON,
         "room_package": {"room_id": "old", "scene": {"path": "whole.glb"},
                          "acoustics": {"path": "whole_house_package"}}}]}
    catalog.write_text(json.dumps(original))
    splits = tmp_path / "splits"
    splits.mkdir()
    split = {"id": "new", "house": "h", "floor_y_m": 0.,
             "floor_polygon_xz_m": POLYGON, "decision": "retain", "new_room": True}
    (splits / "new.json").write_text(json.dumps(split))
    (splits / "discard.json").write_text(json.dumps({**split, "id": "discard", "decision": "discard"}))
    return catalog, splits, original


def test_new_directory_preserves_inputs_original_geometry_and_resource_templates(tmp_path):
    catalog, splits, original = inputs(tmp_path)
    before = catalog.read_bytes()
    out = tmp_path / "out"
    summary = build_manifest(original_rooms=[catalog], split_rooms=splits, output_dir=out,
                             template_map={"h": "old"})
    assert catalog.read_bytes() == before
    result = json.loads((out / "manifest.json").read_text())
    assert result["rooms"][0] == original["rooms"][0]
    new = result["rooms"][1]
    assert new["room_package"]["scene"] == original["rooms"][0]["room_package"]["scene"]
    assert new["room_package"]["acoustics"] == original["rooms"][0]["room_package"]["acoustics"]
    assert new["floor_polygon_xz_m"] == POLYGON  # retain source holes in the record
    assert len(read_records(out / "rooms")[0]) == 2
    assert summary["added_retained_rooms"] == 1
    assert summary["registered_catalog_templates_supplied"]
    from avengine.rooms.room_providers import load_room_catalog
    assert load_room_catalog(out / "manifest.json")["rooms"] == result["rooms"]


def test_refuses_overwrite_and_requires_explicit_template_for_all_rooms(tmp_path):
    catalog, splits, _ = inputs(tmp_path)
    out = tmp_path / "out"
    build_manifest(split_rooms=splits, output_dir=out)
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_manifest(split_rooms=splits, output_dir=out)
    assert (out / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="no explicit registered template"):
        build_manifest(original_rooms=[catalog], split_rooms=splits,
                       output_dir=tmp_path / "missing", template_map={"other": "old"})
    assert not (tmp_path / "missing").exists()
