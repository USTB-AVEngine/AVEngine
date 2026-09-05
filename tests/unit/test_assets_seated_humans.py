from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from avengine.assets.seated_humans import (
    SeatedHumanSpecError,
    avengine_emitter_offset_m,
    seat_root_offset_blender_m,
    build_ue_import_request,
    load_seated_human_batch,
)


SPEC = Path("examples/assets/seated_human_batch_v1.json")


def test_loads_four_assets_and_converts_emitter_frame() -> None:
    values = load_seated_human_batch(SPEC, require_sources=False)
    assert len(values) == 4
    assert values[0].asset_id == "seated_rocketbox_male_adult_01_blue_v1"
    assert avengine_emitter_offset_m(values[0]) == (-0.000163, 1.391552, -0.143839)
    assert avengine_emitter_offset_m(values[1]) == (0.031161, 1.314836, -0.218617)
    assert seat_root_offset_blender_m(values[0]) == pytest.approx((0.0, 0.18, -0.01))
    assert seat_root_offset_blender_m(values[2]) == pytest.approx((0.0, -0.18, -0.01))


def test_batch_rejects_duplicate_asset_ids(tmp_path: Path) -> None:
    raw = json.loads(SPEC.read_text(encoding="utf-8"))
    raw["assets"][1]["asset_id"] = raw["assets"][0]["asset_id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SeatedHumanSpecError, match="duplicate seated asset ID"):
        load_seated_human_batch(path, require_sources=False)


def test_build_request_binds_seat_reference_and_action(tmp_path: Path) -> None:
    values = load_seated_human_batch(SPEC, require_sources=False)
    output_root = tmp_path / "generated"
    for value in values:
        path = output_root / value.asset_id / "asset"
        path.mkdir(parents=True)
        (path / f"{value.asset_id}.glb").write_bytes(b"glTF")
    request = build_ue_import_request(values, output_root=output_root)
    assert request["content_root"] == "/Game/AVEngine/SeatedHumans"
    assert len(request["assets"]) == 4
    assert request["assets"][0]["animation_name"] == "Seated_Idle"
    assert request["assets"][0]["seat_reference"]["seat_top_m"] == 0.53
