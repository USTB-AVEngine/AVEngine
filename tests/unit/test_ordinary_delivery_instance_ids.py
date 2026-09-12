import json
from copy import deepcopy
import pytest
from avengine.dataset.production_runner import (
    ProductionRunError, _ordinary_delivery_manifest_entry,
)


def setup(tmp_path):
    (tmp_path / "plan").mkdir()
    actors = [
        {"actor_id": "source1", "entity_instance_id": "human_target", "asset_id": "human"},
        {"actor_id": "source2", "entity_instance_id": "dog_competitor", "asset_id": "dog"},
    ]
    (tmp_path / "plan" / "episode_plan.json").write_text(json.dumps(
        {"visual_plan": {"actors": actors}}))
    return {"source_assignments": [
        {"actor_id": "human_target", "asset_id": "human", "sound_asset_ids": ["voice"]},
        {"actor_id": "dog_competitor", "asset_id": "dog", "sound_asset_ids": ["bark"]},
    ]}


def test_declared_instances_join_runtime_without_changing_allowed_sounds(tmp_path):
    row = setup(tmp_path)
    original = deepcopy(row)
    result = _ordinary_delivery_manifest_entry(row, tmp_path)
    allowed = {a["actor_id"]: a["sound_asset_ids"] for a in result["source_assignments"]}
    assert allowed == {"source1": ["voice"], "source2": ["bark"]}
    assert "bark" not in allowed["source1"]
    assert "voice" not in allowed["source2"]
    assert row == original


def test_mapping_to_another_asset_is_rejected(tmp_path):
    row = setup(tmp_path)
    row["source_assignments"][0]["asset_id"] = "dog"
    with pytest.raises(ProductionRunError, match="asset differs"):
        _ordinary_delivery_manifest_entry(row, tmp_path)


def test_already_runtime_named_rows_keep_their_sound_lists(tmp_path):
    row = setup(tmp_path)
    for i, assignment in enumerate(row["source_assignments"], 1):
        assignment["actor_id"] = f"source{i}"
    result = _ordinary_delivery_manifest_entry(row, tmp_path)
    assert [a["sound_asset_ids"] for a in result["source_assignments"]] == [["voice"], ["bark"]]
