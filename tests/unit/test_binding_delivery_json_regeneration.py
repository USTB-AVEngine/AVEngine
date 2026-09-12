import json
from copy import deepcopy
import pytest
from avengine.qa import binding_delivery as delivery


def payload(tmp_path, monkeypatch, changed=False):
    root = tmp_path
    (root / "catalog").mkdir()
    config = {"seed": "same", "items_per_type": 1, "qa_sampling": {}}
    expected = {"items": [], "evidence": {"published_window_frames": [15, 30]}}
    (root / "catalog/request_config.json").write_text(json.dumps(config))
    (root / "catalog/facts.json").write_text(json.dumps({"sampling": {}}))
    (root / "catalog/questions.json").write_text(json.dumps(expected))
    actual = deepcopy(expected)
    actual["evidence"]["published_window_frames"] = (15, 31 if changed else 30)
    monkeypatch.setattr(delivery, "generate_unified_questions", lambda *a, **k: actual)
    monkeypatch.setattr(delivery, "whole_degree_display", lambda value: value)
    catalog = {"records": [{
        "record_kind": "episode", "episode_id": "episode", "facts_path": "facts.json",
        "questions_path": "questions.json", "public_question_ids": [],
    }]}
    return root, catalog


def test_tuple_and_json_array_are_the_same_persisted_window(tmp_path, monkeypatch):
    root, catalog = payload(tmp_path, monkeypatch)
    result = delivery._validate_qa_regeneration(root, catalog)
    assert result["status"] == "pass"
    assert result["records_regenerated"] == 1


def test_changed_window_value_is_still_refused(tmp_path, monkeypatch):
    root, catalog = payload(tmp_path, monkeypatch, changed=True)
    with pytest.raises(delivery.BindingDeliveryError, match="regeneration differs"):
        delivery._validate_qa_regeneration(root, catalog)
