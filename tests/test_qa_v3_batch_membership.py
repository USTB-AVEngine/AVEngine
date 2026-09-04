"""Rejected design directories must never enter capture/audio batches."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/qa"))
from qa_v3_request import batch_point_ids, QARequestError


def _point(root, name):
    path = root / name
    path.mkdir()
    (path / "timeline.json").write_text("{}")


def test_realized_rejection_with_timeline_is_not_a_completed_point(tmp_path):
    _point(tmp_path, "good")
    _point(tmp_path, "bad")
    (tmp_path / "batch_manifest.json").write_text(json.dumps({
        "status": "research_dev", "rejections": [{"point_id": "bad"}]}))
    (tmp_path / "facts.jsonl").write_text(json.dumps({"point_id": "good", "variant": "main"}) + "\n"
        + json.dumps({"point_id": "good", "variant": "gateA"}) + "\n")
    assert batch_point_ids(tmp_path) == ["good"]
    with pytest.raises(QARequestError, match="not completed"):
        batch_point_ids(tmp_path, ["bad"])
    assert (tmp_path / "bad/timeline.json").is_file()


def test_extended_batch_uses_recorded_points_not_directories(tmp_path):
    _point(tmp_path, "good")
    _point(tmp_path, "unfinished")
    (tmp_path / "batch_manifest.json").write_text(json.dumps({
        "records": [{"point_id": "good"}], "rejected": []}))
    assert batch_point_ids(tmp_path) == ["good"]


def test_partial_batch_does_not_fall_back_to_directory_scan(tmp_path):
    _point(tmp_path, "unfinished")
    (tmp_path / "batch_manifest.json").write_text("{}")
    with pytest.raises(QARequestError, match="no completed"):
        batch_point_ids(tmp_path)


def test_standalone_capture_inputs_keep_directory_discovery(tmp_path):
    _point(tmp_path, "raw")
    assert batch_point_ids(tmp_path) == ["raw"]


@pytest.mark.parametrize("record", [
    {"records": [{"point_id": "../outside"}]},
    {"records": [{"point_id": "good"}, {"point_id": "good"}]},
    {"records": [{"point_id": "good"}], "rejected": [{"point_id": "good"}]},
    {"status": "failed", "records": [{"point_id": "good"}]},
])
def test_malformed_or_conflicting_batch_membership_fails(tmp_path, record):
    (tmp_path / "batch_manifest.json").write_text(json.dumps(record))
    with pytest.raises(QARequestError):
        batch_point_ids(tmp_path)
