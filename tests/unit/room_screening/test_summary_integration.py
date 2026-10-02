"""A synthetic end-to-end publication catches integration errors before holdout."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.rooms.room_selection.analysis import summarize, second_sample
from tools.rooms.room_selection.run import load_parameters
from tools.rooms.room_selection import review

pytestmark = pytest.mark.fast_unit


def test_summarize_produces_one_shared_manifest_and_keeps_second_sample(
    tmp_path, monkeypatch
):
    p = load_parameters(
        Path(__file__).parents[3] / "tools/rooms/room_selection/thresholds.yaml"
    )[1]
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(review, "MEDIA_ROOT", media)
    rows = []
    for i, (verdict, status) in enumerate([("use", "pass"), ("skip", "fail")]):
        rows.append(
            dict(
                family="hm3d",
                house=f"hm3d_test_{i:05d}_synthetic",
                region_id=0,
                room_label="R0",
                human=dict(verdict=verdict, source="synthetic"),
                legacy_room={},
                stage1=dict(status=status, reason_codes=[]),
                stage2=dict(status="not_triggered", reason_codes=[]),
                stage4=dict(status="pass", source="synthetic"),
                floors=[],
                metrics=None,
                video_exists=False,
            )
        )
    prior = tmp_path / "previous_82382e2"
    prior.mkdir()
    original = second_sample(rows, p)
    (prior / "second_reviewer_sample.json").write_text(json.dumps(original))
    old_baseline = dict(agreement=0.5, precision=0.5, recall=0.5)
    (prior / "agreement.json").write_text(
        json.dumps(
            dict(
                documented_simple_baseline_all=old_baseline,
                documented_simple_baseline_holdout=old_baseline,
            )
        )
    )
    (tmp_path / "house_analysis_split.json").write_text(
        json.dumps(dict(calibration=[rows[0]["house"]], holdout=[rows[1]["house"]]))
    )
    inputs = dict(
        parameters=p,
        calibration_selection={"calibration_only": True},
        family="hm3d",
        house_count=2,
        latest_gate_counts={"pass": 2},
        historical_baseline_task_R={},
        inventory_source="synthetic",
        created_at_sgt="synthetic",
    )
    summarize(
        SimpleNamespace(
            output=tmp_path, manual_splits=None, clean_houses=None, inventory=None
        ),
        rows,
        inputs,
    )
    manifest = json.loads((tmp_path / "review_manifest.json").read_text())
    assert manifest["schema_version"] == "room_screening_review_manifest_v1"
    assert len(manifest["items"]) == 2
    assert all(item["second_review"] for item in manifest["items"])
    assert not (tmp_path / "review.html").exists()
    assert (
        json.loads((tmp_path / "second_reviewer_sample.json").read_text()) == original
    )
    metrics = json.loads((tmp_path / "agreement.json").read_text())
    assert metrics["all"]["confusion_matrix"] == dict(TP=1, FP=0, FN=0, TN=1)
    assert metrics["documented_simple_baseline_holdout"] == old_baseline
