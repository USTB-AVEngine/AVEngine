from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.assets.animal_review_policy import (
    AnimalReviewPolicyError,
    load_review_policy,
    write_visual_review_decision,
    write_visual_review_manifest,
)

from tools.assets import gate_retopology, gate_rigged_asset


POLICY = Path(__file__).resolve().parents[2] / (
    "examples/assets/generated_animal_review_policy_v1.json"
)


def test_default_policy_is_visual_review_and_keeps_ladder_in_json() -> None:
    policy = load_review_policy(POLICY)
    assert policy["strategy"] == "visual_review"
    assert policy["gate_metrics"] is False
    assert policy["require_closed_cycle"] is False
    assert len(policy["ladder"]) == 5
    assert policy["render"]["walking_frames"] > 0
    assert policy["measurement"]["sample_count"] > 0
    assert set(policy["advisory_metrics"]) >= {
        "face_target",
        "head_survival",
        "shard_share",
        "worst_triangle",
        "fixed_bone_count",
    }


def test_strict_strategy_explicitly_restores_metric_gates() -> None:
    policy = load_review_policy(POLICY, strategy="strict_metrics")
    assert policy["strategy"] == "strict_metrics"
    assert policy["gate_metrics"] is True
    assert policy["require_closed_cycle"] is True


def test_policy_rejects_non_finite_config(tmp_path: Path) -> None:
    value = json.loads(POLICY.read_text(encoding="utf-8"))
    value["common"]["measurement"]["sample_count"] = float("nan")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AnimalReviewPolicyError, match="non-finite"):
        load_review_policy(path)


def test_visual_review_manifest_has_no_hash_binding_and_starts_pending(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    manifest_path = write_visual_review_manifest(
        tmp_path / "visual_review.json",
        review_id="review_001",
        asset_id="animal_001",
        review_path=review_dir,
        policy=load_review_policy(POLICY),
        accepted_rung="plain_25000",
        walking_render="walk",
        turntable_render="turntable",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "needs_visual_review"
    assert manifest["review_path"] == str(review_dir.resolve())
    assert "sha256" not in manifest
    assert manifest["dataset_asset_registration_authorized"] is False
    assert manifest["formal_dataset_registration_authorized"] is False


@pytest.mark.parametrize(
    ("decision", "expected_status", "authorized"),
    (
        ("accept", "accepted_for_dataset_asset", True),
        ("reject", "rejected_for_dataset_asset", False),
    ),
)
def test_human_decision_uses_ordinary_fields_only(
    tmp_path: Path,
    decision: str,
    expected_status: str,
    authorized: bool,
) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    output = write_visual_review_decision(
        tmp_path / f"{decision}.json",
        review_id="review_001",
        asset_id="animal_001",
        review_path=review_dir,
        decision=decision,
        notes="ordinary viewing distance",
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["status"] == expected_status
    assert value["dataset_asset_registration_authorized"] is authorized
    assert value["formal_dataset_registration_authorized"] is False
    assert value["notes"] == "ordinary viewing distance"
    assert set(value) == {
        "schema",
        "review_id",
        "asset_id",
        "review_path",
        "decision",
        "notes",
        "status",
        "qualification_claim",
        "dataset_asset_registration_authorized",
        "formal_dataset_registration_authorized",
    }


def test_metric_findings_are_advisory_by_default_but_strict_when_selected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mesh = tmp_path / "prepared.glb"
    mesh.write_bytes(b"glTF")
    report = {
        "target_faces": 100,
        "output": str(mesh),
        "stages": {"decimated": {"faces": 1, "verts": 3}},
        "band_survival": {"front": 0.1},
    }
    report_path = tmp_path / "prepared.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert gate_retopology.main([str(report_path)]) == 0
    visual = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert visual["status"] == "needs_visual_review"
    assert visual["failures"] == []
    assert visual["advisory_failures"]

    assert gate_retopology.main(
        [str(report_path), "--strategy", "strict_metrics"]
    ) == 1
    strict = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert strict["status"] == "rejected_by_metrics"
    assert strict["failures"]


def test_shard_metric_is_advisory_by_default_but_strict_when_selected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mesh = tmp_path / "animated.glb"
    mesh.write_bytes(b"glTF")
    report = {
        "input": str(mesh),
        "action": "Walking",
        "frames_sampled": [0, 1],
        "mesh": {"valid": True, "finite_coordinates": True, "faces": 3},
        "armature": {"present": True, "bones": 2},
        "skinning": {
            "valid": True,
            "finite_weights": True,
            "skinned_meshes": 1,
            "vertex_groups": 2,
        },
        "animation_numeric_bounds": {
            "max_abs_position": 1,
            "max_abs_scale": 1,
            "limits": {
                "maximum_abs_position": 100,
                "maximum_abs_scale": 100,
            },
            "exploded": False,
        },
        "worst_share_area_shards": 0.5,
        "worst_share_area_over_10x": 0.5,
        "worst_frame_by_shards": 1,
    }
    report_path = tmp_path / "deformation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert gate_rigged_asset.main([str(report_path)]) == 0
    visual = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert visual["status"] == "needs_visual_review"
    assert visual["failures"] == []
    assert gate_rigged_asset.main(
        [str(report_path), "--strategy", "strict_metrics"]
    ) == 1
    strict = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert strict["status"] == "rejected_by_metrics"
    assert strict["failures"]


@pytest.mark.parametrize(
    "missing",
    ("mesh", "armature", "skinning", "animation_numeric_bounds"),
)
def test_structural_rigged_evidence_is_hard_in_visual_review(
    tmp_path: Path, missing: str
) -> None:
    mesh = tmp_path / "animated.glb"
    mesh.write_bytes(b"glTF")
    report = {
        "input": str(mesh),
        "action": "Walking",
        "frames_sampled": [0, 1],
        "mesh": {"valid": True, "finite_coordinates": True, "faces": 3},
        "armature": {"present": True, "bones": 2},
        "skinning": {
            "valid": True,
            "finite_weights": True,
            "skinned_meshes": 1,
            "vertex_groups": 2,
        },
        "animation_numeric_bounds": {
            "max_abs_position": 1,
            "max_abs_scale": 1,
            "limits": {
                "maximum_abs_position": 100,
                "maximum_abs_scale": 100,
            },
            "exploded": False,
        },
        "worst_share_area_shards": 0.01,
    }
    report.pop(missing)
    report_path = tmp_path / f"missing_{missing}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        gate_rigged_asset._structural(*gate_rigged_asset._read_report(report_path))


def test_policy_allows_zero_retries_and_rejects_render_path_escape(tmp_path: Path) -> None:
    value = json.loads(POLICY.read_text(encoding="utf-8"))
    value["common"]["runner"]["rig_retries"] = 0
    valid = tmp_path / "zero_retries.json"
    valid.write_text(json.dumps(value), encoding="utf-8")
    assert load_review_policy(valid)["runner"]["rig_retries"] == 0

    value["common"]["render"]["walking_dir"] = "../old"
    escaped = tmp_path / "escaped.json"
    escaped.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AnimalReviewPolicyError, match="relative child"):
        load_review_policy(escaped)


def test_policy_metadata_does_not_freeze_the_reader(tmp_path: Path) -> None:
    value = json.loads(POLICY.read_text(encoding="utf-8"))
    value["schema"] = "future_policy_metadata"
    value["strategies"]["future_optional_strategy"] = {
        "gate_metrics": False,
        "require_closed_cycle": False,
        "expected_bone_count": None,
        "expected_vertex_group_count": None,
        "advisory_metrics": [],
    }
    path = tmp_path / "extended.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_review_policy(path)["strategy"] == "visual_review"


def test_strict_metrics_reject_boolean_metric_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mesh = tmp_path / "prepared.glb"
    mesh.write_bytes(b"glTF")
    report = {
        "target_faces": 100,
        "output": str(mesh),
        "stages": {"decimated": {"faces": 100, "verts": 3}},
        "band_survival": {"front": True},
    }
    report_path = tmp_path / "prepared.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert gate_retopology.main(
        [str(report_path), "--strategy", "strict_metrics"]
    ) == 1
    verdict = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert verdict["failures"]


def test_legacy_deformation_report_remains_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mesh = tmp_path / "animated.glb"
    mesh.write_bytes(b"glTF")
    report = {
        "input": str(mesh),
        "action": "Walking",
        "faces": 3,
        "frames_sampled": [0, 1],
        "worst_share_area_shards": 0.01,
        "worst_share_area_over_10x": 0.0,
        "worst_frame_by_shards": 1,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert gate_rigged_asset.main(
        [str(path), "--strategy", "strict_metrics"]
    ) == 0
    verdict = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert verdict["structural"]["compatibility"] == "legacy_deformation_report"


def test_strict_metric_thresholds_reject_non_finite_cli_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mesh = tmp_path / "prepared.glb"
    mesh.write_bytes(b"glTF")
    report = {
        "target_faces": 100,
        "output": str(mesh),
        "stages": {"decimated": {"faces": 100, "verts": 3}},
        "band_survival": {"front": 1.0},
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert gate_retopology.main([
        str(path), "--strategy", "strict_metrics", "--min-head-survival", "nan"
    ]) == 1
    assert "HARD_FAIL" in capsys.readouterr().out
