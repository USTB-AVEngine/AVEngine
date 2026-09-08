"""Later audio attempts must drive delivery counts and coverage without rewriting history."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "tools/dataset/merge_qa_batch_attempts.py"
    spec = importlib.util.spec_from_file_location("later_attempt_selection_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(attempt, **values):
    return {"episode_id": "episode", "attempt": attempt, "attempt_root": f"/missing/{attempt}",
            "status": "delivered", "room_id": "room", "asset_ids": ["a", "b"], **values}


def test_latest_failure_is_not_hidden_by_old_success_and_commands_stay_historical():
    module = _module()
    original = {"episodes": [_row("attempt_01", command=["/original/tree/run.py"],
        review={"audit": {"command": ["/original/tree/audit.py"]}})]}
    before = deepcopy(original)
    second = {"episodes": [_row("attempt_02", command=["/second/tree/run.py"])]}
    merged = module.merge_episode_records(original, second, runner=object())
    failed = {"episodes": [_row("attempt_04", status="failed", command=["/current/tree/replay.py"],
        failure_stage="audio", gap_state="evidence_missing_or_unsampled", reason_code="audio_peak_exceeded") ]}
    selected = module.merge_episode_records({"episodes": merged}, failed, runner=object())
    assert selected[0]["status"] == "failed"
    assert selected[0]["attempt"] == "attempt_04"
    assert [x["attempt"] for x in selected[0]["prior_attempts"]] == ["attempt_01", "attempt_02"]
    assert selected[0]["prior_attempts"][0]["command"] == ["/original/tree/run.py"]
    assert selected[0]["prior_attempts"][0]["review_audit_command"] == ["/original/tree/audit.py"]
    assert original == before


def test_coverage_uses_selected_attempt_paths_even_if_old_sidecar_exists(tmp_path):
    module = _module()
    selected = [_row("attempt_04", facts_path="/new/facts.json", questions_path="/new/questions.json")]
    old = {"episodes": [{"episode_id": "episode", "facts": "/old/facts.json", "questions": "/old/questions.json"}]}
    rerun = {"episodes": [{"episode_id": "episode", "facts": "/second/facts.json", "questions": "/second/questions.json"}]}
    result = module.build_coverage_manifest(selected, original_inputs=old, rerun_inputs=rerun,
        repository=tmp_path, original_root=tmp_path / "a", rerun_root=tmp_path / "b")
    assert result["episodes"][0]["facts"] == "/new/facts.json"
    assert result["episodes"][0]["questions"] == "/new/questions.json"
    assert old["episodes"][0]["facts"] == "/old/facts.json"


def test_supplemental_witness_must_be_declared_and_keeps_original_denominator():
    module = _module()
    original = {"episodes": [_row("attempt_01")]}
    witness = {"episodes": [_row("attempt_01", episode_id="qa10_witness")]}
    with pytest.raises(ValueError, match="declared in the manifest"):
        module.merge_episode_records(original, witness, runner=object())
    manifest = {"episodes": [{"episode_id": "episode"}, {"episode_id": "qa10_witness"}]}
    rows = module.merge_episode_records(original, witness, manifest=manifest, runner=object())
    assert [row["episode_id"] for row in rows] == ["episode", "qa10_witness"]


def test_failure_never_reuses_old_coverage_episode(tmp_path):
    module = _module()
    failed = [_row("attempt_04", status="failed", failure_stage="audio",
        gap_state="evidence_missing_or_unsampled", failure_reason="audio peak exceeded")]
    result = module.build_coverage_manifest(failed,
        original_inputs={"episodes": [{"episode_id": "episode", "facts": "/old/facts.json"}]},
        rerun_inputs={}, repository=tmp_path, original_root=tmp_path / "a", rerun_root=tmp_path / "b")
    assert result["episodes"] == []
    assert result["failed_episodes"][0]["episode_id"] == "episode"


def test_historical_rerender_failure_keeps_error_and_does_not_invent_command(tmp_path):
    import json
    module = _module()
    episode = tmp_path / "episodes" / "episode" / "attempt_03" / "episode"
    raw = {"episode_id": "episode", "dest": str(episode),
           "status": "finalize_failed", "finalize_returncode": 1}
    (tmp_path / "rerender_summary.json").write_text(json.dumps([raw]))
    record = module.load_attempt_outcomes(tmp_path)["episodes"][0]
    assert record["status"] == "failed"
    assert record["attempt"] == "attempt_03"
    assert record["historical_rerender_summary"] == raw
    assert "command" not in record
    record.update(failure_stage="audio", failure_reason="peak 1.13879 exceeds 1.0",
                  reason_code="audio_peak_exceeded", gap_state="evidence_missing_or_unsampled")
    latest = {"episodes": [_row("attempt_04")]}
    merged = module.merge_episode_records({"episodes": [record]}, latest, runner=object())
    assert merged[0]["prior_attempts"][0]["failure_reason"] == record["failure_reason"]
    assert merged[0]["prior_attempts"][0]["historical_rerender_summary"] == raw


def test_unknown_failure_stays_in_coverage_with_diagnostics():
    module = _module()
    record = _row("attempt_04", status="failed", gap_state="evidence_missing_or_unsampled",
                  reason_code="unclassified_failure", failure_reason="unexpected native abort",
                  classification_unknown=True)
    result = module.failed_coverage_records([record])
    assert result[0]["gap_state"] == "evidence_missing_or_unsampled"
    assert result[0]["failure_reason"] == "unexpected native abort"
    assert result[0]["classification_unknown"] is True


def test_manifest_source_classes_keep_device_in_family_coverage():
    module = _module()
    row = _row("attempt_06", episode_id="supplement")
    entry = {"room_family": "authored",
             "requested_source_classes": ["articulated_human", "rigid_static_object"]}
    result = module.backfill_failed_record(row, manifest_entry=entry, runner=object())
    assert result["family"] == "authored"
    assert result["class_pair"] == "human+device"


def test_latest_coverage_context_supersedes_old_audio_provenance(tmp_path):
    import json
    module = _module()
    original, rerun, latest = (tmp_path / name for name in ["old", "rerun", "latest"])
    for root in [original, rerun, latest]:
        (root / "summary").mkdir(parents=True)
    old = _row("attempt_01")
    new = _row("attempt_06", facts_path="/current/facts.json", questions_path="/current/questions.json")
    (original / "outcomes.json").write_text(json.dumps({"episodes": [old]}))
    (rerun / "outcomes.json").write_text(json.dumps({"episodes": []}))
    (latest / "outcomes.json").write_text(json.dumps({"episodes": [new]}))
    (original / "summary/coverage_inputs.json").write_text(json.dumps({"episodes": [
        {"episode_id": "episode", "source_refs": {"audio_report": "/old/gain1.json"}}]}))
    (latest / "summary/coverage_inputs.json").write_text(json.dumps({"episodes": [
        {"episode_id": "episode", "source_refs": {"audio_report": "/current/gain05.json"},
         "sound_events": [{"event_id": "actual_current_event"}]}]}))
    output = tmp_path / "merged"
    module.merge_attempts(original_root=original, rerun_root=rerun, output_root=output,
        repository=tmp_path, later_attempt_roots=[latest], apply_exposure=False, build_coverage=False,
        dry_run_summary=None, runner=object())
    context = json.loads((output / "coverage_inputs.json").read_text())["episodes"][0]
    assert context["source_refs"]["audio_report"] == "/current/gain05.json"
    assert context["sound_events"] == [{"event_id": "actual_current_event"}]
