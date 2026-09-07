"""Merge original and rerun QA batch attempts, including failed-episode accounting."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from avengine.qa.batch_coverage import CODEX_WORKTREE_PREFIX, rewrite_codex_worktree_path
from avengine.qa.batch_manifest import collect_batch_outcomes


REPO = Path(__file__).resolve().parents[2]


def _load_merge():
    path = REPO / "tools/dataset/merge_qa_batch_attempts.py"
    spec = importlib.util.spec_from_file_location("merge_qa_batch_attempts_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_runner():
    path = REPO / "tools/dataset/run_qa_batch.py"
    spec = importlib.util.spec_from_file_location("qa_batch_runner_merge_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Committed 46-cell roster (original attempt_01 statuses). Rerun IDs overlay these.
PILOT46_ORIG_STATUS = {
    "qa_pilot46_20260907_apartment_human_human": "delivered",
    "qa_pilot46_20260907_apartment_human_animal": "delivered",
    "qa_pilot46_20260907_apartment_human_device": "delivered",
    "qa_pilot46_20260907_apartment_animal_animal": "delivered",
    "qa_pilot46_20260907_apartment_animal_device": "delivered",
    "qa_pilot46_20260907_apartment_device_device": "delivered",
    "qa_pilot46_20260907_apartment_single_active": "delivered",
    "qa_pilot46_20260907_authored_a_human_human": "delivered",
    "qa_pilot46_20260907_authored_a_human_animal": "delivered",
    "qa_pilot46_20260907_authored_a_human_device": "delivered",
    "qa_pilot46_20260907_authored_a_animal_animal": "delivered",
    "qa_pilot46_20260907_authored_a_animal_device": "delivered",
    "qa_pilot46_20260907_authored_a_device_device": "blocked",
    "qa_pilot46_20260907_authored_b_human_human": "delivered",
    "qa_pilot46_20260907_authored_b_human_animal": "delivered",
    "qa_pilot46_20260907_authored_b_human_device": "delivered",
    "qa_pilot46_20260907_authored_b_animal_animal": "delivered",
    "qa_pilot46_20260907_authored_b_animal_device": "delivered",
    "qa_pilot46_20260907_authored_b_device_device": "delivered",
    "qa_pilot46_20260907_authored_c_human_human": "delivered",
    "qa_pilot46_20260907_authored_c_human_animal": "delivered",
    "qa_pilot46_20260907_authored_c_human_device": "delivered",
    "qa_pilot46_20260907_authored_c_animal_animal": "delivered",
    "qa_pilot46_20260907_authored_c_animal_device": "failed",
    "qa_pilot46_20260907_authored_c_device_device": "delivered",
    "qa_pilot46_20260907_kujiale_human_human": "delivered",
    "qa_pilot46_20260907_kujiale_human_animal": "delivered",
    "qa_pilot46_20260907_kujiale_human_device": "delivered",
    "qa_pilot46_20260907_kujiale_animal_animal": "delivered",
    "qa_pilot46_20260907_kujiale_animal_device": "delivered",
    "qa_pilot46_20260907_kujiale_device_device": "blocked",
    "qa_pilot46_20260907_kujiale_single_active": "delivered",
    "qa_pilot46_20260907_mp3d_human_human": "delivered",
    "qa_pilot46_20260907_mp3d_human_animal": "failed",
    "qa_pilot46_20260907_mp3d_human_device": "delivered",
    "qa_pilot46_20260907_mp3d_animal_animal": "delivered",
    "qa_pilot46_20260907_mp3d_animal_device": "delivered",
    "qa_pilot46_20260907_mp3d_device_device": "delivered",
    "qa_pilot46_20260907_mp3d_single_active": "failed",
    "qa_pilot46_20260907_hm3d_human_human": "delivered",
    "qa_pilot46_20260907_hm3d_human_animal": "failed",
    "qa_pilot46_20260907_hm3d_human_device": "delivered",
    "qa_pilot46_20260907_hm3d_animal_animal": "delivered",
    "qa_pilot46_20260907_hm3d_animal_device": "delivered",
    "qa_pilot46_20260907_hm3d_device_device": "delivered",
    "qa_pilot46_20260907_hm3d_single_active": "failed",
}

RERUN_STATUS = {
    "qa_pilot46_20260907_mp3d_human_animal": "delivered",
    "qa_pilot46_20260907_mp3d_single_active": "delivered",
    "qa_pilot46_20260907_hm3d_human_animal": "delivered",
    "qa_pilot46_20260907_hm3d_single_active": "delivered",
    "qa_pilot46_20260907_authored_a_human_human": "delivered",
    "qa_pilot46_20260907_authored_a_human_animal": "delivered",
    "qa_pilot46_20260907_authored_a_human_device": "delivered",
    "qa_pilot46_20260907_authored_a_animal_animal": "delivered",
    "qa_pilot46_20260907_authored_a_animal_device": "delivered",
    "qa_pilot46_20260907_authored_b_human_human": "delivered",
    "qa_pilot46_20260907_authored_b_human_animal": "delivered",
    "qa_pilot46_20260907_authored_b_human_device": "delivered",
    "qa_pilot46_20260907_authored_b_animal_animal": "delivered",
    "qa_pilot46_20260907_authored_b_animal_device": "failed",
    "qa_pilot46_20260907_authored_b_device_device": "delivered",
    "qa_pilot46_20260907_authored_c_human_human": "delivered",
    "qa_pilot46_20260907_authored_c_human_animal": "delivered",
    "qa_pilot46_20260907_authored_c_human_device": "delivered",
    "qa_pilot46_20260907_authored_c_animal_animal": "delivered",
    "qa_pilot46_20260907_authored_c_device_device": "delivered",
}

FAILED_CELLS = {
    "qa_pilot46_20260907_authored_a_device_device": {
        "room_id": "aea_loc3_social_rebuild_v1",
        "asset_ids": [
            "generated_toilet_elevated_tank_exposed_pipe_white_research_v2",
            "generated_fireplace_wood_stove_matte_black_research_v1",
        ],
        "gaps": [{"code": "fixed_sound_identities_exceed_profile_clip_budget",
                  "state": "evidence_missing_or_unsampled"}],
    },
    "qa_pilot46_20260907_kujiale_device_device": {
        "room_id": "kujiale_0020_full_home_v1",
        "asset_ids": [
            "generated_blender_bullet_blender_silver_research_v1",
            "generated_printer_desktop_inkjet_matte_black_research_v1",
        ],
        "gaps": [{"code": "fixed_sound_identities_exceed_profile_clip_budget",
                  "state": "evidence_missing_or_unsampled"}],
    },
    "qa_pilot46_20260907_authored_c_animal_device": {
        "room_id": "authored_open_family_home_room_c_v1",
        "asset_ids": [
            "generated_shiba_inu_red_medium_standard_adult_research_v2",
            "generated_smart_speaker_fabric_wrapped_cylinder_charcoal_research_v1",
        ],
        "gaps": [],
        "histogram": {
            "camera:no_joint_geometry_activity_schedule": 169,
            "routes:initial_source_separation_below_0.95_m": 31,
        },
    },
    "qa_pilot46_20260907_authored_b_animal_device": {
        "room_id": "authored_compact_home_room_b_v1",
        "asset_ids": [
            "generated_burmese_dark_sable_research_v1",
            "generated_smart_speaker_fabric_wrapped_cylinder_sandstone_research_v1",
        ],
        "gaps": [],
        "histogram": {
            "camera:no_joint_geometry_activity_schedule": 172,
            "routes:initial_source_separation_below_0.95_m": 28,
        },
    },
}


def _outcome(episode_id: str, status: str, attempt_root: Path, *, attempt: str, **extra) -> dict:
    rec = {
        "episode_id": episode_id,
        "status": status,
        "attempt_root": str(attempt_root),
        "episode_output_root": str(attempt_root / "episode"),
        "request_path": str(attempt_root.parent / "request.json"),
        "stderr_log": str(attempt_root / "stderr.log"),
        "stdout_log": str(attempt_root / "stdout.log"),
        "attempt": attempt,
    }
    rec.update(extra)
    return rec


def _write_attempt(
    root: Path,
    episode_id: str,
    *,
    attempt: str,
    status: str,
    spec: dict | None = None,
    old_controller_exit: bool = False,
) -> dict:
    episode_dir = root / "episodes" / episode_id
    attempt_root = episode_dir / attempt
    attempt_root.mkdir(parents=True)
    (attempt_root / "stdout.log").write_text("", encoding="utf-8")
    spec = spec or {}
    request = {
        "episode_id": episode_id,
        "room_id": spec.get("room_id") or "room_x",
        "source_asset_ids": list(spec.get("asset_ids") or ["asset_a", "asset_b"]),
    }
    (episode_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    extra = {}
    if status == "blocked":
        (attempt_root / "stderr.log").write_text(
            "preallocation_gap: manifest row has a known preallocation gap\n", encoding="utf-8")
        extra.update(reason="manifest row has a known preallocation gap",
                     reason_code="preallocation_gap")
    elif status == "failed" and spec.get("histogram"):
        episode = attempt_root / "episode"
        episode.mkdir()
        histogram = spec["histogram"]
        (episode / "planning_result.json").write_text(json.dumps({
            "status": "failed",
            "attempts": 200,
            "failure_histogram": histogram,
            "gap_category": "evidence_missing_or_unsampled",
        }), encoding="utf-8")
        (episode / "request.json").write_text(json.dumps(request), encoding="utf-8")
        (attempt_root / "stderr.log").write_text(
            "avengine.rooms.qa_episode.QAPlanningError: no existing room could realize the request: "
            "[{'room_id': '%s', 'status': 'not_selected', 'reason': "
            "\"ConditionedPlanningFailure: fixed condition profile exhausted: %s\"}]\n"
            % (spec["room_id"], histogram),
            encoding="utf-8",
        )
        if old_controller_exit:
            extra.update(reason="controller exited with 1", reason_code="controller_exit",
                         controller_returncode=1)
        else:
            extra.update(
                reason="ConditionedPlanningFailure: fixed condition profile exhausted " + json.dumps(histogram),
                reason_code="planning_exhausted",
                failure_stage="planning",
                gap_state="evidence_missing_or_unsampled",
                failure_reason="ConditionedPlanningFailure: fixed condition profile exhausted " + json.dumps(histogram),
                controller_returncode=1,
            )
    else:
        (attempt_root / "stderr.log").write_text("", encoding="utf-8")
        extra.update(reason=None, reason_code=None)
    return _outcome(episode_id, status, attempt_root, attempt=attempt, **extra)


def _manifest_entry(episode_id: str, spec: dict | None = None) -> dict:
    spec = spec or {}
    asset_ids = list(spec.get("asset_ids") or ["asset_a", "asset_b"])
    return {
        "episode_id": episode_id,
        "room_id": spec.get("room_id") or "room_x",
        "room_family": "authored",
        "source_assignments": [{"actor_id": f"source{i+1}", "asset_id": asset_id}
                               for i, asset_id in enumerate(asset_ids)],
        "preallocation_gaps": list(spec.get("gaps") or []),
        "request": {"room_id": spec.get("room_id") or "room_x", "source_asset_ids": asset_ids},
        "requested_profile": {"separation_bin_deg": [30, 60]},
        "requested_quota_by_qa": {f"QA-{i:02d}": 1 for i in range(1, 25)},
        "condition_group": "identity_binding",
    }


def _build_pilot46_world(tmp_path: Path) -> tuple[dict, dict, dict]:
    original_root = tmp_path / "original"
    rerun_root = tmp_path / "rerun"
    orig_episodes = []
    for eid, status in PILOT46_ORIG_STATUS.items():
        spec = FAILED_CELLS.get(eid)
        orig_episodes.append(_write_attempt(
            original_root, eid, attempt="attempt_01", status=status, spec=spec,
            old_controller_exit=(eid == "qa_pilot46_20260907_authored_c_animal_device"),
        ))
    rerun_episodes = []
    for eid, status in RERUN_STATUS.items():
        spec = FAILED_CELLS.get(eid)
        rerun_episodes.append(_write_attempt(
            rerun_root, eid, attempt="attempt_02", status=status, spec=spec,
        ))
    manifest = {
        "batch_id": "qa_pilot46_20260907",
        "episodes": [_manifest_entry(eid, FAILED_CELLS.get(eid)) for eid in PILOT46_ORIG_STATUS],
    }
    original = {"episodes": orig_episodes}
    rerun = {"episodes": rerun_episodes}
    return original, rerun, manifest


def test_gap_state_rules_cover_three_classes():
    runner = _load_runner()
    planning = runner.gap_state_for_failure(
        failure_stage="planning",
        reason="ConditionedPlanningFailure: fixed condition profile exhausted "
               "{'camera:no_joint_geometry_activity_schedule': 169, "
               "'routes:initial_source_separation_below_0.95_m': 31}",
    )
    preallocation = runner.gap_state_for_failure(
        failure_stage="planning",
        reason="manifest row has a known preallocation gap",
        reason_code="preallocation_gap",
    )
    capture = runner.gap_state_for_failure(
        failure_stage="capture",
        reason="CalledProcessError: capture renderer exited with 2",
    )
    assert planning == "evidence_missing_or_unsampled"
    assert preallocation == "evidence_missing_or_unsampled"
    assert capture == "interface_not_implemented"


def test_classify_controller_failure_three_classes(tmp_path: Path):
    runner = _load_runner()
    planning_root = tmp_path / "plan"
    planning_root.mkdir()
    (planning_root / "planning_result.json").write_text(json.dumps({
        "status": "failed",
        "failure_histogram": {
            "camera:no_joint_geometry_activity_schedule": 169,
            "routes:initial_source_separation_below_0.95_m": 31,
        },
    }), encoding="utf-8")
    (planning_root / "stderr.log").write_text(
        "ConditionedPlanningFailure: fixed condition profile exhausted "
        "{'camera:no_joint_geometry_activity_schedule': 169, "
        "'routes:initial_source_separation_below_0.95_m': 31}\n",
        encoding="utf-8",
    )
    planned = runner.classify_controller_failure(
        episode_output_root=planning_root,
        stderr_path=planning_root / "stderr.log",
        returncode=1,
    )
    assert planned["failure_stage"] == "planning"
    assert planned["gap_state"] == "evidence_missing_or_unsampled"
    assert planned["reason_code"] == "planning_exhausted"

    audio_root = tmp_path / "audio"
    (audio_root / "capture").mkdir(parents=True)
    (audio_root / "execution_commands.json").write_text("{}", encoding="utf-8")
    (audio_root / "capture" / "neutral_readback.json").write_text("{}", encoding="utf-8")
    delivery = audio_root / "delivery"
    delivery.mkdir()
    (delivery / "audio.log").write_text(json.dumps({
        "status": "fail",
        "error": "AudioProgram validation failed: sequential_sources events must not overlap",
    }), encoding="utf-8")
    audio = runner.classify_controller_failure(episode_output_root=audio_root, returncode=1)
    assert audio["failure_stage"] == "audio"
    assert audio["gap_state"] == "interface_not_implemented"

    blocked = runner.gap_state_for_failure(
        failure_stage="planning",
        reason="manifest row has a known preallocation gap",
        reason_code="preallocation_gap",
    )
    assert blocked == "evidence_missing_or_unsampled"


def test_reconstruct_46_cell_merge_table(tmp_path: Path):
    merge = _load_merge()
    original, rerun, manifest = _build_pilot46_world(tmp_path)
    merged = merge.merge_episode_records(original, rerun, manifest=manifest, runner=_load_runner())
    assert len(merged) == 46
    counts = {}
    for rec in merged:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    assert counts == {"delivered": 42, "blocked": 2, "failed": 2}

    by_id = {rec["episode_id"]: rec for rec in merged}
    animal_c = by_id["qa_pilot46_20260907_authored_c_animal_device"]
    assert animal_c["attempt"] == "attempt_01"
    assert animal_c["failure_stage"] == "planning"
    assert animal_c["gap_state"] == "evidence_missing_or_unsampled"
    assert animal_c["reason_code"] == "planning_exhausted"
    assert animal_c["room_id"] == "authored_open_family_home_room_c_v1"
    assert animal_c["asset_ids"] == FAILED_CELLS["qa_pilot46_20260907_authored_c_animal_device"]["asset_ids"]
    assert "169" in (animal_c.get("failure_reason") or "")

    blocked_a = by_id["qa_pilot46_20260907_authored_a_device_device"]
    blocked_k = by_id["qa_pilot46_20260907_kujiale_device_device"]
    for rec in (blocked_a, blocked_k):
        assert rec["status"] == "blocked"
        assert rec["failure_stage"] == "planning"
        assert rec["gap_state"] == "evidence_missing_or_unsampled"
        assert rec["reason_code"] == "preallocation_gap"
        assert rec["room_id"]
        assert rec["asset_ids"]

    failed_b = by_id["qa_pilot46_20260907_authored_b_animal_device"]
    assert failed_b["attempt"] == "attempt_02"
    assert failed_b["gap_state"] == "evidence_missing_or_unsampled"
    assert failed_b["room_id"] == "authored_compact_home_room_b_v1"

    failed = merge.failed_coverage_records(merged)
    assert len(failed) == 4
    assert {row["episode_id"] for row in failed} == set(FAILED_CELLS)
    assert all(row["asset_ids"] and row["room_id"] and row["gap_state"] for row in failed)


def test_merge_passes_failed_episodes_and_rewrites_codex_paths(tmp_path: Path, monkeypatch):
    merge = _load_merge()
    original, rerun, manifest = _build_pilot46_world(tmp_path)
    original_root = tmp_path / "original"
    rerun_root = tmp_path / "rerun"
    (original_root / "outcomes.json").write_text(json.dumps(original), encoding="utf-8")
    (rerun_root / "outcomes.json").write_text(json.dumps(rerun), encoding="utf-8")
    (original_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (original_root / "summary").mkdir()
    (rerun_root / "summary").mkdir()
    (original_root / "summary" / "coverage_inputs.json").write_text(json.dumps({
        "schema": "avengine_qa_batch_episode_input_manifest_v1",
        "asset_inventory": f"{CODEX_WORKTREE_PREFIX}/examples/runtime/source_asset_runtime_profiles.json",
        "room_catalog": f"{CODEX_WORKTREE_PREFIX}/examples/rooms/packages/catalog.json",
        "runtime_registry": f"{CODEX_WORKTREE_PREFIX}/examples/runtime/source_asset_runtime_profiles.json",
        "episodes": [],
    }), encoding="utf-8")
    (rerun_root / "summary" / "coverage_inputs.json").write_text(json.dumps({"episodes": []}), encoding="utf-8")

    captured = {}

    def fake_coverage(manifest_payload, *, repository=None):
        captured["manifest"] = manifest_payload
        captured["repository"] = str(repository)
        return {"rows": [], "provenance": {"asset_inventory": manifest_payload.get("asset_inventory")}}

    def fake_write(result, output_dir):
        Path(output_dir).mkdir(parents=True)
        return {"coverage": str(Path(output_dir) / "coverage.json")}

    monkeypatch.setattr(merge, "build_batch_coverage", fake_coverage)
    monkeypatch.setattr(merge, "write_batch_coverage", fake_write)
    out = tmp_path / "merged"
    summary = merge.merge_attempts(
        original_root=original_root,
        rerun_root=rerun_root,
        output_root=out,
        repository=REPO,
        apply_exposure=False,
        build_coverage=True,
        runner=_load_runner(),
        dry_run_summary=None,
    )
    assert summary["merged_status_counts"] == {"delivered": 42, "blocked": 2, "failed": 2}
    assert captured["manifest"]["failed_episodes"]
    assert len(captured["manifest"]["failed_episodes"]) == 4
    for key in ("asset_inventory", "room_catalog", "runtime_registry"):
        value = captured["manifest"][key]
        assert CODEX_WORKTREE_PREFIX not in str(value)
        assert str(REPO) in str(value)
    table = json.loads((out / "merged_episodes.json").read_text())
    assert table["episode_denominator"] == 46
    animal_c = next(r for r in table["episodes"] if r["episode_id"].endswith("authored_c_animal_device"))
    assert animal_c["gap_state"] == "evidence_missing_or_unsampled"


def test_rewrite_codex_worktree_path_uses_repository(tmp_path: Path):
    inventory = tmp_path / "examples/runtime/source_asset_runtime_profiles.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("{}", encoding="utf-8")
    rewritten = rewrite_codex_worktree_path(
        f"{CODEX_WORKTREE_PREFIX}/examples/runtime/source_asset_runtime_profiles.json",
        repository=tmp_path,
        fallback=tmp_path / "missing.json",
    )
    assert rewritten == str(inventory.resolve())
    assert CODEX_WORKTREE_PREFIX not in rewritten


def test_collect_batch_outcomes_fills_failure_stage_and_gap_state():
    manifest = {
        "batch_id": "test",
        "episodes": [
            _manifest_entry("batch_001", FAILED_CELLS["qa_pilot46_20260907_authored_a_device_device"]),
            _manifest_entry("batch_002", FAILED_CELLS["qa_pilot46_20260907_authored_c_animal_device"]),
        ],
    }
    # Align episode_ids with the manifest rows built above.
    manifest["episodes"][0]["episode_id"] = "batch_001"
    manifest["episodes"][1]["episode_id"] = "batch_002"
    result = collect_batch_outcomes(manifest, [
        {"episode_id": "batch_001", "status": "preallocation_blocked",
         "reason_code": "preallocation_gap",
         "failure_reason": "manifest row has a known preallocation gap"},
        {"episode_id": "batch_002", "status": "planning_failed",
         "failure_histogram": {"camera:no_joint_geometry_activity_schedule": 169},
         "failure_reason": "ConditionedPlanningFailure: fixed condition profile exhausted"},
    ])
    by_id = {row["episode_id"]: row for row in result["episodes"]}
    assert by_id["batch_001"]["failure_stage"] == "planning"
    assert by_id["batch_001"]["gap_state"] == "evidence_missing_or_unsampled"
    assert by_id["batch_002"]["failure_stage"] == "planning"
    assert by_id["batch_002"]["gap_state"] == "evidence_missing_or_unsampled"
    assert by_id["batch_001"]["outcome"]["gap_state"] == "evidence_missing_or_unsampled"
