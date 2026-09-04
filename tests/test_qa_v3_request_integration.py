"""Integration checks for the scheduler's request planning entry point."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from qa_v3_request import plan_room_questions, read_qa_params  # noqa: E402
from run_qa_v3_room_profile_scheduler import (  # noqa: E402
    SceneSpec,
    run_scheduler,
)


SCHEDULER = TOOLS / "run_qa_v3_room_profile_scheduler.py"


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _manifest(candidates=1):
    return {
        "evidence_class": "geometry_candidate",
        "counts": {
            "cells_requested": candidates,
            "geometry_candidates": candidates,
            "rejected": 0,
        },
        "search": {
            "combinations_evaluated": 1,
            "budget_exhausted": 0,
            "by_reason": {},
        },
    }


def _cli_inputs(tmp_path: Path, *, profile_count: int, scene_count: int,
                params: dict) -> tuple[Path, list[Path], Path]:
    scenes = [
        _write_json(
            tmp_path / f"scene_{index}.json",
            {"scene_id": f"scene_{index}"},
        )
        for index in range(scene_count)
    ]
    profiles = _write_json(
        tmp_path / "profiles.json",
        [{"id": f"profile_{index:02d}"} for index in range(profile_count)],
    )
    params_path = _write_json(tmp_path / "params.json", params)
    return profiles, scenes, params_path


def _plan_only_command(
    profiles: Path,
    scenes: list[Path],
    params: Path,
    output: Path,
    *extra: str,
) -> subprocess.CompletedProcess:
    argv = [
        sys.executable, str(SCHEDULER),
        "--profiles", str(profiles),
        "--params", str(params),
        "--seed", "integration-test",
        "--out-root", str(output),
        "--plan-only",
    ]
    for scene in scenes:
        argv.extend(["--scene-config", str(scene)])
    argv.extend(extra)
    return subprocess.run(argv, capture_output=True, text=True)


def test_plan_only_uses_final_item_budget_across_two_scenes_and_21_profiles(
    tmp_path,
):
    profiles, scenes, params = _cli_inputs(
        tmp_path,
        profile_count=21,
        scene_count=2,
        params={
            "ITEMS_PER_ROOM_DEFAULT": 300,
            "ANSWER_FORMS_DEFAULT": ["mcq", "open"],
        },
    )
    output = tmp_path / "plan"
    completed = _plan_only_command(profiles, scenes, params, output)
    assert completed.returncode == 0, completed.stderr

    plan = json.loads((output / "question_request.json").read_text())
    per_room = plan["per_room"]
    assert per_room["requested_budget"] == 300
    assert per_room["forms_per_candidate"] == 2
    assert per_room["planned_candidates"] == 150
    assert per_room["planned_question_count"] == 300
    assert sum(per_room["cells"].values()) == 150
    assert plan["planned_question_count"] == 600
    assert plan["scene_ids"] == ["scene_0", "scene_1"]


def test_plan_only_explicit_budget_and_forms_override_params_and_report_remainder(
    tmp_path,
):
    profiles, scenes, params = _cli_inputs(
        tmp_path,
        profile_count=2,
        scene_count=1,
        params={
            "ITEMS_PER_ROOM_DEFAULT": 9,
            "ANSWER_FORMS_DEFAULT": ["open"],
        },
    )
    output = tmp_path / "override-plan"
    completed = _plan_only_command(
        profiles,
        scenes,
        params,
        output,
        "--question-budget", "5",
        "--answer-form", "mcq",
        "--answer-form", "open",
    )
    assert completed.returncode == 0, completed.stderr

    per_room = json.loads(
        (output / "question_request.json").read_text())["per_room"]
    assert per_room["requested_budget"] == 5
    assert per_room["answer_forms"] == ["mcq", "open"]
    assert per_room["planned_candidates"] == 2
    assert per_room["planned_question_count"] == 4
    assert per_room["unallocated_budget"] == 1


def test_run_scheduler_only_calls_positive_quota_profiles_and_isolates_failure(
    tmp_path,
):
    scenes = [
        SceneSpec(tmp_path / "room_a.json", "room_a", {"scene_id": "room_a"}),
        SceneSpec(tmp_path / "room_b.json", "room_b", {"scene_id": "room_b"}),
    ]
    profiles = {
        "p0": {"id": "p0"},
        "p1": {"id": "p1"},
        "p2": {"id": "p2"},
    }
    params = {
        "ITEMS_PER_ROOM_DEFAULT": 1,
        "ANSWER_FORMS_DEFAULT": ["open"],
    }
    plan = plan_room_questions(["p0", "p1", "p2"], params)
    assert plan["cells"] == {"p0": 1, "p1": 0, "p2": 0}

    calls = []

    def fake_runner(**kwargs):
        scene = json.loads(
            kwargs["scene_config"].read_text(encoding="utf-8"))["scene_id"]
        profile = json.loads(
            kwargs["profile_config"].read_text(encoding="utf-8"))[0]["id"]
        calls.append((scene, profile, kwargs["cells"]))
        if scene == "room_a":
            raise RuntimeError("synthetic pair failure")
        result = _manifest(kwargs["cells"])
        kwargs["batch_root"].mkdir(parents=True)
        (kwargs["batch_root"] / "batch_manifest.json").write_text(
            json.dumps(result), encoding="utf-8")
        return result

    matrix = run_scheduler(
        scene_specs=scenes,
        profile_catalog=profiles,
        requested_profiles=["p0", "p1", "p2"],
        params_value=params,
        params_source=tmp_path / "params.json",
        out_root=tmp_path / "scheduler-out",
        cells=plan["cells"],
        seed="integration-test",
        snapshot_content="/unused",
        pixel_results={},
        runner=fake_runner,
        request_plan=plan,
    )

    assert calls == [("room_a", "p0", 1), ("room_b", "p0", 1)]
    assert matrix["counts_by_status"] == {
        "pipeline_error": 1,
        "generated": 1,
        "not_scheduled": 4,
    }
    rows = {
        (row["scene_id"], row["profile_id"]): row
        for row in matrix["matrix"]
    }
    assert rows[("room_a", "p0")]["attempt_status"] == "pipeline_error"
    assert rows[("room_b", "p0")]["attempt_status"] == "generated"
    for scene in ("room_a", "room_b"):
        for profile in ("p1", "p2"):
            row = rows[(scene, profile)]
            assert row["attempt_status"] == "not_scheduled"
            assert row["requested_cells"] == 0
    assert matrix["counts_by_status"].get("generated", 0) == 1


def test_read_qa_params_resolves_relative_pool_from_source_and_snapshot_does_not_drift(
    tmp_path, monkeypatch,
):
    params_path = tmp_path / "config" / "params.json"
    expected_pool = params_path.parent / "pool" / "events.json"
    _write_json(
        params_path,
        {
            "SOUND_EVENT_POOL": "pool/events.json",
            "ITEMS_PER_ROOM_DEFAULT": 1,
            "ANSWER_FORMS_DEFAULT": ["open"],
        },
    )
    monkeypatch.chdir(tmp_path)
    params = read_qa_params(params_path)
    assert params["SOUND_EVENT_POOL"] == str(expected_pool.resolve())

    scene = SceneSpec(
        tmp_path / "scene.json", "room", {"scene_id": "room"})
    output = tmp_path / "snapshot-out"
    run_scheduler(
        scene_specs=[scene],
        profile_catalog={"p0": {"id": "p0"}},
        requested_profiles=["p0"],
        params_value=params,
        params_source=params_path,
        out_root=output,
        cells={"p0": 0},
        seed="integration-test",
        snapshot_content="/unused",
        pixel_results={},
        runner=lambda **kwargs: pytest.fail("zero quota must not run"),
    )
    snapshot = json.loads(
        (output / "inputs" / "params.json").read_text(encoding="utf-8"))
    assert snapshot["SOUND_EVENT_POOL"] == params["SOUND_EVENT_POOL"]
