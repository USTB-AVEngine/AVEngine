"""Fleet planning: no house twice, no flood, deterministic order."""

from __future__ import annotations

from avengine.studio.fleet import fleet_status, plan_fleet, scene_dir_of


def _scene(name: str, split: str = "val") -> dict:
    return {"name": name, "split": split, "scene_dir": f"/hm3d/{split}/{name}"}


def _task(name: str, status: str, split: str = "val") -> dict:
    return {
        "template": "hm3d_end_to_end",
        "status": status,
        "argv": ["python", "run.py", "--scene-dir", f"/hm3d/{split}/{name}",
                 "--split", split],
    }


def test_scene_dir_is_read_from_the_tasks_own_argv() -> None:
    assert scene_dir_of(_task("00800-a", "pass")) == "/hm3d/val/00800-a"
    assert scene_dir_of({"argv": ["python", "run.py"]}) is None


def test_attempted_houses_are_never_resubmitted() -> None:
    scenes = [_scene("00800-a"), _scene("00802-b"), _scene("00803-c")]
    tasks = [_task("00800-a", "pass"), _task("00802-b", "fail")]
    plan = plan_fleet(scenes, tasks, keep=5)
    assert [entry["name"] for entry in plan["submit"]] == ["00803-c"]
    assert plan["passed"] == 1 and plan["failed"] == 1


def test_in_flight_chains_hold_the_feeder_back() -> None:
    scenes = [_scene(f"0080{i}-x") for i in range(6)]
    tasks = [_task("00800-x", "running"), _task("00801-x", "queued")]
    plan = plan_fleet(scenes, tasks, keep=2)
    assert plan["in_flight"] == 2 and plan["submit"] == []
    plan = plan_fleet(scenes, tasks, keep=3)
    assert [entry["name"] for entry in plan["submit"]] == ["00802-x"]


def test_a_rerun_pass_outranks_the_failure_it_fixed() -> None:
    tasks = [_task("00800-a", "fail"), _task("00800-a", "pass")]
    status = fleet_status([_scene("00800-a")], tasks)
    assert status["by_scene_dir"] == {"/hm3d/val/00800-a": "pass"}
    assert status["failed"] == 0


def test_the_plan_is_deterministic_and_val_goes_first() -> None:
    scenes = [_scene("00800-a", "val"), _scene("00006-t", "train")]
    first = plan_fleet(scenes, [], keep=1)
    again = plan_fleet(scenes, [], keep=1)
    assert first == again
    assert first["submit"][0]["name"] == "00800-a"
    # other templates' tasks are not the fleet's business
    unrelated = [{"template": "hm3d_episode", "status": "running", "argv": []}]
    assert plan_fleet(scenes, unrelated, keep=1)["in_flight"] == 0
