"""Fleet planning: which houses the queue should run next, decided by code.

Mass production is not a job for a person with a mouse. The owner's rule
is that humans appear in this pipeline exactly twice - contributing
source material, and reading a red stamp's one-line reason - so the
feeder below keeps the task queue topped up with the next un-attempted
houses and otherwise does nothing.

Two deliberate refusals shape the plan:

* A house that was ever attempted is never auto-resubmitted, whatever
  the outcome. Re-running a failure without a decision would grind a
  systematic problem into noise (standing decision from 2026-08-27);
  re-running a pass would only burn compute. Retry is a human choice,
  made by clicking the house on the homepage.
* The feeder holds at most ``keep`` chains queued or running. The queue
  is serial, so this only bounds how much work a mistake can enqueue,
  not how fast the fleet goes.
"""

from __future__ import annotations

from typing import Any, Mapping

FLEET_TEMPLATE = "hm3d_end_to_end"


def scene_dir_of(task: Mapping[str, Any]) -> str | None:
    """The house a fleet task runs on, read from its own argv."""

    argv = [str(item) for item in task.get("argv") or []]
    try:
        return argv[argv.index("--scene-dir") + 1]
    except (ValueError, IndexError):
        return None


def fleet_status(
    scenes: list[Mapping[str, Any]], tasks: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Per-house outcome of every fleet task so far.

    A house's status is its best outcome: pass beats an in-flight
    attempt, which beats fail - a re-run that succeeded should not leave
    the house painted by the failure it fixed.
    """

    rank = {"pass": 3, "running": 2, "queued": 2, "fail": 1}
    by_dir: dict[str, str] = {}
    in_flight = 0
    for task in tasks:
        if str(task.get("template")) != FLEET_TEMPLATE:
            continue
        status = str(task.get("status"))
        if status in ("queued", "running"):
            in_flight += 1
        directory = scene_dir_of(task)
        if directory is None:
            continue
        current = by_dir.get(directory)
        if current is None or rank.get(status, 0) > rank.get(current, 0):
            by_dir[directory] = status
    passed = sorted(d for d, s in by_dir.items() if s == "pass")
    failed = sorted(d for d, s in by_dir.items() if s == "fail")
    return {
        "total_scenes": len(scenes),
        "attempted": len(by_dir),
        "passed": len(passed),
        "failed": len(failed),
        "failed_scene_dirs": failed,
        "in_flight": in_flight,
        "by_scene_dir": by_dir,
    }


def plan_fleet(
    scenes: list[Mapping[str, Any]],
    tasks: list[Mapping[str, Any]],
    *,
    keep: int = 2,
) -> dict[str, Any]:
    """The submissions that keep ``keep`` chains in flight, and why.

    Candidates are houses never attempted, in the scene listing's own
    order (val first, then train - the split the paper evaluates on gets
    its houses first). The plan is deterministic: same inputs, same
    submissions, so a feeder run twice by accident submits nothing the
    second time.
    """

    status = fleet_status(scenes, tasks)
    slots = max(0, int(keep) - status["in_flight"])
    submit = [
        {
            "scene_dir": str(scene.get("scene_dir")),
            "split": str(scene.get("split")),
            "name": scene.get("name"),
        }
        for scene in scenes
        if str(scene.get("scene_dir")) not in status["by_scene_dir"]
    ][:slots]
    return {**status, "keep": keep, "slots": slots, "submit": submit}
