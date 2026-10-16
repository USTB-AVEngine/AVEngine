#!/usr/bin/env python3
"""Keep the studio queue fed with the next un-attempted HM3D houses.

Run from cron. Each invocation asks the studio which houses exist and
which fleet tasks ran, plans the next submissions with
avengine.studio.fleet (never re-attempting a house, holding at most
--keep chains in flight), submits them, prints one JSON line, exits.

A pause file stops the fleet without touching cron: create it and the
feeder does nothing until it is removed. Failures are never retried by
this script - a red house is a decision for a person, made by clicking
that house on the homepage.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.studio.fleet import plan_fleet  # noqa: E402


def fetch(studio: str, path: str) -> dict:
    with urllib.request.urlopen(f"{studio}{path}", timeout=30) as response:
        return json.load(response)


def submit(studio: str, overrides: dict) -> dict:
    request = urllib.request.Request(
        f"{studio}/api/tasks",
        data=json.dumps(
            {"template": "hm3d_end_to_end", "overrides": overrides}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio", default="http://127.0.0.1:8765")
    parser.add_argument("--keep", type=int, default=2,
                        help="maximum fleet chains queued or running at once")
    parser.add_argument(
        "--pause-file",
        type=Path,
        default=Path("/data/avengine_external/studio/FLEET_PAUSE"),
        help="if this file exists the feeder does nothing",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.pause_file.exists():
        print(json.dumps({"paused_by": str(args.pause_file)}))
        return 0

    scenes = fetch(args.studio, "/api/hm3d-scenes").get("scenes") or []
    tasks = fetch(args.studio, "/api/tasks").get("tasks") or []
    plan = plan_fleet(scenes, tasks, keep=args.keep)

    submitted = []
    for entry in plan["submit"]:
        if args.dry_run:
            submitted.append({"name": entry["name"], "dry_run": True})
            continue
        payload = submit(
            args.studio,
            {"scene_dir": entry["scene_dir"], "split": entry["split"]},
        )
        submitted.append(
            {
                "name": entry["name"],
                "task_id": (payload.get("task") or {}).get("task_id"),
                "error": payload.get("error"),
            }
        )

    print(
        json.dumps(
            {
                "total_scenes": plan["total_scenes"],
                "passed": plan["passed"],
                "failed": plan["failed"],
                "in_flight": plan["in_flight"],
                "submitted": submitted,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
