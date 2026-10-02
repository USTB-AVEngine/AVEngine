#!/usr/bin/env python3
"""Re-submit failed hm3d_end_to_end tasks via the studio API.

The fleet never auto-resubmits a house that was attempted once (fleet.py), so
the 145 failed train houses need a manual re-submission now that their
.basis.navmesh files exist. This script drives that re-submission strictly
through the public API - no touching of task dirs, no fleet restart, no
deleting old tasks.

How it keeps the queue at the caller's chosen pace: it counts *in-flight*
tasks (status queued/running, template hm3d_end_to_end) via GET /api/tasks
and only submits when that count is below the cap. The studio server runs one
daemon worker that executes strictly in submission order (tasks.py), so the
cap bounds the queue, never the parallelism.

The argv for each submission is reconstructed from the *historical* failed
task's argv (key/value pairs of --flags), with only scene_dir and split
overridden - the behaviour of the resubmission is therefore identical to the
original, just with a fresh task_id/output directory.

Usage:
  python submit_hm3d_tasks.py --house 00506-QVAA6zecMHu --wait   # one house, poll to pass/fail
  python submit_hm3d_tasks.py --all --watch --in-flight-cap 2    # drive the remaining 144
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "http://localhost:8765"
TASKS_ROOT = Path("/data/avengine_external/studio/tasks")
TEMPLATE = "hm3d_end_to_end"


def http_json(path: str, body: dict | None = None, timeout: int = 120) -> dict:
    url = f"{API}{path}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} on {path}: {detail}") from error


def collect_failed_houses() -> dict[str, dict]:
    """Map scene_dir -> argv overrides, from the historical failed tasks."""
    by_scene: dict[str, dict] = {}
    for task_json in TASKS_ROOT.glob(f"*{TEMPLATE}/task.json"):
        record = json.loads(task_json.read_text(encoding="utf-8"))
        if record.get("status") != "fail":
            continue
        argv = record.get("argv") or []
        overrides = {}
        for i in range(len(argv) - 1):
            if argv[i].startswith("--"):
                overrides[argv[i][2:].replace("-", "_")] = argv[i + 1]
        scene_dir = overrides.get("scene_dir")
        if scene_dir and "/train/" in scene_dir:
            by_scene[scene_dir] = overrides
    return by_scene


def submit(overrides: dict) -> dict:
    body = {"template": TEMPLATE, "overrides": overrides}
    return http_json("/api/tasks", body=body)


def in_flight_count(exclude_ids: set[str]) -> int:
    data = http_json("/api/tasks")
    count = 0
    for task in data.get("tasks", []):
        if task.get("template") != TEMPLATE:
            continue
        if task.get("task_id") in exclude_ids:
            continue
        if task.get("status") in ("queued", "running"):
            count += 1
    return count


def task_status(task_id: str) -> str:
    data = http_json(f"/api/tasks/{task_id}")
    task = data.get("task") or data
    return str(task.get("status") or "")


def wait_finished(task_id: str, poll_s: int = 60, timeout_s: int = 7200) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = task_status(task_id)
        print(f"  [{task_id}] status={status} {time.strftime('%H:%M:%S')}", flush=True)
        if status in ("pass", "fail", "error"):
            return status
        time.sleep(poll_s)
    return "timeout"


def check_registered(house: str) -> bool:
    data = http_json("/api/room-curation")
    return any(h.get("house") == house for h in data.get("houses", []))


def rooms_json_for(task_id: str, scene_dir: str) -> Path | None:
    # scene_dir 如 .../train/00506-QVAA6zecMHu → house 名 hm3d_train_00506_QVAA6zecMHu
    scene = Path(scene_dir)
    split = scene.parent.name  # train / val
    house = f"hm3d_{split}_{scene.name.replace('-', '_')}"
    for candidate in (
        TASKS_ROOT / task_id / "output" / "render" / "rooms" / house / "rooms.json",
        TASKS_ROOT / task_id / "output" / "render" / house / "rooms.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", help="scene dir basename, e.g. 00506-QVAA6zecMHu")
    parser.add_argument("--wait", action="store_true", help="poll until pass/fail")
    parser.add_argument("--all", action="store_true", help="drive every failed train house")
    parser.add_argument("--watch", action="store_true", help="loop until every house submitted")
    parser.add_argument("--in-flight-cap", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    houses = collect_failed_houses()
    print(f"failed train houses on record: {len(houses)}", flush=True)
    if not houses:
        return 1

    targets: dict[str, dict] = {}
    if args.house:
        scene_dir = f"/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/train/{args.house}"
        if scene_dir not in houses:
            print(f"house {args.house} not among failed train houses")
            return 1
        targets[scene_dir] = houses[scene_dir]
    elif args.all:
        targets = dict(houses)
    else:
        print("need --house or --all")
        return 1

    done: dict[str, str] = {}
    for scene_dir, overrides in sorted(targets.items()):
        if args.watch:
            while in_flight_count(set()) >= args.in_flight_cap:
                print("  queue full, waiting...", flush=True)
                time.sleep(60)
        house = Path(scene_dir).name
        house_name = f"hm3d_{Path(scene_dir).parent.name}_{house.replace('-', '_')}"
        if check_registered(house_name):
            print(f"skipped {house}: already registered", flush=True)
            done[scene_dir] = "skipped-registered"
            continue
        # 模板只允许覆盖 scene_dir/seed/split，其余键（runtime、素材、dataset
        # config 等）由 server 从 studio_config 取默认，与历史任务 argv 一致。
        overrides = {
            "scene_dir": scene_dir,
            "split": "train",
            "seed": overrides.get("seed", 20260826),
        }
        if args.dry_run:
            print(f"  would submit {house} split=train", flush=True)
            done[scene_dir] = "dry-run"
            continue
        result = submit(overrides)
        task = result.get("task", {})
        task_id = task.get("task_id", "?")
        print(f"submitted {house} -> {task_id}", flush=True)
        if args.wait:
            status = wait_finished(task_id)
            house_name = "hm3d_" + "_".join(Path(scene_dir).name.split("-"))
            rooms = rooms_json_for(task_id, scene_dir)
            registered = check_registered(house_name)
            print(
                f"RESULT {house}: status={status} rooms.json={'yes' if rooms else 'NO'} "
                f"registered={registered}",
                flush=True,
            )
            done[scene_dir] = status
        else:
            done[scene_dir] = "submitted"
        if args.house and args.wait:
            break  # single-house verification mode: stop after this one

    print(f"SUMMARY: {len(done)} houses: {sorted(set(done.values()))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
