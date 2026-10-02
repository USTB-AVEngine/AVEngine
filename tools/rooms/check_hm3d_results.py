#!/usr/bin/env python3
"""Check final results for the 145 resubmitted hm3d_end_to_end houses.

Waits until no hm3d_end_to_end task is in flight, then for each failed train
house reports: registration status, latest task status, rooms.json presence.
Outputs a CSV summary line per house plus an overall tally.

Usage:
  nohup python3 tools/rooms/check_hm3d_results.py > logs/check_145.log 2>&1 &
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit_hm3d_tasks import (  # noqa: E402
    TASKS_ROOT,
    TEMPLATE,
    check_registered,
    http_json,
    rooms_json_for,
)

TRAIN_ROOT = Path("/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/train")


def all_inflight() -> int:
    data = http_json("/api/tasks")
    return sum(
        1
        for t in data.get("tasks", [])
        if t.get("template") == TEMPLATE and t.get("status") in ("queued", "running")
    )


def main() -> int:
    # 145 栋 = 历史失败任务去重
    scene_dirs: set[Path] = set()
    for task_json in TASKS_ROOT.glob(f"*{TEMPLATE}/task.json"):
        record = json.loads(task_json.read_text(encoding="utf-8"))
        if record.get("status") != "fail":
            continue
        argv = record.get("argv") or []
        for i in range(len(argv) - 1):
            if argv[i] == "--scene-dir" and "/train/" in str(argv[i + 1]):
                scene_dirs.add(Path(argv[i + 1]))
    scene_dirs = sorted(scene_dirs)
    print(f"houses to verify: {len(scene_dirs)}", flush=True)

    # 等所有任务结束(最长 48h 兜底)
    deadline = time.time() + 48 * 3600
    while all_inflight() > 0:
        print(f"  waiting, in-flight={all_inflight()} {time.strftime('%H:%M:%S')}", flush=True)
        if time.time() > deadline:
            print("TIMEOUT waiting for tasks", flush=True)
            break
        time.sleep(300)

    rows: list[dict] = []
    for scene in scene_dirs:
        house = f"hm3d_train_{scene.name.replace('-', '_')}"
        registered = check_registered(house)
        rooms = None
        status = "no-new-task"
        # 找该房子的最新任务(按 task_id 排序取最新)
        latest: tuple[str, str] | None = None
        for task_json in TASKS_ROOT.glob(f"*{TEMPLATE}/task.json"):
            record = json.loads(task_json.read_text(encoding="utf-8"))
            argv = record.get("argv") or []
            hit = any(
                argv[i] == "--scene-dir" and argv[i + 1] == str(scene)
                for i in range(len(argv) - 1)
            )
            if not hit:
                continue
            tid = record.get("task_id", task_json.parent.name)
            st = record.get("status", "?")
            if latest is None or tid > latest[0]:
                latest = (tid, st)
        if latest:
            tid, status = latest
            rooms = rooms_json_for(tid, str(scene))
        rows.append(
            {
                "house": house,
                "registered": registered,
                "status": status,
                "rooms_json": bool(rooms),
            }
        )
        print(
            f"{house} registered={registered} latest_status={status} rooms_json={bool(rooms)}",
            flush=True,
        )

    ok = sum(1 for r in rows if r["registered"] and r["rooms_json"])
    print(f"\nSUMMARY: {ok}/{len(rows)} registered+rooms_json", flush=True)
    for r in rows:
        if not (r["registered"] and r["rooms_json"]):
            print(f"  ISSUE {r['house']}: registered={r['registered']} status={r['status']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
