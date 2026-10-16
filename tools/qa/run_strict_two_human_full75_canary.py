#!/usr/bin/env python3
"""Run one planned full75 canary only after the physical-GPU1 idle gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _nvidia_csv(query: str) -> list[list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query.split(':', 1)[0]}={query.split(':', 1)[1]}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _gpu_snapshot() -> dict[str, Any]:
    gpus = _nvidia_csv("gpu:index,uuid,name")
    apps = _nvidia_csv("compute-apps:gpu_uuid,pid,process_name")
    return {
        "gpus": [
            {"physical_index": int(index), "uuid": uuid, "name": name}
            for index, uuid, name in gpus
        ],
        "compute_apps": [
            {"gpu_uuid": uuid, "pid": int(pid), "process_name": name}
            for uuid, pid, name in apps
        ],
    }


def run(plan_path: Path, canary_index: int, receipt_path: Path, *, dry_run: bool) -> int:
    plan = _load(plan_path)
    _require(
        plan.get("schema") == "avengine_native_strict_two_human_full75_canary_plan_v1",
        "canary plan schema drift",
    )
    matches = [
        item for item in plan["canaries"] if int(item["canary_index"]) == canary_index
    ]
    _require(len(matches) == 1, "canary index must resolve exactly once")
    canary = matches[0]
    gpu_policy = plan["gpu_policy"]
    _require(
        gpu_policy["physical_gpu_index"] == 1
        and gpu_policy["graphics_adapter_argument"] == 1
        and gpu_policy["forbidden_physical_gpu_indices"] == [0, 3],
        "GPU1-only policy drift",
    )
    argv = [str(value) for value in canary["capture_argv"]]
    _require("--frame-index" not in argv, "full75 canary cannot use a sparse frame selector")
    adapter_position = argv.index("--graphics-adapter")
    _require(argv[adapter_position + 1] == "1", "capture adapter must be GPU1")
    before = _gpu_snapshot()
    gpu1 = [item for item in before["gpus"] if item["physical_index"] == 1]
    _require(len(gpu1) == 1, "physical GPU1 did not resolve exactly once")
    gpu1_uuid = gpu1[0]["uuid"]
    gpu1_apps = [
        item for item in before["compute_apps"] if item["gpu_uuid"] == gpu1_uuid
    ]
    _require(
        len(gpu1_apps) == int(gpu_policy["required_idle_compute_process_count"]),
        f"physical GPU1 is not idle: {gpu1_apps}",
    )
    receipt: dict[str, Any] = {
        "schema": "avengine_native_strict_two_human_full75_gpu_launch_receipt_v1",
        "status": "dry_run_pass" if dry_run else "running",
        "canary_index": canary_index,
        "episode_id": canary["episode_id"],
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "forbidden_physical_gpu_indices_used": [],
        "gpu1_uuid": gpu1_uuid,
        "prelaunch_snapshot": before,
        "capture_argv": argv,
        "capture_process_exit_code": None,
    }
    if dry_run:
        _write(receipt_path, receipt)
        return 0
    completed = subprocess.run(argv, check=False)
    receipt["capture_process_exit_code"] = completed.returncode
    receipt["status"] = "pass" if completed.returncode == 0 else "fail"
    _write(receipt_path, receipt)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--canary-index", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(
        args.canary_plan.resolve(),
        args.canary_index,
        args.receipt.resolve(),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
