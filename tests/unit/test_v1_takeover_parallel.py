"""Bounded production-runner concurrency and atomic native reservation tests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from avengine.dataset import production_runner as runner
from avengine.dataset.production_resources import (
    GpuDevice,
    ResourceAllocator,
    ResourcePolicy,
    backend_profiles,
)


_WORKER = r"""
import json
import sys
import time
from pathlib import Path

task_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
marker = sys.argv[3]
task = json.loads(task_path.read_text(encoding="utf-8"))
context = task["context"]
root = Path(context["output_root"]) / "episode"
stage = context["stage"]
time.sleep(0.25)
if marker.startswith("bad"):
    result_path.write_text(json.dumps({
        "schema": "avengine_v1_production_run_v1",
        "status": "error",
        "reason": "synthetic CPU worker failure",
        "reason_code": "worker_failed",
    }), encoding="utf-8")
    raise SystemExit(0)
if stage == "plan":
    plan = root / "plan" / "episode_plan.json"
    request = root / "request.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps(context["work_item"]["inputs"]["request"]), encoding="utf-8")
    plan.write_text(json.dumps({
        "episode_id": context["scope_id"],
        "clock": {
            "clip_seconds": 10.0,
            "duration_seconds": 10.0,
            "frame_count": 150,
            "frame_rate_hz": 15.0,
            "sample_count": 160000,
            "sample_rate_hz": 16000,
            "time_base_hz": 48000,
            "ticks_per_frame": 3200,
        },
        "resources": {"renderer": "habitat"},
        "visual_plan": {"camera": {"motion": "static"}},
    }), encoding="utf-8")
    outcome = {
        "status": "pass",
        "facts": {
            "episode_plan_path": str(plan),
            "renderer": "habitat",
            "clock": json.loads(plan.read_text())["clock"],
        },
        "outputs": {
            "output_root": str(root),
            "plan_root": str(root),
            "episode_plan": str(plan),
            "request_path": str(request),
        },
    }
else:
    capture = root / "capture"
    capture.mkdir(parents=True, exist_ok=True)
    receipt = capture / "research_receipt.json"
    neutral = capture / "neutral_readback.json"
    frames = capture / "frame_records.json"
    video = capture / "ue_visual_only.mp4"
    receipt.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    neutral.write_text(json.dumps({"frames": [{}, {}]}), encoding="utf-8")
    frames.write_text(json.dumps({"frames": [{}, {}]}), encoding="utf-8")
    video.write_bytes(b"cpu-test-video")
    outcome = {
        "status": "pass",
        "facts": {
            "capture_receipt_path": str(receipt),
            "captured_frame_count": 2,
        },
        "outputs": {
            "capture": str(capture),
            "capture_root": str(root),
            "neutral_readback": str(neutral),
            "frame_readbacks": str(frames),
            "visual_video": str(video),
            "native_visual_worlds_created": 0,
        },
    }
result_path.write_text(json.dumps({
    "schema": "avengine_v1_production_run_v1",
    "status": "ok",
    "outcome": outcome,
    "worker": {"pid": __import__("os").getpid()},
}), encoding="utf-8")
"""


def _request(episode_id: str) -> dict:
    return {
        "schema": "avengine_native_qa_room_request_v1",
        "episode_id": episode_id,
        "room_id": "t09_cpu_room",
        "seed": 7,
        "frame_count": 150,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16000,
        "camera": {"motion": "static", "resolution_hw": [720, 1280], "fov_deg": 85.0},
        "entities": {
            "total_count": 2,
            "silent_count": 0,
            "source_classes": ["articulated_human", "articulated_human"],
        },
        "source_asset_ids": ["asset_a", "asset_b"],
        "profile": {
            "anchor_count": 1,
            "reserve_tail_s": 3.0,
            "separation_bin_deg": [30, 60],
        },
        "qa_ids": ["QA-01"],
        "qa_sampling": {"items_per_type": 1},
        "audio_layouts": [{"type": "binaural", "channel_count": 2, "role": "primary"}],
        "post_assembly_convolution_gain": 0.5,
        "production": {
            "retry": {"attempts_per_stage": 1},
            "stage_resources": {
                "plan": {"kind": "cpu", "execution": "cpu", "runtime_context": "pure_python"},
                "capture": {
                    "kind": "gpu_native_visual",
                    "execution": "gpu",
                    "runtime_context": "renderer_native",
                    "min_free_vram_mb": 8192,
                },
            },
        },
    }


def _manifest(tmp_path: Path, ids=("good", "bad")) -> tuple[dict, Path]:
    manifest = {
        "schema": "avengine_qa_batch_manifest_v1",
        "batch_id": "t09_parallel",
        "episodes": [
            {
                "episode_id": episode_id,
                "room_id": "t09_cpu_room",
                "request": _request(episode_id),
            }
            for episode_id in ids
        ],
        "production": {"core_groups": []},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, path


def _broker(tmp_path: Path) -> runner.ResourceBroker:
    devices = (GpuDevice(
        index=0,
        uuid="t09-test-gpu",
        name="t09-test",
        total_memory_mb=16000,
        free_memory_mb=16000,
    ),)
    allocator = ResourceAllocator(
        ResourcePolicy(backends=backend_profiles(None)),
        inventory_reader=lambda: devices,
        process_reader=lambda: (),
        port_probe=lambda port, host: True,
        descendants_reader=lambda pid: (),
    )
    return runner.ResourceBroker(allocator=allocator, repository=tmp_path)


def _room(_request_value):
    return {
        "renderer": "habitat",
        "effective": {"runtime_prefix": "/fake/prefix", "mp3d_root": "/fake/mp3d"},
        "isolation_keys": ("runtime_prefix",),
        "requires_fresh_interpreter": True,
    }


def _launcher_factory(calls, barrier=None):
    lock = threading.Lock()

    def launch(*, plan, work_dir, work_item_id_value):
        task_path = Path(plan["argv"][plan["argv"].index("--execute-work-item") + 1])
        result_path = Path(plan["argv"][plan["argv"].index("--result") + 1])
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if barrier is not None and task["context"]["stage"] == "plan":
            barrier.wait(timeout=3.0)
        marker = work_item_id_value.split(":", 1)[0]
        stdout = work_dir / "stdout.log"
        stderr = work_dir / "stderr.log"
        stdout_handle = stdout.open("x", encoding="utf-8")
        stderr_handle = stderr.open("x", encoding="utf-8")
        started = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(task_path), str(result_path), marker],
            cwd=str(work_dir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
        )
        with lock:
            calls.append((work_item_id_value, started, process.pid))
        returncode = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        return {
            "pid": process.pid,
            "cmdline_marker": work_item_id_value,
            "argv": plan["argv"],
            "cwd": str(work_dir),
            "stdout_log": str(stdout),
            "stderr_log": str(stderr),
            "returncode": returncode,
        }

    return launch


def test_ready_wave_runs_two_real_cpu_children_and_failure_blocks_only_dependents(tmp_path):
    manifest, manifest_path = _manifest(tmp_path)
    calls = []
    runner_instance = runner.ProductionRunner(
        manifest=manifest,
        manifest_path=manifest_path,
        run_root=tmp_path / "run",
        repository=tmp_path,
        max_parallel=2,
        max_waves=2,
        launcher=_launcher_factory(calls, threading.Barrier(2)),
        broker=_broker(tmp_path),
        room_resolver=_room,
        argv=["t09_parallel"],
    )
    started = time.monotonic()
    summary = runner_instance.run()
    elapsed = time.monotonic() - started
    assert len(calls) >= 2
    assert elapsed < 1.5
    journal = [
        json.loads(line)
        for line in (tmp_path / "run/journal.jsonl").read_text().splitlines()
    ]
    ready_ids = [row["work_item_id"] for row in journal if row["event"] == "work_item_ready"]
    assert any(value.startswith("good:capture") for value in ready_ids)
    assert not any(value.startswith("bad:capture") for value in ready_ids)
    assert summary["status"] == "completed_with_diagnostics"


def test_parallel_capture_budget_reservation_allows_only_one_launch(tmp_path):
    manifest, manifest_path = _manifest(tmp_path, ids=("one", "two"))
    calls = []
    runner_instance = runner.ProductionRunner(
        manifest=manifest,
        manifest_path=manifest_path,
        run_root=tmp_path / "run",
        repository=tmp_path,
        native_visual_world_budget=1,
        max_parallel=2,
        launcher=_launcher_factory(calls),
        broker=_broker(tmp_path),
        room_resolver=_room,
        argv=["t09_budget_reservation"],
    )
    scopes = runner_instance.scopes
    items = []
    for scope in scopes:
        items.append(
            {
                "work_item_id": f"{scope.scope_key}:capture:01",
                "stage": "capture",
                "request_id": scope.scope_key,
                "attempt": 1,
                "depends_on": [],
                "fresh_output_relative": f"{scope.scope_key}/capture/attempt_01",
                "resource": {
                    "kind": "cpu",
                    "execution": "cpu",
                    "runtime_context": "pure_python",
                },
                "payload": {"unit_kind": "capture"},
                "inputs": {"request": _request(scope.scope_key)},
                "group_id": None,
                "task_family": None,
                "unit_id": None,
                "member_request_ids": [],
            }
        )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda pair: runner_instance.execute_work_item(*pair),
                zip(scopes, items, strict=True),
            )
        )
    assert sorted(row["status"] for row in results) == ["blocked", "pass"]
    assert len(calls) == 1
    assert runner_instance.native_visual_worlds_used == 1
