from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
import sys
from pathlib import Path
import socket
import threading
import time

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "tools/dataset/run_qa_batch.py"
    spec = importlib.util.spec_from_file_location("qa_batch_runner_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _request(episode_id: str, gpu: int = 0, port: int = 43001) -> dict:
    return {
        "schema": "avengine_native_qa_room_request_v1",
        "episode_id": episode_id,
        "room_id": "test_room",
        "source_asset_ids": ["asset_a", "asset_b"],
        "profile": {"separation_bin_deg": [30, 60], "anchor_count": 1},
        "runtime": {"graphics_adapter": gpu, "rpc_port": port},
        "nested": {"selection": [episode_id, {"fixed": True}]},
    }


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "batch_manifest.json"
    path.write_text(json.dumps({"schema": "avengine_qa_batch_manifest_v1",
                                "batch_id": "test_batch", "episodes": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _row(episode_id: str, *, gpu: int = 0, port: int = 43001, gaps=None) -> dict:
    return {"episode_id": episode_id, "room_id": "test_room",
            "source_assignments": [{"actor_id": "source1", "asset_id": "asset_a"}],
            "requested_profile": {"separation_bin_deg": [30, 60]},
            "preallocation_gaps": list(gaps or []),
            "request": _request(episode_id, gpu=gpu, port=port)}


class FakeProcess:
    next_pid = 12000
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    fail_ids: set[str] = set()
    sleep_s = 0.0

    def __init__(self, command, *, cwd, stdout, stderr, stdin, start_new_session, text):
        del cwd, stderr, stdin, start_new_session, text
        self.command = command
        self.stdout = stdout
        self.output_root = Path(command[command.index("--output") + 1])
        if self.output_root.exists():
            raise AssertionError("controller output must be a fresh path")
        self.returncode = 7 if self.episode_id in self.fail_ids else 0
        self.pid = self.next_pid
        type(self).next_pid += 1

    @property
    def episode_id(self) -> str:
        request = json.loads(Path(self.command[self.command.index("--request") + 1]).read_text())
        return request["episode_id"]

    def wait(self):
        cls = type(self)
        with cls.active_lock:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
        if self.returncode == 0:
            output = self.output_root
            output.mkdir(parents=True)
            (output / "episode_result.json").write_text(
                json.dumps({"status": "research_only", "episode_id": self.episode_id,
                            "delivery": {"status": "captured"}}), encoding="utf-8")
            self.stdout.write(json.dumps({"status": "research_only", "episode_id": self.episode_id}))
            self.stdout.flush()
        time.sleep(cls.sleep_s)
        with cls.active_lock:
            cls.active -= 1
        return self.returncode


def _patch_success(monkeypatch, runner, *, review_status="delivered"):
    monkeypatch.setattr(runner, "_producer_metadata",
                        lambda manifest_path, repository: {"manifest_path": str(manifest_path),
                                                           "repository": str(repository),
                                                           "cwd": str(repository),
                                                           "git_commit": "test-head",
                                                           "python_executable": "test-python"})
    monkeypatch.setattr(runner, "_check_resources",
                        lambda request, min_free_gpu_mb, require_rpc_port=True: {"gpu": {"free_memory_mb": 32000},
                                                          "rpc_port": None})
    monkeypatch.setattr(runner, "_finalize_delivery",
                        lambda attempt_root, entry, repository: {"status": review_status,
                                                                   "review_root": str(attempt_root / "review")})
    monkeypatch.setattr(runner, "_finalize_batch_outputs",
                        lambda output_root, manifest, execution_summary, repository: {
                            "status": "machine_artifacts_complete",
                            "summary_root": str(output_root / "summary"),
                            "five_clip_listening": str(output_root / "summary/five_clip_listening_pending.json"),
                            "coverage_outputs": {"coverage": str(output_root / "summary/coverage")},
                        })
    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    FakeProcess.fail_ids = set()
    FakeProcess.sleep_s = 0.0
    FakeProcess.active = 0
    FakeProcess.max_active = 0


def test_exact_saved_request_and_durable_progress(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    request = _request("episode_exact", gpu=0, port=43011)
    row = _row("episode_exact")
    row["request"] = request
    manifest = _manifest(tmp_path, [row])
    output = tmp_path / "execution"

    summary = runner.execute_batch(manifest, output, max_parallel=1, min_free_gpu_mb=0)

    saved = json.loads((output / "episodes/episode_exact/request.json").read_text())
    assert saved == request
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["counts"]["delivered"] == 1
    assert progress["episodes"]["episode_exact"]["status"] == "delivered"
    assert json.loads((output / "producer.json").read_text())["python_executable"]
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert {event["event"] for event in events} >= {"queued", "launching", "started", "completed", "batch_complete"}
    assert summary["outcome_counts"] == {"delivered": 1}
    assert summary["aggregate_status"] == "complete"
    assert summary["batch_summary_paths"]["summary_root"].endswith("/summary")
    final_progress = json.loads((output / "progress.json").read_text())
    assert final_progress["aggregate_status"] == "complete"
    assert final_progress["batch_summary_paths"]["five_clip_listening"].endswith("five_clip_listening_pending.json")
    outcome = json.loads((output / "episodes/episode_exact/attempt_01/outcome.json").read_text())
    assert outcome["command"][outcome["command"].index("--request") + 1].endswith("/episode_exact/request.json")
    assert outcome["command"][outcome["command"].index("--output") + 1].endswith("/episode_exact/attempt_01/episode")
    assert outcome["episode_output_root"].endswith("/episode_exact/attempt_01/episode")
    assert outcome["captured_delivery_status"] == "captured"


def test_output_root_is_no_clobber(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    manifest = _manifest(tmp_path, [_row("episode_existing")])
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing existing output"):
        runner.execute_batch(manifest, output, min_free_gpu_mb=0)


def test_independent_controller_failure_does_not_cancel_siblings(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    FakeProcess.fail_ids = {"episode_bad"}
    manifest = _manifest(tmp_path, [_row("episode_bad", port=43012), _row("episode_good", port=43013)])

    summary = runner.execute_batch(manifest, tmp_path / "execution", max_parallel=2, min_free_gpu_mb=0)

    assert summary["status"] == "completed_with_diagnostics"
    assert summary["outcome_counts"] == {"failed": 1, "delivered": 1}
    assert (tmp_path / "execution/episodes/episode_good/attempt_01/outcome.json").is_file()
    bad = json.loads((tmp_path / "execution/episodes/episode_bad/attempt_01/outcome.json").read_text())
    assert bad["status"] == "failed"
    assert bad["controller_returncode"] == 7


def test_known_preallocation_gap_is_blocked_without_controller(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    called = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: called.append(args) or FakeProcess(*args, **kwargs))
    manifest = _manifest(tmp_path, [_row("episode_gap", gaps=[{"code": "no_sound_identity"}])])

    summary = runner.execute_batch(manifest, tmp_path / "execution", min_free_gpu_mb=0)

    assert not called
    assert summary["outcome_counts"] == {"blocked": 1}
    outcome = json.loads((tmp_path / "execution/episodes/episode_gap/attempt_01/outcome.json").read_text())
    assert outcome["status"] == "blocked"
    assert outcome["reason_code"] == "preallocation_gap"
    assert outcome["failure_stage"] == "planning"
    assert outcome["gap_state"] == "evidence_missing_or_unsampled"


def test_per_gpu_lock_serializes_same_gpu(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    FakeProcess.sleep_s = 0.05
    manifest = _manifest(tmp_path, [_row("episode_a", gpu=1, port=43021), _row("episode_b", gpu=1, port=43022)])

    summary = runner.execute_batch(manifest, tmp_path / "execution", max_parallel=2, min_free_gpu_mb=0)

    assert summary["status"] == "complete"
    assert FakeProcess.max_active == 1


def test_resource_failure_without_gpu_is_explicit_and_independent(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    called = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: called.append(args) or FakeProcess(*args, **kwargs))
    monkeypatch.setattr(runner, "_check_resources",
                        lambda request, min_free_gpu_mb, require_rpc_port=True: (_ for _ in ()).throw(
                            runner.ResourceBlocked("nvidia_smi_unavailable", "nvidia-smi unavailable")))
    manifest = _manifest(tmp_path, [_row("episode_no_gpu")])

    summary = runner.execute_batch(manifest, tmp_path / "execution", min_free_gpu_mb=8192)

    assert not called
    assert summary["outcome_counts"] == {"blocked": 1}
    outcome = json.loads((tmp_path / "execution/episodes/episode_no_gpu/attempt_01/outcome.json").read_text())
    assert outcome["reason_code"] == "nvidia_smi_unavailable"


def test_review_failure_preserves_captured_delivery(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner, review_status="review_failed")
    manifest = _manifest(tmp_path, [_row("episode_review_failed")])

    summary = runner.execute_batch(manifest, tmp_path / "execution", min_free_gpu_mb=0)

    assert summary["outcome_counts"] == {"review_failed": 1}
    outcome = json.loads((tmp_path / "execution/episodes/episode_review_failed/attempt_01/outcome.json").read_text())
    assert outcome["status"] == "review_failed"
    assert outcome["review_status"] == "review_failed"
    assert outcome["captured_delivery_status"] == "captured"


def test_episode_id_selects_only_original_rows(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    manifest = _manifest(tmp_path, [_row("episode_a", port=43031), _row("episode_b", port=43032)])

    summary = runner.execute_batch(manifest, tmp_path / "execution", episode_ids=["episode_b"], min_free_gpu_mb=0)

    assert summary["selected_episode_ids"] == ["episode_b"]
    assert (tmp_path / "execution/episodes/episode_b").is_dir()
    assert not (tmp_path / "execution/episodes/episode_a").exists()


def test_gpu_and_rpc_probe_are_narrow_and_fail_closed(monkeypatch, runner):
    monkeypatch.setattr(runner, "_query_free_gpu",
                        lambda gpu: {"gpu": gpu, "free_memory_mb": 20000})
    request = _request("episode_probe", gpu=2, port=43091)
    checked = runner._check_resources(request, min_free_gpu_mb=8192, require_rpc_port=True)
    assert checked["gpu"]["gpu"] == 2
    assert checked["rpc_port"]["probe"] == "bind_then_release"

    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    try:
        blocked_request = _request("episode_probe_blocked", gpu=2, port=held.getsockname()[1])
        with pytest.raises(runner.ResourceBlocked, match="unavailable"):
            runner._check_resources(blocked_request, min_free_gpu_mb=8192, require_rpc_port=True)
    finally:
        held.close()


def test_initial_progress_counts_all_jobs_as_queued(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    rows = [_row("episode_queued_a", port=43101), _row("episode_queued_b", port=43102)]
    manifest = _manifest(tmp_path, rows)
    loaded, selected = runner.load_manifest(manifest)
    output = tmp_path / "queued_snapshot"
    jobs = runner._build_jobs(manifest, loaded, selected, output)
    executor = runner.BatchExecutor(manifest_path=manifest, manifest=loaded, jobs=jobs,
                                    output_root=output, max_parallel=2, min_free_gpu_mb=0)
    executor._prepare()
    progress = json.loads((output / "progress.json").read_text())
    assert progress["counts"]["queued"] == 2
    assert progress["counts"]["running"] == 0


def test_internal_executor_error_preserves_existing_logs(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    manifest = _manifest(tmp_path, [_row("episode_internal_error")])

    def explode(self, job):
        (job.attempt_root / "stdout.log").write_text("stdout before error\n", encoding="utf-8")
        (job.attempt_root / "stderr.log").write_text("stderr before error\n", encoding="utf-8")
        raise RuntimeError("synthetic executor error")

    monkeypatch.setattr(runner.BatchExecutor, "_run_job", explode)
    summary = runner.execute_batch(manifest, tmp_path / "execution", min_free_gpu_mb=0)

    root = tmp_path / "execution/episodes/episode_internal_error/attempt_01"
    assert summary["outcome_counts"] == {"blocked": 1}
    assert (root / "stdout.log").read_text() == "stdout before error\n"
    assert (root / "stderr.log").read_text() == "stderr before error\n"
    assert "executor_internal_error" in (root / "diagnostic.log").read_text()


def test_aggregate_failure_preserves_all_episode_outcomes(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)
    monkeypatch.setattr(runner, "_finalize_batch_outputs",
                        lambda output_root, manifest, execution_summary, repository: (_ for _ in ()).throw(
                            RuntimeError("synthetic aggregate failure")))
    manifest = _manifest(tmp_path, [_row("episode_aggregate_failure")])

    summary = runner.execute_batch(manifest, tmp_path / "execution", min_free_gpu_mb=0)

    output = tmp_path / "execution"
    assert summary["status"] == "aggregate_failed"
    assert summary["operational_status"] == "complete"
    assert summary["aggregate_status"] == "aggregate_failed"
    assert summary["outcome_counts"] == {"delivered": 1}
    assert summary["episodes"][0]["status"] == "delivered"
    error_path = output / "aggregate_error.log"
    assert error_path.is_file()
    assert "synthetic aggregate failure" in error_path.read_text()
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "aggregate_failed"
    assert progress["episodes"]["episode_aggregate_failure"]["status"] == "delivered"

def _failing_process(writer):
    class Process(FakeProcess):
        def wait(self):
            self.output_root.mkdir(parents=True, exist_ok=True)
            writer(self.output_root)
            self.returncode = 1
            return 1
    return Process


def test_failure_accounting_planning_exhausted(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)

    def write(root: Path) -> None:
        (root / "planning_result.json").write_text(json.dumps({
            "status": "failed",
            "failure_histogram": {
                "camera:no_joint_geometry_activity_schedule": 169,
                "routes:initial_source_separation_below_0.95_m": 31,
            },
            "gap_category": "evidence_missing_or_unsampled",
        }), encoding="utf-8")

    monkeypatch.setattr(runner.subprocess, "Popen", _failing_process(write))
    summary = runner.execute_batch(_manifest(tmp_path, [_row("episode_plan_fail")]),
                                   tmp_path / "execution", min_free_gpu_mb=0)
    outcome = json.loads((tmp_path / "execution/episodes/episode_plan_fail/attempt_01/outcome.json").read_text())
    assert summary["outcome_counts"] == {"failed": 1}
    assert outcome["failure_stage"] == "planning"
    assert outcome["gap_state"] == "evidence_missing_or_unsampled"
    assert "169" in outcome["failure_reason"]
    assert "fixed condition profile exhausted" in outcome["failure_reason"]


def test_failure_accounting_audio_interface_defect(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)

    def write(root: Path) -> None:
        (root / "execution_commands.json").write_text("{}", encoding="utf-8")
        (root / "capture").mkdir()
        (root / "capture" / "neutral_readback.json").write_text("{}", encoding="utf-8")
        delivery = root / "delivery"
        delivery.mkdir()
        (delivery / "audio.log").write_text(json.dumps({
            "status": "fail",
            "error": "AudioProgram validation failed: sequential_sources events must not overlap",
        }), encoding="utf-8")

    monkeypatch.setattr(runner.subprocess, "Popen", _failing_process(write))
    runner.execute_batch(_manifest(tmp_path, [_row("episode_audio_fail")]),
                         tmp_path / "execution", min_free_gpu_mb=0)
    outcome = json.loads((tmp_path / "execution/episodes/episode_audio_fail/attempt_01/outcome.json").read_text())
    assert outcome["failure_stage"] == "audio"
    assert outcome["gap_state"] == "interface_not_implemented"
    assert "AudioProgram validation failed" in outcome["failure_reason"]


def test_failure_accounting_capture_or_finalize(monkeypatch, runner, tmp_path):
    _patch_success(monkeypatch, runner)

    def write(root: Path) -> None:
        (root / "execution_commands.json").write_text("{}", encoding="utf-8")
        (root / "capture.log").write_text(
            "CalledProcessError: capture renderer exited with 2\n", encoding="utf-8")

    monkeypatch.setattr(runner.subprocess, "Popen", _failing_process(write))
    runner.execute_batch(_manifest(tmp_path, [_row("episode_capture_fail")]),
                         tmp_path / "execution", min_free_gpu_mb=0)
    outcome = json.loads((tmp_path / "execution/episodes/episode_capture_fail/attempt_01/outcome.json").read_text())
    assert outcome["failure_stage"] == "capture"
    assert outcome["gap_state"] == "interface_not_implemented"
    assert "CalledProcessError" in outcome["failure_reason"]

