from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

LAUNCHER_NAME = "tools/qa/run_strict_two_human_mp3d_f15_probe.py"
LAUNCHER_PATH = next(
    candidate / LAUNCHER_NAME
    for candidate in Path(__file__).resolve().parents
    if (candidate / LAUNCHER_NAME).is_file()
)


def _load_launcher() -> ModuleType:
    name = "avengine_test_mp3d_f15_launcher"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load_launcher()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _idle_snapshot() -> dict[str, object]:
    return {
        "captured_at_utc": "2026-08-12T00:00:00Z",
        "gpus": [
            {
                "physical_index": 1,
                "uuid": LAUNCHER.GPU1_UUID,
                "name": "GPU",
                "memory_used_mib": 19,
                "utilization_percent": 0,
            }
        ],
        "compute_apps": [],
    }


def _request(attempt_root: Path) -> dict[str, object]:
    return {
        "attempt_root": str(attempt_root),
        "capture_output": str(attempt_root.parent / "diagnostic_f15_capture_attempt_01"),
        "required_repo_commit": "a" * 40,
        "rpc_port": 39631,
    }


class Mp3dF15LauncherTests(unittest.TestCase):
    def test_capture_argv_is_exactly_one_f15_on_adapter1(self) -> None:
        request = {
            "capture_python": "/runtime/python",
            "capture_script": "/repo/capture.py",
            "suite_plan": "/evidence/suite.json",
            "room_adapter": "/evidence/room.json",
            "spear_root": "/runtime/SPEAR",
            "capture_output": "/evidence/capture",
            "rpc_port": 39631,
        }
        argv = LAUNCHER._capture_argv(request)
        self.assertEqual(argv.count("--frame-index"), 1)
        self.assertEqual(argv[argv.index("--frame-index") + 1], "15")
        self.assertEqual(argv.count("--graphics-adapter"), 1)
        self.assertEqual(argv[argv.index("--graphics-adapter") + 1], "1")

    def test_gpu_gate_rejects_uuid_drift_and_busy_gpu1(self) -> None:
        snapshot = _idle_snapshot()
        self.assertEqual(
            LAUNCHER._validate_gpu1_idle(snapshot)["uuid"], LAUNCHER.GPU1_UUID
        )
        wrong = _idle_snapshot()
        wrong["gpus"][0]["uuid"] = "GPU-wrong"
        with self.assertRaisesRegex(RuntimeError, "UUID drift"):
            LAUNCHER._validate_gpu1_idle(wrong)
        busy = _idle_snapshot()
        busy["compute_apps"] = [
            {"gpu_uuid": LAUNCHER.GPU1_UUID, "pid": 7, "process_name": "python"}
        ]
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            LAUNCHER._validate_gpu1_idle(busy)

    def test_artifact_binding_detects_single_byte_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text("{}\n", encoding="utf-8")
            record = LAUNCHER._file_record(path)
            self.assertEqual(
                LAUNCHER._validate_file_record(record, owner="evidence"),
                path.resolve(),
            )
            path.write_text("{ }\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "binding drift"):
                LAUNCHER._validate_file_record(record, owner="evidence")

    def test_capture_python_symlink_resolves_to_only_pinned_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "python3.11"
            real.write_text("runtime", encoding="utf-8")
            logical = root / "python"
            logical.symlink_to(real.name)
            wrong = root / "other-python"
            wrong.write_text("wrong", encoding="utf-8")
            with mock.patch.object(
                LAUNCHER, "CAPTURE_PYTHON_LOGICAL", logical
            ):
                self.assertTrue(LAUNCHER._is_authoritative_capture_python(logical))
                self.assertTrue(LAUNCHER._is_authoritative_capture_python(real))
                self.assertFalse(LAUNCHER._is_authoritative_capture_python(wrong))

    def test_prepare_failure_archive_preserves_request_without_consuming_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
            attempt = atom / "diagnostic_f15_launch_attempt_01"
            _write(attempt / "request.json", {"required_repo_commit": "a" * 40})
            with mock.patch.object(LAUNCHER, "REPOSITORY", repo):
                receipt_path = LAUNCHER.archive_preparation_failure(
                    atom_root=atom,
                    error="canonical interpreter symlink mismatch",
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])
            self.assertTrue(receipt_path.with_name("request.json").is_file())
            self.assertFalse(attempt.exists())

    def test_dry_run_writes_only_non_consuming_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "diagnostic_f15_launch_attempt_01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER, "_gpu_snapshot", return_value=_idle_snapshot()
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available"),
            ):
                self.assertEqual(LAUNCHER.run(request_path, dry_run=True), 0)
            receipt = json.loads(
                (attempt / "dry_run_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "dry_run_pass_not_launched")
            self.assertEqual(receipt["frame_indices"], [15])
            self.assertFalse(receipt["full75_allowed"])
            self.assertFalse((attempt / "running_receipt.json").exists())
            self.assertFalse((attempt / "final_receipt.json").exists())

    def test_real_attempt_has_immutable_running_and_separate_final_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "diagnostic_f15_launch_attempt_01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            validation = {
                "status": "pass_diagnostic_f15_review_ready",
                "qualification_claim": False,
                "formal_dataset_count": 0,
            }
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER,
                    "_gpu_snapshot",
                    side_effect=[_idle_snapshot(), _idle_snapshot()],
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available"),
                mock.patch.object(
                    LAUNCHER,
                    "subprocess",
                    wraps=LAUNCHER.subprocess,
                ) as subprocess_module,
                mock.patch.object(
                    LAUNCHER, "_validate_capture", return_value=validation
                ),
            ):
                subprocess_module.run.return_value = SimpleNamespace(returncode=0)
                self.assertEqual(LAUNCHER.run(request_path, dry_run=False), 0)
            running = json.loads(
                (attempt / "running_receipt.json").read_text(encoding="utf-8")
            )
            final = json.loads(
                (attempt / "final_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(running["status"], "running")
            self.assertIsNone(running["capture_process_exit_code"])
            self.assertEqual(final["status"], "pass_diagnostic_f15_review_ready")
            self.assertEqual(final["capture_process_exit_code"], 0)
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request", return_value=(request, argv)
                ),
                self.assertRaisesRegex(RuntimeError, "final receipt"),
            ):
                LAUNCHER.run(request_path, dry_run=False)


if __name__ == "__main__":
    unittest.main()
