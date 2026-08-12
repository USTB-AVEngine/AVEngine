from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

LAUNCHER_NAME = "tools/qa/run_strict_two_human_mp3d_f15_probe_v4.py"
LAUNCHER_PATH = next(
    candidate / LAUNCHER_NAME
    for candidate in Path(__file__).resolve().parents
    if (candidate / LAUNCHER_NAME).is_file()
)
TOOLS = LAUNCHER_PATH.parent
sys.path.insert(0, str(TOOLS))


def _load_launcher() -> ModuleType:
    name = "avengine_test_mp3d_f15_v4_launcher"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load_launcher()
BASE = LAUNCHER.base


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _idle_snapshot() -> dict[str, object]:
    return {
        "captured_at_utc": "2026-08-12T00:00:00Z",
        "gpus": [
            {
                "physical_index": 1,
                "uuid": BASE.GPU1_UUID,
                "name": "GPU",
                "memory_used_mib": 19,
                "utilization_percent": 0,
            }
        ],
        "compute_apps": [],
    }


def _write_v3_terminal(atom: Path) -> dict[str, Path]:
    paths = LAUNCHER._v3_attempt_paths(atom)
    capture = paths["capture_root"]
    capture.mkdir(parents=True)
    request = {
        "schema": LAUNCHER.predecessor.REQUEST_SCHEMA,
        "candidate_revision": LAUNCHER.predecessor.CANDIDATE_REVISION,
        "required_repo_commit": "b" * 40,
        "capture_output": str(capture),
        "frame_indices": [BASE.FRAME_INDEX],
        "full75_allowed": False,
        "formal_dataset_count": 0,
    }
    _write(paths["request"], request)
    _write(
        paths["dry_run"],
        {
            "schema": LAUNCHER.predecessor.RECEIPT_SCHEMA,
            "status": "dry_run_pass_not_launched",
            "gpu_started": False,
            "attempt_consumed": False,
            "formal_dataset_count": 0,
        },
    )
    _write(
        paths["running"],
        {
            "schema": LAUNCHER.predecessor.RECEIPT_SCHEMA,
            "status": "running",
            "attempt_consumed": True,
            "formal_dataset_count": 0,
        },
    )
    phase_values = (
        (paths["phase_preconnect"], 0, "preconnect"),
        (paths["phase_post_entry"], 1, "post-entry"),
        (paths["phase_mesh"], 2, "mesh"),
        (paths["phase_lighting"], 3, "lighting"),
        (paths["phase_camera"], 4, "camera"),
    )
    for path, sequence, phase in phase_values:
        _write(
            path,
            {
                "schema": LAUNCHER.CAPTURE_PHASE_SCHEMA,
                "status": "entered",
                "phase": phase,
                "sequence": sequence,
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        )
    failure = {
        "schema": BASE.CAPTURE_FAILURE_SCHEMA,
        "status": "failed",
        "phase": "camera",
        "exception_type": LAUNCHER.V3_EXCEPTION_TYPE,
        "exception_message": LAUNCHER.V3_EXCEPTION_MESSAGE,
        "traceback": (
            "Traceback (most recent call last):\n"
            "  File 'capture.py', in _set_camera_hfov\n"
            "    game.unreal_service.get_component_by_class(...)\n"
            "  File 'unreal_object.py', in __init__\n"
            "    assert self.uobject != 0\n"
            "AssertionError\n"
        ),
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    _write(paths["failure"], failure)
    paths["stdout"].write_text(
        "Game Engine Initialized.\n"
        "LoadMap Load map complete /Engine/Maps/Entry\n"
        "ERROR get_component_by_class\n",
        encoding="utf-8",
    )
    paths["stderr"].write_text(
        "Assertion 'vector.size() == 1' failed (DEBUG)\n", encoding="utf-8"
    )
    _write(
        paths["final"],
        {
            "schema": LAUNCHER.predecessor.RECEIPT_SCHEMA,
            "status": "failed",
            "attempt_consumed": True,
            "retry_same_candidate_forbidden": True,
            "child_exit_code": 1,
            "capture_process_exit_code": 1,
            "failure_observability_status": ("phase_and_complete_traceback_persisted"),
            "capture_observability": {
                "capture_failure_detail": failure,
                "capture_failure_artifact": BASE._file_record(paths["failure"]),
            },
            "formal_dataset_count": 0,
        },
    )
    return paths


def _request_v4(attempt: Path) -> dict[str, object]:
    return {
        "attempt_root": str(attempt),
        "capture_output": str(attempt.parent / LAUNCHER.V4_CAPTURE_DIRECTORY),
        "capture_stdout": str(attempt / "capture_stdout.log"),
        "capture_stderr": str(attempt / "capture_stderr.log"),
        "required_repo_commit": "c" * 40,
        "rpc_port": LAUNCHER.V4_RPC_PORT,
    }


class Mp3dF15V4LauncherTests(unittest.TestCase):
    def test_v3_ledger_freezes_camera_cardinality_and_zero_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
            _write_v3_terminal(atom)
            with mock.patch.object(LAUNCHER, "REPOSITORY", repo):
                ledger_path = LAUNCHER.record_v3_terminal_failure(atom_root=atom)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["status"], LAUNCHER.V3_FAILURE_STATUS)
            self.assertEqual(ledger["root_cause"]["phase"], "camera")
            self.assertEqual(
                ledger["root_cause"]["stderr_assertion"],
                LAUNCHER.V3_STDERR_ASSERTION,
            )
            self.assertEqual(ledger["captured_frame_count"], 0)
            self.assertEqual(ledger["capture_artifact_count"], 6)
            self.assertTrue(
                ledger["prior_phase_closure"][
                    "live_71_mesh_handle_gate_returned_before_lighting"
                ]
            )
            self.assertEqual(
                ledger["prior_phase_closure"]["expected_static_mesh_count"], 71
            )
            self.assertTrue(ledger["attempt_consumed"])
            self.assertTrue(ledger["retry_same_candidate_forbidden"])
            self.assertEqual(ledger["formal_dataset_count"], 0)
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                self.assertRaisesRegex(RuntimeError, "already exists"),
            ):
                LAUNCHER.record_v3_terminal_failure(atom_root=atom)

    def test_v3_ledger_rejects_wrong_failure_and_materialized_frame(self) -> None:
        for mutation in ("wrong_exception", "frame"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                repo = Path(directory).resolve()
                atom = repo / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
                paths = _write_v3_terminal(atom)
                if mutation == "wrong_exception":
                    failure = json.loads(paths["failure"].read_text(encoding="utf-8"))
                    failure["exception_type"] = "RuntimeError"
                    _write(paths["failure"], failure)
                else:
                    (paths["capture_root"] / "rgb_frames").mkdir()
                    (paths["capture_root"] / "rgb_frames/15.png").write_bytes(b"png")
                with (
                    mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "camera failure detail drift|unexpected artifacts",
                    ),
                ):
                    LAUNCHER.record_v3_terminal_failure(atom_root=atom)

    def test_prepare_v4_binds_independent_candidate_and_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
            _write_v3_terminal(atom)
            with mock.patch.object(LAUNCHER, "REPOSITORY", repo):
                ledger = LAUNCHER.record_v3_terminal_failure(atom_root=atom)
            evidence = {}
            for name in (
                "preflight",
                "room_adapter",
                "suite_plan",
                "rir_runtime_probe",
                "package_manifest",
                "package_material_coverage",
                "rir_cache_receipt",
                "rir_cache_index",
            ):
                path = atom / "fake_evidence" / f"{name}.json"
                _write(path, {"name": name})
                evidence[name] = path
            sources = {}
            for name in (
                "capture_script",
                "room_adapter_source",
                "preflight_builder_source",
                "v4_launcher_source",
                "v3_launcher_source",
                "base_launcher_source",
            ):
                path = repo / "fake_sources" / f"{name}.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {name}\n", encoding="utf-8")
                sources[name] = path
            runtime = repo / "runtime/python"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("runtime\n", encoding="utf-8")
            spear = repo / "SPEAR"
            spear.mkdir()
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                mock.patch.object(LAUNCHER, "_require_clean_repository"),
                mock.patch.object(LAUNCHER, "_git_head", return_value="c" * 40),
                mock.patch.object(LAUNCHER, "_v4_source_paths", return_value=sources),
                mock.patch.object(
                    LAUNCHER, "_v4_artifact_paths", return_value=evidence
                ),
                mock.patch.object(BASE, "_validate_cpu_evidence"),
                mock.patch.object(
                    BASE, "_is_authoritative_capture_python", return_value=True
                ),
                mock.patch.object(BASE, "SPEAR_ROOT", spear),
            ):
                request_path = LAUNCHER.prepare_request_v4(
                    atom_root=atom,
                    capture_python=runtime,
                    spear_root=spear,
                    rpc_port=LAUNCHER.V4_RPC_PORT,
                )
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(request["candidate_revision"], LAUNCHER.CANDIDATE_REVISION)
            self.assertEqual(request["frame_indices"], [15])
            self.assertFalse(request["full75_allowed"])
            self.assertEqual(request["formal_dataset_count"], 0)
            self.assertEqual(
                request["predecessor_v3_failure_ledger"], BASE._file_record(ledger)
            )
            change = request["candidate_change_contract"]
            self.assertEqual(
                change["required_named_components"], ["rgb", "depth", "object_ids"]
            )
            self.assertEqual(change["component_class"], "USpSceneCaptureComponent2D")
            self.assertEqual(change["property"], "FOVAngle")
            self.assertFalse(change["actor_wide_u_camera_component_query_allowed"])
            self.assertTrue(change["distinct_live_component_handles_required"])
            self.assertFalse(change["avengine_dot_venv_is_capture_authority"])
            self.assertEqual(
                change["capture_runtime"], str(BASE.CAPTURE_PYTHON_LOGICAL)
            )
            self.assertEqual(request_path.parent.name, LAUNCHER.V4_ATTEMPT_DIRECTORY)

    def test_v4_dry_run_stops_before_child_and_does_not_consume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V4_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v4(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v4", return_value=(request, argv)
                ),
                mock.patch.object(BASE, "_gpu_snapshot", return_value=_idle_snapshot()),
                mock.patch.object(BASE, "_assert_port_available"),
                mock.patch.object(LAUNCHER.subprocess, "run") as child,
            ):
                self.assertEqual(
                    LAUNCHER.run_v4(
                        request_path,
                        dry_run=True,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
            child.assert_not_called()
            receipt = json.loads(
                (attempt / "dry_run_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "dry_run_pass_not_launched")
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])
            self.assertEqual(receipt["formal_dataset_count"], 0)
            self.assertFalse((attempt / "running_receipt.json").exists())
            self.assertFalse((attempt / "capture_stdout.log").exists())
            self.assertFalse((attempt / "capture_stderr.log").exists())

    def test_v4_capture_validation_binds_hfov_to_all_three_pass_handles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            suite = root / "suite.json"
            _write(
                suite,
                {"scenarios": [{"plan": {"camera": {"horizontal_fov_deg": 90.0}}}]},
            )
            identities = [
                {
                    "pass_id": pass_id,
                    "camera_actor_handle": 5000,
                    "rgb_component_handle": 5001,
                    "metric_depth_component_handle": 5002,
                    "object_id_component_handle": 5003,
                }
                for pass_id in (
                    "normal",
                    "source1_target_only",
                    "source2_target_only",
                )
            ]
            manifest = {
                "camera_contract": {
                    "pass_identities": identities,
                    "hfov_readback": {
                        "status": "pass",
                        "camera_actor_handle": 5000,
                        "component_handles": {
                            "rgb": 5001,
                            "depth": 5002,
                            "object_ids": 5003,
                        },
                        "requested_horizontal_fov_deg": 90.0,
                        "observed_horizontal_fov_deg_by_component": {
                            "rgb": 90.0,
                            "depth": 90.0,
                            "object_ids": 90.0,
                        },
                        "write_method": (
                            "named_USpSceneCaptureComponent2D.FOVAngle_property"
                        ),
                    },
                }
            }
            _write(capture / "manifest.json", manifest)
            request = {"capture_output": str(capture), "suite_plan": str(suite)}
            with mock.patch.object(
                BASE, "_validate_capture", return_value={"status": "pass"}
            ):
                validation = LAUNCHER._validate_v4_capture(request)
                self.assertEqual(
                    validation["named_scene_capture_hfov"]["component_handles"],
                    {"rgb": 5001, "depth": 5002, "object_ids": 5003},
                )
                manifest["camera_contract"]["hfov_readback"]["component_handles"][
                    "depth"
                ] = 5001
                _write(capture / "manifest.json", manifest)
                with self.assertRaisesRegex(RuntimeError, "HFOV evidence drift"):
                    LAUNCHER._validate_v4_capture(request)

    def test_v4_real_launch_requires_authorization_before_gpu_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V4_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v4(attempt)
            with (
                mock.patch.object(
                    LAUNCHER,
                    "_validate_request_v4",
                    return_value=(request, ["python", "capture.py"]),
                ),
                mock.patch.object(BASE, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "explicit launch authorization"),
            ):
                LAUNCHER.run_v4(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=False,
                )
            snapshot.assert_not_called()
            self.assertFalse((attempt / "running_receipt.json").exists())

    def test_v4_rejects_preexisting_child_log_before_gpu_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V4_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v4(attempt)
            stdout = Path(str(request["capture_stdout"]))
            stdout.write_text("do not replace\n", encoding="utf-8")
            with (
                mock.patch.object(
                    LAUNCHER,
                    "_validate_request_v4",
                    return_value=(request, ["python", "capture.py"]),
                ),
                mock.patch.object(BASE, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "already exists"),
            ):
                LAUNCHER.run_v4(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=True,
                )
            snapshot.assert_not_called()
            self.assertEqual(stdout.read_text(encoding="utf-8"), "do not replace\n")
            self.assertFalse((attempt / "running_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
