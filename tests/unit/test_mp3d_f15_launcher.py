from __future__ import annotations

import importlib.util
import inspect
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
CAPTURE_PATH = LAUNCHER_PATH.with_name(
    "capture_spear_imported_glb_strict_two_human_episode.py"
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


def _load_capture() -> ModuleType:
    name = "avengine_test_mp3d_f15_capture"
    spec = importlib.util.spec_from_file_location(name, CAPTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_capture()


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
        "capture_output": str(
            attempt_root.parent / "diagnostic_f15_capture_attempt_01"
        ),
        "required_repo_commit": "a" * 40,
        "rpc_port": 39631,
    }


def _request_v2(attempt_root: Path) -> dict[str, object]:
    return {
        "attempt_root": str(attempt_root),
        "capture_output": str(attempt_root.parent / LAUNCHER.V2_CAPTURE_DIRECTORY),
        "capture_stdout": str(attempt_root / "capture_stdout.log"),
        "capture_stderr": str(attempt_root / "capture_stderr.log"),
        "required_repo_commit": "b" * 40,
        "rpc_port": LAUNCHER.V2_RPC_PORT,
    }


class Mp3dF15LauncherTests(unittest.TestCase):
    def test_capture_argv_is_exactly_one_f15_on_adapter1(self) -> None:
        request = {
            "episode_id": "dynamic_episode_0002",
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
        self.assertEqual(argv[argv.index("--scenario-id") + 1], "dynamic_episode_0002")

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

    def test_artifact_binding_is_path_only_and_ignores_legacy_digest_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text("{}\n", encoding="utf-8")
            record = LAUNCHER._file_record(path)
            self.assertEqual(record, {"path": str(path.resolve())})
            self.assertEqual(
                LAUNCHER._validate_file_record(record, owner="evidence"),
                path.resolve(),
            )
            path.write_text("{ }\n", encoding="utf-8")
            legacy_record = {**record, "legacy_metadata": "ignored"}
            self.assertEqual(
                LAUNCHER._validate_file_record(legacy_record, owner="evidence"),
                path.resolve(),
            )

    def test_execution_plan_resolver_rejects_suite_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            preflight = repo / "tmp/atom/cpu_preflight_v3"
            preflight.mkdir(parents=True)
            plan_path = preflight / "execution_plan.json"
            atom = preflight.parent
            cpu_steps = [
                {
                    "step_id": "probe_authoritative_habitat_rir_runtime",
                    "expected": {"receipt": str(preflight / "runtime.json")},
                },
                {
                    "step_id": "fresh_compile_mp3d_rlr_materials",
                    "expected": {
                        "manifest": str(atom / "package/manifest.json"),
                        "semantic_material_coverage": str(
                            atom / "package/coverage.json"
                        ),
                    },
                },
                {
                    "step_id": "render_two_exact_rirs",
                    "expected": {
                        "receipt": str(atom / "cache/receipt.json"),
                        "index": str(atom / "cache/index.json"),
                    },
                },
            ]
            sparse_argv = [
                "python",
                "capture.py",
                "--suite-plan",
                str(repo.parent / "escaped_suite.json"),
                "--room-adapter",
                str(preflight / "room_adapter.json"),
                "--output",
                str(atom / "capture"),
                "--frame-index",
                "15",
                "--graphics-adapter",
                "1",
            ]
            _write(
                plan_path,
                {
                    "schema": "avengine_native_strict_two_human_mp3d_execution_plan_v2",
                    "qualification_claim": False,
                    "formal_dataset_count": 0,
                    "local_staging_output": str(preflight),
                    "remote_target_root": str(atom),
                    "cpu_steps": cpu_steps,
                    "gpu_steps": [
                        {"step_id": "sparse_f15_probe", "argv": sparse_argv},
                        {"step_id": "full75_episode", "argv": []},
                    ],
                },
            )
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                self.assertRaisesRegex(RuntimeError, "suite plan escapes"),
            ):
                LAUNCHER._execution_plan_artifact_paths(plan_path)

    def test_execution_plan_package_id_mismatch_fails_closed(self) -> None:
        plan = {
            "cpu_steps": [
                {
                    "step_id": "fresh_compile_mp3d_rlr_materials",
                    "argv": ["compile", "--package-id", "package_from_plan"],
                }
            ]
        }
        package = {
            "schema": "avengine_acoustic_scene_package_v1",
            "package_id": "different_package",
            "package_mode": "research_candidate",
            "room_kind": "habitat_native",
            "geometry": {"triangle_count": 10, "vertex_count": 9},
        }
        coverage = {
            "schema": "avengine_m3_rlr_semantic_material_coverage_v1",
            "status": "research_candidate",
            "qualification_claim": False,
            "compiled_triangle_count": 10,
            "triangle_coverage": {"triangle_count": 10},
            "runtime_one_to_one": {"passed": True},
        }
        with self.assertRaisesRegex(RuntimeError, "fresh acoustic package drift"):
            LAUNCHER._validate_execution_plan_package(plan, package, coverage)

    def test_v5_prepare_emits_path_only_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = root / "atom/cpu_preflight_v3/execution_plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("{}\n", encoding="utf-8")
            validation = {
                "episode_id": "episode_0002",
                "scene_id": "scene_dynamic",
                "execution_plan": str(plan),
                "evidence_paths": {
                    "preflight": str(plan.with_name("preflight.json")),
                    "suite_plan": str(plan.with_name("suite_execution_plan.json")),
                    "room_adapter": str(plan.with_name("room_adapter.json")),
                },
                "capture_output": str(root / "atom/native_sparse_f15_v1"),
                "capture_argv": ["python", "capture.py", "--frame-index", "15"],
            }
            with (
                mock.patch.object(
                    LAUNCHER,
                    "offline_validate_execution_plan",
                    return_value=validation,
                ),
                mock.patch.object(LAUNCHER, "_git_head", return_value="c" * 40),
            ):
                request_path = LAUNCHER.prepare_request_v5(execution_plan_path=plan)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(request["schema"], LAUNCHER.REQUEST_SCHEMA_V5)
            self.assertEqual(request["scene_id"], "scene_dynamic")
            self.assertIn("evidence_paths", request)
            self.assertEqual(
                request["suite_plan"], validation["evidence_paths"]["suite_plan"]
            )
            self.assertEqual(
                request["room_adapter"], validation["evidence_paths"]["room_adapter"]
            )

    def test_v5_offline_validate_and_dry_run_never_query_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V5_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = {
                "attempt_root": str(attempt),
                "episode_id": "episode_0002",
                "scene_id": "scene_dynamic",
                "required_repo_commit": "d" * 40,
                "execution_plan": "/evidence/execution_plan.json",
                "capture_output": str(attempt.parent / "capture"),
            }
            argv = ["python", "capture.py", "--rpc-port", "39631"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v5", return_value=(request, argv)
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as snapshot,
            ):
                self.assertEqual(
                    LAUNCHER.run_v5(
                        request_path,
                        offline_validate=True,
                        dry_run=False,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
                self.assertEqual(
                    LAUNCHER.run_v5(
                        request_path,
                        offline_validate=False,
                        dry_run=True,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
            snapshot.assert_not_called()
            receipt = json.loads(
                (attempt / "dry_run_receipt.json").read_text(encoding="utf-8")
            )
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["gpu_started"])

    def test_capture_python_symlink_resolves_to_only_pinned_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "python3.11"
            real.write_text("runtime", encoding="utf-8")
            logical = root / "python"
            logical.symlink_to(real.name)
            wrong = root / "other-python"
            wrong.write_text("wrong", encoding="utf-8")
            with mock.patch.object(LAUNCHER, "CAPTURE_PYTHON_LOGICAL", logical):
                self.assertTrue(LAUNCHER._is_authoritative_capture_python(logical))
                self.assertTrue(LAUNCHER._is_authoritative_capture_python(real))
                self.assertFalse(LAUNCHER._is_authoritative_capture_python(wrong))

    def test_prepare_failure_archive_preserves_request_without_consuming_attempt(
        self,
    ) -> None:
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

    def test_real_attempt_has_immutable_running_and_separate_final_receipt(
        self,
    ) -> None:
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

    def test_attempt01_failure_ledger_freezes_only_observed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
            attempt = atom / "diagnostic_f15_launch_attempt_01"
            capture = atom / "diagnostic_f15_capture_attempt_01"
            capture.mkdir(parents=True)
            _write(
                attempt / "request.json",
                {
                    "schema": LAUNCHER.REQUEST_SCHEMA,
                    "capture_output": str(capture),
                },
            )
            _write(
                attempt / "dry_run_receipt.json",
                {
                    "schema": LAUNCHER.RECEIPT_SCHEMA,
                    "status": "dry_run_pass_not_launched",
                },
            )
            _write(
                attempt / "running_receipt.json",
                {"schema": LAUNCHER.RECEIPT_SCHEMA, "status": "running"},
            )
            _write(
                attempt / "final_receipt.json",
                {
                    "schema": LAUNCHER.RECEIPT_SCHEMA,
                    "status": "failed",
                    "capture_process_exit_code": 1,
                },
            )
            spear_log = repo / "SpearSim_rpc_39631.log"
            spear_log.write_text(
                "LogInit: Display: Game Engine Initialized.\n"
                "LogGlobalStatus: LoadMap Load map complete /Engine/Maps/Entry\n"
                "LogInit: Display: Engine is initialized. "
                "Leaving FEngineLoop::Init()\n",
                encoding="utf-8",
            )
            with mock.patch.object(LAUNCHER, "REPOSITORY", repo):
                ledger_path = LAUNCHER.record_attempt01_failure_ledger(
                    atom_root=atom,
                    spear_log=spear_log,
                )
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["status"], LAUNCHER.ATTEMPT01_FAILURE_STATUS)
            self.assertEqual(ledger["root_cause"], "undetermined")
            self.assertTrue(ledger["attempt_consumed"])
            self.assertTrue(ledger["retry_same_candidate_forbidden"])
            self.assertEqual(ledger["captured_frame_count"], 0)
            self.assertEqual(ledger["capture_artifact_count"], 0)
            self.assertEqual(ledger["first_capture_artifact_count"], 0)
            self.assertFalse(ledger["causal_exclusions"]["mesh_failure_claimed"])
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                self.assertRaises(FileExistsError),
            ):
                LAUNCHER.record_attempt01_failure_ledger(
                    atom_root=atom,
                    spear_log=spear_log,
                )

    def test_capture_failure_journal_covers_all_runtime_phases(self) -> None:
        phases = (
            "preconnect",
            "post-entry",
            "mesh",
            "lighting",
            "camera",
            "actor",
            "capture",
        )
        for phase in phases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "capture"
                output.mkdir()

                def fail_at_phase(
                    _args: SimpleNamespace,
                    journal: object,
                    *,
                    selected_phase: str = phase,
                ) -> Path:
                    journal.enter(selected_phase)
                    raise RuntimeError(f"fake runtime failure at {selected_phase}")

                with (
                    mock.patch.object(CAPTURE, "_run_impl", side_effect=fail_at_phase),
                    self.assertRaisesRegex(RuntimeError, phase),
                ):
                    CAPTURE.run(SimpleNamespace(output=output))
                failure = json.loads(
                    (output / "capture_failure.json").read_text(encoding="utf-8")
                )
                self.assertEqual(failure["phase"], phase)
                self.assertEqual(failure["exception_type"], "RuntimeError")
                self.assertIn(
                    f"RuntimeError: fake runtime failure at {phase}",
                    failure["traceback"],
                )
                markers = list(output.glob(f"capture_phase_*_{phase}.json"))
                self.assertEqual(len(markers), 1)

    def test_capture_runtime_wires_required_phases_in_order(self) -> None:
        source = inspect.getsource(CAPTURE._run_impl)
        phases = (
            "preconnect",
            "post-entry",
            "mesh",
            "lighting",
            "camera",
            "actor",
            "capture",
        )
        offsets = [source.index(f'journal.enter("{phase}")') for phase in phases]
        self.assertEqual(offsets, sorted(offsets))

    def test_revision_v2_dry_run_writes_no_child_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v2(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER, "_gpu_snapshot", return_value=_idle_snapshot()
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available"),
            ):
                self.assertEqual(
                    LAUNCHER.run_v2(
                        request_path,
                        dry_run=True,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
            receipt = json.loads(
                (attempt / "dry_run_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "dry_run_pass_not_launched")
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])
            self.assertFalse((attempt / "capture_stdout.log").exists())
            self.assertFalse((attempt / "capture_stderr.log").exists())

    def test_revision_v2_real_launch_requires_explicit_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v2(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "explicit launch authorization"),
            ):
                LAUNCHER.run_v2(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=False,
                )
            snapshot.assert_not_called()
            self.assertFalse((attempt / "running_receipt.json").exists())

    def test_revision_v2_rejects_preexisting_child_log_before_gpu_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v2(attempt)
            Path(str(request["capture_stdout"])).write_text(
                "do not replace\n", encoding="utf-8"
            )
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "already exists"),
            ):
                LAUNCHER.run_v2(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=True,
                )
            snapshot.assert_not_called()
            self.assertEqual(
                Path(str(request["capture_stdout"])).read_text(encoding="utf-8"),
                "do not replace\n",
            )
            self.assertFalse((attempt / "running_receipt.json").exists())

    def test_revision_v2_failure_persists_observability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / LAUNCHER.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v2(attempt)
            capture_output = Path(str(request["capture_output"]))
            argv = ["python", "capture.py", "--frame-index", "15"]

            def fake_child(*_args: object, **kwargs: object) -> SimpleNamespace:
                stdout = kwargs["stdout"]
                stderr = kwargs["stderr"]
                stdout.write(b"exclusive child stdout\n")
                stderr.write(b"Traceback: exclusive child stderr\n")
                capture_output.mkdir()
                _write(
                    capture_output / "capture_phase_00_mesh.json",
                    {
                        "schema": "avengine_mp3d_f15_capture_phase_v1",
                        "status": "entered",
                        "phase": "mesh",
                        "sequence": 0,
                        "qualification_claim": False,
                        "formal_dataset_count": 0,
                    },
                )
                _write(
                    capture_output / "capture_failure.json",
                    {
                        "schema": LAUNCHER.CAPTURE_FAILURE_SCHEMA,
                        "status": "failed",
                        "phase": "mesh",
                        "exception_type": "RuntimeError",
                        "exception_message": "fake mesh failure",
                        "traceback": (
                            "Traceback (most recent call last):\n"
                            "RuntimeError: fake mesh failure\n"
                        ),
                        "qualification_claim": False,
                        "formal_dataset_count": 0,
                    },
                )
                return SimpleNamespace(returncode=23)

            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER,
                    "_gpu_snapshot",
                    side_effect=[_idle_snapshot(), _idle_snapshot()],
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available"),
                mock.patch.object(LAUNCHER.subprocess, "run", side_effect=fake_child),
            ):
                self.assertEqual(
                    LAUNCHER.run_v2(
                        request_path,
                        dry_run=False,
                        authorize_gpu_capture=True,
                    ),
                    23,
                )
            final = json.loads(
                (attempt / "final_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final["status"], "failed")
            self.assertEqual(final["child_exit_code"], 23)
            self.assertEqual(final["capture_process_exit_code"], 23)
            self.assertEqual(final["child_exit"], {"observed": True, "returncode": 23})
            self.assertEqual(
                final["failure_observability_status"],
                "phase_and_complete_traceback_persisted",
            )
            self.assertEqual(
                final["capture_observability"]["capture_failure_detail"]["phase"],
                "mesh",
            )
            self.assertIn(
                "RuntimeError: fake mesh failure",
                final["capture_observability"]["capture_failure_detail"]["traceback"],
            )
            self.assertEqual(
                (attempt / "capture_stdout.log").read_text(encoding="utf-8"),
                "exclusive child stdout\n",
            )
            self.assertEqual(
                (attempt / "capture_stderr.log").read_text(encoding="utf-8"),
                "Traceback: exclusive child stderr\n",
            )
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                self.assertRaisesRegex(RuntimeError, "final receipt"),
            ):
                LAUNCHER.run_v2(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=True,
                )


if __name__ == "__main__":
    unittest.main()
