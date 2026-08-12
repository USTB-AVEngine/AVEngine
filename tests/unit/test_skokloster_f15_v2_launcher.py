from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools/qa"
sys.path.insert(0, str(TOOLS))


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load(
    TOOLS / "run_strict_two_human_skokloster_f15_probe_v2.py",
    "avengine_test_skokloster_f15_v2_launcher",
)
BASE = V2.base


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_recipe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "name: spear-env",
                "# /data/jzy/miniconda3/envs/spear-env/bin/pip install -e ",
                "#   <AVEngine>/external/SPEAR/python",
                "# /data/jzy/miniconda3/envs/spear-env/bin/pip install -e ",
                "#   <AVEngine>/external/SPEAR/python_ext",
                "dependencies:",
                "  - python=3.11",
                "  - pip:",
                "    - numpy==2.0.2",
                "    - opencv-python==4.10.0.84",
                "    - PyYAML==6.0.3",
            )
        )
        + "\n",
        encoding="utf-8",
    )


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


def _write_v1_terminal(atom: Path, *, bound_python: str) -> dict[str, Path]:
    paths = V2._v1_paths(atom)
    request = {
        "schema": BASE.REQUEST_SCHEMA,
        "status": "prepared_not_launched",
        "required_repo_commit": "a" * 40,
        "capture_python": bound_python,
        "capture_output": str(paths["capture_root"]),
        "frame_indices": [BASE.FRAME_INDEX],
        "full75_allowed": False,
        "formal_dataset_count": 0,
    }
    _write_json(paths["request"], request)
    _write_json(
        paths["dry_run"],
        {
            "schema": BASE.RECEIPT_SCHEMA,
            "status": "dry_run_pass_not_launched",
            "gpu_started": False,
            "attempt_consumed": False,
            "formal_dataset_count": 0,
        },
    )
    _write_json(
        paths["running"],
        {
            "schema": BASE.RECEIPT_SCHEMA,
            "status": "running",
            "attempt_consumed": True,
            "formal_dataset_count": 0,
        },
    )
    phase_values = (
        (paths["phase_prelaunch"], 0, "prelaunch_closed"),
        (paths["phase_child_start"], 1, "child_invocation_started"),
        (paths["phase_child_exit"], 2, "child_exit_observed"),
    )
    phases = []
    for path, sequence, phase in phase_values:
        marker = {
            "schema": BASE.PHASE_SCHEMA,
            "status": "entered",
            "sequence": sequence,
            "phase": phase,
            "formal_dataset_count": 0,
        }
        _write_json(path, marker)
        phases.append({"sequence": sequence, "phase": phase})
    paths["stdout"].write_bytes(b"")
    paths["stderr"].write_text(
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'numpy'\n",
        encoding="utf-8",
    )
    _write_json(
        paths["final"],
        {
            "schema": BASE.RECEIPT_SCHEMA,
            "status": "failed",
            "attempt_consumed": True,
            "retry_same_candidate_forbidden": True,
            "child_invocation_attempted": True,
            "child_exit_code": 1,
            "capture_process_exit_code": 1,
            "failure_phase": "child_exit_observed",
            "capture_argv": [bound_python, "capture.py"],
            "launcher_phases": phases,
            "formal_dataset_count": 0,
        },
    )
    return paths


@contextlib.contextmanager
def _identity_patch(repo: Path, official_python: Path):
    def origin_main_contract():
        return {
            "repository": "Eastforward/AVEngine",
            "remote_url": "git@github.com:Eastforward/AVEngine.git",
            "git_ref": V2.OFFICIAL_ENV_GIT_REF,
            "commit": "f" * 40,
            "recipe_path": str(V2.OFFICIAL_ENV_RECIPE_RELATIVE),
            "recipe": BASE._file_record(repo / V2.OFFICIAL_ENV_RECIPE_RELATIVE),
            "environment_name": "spear-env",
            "logical_interpreter": str(official_python),
        }

    with (
        mock.patch.object(V2, "REPOSITORY", repo),
        mock.patch.object(V2, "AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL", official_python),
        mock.patch.object(
            V2, "AUTHORITATIVE_ENV_PREFIX", official_python.parent.parent
        ),
        mock.patch.object(BASE, "CAPTURE_PYTHON_LOGICAL", repo / ".venv/bin/python"),
        mock.patch.object(
            V2, "_origin_main_env_contract", side_effect=origin_main_contract
        ),
    ):
        yield


class SkoklosterF15V2LauncherTests(unittest.TestCase):
    def test_v1_ledger_freezes_environment_failure_and_zero_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / BASE.ATOM_DIRECTORY
            official = repo / "conda/envs/spear-env/bin/python"
            official.parent.mkdir(parents=True)
            official.write_bytes(b"python")
            _write_recipe(repo / V2.OFFICIAL_ENV_RECIPE_RELATIVE)
            _write_v1_terminal(atom, bound_python="/uv/base/python3.11")
            with _identity_patch(repo, official):
                ledger_path = V2.record_v1_terminal_failure(atom_root=atom)
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                self.assertEqual(ledger["status"], V2.V1_FAILURE_STATUS)
                self.assertEqual(ledger["captured_frame_count"], 0)
                self.assertEqual(ledger["capture_artifact_count"], 0)
                self.assertFalse(ledger["capture_output_materialized"])
                self.assertTrue(ledger["attempt_consumed"])
                self.assertTrue(ledger["retry_same_candidate_forbidden"])
                self.assertEqual(
                    ledger["root_cause"]["official_environment"], "spear-env"
                )
                self.assertEqual(
                    ledger["root_cause"]["request_bound_interpreter"],
                    "/uv/base/python3.11",
                )
                self.assertEqual(ledger["formal_dataset_count"], 0)
                self.assertEqual(
                    V2._validate_v1_ledger(ledger_path)["status"],
                    V2.V1_FAILURE_STATUS,
                )
                with self.assertRaisesRegex(RuntimeError, "ledger exists"):
                    V2.record_v1_terminal_failure(atom_root=atom)

    def test_v1_ledger_rejects_any_materialized_capture_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / BASE.ATOM_DIRECTORY
            official = repo / "conda/envs/spear-env/bin/python"
            official.parent.mkdir(parents=True)
            official.write_bytes(b"python")
            _write_recipe(repo / V2.OFFICIAL_ENV_RECIPE_RELATIVE)
            paths = _write_v1_terminal(atom, bound_python="/uv/base/python3.11")
            paths["capture_root"].mkdir(parents=True)
            with (
                _identity_patch(repo, official),
                self.assertRaisesRegex(RuntimeError, "materialized capture output"),
            ):
                V2.record_v1_terminal_failure(atom_root=atom)

    def test_origin_main_recipe_contract_rejects_wrong_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "spear-env.yml"
            _write_recipe(recipe)
            V2._validate_official_env_recipe(recipe)
            recipe.write_text(
                recipe.read_text(encoding="utf-8").replace(
                    "name: spear-env", "name: repository-venv"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "name: spear-env"):
                V2._validate_official_env_recipe(recipe)

    def test_origin_main_contract_pins_remote_commit_recipe_and_logical_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            recipe = repo / V2.OFFICIAL_ENV_RECIPE_RELATIVE
            _write_recipe(recipe)
            official = repo / "conda/envs/spear-env/bin/python"
            expected_commit = "b" * 40

            def git_text(*args: str) -> str:
                if args == ("remote", "get-url", "origin"):
                    return "git@github.com:Eastforward/AVEngine.git\n"
                if args == ("rev-parse", "origin/main"):
                    return expected_commit + "\n"
                if args == ("show", "origin/main:envs/spear-env.yml"):
                    return recipe.read_text(encoding="utf-8")
                raise AssertionError(f"unexpected git query: {args}")

            with (
                mock.patch.object(V2, "REPOSITORY", repo),
                mock.patch.object(V2, "AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL", official),
                mock.patch.object(V2, "_git_text", side_effect=git_text),
            ):
                contract = V2._origin_main_env_contract()

            self.assertEqual(contract["repository"], "Eastforward/AVEngine")
            self.assertEqual(contract["git_ref"], "origin/main")
            self.assertEqual(contract["commit"], expected_commit)
            self.assertEqual(contract["environment_name"], "spear-env")
            self.assertEqual(contract["logical_interpreter"], str(official))
            self.assertEqual(contract["recipe"]["path"], str(recipe))

    def test_origin_main_contract_rejects_worktree_recipe_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            recipe = repo / V2.OFFICIAL_ENV_RECIPE_RELATIVE
            _write_recipe(recipe)

            def git_text(*args: str) -> str:
                if args == ("remote", "get-url", "origin"):
                    return "https://github.com/Eastforward/AVEngine.git\n"
                if args == ("rev-parse", "origin/main"):
                    return "c" * 40 + "\n"
                if args == ("show", "origin/main:envs/spear-env.yml"):
                    return recipe.read_text(encoding="utf-8") + "# drift\n"
                raise AssertionError(f"unexpected git query: {args}")

            with (
                mock.patch.object(V2, "REPOSITORY", repo),
                mock.patch.object(V2, "_git_text", side_effect=git_text),
                self.assertRaisesRegex(RuntimeError, "differs from origin/main"),
            ):
                V2._origin_main_env_contract()

    def test_interpreter_probe_executes_logical_path_and_imports_full_graph(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            logical = repo / "conda/envs/spear-env/bin/python"
            real = repo / "base/python3.11"
            real.parent.mkdir(parents=True)
            real.write_bytes(b"python")
            logical.parent.mkdir(parents=True)
            logical.symlink_to(real)
            avengine = repo / "src/avengine/__init__.py"
            capture = repo / "tools/qa/capture_skokloster_strict_two_human_episode.py"
            avengine.parent.mkdir(parents=True)
            capture.parent.mkdir(parents=True)
            avengine.write_text("# avengine\n", encoding="utf-8")
            capture.write_text("# capture\n", encoding="utf-8")
            origins = {}
            for name in V2.EXPECTED_DEPENDENCIES:
                path = repo / "site-packages" / name / "__init__.py"
                path.parent.mkdir(parents=True)
                path.write_text(f"# {name}\n", encoding="utf-8")
                origins[name] = path
            payload = {
                "logical_interpreter": str(logical),
                "interpreter_realpath": str(real),
                "sys_executable": str(logical),
                "sys_prefix": str(logical.parent.parent),
                "sys_base_prefix": str(real.parent),
                "python_version": "3.11.15",
                "dependencies": {
                    name: {**expected, "origin": str(origins[name])}
                    for name, expected in V2.EXPECTED_DEPENDENCIES.items()
                },
                "avengine_origin": str(avengine),
                "capture_module_origin": str(capture),
                "capture_module_imported": True,
                "loaded_cuda_libraries": [],
                "cuda_initialized": False,
                "probe_pid": 123,
            }

            def child(argv, **kwargs):
                self.assertEqual(argv[0], str(logical))
                self.assertNotEqual(argv[0], str(real))
                kwargs["stdout"].write(json.dumps(payload).encode("utf-8"))
                kwargs["stderr"].write(b"")
                return SimpleNamespace(returncode=0)

            attempt = repo / "attempt"
            attempt.mkdir()
            with (
                _identity_patch(repo, logical),
                mock.patch.object(
                    BASE,
                    "_gpu_snapshot",
                    side_effect=[_idle_snapshot(), _idle_snapshot()],
                ),
                mock.patch.object(V2.subprocess, "run", side_effect=child),
            ):
                receipt_path = V2._probe_authoritative_interpreter(
                    attempt_root=attempt, logical_python=logical
                )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], "pass")
                self.assertEqual(receipt["execution_argv"][0], str(logical))
                self.assertEqual(receipt["interpreter_realpath"], str(real))
                self.assertFalse(receipt["gpu_process_started"])
                self.assertFalse(receipt["payload"]["cuda_initialized"])
                self.assertEqual(receipt["payload"]["loaded_cuda_libraries"], [])
                self.assertEqual(
                    set(receipt["payload"]["dependencies"]),
                    {"numpy", "cv2", "yaml", "spear", "spear_ext"},
                )
                self.assertTrue(receipt["payload"]["capture_module_imported"])
                self.assertEqual(receipt["formal_dataset_count"], 0)
                self.assertEqual(
                    V2._validate_probe_receipt(receipt_path)["status"], "pass"
                )

    def test_capture_argv_preserves_logical_python_not_realpath(self) -> None:
        request = {
            "capture_python_logical": "/conda/envs/spear-env/bin/python",
            "capture_python_realpath": "/conda/bin/python3.11",
            "capture_script": "/repo/capture.py",
            "suite_plan": "/evidence/suite.json",
            "audio_wav": "/evidence/audio.wav",
            "spear_root": "/runtime/SPEAR",
            "packaged_executable": "/archive/SpearSim.sh",
            "capture_output": "/evidence/v2-capture",
            "rpc_port": V2.V2_RPC_PORT,
        }
        argv = V2._capture_argv_v2(request)
        self.assertEqual(argv[0], request["capture_python_logical"])
        self.assertNotEqual(argv[0], request["capture_python_realpath"])
        self.assertEqual(argv[argv.index("--frame-index") + 1], "15")
        self.assertEqual(argv[argv.index("--graphics-adapter") + 1], "1")
        self.assertEqual(argv.count("--authorize-gpu-capture"), 1)

    def test_prepare_rejects_repository_venv_before_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / BASE.ATOM_DIRECTORY
            wrong = repo / ".venv/bin/python"
            official = repo / "conda/envs/spear-env/bin/python"
            official.parent.mkdir(parents=True)
            official.write_bytes(b"python")
            with (
                _identity_patch(repo, official),
                mock.patch.object(BASE, "_require_clean_head", return_value="a" * 40),
                mock.patch.object(V2, "_probe_authoritative_interpreter") as probe,
                self.assertRaisesRegex(RuntimeError, "official spear-env"),
            ):
                V2.prepare_request_v2(
                    atom_root=atom,
                    capture_python=wrong,
                    spear_root=repo / "SPEAR",
                    rpc_port=V2.V2_RPC_PORT,
                )
            probe.assert_not_called()

    def test_v2_dry_run_does_not_consume_or_invoke_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / V2.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = {
                "attempt_root": str(attempt),
                "capture_stdout": str(attempt / "capture_stdout.log"),
                "capture_stderr": str(attempt / "capture_stderr.log"),
                "capture_output": str(attempt.parent / V2.V2_CAPTURE_DIRECTORY),
                "required_repo_commit": "a" * 40,
                "capture_python_logical": str(V2.AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
                "capture_python_realpath": "/base/python3.11",
                "rpc_port": V2.V2_RPC_PORT,
            }
            argv = [str(V2.AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL), "capture.py"]
            with (
                mock.patch.object(
                    V2, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(BASE, "_gpu_snapshot", return_value=_idle_snapshot()),
                mock.patch.object(BASE, "_assert_port_available"),
                mock.patch.object(V2.subprocess, "run") as child,
            ):
                code = V2.run_v2(
                    request_path,
                    dry_run=True,
                    authorize_gpu_capture=False,
                    mp3d_v2_terminal_receipt=Path("/unused"),
                )
            self.assertEqual(code, 0)
            child.assert_not_called()
            receipt = json.loads(
                (attempt / "dry_run_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "dry_run_pass_not_launched")
            self.assertFalse(receipt["attempt_consumed"])
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse((attempt / "running_receipt.json").exists())
            self.assertFalse((attempt / "capture_stdout.log").exists())
            self.assertEqual(receipt["formal_dataset_count"], 0)

    def test_v2_real_launch_requires_authorization_before_gpu_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / V2.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = {
                "attempt_root": str(attempt),
                "capture_stdout": str(attempt / "capture_stdout.log"),
                "capture_stderr": str(attempt / "capture_stderr.log"),
            }
            with (
                mock.patch.object(
                    V2,
                    "_validate_request_v2",
                    return_value=(request, ["spear-env/bin/python", "capture.py"]),
                ),
                mock.patch.object(BASE, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "explicit launch authorization"),
            ):
                V2.run_v2(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=False,
                    mp3d_v2_terminal_receipt=Path("/unused"),
                )
            snapshot.assert_not_called()
            self.assertFalse((attempt / "running_receipt.json").exists())

    def test_v2_real_launch_requires_bound_dry_receipt_before_gpu_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / V2.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = {
                "attempt_root": str(attempt),
                "capture_stdout": str(attempt / "capture_stdout.log"),
                "capture_stderr": str(attempt / "capture_stderr.log"),
            }
            with (
                mock.patch.object(
                    V2,
                    "_validate_request_v2",
                    return_value=(request, ["spear-env/bin/python", "capture.py"]),
                ),
                mock.patch.object(BASE, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "missing Skokloster v2 dry-run"),
            ):
                V2.run_v2(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=True,
                    mp3d_v2_terminal_receipt=Path("/unused"),
                )
            snapshot.assert_not_called()
            self.assertFalse((attempt / "running_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
