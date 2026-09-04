"""Focused checks for the AVEngine-local SkinTokens closure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import numpy as np
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY / "tools/assets/run_skintokens_rig.py"
SPEC = importlib.util.spec_from_file_location("run_skintokens_rig", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

CHAIN = REPOSITORY / "tools/assets/run_generated_animal_chain.sh"
SOURCE_ROOT = REPOSITORY / "src/avengine/assets/skintokens"


def test_runner_rejects_external_rigger_root_before_loading_any_input() -> None:
    assert RUNNER.main([
        "--rigger-root", "/data/jzy/code/SkinTokens",
        "--input", "/tmp/missing.glb",
        "--output", "/tmp/fresh-skintokens.glb",
    ]) == 2


def test_runner_and_chain_have_no_upstream_checkout_execution_path() -> None:
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    chain_source = CHAIN.read_text(encoding="utf-8")
    combined = runner_source + "\n" + chain_source
    assert "demo.py" not in combined
    assert ".venv" not in combined
    assert "runpy" not in combined
    server_source = (
        REPOSITORY / "tools/assets/skintokens_loopback_bpy_server.py"
    ).read_text(encoding="utf-8")
    assert "$RIGGER_ROOT/" not in chain_source
    assert "run_skintokens_rig.py" in chain_source
    assert "AF_UNIX" in runner_source
    assert "UnixStreamServer" in server_source
    assert "HTTPServer((" not in server_source
    assert "127.0.0.1" not in runner_source


def test_model_registry_declares_local_skintokens_and_qwen_roots() -> None:
    registry = json.loads(
        (REPOSITORY / "examples/assets/model_roots_v1.json").read_text(
            encoding="utf-8"
        )
    )
    models = registry["models"]
    assert {"skintokens", "qwen3_0_6b"} <= set(models)


def test_local_source_has_no_absolute_src_import_or_external_checkout_path() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SOURCE_ROOT.rglob("*.py")
    )
    assert "from src." not in source
    assert "import src." not in source
    assert "/data/jzy/code/" not in source
    assert "gradio" not in source.lower()


def test_shell_entry_is_syntactically_valid() -> None:
    subprocess.run(["bash", "-n", str(CHAIN)], check=True)


def test_runtime_module_compiles_without_model_execution() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(SOURCE_ROOT / "runtime.py"),
            str(RUNNER_PATH),
            str(REPOSITORY / "tools/assets/skintokens_loopback_bpy_server.py"),
        ],
        check=True,
    )


def test_numpy_knn_fallback_visits_both_dimensions_in_bounded_tiles(
    monkeypatch,
) -> None:
    from skintokens.rig_package import utils

    queries = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    references = np.array(
        [[1.0, 0.0, 0.0], [3.1, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    monkeypatch.setattr(utils, "cKDTree", None)
    monkeypatch.setattr(utils, "_blender_kdtree", None)
    distances, indices = utils.nearest_neighbors(queries, references, k=2)
    assert indices.tolist() == [[0, 1], [1, 0]]
    assert np.allclose(distances, [[1.0, 3.1], [0.1, 2.0]], atol=1e-6)
    assert utils.NEAREST_NEIGHBOR_TEMP_BYTES == 8 * 1024 * 1024


def test_runner_rejects_legacy_tcp_port() -> None:
    assert RUNNER.main([
        "--port", "59877",
        "--input", "/tmp/missing.glb",
        "--output", "/tmp/fresh-skintokens.glb",
    ]) == 2


def test_private_rpc_directory_is_removed_when_blender_is_unavailable(
    tmp_path: Path,
) -> None:
    rpc = RUNNER._BlenderRPC(
        blender=tmp_path / "missing-blender",
        workdir=tmp_path,
        timeout=1.0,
    )
    session_dir = rpc.session_dir
    with pytest.raises(ValueError, match="Blender executable is missing"):
        rpc.__enter__()
    assert not session_dir.exists()
