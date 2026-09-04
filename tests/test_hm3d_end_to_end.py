from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools" / "studio" / "run_hm3d_end_to_end.py"
SPEC = importlib.util.spec_from_file_location("run_hm3d_end_to_end", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_source_provenance_records_repo_commit_branch_entrypoint_and_python():
    state = RUNNER._source_provenance()

    assert Path(state["repository"]) == RUNNER.REPOSITORY.resolve()
    assert Path(state["entrypoint"]) == SCRIPT.resolve()
    assert isinstance(state["git_commit"], str) and state["git_commit"]
    assert state["git_branch"] is None or (
        isinstance(state["git_branch"], str) and state["git_branch"]
    )
    assert Path(state["python_executable"]).is_file()


def test_source_provenance_git_failure_is_recorded_without_becoming_a_gate(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(RUNNER.subprocess, "run", unavailable)
    state = RUNNER._source_provenance()

    assert Path(state["repository"]) == RUNNER.REPOSITORY.resolve()
    assert state["git_commit"] is None
    assert state["git_branch"] is None
    assert "git_error" in state


def test_end_to_end_receipt_carries_source_provenance(monkeypatch, tmp_path):
    source = {
        "repository": "/repo",
        "git_commit": "abc123",
        "git_branch": "main",
        "entrypoint": "/repo/tools/studio/run_hm3d_end_to_end.py",
        "python_executable": "/env/bin/python",
    }
    monkeypatch.setattr(RUNNER, "_source_provenance", lambda: source)
    chosen = {
        "region_id": 4,
        "floor_area_m2": 12.5,
        "_furnished": ["chair"],
    }
    receipt = RUNNER._build_receipt(
        scene_dir=tmp_path / "00123-example",
        scene_id="example",
        chosen=chosen,
        attempts=[],
        bank=tmp_path / "routes" / "example.bank.json",
        rooms_dir=tmp_path / "rooms",
        routes_dir=tmp_path / "routes",
        output=tmp_path / "output",
    )

    assert receipt["source"] == source
    assert receipt["schema"] == "avengine_hm3d_end_to_end_receipt_v1"


def test_zero_exit_scan_geometry_fail_is_reported_not_a_package_command_failure(
    monkeypatch, tmp_path
):
    completed = SimpleNamespace(
        returncode=0,
        stdout=b'{"qa_geometry_status": "fail", "qa_note": "scan mesh"}\n',
    )
    monkeypatch.setattr(RUNNER.subprocess, "run", lambda *args, **kwargs: completed)

    RUNNER.run("package", ["package"], tmp_path)

    assert (tmp_path / "package.log").read_bytes() == completed.stdout


def test_nonzero_frame_parity_command_stops_package_stage(monkeypatch, tmp_path):
    completed = SimpleNamespace(
        returncode=1,
        stdout=b"frame parity FAILED: package disagrees with Habitat\n",
    )
    monkeypatch.setattr(RUNNER.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(SystemExit, match="package failed with exit code 1"):
        RUNNER.run("package", ["package"], tmp_path)
