from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import os

import pytest

from avengine.dataset import binding_group_native as native
from avengine.dataset import binding_group_motion as motion

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "tmp/binding_v1_parallel_20260910/TAKEOVER/M23_STATE_QUALIFICATION/attempt_20260911T014229Z_pid941113/run/work/takeover_cross_time_state_mp3d_g01/v0/plan/attempt_01"
THIN = BASE / "episode/plan/episode_plan.json"
pytestmark = pytest.mark.skipif(not THIN.is_file(), reason="real M23 thin state plan unavailable")


class CaptureReached(Exception):
    pass


@pytest.fixture
def boundary(monkeypatch):
    candidate = os.environ.get("AVENGINE_CAPTURE_FUNCTION_CANDIDATE")
    if candidate:
        original = native.run_visual_capture_unit
        exec(Path(candidate).read_text(), native.__dict__)
        replacement = native.run_visual_capture_unit
        native.run_visual_capture_unit = original
        monkeypatch.setattr(native, "run_visual_capture_unit", replacement)
    unit = {"unit_id": "v0_capture", "depends_on_units": ["v0"]}
    context = {"group_id": "M23_readonly_probe", "group_spec": {"stage_units": [unit]}}
    item = {"unit_id": "v0_capture", "inputs": {"v0": {
        "facts": {"episode_plan_path": str(THIN)},
        "outputs": {"request_path": str(BASE / "request.json")},
    }}}
    monkeypatch.setattr(native, "_unit_row", lambda *_: unit)
    monkeypatch.setattr(native, "_visual_unit_for", lambda *_: {"source_asset_ids": []})
    monkeypatch.setattr(native, "resolve_instance_runtime", lambda *a, **k: {"graphics_adapter": None, "rpc_port": None})
    monkeypatch.setattr(native, "group_world_equivalence", lambda *a, **k: {"status": "pass"})
    monkeypatch.setattr(native, "_retained_root_for", lambda *_: None)
    spec = importlib.util.spec_from_file_location("qa_capture_cli_probe", REPO / "tools/studio/run_qa_episode.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    def stop_before_launch(request, output, **kwargs):
        command = cli.capture_command(request, output)
        assert "--case-manifest" in command
        assert "--m1-request" in command
        assert "--room-manifest" in command
        assert (Path(output) / "plan/habitat_execution/case_manifest.json").is_file()
        calls.append(command)
        raise CaptureReached("native process intentionally not started")
    monkeypatch.setattr(native, "capture_visual_plan", stop_before_launch)
    return context, item, calls


def test_state_thin_plan_repair_reaches_real_capture_delegate(tmp_path, boundary):
    context, item, calls = boundary
    before = sorted(p.name for p in THIN.parent.iterdir())
    with pytest.raises(CaptureReached):
        motion._run_state_visual_capture_unit(item, context, tmp_path / "attempt_03", output_root=tmp_path)
    assert len(calls) == 1
    assert sorted(p.name for p in THIN.parent.iterdir()) == before == ["episode_plan.json"]


def test_capture_delegate_rejects_reused_attempt_without_overwrite(tmp_path, boundary):
    context, item, calls = boundary
    root = tmp_path / "attempt_03"
    patched, _ = motion._state_capture_item_with_materialized_plan(item, context, root)
    with pytest.raises(CaptureReached):
        native.run_visual_capture_unit(patched, context, root, output_root=tmp_path)
    saved = (root / "capture_request.json").read_bytes()
    with pytest.raises(FileExistsError):
        native.run_visual_capture_unit(patched, context, root, output_root=tmp_path)
    assert (root / "capture_request.json").read_bytes() == saved
    assert len(calls) == 1


def test_capture_delegate_preserves_unrelated_preexisting_directory(tmp_path, boundary):
    context, item, calls = boundary
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        native.run_visual_capture_unit(item, context, root, output_root=tmp_path)
    assert marker.read_text() == "preserve"
    assert calls == []
