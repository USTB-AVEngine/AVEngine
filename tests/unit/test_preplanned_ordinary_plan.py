from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from avengine.dataset.production_runner import (
    ProductionRunError, StageOutputMissing, _ordinary_plan_executor,
)


def context(tmp_path):
    saved = tmp_path / "cpu_plan"
    (saved / "plan").mkdir(parents=True)
    request = {
        "episode_id": "preplanned_test", "seed": 7,
        "camera": {"fov_deg": 85}, "runtime": {"runtime_prefix": "/runtime"},
        "sound_selection": {"preallocated_sound_asset_ids_by_actor": {"human": ["voice_a"]}},
    }
    (saved / "request.json").write_text(json.dumps(request))
    (saved / "plan" / "episode_plan.json").write_text(json.dumps({
        "episode_id": request["episode_id"], "clock": {"frame_count": 150},
        "resources": {"room_package": {"renderer": "habitat"}},
    }))
    return SimpleNamespace(
        output_root=tmp_path / "stage", request=deepcopy(request), scope_id="preplanned_test",
        recipe_options={"preplanned_episode_root": str(saved)},
    ), saved


def test_cpu_plan_is_used_without_a_second_sampler_call(tmp_path, monkeypatch):
    ctx, saved = context(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("preplanned Episode was sampled again")
    monkeypatch.setattr("avengine.dataset.binding_group_native.plan_visual_variant", forbidden)
    ctx.request["runtime"]["graphics_adapter"] = 2
    ctx.request["runtime"]["rpc_port"] = 21001
    ctx.request["sampling_candidate_index"] = 0
    result = _ordinary_plan_executor(ctx)
    assert result.status == "pass"
    assert result.outputs["ordinary_execution"] == "reused_cpu_preplanned_episode"
    assert result.outputs["episode_plan"] == str(ctx.output_root / "episode" / "plan" / "episode_plan.json")
    assert json.loads((saved / "plan" / "episode_plan.json").read_text()) == json.loads(
        (ctx.output_root / "episode" / "plan" / "episode_plan.json").read_text())
    assert "graphics_adapter" not in json.loads((saved / "request.json").read_text())["runtime"]


@pytest.mark.parametrize("changed_key", ["camera", "sound_selection"])
def test_preplanned_request_cannot_change_geometry_or_audio(tmp_path, changed_key):
    ctx, saved = context(tmp_path)
    ctx.request[changed_key] = {"changed": True}
    with pytest.raises(ProductionRunError, match=changed_key):
        _ordinary_plan_executor(ctx)


def test_preplanned_root_cannot_smuggle_old_capture(tmp_path):
    ctx, saved = context(tmp_path)
    (saved / "capture").mkdir()
    with pytest.raises(ProductionRunError, match="already contains capture"):
        _ordinary_plan_executor(ctx)


def test_missing_plan_is_rejected_before_capture(tmp_path):
    ctx, saved = context(tmp_path)
    (saved / "plan" / "episode_plan.json").unlink()
    with pytest.raises(StageOutputMissing, match="lacks request/plan"):
        _ordinary_plan_executor(ctx)
