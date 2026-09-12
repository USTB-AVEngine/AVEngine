
from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.dataset import binding_group_identity as identity

REPOSITORY = Path(__file__).resolve().parents[2]
ATTEMPT = REPOSITORY / (
    "tmp/binding_v1_parallel_20260910/TAKEOVER/L34_IDENTITY/"
    "attempt_20260911T125810Z_adapter"
)
PLAN = ATTEMPT / (
    "run/work/takeover_cross_event_identity_apartment_g01/"
    "identity_probe_plan/plan/attempt_01/stage_result.json"
)
TASK = ATTEMPT / (
    "run/workers/takeover_cross_event_identity_apartment_g01__"
    "identity_probe_capture_capture_01/task.json"
)

@pytest.mark.skipif(not PLAN.is_file() or not TASK.is_file(), reason="L34 real plan/lease unavailable")
@pytest.mark.parametrize("corrupt_rpc", [False, True])
def test_probe_command_keeps_real_allocator_placement(tmp_path, monkeypatch, corrupt_rpc):
    plan_row = json.loads(PLAN.read_text())
    task = json.loads(TASK.read_text())["context"]
    lease = {
        "lease_id": task["lease"]["lease_id"],
        "graphics_adapter": task["lease"]["device_index"],
        "rpc_port": task["lease"]["rpc_port"],
    }
    assert lease["graphics_adapter"] is not None and lease["rpc_port"] is not None
    request_path = Path(plan_row["outputs"]["request_path"])
    original = request_path.read_bytes()
    item = deepcopy(task["work_item"])
    root = tmp_path / "probe"
    called = []
    class LaunchBoundary(Exception):
        pass
    def stop_capture(request, visual_root, label):
        called.append(deepcopy(request))
        raise LaunchBoundary
    monkeypatch.setattr(identity, "_capture", stop_capture)
    if corrupt_rpc:
        qa = identity.native._qa_module()
        capture_command = qa.capture_command
        def wrong_port(*args, **kwargs):
            command = capture_command(*args, **kwargs)
            command[command.index("--rpc-port") + 1] = str(lease["rpc_port"] + 1)
            return command
        monkeypatch.setattr(qa, "capture_command", wrong_port)
        monkeypatch.setattr(identity.native, "_qa_module", lambda: qa)
    error = identity.IdentityNativeError if corrupt_rpc else LaunchBoundary
    with pytest.raises(error):
        identity._run_identity_probe_capture_unit(
            item, {}, root, output_root=tmp_path,
            results=[plan_row], lease=lease,
        )
    assert request_path.read_bytes() == original
    if corrupt_rpc:
        assert not called
    else:
        readback = json.loads((root / "capture_launch_preflight.json").read_text())
        assert readback["lease_checked"] is True
        assert readback["command_runtime"] == {
            key: lease[key] for key in ("graphics_adapter", "rpc_port")
        }
        assert len(called) == 1
        assert called[0]["runtime"]["graphics_adapter"] == lease["graphics_adapter"]
        assert called[0]["runtime"]["rpc_port"] == lease["rpc_port"]

@pytest.mark.skipif(not (ATTEMPT / "run/state.json").is_file(), reason="L34 real audio unavailable")
@pytest.mark.parametrize("mutation", [None, "emitter", "gain", "sound"])
def test_canonical_audio_requires_physical_event_equivalence(monkeypatch, mutation):
    state = json.loads((ATTEMPT / "run/state.json").read_text())
    rows = {r["scope_id"].rsplit("/", 1)[-1]: r
            for s in state["scopes"] for r in s["results"] if r["status"] == "pass"}
    if not all(k in rows for k in ("v0_a0", "v1_a0")):
        pytest.skip("real public audio not finished")
    variants, plans = [], []
    for key in ("v0_a0", "v1_a0"):
        row = rows[key]
        variants.append({"audio_report": row["facts"]["audio_report_path"],
                         "visual_capture_root": row["outputs"]["capture"]})
        plans.append(json.loads(Path(row["outputs"]["assignment_plan_path"]).read_text()))
    original_load = identity._load
    right_report = Path(variants[1]["audio_report"]).resolve()
    right_readback = Path(variants[1]["visual_capture_root"]).resolve() / "neutral_readback.json"
    def changed_load(path):
        value = original_load(path)
        if mutation == "gain" and Path(path).resolve() == right_report:
            value["gain_application"]["post_assembly_convolution_gain"] = 0.25
        if mutation == "emitter" and Path(path).resolve() == right_readback:
            value["entities"]["source2"][72]["emitter"][0] += 0.1
        return value
    monkeypatch.setattr(identity, "_load", changed_load)
    if mutation == "sound":
        plans[1]["audio_events"][1]["path"] += ".different"
    if mutation is None:
        proof = identity.identity_shared_audio_input_equivalence(*variants, *plans)
        assert proof["status"] == "pass"
        assert set(proof["layouts"]) == {"binaural", "ambisonics"}
    else:
        with pytest.raises(identity.IdentityNativeError):
            identity.identity_shared_audio_input_equivalence(*variants, *plans)
