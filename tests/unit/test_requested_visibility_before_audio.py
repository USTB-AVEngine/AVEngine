from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import numpy as np
import pytest

from avengine.dataset import binding_group_native as native
from avengine.dataset.production_runner import ProductionRunner
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth
from avengine.rooms import qa_delivery

REPO = Path(__file__).resolve().parents[2]


def case(tmp_path, *, enters=True, side="right"):
    frames = 150
    normal = []
    for i in range(frames):
        frame = np.zeros((20, 30), dtype=np.int32)
        if not enters or i >= 30:
            frame[4:8, 24:27] = 41
        normal.append(frame)
    context = {"renderer_backend": "habitat", "rgb_renderer_backend": "habitat",
               "camera_contract_id": "test_fixed", "semantic_id_namespace": "test_ids",
               "resolution_hw": [20, 30], "frame_indices": list(range(frames)),
               "camera_pose_ids": ["fixed"] * frames}
    truth = compile_pixel_visibility_truth(
        normal_semantic_masks=normal,
        target_only_semantic_masks_by_instance={"source1": normal},
        semantic_ids_by_instance={"source1": 41},
        normal_context={**context, "pass_kind": "modal_scene"},
        target_only_contexts_by_instance={"source1": {
            **context, "pass_kind": "target_only", "target_instance_id": "source1"}})
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "pixel_visibility_truth.json").write_text(json.dumps(truth))
    request = {"qa_targets": [{"qa_id": "QA-07", "branch": side,
                              "target_instance_ids": ["speaker"]}]}
    requirement = {"kind": "out_of_view_to_visible", "subject": "speaker",
                   "side": side, "qa_id": "QA-07", "source_condition_key": "entry_transition",
                   "require_publishable_window": True, "observation_windows": [[0, frames]]}
    plan = {"episode_id": "test", "clock": {"frame_count": frames, "frame_rate_hz": 15},
            "visual_plan": {"camera": {"resolution_hw": [20, 30]},
                            "actors": [{"actor_id": "source1", "entity_instance_id": "speaker"}]},
            "camera_condition_sampling": {"visibility_solver": {"requirements": [requirement]}}}
    return plan, request, capture


def test_judges_mask_transition_and_preserves_failed_report(tmp_path):
    plan, request, capture = case(tmp_path)
    assert native.check_requested_visibility(plan, request, capture)["status"] == "pass"
    request["qa_targets"][0]["branch"] = "left"
    plan["camera_condition_sampling"]["visibility_solver"]["requirements"][0]["side"] = "left"
    report = tmp_path / "rejected.json"
    with pytest.raises(native.RequestedVisibilityError):
        native.check_requested_visibility(plan, request, capture, report_path=report)
    saved = json.loads(report.read_text())
    assert saved["binding"]["status"] == "pass"
    assert saved["requirements"][0]["status"] == "fail"


def test_bare_qa_ids_do_not_impose_every_question_on_every_world(tmp_path):
    assert native.check_requested_visibility({}, {"qa_ids": ["QA-07"]}, tmp_path)[
        "status"] == "not_requested"


def test_missing_pixel_data_is_not_pass(tmp_path):
    plan, request, capture = case(tmp_path)
    with pytest.raises(native.RequestedVisibilityError, match="missing"):
        native.check_requested_visibility(plan, request, tmp_path / "absent")


def test_monolithic_run_never_calls_audio_after_rejected_visual_condition(tmp_path, monkeypatch):
    plan, request, capture = case(tmp_path, enters=False)
    spec = importlib.util.spec_from_file_location("qa_entry_under_test", REPO / "tools/studio/run_qa_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "plan_request", lambda request, output: plan)
    monkeypatch.setattr(module, "capture_command", lambda request, output: ["test_renderer"])
    launched = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(module, "read_json", lambda path: {})
    monkeypatch.setattr(module, "renderer_for_room", lambda package: "spear_unreal")
    def forbidden_audio(*args, **kwargs):
        pytest.fail("audio was called after a rejected pixel condition")
    monkeypatch.setattr(qa_delivery, "finalize_qa_episode", forbidden_audio)
    with pytest.raises(native.RequestedVisibilityError):
        module.run(request, tmp_path)
    assert len(launched) == 1
    assert json.loads((tmp_path / "native_visibility_acceptance.json").read_text())["status"] == "fail"


def test_native_condition_rejection_is_not_retried_as_same_capture():
    runner = object.__new__(ProductionRunner)
    scope = SimpleNamespace(blockers=[])
    decision = runner._may_retry(
        scope, {"stage": "capture", "work_item_id": "e:capture:01"},
        {"reason": "RequestedVisibilityError: requested visibility conditions not satisfied"})
    assert decision["retry"] is False
    assert decision["kind"] == "requires_replan"


def test_registered_occluder_precheck_builds_shared_visual_evidence_without_mutating_capture(tmp_path, monkeypatch):
    from avengine.rooms import qa_evidence, conditioned_visibility
    plan, request, capture = case(tmp_path)
    request['qa_targets'] = [{'qa_id': 'QA-10', 'target_instance_ids': ['speaker']}]
    request['qa_sampling'] = {'acceptance_policy': {'keep_scene_when_target_unmet': True}}
    plan['camera_condition_sampling']['visibility_solver']['requirements'] = [
        {'kind': 'registered_occluder_visible', 'subject': 'speaker', 'qa_id': 'QA-10',
         'observation_windows': [[30, 48]], 'require_publishable_window': True}]
    (capture / 'native_pixel_masks_depth_authority_v1.npz').write_bytes(b'test dependency marker')
    calls = []
    witness = {'status': 'pass', 'frame_records': []}
    def prepare(root, scene, truth, **kwargs):
        calls.append(('prepare', kwargs['shared_root']))
        return {'actor_occluders': witness, 'appearance_review': {'actors': {}}}
    def judge(requirements, **kwargs):
        calls.append(('judge', kwargs['occluder_evidence']))
        return {'status': 'fail', 'pixel_truth_authority_registered': True,
                'requirements': [{'status': 'fail', 'reason': 'no attributed occlusion'}]}
    monkeypatch.setattr(qa_evidence, 'acquire_shared_visual_evidence', prepare)
    monkeypatch.setattr(conditioned_visibility, 'accept_native_visibility', judge)
    report = tmp_path / 'checks' / 'native_visibility_acceptance.json'
    before = set(capture.iterdir())
    result = native.check_requested_visibility(plan, request, capture, report_path=report)
    assert calls == [('prepare', report.parent / 'native_visibility_visual_evidence'), ('judge', witness)]
    assert set(capture.iterdir()) == before
    assert result['status'] == 'fail' and result['requested_target_unmet'] is True
    assert result['shared_visual_root'] == str((report.parent / 'native_visibility_visual_evidence').resolve())
