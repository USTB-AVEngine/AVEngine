from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.dataset import binding_group_motion as motion
from avengine.dataset import binding_group_native as native
from avengine.dataset.binding_group_native import (
    BindingNativeError,
    compare_group_native_visuals,
    compare_native_visuals,
    controlled_world_contract,
    declared_audio_delivery,
    verify_delivered_audio_layouts,
    verify_retained_visual_request,
)
from avengine.dataset.production_spec import (
    CoreGroupRequest,
    production_request_from_legacy,
)


REPOSITORY = Path(__file__).resolve().parents[2]
STATE_ROOT = REPOSITORY / (
    "tmp/binding_dataset_20260909_v2/state_first/hm3d/group_v1"
)
IDENTITY_ROOT = REPOSITORY / (
    "tmp/binding_dataset_20260909_v2/identity_first/group_v14"
)
VISIBLE_REVIEW = REPOSITORY / (
    "tmp/binding_v1_parallel_20260910/TAKEOVER/MAIN/"
    "attempt_20260911T0129_controller/"
    "T02_VISIBLE_NATIVE_REVIEW_COUNTEREXAMPLE.json"
)
VISIBLE_RETAINED_REQUEST = REPOSITORY / (
    "tmp/binding_dataset_20260909_v2/visible_first/hm3d/visual/v0/request.json"
)


def _state_request() -> dict:
    return json.loads((STATE_ROOT / "request.json").read_text(encoding="utf-8"))


def _early_facts() -> dict[str, dict]:
    return {
        assignment: json.loads(
            (STATE_ROOT / f"variants/v0_{assignment}/delivery/facts.json").read_text(
                encoding="utf-8"
            )
        )
        for assignment in ("a0", "a1")
    }


def _native_contract() -> dict:
    order = [
        "rocketbox_human_male_adult_01_top_blue_research_v1",
        "rocketbox_human_male_adult_01_top_green_research_v1",
    ]
    return {
        "plan_equivalence": "world",
        "visual_intervention": "after_wet_tail_motion",
        "visual_units": {
            "v0_capture": {"source_asset_ids": order, "plan_unit_ids": ["v0"]},
            "v1_capture": {"source_asset_ids": order, "plan_unit_ids": ["v1"]},
        },
    }


def _visible_review() -> dict:
    return json.loads(VISIBLE_REVIEW.read_text(encoding="utf-8"))


def test_real_visible_slot_swap_uses_source_slot_and_keeps_strict_audio_compare():
    record = _visible_review()
    strict = record["strict_camera_clock_emitter_by_slot"]
    left = {"neutral_readback": strict["left"]}
    right = {"neutral_readback": strict["right"]}
    strict_result = compare_native_visuals(left, right)
    group_result = compare_group_native_visuals(
        left,
        right,
        contract=record["contract"],
        left_unit_id="v0_capture",
        right_unit_id="v1_capture",
    )
    assert strict_result["max_numeric_delta"] == 0.0
    assert group_result["status"] == "pass"
    assert group_result["max_camera_delta"] == 0.0
    assert group_result["entity_binding_source"] == {
        "left": "native_entity_identities",
        "right": "native_entity_identities",
    }
    assert group_result["native_entity_binding_verified"] is True
    assert group_result["authority"].endswith(
        "strict_entity_geometry_by_source_slot"
    )


@pytest.mark.parametrize(
    ("label", "root"),
    [("state", STATE_ROOT), ("identity", IDENTITY_ROOT)],
)
def test_real_state_and_identity_path_variants_keep_native_world_constraints(
    label: str, root: Path
):
    result = compare_group_native_visuals(
        {"neutral_readback": root / "visual/v0/capture/neutral_readback.json"},
        {"neutral_readback": root / "visual/v1/capture/neutral_readback.json"},
        contract=_native_contract(),
        left_unit_id="v0_capture",
        right_unit_id="v1_capture",
    )
    assert result["status"] == "pass", label
    assert result["path_intervention_allowed"] is True
    assert result["native_entity_binding_verified"] is True
    assert result["entity_binding_source"] == {
        "left": "native_entity_identities",
        "right": "native_entity_identities",
    }
    assert result["authority"].endswith(
        "declared_path_intervention_by_source_slot"
    )


def _synthetic_readback(*, source1_x: float, source2_x: float) -> dict:
    return {
        "schema": "synthetic",
        "clock": {"frame_count": 1, "frame_rate_hz": 15},
        "coordinate_frame": "world",
        "camera": [{
            "frame_index": 0,
            "position_m": [0.0, 0.0, 0.0],
            "basis": {"forward": [0.0, 0.0, 1.0]},
            "pts_ticks": 0,
        }],
        "entities": {
            "source1": [{
                "frame_index": 0,
                "root": [source1_x, 0.0, 0.0],
                "emitter": [source1_x, 1.0, 0.0],
                "moving": False,
            }],
            "source2": [{
                "frame_index": 0,
                "root": [source2_x, 0.0, 0.0],
                "emitter": [source2_x, 1.0, 0.0],
                "moving": False,
            }],
        },
        "entity_identities": {
            "source1": {"actor_id": "source1", "asset_id": "duplicate_asset"},
            "source2": {"actor_id": "source2", "asset_id": "duplicate_asset"},
        },
    }


def test_duplicate_asset_instances_are_compared_by_physical_source_slot(
    tmp_path: Path,
):
    left_path = tmp_path / "duplicate_left.json"
    right_path = tmp_path / "duplicate_right.json"
    left_path.write_text(
        json.dumps(_synthetic_readback(source1_x=0.0, source2_x=10.0)),
        encoding="utf-8",
    )
    right_path.write_text(
        json.dumps(_synthetic_readback(source1_x=20.0, source2_x=10.0)),
        encoding="utf-8",
    )
    contract = {
        "plan_equivalence": "controlled_slots",
        "visual_intervention": "source_slot_permutation",
        "visual_units": {
            "v0_capture": {"source_asset_ids": ["duplicate_asset", "duplicate_asset"]},
            "v1_capture": {"source_asset_ids": ["duplicate_asset", "duplicate_asset"]},
        },
    }
    with pytest.raises(BindingNativeError, match="native group geometry"):
        compare_group_native_visuals(
            {"neutral_readback": left_path},
            {"neutral_readback": right_path},
            contract=contract,
            left_unit_id="v0_capture",
            right_unit_id="v1_capture",
        )


def test_legacy_group_comparison_reports_plan_binding_fallback(tmp_path: Path):
    left = _synthetic_readback(source1_x=0.0, source2_x=10.0)
    right = _synthetic_readback(source1_x=0.0, source2_x=10.0)
    left.pop("entity_identities")
    right.pop("entity_identities")
    left_path = tmp_path / "legacy_left.json"
    right_path = tmp_path / "legacy_right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")
    contract = {
        "plan_equivalence": "controlled_slots",
        "visual_intervention": "source_slot_permutation",
        "visual_units": {
            "v0_capture": {"source_asset_ids": ["duplicate_asset", "duplicate_asset"]},
            "v1_capture": {"source_asset_ids": ["duplicate_asset", "duplicate_asset"]},
        },
    }
    result = compare_group_native_visuals(
        {"neutral_readback": left_path},
        {"neutral_readback": right_path},
        contract=contract,
        left_unit_id="v0_capture",
        right_unit_id="v1_capture",
    )
    assert result["status"] == "pass"
    assert result["entity_binding_source"] == {
        "left": "plan_fallback",
        "right": "plan_fallback",
    }
    assert result["native_entity_binding_verified"] is False
    assert "plan_fallback_for_missing_entity_identities" in result["authority"]
    assert "actual_native_entity_binding" not in result["authority"]


def test_native_entity_binding_mismatch_is_rejected(tmp_path: Path):
    record = _visible_review()
    strict = record["strict_camera_clock_emitter_by_slot"]
    changed = json.loads(
        Path(strict["left"]).read_text(encoding="utf-8")
    )
    changed["entity_identities"]["source1"]["asset_id"] = (
        record["contract"]["visual_units"]["v1_capture"]["source_asset_ids"][0]
    )
    changed_path = tmp_path / "binding_mismatch.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(BindingNativeError, match="native entity binding"):
        compare_group_native_visuals(
            {"neutral_readback": changed_path},
            {"neutral_readback": strict["right"]},
            contract=record["contract"],
            left_unit_id="v0_capture",
            right_unit_id="v1_capture",
        )


def _retained_audio_view_case() -> tuple[dict, dict]:
    retained = json.loads(VISIBLE_RETAINED_REQUEST.read_text(encoding="utf-8"))
    requested = deepcopy(retained)
    requested["audio_layouts"] = [
        {"type": "binaural", "channel_count": 2, "role": "primary"},
        {
            "type": "ambisonics",
            "channel_count": 4,
            "role": "attached_view",
            "ambisonic_order": 1,
        },
    ]
    requested["foa_normalization"] = "sn3d"
    runtime = dict(retained.get("runtime") or {})
    contract = {
        "shared_world": requested,
        "shared_runtime": {
            key: runtime.get(key)
            for key in native.CONTROLLED_WORLD_RUNTIME_KEYS
        },
        "shared_audio_policy": None,
    }
    return retained, contract


def test_real_p09_retained_visual_allows_audio_view_change_and_records_both():
    retained, contract = _retained_audio_view_case()
    checked = verify_retained_visual_request(retained, contract, label="v0")
    assert checked["status"] == "pass"
    assert checked["separately_recorded_audio_view_fields"] == [
        "audio_layouts", "foa_normalization"
    ]
    fields = checked["audio_view_fields"]["fields"]
    assert fields["audio_layouts"]["same"] is False
    assert fields["foa_normalization"]["same"] is False
    assert checked["audio_view_fields"]["requested"]["foa_normalization"] == "sn3d"
    assert checked["audio_view_fields"]["retained"]["foa_normalization"] == "native_n3d"
    assert "audio_layouts" not in checked["checked_fields"]
    assert "foa_normalization" not in checked["checked_fields"]


def test_retained_audio_view_split_does_not_relax_camera_or_seed():
    _retained, base_contract = _retained_audio_view_case()
    for field in ("camera", "seed"):
        contract = deepcopy(base_contract)
        if field == "camera":
            contract["shared_world"]["camera"]["fov_deg"] += 1
        else:
            contract["shared_world"]["seed"] += 1
        with pytest.raises(BindingNativeError, match="different world"):
            verify_retained_visual_request(
                _retained, contract, label=f"changed_{field}"
            )


def _small_visible_core_group() -> CoreGroupRequest:
    orders = [
        ["asset_blue", "asset_green"],
        ["asset_blue", "asset_green"],
        ["asset_green", "asset_blue"],
        ["asset_green", "asset_blue"],
    ]
    members = []
    for index, order in enumerate(orders):
        request = {
            "schema": "avengine_native_qa_room_request_v1",
            "episode_id": f"foa_member_{index}",
            "room_id": "foa_room",
            "seed": 7,
            "sampling_policy": "conditioned_static_v2",
            "camera": {
                "motion": "static",
                "fov_deg": 85,
                "resolution_hw": [720, 1280],
            },
            "frame_count": 150,
            "frame_rate_hz": 15,
            "sample_rate_hz": 16000,
            "entities": {"total_count": 2, "silent_count": 0},
            "entity_instances": [
                {
                    "instance_id": f"inst_{index}_{position}",
                    "source_class": "articulated_human",
                    "asset_id": asset_id,
                }
                for position, asset_id in enumerate(order)
            ],
            "source_asset_ids": order,
            "profile": {"reserve_tail_s": 3.0},
            "qa_ids": ["QA-20"],
            "qa_sampling": {"items_per_type": 1},
            "rir_stride": 5,
            "post_assembly_convolution_gain": 0.5,
            "runtime": {
                "hrtf": "/tmp/hrtf.sofa",
                "graphics_adapter": 1,
                "rpc_port": 39500,
            },
            "sound_pool": "/tmp/pool.json",
            "sound_selection": {},
            "task_family": "visible_binding",
            "group_id": "foa_group",
            "motion_timing": "none",
            "production": {},
        }
        members.append(
            production_request_from_legacy(
                request, request_id=request["episode_id"], kind="core_group_member"
            )
        )
    return CoreGroupRequest(
        group_id="foa_group",
        task_family="visible_binding",
        room_id="foa_room",
        members=tuple(members),
    )


def test_group_members_still_reject_audio_view_disagreement():
    group = _small_visible_core_group()
    requests = {
        member.request_id: member.to_legacy_request()
        for member in group.members
    }
    requests["foa_member_2"]["audio_layouts"] = [
        {"type": "binaural", "channel_count": 2, "role": "primary"},
        {"type": "ambisonics", "channel_count": 4, "role": "attached_view"},
    ]
    with pytest.raises(BindingNativeError, match="one controlled world"):
        controlled_world_contract(group.to_dict(), requests)


def _audio_unit_fixture(tmp_path: Path, member_request: dict) -> tuple[dict, dict, Path]:
    capture = tmp_path / "capture"
    capture.mkdir()
    plan = {
        "clock": {
            "frame_count": 1,
            "frame_rate_hz": 15,
            "sample_rate_hz": 16000,
        },
        "audio_events": [
            {"event_id": "event_001", "sound_asset_id": "sound_one"}
        ],
        "voice_bindings": [
            {"sound_asset_id": "sound_one", "path": "/tmp/sound_one.wav"}
        ],
        "visual_plan": {
            "actors": [{
                "actor_id": "source1",
                "asset_id": "asset_blue",
                "source_endpoint_id": "source1_emitter",
            }]
        },
    }
    plan_path = tmp_path / "capture_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    capture_request = deepcopy(member_request)
    capture_request["audio_layouts"] = [
        {"type": "binaural", "channel_count": 2, "role": "primary"}
    ]
    capture_request["foa_normalization"] = "native_n3d"
    request_path = tmp_path / "capture_request.json"
    request_path.write_text(json.dumps(capture_request), encoding="utf-8")
    neutral_path = tmp_path / "neutral_readback.json"
    neutral_path.write_text("{}", encoding="utf-8")
    contract = {
        "audio_columns": {"a0": ["v0_a0"]},
        "stage_units": [{
            "unit_id": "v0_a0",
            "unit_kind": "audio",
            "visual_unit_id": "v0",
            "member_request_ids": ["member_0"],
        }],
    }
    context = {
        "group_id": "foa_group",
        "member_requests": {"member_0": deepcopy(member_request)},
        "group_spec": contract,
        "contract": contract,
    }
    item = {
        "work_item_id": "foa_group/v0_a0:audio:01",
        "stage": "audio",
        "scope_id": "foa_group/v0_a0",
        "unit_id": "v0_a0",
        "member_request_ids": ["member_0"],
        "inputs": {
            "v0": {
                "outputs": {
                    "capture": str(capture),
                    "episode_plan": str(plan_path),
                    "request_path": str(request_path),
                    "neutral_readback": str(neutral_path),
                }
            }
        },
        "resource": {},
    }
    return context, item, plan_path


def test_run_audio_unit_consumes_current_member_audio_view_and_persists_it(
    monkeypatch, tmp_path: Path
):
    _retained, contract = _retained_audio_view_case()
    member = deepcopy(contract["shared_world"])
    context, item, _plan_path = _audio_unit_fixture(tmp_path, member)
    seen = {}

    def fake_build(plan, request, assignment, **kwargs):
        seen["request"] = deepcopy(request)
        assigned = deepcopy(plan)
        assigned["request"] = deepcopy(request)
        return assigned, deepcopy(request)

    facts_path = tmp_path / "facts.json"
    facts_path.write_text(
        json.dumps({"audio": {"wet_tail_intervals": [{"end_s": 0.5}]}}),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(native, "build_audio_assignment_plan", fake_build)
    monkeypatch.setattr(
        native, "_neutral_endpoint_bindings",
        lambda *args, **kwargs: {"source1": "source1_emitter"},
    )
    monkeypatch.setattr(
        native,
        "acoustic_identity",
        lambda request, plan: {"audio_view": native._audio_view_fields(request)},
    )
    monkeypatch.setattr(
        native,
        "_audio_reuse_candidate",
        lambda *args, **kwargs: {
            "reused": False,
            "source_unit_id": None,
            "audio_report": None,
            "rejected": [],
        },
    )

    def fake_materialize(_visual, output, _plan, _request, *, member_id):
        output = Path(output)
        output.mkdir(parents=True)
        return output

    monkeypatch.setattr(native, "materialize_audio_variant", fake_materialize)
    monkeypatch.setattr(
        native,
        "finalize_audio_assignment",
        lambda *args, **kwargs: {
            "facts": str(facts_path),
            "audio_report": str(report_path),
            "audio": str(tmp_path / "audio.wav"),
            "questions": str(tmp_path / "questions.json"),
            "visual_video": None,
            "preview": str(tmp_path / "preview.mp4"),
            "delivered_audio_layouts": {"status": "pass"},
            "result": {},
            "elapsed_s": 0.0,
        },
    )

    result = native.run_audio_unit(
        item, context, tmp_path / "unit", output_root=tmp_path
    )
    assert result["status"] == "pass"
    assert seen["request"]["audio_layouts"] == member["audio_layouts"]
    assert seen["request"]["foa_normalization"] == "sn3d"
    assignment_request = json.loads(
        (tmp_path / "unit/assignment_request.json").read_text(encoding="utf-8")
    )
    assignment_plan = json.loads(
        (tmp_path / "unit/assignment_plan.json").read_text(encoding="utf-8")
    )
    assert assignment_request["audio_layouts"] == member["audio_layouts"]
    assert assignment_request["foa_normalization"] == "sn3d"
    assert assignment_plan["request"]["audio_layouts"] == member["audio_layouts"]
    assert result["outputs"]["member_request_id"] == "member_0"
    assert result["outputs"]["audio_view_fields"]["fields"]["audio_layouts"]["same"] is False


def test_state_audio_missing_attached_view_blocks_stage(
    monkeypatch, tmp_path: Path
):
    _retained, contract_case = _retained_audio_view_case()
    member = deepcopy(contract_case["shared_world"])
    capture = tmp_path / "capture"
    capture.mkdir()
    neutral = tmp_path / "neutral.json"
    neutral.write_text("{}", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"clock": {"frame_rate_hz": 15}}), encoding="utf-8")
    request = tmp_path / "request.json"
    request.write_text(json.dumps(member), encoding="utf-8")
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps({"audio": {"wet_tail_intervals": [{"end_s": 0.5}]}}),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    visual_video = tmp_path / "visual.mp4"
    visual_video.write_bytes(b"retained visual")
    context = {
        "group_id": "foa_state_group",
        "member_requests": {"member_0": member},
        "group_spec": {
            "stage_units": [{
                "unit_id": "v0_a0",
                "unit_kind": "audio",
                "visual_unit_id": "v0",
                "member_request_ids": ["member_0"],
            }],
        },
        "contract": contract_case | {
            "audio_columns": {"a0": ["v0_a0"]},
        },
    }
    item = {
        "work_item_id": "foa_state_group/v0_a0:audio:01",
        "scope_id": "foa_state_group/v0_a0",
        "stage": "audio",
        "unit_id": "v0_a0",
        "member_request_ids": ["member_0"],
    }
    visual_result = {
        "work_item_id": "foa_state_group/v0:visual_capture:01",
        "scope_id": "foa_state_group/v0",
        "status": "pass",
        "facts": {"episode_plan_path": str(plan)},
        "outputs": {
            "capture": str(capture),
            "neutral_readback": str(neutral),
            "request_path": str(request),
        },
    }

    def fake_render(*args, **kwargs):
        return {
            "facts": str(facts),
            "audio_report": str(report),
            "audio": str(tmp_path / "audio.wav"),
            "questions": str(tmp_path / "questions.json"),
            "visual_video": str(visual_video),
            "variant_root": str(tmp_path / "variant"),
            "assignment_plan_path": str(tmp_path / "assignment_plan.json"),
            "assignment_request_path": str(tmp_path / "assignment_request.json"),
            "delivered_audio_layouts": {
                "status": "blocked",
                "reason": "declared ambisonics output is missing",
            },
            "declared_audio_delivery": declared_audio_delivery(member),
            "shared_visual_root": None,
        }

    monkeypatch.setattr(motion, "_render_state_column", fake_render)
    result = motion._run_state_audio_unit(
        item,
        context,
        tmp_path / "unit",
        output_root=tmp_path,
        results=[visual_result],
    )
    assert result["status"] == "blocked"
    assert "ambisonics" in result["reason"]
    assert result["outputs"]["delivered_audio_layouts"]["status"] == "blocked"


def test_cached_binaural_audio_cannot_satisfy_member_foa(tmp_path: Path):
    member = {
        "audio_layouts": [
            {"type": "binaural", "channel_count": 2, "role": "primary"},
            {"type": "ambisonics", "channel_count": 4, "role": "attached_view"},
        ],
        "foa_normalization": "sn3d",
    }
    context = {
        "contract": {"audio_columns": {"a0": ["v0_a0", "v1_a0"]}},
    }
    item = {"unit_id": "v1_a0"}
    report_path = tmp_path / "old_report.json"
    report_path.write_text("{}", encoding="utf-8")
    old_sibling = {
        "work_item_id": "foa_group/v0_a0:audio:01",
        "scope_id": "foa_group/v0_a0",
        "status": "pass",
        "facts": {"audio_report_path": str(report_path)},
        "outputs": {
            "acoustic_identity": {
                "audio_view": {
                    "audio_layouts": [
                        {"type": "binaural", "channel_count": 2, "role": "primary"}
                    ],
                    "foa_normalization": "native_n3d",
                }
            }
        },
    }
    result = native._audio_reuse_candidate(
        context,
        item,
        [old_sibling],
        column="a0",
        assignment_plan={"audio_events": [], "voice_bindings": []},
        acoustics={"audio_view": native._audio_view_fields(member)},
        neutral_readback="/tmp/not_reached.json",
    )
    assert result["reused"] is False
    assert result["rejected"][0]["reason"] == (
        "acoustic input or configuration identity differs"
    )
    declared = declared_audio_delivery(member)
    blocked = verify_delivered_audio_layouts(
        declared, {"ancillary_audio_outputs": []}
    )
    assert blocked["status"] == "blocked"
    assert "ambisonics" in blocked["undelivered_attached_view_layouts"]


def test_real_early_facts_define_the_late_motion_window():
    request = _state_request()
    window = motion.measured_motion_window_from_facts(
        _early_facts(),
        clock=_early_facts()["a0"]["time"],
        binding_motion=request["binding_motion"],
        reserve_tail_s=request["profile"]["reserve_tail_s"],
    )
    assert window["first_motion_frame"] == 89
    assert window["last_motion_frame"] == 141
    assert window["minimum_motion_frames"] == 30
    assert window["available_motion_frames"] == 53
    assert window["measured_wet_end_s"] == pytest.approx(5.8115)
    assert window["wet_tail_end_s_by_assignment"] == {
        "a0": pytest.approx(5.8115),
        "a1": pytest.approx(4.4355625),
    }
    assert window["boundary_formula"] == "ceil(max(wet_end_s) * fps) + 1"
    assert window["requested_terminal_tail_s"] == pytest.approx(3.0)


def test_overlong_measured_tail_is_rejected_before_late_path_sampling():
    request = _state_request()
    facts = _early_facts()
    facts["a0"]["audio"]["wet_tail_intervals"][0]["end_s"] = 9.5
    with pytest.raises(BindingNativeError, match="insufficient declared movement time"):
        motion.measured_motion_window_from_facts(
            facts,
            clock=facts["a0"]["time"],
            binding_motion=request["binding_motion"],
            reserve_tail_s=request["profile"]["reserve_tail_s"],
        )


def test_real_retained_state_preflights_pass_before_late_audio():
    captured = {
        visual_id: {"capture": str(STATE_ROOT / f"visual/{visual_id}/capture")}
        for visual_id in ("v0", "v1")
    }
    history = motion.native_state_visual_history_preflight(
        captured, _early_facts()
    )
    angle = motion.native_state_visual_angle_preflight(
        captured,
        _early_facts()["a0"],
        ["source1", "source2"],
        _state_request()["binding_motion"]["angle_tolerance_deg"],
    )
    assert history["status"] == "pass"
    assert len(history["checks"]) == 8
    assert angle["status"] == "pass"
    assert all(row["pass"] for row in angle["comparisons"])


def test_retained_state_assembly_keeps_clip_end_query_and_media_readback():
    spec = json.loads(
        (STATE_ROOT / "group_spec.json").read_text(encoding="utf-8")
    )
    assembled = json.loads(
        (STATE_ROOT / "assembled/binding_groups.json").read_text(
            encoding="utf-8"
        )
    )
    group = spec["groups"][0]
    assert group["task_family"] == "cross_time_state"
    assert group["query"] == {"event_number": 1, "query_anchor": "clip_end"}
    assert group["angle_tolerance_deg"] == 10
    assert assembled["status"] == "research_candidate"
    assert assembled["group_count"] == 1
    assert assembled["world_count"] == 1
    assert assembled["sample_count"] == 4
    assert assembled["validation"] == "media_checked"


def test_retained_state_camera_drift_is_rejected(tmp_path: Path):
    left = STATE_ROOT / "visual/v0/capture/neutral_readback.json"
    changed = json.loads(
        (STATE_ROOT / "visual/v1/capture/neutral_readback.json").read_text(
            encoding="utf-8"
        )
    )
    changed["camera"][0]["position_m"][0] += 1.0
    right = tmp_path / "changed_readback.json"
    right.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(BindingNativeError, match="group native camera"):
        compare_group_native_visuals(
            {"neutral_readback": left},
            {"neutral_readback": right},
            contract=_native_contract(),
            left_unit_id="v0_capture",
            right_unit_id="v1_capture",
        )


def test_state_plan_helpers_preserve_explicit_qa_request_and_bind_slots():
    request = {
        "episode_id": "state_member",
        "qa_ids": ["QA-13", "QA-17"],
        "qa_targets": [{"qa_id": "QA-13", "event": {"kind": "clip_tail"}}],
        "source_asset_ids": ["asset_blue", "asset_green"],
    }
    context = {"member_requests": {"state_member": request}}
    item = {"member_request_ids": ["state_member"]}
    selected = motion._state_member_request(context, item)
    assert selected == request
    plan = {
        "scene": {"room_id": "room"},
        "clock": {"frame_count": 2, "frame_rate_hz": 15, "sample_rate_hz": 16000},
        "visual_plan": {
            "camera": {"motion": "static"},
            "actors": [
                {"actor_id": "source1", "asset_id": "asset_blue"},
                {"actor_id": "source2", "asset_id": "asset_green"},
            ],
            "frames": [
                {"actor_states": [
                    {"actor_id": "source1", "asset_id": "asset_blue",
                     "root_transform": {"translation_m": [0, 0, 0]},
                     "planned_emitter_m": [0, 1, 0], "moving": False},
                    {"actor_id": "source2", "asset_id": "asset_green",
                     "root_transform": {"translation_m": [1, 0, 0]},
                     "planned_emitter_m": [1, 1, 0], "moving": False},
                ]},
                {"actor_states": [
                    {"actor_id": "source1", "asset_id": "asset_blue",
                     "root_transform": {"translation_m": [0, 0, 0]},
                     "planned_emitter_m": [0, 1, 0], "moving": False},
                    {"actor_id": "source2", "asset_id": "asset_green",
                     "root_transform": {"translation_m": [1, 0, 0]},
                     "planned_emitter_m": [1, 1, 0], "moving": False},
                ]},
            ],
        },
    }
    motion._state_validate_static_base(plan, {
        **request, "room_id": "room", "frame_count": 2,
        "frame_rate_hz": 15, "sample_rate_hz": 16000,
    })
    motion._validate_stationary_plan(plan)
    swapped = motion._state_bind_plan_request(
        plan, {**request, "source_asset_ids": ["asset_green", "asset_blue"]}
    )
    assert [actor["asset_id"] for actor in swapped["visual_plan"]["actors"]] == [
        "asset_green", "asset_blue"
    ]
    assert selected["qa_ids"] == ["QA-13", "QA-17"]


def test_state_plan_requires_an_existing_base_and_does_not_start_a_planner():
    with pytest.raises(BindingNativeError, match="no native planner is started implicitly"):
        motion._state_base_plan_path({}, {})


def test_stage_result_window_reads_both_real_fact_files():
    request = _state_request()
    results = []
    for assignment in ("a0", "a1"):
        results.append({
            "work_item_id": f"state/v0_{assignment}:audio:01",
            "scope_id": f"state/v0_{assignment}",
            "status": "pass",
            "facts": {
                "facts_path": str(
                    STATE_ROOT
                    / f"variants/v0_{assignment}/delivery/facts.json"
                )
            },
            "outputs": {},
        })
    window = motion.motion_window_from_stage_results(
        results,
        clock=_early_facts()["a0"]["time"],
        binding_motion=request["binding_motion"],
        reserve_tail_s=request["profile"]["reserve_tail_s"],
    )
    assert (window["first_motion_frame"], window["last_motion_frame"]) == (89, 141)


def test_state_dispatch_wrapper_passes_only_its_scoped_runner_map(
    monkeypatch, tmp_path: Path
):
    seen = {}

    def fake_dispatch(item, context, **kwargs):
        seen.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(motion.native, "run_group_stage_work_item", fake_dispatch)
    result = motion.run_group_stage_work_item(
        {"unit_id": "v0"},
        {"group_id": "state"},
        output_root=tmp_path,
    )
    assert result == {"status": "pass"}
    assert seen["stage_runners"] is motion.STATE_STAGE_RUNNERS
    assert set(seen["stage_runners"]) == {
        "visual_plan", "visual_capture", "audio", "assembly"
    }


def test_state_module_reuses_native_context_and_result_loader(monkeypatch):
    sentinel = {"schema": "sentinel"}
    monkeypatch.setattr(motion.native, "group_stage_context",
                        lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(motion.native, "load_group_stage_results",
                        lambda *args, **kwargs: [sentinel])
    assert motion.group_stage_context() is sentinel
    assert motion.load_group_stage_results("root", "state") == [sentinel]


def test_retained_identity_reference_stays_a_world_comparison():
    v0 = IDENTITY_ROOT / "visual/v0/plan/episode_plan.json"
    v1 = IDENTITY_ROOT / "visual/v1/plan/episode_plan.json"
    contract = {
        "plan_equivalence": "world",
        "query_identity_policy": "exact",
        "plan_equivalence_rule": "identical_planned_scene_clock_and_camera_route_only",
    }
    from avengine.dataset.binding_group_native import compare_group_visual_plans
    assert compare_group_visual_plans(v0, v1, contract=contract)["status"] == "pass"


# --- generic audio recovery reachability -------------------------------------
# The ordinary runner recovers an interrupted audio attempt by looking
# recover_rendered_audio_attempt up on the recipe module
# (production_runner._invoke_recipe_audio_recovery). A module without it gets no
# recovery at all and the runner says nothing, so both the name and the attempt
# layout it resolves are part of this recipe's contract.
STATE_MATERIALIZED_ATTEMPT = Path(
    "/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/"
    "binding_v1_parallel_20260910/TAKEOVER/T08_RELATION/"
    "attempt_20260910T205814Z_pid771755/state_audio_fresh/v0_a0/audio/attempt_01"
)


def test_recipe_module_exposes_the_runner_audio_recovery_hook():
    from avengine.dataset.production_runner import recipe_binding
    import importlib
    binding = recipe_binding("cross_time_state")
    module = importlib.import_module(binding.module)
    recover = getattr(module, "recover_rendered_audio_attempt", None)
    assert callable(recover), binding.module
    assert "recover_rendered_audio_attempt" in module.__all__


def test_audio_recovery_resolves_this_recipe_variant_layout(tmp_path: Path):
    module = motion
    resolve = module._recipe_audio_attempt_root
    # episode layout, the shared native one
    episode = tmp_path / "attempt_01/episode"
    episode.mkdir(parents=True)
    (episode / "request.json").write_text("{}", encoding="utf-8")
    assert resolve(tmp_path / "attempt_01", {"unit_id": "v0_a0"}, {}) == episode
    # variants/<unit_id> layout, the one this recipe materializes
    variant = tmp_path / "attempt_02/variants/v0_a1"
    variant.mkdir(parents=True)
    (variant / "request.json").write_text("{}", encoding="utf-8")
    assert resolve(tmp_path / "attempt_02", {"unit_id": "v0_a1"}, {}) == variant
    # the unit id may only arrive through the lineage the runner builds
    assert resolve(tmp_path / "attempt_02", {}, {"unit_id": "v0_a1"}) == variant
    # a bare attempt root names what it looked at instead of failing deep inside
    with pytest.raises(Exception) as raised:
        resolve(tmp_path / "attempt_03", {"unit_id": "v0_a0"}, {})
    assert "looked at" in str(raised.value)


@pytest.mark.skipif(
    not STATE_MATERIALIZED_ATTEMPT.is_dir(),
    reason="retained materialized state audio attempt is not present",
)
def test_audio_recovery_delegation_reaches_the_shared_receipt_check(tmp_path: Path):
    module = motion
    item = {"unit_id": "v0_a0", "scope_id": "cross_time_state_hm3d_0001/v0_a0",
            "group_id": "cross_time_state_hm3d_0001"}
    # This retained attempt was materialized but never rendered, so the shared
    # implementation has to reject it by name rather than invent a delivery.
    with pytest.raises(native.BindingNativeError) as raised:
        module.recover_rendered_audio_attempt(
            item, {"group_id": "cross_time_state_hm3d_0001"},
            output_root=tmp_path / "recovery",
            previous_attempt_root=STATE_MATERIALIZED_ATTEMPT,
            lineage={"scope_id": "cross_time_state_hm3d_0001/v0_a0",
                     "unit_id": "v0_a0"},
        )
    message = str(raised.value)
    assert "no raw native audio receipt" in message
    assert message.endswith("variants/v0_a0")
    assert not (tmp_path / "recovery").exists()


# --- state variant execution materialization ---------------------------------
# M23 stopped here: the v0 plan unit reported a directory holding only
# episode_plan.json, run_visual_capture_unit symlinked that directory, and
# capture_command refused because it found no Habitat execution inputs. The
# base plan is not a stand-in: it differs from the variant.
M23_PLAN_ATTEMPT = REPOSITORY / (
    "tmp/binding_v1_parallel_20260910/TAKEOVER/M23_STATE_QUALIFICATION/"
    "attempt_20260911T014229Z_pid941113/run/work/"
    "takeover_cross_time_state_mp3d_g01/v0/plan/attempt_01"
)
RETAINED_STATE_GROUP = Path(
    "/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/"
    "binding_dataset_20260909_v2/state_first/hm3d/group_v1/visual"
)


@pytest.mark.skipif(
    not (M23_PLAN_ATTEMPT / "episode/plan/episode_plan.json").is_file(),
    reason="the M23 state plan attempt is not present",
)
def test_thin_state_variant_plan_directory_is_reported_incomplete():
    thin = M23_PLAN_ATTEMPT / "episode/plan/episode_plan.json"
    status = motion.state_variant_materialization_status(thin)
    assert status["status"] == "incomplete"
    assert status["room_family"] == "mp3d"
    assert status["present"] == ["episode_plan.json"]
    assert status["missing"] == [
        "audio_events.json",
        "voice_bindings.json",
        "habitat_room_manifest.json",
        "habitat_execution/case_manifest.json",
        "habitat_execution/m1_capture_request.json",
    ]


@pytest.mark.skipif(
    not (M23_PLAN_ATTEMPT / "episode/plan/episode_plan.json").is_file(),
    reason="the M23 state plan attempt is not present",
)
def test_repairing_the_real_m23_variant_yields_a_capturable_tree(tmp_path: Path):
    thin = M23_PLAN_ATTEMPT / "episode/plan/episode_plan.json"
    before = sorted(p.name for p in thin.parent.iterdir())
    repaired = motion.repair_state_variant_materialization(
        thin, tmp_path / "repair/episode"
    )
    assert repaired["status"] == "pass"
    assert repaired["native_visual_worlds_created"] == 0
    assert repaired["native_acoustic_contexts_created"] == 0
    # the reported directory of the attempt that already passed stays read-only
    assert sorted(p.name for p in thin.parent.iterdir()) == before == [
        "episode_plan.json"
    ]
    assert motion.state_variant_materialization_status(
        repaired["episode_plan_path"]
    )["missing"] == []
    # camera, clock, audio program and tracks are the variant's, not the base's
    evidence = repaired["variant_evidence"]
    assert evidence["episode_plan_equals_variant"] is True
    assert evidence["episode_plan_differs_from_base"] is True
    assert evidence["audio_events_equals_variant"] is True
    assert evidence["audio_events_differs_from_base"] is True
    assert evidence["voice_bindings_equals_variant"] is True
    assert evidence["camera_equals_variant"] is True
    assert evidence["clock_equals_variant"] is True
    assert evidence["case_manifest_clock_equals_variant"] is True
    assert evidence["tracks"]["all_tracks_match_the_variant"] is True
    for row in evidence["tracks"]["by_actor"].values():
        assert row["frame_count"] == 150
        assert row["max_delta_to_variant_m"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(
    not (M23_PLAN_ATTEMPT / "episode/plan/episode_plan.json").is_file(),
    reason="the M23 state plan attempt is not present",
)
def test_capture_command_accepts_the_repaired_variant_and_still_refuses_the_thin_one(
    tmp_path: Path,
):
    import sys
    sys.path.insert(0, str(REPOSITORY / "tools/studio"))
    import run_qa_episode as qa

    thin = M23_PLAN_ATTEMPT / "episode/plan/episode_plan.json"
    request = json.loads(
        (M23_PLAN_ATTEMPT / "request.json").read_text(encoding="utf-8")
    )
    repaired = motion.repair_state_variant_materialization(
        thin, tmp_path / "repair/episode"
    )
    good = tmp_path / "capture_good"
    good.mkdir()
    (good / "plan").symlink_to(
        Path(repaired["episode_plan_path"]).parent, target_is_directory=True
    )
    argv = qa.capture_command(dict(request), good)
    flags = {argv[index]: argv[index + 1] for index in range(len(argv) - 1)}
    assert flags["--case-manifest"].endswith(
        "habitat_execution/case_manifest.json"
    )
    assert flags["--m1-request"].endswith(
        "habitat_execution/m1_capture_request.json"
    )
    assert flags["--room-manifest"].endswith("habitat_room_manifest.json")
    assert flags["--episode-plan"].endswith("plan/episode_plan.json")
    # the original refusal is still a refusal, so nothing was loosened
    bad = tmp_path / "capture_thin"
    bad.mkdir()
    (bad / "plan").symlink_to(thin.parent.resolve(), target_is_directory=True)
    with pytest.raises(Exception) as raised:
        qa.capture_command(dict(request), bad)
    assert "materialized case_manifest" in str(raised.value)


@pytest.mark.skipif(
    not (RETAINED_STATE_GROUP / "v1/plan/habitat_execution/tracks").is_dir(),
    reason="the retained state group is not present",
)
def test_late_plan_v1_materialized_tracks_follow_v1_and_not_v0():
    v0 = json.loads(
        (RETAINED_STATE_GROUP / "v0/plan/episode_plan.json").read_text(
            encoding="utf-8"
        )
    )
    v1 = json.loads(
        (RETAINED_STATE_GROUP / "v1/plan/episode_plan.json").read_text(
            encoding="utf-8"
        )
    )
    checks = motion._state_variant_materialization_checks(
        v1, v0, RETAINED_STATE_GROUP / "v1"
    )
    assert checks["episode_plan_equals_variant"] is True
    assert checks["camera_equals_variant"] is True
    tracks = checks["tracks"]
    assert tracks["all_tracks_match_the_variant"] is True
    # the late plan really moves the actors, so a base-derived tree would show
    assert all(
        row["max_delta_to_base_m"] > 1.0 for row in tracks["by_actor"].values()
    )


def test_repair_refuses_missing_and_mismatched_inputs(tmp_path: Path):
    # no plan at all
    with pytest.raises(native.BindingNativeError):
        motion.state_variant_materialization_status(tmp_path / "absent.json")
    # a plan with no resolvable base episode
    lonely = tmp_path / "lonely/plan"
    lonely.mkdir(parents=True)
    plan = {
        "clock": {"frame_count": 1, "frame_rate_hz": 15.0, "sample_rate_hz": 16000},
        "resources": {"room_package": {"family": "mp3d", "renderer": "habitat"}},
        "visual_plan": {"camera": {}, "actors": [], "frames": []},
        "request": {},
    }
    (lonely / "episode_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(native.BindingNativeError) as raised:
        motion.repair_state_variant_materialization(
            lonely / "episode_plan.json", tmp_path / "out/episode"
        )
    assert "looked at" in str(raised.value)
    # an unsupported room family is named rather than silently materialized
    plan["resources"]["room_package"]["family"] = "not_a_room_family"
    other = tmp_path / "other/plan"
    other.mkdir(parents=True)
    (other / "episode_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(native.BindingNativeError):
        motion.state_variant_materialization_status(other / "episode_plan.json")


@pytest.mark.skipif(
    not (M23_PLAN_ATTEMPT / "episode/plan/episode_plan.json").is_file(),
    reason="the M23 state plan attempt is not present",
)
def test_repair_refuses_a_base_episode_from_another_room(tmp_path: Path):
    thin = M23_PLAN_ATTEMPT / "episode/plan/episode_plan.json"
    foreign = tmp_path / "foreign"
    (foreign / "plan").mkdir(parents=True)
    base = json.loads(
        (M23_PLAN_ATTEMPT / "base_plan/plan/episode_plan.json").read_text(
            encoding="utf-8"
        )
    )
    base["scene"] = {"room_id": "somewhere_else_v1", "scene_id": "somewhere_else_v1"}
    (foreign / "plan/episode_plan.json").write_text(
        json.dumps(base), encoding="utf-8"
    )
    with pytest.raises(native.BindingNativeError) as raised:
        motion.repair_state_variant_materialization(
            thin, tmp_path / "out/episode", base_root=foreign
        )
    assert "another" in str(raised.value)
    assert not (tmp_path / "out").exists()
