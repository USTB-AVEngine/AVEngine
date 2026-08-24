"""CPU-only tests for the multi-actor apply-plan boundary."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.m5.current_visual_multi_actor import (
    ActorTrackInput,
    CurrentVisualMultiActorError,
    MultiActorCaseInput,
    _safe_relative_file,
    build_apply_plan,
    classify_collision_contacts,
    load_multi_actor_case,
)


def _transform(x: float) -> dict[str, list[float]]:
    return {
        "translation_m": [x, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _fake_track(
    *, actor_id: str, source_id: str, asset_id: str, joint_count: int
) -> ActorTrackInput:
    joints = [
        {
            "joint_id": f"joint{index}",
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        for index in range(joint_count)
    ]
    states = []
    frames = []
    positions = []
    for index in range(75):
        x = float(index) * 0.01
        positions.append([x, 0.0, 0.0])
        states.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "action_id": "walk",
                "action_sample_index": index,
                "root_transform": _transform(x),
                "joint_states": joints,
            }
        )
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "action_id": "walk",
                "action_sample_index": index,
                "route_source_center_m": [x, 0.0, 0.0],
                "planned_world_from_actor": _transform(x),
                "derived_world_from_skin_root": _transform(x),
            }
        )
    track = {
        "route_source_center_plan": {
            "region_instance_id": "room:region:000",
            "route_family_id": "family0",
            "motion_case": "both_moving",
            "source_id": source_id,
            "positions_m": positions,
        },
        "states": states,
        "frames": frames,
    }
    m2_inputs = SimpleNamespace(asset={"asset_id": asset_id})
    bundle = SimpleNamespace(actor_from_skin_root=np.eye(4, dtype=np.float64))
    return ActorTrackInput(
        actor_id=actor_id,
        source_id=source_id,
        semantic_id=210 if actor_id == "actor0" else 211,
        emitter_link_name="mouth",
        track_path=Path(f"/tmp/{actor_id}.json"),
        track=track,
        m2_inputs=m2_inputs,
        bundle=bundle,
    )


def _fake_case(left: ActorTrackInput, right: ActorTrackInput) -> MultiActorCaseInput:
    return MultiActorCaseInput(
        manifest_path=Path("/tmp/case_manifest.json"),
        manifest={
            "region_instance_id": "room:region:000",
            "route_family_id": "family0",
            "motion_case": "both_moving",
        },
        actor_tracks=(left, right),
        camera_m1_request_path=Path("/tmp/camera.json"),
    )


def _contact(a: int, b: int) -> dict[str, object]:
    return {
        "object_id_a": a,
        "object_id_b": b,
        "link_id_a": 1,
        "link_id_b": 2,
        "position_on_a_in_ws": [0.0, 0.0, 0.0],
        "position_on_b_in_ws": [0.0, 0.0, 0.0],
        "contact_normal_on_b_in_ws": [0.0, 1.0, 0.0],
        "contact_distance": 0.0,
        "is_active": True,
        "normal_force": 0.0,
    }


def test_two_independent_tracks_keep_joint_axes_and_no_fixed_offsets() -> None:
    left = _fake_track(
        actor_id="actor0", source_id="source1", asset_id="asset-a", joint_count=2
    )
    right = _fake_track(
        actor_id="actor1", source_id="source2", asset_id="asset-b", joint_count=3
    )
    plan = build_apply_plan(_fake_case(left, right))
    assert plan.case_id == "family0::both_moving"
    assert [len(track.frames[0].joint_states) for track in plan.actor_tracks] == [2, 3]
    assert plan.actor_tracks[0].frames[10].planned_world_from_actor[0][
        3
    ] == pytest.approx(0.1)
    assert plan.actor_tracks[1].frames[10].planned_world_from_actor[0][
        3
    ] == pytest.approx(0.1)
    assert (
        "CURRENT_ACTOR_OFFSETS_M"
        not in Path(
            __import__(
                "avengine.m5.current_visual_multi_actor", fromlist=["__file__"]
            ).__file__
        ).read_text()
    )


def test_camera_path_does_not_change_actor_plan() -> None:
    case = _fake_case(
        _fake_track(
            actor_id="actor0", source_id="source1", asset_id="a", joint_count=1
        ),
        _fake_track(
            actor_id="actor1", source_id="source2", asset_id="b", joint_count=1
        ),
    )
    first = build_apply_plan(case)
    second = build_apply_plan(
        replace(case, camera_m1_request_path=Path("/tmp/other.json"))
    )
    assert first.actor_tracks == second.actor_tracks
    assert first.camera_m1_request_path != second.camera_m1_request_path


def test_different_actor_emitter_links_remain_independent() -> None:
    left = _fake_track(
        actor_id="actor0", source_id="source1", asset_id="a", joint_count=1
    )
    right = replace(
        _fake_track(
            actor_id="actor1", source_id="source2", asset_id="b", joint_count=1
        ),
        emitter_link_name="different-mouth",
    )
    plan = build_apply_plan(_fake_case(left, right))
    assert [track.emitter_link_name for track in plan.actor_tracks] == [
        "mouth",
        "different-mouth",
    ]


def test_clock_damage_fails_before_two_actor_plan() -> None:
    left = _fake_track(
        actor_id="actor0", source_id="source1", asset_id="a", joint_count=1
    )
    right = _fake_track(
        actor_id="actor1", source_id="source2", asset_id="b", joint_count=1
    )
    right.track["frames"][7]["pts_ticks"] = 99
    with pytest.raises(CurrentVisualMultiActorError, match="clock"):
        build_apply_plan(_fake_case(left, right))


def test_safe_relative_file_rejects_escape_and_symlink(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    manifest = case_dir / "case_manifest.json"
    manifest.write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(CurrentVisualMultiActorError, match="escape"):
        _safe_relative_file(manifest, "../outside.json", owner="track")
    link = case_dir / "link.json"
    link.symlink_to(outside)
    with pytest.raises(CurrentVisualMultiActorError, match="symlink"):
        _safe_relative_file(manifest, "link.json", owner="track")


def test_collision_classifier_separates_pair_stage_and_unrelated() -> None:
    summary = classify_collision_contacts(
        [_contact(10, 11), _contact(10, 99), _contact(88, 99)],
        runtime_actor_object_ids=[10, 11],
        stage_object_id=99,
    )
    assert len(summary.actor_actor_contacts) == 1
    assert len(summary.actor_stage_contacts) == 1
    assert len(summary.actor_other_contacts) == 0
    assert len(summary.unrelated_contacts) == 1
    assert summary.actor_pair_collision_free is False
    assert (
        summary.actor_stage_contacts[0]["stage_contact_recorded_unclassified"] is True
    )


def test_zero_object_id_is_not_a_plan_or_collision_legality_gate() -> None:
    summary = classify_collision_contacts(
        [_contact(10, 0)], runtime_actor_object_ids=[10, 11]
    )
    assert summary.actor_pair_collision_free is True
    assert summary.actor_stage_contacts


def test_non_stage_actor_contact_is_retained_separately() -> None:
    summary = classify_collision_contacts(
        [_contact(10, 99)], runtime_actor_object_ids=[10, 11], stage_object_id=0
    )
    assert not summary.actor_stage_contacts
    assert len(summary.actor_other_contacts) == 1
    assert summary.actor_other_contacts[0]["scope"] == "actor_other"


def test_collision_contact_fields_must_be_finite_and_typed() -> None:
    damaged = _contact(10, 11)
    damaged["contact_distance"] = float("nan")
    with pytest.raises(CurrentVisualMultiActorError, match="finite"):
        classify_collision_contacts([damaged], runtime_actor_object_ids=[10, 11])
    damaged = _contact(10, 11)
    damaged["link_id_a"] = True
    with pytest.raises(CurrentVisualMultiActorError, match="integer"):
        classify_collision_contacts([damaged], runtime_actor_object_ids=[10, 11])


def test_case_manifest_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    target = tmp_path / "case.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(CurrentVisualMultiActorError, match="symlink"):
        load_multi_actor_case(link)


def test_case_manifest_rejects_cross_wired_actor_source_axes(tmp_path: Path) -> None:
    camera = tmp_path / "camera.json"
    camera.write_text("{}")
    manifest = tmp_path / "case_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "research_only": True,
                "formal_m2_request": False,
                "episode_counted": False,
                "receipt": {
                    "status": "research_only",
                    "formal_m2_request": False,
                    "episode_counted": False,
                },
                "region_instance_id": "room:region:000",
                "route_family_id": "family0",
                "motion_case": "both_moving",
                "native_pending": {
                    key: None
                    for key in (
                        "support_contact",
                        "emitter_actual_position_m",
                        "articulated_collision",
                        "object_id",
                        "native_execution",
                        "rlr",
                    )
                },
                "camera_m1_reference": {
                    "status": "resolved",
                    "resolved_path": str(camera),
                },
                "route_source_center_plan": {
                    "camera_participation_in_root_generation": False,
                },
                "actors": [
                    {"actor_id": "actor0", "source_id": "source2"},
                    {"actor_id": "actor1", "source_id": "source1"},
                ],
            }
        )
    )
    with pytest.raises(CurrentVisualMultiActorError, match="cross-wire"):
        load_multi_actor_case(manifest)


@pytest.mark.skipif(
    not Path("/data/jzy/tmp/mp3d_region_actor_tracks_2b0d4f9_r4/cases").is_dir(),
    reason="server retained r4 artifact is unavailable",
)
def test_retained_r4_case_passes_exact_loader_and_apply_plan() -> None:
    case_path = next(
        Path("/data/jzy/tmp/mp3d_region_actor_tracks_2b0d4f9_r4/cases").glob(
            "*/case_manifest.json"
        )
    )
    plan = build_apply_plan(load_multi_actor_case(case_path))
    assert plan.frame_count == 75
    assert len(plan.actor_tracks) == 2
    assert all(len(track.frames) == 75 for track in plan.actor_tracks)
