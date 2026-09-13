"""Fixed-camera visibility crossings and occlusion sequences.

Two things these checks exist to keep true.

The screen never confirms a pixel state.  A nine-point body sample set is
sound when it *refutes* - one in-frustum unblocked sample proves the subject is
neither out of view nor fully occluded - and can never prove that no pixel of a
silhouette survives.  So every positive verdict here is ``consistent``, meaning
"worth rendering", and the pass/fail belongs to
:func:`accept_native_visibility` on the renderer's own pixel truth.

A subject that cannot walk is not a reason to refuse a group.  A bolted-down
device blocks exactly one requirement - the crossing that needs the subject's
own movement - and keeps every occlusion requirement, because a walking
occluder or a piece of scene supplies the change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.dataset.source_capabilities import (
    STATE_AVAILABLE,
    STATE_EVIDENCE_MISSING,
    STATE_NOT_APPLICABLE,
    STATE_NOT_IMPLEMENTED,
)
from avengine.qa import generation_conditions as gc
from avengine.qa.answerability import MeshHandle, line_of_sight
from avengine.qa.pixel_visibility import compile_pixel_visibility_truth
from avengine.rooms import conditioned_visibility as cv
from avengine.rooms.room_providers import (
    load_room_catalog,
    resolve_catalog_room,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "examples/rooms/packages/catalog.json"

FRAME_RATE_HZ = 15.0
# The registered rocketbox mouth anchor and border collie muzzle anchor, so the
# body reach in these checks is a real registry value rather than a constant
# invented for the test.
HUMAN_MOUTH_OFFSET_M = [0.0, 1.641311, 0.0]
COLLIE_MUZZLE_OFFSET_M = [0.40465284499021825, 0.6465226403698251, 0.0]


def camera(resolution_hw=(720, 1280), fov_deg=85.0, candidate_id="grid_00000_yaw_000"):
    """A fixed camera at eye height looking down -Z, screen right = +X."""

    return cv.CameraPose(
        candidate_id=candidate_id,
        position_m=(0.0, 1.55, 0.0),
        forward=(0.0, 0.0, -1.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        horizontal_fov_deg=fov_deg,
        resolution_hw=tuple(resolution_hw),
    )


def track(instance_id, positions, *, offset=HUMAN_MOUTH_OFFSET_M, entity_class="articulated_human"):
    return cv.ActorTrack(
        instance_id=instance_id,
        positions_m=np.asarray(positions, dtype=float),
        body=cv.body_proxy_from_emitter_anchor(offset),
        entity_class=entity_class,
    )


def hold(point, frames):
    return np.repeat(np.asarray([point], dtype=float), frames, axis=0)


def stitch(*segments):
    return np.concatenate([np.asarray(segment, dtype=float) for segment in segments])


# A subject 4 m in front of the camera is inside the horizontal frustum out to
# 4 * tan(42.5 deg) = 3.67 m, so x = 1.0 is well inside and x = 6.0 is well out.
INSIDE_RIGHT = (1.0, 0.0, -4.0)
INSIDE_LEFT = (-1.0, 0.0, -4.0)
OUTSIDE_RIGHT = (6.0, 0.0, -4.0)


def wall_mesh(z_m=-2.0, half_width=4.0, height=3.0):
    """A single vertical quad across the view at ``z_m``."""

    vertices = np.asarray(
        [
            [-half_width, -0.5, z_m],
            [half_width, -0.5, z_m],
            [half_width, height, z_m],
            [-half_width, height, z_m],
        ],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return MeshHandle(vertices, triangles, {"source": "test_wall_quad"})


def open_mesh():
    """Geometry that exists but never crosses a sight line in these checks."""

    vertices = np.asarray(
        [
            [-40.0, -6.0, 40.0],
            [-39.0, -6.0, 40.0],
            [-39.0, -5.0, 40.0],
        ],
        dtype=float,
    )
    return MeshHandle(vertices, np.asarray([[0, 1, 2]], dtype=np.int64), {"source": "far"})


# ---------------------------------------------------------------------------
# Camera and body contracts
# ---------------------------------------------------------------------------


def test_camera_pose_refuses_a_moving_camera() -> None:
    with pytest.raises(cv.ConditionedVisibilityError, match="fixed cameras only"):
        cv.CameraPose.from_mapping(
            {
                "candidate_id": "c",
                "position_m": [0.0, 1.5, 0.0],
                "basis": {"forward": [0, 0, -1], "right": [1, 0, 0], "up": [0, 1, 0]},
                "horizontal_fov_deg": 85.0,
                "resolution_hw": [720, 1280],
                "motion": "orbit",
            }
        )


def test_camera_centre_and_dead_zone_are_the_judges_own() -> None:
    pose = camera()
    assert pose.center_column_px == (1280 - 1) / 2
    assert pose.dead_zone_px == max(1.0, 1280 * 0.02)
    narrow = camera(resolution_hw=(8, 16))
    assert narrow.dead_zone_px == 1.0


def test_camera_mapping_round_trips_the_sampler_shape() -> None:
    pose = camera()
    again = cv.CameraPose.from_mapping(pose.as_report())
    assert again == pose


def test_entry_side_sign_follows_the_camera_right_basis() -> None:
    pose = camera()
    right = cv.screen_visibility_series(
        camera=pose,
        tracks=[track("source1", hold(INSIDE_RIGHT, 2))],
        subject="source1",
        cast_rays=False,
    )
    left = cv.screen_visibility_series(
        camera=pose,
        tracks=[track("source1", hold(INSIDE_LEFT, 2))],
        subject="source1",
        cast_rays=False,
    )
    assert right["frames"][0]["side"] == "right"
    assert right["frames"][0]["predicted_column_offset_px"] > 0.0
    assert left["frames"][0]["side"] == "left"
    assert left["frames"][0]["predicted_column_offset_px"] < 0.0
    # Both sit far outside the judge's dead zone, so the sign is the answer.
    assert right["frames"][0]["side_beyond_dead_zone"]
    assert left["frames"][0]["side_beyond_dead_zone"]


def test_body_proxy_records_where_its_reach_came_from() -> None:
    registered = cv.body_proxy_from_emitter_anchor(HUMAN_MOUTH_OFFSET_M)
    assert registered.height_source == "registered_emitter_anchor_height"
    assert registered.height_m == pytest.approx(HUMAN_MOUTH_OFFSET_M[1])
    assert registered.width_source == "placeholder_default"
    animal = cv.body_proxy_from_emitter_anchor(COLLIE_MUZZLE_OFFSET_M)
    assert animal.height_m == pytest.approx(COLLIE_MUZZLE_OFFSET_M[1])
    assert animal.height_m < registered.height_m
    missing = cv.body_proxy_from_emitter_anchor(None)
    assert missing.height_source == "placeholder_default"
    declared = cv.body_proxy_from_emitter_anchor(HUMAN_MOUTH_OFFSET_M, width_m=0.55)
    assert declared.width_source == "caller_declared"
    assert "not the asset's reviewed geometry" in declared.as_report()["claim_boundary"]


def test_body_samples_span_the_width_perpendicular_to_the_sight_line() -> None:
    body = cv.BodyProxy(height_m=1.8, width_m=0.4)
    points = cv.body_sample_points((0.0, 1.55, 0.0), (0.0, 0.0, -4.0), body)
    assert points.shape == (9, 3)
    # The sight line runs along -Z, so the lateral span is on X only.
    assert set(np.round(points[:, 0], 6)) == {-0.2, 0.0, 0.2}
    assert np.allclose(np.round(points[:, 2], 6), -4.0)
    assert sorted(set(np.round(points[:, 1], 6))) == [0.36, 1.08, 1.8]

def test_full_body_envelope_reports_pose_coverage_and_rejects_proxy_false_out() -> None:
    positions = hold(OUTSIDE_RIGHT, 4)
    full_points = np.repeat(
        np.asarray([[[6.0, 1.2, -4.0], [1.0, 1.2, -4.0], [6.0, 1.8, -4.0]]]),
        4,
        axis=0,
    )
    envelope = cv.ActorBodyEnvelope(
        vertices_m=full_points,
        source_ref="registered/visual.glb",
        pose_coverage="axis_aligned_enclosure_of_all_declared_idle_walk_poses",
    )
    subject = cv.ActorTrack(
        instance_id="source1",
        positions_m=positions,
        body=cv.BodyProxy(height_m=1.6),
        body_envelope=envelope,
    )
    old_proxy = cv.screen_visibility_series(
        camera=camera(), tracks=[track("source1", positions)], subject="source1",
        cast_rays=False,
    )
    full = cv.screen_visibility_series(
        camera=camera(), tracks=[subject], subject="source1", cast_rays=False
    )
    assert envelope.as_report()["pose_coverage"].startswith("axis_aligned_enclosure")
    assert old_proxy["out_of_view_frames"] == 4
    assert full["out_of_view_frames"] == 0
    assert full["body_geometry"] == "full_body_envelope"
    frame = full["frames"][0]
    assert frame["in_view"] is True
    bbox = frame["projected_body_bbox_px"]
    assert bbox[0] < 1280 - 0.5
    assert bbox[2] > -0.5
    assert frame["state"] is None


def test_full_body_envelope_refuses_initial_partial_proxy_candidate() -> None:
    positions = stitch(hold(OUTSIDE_RIGHT, 10), hold(INSIDE_RIGHT, 30))
    partial_first = np.repeat(
        np.asarray([[[6.0, 1.2, -4.0], [1.0, 1.2, -4.0], [6.0, 1.8, -4.0]]]),
        10,
        axis=0,
    )
    inside = np.repeat(
        np.asarray([[[1.0, 1.2, -4.0], [1.1, 1.2, -4.0], [1.0, 1.8, -4.0]]]),
        30,
        axis=0,
    )
    envelope = cv.ActorBodyEnvelope(
        vertices_m=np.concatenate([partial_first, inside]),
        source_ref="registered/visual.glb",
        pose_coverage="axis_aligned_enclosure_of_all_declared_idle_walk_poses",
    )
    requirement = cv.VisibilityRequirement(
        kind="out_of_view_to_visible",
        subject="source1",
        side="right",
    )
    old = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[camera()],
        tracks=[track("source1", positions)],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=None,
    )
    full = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[camera()],
        tracks=[
            cv.ActorTrack(
                instance_id="source1",
                positions_m=positions,
                body=cv.BodyProxy(height_m=1.6),
                body_envelope=envelope,
            )
        ],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=None,
    )
    assert old["candidates"], old["refuted"]
    assert full["candidates"] == []
    assert full["refuted"][0]["tier"] == "frustum"


def test_full_body_envelope_keeps_a_true_crossing_candidate() -> None:
    positions = stitch(hold(OUTSIDE_RIGHT, 10), hold(INSIDE_RIGHT, 30))
    outside = np.repeat(
        np.asarray([[[6.0, 1.2, -4.0], [6.1, 1.2, -4.0], [6.0, 1.8, -4.0]]]),
        10,
        axis=0,
    )
    inside = np.repeat(
        np.asarray([[[1.0, 1.2, -4.0], [1.1, 1.2, -4.0], [1.0, 1.8, -4.0]]]),
        30,
        axis=0,
    )
    track_with_envelope = cv.ActorTrack(
        instance_id="source1",
        positions_m=positions,
        body=cv.BodyProxy(height_m=1.6),
        body_envelope=cv.ActorBodyEnvelope(
            vertices_m=np.concatenate([outside, inside]),
            source_ref="registered/visual.glb",
            pose_coverage="axis_aligned_enclosure_of_all_declared_idle_walk_poses",
        ),
    )
    solved = cv.solve_visibility_candidates(
        [
            cv.VisibilityRequirement(
                kind="out_of_view_to_visible",
                subject="source1",
                side="right",
            )
        ],
        camera_poses=[camera()],
        tracks=[track_with_envelope],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=None,
    )
    assert solved["candidates"], solved["refuted"]
    crossing = solved["candidates"][0]["requirement_screens"][0]["crossings"][0]
    assert crossing["entry_frame"] == 10
    assert crossing["initial_outside_side"] == "right"



# ---------------------------------------------------------------------------
# Requirements from P02 planning knobs
# ---------------------------------------------------------------------------


def test_requirements_from_planning_covers_the_recorded_knob_values() -> None:
    entry = cv.requirements_from_planning(
        {"visibility_transition": "out_of_view_to_visible", "entry_side": "left"},
        subject="source1",
        qa_id="QA-07",
    )
    assert [item.kind for item in entry] == ["out_of_view_to_visible"]
    assert entry[0].side == "left"
    assert entry[0].require_publishable_window

    for value, kind in (
        ("fully_occluded_then_visible", "fully_occluded_then_visible"),
        ("fully_occluded_without_return", "fully_occluded_without_return"),
        ("registered_occluder_visible", "registered_occluder_visible"),
    ):
        compiled = cv.requirements_from_planning(
            {"pixel_occlusion_transition": value}, subject="source1"
        )
        assert [item.kind for item in compiled] == [kind]
    negative = cv.requirements_from_planning(
        {"pixel_occlusion_transition": "fully_occluded_without_return"},
        subject="source1",
    )
    assert negative[0].require_complete_coverage

    state = cv.requirements_from_planning(
        {"anchor_visibility": "in_fov", "pixel_occlusion_transition": "fully_occluded"},
        subject="source1",
        evidence={"visibility_state": "fully_occluded"},
    )
    assert [(item.kind, item.state) for item in state] == [
        ("visibility_state", "fully_occluded")
    ]


def test_a_frustum_knob_alone_is_not_a_pixel_requirement() -> None:
    # anchor_visibility and anchor_line_of_sight stay what the sampler means by
    # them; only a condition whose evidence asks for a pixel state produces one.
    assert (
        cv.requirements_from_planning(
            {"anchor_visibility": "in_fov", "anchor_line_of_sight": "occluded"},
            subject="source1",
        )
        == ()
    )
    derived = cv.requirements_from_planning(
        {"anchor_line_of_sight": "occluded"},
        subject="source1",
        evidence={"previous_state": "visible_occluded", "current_state": "visible_clear"},
    )
    assert [item.kind for item in derived] == ["visible_occluded_to_visible_clear"]


def test_requirements_from_planning_refuses_an_unknown_transition() -> None:
    with pytest.raises(cv.ConditionedVisibilityError, match="visibility_transition"):
        cv.requirements_from_planning(
            {"visibility_transition": "visible_to_out_of_view"}, subject="source1"
        )
    with pytest.raises(cv.ConditionedVisibilityError, match="pixel_occlusion_transition"):
        cv.requirements_from_planning(
            {"pixel_occlusion_transition": "half_occluded"}, subject="source1"
        )


def test_knob_support_answers_exactly_the_two_recorded_gaps() -> None:
    support = cv.visibility_knob_support()
    implemented = set(support["implemented"])
    assert implemented == {"visibility_transition", "pixel_occlusion_transition"}
    # Implemented knobs no longer remain in the live gap table.
    assert implemented.isdisjoint(gc.KNOB_GAPS)
    for detail in support["implemented"].values():
        assert detail["solver"].endswith("solve_visibility_candidates")
        for kind in detail["requirement_kinds"]:
            assert kind in cv.REQUIREMENT_KINDS
    assert support["confirmation_authority"] == "native_pixel_witness"


def test_requirements_from_conditions_reads_compiled_condition_dicts() -> None:
    instances = [
        {"entity_instance_id": "source1", "asset_id": "asset_a"},
        {"entity_instance_id": "source2", "asset_id": "asset_b"},
    ]
    compiled = gc.compile_generation_conditions(
        {
            "qa_id": "QA-07",
            "branch": "right",
            "target_instance_ids": ["source1"],
            "competitor_instance_ids": ["source2"],
        },
        branch="right",
        instances=instances,
        registry=None,
    )
    payload = compiled.to_dict()
    requirements = cv.requirements_from_conditions(
        payload, observation_windows_by_subject={"source1": [[0, 40]]}
    )
    assert [item.kind for item in requirements] == ["out_of_view_to_visible"]
    assert requirements[0].subject == "source1"
    assert requirements[0].side == "right"
    assert requirements[0].observation_windows == ((0, 40),)
    assert requirements[0].source_condition_key == "entry_transition"


def test_requirements_from_conditions_covers_the_pixel_occlusion_types() -> None:
    instances = [
        {"entity_instance_id": "source1", "asset_id": "asset_a"},
        {"entity_instance_id": "source2", "asset_id": "asset_b"},
    ]
    seen: dict[str, list[str]] = {}
    for qa_id, branch in (("QA-09", "yes"), ("QA-09", "no"), ("QA-10", None), ("QA-11", None)):
        compiled = gc.compile_generation_conditions(
            {
                "qa_id": qa_id,
                "target_instance_ids": ["source1"],
                "competitor_instance_ids": ["source2"],
            },
            branch=branch,
            instances=instances,
            registry=None,
        )
        seen[f"{qa_id}:{branch}"] = [
            item.kind for item in cv.requirements_from_conditions(compiled.to_dict())
        ]
    assert seen["QA-09:yes"] == ["fully_occluded_then_visible"]
    assert seen["QA-09:no"] == ["fully_occluded_without_return"]
    assert seen["QA-10:None"] == ["registered_occluder_visible"]
    assert seen["QA-11:None"] == ["visible_occluded_to_visible_clear"]


# ---------------------------------------------------------------------------
# Tier one and tier two screening
# ---------------------------------------------------------------------------


def entry_clip(out_frames=10, in_frames=30, inside=INSIDE_RIGHT):
    return stitch(hold(OUTSIDE_RIGHT, out_frames), hold(inside, in_frames))


def test_entry_crossing_is_screened_with_its_side_and_sustain() -> None:
    subject = track("source1", entry_clip())
    series = cv.screen_visibility_series(
        camera=camera(), tracks=[subject], subject="source1", cast_rays=False
    )
    assert series["out_of_view_frames"] == 10
    assert series["in_view_frames"] == 30
    requirement = cv.VisibilityRequirement(
        kind="out_of_view_to_visible",
        subject="source1",
        side="right",
        require_publishable_window=True,
    )
    solved = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[camera()],
        tracks=[subject],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
    )
    assert solved["candidates"], solved["refuted"]
    screen = solved["candidates"][0]["requirement_screens"][0]
    assert screen["verdict"] == "consistent"
    assert screen["selected"]["entry_frame"] == 10
    assert screen["selected"]["side"] == "right"
    assert screen["selected"]["same_side_sustain_frames"] == 30
    assert screen["selected"]["publishable_extent"]["publishable"]


def test_an_exit_only_clip_is_refuted() -> None:
    # The retained native captures carry out_of_view frames only at the end of
    # the clip: a subject leaving the frame is not an entry, and reading that
    # as one is how a QA-07 candidate gets rendered and then thrown away.
    subject = track("source1", stitch(hold(INSIDE_RIGHT, 30), hold(OUTSIDE_RIGHT, 10)))
    solved = cv.solve_visibility_candidates(
        [cv.VisibilityRequirement(kind="out_of_view_to_visible", subject="source1")],
        camera_poses=[camera()],
        tracks=[subject],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
    )
    assert solved["candidates"] == []
    assert solved["status"] == "no_candidate_survived_screening"
    (refusal,) = solved["refuted"]
    assert refusal["tier"] == "frustum"
    assert "no frame where the subject goes from having no in-image body sample" in (
        refusal["reason"]
    )


def test_an_entry_too_late_to_print_is_refuted_with_the_frames_it_needed() -> None:
    subject = track("source1", stitch(hold(OUTSIDE_RIGHT, 34), hold(INSIDE_RIGHT, 6)))
    requirement = cv.VisibilityRequirement(
        kind="out_of_view_to_visible",
        subject="source1",
        require_publishable_window=True,
    )
    solved = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[camera()],
        tracks=[subject],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
    )
    assert solved["candidates"] == []
    (refusal,) = solved["refuted"]
    assert refusal["tier"] == "frustum_and_ray"
    assert "public time precision" in refusal["reason"]
    assert "needs an earlier crossing" in refusal["reason"]


def test_publishable_extent_at_fifteen_hertz_needs_about_a_second() -> None:
    view = cv._facts_view(
        states_by_instance={}, frame_count=150, frame_rate_hz=FRAME_RATE_HZ, precision=0
    )
    # Two consecutive frames satisfy the judge's state test and still cannot be
    # printed as a whole-second interval, which is the gap a solver that only
    # counts state frames walks into.
    assert cv.publishable_window(view, [10, 12])["publishable"] is False
    extent = cv.first_publishable_extent(view, 10, frame_count=150)
    assert extent["publishable"]
    assert extent["frames_needed"] == 20
    assert extent["public_seconds"] == [1.0, 2.0]
    # Nothing can be printed from a start that leaves less than that behind it.
    assert cv.first_publishable_extent(view, 148, frame_count=150)["publishable"] is False


def test_the_screen_never_confirms_out_of_view_or_a_full_occlusion() -> None:
    away = cv.screen_visibility_series(
        camera=camera(),
        tracks=[track("source1", hold(OUTSIDE_RIGHT, 1))],
        subject="source1",
        cast_rays=False,
    )
    frame = away["frames"][0]
    assert frame["state"] == "out_of_view"
    assert frame["decisive"] is False
    # See test_an_out_of_view_screen_call_refutes_nothing: this prediction was
    # measured wrong on retained captures, so it rules nothing out.
    assert frame["refutes"] == []

    behind_wall = cv.screen_visibility_series(
        camera=camera(),
        tracks=[track("source1", hold(INSIDE_RIGHT, 1))],
        subject="source1",
        mesh=wall_mesh(),
    )
    hidden = behind_wall["frames"][0]
    assert hidden["state"] == "fully_occluded"
    assert hidden["decisive"] is False
    assert hidden["refutes"] == ["out_of_view"]
    assert "only the native pixel pass" in hidden["reason"]
    assert away["confirmation_authority"] == "native_pixel_witness"
    assert set(cv.SCREEN_VERDICTS) == {"consistent", "undetermined", "refuted"}


def test_a_clear_screen_is_never_decisive_about_being_clear() -> None:
    series = cv.screen_visibility_series(
        camera=camera(),
        tracks=[track("source1", hold(INSIDE_RIGHT, 1))],
        subject="source1",
        mesh=open_mesh(),
        geometry_authority="visual_mesh",
    )
    frame = series["frames"][0]
    assert frame["state"] == "visible_clear"
    assert frame["decisive"] is False
    assert set(frame["refutes"]) == {"out_of_view", "fully_occluded"}


def test_only_the_geometry_that_renders_may_refute_an_occlusion() -> None:
    # Measured on a retained UE/SPEAR capture: the acoustic proxy missed the
    # occluder for all 150 frames the renderer called fully_occluded, so a
    # clear ray against a proxy must not rule a full occlusion out.
    for authority, expected in (
        ("visual_mesh", {"out_of_view", "fully_occluded"}),
        ("acoustic_proxy_mesh", {"out_of_view"}),
        ("unknown", {"out_of_view"}),
    ):
        series = cv.screen_visibility_series(
            camera=camera(),
            tracks=[track("source1", hold(INSIDE_RIGHT, 1))],
            subject="source1",
            mesh=open_mesh(),
            geometry_authority=authority,
        )
        assert set(series["frames"][0]["refutes"]) == expected
        assert series["geometry_authority"] == authority
    with pytest.raises(cv.ConditionedVisibilityError, match="geometry_authority"):
        cv.screen_visibility_series(
            camera=camera(),
            tracks=[track("source1", hold(INSIDE_RIGHT, 1))],
            subject="source1",
            geometry_authority="ue_visual_guess",
        )


def test_an_out_of_view_screen_call_refutes_nothing() -> None:
    # The nine-sample hull under-covers a silhouette, and the retained captures
    # show the renderer seeing targets on frames this call would have ruled out.
    series = cv.screen_visibility_series(
        camera=camera(),
        tracks=[track("source1", hold(OUTSIDE_RIGHT, 1))],
        subject="source1",
        cast_rays=False,
    )
    assert series["frames"][0]["state"] == "out_of_view"
    assert series["frames"][0]["refutes"] == []
    # The one sound direction: a sample inside the image rules out an empty
    # footprint, and it needs no geometry at all.
    inside = cv.screen_visibility_series(
        camera=camera(),
        tracks=[track("source1", hold(INSIDE_RIGHT, 1))],
        subject="source1",
        cast_rays=False,
    )
    assert inside["frames"][0]["refutes"] == ["out_of_view"]
    assert "rules out out_of_view" in inside["sound_direction"]


def test_a_proxy_mesh_ranks_an_occlusion_requirement_instead_of_refusing_it() -> None:
    subject = track("source1", hold(FAR_SUBJECT, 20))
    terminal = track(
        "source2", stitch(hold(ASIDE_BLOCKER, 10), hold(ON_AXIS_BLOCKER, 10))
    )
    requirement = cv.VisibilityRequirement(
        kind="fully_occluded_then_visible", subject="source1"
    )
    proxied = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[camera()],
        tracks=[subject, terminal],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
        geometry_authority="acoustic_proxy_mesh",
    )
    assert proxied["candidates"], proxied["refuted"]
    screen = proxied["candidates"][0]["requirement_screens"][0]
    assert screen["verdict"] == "undetermined"
    assert "never becomes visible again" in (
        screen["screen_refutation_withheld"]["would_have_refuted"]
    )
    assert "not the geometry that renders" in screen["reason"]
    assert "not a proven impossibility" in proxied["pruning_boundary"]
    # A crossing requirement is a frustum fact, so the proxy does not soften it.
    exit_only = cv.solve_visibility_candidates(
        [cv.VisibilityRequirement(kind="out_of_view_to_visible", subject="source1")],
        camera_poses=[camera()],
        tracks=[
            track("source1", stitch(hold(INSIDE_RIGHT, 30), hold(OUTSIDE_RIGHT, 10))),
        ],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
        geometry_authority="acoustic_proxy_mesh",
    )
    assert exit_only["candidates"] == []


def test_actor_occlusion_is_screened_without_any_static_mesh() -> None:
    # An instance occluder is measured analytically, so a room that supplies no
    # static geometry still screens a full occlusion by another source.
    subject = track("source1", hold((0.0, 0.0, -6.0), 3))
    blocker = track("source2", hold((0.0, 0.0, -3.0), 3))
    series = cv.screen_visibility_series(
        camera=camera(), tracks=[subject, blocker], subject="source1", mesh=None
    )
    frame = series["frames"][0]
    assert frame["state"] == "fully_occluded"
    assert frame["occluder_ids"] == ["source2"]
    assert frame["occluder_attribution"] == {"source2": frame["samples"]}


def test_unresolved_static_geometry_blocks_a_named_occluder() -> None:
    subject = track("source1", hold(INSIDE_RIGHT, 20))
    solved = cv.solve_visibility_candidates(
        [
            cv.VisibilityRequirement(
                kind="registered_occluder_visible", subject="source1"
            )
        ],
        camera_poses=[camera()],
        tracks=[subject, track("source2", hold((-3.0, 0.0, -4.0), 20))],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=wall_mesh(),
    )
    # The wall hides the subject, but nothing declares what the wall is, so no
    # unique registered occluder can be named for the question.
    screens = [
        item
        for candidate in solved["candidates"]
        for item in candidate["requirement_screens"]
    ] or [
        {"verdict": "refuted", "reason": row["reason"]} for row in solved["refuted"]
    ]
    assert all(item["verdict"] != "consistent" for item in screens)
    series = cv.screen_visibility_series(
        camera=camera(), tracks=[subject], subject="source1", mesh=wall_mesh()
    )
    assert series["frames"][0]["occluder_attribution"] == {
        "unresolved_static_geometry": series["frames"][0]["samples"]
    }
    assert series["frames"][0]["occluder_ids"] == []


def test_a_declared_static_box_names_the_occluder() -> None:
    subject = track("source1", hold(INSIDE_RIGHT, 20))
    box = cv.StaticOccluder(
        occluder_id="sideboard_01",
        minimum_m=(-4.0, -0.5, -2.2),
        maximum_m=(4.0, 3.0, -1.8),
        semantic_class="cabinet",
    )
    series = cv.screen_visibility_series(
        camera=camera(),
        tracks=[subject],
        subject="source1",
        mesh=wall_mesh(),
        static_occluders=[box],
    )
    frame = series["frames"][0]
    assert frame["occluder_ids"] == ["sideboard_01"]
    assert "unresolved_static_geometry" not in frame["occluder_attribution"]


def test_static_occluders_from_layout_refuses_to_guess_a_frame() -> None:
    layout = {
        "objects": [
            {
                "object_id": "sofa_01",
                "semantic_class": "sofa",
                "static": True,
                "bounds_xyz_m": [[0.0, 0.0, 0.0], [2.0, 1.0, 0.8]],
            },
            {
                "object_id": "stool_01",
                "static": False,
                "bounds_xyz_m": [[3.0, 0.0, 0.0], [3.4, 0.4, 0.5]],
            },
        ]
    }
    guessed, report = cv.static_occluders_from_layout(layout, authoring_to_shared=None)
    assert guessed == ()
    assert report["state"] == STATE_EVIDENCE_MISSING
    assert "refusing to guess" in report["reason"]

    boxes, report = cv.static_occluders_from_layout(
        layout, authoring_to_shared="authoring_xyz_m_to_xz_negative_y_m"
    )
    assert [item.occluder_id for item in boxes] == ["sofa_01"]
    assert report["state"] == STATE_AVAILABLE
    assert report["skipped"] == [{"object_id": "stool_01", "reason": "declared movable"}]
    # Authoring Z becomes shared Y and authoring Y becomes shared -Z.
    assert boxes[0].minimum_m == pytest.approx((0.0, 0.0, -1.0))
    assert boxes[0].maximum_m == pytest.approx((2.0, 0.8, 0.0))

    empty, report = cv.static_occluders_from_layout(
        {}, authoring_to_shared="identity_shared_meter_y_up"
    )
    assert empty == ()
    assert "declares no object list" in report["reason"]

    with pytest.raises(cv.ConditionedVisibilityError, match="unknown authoring_to_shared"):
        cv.static_occluders_from_layout(layout, authoring_to_shared="ue_cm_guess")


def test_mesh_narrowing_does_not_change_a_ray_verdict() -> None:
    mesh = wall_mesh()
    far = MeshHandle(
        np.concatenate([mesh.vertices, np.asarray([[80.0, 80.0, 80.0], [81.0, 80.0, 80.0], [81.0, 81.0, 80.0]])]),
        np.concatenate([mesh.triangles, np.asarray([[4, 5, 6]], dtype=np.int64)]),
        {"source": "wall_plus_far_triangle"},
    )
    origin = np.asarray([0.0, 1.55, 0.0])
    target = np.asarray([1.0, 1.0, -4.0])
    narrowed = cv.narrow_mesh_to_segments(far, np.asarray([origin, target]))
    assert narrowed.triangles.shape[0] < far.triangles.shape[0]
    assert line_of_sight(narrowed, origin, target) == line_of_sight(far, origin, target)
    series = cv.screen_visibility_series(
        camera=camera(),
        tracks=[track("source1", hold(INSIDE_RIGHT, 2))],
        subject="source1",
        mesh=far,
    )
    assert series["static_geometry_narrowing"]["triangles_retained"] == 2
    assert series["static_geometry_narrowing"]["triangles_declared"] == 3


# ---------------------------------------------------------------------------
# A device subject removes one requirement, never the group
# ---------------------------------------------------------------------------


def test_an_immobile_subject_only_blocks_the_crossing_requirement() -> None:
    device = track(
        "source1",
        hold(INSIDE_RIGHT, 20),
        offset=[0.03, 0.1, 0.0],
        entity_class="rigid_static_object",
    )
    walker = track("source2", stitch(hold((3.0, 0.0, -2.0), 10), hold((0.0, 0.0, -2.0), 10)))
    requirements = [
        cv.VisibilityRequirement(kind="out_of_view_to_visible", subject="source1"),
        cv.VisibilityRequirement(
            kind="fully_occluded_without_return", subject="source1"
        ),
        cv.VisibilityRequirement(
            kind="registered_occluder_visible", subject="source1"
        ),
    ]
    solved = cv.solve_visibility_candidates(
        requirements,
        camera_poses=[camera()],
        tracks=[device, walker],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
    )
    states = {row["kind"]: row for row in solved["requirements"]}
    assert states["out_of_view_to_visible"]["capability_state"] == STATE_NOT_APPLICABLE
    # The pixel field keeps the pixel vocabulary: this kind asks for no
    # explicit state, so it stays None rather than becoming "available".
    assert states["out_of_view_to_visible"]["state"] is None
    assert "fixed camera" in states["out_of_view_to_visible"]["reason"]
    assert states["fully_occluded_without_return"]["capability_state"] == STATE_AVAILABLE
    assert states["registered_occluder_visible"]["capability_state"] == STATE_AVAILABLE
    # A device that cannot walk must not collapse the whole group.
    assert solved["requirement_group_state"] == STATE_AVAILABLE
    assert states["out_of_view_to_visible"]["subject_can_move"] is False
    assert states["registered_occluder_visible"]["mobile_other_instances"] == ["source2"]


def test_a_still_scene_blocks_only_the_requirements_that_need_a_change() -> None:
    device = track(
        "source1",
        hold(INSIDE_RIGHT, 10),
        offset=[0.03, 0.1, 0.0],
        entity_class="rigid_static_object",
    )
    other = track(
        "source2",
        hold((-3.0, 0.0, -2.0), 10),
        offset=[0.03, 0.1, 0.0],
        entity_class="rigid_static_object",
    )
    solved = cv.solve_visibility_candidates(
        [
            cv.VisibilityRequirement(
                kind="fully_occluded_then_visible", subject="source1"
            ),
            cv.VisibilityRequirement(
                kind="fully_occluded_without_return", subject="source1"
            ),
        ],
        camera_poses=[camera()],
        tracks=[device, other],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=wall_mesh(),
    )
    states = {row["kind"]: row for row in solved["requirements"]}
    assert states["fully_occluded_then_visible"]["capability_state"] == STATE_NOT_APPLICABLE
    assert "neither the subject nor any other instance moves" in (
        states["fully_occluded_then_visible"]["capability_reason"]
    )
    assert states["fully_occluded_without_return"]["capability_state"] == STATE_AVAILABLE
    # The permanently hidden device is exactly the negative reappearance case.
    assert solved["candidates"], solved["refuted"]
    screen = solved["candidates"][0]["requirement_screens"][0]
    assert screen["kind"] == "fully_occluded_without_return"
    assert screen["verdict"] == "consistent"


# The camera sits at the origin looking down -Z, so an occluder standing on
# that axis at z = -3 covers a subject at z = -6 and one at x = 2.5 does not.
FAR_SUBJECT = (0.0, 0.0, -6.0)
ON_AXIS_BLOCKER = (0.0, 0.0, -3.0)
ASIDE_BLOCKER = (2.5, 0.0, -3.0)


def test_a_reappearance_requirement_needs_a_return_not_a_terminal_hide() -> None:
    subject = track("source1", hold(FAR_SUBJECT, 20))
    terminal = track(
        "source2", stitch(hold(ASIDE_BLOCKER, 10), hold(ON_AXIS_BLOCKER, 10))
    )
    solved = cv.solve_visibility_candidates(
        [cv.VisibilityRequirement(kind="fully_occluded_then_visible", subject="source1")],
        camera_poses=[camera()],
        tracks=[subject, terminal],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
        geometry_authority="visual_mesh",
    )
    assert solved["candidates"] == []
    (refusal,) = solved["refuted"]
    assert "never becomes visible again" in refusal["reason"]

    returning = track(
        "source2",
        stitch(hold(ASIDE_BLOCKER, 5), hold(ON_AXIS_BLOCKER, 5), hold(ASIDE_BLOCKER, 10)),
    )
    solved = cv.solve_visibility_candidates(
        [
            cv.VisibilityRequirement(
                kind="fully_occluded_then_visible", subject="source1"
            ),
            cv.VisibilityRequirement(
                kind="registered_occluder_visible",
                subject="source1",
                occluder_subject="source2",
            ),
        ],
        camera_poses=[camera()],
        tracks=[subject, returning],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
    )
    assert solved["candidates"]
    screens = {
        item["kind"]: item
        for item in solved["candidates"][0]["requirement_screens"]
    }
    assert screens["fully_occluded_then_visible"]["verdict"] == "consistent"
    hidden = screens["fully_occluded_then_visible"]["fully_occluded_runs"]
    assert [[item["start"], item["end"]] for item in hidden] == [[5, 10]]
    # The same occluder is the one a QA-10 question would have to name.
    assert screens["registered_occluder_visible"]["verdict"] == "consistent"
    assert screens["registered_occluder_visible"]["selected"]["value"] == "source2"


def test_the_solver_refutes_at_the_frustum_tier_before_casting_rays() -> None:
    behind = track("source1", hold((0.0, 0.0, 4.0), 10))
    solved = cv.solve_visibility_candidates(
        [cv.VisibilityRequirement(kind="visibility_state", subject="source1", state="visible_clear")],
        camera_poses=[camera(), camera(candidate_id="grid_00001_yaw_000")],
        tracks=[behind],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=wall_mesh(),
    )
    assert solved["candidates"] == []
    assert solved["stages"]["poses_screened_frustum"] == 2
    assert solved["stages"]["poses_surviving_frustum"] == 0
    assert solved["stages"]["poses_screened_with_rays"] == 0
    assert {row["tier"] for row in solved["refuted"]} == {"frustum"}


def test_the_solver_reports_observation_window_retention() -> None:
    subject = track("source1", entry_clip())
    requirement = cv.VisibilityRequirement(
        kind="out_of_view_to_visible",
        subject="source1",
        require_publishable_window=True,
        observation_windows=((0, 40),),
    )
    solved = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[camera()],
        tracks=[subject],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
    )
    observation = solved["candidates"][0]["requirement_screens"][0]["observation"]
    assert observation["declared"] is True
    assert observation["observation_frames_total"] == 40
    # The published entry interval consumes frames 10..30 of the legal window.
    assert observation["frames_consumed_by_transition"] == [[10, 30]]
    assert observation["observation_frames_retained"] == 20
    assert solved["candidates"][0]["observation_frames_retained"] == 20


def test_a_visibility_state_requirement_honours_its_observation_windows() -> None:
    subject = track("source1", entry_clip())
    inside = cv.VisibilityRequirement(
        kind="visibility_state",
        subject="source1",
        state="out_of_view",
        observation_windows=((0, 5),),
    )
    outside = cv.VisibilityRequirement(
        kind="visibility_state",
        subject="source1",
        state="out_of_view",
        observation_windows=((20, 40),),
    )
    both = cv.solve_visibility_candidates(
        [inside, outside],
        camera_poses=[camera()],
        tracks=[subject],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
    )
    # The subject is out of view only for frames 0..9, so the first window can
    # be answered and the second cannot; the candidate is not consistent.
    verdicts = {
        tuple(item["observation_windows"] for item in ())
        or item["kind"]: item["verdict"]
        for candidate in both["candidates"]
        for item in candidate["requirement_screens"]
    }
    assert both["candidates"]
    assert both["candidates"][0]["verdict"] == "undetermined"
    screens = both["candidates"][0]["requirement_screens"]
    assert [item["verdict"] for item in screens] == ["consistent", "undetermined"]
    assert screens[0]["frames_in_state_inside_observation"] == list(range(5))
    assert screens[1]["frames_in_state_inside_observation"] == []
    assert "inside the observation windows" in screens[1]["reason"]

    solved_one = cv.solve_visibility_candidates(
        [inside],
        camera_poses=[camera()],
        tracks=[subject],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=open_mesh(),
    )
    assert solved_one["candidates"][0]["verdict"] == "consistent"


def test_a_track_clock_mismatch_is_refused() -> None:
    with pytest.raises(cv.ConditionedVisibilityError, match="frame clock"):
        cv.solve_visibility_candidates(
            [cv.VisibilityRequirement(kind="visibility_state", subject="source1", state="visible_clear")],
            camera_poses=[camera()],
            tracks=[
                track("source1", hold(INSIDE_RIGHT, 10)),
                track("source2", hold(INSIDE_LEFT, 9)),
            ],
            frame_rate_hz=FRAME_RATE_HZ,
        )


def test_an_unknown_requirement_subject_is_refused() -> None:
    with pytest.raises(cv.ConditionedVisibilityError, match="no track"):
        cv.solve_visibility_candidates(
            [cv.VisibilityRequirement(kind="visibility_state", subject="ghost", state="visible_clear")],
            camera_poses=[camera()],
            tracks=[track("source1", hold(INSIDE_RIGHT, 4))],
            frame_rate_hz=FRAME_RATE_HZ,
        )


# ---------------------------------------------------------------------------
# Tier three: native pixel acceptance
# ---------------------------------------------------------------------------

RESOLUTION_HW = (8, 16)
SEMANTIC_ID = 7


def masks_for(states, *, instance="source1"):
    """Build a real modal/target-only mask pair for a state sequence."""

    height, width = RESOLUTION_HW
    normal = []
    target_only = []
    for state in states:
        modal = np.zeros((height, width), dtype=np.int32)
        footprint = np.zeros((height, width), dtype=np.int32)
        if state != "out_of_view":
            # A footprint on the right half so the centroid column sits well
            # outside the judge's dead zone.
            footprint[2:6, 12:15] = SEMANTIC_ID
            if state == "visible_clear":
                modal[2:6, 12:15] = SEMANTIC_ID
            elif state == "visible_occluded":
                modal[2:4, 12:15] = SEMANTIC_ID
        normal.append(modal)
        target_only.append(footprint)
    return normal, target_only


def truth_for(states_by_instance, *, camera_pose_id="grid_00000_yaw_000"):
    instances = sorted(states_by_instance)
    frame_count = len(states_by_instance[instances[0]])
    context = {
        "renderer_backend": "habitat",
        "rgb_renderer_backend": "habitat",
        "camera_contract_id": "fixed_static_v1",
        "semantic_id_namespace": "test_ns",
        "resolution_hw": list(RESOLUTION_HW),
        "frame_indices": list(range(frame_count)),
        "camera_pose_ids": [camera_pose_id] * frame_count,
    }
    normal = [np.zeros(RESOLUTION_HW, dtype=np.int32) for _ in range(frame_count)]
    target_only = {}
    contexts = {}
    semantic_ids = {}
    for index, instance in enumerate(instances):
        own_normal, own_target = masks_for(states_by_instance[instance])
        semantic_ids[instance] = SEMANTIC_ID + index
        for frame in range(frame_count):
            own_target[frame][own_target[frame] > 0] = semantic_ids[instance]
            own_normal[frame][own_normal[frame] > 0] = semantic_ids[instance]
            normal[frame] = np.where(own_normal[frame] > 0, own_normal[frame], normal[frame])
        target_only[instance] = own_target
        contexts[instance] = {
            **context,
            "pass_kind": "target_only",
            "target_instance_id": instance,
        }
    return compile_pixel_visibility_truth(
        normal_semantic_masks=normal,
        target_only_semantic_masks_by_instance=target_only,
        semantic_ids_by_instance=semantic_ids,
        normal_context={**context, "pass_kind": "modal_scene"},
        target_only_contexts_by_instance=contexts,
    )


def test_acceptance_passes_a_real_entry_and_fails_an_exit() -> None:
    entering = ["out_of_view"] * 20 + ["visible_clear"] * 30
    truth = truth_for({"source1": entering})
    requirement = cv.VisibilityRequirement(
        kind="out_of_view_to_visible",
        subject="source1",
        side="right",
        require_publishable_window=True,
    )
    accepted = cv.accept_native_visibility(
        [requirement],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        camera_pose_ids=["grid_00000_yaw_000"] * 50,
    )
    assert accepted["binding"]["status"] == "pass"
    assert accepted["status"] == "pass"
    (row,) = accepted["requirements"]
    assert row["status"] == "pass"
    assert row["measured"]["selected"]["entry_frame"] == 20
    assert row["measured"]["selected"]["side"] == "right"
    assert row["measured"]["selected"]["publishable"] is True
    assert row["judge_source"] == "avengine/qa/unified_catalog.py"

    leaving = ["visible_clear"] * 30 + ["out_of_view"] * 20
    accepted = cv.accept_native_visibility(
        [requirement],
        pixel_truth=truth_for({"source1": leaving}),
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
    )
    (row,) = accepted["requirements"]
    assert row["status"] == "fail"
    assert "no consecutive out_of_view to visible pixel pair" in row["reason"]


def test_acceptance_refuses_the_wrong_entry_side() -> None:
    truth = truth_for({"source1": ["out_of_view"] * 20 + ["visible_clear"] * 30})
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="out_of_view_to_visible", subject="source1", side="left"
            )
        ],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
    )
    (row,) = accepted["requirements"]
    assert row["status"] == "fail"
    assert "this branch needs left" in row["reason"]


def test_acceptance_keeps_the_two_reappearance_branches_apart() -> None:
    returning = ["visible_clear"] * 10 + ["fully_occluded"] * 10 + ["visible_clear"] * 10
    terminal = ["visible_clear"] * 10 + ["fully_occluded"] * 20
    yes = cv.VisibilityRequirement(kind="fully_occluded_then_visible", subject="source1")
    no = cv.VisibilityRequirement(
        kind="fully_occluded_without_return",
        subject="source1",
        require_complete_coverage=True,
    )
    accepted = cv.accept_native_visibility(
        [yes, no],
        pixel_truth=truth_for({"source1": returning}),
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
    )
    statuses = {row["kind"]: row for row in accepted["requirements"]}
    assert statuses["fully_occluded_then_visible"]["status"] == "pass"
    assert statuses["fully_occluded_without_return"]["status"] == "fail"

    accepted = cv.accept_native_visibility(
        [yes, no],
        pixel_truth=truth_for({"source1": terminal}),
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
    )
    statuses = {row["kind"]: row for row in accepted["requirements"]}
    assert statuses["fully_occluded_then_visible"]["status"] == "fail"
    assert statuses["fully_occluded_without_return"]["status"] == "pass"
    assert statuses["fully_occluded_without_return"]["visibility_complete"] is True


def test_a_negative_reappearance_needs_complete_observation() -> None:
    truth = truth_for({"source1": ["visible_clear"] * 10 + ["fully_occluded"] * 20})
    # Drop two frames: the answer "it never came back" is not observable when
    # the episode has no state for part of the clip.
    sparse = json.loads(json.dumps(truth))
    sparse["per_instance"]["source1"]["frames"] = [
        frame
        for frame in sparse["per_instance"]["source1"]["frames"]
        if frame["frame_index"] not in {25, 26}
    ]
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="fully_occluded_without_return",
                subject="source1",
                require_complete_coverage=True,
            )
        ],
        pixel_truth=sparse,
        frame_rate_hz=FRAME_RATE_HZ,
        frame_count=30,
    )
    (row,) = accepted["requirements"]
    assert row["status"] == "not_run"
    assert row["visibility_complete"] is False
    assert "explicit pixel state at every frame" in row["reason"]
    assert accepted["status"] == "incomplete"


def test_a_full_occlusion_requirement_fails_without_the_pixel_state() -> None:
    truth = truth_for({"source1": ["visible_occluded"] * 30})
    accepted = cv.accept_native_visibility(
        [cv.VisibilityRequirement(kind="fully_occluded_then_visible", subject="source1")],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
    )
    (row,) = accepted["requirements"]
    assert row["status"] == "fail"
    assert "a blocked ray is a different measurement" in row["reason"]


def test_acceptance_reads_the_occluder_evidence_frame_records() -> None:
    states = ["visible_clear"] * 6 + ["visible_occluded"] * 24
    truth = truth_for({"source1": states, "source2": ["visible_clear"] * 30})
    evidence = {
        "authority": "intersection_of_native_depth_modal_and_target_only_masks",
        "frame_records": [
            {
                "frame_index": frame,
                "target_instance_id": "source1",
                "occluder_instance_ids": ["source2"],
                "explained_fraction": 1.0,
            }
            for frame in range(6, 30)
        ],
    }
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="registered_occluder_visible",
                subject="source1",
                require_publishable_window=True,
            )
        ],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        occluder_evidence=evidence,
        occluder_registry={"source2": {"display_label": "Human (blue top)"}},
    )
    (row,) = accepted["requirements"]
    assert row["status"] == "pass"
    selected = row["measured"]["selected"]
    assert selected["occluder_instance_id"] == "source2"
    assert selected["publishable"] is True
    assert selected["public_seconds"] == [1.0, 2.0]


def test_an_unregistered_pixel_occluder_is_missing_evidence_not_a_failure() -> None:
    states = ["visible_occluded"] * 30
    truth = truth_for({"source1": states})
    evidence = {
        "frame_records": [
            {
                "frame_index": frame,
                "target_instance_id": "source1",
                "occluder_instance_ids": ["native_static_object::counter_07"],
            }
            for frame in range(30)
        ]
    }
    accepted = cv.accept_native_visibility(
        [cv.VisibilityRequirement(kind="registered_occluder_visible", subject="source1")],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        occluder_evidence=evidence,
        occluder_registry={"source2": {"display_label": "other"}},
    )
    (row,) = accepted["requirements"]
    assert row["status"] == "not_run"
    assert row["capability_state"] == STATE_EVIDENCE_MISSING
    assert row["measured"]["unregistered_occluder_ids"] == [
        "native_static_object::counter_07"
    ]


def test_a_missing_appearance_label_is_missing_evidence_not_an_absent_target() -> None:
    truth = truth_for({"source1": ["visible_clear"] * 30})
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="visibility_state", subject="source1", state="visible_clear"
            )
        ],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        appearance_reviewed=[],
    )
    (row,) = accepted["requirements"]
    assert row["status"] == "not_run"
    assert row["capability_state"] == STATE_EVIDENCE_MISSING
    assert row["in_view_frames"] == 30
    assert "missing label is missing evidence, not an" in row["reason"]
    # With the label present the same episode answers the same requirement.
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="visibility_state", subject="source1", state="visible_clear"
            )
        ],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        appearance_reviewed=["source1"],
    )
    assert accepted["requirements"][0]["status"] == "pass"


def test_acceptance_reports_a_binding_that_disagrees_with_the_camera() -> None:
    truth = truth_for({"source1": ["visible_clear"] * 30})
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="visibility_state", subject="source1", state="visible_clear"
            )
        ],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=(8, 32),
    )
    assert accepted["binding"]["status"] == "fail"
    assert "resolution" in accepted["binding"]["reason"]
    assert accepted["status"] == "fail"


def test_acceptance_binds_complete_actor_mapping_but_judges_requested_subjects() -> None:
    entering = ["out_of_view"] * 20 + ["visible_clear"] * 30
    truth = truth_for(
        {"source1": entering, "source2": ["out_of_view"] * 50}
    )
    requirement = cv.VisibilityRequirement(
        kind="out_of_view_to_visible",
        subject="human_target",
        side="right",
        require_publishable_window=True,
    )
    accepted = cv.accept_native_visibility(
        [requirement],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        camera_pose_ids=["grid_00000_yaw_000"] * 50,
        actor_by_instance={
            "human_target": "source1",
            "dog_competitor": "source2",
        },
    )
    assert accepted["binding"]["status"] == "pass"
    assert accepted["status"] == "pass"
    assert len(accepted["requirements"]) == 1
    assert accepted["requirements"][0]["instance_id"] == "source1"


def test_acceptance_rejects_a_missing_actor_from_complete_mapping() -> None:
    entering = ["out_of_view"] * 20 + ["visible_clear"] * 30
    truth = truth_for({"source1": entering})
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="out_of_view_to_visible",
                subject="human_target",
                side="right",
            )
        ],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        camera_pose_ids=["grid_00000_yaw_000"] * 50,
        actor_by_instance={
            "human_target": "source1",
            "dog_competitor": "source2",
        },
    )
    assert accepted["binding"]["status"] == "fail"
    assert "instances differ" in accepted["binding"]["reason"]
    assert accepted["status"] == "fail"


def test_acceptance_complete_mapping_keeps_resolution_and_camera_rejections() -> None:
    truth = truth_for(
        {"source1": ["visible_clear"] * 30, "source2": ["out_of_view"] * 30}
    )
    requirement = cv.VisibilityRequirement(
        kind="visibility_state",
        subject="human_target",
        state="visible_clear",
    )
    mapping = {
        "human_target": "source1",
        "dog_competitor": "source2",
    }
    bad_resolution = cv.accept_native_visibility(
        [requirement],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=(8, 32),
        actor_by_instance=mapping,
    )
    assert bad_resolution["binding"]["status"] == "fail"
    assert bad_resolution["status"] == "fail"
    bad_camera = cv.accept_native_visibility(
        [requirement],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        camera_pose_ids=["different_pose"] * 30,
        actor_by_instance=mapping,
    )
    assert bad_camera["binding"]["status"] == "fail"
    assert bad_camera["status"] == "fail"


def test_acceptance_maps_an_instance_id_onto_the_captured_actor() -> None:
    truth = truth_for({"source1": ["visible_clear"] * 30})
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="visibility_state", subject="human_instance_1", state="visible_clear"
            )
        ],
        pixel_truth=truth,
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        actor_by_instance={"human_instance_1": "source1"},
    )
    (row,) = accepted["requirements"]
    assert row["status"] == "pass"
    assert row["instance_id"] == "source1"


def test_compare_screen_to_witness_separates_the_four_error_causes() -> None:
    subject = track("source1", hold(INSIDE_RIGHT, 4))
    hidden = cv.screen_visibility_series(
        camera=camera(), tracks=[subject], subject="source1", mesh=wall_mesh()
    )
    assert hidden["predicted_state_counts"]["fully_occluded"] == 4
    # Over-predicting an occlusion is allowed: the screen never confirms one.
    over = cv.compare_screen_to_witness(
        hidden, truth_for({"source1": ["visible_clear"] * 4})
    )
    assert over["frames_compared"] == 4
    assert over["agreement"] == 0.0
    assert over["occlusion_overprediction"] == 4
    assert over["occlusion_underprediction"] == 0
    assert over["in_view_direction_violation_count"] == 0
    assert over["unsound_refutation_count"] == 0

    # Under-predicting one is the proxy-geometry case measured on real data.
    clear = cv.screen_visibility_series(
        camera=camera(),
        tracks=[subject],
        subject="source1",
        mesh=open_mesh(),
        geometry_authority="acoustic_proxy_mesh",
    )
    under = cv.compare_screen_to_witness(
        clear, truth_for({"source1": ["fully_occluded"] * 4})
    )
    assert under["occlusion_underprediction"] == 4
    assert under["geometry_authority"] == "acoustic_proxy_mesh"
    # Withholding the occlusion refutation is what keeps this sound.
    assert under["unsound_refutation_count"] == 0

    # An out_of_view prediction the renderer disagrees with is a pruning cost.
    away = cv.screen_visibility_series(
        camera=camera(),
        tracks=[track("source1", hold(OUTSIDE_RIGHT, 4))],
        subject="source1",
        cast_rays=False,
    )
    pruned = cv.compare_screen_to_witness(
        away, truth_for({"source1": ["visible_clear"] * 4})
    )
    assert pruned["out_of_view_prediction_errors"] == 4
    assert pruned["unsound_refutation_count"] == 0

    # The one direction that must never fail: a sample inside the image while
    # the renderer reports an empty footprint.
    violated = cv.compare_screen_to_witness(
        clear, truth_for({"source1": ["out_of_view"] * 4})
    )
    assert violated["in_view_direction_violation_count"] == 4
    assert violated["unsound_refutation_count"] == 4
    assert violated["unsound_refutations"][0]["witnessed"] == "out_of_view"


def test_screen_agreement_travels_with_an_acceptance_record() -> None:
    subject = track("source1", hold(INSIDE_RIGHT, 30))
    series = cv.screen_visibility_series(
        camera=camera(),
        tracks=[subject],
        subject="source1",
        mesh=open_mesh(),
        geometry_authority="visual_mesh",
    )
    accepted = cv.accept_native_visibility(
        [
            cv.VisibilityRequirement(
                kind="visibility_state", subject="source1", state="visible_clear"
            )
        ],
        pixel_truth=truth_for({"source1": ["visible_clear"] * 30}),
        frame_rate_hz=FRAME_RATE_HZ,
        resolution_hw=RESOLUTION_HW,
        screen=series,
    )
    assert accepted["screen_agreement"]["agreement"] == 1.0
    assert accepted["screen_agreement"]["unsound_refutation_count"] == 0


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


def resolved(room_id):
    catalog = load_room_catalog(CATALOG)
    return resolve_catalog_room(catalog, room_id, catalog_path=CATALOG)


@pytest.mark.parametrize(
    "room_id,renderer,witness_mode",
    [
        ("habitat_mp3d_example_17DRP5sb8fy", "habitat", "habitat_modal_target_only_semantic_v1"),
        ("hm3d_val_00800_TEEsavR23oF", "habitat", "habitat_modal_target_only_semantic_v1"),
        ("legacy_ue_apartment_0000_v1", "ue_spear", "spear_modal_target_only_metric_depth_v1"),
        ("kujiale_0020_full_home_v1", "ue_spear", "spear_modal_target_only_metric_depth_v1"),
    ],
)
def test_all_four_production_rooms_dispatch_by_declared_renderer(
    room_id, renderer, witness_mode
) -> None:
    report = cv.provider_visibility_capabilities(resolved(room_id), other_instance_count=1)
    assert report["renderer"] == renderer
    assert report["dispatch"] == "declared_renderer_and_room_package_dimensions"
    facilities = report["facilities"]
    assert facilities["native_pixel_witness"]["witness_mode"] == witness_mode
    assert facilities["native_pixel_witness"]["state"] == STATE_AVAILABLE
    assert facilities["frustum_screen"]["state"] == STATE_AVAILABLE
    assert facilities["ray_occlusion_screen"]["state"] == STATE_AVAILABLE
    assert facilities["actor_occluder_identity_screen"]["state"] == STATE_AVAILABLE
    assert facilities["native_actor_occluder_witness"]["state"] == STATE_AVAILABLE
    # Static occluder identity from pixels needs per-object ids in the modal
    # pass, which only the UE route emits today.
    expected = STATE_AVAILABLE if renderer == "ue_spear" else STATE_NOT_IMPLEMENTED
    assert facilities["native_static_occluder_witness"]["state"] == expected
    assert report["screen_can_confirm_a_pixel_state"] is False
    assert set(report["facilities"]) == set(cv.VISIBILITY_FACILITIES)
    package = resolved(room_id).package
    render_surface = package['static_geometry'].get('render_surface') is True
    assert report['geometry_authority'] == ('visual_mesh' if render_surface else 'acoustic_proxy_mesh')
    assert facilities['ray_occlusion_screen']['may_refute_an_occlusion_state'] is render_surface
    if not render_surface:
        assert 'not the geometry that renders' in facilities['ray_occlusion_screen']['geometry_note']



def test_no_room_identifier_appears_in_the_dispatch() -> None:
    # Adding a room is a registration. Every string constant in this module is
    # compared against the registered room ids and the production family names,
    # so a per-room branch cannot be introduced without this failing.
    import ast

    literals = {
        node.value
        for node in ast.walk(ast.parse(Path(cv.__file__).read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    catalog = load_room_catalog(CATALOG)
    for entry in catalog["rooms"]:
        assert str(entry["room_id"]) not in literals
    for family in ("apartment", "kujiale", "mp3d", "hm3d", "authored"):
        assert family not in literals
    # The only dispatch keys are the declared renderers.
    assert set(cv.NATIVE_WITNESS_ROUTES) == {"habitat", "ue_spear"}


def test_geometry_authority_comes_from_the_declared_static_geometry_source() -> None:
    catalog = load_room_catalog(CATALOG)
    package = resolve_catalog_room(
        catalog, "habitat_mp3d_example_17DRP5sb8fy", catalog_path=CATALOG
    ).package
    assert package["static_geometry"]["source"] == "acoustic_package_arrays"
    assert package['static_geometry']['render_surface'] is True
    assert cv.geometry_authority_for_package(package) == "visual_mesh"
    legacy = {"static_geometry": {"source": "acoustic_package_arrays"}}
    assert cv.geometry_authority_for_package(legacy) == "acoustic_proxy_mesh"
    assert (
        cv.geometry_authority_for_package(
            {"static_geometry": {"representation": "visual_surface_mesh"}}
        )
        == "visual_mesh"
    )
    assert cv.geometry_authority_for_package({}) == "unknown"
    assert cv.geometry_authority_for_package(None) == "unknown"


def test_a_room_without_a_route_has_no_facilities_to_report() -> None:
    with pytest.raises(cv.ConditionedVisibilityError, match="route"):
        cv.provider_visibility_capabilities({"room_id": "x", "capabilities": {}})


def test_an_unknown_renderer_is_not_implemented_rather_than_absent() -> None:
    report = cv.provider_visibility_capabilities(
        {
            "room_id": "future_room_v1",
            "route": {"family": "future", "renderer": "some_new_backend"},
            "capabilities": {
                "dimensions": {
                    "visual_scene": {"status": "pass"},
                    "semantics": {"status": "pass"},
                    "static_geometry": {"status": "pass"},
                }
            },
        },
        other_instance_count=1,
    )
    assert report["facilities"]["native_pixel_witness"]["state"] == STATE_NOT_IMPLEMENTED
    # The geometry screen still works, because it needs no renderer.
    assert report["facilities"]["frustum_screen"]["state"] == STATE_AVAILABLE
    assert report["facilities"]["ray_occlusion_screen"]["state"] == STATE_AVAILABLE


def test_a_blocked_semantics_dimension_is_missing_evidence() -> None:
    report = cv.provider_visibility_capabilities(
        {
            "room_id": "partial_room_v1",
            "route": {"family": "mp3d_like", "renderer": "habitat"},
            "capabilities": {
                "dimensions": {
                    "visual_scene": {"status": "pass"},
                    "semantics": {"status": "blocked"},
                    "static_geometry": {"status": "not_run"},
                }
            },
        }
    )
    facilities = report["facilities"]
    assert facilities["native_pixel_witness"]["state"] == STATE_EVIDENCE_MISSING
    assert "semantics=blocked" in facilities["native_pixel_witness"]["reason"]
    assert facilities["ray_occlusion_screen"]["state"] == STATE_EVIDENCE_MISSING
    assert facilities["actor_occluder_identity_screen"]["state"] == STATE_EVIDENCE_MISSING
    assert report["status"] == "partial"


def test_the_acceptance_plan_names_the_artifacts_and_the_judges() -> None:
    capabilities = cv.provider_visibility_capabilities(
        resolved("kujiale_0020_full_home_v1"), other_instance_count=1
    )
    plan = cv.native_acceptance_plan(
        [
            cv.VisibilityRequirement(
                kind="out_of_view_to_visible", subject="source1", side="right"
            ),
            cv.VisibilityRequirement(
                kind="fully_occluded_without_return",
                subject="source1",
                require_complete_coverage=True,
            ),
            cv.VisibilityRequirement(
                kind="registered_occluder_visible", subject="source1"
            ),
        ],
        capabilities=capabilities,
    )
    assert plan["state"] == STATE_AVAILABLE
    assert plan["witness_mode"] == "spear_modal_target_only_metric_depth_v1"
    rows = {row["kind"]: row for row in plan["requirements"]}
    assert "native_pixel_masks_depth_authority_v1.npz" in rows[
        "out_of_view_to_visible"
    ]["required_artifacts"]
    assert (
        "unified_catalog._entry_transition_window"
        in rows["out_of_view_to_visible"]["required_checks"]
    )
    assert (
        "unified_catalog._visibility_is_complete"
        in rows["fully_occluded_without_return"]["required_checks"]
    )
    occluder = rows["registered_occluder_visible"]
    assert "actor_occluders.json" in occluder["required_artifacts"]
    assert occluder["occluder_witnesses_available"] == ["actor", "static_object"]


def test_the_acceptance_plan_reports_a_blocked_witness_as_a_blocker() -> None:
    plan = cv.native_acceptance_plan(
        [cv.VisibilityRequirement(kind="visibility_state", subject="source1", state="fully_occluded")],
        capabilities={
            "facilities": {
                "native_pixel_witness": {
                    "state": STATE_EVIDENCE_MISSING,
                    "reason": "the room package does not resolve semantics=blocked",
                }
            }
        },
    )
    assert plan["state"] == STATE_EVIDENCE_MISSING
    assert plan["requirements"][0]["capability_state"] == STATE_EVIDENCE_MISSING
    assert "semantics=blocked" in plan["requirements"][0]["blockers"][0]


# ---------------------------------------------------------------------------
# C07: the pixel state and the planner's capability word are two vocabularies.
# ---------------------------------------------------------------------------


def test_a_requirement_row_keeps_its_pixel_state_beside_the_capability_word() -> None:
    """A ``fully_occluded`` requirement must not read back as ``available``.

    The planner verdict used to be written into the same ``state`` key the
    requirement's own pixel state lives in, so a request for a specific pixel
    state was reported back in the capability vocabulary and a reader could not
    tell which state had been solved.
    """

    hidden = track("source1", hold((0.0, 0.0, -4.0), 12))
    solved = cv.solve_visibility_candidates(
        [
            cv.VisibilityRequirement(
                kind="visibility_state",
                subject="source1",
                state="fully_occluded",
                qa_id="QA-24",
            )
        ],
        camera_poses=[camera()],
        tracks=[hidden],
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=wall_mesh(),
    )
    row = solved["requirements"][0]
    assert row["state"] == "fully_occluded"
    assert row["capability_state"] == STATE_AVAILABLE
    assert row["capability_state"] != row["state"]
    # And the planner verdict still drives the group decision it always did.
    assert solved["requirement_group_state"] == STATE_AVAILABLE


def test_the_capability_word_never_overwrites_a_state_in_the_acceptance_plan() -> None:
    plan = cv.native_acceptance_plan(
        [
            cv.VisibilityRequirement(
                kind="visibility_state", subject="source1", state="fully_occluded"
            )
        ],
        capabilities={
            "facilities": {
                "native_pixel_witness": {"state": STATE_AVAILABLE, "reason": None}
            }
        },
    )
    row = plan["requirements"][0]
    assert row["state"] == "fully_occluded"
    assert row["capability_state"] == STATE_AVAILABLE


# ---------------------------------------------------------------------------
# C07: what the routes have to do, said before the routes are drawn.
# ---------------------------------------------------------------------------


def test_a_crossing_asks_the_subject_itself_to_walk() -> None:
    rows = cv.route_dynamics_requirements(
        [
            cv.VisibilityRequirement(
                kind="out_of_view_to_visible",
                subject="human_target",
                side="left",
                qa_id="QA-07",
            )
        ]
    )
    assert [row.dynamic for row in rows] == ["subject_frustum_crossing"]
    assert rows[0].instance_id == "human_target"
    assert rows[0].subject_must_move is True
    assert rows[0].any_mover_satisfies is False
    assert rows[0].qa_ids == ("QA-07",)


def test_a_persistent_state_asks_the_routes_for_nothing() -> None:
    """A permanently hidden bolted-down device is a legal QA-09 ``no``."""

    assert (
        cv.route_dynamics_requirements(
            [
                cv.VisibilityRequirement(
                    kind="fully_occluded_without_return",
                    subject="source1",
                    qa_id="QA-09",
                ),
                cv.VisibilityRequirement(
                    kind="registered_occluder_visible", subject="source1"
                ),
            ]
        )
        == ()
    )


def test_an_occlusion_change_may_be_carried_by_another_mover() -> None:
    requirement = cv.VisibilityRequirement(
        kind="fully_occluded_then_visible", subject="device", qa_id="QA-09"
    )
    rows = cv.route_dynamics_requirements([requirement])
    assert rows[0].dynamic == "subject_or_occluder_motion"
    assert rows[0].subject_must_move is False
    plan = cv.route_motion_plan(
        [requirement], mobile_instance_ids=["walker"]
    )
    assert plan["must_move"] == ("walker",)
    assert plan["unsatisfiable"] == []
    # And the subject is preferred when it can walk itself.
    assert cv.route_motion_plan(
        [requirement], mobile_instance_ids=["device", "walker"]
    )["must_move"] == ("device",)


def test_a_crossing_asked_of_an_immobile_subject_is_named_not_silently_moved() -> None:
    plan = cv.route_motion_plan(
        [
            cv.VisibilityRequirement(
                kind="out_of_view_to_visible", subject="speaker_box", qa_id="QA-07"
            )
        ],
        mobile_instance_ids=["human_target"],
    )
    assert plan["must_move"] == ()
    assert len(plan["unsatisfiable"]) == 1
    assert plan["unsatisfiable"][0]["instance_id"] == "speaker_box"
    assert "no other instance can stand in" in plan["unsatisfiable"][0]["why"]


@pytest.mark.parametrize(
    "kind",
    [
        "out_of_view_to_visible",
        "visibility_state",
        "fully_occluded_then_visible",
        "fully_occluded_without_return",
        "visible_occluded_to_visible_clear",
        "registered_occluder_visible",
    ],
)
def test_the_route_plan_and_the_solver_agree_on_what_a_still_scene_refuses(
    kind: str,
) -> None:
    """The pre-route statement has to match the post-route verdict exactly.

    ``route_motion_plan`` is a second reading of ``_REQUIRED_DYNAMICS``.  If it
    ever disagreed with ``solve_visibility_candidates``, a route sampler would
    either move an instance that did not need to, or draw a still route and be
    refused afterwards for a reason it was told did not apply.
    """

    requirement = cv.VisibilityRequirement(
        kind=kind,
        subject="source1",
        state="fully_occluded" if kind == "visibility_state" else None,
    )
    still = [
        track("source1", hold((0.0, 0.0, -4.0), 12)),
        track("source2", hold((2.0, 0.0, -4.0), 12)),
    ]
    solved = cv.solve_visibility_candidates(
        [requirement],
        camera_poses=[camera()],
        tracks=still,
        frame_rate_hz=FRAME_RATE_HZ,
        mesh=wall_mesh(),
    )
    solver_says_unreachable = (
        solved["requirements"][0]["capability_state"] == STATE_NOT_APPLICABLE
    )
    plan = cv.route_motion_plan([requirement], mobile_instance_ids=[])
    plan_says_unreachable = bool(plan["unsatisfiable"])
    assert solver_says_unreachable == plan_says_unreachable, (
        solved["requirements"][0],
        plan,
    )
    # And when the caller *can* move something, the plan names who.
    mobile_plan = cv.route_motion_plan(
        [requirement], mobile_instance_ids=["source1", "source2"]
    )
    assert bool(mobile_plan["must_move"]) == plan_says_unreachable
    assert mobile_plan["unsatisfiable"] == []


def test_route_motion_plan_refuses_a_non_requirement() -> None:
    with pytest.raises(cv.ConditionedVisibilityError):
        cv.route_dynamics_requirements([{"kind": "out_of_view_to_visible"}])
