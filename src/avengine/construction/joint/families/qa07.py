"""Construct an intended side of entry; rendered pixel streams remain QA truth.

The family supplies an analytic tangent to the requested side of the frustum.
It does not search arbitrary route headings and does not change a QA threshold.
Geometric predictions are explicitly labelled; they must never be copied into
native visibility facts or used as an export admission receipt.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from avengine.construction.joint.entry import EntryEnvelope, tangent_entry_anchors, entry_timing

REVISION = "qa07-side-first-tangent-v1"
SIDES = {"left": -1, "right": 1}




def requested_side(state: Mapping, scene_index: int, *, declared_side=None) -> str:
    """Honor a declared branch, or balance actual exports before solving."""
    if declared_side is not None:
        if declared_side not in SIDES:
            raise ValueError("declared QA-07 entry side must be left or right")
        return declared_side
    tables = state.get("answer_counts", state.get("family_answer_counts", {}))
    counts = tables.get("QA-07", {})
    if not counts:
        counts = state.get("answer_branch_distribution", {}).get("families", {}).get("QA-07", {}).get("answers", {})
    left, right = int(counts.get("left", 0)), int(counts.get("right", 0))
    if left != right:
        return "left" if left < right else "right"
    return "left" if int(scene_index) % 2 == 0 else "right"






def entry_prediction_certificate(frames: Sequence[Mapping], resolution_hw, side: str,
                                 *, fps: float = 15., minimum_stable_s: float = .6,
                                 minimum_outside_s: float = .5,
                                 pixel_margin_px: float = 8.) -> dict:
    """Use the original stable-entry predicate on a predicted pixel stream.

    Input rows use frame_index/state/target_centroid_xy_px. Predictions must
    be kept separate from runtime facts; this result never admits an export.
    Missing frames and unknown states fail explicitly, rather than becoming
    out_of_view. A continuous visible interval may begin with a few edge pixels;
    it must then hold the requested side for a useful display interval.
    """
    from avengine.qa.unified_catalog import _entry_transition_window, VISIBLE_STATES, publishable_query_window

    if side not in SIDES or fps <= 0 or not math.isfinite(fps):
        raise ValueError("invalid entry branch or sample rate")
    if len(resolution_hw) != 2 or min(resolution_hw) <= 0:
        raise ValueError("entry prediction needs a positive H,W resolution")
    rows = list(frames)
    width = float(resolution_hw[1]); center = (width - 1.) / 2.
    dead_zone = max(1., width * .02)
    reasons = []
    first_frame = rows[0].get("frame_index") if rows else None
    if (not isinstance(first_frame, int) or first_frame < 0
            or [r.get("frame_index") for r in rows] != list(range(first_frame, first_frame+len(rows)))):
        reasons.append("prediction_frames_missing_or_not_contiguous")
    if any(r.get("state") not in {*VISIBLE_STATES, "out_of_view", "fully_occluded"} for r in rows):
        reasons.append("prediction_contains_unknown_state")
    if any(r.get("state") in VISIBLE_STATES and (
            not isinstance(r.get("target_centroid_xy_px"), (list, tuple))
            or len(r["target_centroid_xy_px"]) != 2
            or not np.isfinite(r["target_centroid_xy_px"]).all()) for r in rows):
        reasons.append("prediction_centroid_missing")
    transitions = []
    if not reasons:
        by_frame = {int(r["frame_index"]): dict(r) for r in rows}
        facts = {"visibility": {"target": {str(i): r for i, r in by_frame.items()}},
                 "time": {"frame_count": first_frame+len(rows), "frame_rate_hz": fps}}
        for index in range(1, len(rows)):
            previous, current = rows[index-1], rows[index]
            if previous["state"] != "out_of_view" or current["state"] not in VISIBLE_STATES:
                continue
            outside_begin = index-1
            while outside_begin > 0 and rows[outside_begin-1]["state"] == "out_of_view":
                outside_begin -= 1
            entry_frame = int(current["frame_index"])
            stable = _entry_transition_window(facts, "target", entry_frame, center=center, dead_zone=dead_zone)
            if stable is None or not publishable_query_window(facts, stable, start_rounding="floor")["publishable"]:
                continue
            offset = float(current["target_centroid_xy_px"][0]) - center
            actual_side = "right" if offset > 0 else "left"
            stable_offsets = [SIDES[side] * (float(by_frame[j]["target_centroid_xy_px"][0])-center)
                              for j in range(*stable)]
            if (actual_side == side and (index-outside_begin)/fps >= minimum_outside_s
                    and (stable[1]-stable[0])/fps >= minimum_stable_s
                    and min(stable_offsets) >= dead_zone + pixel_margin_px):
                transitions.append({"entry_frame": entry_frame, "stable_frames": stable,
                                    "side": side, "outside_s": (index-outside_begin)/fps,
                                    "stable_s": (stable[1]-stable[0])/fps,
                                    "minimum_side_offset_px": min(stable_offsets)})
        if not transitions:
            reasons.append("no_predicted_stable_entry_on_requested_side_with_margin")
    return {"status": "pass" if not reasons else "fail", "family": "QA-07", "branch": side,
            "reasons": reasons, "transitions": transitions,
            "resolution_hw": list(resolution_hw), "fps": float(fps),
            "original_entry_predicate": "qa.unified_catalog._entry_transition_window",
            "evidence_kind": "geometry_prediction_only", "native_pixel_truth_verified": False}


def native_entry_route_proposals(video, entry, scene, distance, local_forward, request):
    """Consume the completed gait clock without changing it or choosing a new side."""
    from types import SimpleNamespace
    from shapely.geometry import LineString, shape
    from avengine.rooms.walkable_map import load_room_map
    from avengine.timeline.v7_spatial_layout import spacing_rejection
    from avengine.capture.v7_camera_path_proposals import matches_camera_path

    route = request["joint_entry_route"]
    distance = np.asarray(distance, float)
    if distance.shape != (901,) or not np.isfinite(distance).all() or np.any(np.diff(distance) < -1e-9):
        raise ValueError("QA-07 needs the final monotone 901-frame gait distance clock")
    if abs(float(distance[0])) > 1e-8:
        raise ValueError("QA-07 gait distance must start at zero")
    required = ("planned_distance_m", "planned_positions_m", "planned_native_yaws_rad")
    if any(key not in route for key in required):
        raise ValueError("QA-07 native execution lacks solved clock/positions/yaws; return to joint solve")
    planned_distance = np.asarray(route["planned_distance_m"], float)
    positions = np.asarray(route["planned_positions_m"], float).copy()
    yaws = np.asarray(route["planned_native_yaws_rad"], float).copy()
    if (planned_distance.shape != (901,) or positions.shape != (901, 3) or yaws.shape != (901,)
            or not all(np.isfinite(value).all() for value in (planned_distance, positions, yaws))):
        raise ValueError("QA-07 solved native arrays must contain the complete finite 30Hz clock")
    if not np.allclose(distance, planned_distance, rtol=0., atol=1e-7):
        raise ValueError("QA-07 native completed gait clock differs from joint solve; execution cannot refit")
    if np.linalg.norm(np.asarray(entry["applied_root_input"])[:3, 3]-positions[0]) > .002:
        raise ValueError("QA-07 initial native body root differs from the solved root")
    length = float(distance[-1])
    expected = float(route["planning_length_m"])
    start, endpoint = positions[0], positions[-1]
    listener = np.asarray(scene["listener_m"], float)
    if not np.allclose(listener, route["listener_m"], atol=1e-6, rtol=0):
        raise ValueError("QA-07 native camera differs from the solved fixed camera")
    if not np.allclose(scene["camera_forward"], route["camera_forward"], atol=1e-6, rtol=0):
        raise ValueError("QA-07 native camera orientation differs from the solved yaw")
    poly = shape(scene["production_scenario"]["motion_floor_polygon"])
    # Original scanned floor polygons can contain nanometre assembly seams.
    # This is a numerical closure only; the v2 full segment, native 2mm snap,
    # standoff and subsequent actual full-pose checks are still mandatory.
    if not poly.buffer(1e-6).covers(LineString(positions[:, [0, 2]])):
        raise ValueError("QA-07 final gait route leaves its original floor polygon")
    wm = load_room_map(route["room_id"], path=route.get("walkable_map_path"))
    moving_segments = np.flatnonzero(np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1) > 1e-10)
    if (wm is None or not wm.contains_many(positions[:, [0, 2]]).all()
            or any(not wm.segment_free(positions[i, [0, 2]], positions[i+1, [0, 2]]) for i in moving_segments)):
        raise ValueError("QA-07 final gait route fails the complete v2 walkable segment")
    others = [a for a in video.actors if a is not entry and a.get("phase2_layout_qualified") is not False]
    reason = spacing_rejection(entry["declaration"], positions, others,
                               camera_position=listener, camera_minimum_m=1.5)
    if reason:
        raise ValueError("QA-07 final gait violates owner spacing: " + str(reason))
    if not matches_camera_path(positions, np.arange(901)/30., listener, request):
        raise ValueError("QA-07 final gait misses the requested entry-side bounds")
    snaps = np.array([video.pathfinder.snap_point(p) for p in positions])
    error = np.linalg.norm((snaps-positions)[:, [0, 2]], axis=1)
    if not np.isfinite(snaps).all() or float(error.max()) > .002:
        raise ValueError("QA-07 final gait horizontal native snap exceeds 2mm")
    native_path = video.hs.ShortestPath()
    native_path.requested_start = snaps[0]; native_path.requested_end = snaps[-1]
    if not video.pathfinder.find_path(native_path):
        raise ValueError("QA-07 final gait endpoints are not natively connected")
    if float(native_path.geodesic_distance) > length + .05:
        raise ValueError("QA-07 straight route crosses a native navigation obstruction")
    yield SimpleNamespace(
        positions=positions, yaws=yaws,
        receipt={"route_kind": "joint QA-07 analytic side entry", "branch": route["branch"],
                 "root_samples": len(positions), "planning_length_m": expected,
                 "executed_complete_gait_length_m": length,
                 "gait_extension_m": length-expected,
                 "maximum_horizontal_snap_error_m": float(error.max()),
                 "native_geodesic_m": float(native_path.geodesic_distance),
                 "camera_changed": False, "proposal_clock_changed": False,
                 "native_full_pose_checks_still_required": True,
                 "native_entry_preflight_required": True,
                 "same_solved_arrays": True, "execution_replanning_allowed": False})



def _native_rest_framing(video, entry):
    from avengine.construction.joint.native import native_rest_framing
    return native_rest_framing(video, entry)



def entry_preflight_frame_indices(positions_m, *, root_fps=30., video_fps=15.):
    """Exact moving interval plus observed outside/stable margins for QA-07.

    QA-07 asks for an entry existence witness. Frame zero is independently
    checked; the contiguous temporal proof covers every frame of locomotion,
    at least .6s before it and 1.2s afterwards inside the first 10s window.
    """
    positions = np.asarray(positions_m, float)
    if positions.shape != (901, 3) or not np.isfinite(positions).all():
        raise ValueError("native entry probe needs all original 901 root samples")
    moving = np.flatnonzero(np.linalg.norm(np.diff(positions, axis=0), axis=1) > 1e-8)
    if not len(moving):
        raise ValueError("native entry probe has no actual motion interval")
    start = max(0, math.floor(moving[0]/root_fps*video_fps)-math.ceil(.6*video_fps))
    end = min(150, math.ceil((moving[-1]+1)/root_fps*video_fps)+math.ceil(1.2*video_fps)+1)
    return list(range(start, end))


def original_qa07_witness(rows, reviews, assets, target_instance_id, resolution_hw, branch, scene_id):
    """Run the original generator on measured probe rows and native ROI review."""
    from avengine.qa.unified_catalog import _generate_qa_07, _Deferred
    tid = str(target_instance_id)
    facts = {"episode_id": scene_id, "time": {"frame_rate_hz": 15.,
             "frame_count": max(int(row["frame_index"]) for row in rows)+1},
             "visibility_meta": {"resolution_hw": list(resolution_hw)},
             "visibility": {tid: {str(row["frame_index"]): row for row in rows}},
             "actors": {}, "appearance_review": reviews}
    for asset in assets:
        aid = str(asset["native_instance_id"])
        record = {"asset_id": asset["asset_id"]}
        if asset.get("label_color") is not None:
            record["appearance"] = {"value": asset["label_color"]}
        facts["actors"][aid] = record
    try:
        question = _generate_qa_07(facts, "qa07-native-preflight")
    except _Deferred as error:
        return {"status": "fail", "reason": str(error), "original_generator": "_generate_qa_07"}
    actual = question["truth"]["value"]
    return {"status": "pass" if actual == branch else "fail",
            "observed_answer": actual, "requested_answer": branch, "question": question,
            "original_generator": "_generate_qa_07", "final_complete_stream_still_required": True}

def native_entry_preflight(video, scene, entries):
    """Measure real adjacent modal/target-only masks before full media rendering.

    Capture the SAME final native scene over the whole actual entry motion
    and a publishable stable interval, plus an independent frame-zero proof.
    All moving actors and controls are restored to frame zero on every exit.
    A failed witness is reported to the caller; no truth fields are edited.
    """
    from avengine.capture.continuous_native_visibility import NativeVisibilityStream
    from avengine.capture.v7_offscreen_layout import native_projection_proof

    requests = scene.get("production_scenario", {}).get("motion_requests", {})
    targets = [(entry, requests[entry["endpoint_id"]]["joint_entry_route"])
               for entry in entries if requests.get(entry["endpoint_id"], {}).get("joint_entry_route")]
    if not targets:
        return {"status": "not_applicable", "family": "QA-07"}
    if len(targets) != 1:
        raise ValueError("QA-07 entry preflight currently supports one moving target")
    target, route = targets[0]
    if not target.get("motion_episode"):
        raise ValueError("QA-07 native preflight requires the completed actual gait")
    if {id(e) for e in entries} != {id(e) for e in video.actors}:
        raise ValueError("QA-07 native preflight must see every final scene actor")
    tid = str(target["declaration"]["native_instance_id"])
    identities = [{"endpoint_id": e["endpoint_id"], "actor_id": e["declaration"]["native_instance_id"]}
                  for e in entries]
    stream = NativeVisibilityStream(video)
    rows = []
    initial_projection = None
    rest_frames = []
    temporal_indices = entry_preflight_frame_indices(target["motion_episode"]["world_roots_m"])
    temporal_set = set(temporal_indices)
    probe_indices = sorted(temporal_set | {0})
    initial_mask = None
    appearance_reviews = {}
    appearance_observations = []
    try:
        for index in probe_indices:
            for actor in entries:
                if actor.get("motion_episode"):
                    video._apply_motion_entry(actor, index*2)
            counts = video._native_pair_pixels(scene, entries)
            if index in (0, temporal_indices[-1]):
                rest_frames.append({"frame_index": index, "actors": {e["declaration"]["asset_id"]: _native_rest_framing(video, e) for e in entries}})
            modal = np.asarray(video.sim.get_sensor_observations()["semantic"]).copy()
            measured = stream.capture(index, modal, identities)
            measured_row = measured["per_instance"][tid]["frames"][0]
            if index in temporal_set:
                rows.append(measured_row)
            if index == 0:
                initial_mask = measured_row
                initial_projection = native_projection_proof(video, scene, target,
                                                              counts[target["declaration"]["asset_id"]])
            # Match the producer's one-second colour sampling during entry,
            # not only a fully visible endpoint. A clipped shoe/skin sample
            # can invalidate the original aggregate appearance review.
            if index % 15 == 0 or index == temporal_indices[-1]:
                from avengine.assets.v7_visible_color import rebuild_appearance_review
                lengths = {name: len(getattr(video, name, [])) for name in
                           ("color_checks", "material_checks", "phase2_chair_pixel_checks")}
                try:
                    observation = video.sim.get_sensor_observations()
                    video._check_source_frame(observation["rgb"][:, :, :3], observation["semantic"], index)
                    appearance_observations.extend(video.color_checks[lengths["color_checks"]:])
                    review, _ = rebuild_appearance_review(appearance_observations,
                                                            [e["declaration"] for e in entries])
                    appearance_reviews = review["actors"]
                finally:
                    for name, length in lengths.items():
                        if hasattr(video, name):
                            del getattr(video, name)[length:]
    finally:
        for actor in entries:
            if actor.get("motion_episode"):
                video._apply_motion_entry(actor, 0)
        video._native_pair_pixels(scene, entries)
        stream.close()
    result = entry_prediction_certificate(rows, [video.height, video.width], route["branch"])
    peak = max((int(r["visible_pixels"]) for r in rows), default=0)
    if initial_projection is None or initial_projection.get("status") != "pass":
        result["reasons"].append("initial_actual_mesh_not_wholly_outside")
    if peak < 100:
        result["reasons"].append("actual_modal_target_never_reaches_100_pixels")
    if any(r["state"] == "edge" for sample in rest_frames for r in sample["actors"].values()):
        result["reasons"].append("actual_rest_body_straddles_image_edge")
    if rest_frames[-1]["actors"][target["declaration"]["asset_id"]]["state"] != "inside":
        result["reasons"].append("actual_entry_endpoint_not_wholly_inside")
    witness_rows = rows if 0 in temporal_set else [initial_mask, *rows]
    witness = original_qa07_witness(witness_rows, appearance_reviews, [e["declaration"] for e in entries],
                                   tid, [video.height, video.width], route["branch"], scene["scene_id"])
    if witness["status"] != "pass":
        result["reasons"].append("original_qa07_not_publishable_on_actual_native_probe")
    result.update(status="pass" if not result["reasons"] else "fail",
                  evidence_kind="same_native_final_scene_modal_target_only_preflight",
                  native_pixel_truth_verified=True, target_asset_id=target["declaration"]["asset_id"],
                  target_instance_id=tid, peak_actual_visible_pixels=peak,
                  initial_projection=initial_projection, native_control_restoration=stream.receipt(),
                  measured_frames=rows, initial_native_mask=initial_mask,
                  native_probe_frame_indices=probe_indices, native_rest_framing=rest_frames,
                  native_temporal_coverage="all actual motion frames plus .6s before and 1.2s after",
                  native_appearance_review=appearance_reviews, native_appearance_observations=appearance_observations,
                  native_colour_sampling="original producer stride 15 during actual motion, plus initial and final held poses",
                  original_qa07_witness=witness, final_complete_stream_still_required=True)
    return result



def native_entry_counterfactual_checks(video, scene):
    """Require the original real wrong-emitter witness; core owns all RIR gates."""
    from avengine.construction.joint import CheckResult
    from avengine.timeline.v7_phase2_motion_audio import wrong_registered_pose
    actual_mouths = {entry["endpoint_id"]:
        np.asarray(video._apply_entry(entry, scene, frame_index=0)["emitter_position_m"], float)
        for entry in video.actors}
    # A separate read-only predicate input, populated only by actual native
    # registered emitters. It is never assigned back to the rendered scene.
    probe = dict(scene, sources={endpoint: {
        **scene["sources"][endpoint],
        "trajectory": [{"time_s": t, "position_m": point.tolist()} for t in (0., 30.)]}
        for endpoint, point in actual_mouths.items() if endpoint in scene["sources"]})
    active_emitters = {event["endpoint_id"] for event in scene.get("events", ())}
    for entry in video.actors:
        endpoint = entry["endpoint_id"]
        if not entry.get("motion_episode") or endpoint not in active_emitters:
            continue
        try:
            other, position, angle = wrong_registered_pose(probe, actual_mouths[endpoint], 0.)
            evidence = {"actual_other_registered_endpoint": other,
                        "actual_other_native_emitter_m": position.tolist(),
                        "angle_deg": angle, "original_predicate": "wrong_registered_pose",
                        "readback_frame": 0, "rendered_scene_modified": False}
            yield CheckResult("QA-07:initial_real_wrong_emitter:"+endpoint, "pass",
                              evidence=evidence)
        except ValueError as exc:
            yield CheckResult("QA-07:initial_real_wrong_emitter:"+endpoint, "fail", str(exc))


def qa07_sound_requirements(scene, program, assets):
    """Declare the selected complete utterances and their measured activity."""
    from avengine.construction.joint.scheduling import SoundRequirement
    owners = {a["endpoint_id"]: a["asset_id"] for a in assets}
    recordings = {r["sound_asset_id"]: r for r in program["records"]}
    bindings = {b["event_id"]: b["sound_asset_id"] for b in program["events"]}
    requirements = []
    for event in scene["events"]:
        record = recordings[bindings[event["event_id"]]]
        intervals = (record.get("screens", {}).get("automatic_activity", {}).get("activity_intervals_samples")
                     or record.get("source_normalization", {}).get("measured_activity", {}).get("source_activity_intervals_samples"))
        if not intervals:
            raise ValueError("QA-07 companion or walker speech lacks original measured activity")
        rate = float(record["header"]["sample_rate_hz"])
        start = event["start_tick"]/10.
        requirements.append(SoundRequirement(owners[event["endpoint_id"]], record["sound_asset_id"],
            record["sound_class"], tuple((lo/rate, hi/rate) for lo, hi in intervals),
            (start, start), wet_tail_s=.5))
    return tuple(requirements)

def entry_edge_prediction_certificate(predictions, resolution_hw, branch, *, first_inside_frame):
    """Reject an obstructed frustum crossing before spending a native render.

    The body proxy is deliberately a proposal filter, never rendered truth.
    Its edge interval includes all partially framed poses and a stable pause;
    this catches furniture that hides the actual first in-frame limb.
    """
    rows = []
    for index, prediction in enumerate(predictions):
        state = prediction.state
        evidence = prediction.evidence
        # The core box can intersect the frame before its projected cylinder.
        # Empty *proxy* raster evidence is outside only for this prediction.
        if state == "unknown" and evidence.get("target_proxy_pixels") == 0:
            state = "out_of_view"
        rows.append({"frame_index": index, "state": state,
                     "target_centroid_xy_px": evidence.get("target_centroid_xy_px")})
    result = entry_prediction_certificate(rows, resolution_hw, branch)
    transition = result["transitions"][0] if result["transitions"] else None
    edge = []
    if transition is not None:
        end = min(len(predictions), max(first_inside_frame+10, transition["entry_frame"]+18))
        edge = predictions[transition["entry_frame"]:end]
    fractions = [p.occluded_fraction for p in edge]
    clear = bool(fractions) and all(v is not None and math.isfinite(v) and v <= .02 for v in fractions)
    if not clear:
        result["reasons"].append("predicted_entry_edge_has_occlusion_or_unknown_depth")
        result["status"] = "fail"
    result.update(evidence_kind="proxy_frustum_crossing_proposal_filter",
                  native_pixel_truth_verified=False,
                  first_fully_inside_proxy_frame=int(first_inside_frame),
                  edge_probe_frames=len(edge),
                  maximum_edge_proxy_occlusion=max((v for v in fractions if v is not None), default=None),
                  native_mask_transition_still_required=True)
    return result


def entry_stage_seed_preview(context, camera, actor):
    """Screen the same edge obligation before a seed consumes the camera beam.

    This is only a necessary subset of the later complete family prediction.
    Every exported candidate still gets all-frame prediction and native masks.
    """
    from avengine.construction.joint.geometry import framing
    roots = actor.positions_m[:300:2]
    frames = [framing(camera, root, actor.body)[0] for root in roots]
    inside = next((i for i, state in enumerate(frames) if state == "inside"), None)
    if inside is None:
        return False
    first = next((i for i, state in enumerate(frames) if state != "outside"), None)
    if first is None:
        return False
    predictions = {}
    def predict(index):
        if index not in predictions:
            predictions[index] = context.predict(camera, roots[index], actor.body)
        return predictions[index]
    while first <= inside and not predict(first).evidence.get("target_proxy_pixels"):
        first += 1
    if first > inside:
        return False
    samples = {first, min(inside, first+5), inside, min(len(roots)-1, inside+10)}
    for index in samples:
        value = predict(index)
        if (value.state not in {"visible_clear", "visible_occluded"}
                or value.occluded_fraction is None or value.occluded_fraction > .02):
            return False
    return True


class QA07Plugin:
    """Family-specific side-plane construction using the shared room/route core."""

    def __init__(self, bodies, sound_requirements=(), *, clock_source=None, assets=None, seed=0, clock_evaluator=None, silent_target=False):
        self.bodies = dict(bodies)
        self.sound_requirements = tuple(sound_requirements)
        self.clock_source = clock_source
        self.assets = {} if assets is None else dict(assets)
        self.seed = int(seed)
        self.clock_evaluator = clock_evaluator
        self._static_meshes = {}
        self.silent_target = bool(silent_target)
        self.seed_diagnostics = {"stage_previews": 0, "stage_preview_rejected": 0}

    def candidates(self, context, request):
        from copy import copy
        from avengine.construction.joint import CameraPose, ScenePlan
        from avengine.construction.joint.geometry import framing
        from avengine.construction.joint.routes import bind_distance_clock, ConstructionError
        from avengine.construction.joint.layouts import enumerate_static_layouts
        from avengine.construction.joint.context import camera_legal
        from avengine.construction.joint.grounding import floor_contact_variants
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        from avengine.construction.joint.scheduling import schedule_sound_events
        if self.clock_evaluator is None:
            from avengine.construction.joint.gait import complete_human_gait_clock
        else:
            complete_human_gait_clock = self.clock_evaluator
        from avengine.construction.joint.native import native_yaw
        from avengine.timeline.v7_natural_orientation import sample_initial_orientation

        if request.family != "QA-07" or request.branch not in SIDES or len(request.target_ids) != 1:
            raise ValueError("QA07Plugin requires a named target and a fixed left/right branch")
        if self.clock_source is None:
            raise ValueError("QA07Plugin requires the admitted complete-step source inside solve")
        schedule = schedule_sound_events(self.sound_requirements)
        if schedule.status != "solved":
            return
        target = request.target_ids[0]; body = self.bodies[target]
        points = np.asarray(context.points_m, float)
        points = points[context.walkable_map.contains_many(points[:, [0, 2]], "human")]
        floor = float(context.room["measured_floor_y_m"])
        points = points[np.abs(points[:, 1]-floor) <= .03]
        if not len(points):
            return
        envelope = EntryEnvelope(radius_m=math.sqrt(2.)*(body.radius_m+body.margin_m),
                                 height_m=body.height_m+2*body.margin_m)
        companions = {aid: spec for aid, spec in self.bodies.items() if aid != target}
        speed = float(request.options.get("speed_m_s", 1.05))
        emitted = 0; seen_cameras = set()
        for inherited in context.cameras:
            position = tuple(inherited.position_m)
            if position in seen_cameras:
                continue
            seen_cameras.add(position)
            per_camera = 0
            camera_previews = 0
            distances = np.linalg.norm(points[:, [0, 2]]-np.asarray(position)[[0, 2]], axis=1)
            for index in np.argsort(np.abs(distances-3.8), kind="stable")[:600]:
                if per_camera >= 1 or camera_previews >= 16:
                    break
                if not 3.1 <= distances[index] <= 7.:
                    continue
                endpoint = points[index]
                for theta in (26., 30., 22.):
                    if per_camera >= 1 or camera_previews >= 16:
                        break
                    try:
                        anchors = tangent_entry_anchors(position, endpoint-[0., body.margin_m, 0.],
                            request.branch, final_azimuth_deg=theta, envelope=envelope)
                        timing = entry_timing(anchors["planning_length_m"], speed)
                    except ValueError:
                        continue
                    anchors["endpoint_m"][1] = float(endpoint[1])
                    camera = CameraPose(position, anchors["camera_yaw_deg"],
                        hfov_deg=inherited.hfov_deg, width=inherited.width, height=inherited.height)
                    if not camera_legal(context.room, camera)[0]:
                        continue
                    clock = complete_human_gait_clock(self.clock_source, speed, [timing["requested_interval_s"]])
                    distance = np.asarray(clock["distance_m"], float)
                    # Family seed: the final whole-body inside anchor fixes the
                    # origin of the completed source clock BEFORE certification.
                    direction = np.array([anchors["inward_unit_xz"][0], 0., anchors["inward_unit_xz"][1]])
                    origin = endpoint-distance[-1]*direction
                    # Necessary straight-segment condition from the same map;
                    # do it before 901-point original NavMesh certification.
                    if not context.walkable_map.segment_free(origin[[0, 2]], endpoint[[0, 2]], "human"):
                        continue
                    # Query the core's original visual mesh at this exact seed,
                    # rather than let native execution change the root height.
                    if context.raycaster is None:
                        continue
                    ranges, _ = context.raycaster.first_hit(origin+[0., .2, 0.], np.array([[0., -1., 0.]]))
                    ground = float(origin[1]+.2-ranges[0])
                    if not math.isfinite(ground) or abs(ground-floor) > .03:
                        continue
                    origin[1] = ground
                    heading = math.atan2(float(direction[2]), float(direction[0]))
                    try:
                        mover = bind_distance_clock(context, distance, origin_m=origin, heading_rad=heading,
                            species="human", camera=camera, entity_id=target, body=body, ramp_s=.25)
                    except ConstructionError:
                        continue
                    if framing(camera, mover.positions_m[0], body)[0] != "outside" or framing(camera, mover.positions_m[-1], body)[0] != "inside":
                        continue
                    if self.seed_diagnostics["stage_previews"] >= int(request.options.get("seed_preview_limit", 512)):
                        return
                    self.seed_diagnostics["stage_previews"] += 1
                    camera_previews += 1
                    if not entry_stage_seed_preview(context, camera, mover):
                        self.seed_diagnostics["stage_preview_rejected"] += 1
                        continue
                    mover.metadata["admitted_gait_clock_certificate"] = clock["certificate"]
                    mover.metadata["admitted_gait_source"] = clock["source_lineage"]
                    anchors.update(start_m=mover.positions_m[0].tolist(), endpoint_m=mover.positions_m[-1].tolist(),
                        room_id=context.room.get("room_label", context.room.get("room_id")),
                        walkable_map_path=getattr(context.walkable_map, "path", None),
                        speed_m_s=speed, motion_interval_s=timing["requested_interval_s"],
                        timing=timing, physical_start_floor_y_m=ground, native_preflight_required=True)
                    # All companion station selection and floor clearance is
                    # delegated to the core, with this same fixed camera.
                    view = copy(context); view.cameras = (camera,)
                    # Declare the opposite-side domain BEFORE the shared
                    # layout beam ranks nearby points; post-ranking rejection
                    # can otherwise consume all 24 slots on the wrong side.
                    relative = context.points_m-np.asarray(camera.position_m)
                    azimuths = np.degrees(np.arctan2(relative@camera.right, relative@camera.forward))
                    opposite = SIDES[request.branch]*azimuths <= -10.
                    point_masks = {aid: opposite for aid in companions}
                    companion_selected = False
                    companion_views = ("inside", "outside") if self.silent_target else ("outside", "inside")
                    for companion_view in companion_views:
                        for layout in enumerate_static_layouts(view, companions,
                                {aid: companion_view for aid in companions},
                                max_candidates=24, floor_clearance_m=.002, point_masks=point_masks):
                            companions_ready = {}
                            for aid, actor in layout.actors.items():
                                delta = actor.positions_m[0]-np.asarray(camera.position_m)
                                azimuth = math.degrees(math.atan2(float(delta@camera.right), float(delta@camera.forward)))
                                # A real other emitter on the opposite side
                                # supplies the producer's 20deg negative control.
                                if SIDES[request.branch]*azimuth > -10.:
                                    break
                                policy = sample_initial_orientation(self.assets[aid], self.seed,
                                    actor.positions_m[0], camera.position_m,
                                    other_people=({"position_m": mover.positions_m[0].tolist(),
                                                   "yaw_rad": native_yaw(mover, self.assets[target])},))
                                actor.metadata["native_yaw_rad"] = policy["yaw_rad"]
                                actor.metadata["initial_orientation_policy"] = policy
                                if aid not in self._static_meshes:
                                    self._static_meshes[aid] = indexed_static_mesh(self.assets[aid]["habitat_glb"])
                                try:
                                    refined = next(floor_contact_variants(context, actor, self._static_meshes[aid],
                                                   asset=self.assets[aid], yaw_rad=policy["yaw_rad"]), None)
                                except ConstructionError:
                                    refined = None
                                if refined is None:
                                    break
                                refined.metadata["entry_counterfactual_opposite_side"] = {
                                    "azimuth_deg": azimuth, "entry_side": request.branch,
                                    "original_wrong_registered_pose_still_required": True}
                                companions_ready[aid] = refined
                            if len(companions_ready) != len(companions):
                                continue
                            actors = {target: mover, **companions_ready}
                            roles = {target: "outside", **layout.metadata["roles"]}
                            companion_selected = True
                            yield ScenePlan(camera, actors, sounds=schedule.events, metadata={
                                "joint_entry_route": anchors, "roles": roles,
                                "rest_views": roles, "native_entry_preflight_required": True,
                                "sound_schedule_starts_s": list(schedule.starts_s),
                                "sound_schedule_diagnostics": schedule.diagnostics,
                                "planning_prediction_is_not_pixel_truth": True})
                            emitted += 1; per_camera += 1
                            if emitted >= int(request.options.get("candidate_limit", 128)):
                                return
                            # Keep one sound companion per route geometry;
                            # spend the per-camera beam on different entry lines.
                            break
                        if companion_selected:
                            break

    def verify(self, context, request, plan):
        from avengine.construction.joint import CheckResult
        from avengine.construction.joint.geometry import framing, body_box, project_points

        target = request.target_ids[0]
        actor = plan.actors[target]
        camera = plan.camera
        states = [framing(camera, actor.positions_m[i], actor.body)[0] for i in (0, -1)]
        yield CheckResult("entry_whole_body_endpoints", "pass" if states == ["outside", "inside"] else "fail",
                          evidence={"states": states, "native_preflight_required": True})
        # The entire possible target projection stays on the intended side.
        # This certifies the side even when the first visible sliver is uncertain.
        offsets = []
        predictions = []
        first_inside_frame = 449
        seen = {}
        for frame, root in enumerate(actor.positions_m[:900:2]):
            key = tuple(np.round(root, 8))
            if key not in seen:
                seen[key] = context.predict(camera, root, actor.body,
                    occluders=tuple((aid, other.positions_m[min(frame*2, 900)], other.body)
                                    for aid, other in plan.actors.items() if aid != target))
            prediction = seen[key]
            predictions.append(prediction)
            framing_state = framing(camera, root, actor.body)[0]
            if framing_state == "inside":
                first_inside_frame = min(first_inside_frame, frame)
            if framing_state == "outside":
                continue
            uv, depth = project_points(camera, body_box(root, actor.body))
            if np.any(depth <= 0):
                offsets.append(-math.inf)
                continue
            xmin, xmax = max(0., float(uv[:, 0].min())), min(float(camera.width), float(uv[:, 0].max()))
            offsets.append(xmin-(camera.width-1)/2 if request.branch == "right" else (camera.width-1)/2-xmax)
        margin = min(offsets, default=-math.inf)-max(1., camera.width*.02)
        yield CheckResult("entry_requested_side_projection", "pass" if margin >= 8. else "fail",
                          margin=margin, evidence={"branch": request.branch, "all_projected_body_points_on_side": margin >= 8.})
        edge = entry_edge_prediction_certificate(predictions, [camera.height, camera.width],
                                                 request.branch, first_inside_frame=first_inside_frame)
        yield CheckResult("entry_unobstructed_frustum_crossing", edge["status"],
                          "predicted edge must lead directly from outside into visible pixels",
                          evidence=edge)
        outside = sum(p.state == "out_of_view" for p in predictions[:45])
        end = predictions[-1]
        # Core deliberately reports zero for a real-silhouette lower bound
        # from a proxy. The inner cylinder count is separately named evidence.
        # It is a construction preference, followed by actual native masks.
        proxy_pixels = int(end.evidence.get("inner_proxy_visible_pixels", 0))
        ready = outside >= 8 and end.state in {"visible_clear", "visible_occluded"} and proxy_pixels >= 200
        yield CheckResult("entry_geometry_with_pixel_margin", "pass" if ready else "fail",
                          evidence={"initial_outside_samples": outside,
                                    "final_state": end.state,
                                    "final_inner_proxy_pixels": proxy_pixels,
                                    "unknown_prediction_samples": sum(p.state == "unknown" for p in predictions),
                                    "native_preflight_required": True,
                                    "lower_pixels_are_inner_proxy_only": True})



def declare_silent_entry_target(row, target):
    """Declare the visual-only moving role before any scene is prepared."""
    if row.get("joint_prepared_scene_required") or row.get("joint_solution"):
        raise ValueError("silent role must be declared before native preparation")
    if target not in row["asset_ids"]:
        raise ValueError("entry target must remain a real registered body")
    authored = list(row.get("semantic_human_asset_ids", ()))
    if target in authored:
        index = authored.index(target)
        for key in ("semantic_human_asset_ids", "semantic_sound_ids", "semantic_start_offsets_s"):
            row[key].pop(index)
    if not row.get("semantic_human_asset_ids"):
        raise ValueError("QA-07 needs its real static authored-speech companion")
    for key in ("source_classes", "proposed_sound_asset_ids", "phase2_authored_speech_schedule_s",
                "phase2_recorded_speech_schedule_s", "phase2_recorded_sound_ids",
                "phase2_source_event_schedule_s"):
        row.get(key, {}).pop(target, None)
    row["silent_asset_ids"] = list(dict.fromkeys([*row.get("silent_asset_ids", ()), target]))
    row["qa07_sound_policy"] = {
        "target_asset_id": target, "target_silent": True,
        "static_authored_speech_asset_ids": list(row["semantic_human_asset_ids"]),
        "declared_before_joint_solve": True, "real_target_body_retained": True,
        "reason": "QA-07 original answer reads actual target pixels, not target sound"}


def _entry_row_for_plan(base_row, plan, target, state, catalog, contract, runtime=None):
    """Translate the already solved arrays; no route or timing search here."""
    from copy import deepcopy
    from dataclasses import asdict
    from avengine.construction.joint.native import native_yaw
    from avengine.assets.roster_motion.navmesh_execution import prepare_scene_motion
    from avengine.timeline.v7_family_constructors import _request
    from avengine.timeline.v7_production_inputs import validate_recipe

    row = deepcopy(base_row)
    actor = plan.actors[target]
    if not actor.metadata.get("distance_clock_bound"):
        raise ValueError("QA-07 row needs a core-bound original complete-step clock")
    route = deepcopy(plan.metadata["joint_entry_route"])
    assets = {a["asset_id"]: a for a in catalog}
    first_yaw = native_yaw(actor, assets[target])
    yaws = first_yaw-(np.asarray(actor.headings_rad)-actor.headings_rad[0])
    route.update(planned_distance_m=list(actor.metadata["distance_clock_m"]),
                 planned_positions_m=actor.positions_m.tolist(), planned_native_yaws_rad=yaws.tolist(),
                 strict_planned_motion=True)
    row["listener_m"] = list(plan.camera.position_m)
    row["camera_yaw_deg"] = plan.camera.yaw_deg
    row["camera_yaw_knots"] = [{"time_s": t, "yaw_deg": plan.camera.yaw_deg} for t in (0., 30.)]
    row["native_view_roles"] = {}
    for aid, current in plan.actors.items():
        view = plan.metadata["roles"][aid]
        role = {"view": view, "static": aid != target}
        if view == "outside":
            delta = current.positions_m[0]-np.asarray(plan.camera.position_m)
            azimuth = math.degrees(math.atan2(float(delta@plan.camera.right), float(delta@plan.camera.forward)))
            role.update(minimum_abs_azimuth_deg=max(0., abs(azimuth)-1.),
                        preferred_azimuth_deg=azimuth, real_registered_drawable_required=True)
        if aid == target:
            role["required_visible_sometime"] = True
        row["native_view_roles"][aid] = role
    delta = actor.positions_m[0]-np.asarray(plan.camera.position_m)
    angle = abs(math.degrees(math.atan2(float(delta@plan.camera.right), float(delta@plan.camera.forward))))
    row["motion_requests"] = {target: {"action": "walk", "speed_m_s": route["speed_m_s"],
        "intervals_s": [route["motion_interval_s"]], "minimum_listener_horizontal_distance_m": 1.5,
        "joint_entry_route": route,
        "camera_path_condition": {"kind": "enter", "camera_forward": route["camera_forward"],
            "required_entry_side": SIDES[route["branch"]], "initial_outside_minimum_abs_deg": angle-1.,
            "final_inside_maximum_abs_deg": abs(route["final_azimuth_deg"])+.5}}}
    row["automatic_construction"].update(family="QA-07", constructor="construct_qa07_scene",
        revision=REVISION, requested_entry_side=route["branch"], native_entry_preflight_required=True,
        planned_sound_events=[asdict(event) for event in plan.sounds],
        sound_schedule_starts_s=plan.metadata["sound_schedule_starts_s"],
        actual_source_clock_solved_before_native=True)
    # Entry is certified in the first complete 10-second window. The original
    # producer supports a subset of crops while rendering/checking all 30s.
    if runtime is not None and runtime.get("qa07_entry_window_only", True):
        row["acceptance_window_starts_s"] = [0.]
        row["window_count"] = 1
        row["automatic_construction"]["delivery_scope"] = {
            "window_start_s": 0., "window_duration_s": 10.,
            "complete_native_world_duration_s": 30.,
            "full_901_pose_physics_required": True,
            "same_entry_in_overlapping_crops_is_not_another_question": True}
    row["joint_prepared_scene_required"] = True
    row = prepare_scene_motion(row, catalog)
    _request(row, state, "QA-07")
    validate_recipe(row, catalog, contract)
    if runtime is not None:
        from avengine.timeline.v7_production_inputs import attach_runtime_contract
        row = attach_runtime_contract(row, runtime)
    return row


def construct_qa07_scene(room, house, state, catalog, records, semantic, contract, runtime,
                         *, scene_id, seed, family="QA-07", attempt=0):
    """One joint solve prepares, certifies, and returns the exact live native world."""
    from copy import deepcopy
    from dataclasses import asdict
    from pathlib import Path
    from avengine.construction.joint import BodySpec, FamilyRequest, CheckResult, solve
    from avengine.construction.joint.context import build_room_context
    from avengine.construction.joint.native import prepare_native
    from avengine.construction.joint.gait import admitted_human_clock_source
    from avengine.timeline.v7_family_constructors import construct_family_scene, room_context, ConstructionUnavailable
    from avengine.timeline.v7_visibility_construction import walking_body, whole_walking_speech
    from avengine.timeline.v7_family_clock import build_constructed_scene

    if family != "QA-07":
        raise ValueError("QA-07 constructor cannot serve a different family")
    if (house.get("split") != "train" or str(house["house"]).startswith("hm3d_val_")
            or house["house"] in set(contract.get("test_houses", []))):
        raise ConstructionUnavailable("QA-07 only constructs on the allowed non-val train inventory")
    if Path(scene_id).name != scene_id or not runtime.get("joint_cache_directory"):
        raise ValueError("QA-07 needs a plain scene ID and an explicitly owned native preparation directory")
    base = construct_family_scene(room, house, state, catalog, records, semantic, contract, runtime,
                                  scene_id=scene_id, seed=seed, family="QA-03", attempt=attempt)
    row = base["row"]
    row["asset_ids"] = list(row["semantic_human_asset_ids"])
    row["silent_asset_ids"] = []
    row["automatic_construction"]["required_role_count"] = len(row["asset_ids"])
    target = row["semantic_human_asset_ids"][0]
    row, target = walking_body(row, target, catalog, room, contract, state, seed)
    moving_target_speech = bool(runtime.get("qa07_moving_target_speech", False))
    if moving_target_speech:
        whole_walking_speech(row, target, catalog, records, semantic, seed)
    else:
        declare_silent_entry_target(row, target)
    _, sound_scene, sound_program, sound_assets = build_constructed_scene(row, house, records, catalog, semantic)
    sound_requirements = qa07_sound_requirements(sound_scene, sound_program, sound_assets)
    side = requested_side(state, seed, declared_side=runtime.get("qa07_entry_side"))
    assets = {a["asset_id"]: a for a in catalog if a["asset_id"] in row["asset_ids"]}
    clock_source = admitted_human_clock_source(assets[target], runtime, house)
    bodies = {aid: BodySpec("human", radius_m=.35, height_m=1.9, margin_m=.05) for aid in row["asset_ids"]}
    actual_room = deepcopy(room_context(room, house, attempt)["original_room"])
    request = FamilyRequest("QA-07", side, (target,), {
        "speed_m_s": 1.05, "candidate_limit": 128, "require_native_preflight": True})
    plugin = QA07Plugin(bodies, sound_requirements, clock_source=clock_source, assets=assets,
                        seed=seed, silent_target=not moving_target_speech)
    directory = Path(runtime["joint_cache_directory"])/"qa07"/"native_preflight"/scene_id

    def make_row(plan):
        return _entry_row_for_plan(row, plan, target, state, catalog, contract, runtime)

    def family_native(video, scene, entries):
        report = getattr(video, "family_native_preflights", {}).get("QA-07")
        if report is None:
            yield CheckResult("QA-07:native_entry_witness", "unknown", "final native family probe was not recorded")
            return
        witness = report.get("original_qa07_witness", {})
        ok = report.get("status") == "pass" and witness.get("status") == "pass" and witness.get("observed_answer") == side
        yield CheckResult("QA-07:original_native_question", "pass" if ok else "fail",
            "original mask predicate, reviewed appearance and publishable absolute interval",
            evidence={"branch": side, "actual_answer": witness.get("observed_answer"),
                      "target_asset_id": report.get("target_asset_id"),
                      "transitions": report.get("transitions"),
                      "peak_native_pixels": report.get("peak_actual_visible_pixels")})
        if ok:
            # The owner's no-furniture-intersection rule needs every actual
            # pose. Stride-10 record-only observations are not that proof.
            from avengine.construction.joint.physics import native_full_mesh_sweep
            yield from native_full_mesh_sweep(video, entries)
            yield from native_entry_counterfactual_checks(video, scene)

    def oracle(context, requested, plan):
        initial_visible = {aid: 100 for aid, role in plan.metadata["roles"].items() if role == "inside"}
        return prepare_native(context, requested, plan, row_factory=make_row, house=house,
            catalog=catalog, records=records, semantic=semantic, runtime=runtime,
            output_dir=directory, family_check=family_native, roles=plan.metadata["roles"],
            minimum_visible_pixels=initial_visible)

    with build_room_context(actual_room, house=house, runtime=runtime,
                            cache_dir=runtime["joint_cache_directory"],
                            camera_limit=216, grid_step_m=.3, camera_heights_m=[1.1, 1.25, 1.45],
                            width=1280, height=720) as context:
        solved = solve(context, request, plugin, prepare_and_verify=oracle,
                       budget_s=float(runtime.get("qa07_solve_budget_s", 900.)),
                       max_candidates=128, max_native_preparations=3)
    solved.diagnostics["family_seed_search"] = dict(plugin.seed_diagnostics)
    from avengine.timeline.continuous_pilot import write_json
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"joint_solve_result.json", {
        "status": solved.status, "reasons": list(solved.reasons),
        "diagnostics": solved.diagnostics, "formal_export_completed": False})
    if solved.status != "certified" or solved.plan is None or solved.prepared is None:
        raise ConstructionUnavailable("QA-07 strict joint construction " + solved.status + ": " + "; ".join(solved.reasons))
    prepared = solved.prepared.payload
    try:
        prepared.row["automatic_construction"].update(core_certificate=asdict(solved.certificate),
                                                       core_diagnostics=solved.diagnostics)
        return {"profile": "phase2", "row": prepared.row, "prepared_native": prepared,
                "joint_plan": solved.plan, "joint_certificate": solved.certificate}
    except Exception:
        prepared.close()
        raise
