"""Construct an intended side of entry; rendered pixel streams remain QA truth.

The family supplies an analytic tangent to the requested side of the frustum.
It does not search arbitrary route headings and does not change a QA threshold.
Geometric predictions are explicitly labelled; they must never be copied into
native visibility facts or used as an export admission receipt.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

REVISION = "qa07-side-first-tangent-v1"
SIDES = {"left": -1, "right": 1}


@dataclass(frozen=True)
class EntryEnvelope:
    """Conservative upright body envelope, including pose and root uncertainty."""

    radius_m: float = .4
    height_m: float = 1.9
    hfov_deg: float = 85.
    aspect: float = 16. / 9.
    edge_margin_deg: float = 2.
    pixel_margin_px: float = 8.
    minimum_camera_distance_m: float = 1.5
    minimum_stable_s: float = .6

    def __post_init__(self):
        values = (self.radius_m, self.height_m, self.hfov_deg, self.aspect,
                  self.edge_margin_deg, self.pixel_margin_px,
                  self.minimum_camera_distance_m, self.minimum_stable_s)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("entry envelope needs positive finite dimensions")
        if self.hfov_deg / 2 + self.edge_margin_deg >= 89.:
            raise ValueError("entry field of view has no usable side plane")


def requested_side(state: Mapping, scene_index: int) -> str:
    """Balance actual exported answers before solving any geometry."""
    tables = state.get("answer_counts", state.get("family_answer_counts", {}))
    counts = tables.get("QA-07", {})
    if not counts:
        counts = state.get("answer_branch_distribution", {}).get("families", {}).get("QA-07", {}).get("answers", {})
    left, right = int(counts.get("left", 0)), int(counts.get("right", 0))
    if left != right:
        return "left" if left < right else "right"
    return "left" if int(scene_index) % 2 == 0 else "right"


def tangent_entry_anchors(camera_m, endpoint_m, side: str, *,
                          final_azimuth_deg: float = 26.,
                          envelope: EntryEnvelope | None = None) -> dict:
    """Invert the side-frustum condition to obtain yaw and a straight entry.

    Camera coordinates are right/up/forward, with right = forward x world-up.
    The target translates towards the centre at constant camera depth. The
    start's entire envelope lies beyond the chosen side plane. At the end the
    entire envelope lies within both horizontal and vertical frustum planes.
    Room support, obstruction and body-to-body spacing are certified by core.
    """
    env = envelope or EntryEnvelope()
    if side not in SIDES:
        raise ValueError("entry side must be left or right")
    camera = np.asarray(camera_m, float)
    endpoint = np.asarray(endpoint_m, float)
    if camera.shape != (3,) or endpoint.shape != (3,) or not np.isfinite([camera, endpoint]).all():
        raise ValueError("entry camera and endpoint must be finite XYZ positions")
    if not 0 < final_azimuth_deg < env.hfov_deg / 2 - env.edge_margin_deg:
        raise ValueError("entry endpoint needs a noncentral inside azimuth")
    delta = endpoint - camera
    radius = float(np.linalg.norm(delta[[0, 2]]))
    if radius < env.minimum_camera_distance_m:
        raise ValueError("entry endpoint violates camera standoff")
    sign = SIDES[side]
    bearing = math.atan2(float(delta[0]), float(-delta[2]))
    yaw = bearing - sign * math.radians(final_azimuth_deg)
    forward = np.array([math.sin(yaw), 0., -math.cos(yaw)])
    right = np.cross(forward, [0., 1., 0.])
    depth = float(delta @ forward)
    lateral = float(delta @ right)
    body_radius = env.radius_m
    h_inside = math.tan(math.radians(env.hfov_deg / 2 - env.edge_margin_deg))
    vfov_half = math.atan(math.tan(math.radians(env.hfov_deg / 2)) / env.aspect)
    v_inside = math.tan(vfov_half - math.radians(env.edge_margin_deg))
    near_depth = depth - body_radius
    vertical = max(abs(float(delta[1])), abs(float(delta[1] + env.height_m)))
    if near_depth <= 0 or abs(lateral) + body_radius > near_depth * h_inside:
        raise ValueError("entry endpoint envelope is not wholly inside horizontally")
    if vertical > near_depth * v_inside:
        raise ValueError("entry endpoint envelope is not wholly inside vertically")
    outside_lateral = (depth + body_radius) * math.tan(math.radians(env.hfov_deg / 2 + env.edge_margin_deg)) + body_radius
    length = outside_lateral - sign * lateral
    start = endpoint + sign * right * length
    inward = -sign * right
    return {
        "revision": REVISION, "branch": side,
        "listener_m": camera.tolist(),
        "camera_yaw_deg": (math.degrees(yaw) + 180.) % 360. - 180.,
        "camera_forward": forward.tolist(), "camera_right": right.tolist(),
        "start_m": start.tolist(), "endpoint_m": endpoint.tolist(),
        "inward_unit_xz": inward[[0, 2]].tolist(), "planning_length_m": length,
        "depth_m": depth, "final_azimuth_deg": sign * final_azimuth_deg,
        "body_radius_m": body_radius, "body_height_m": env.height_m,
        "edge_margin_deg": env.edge_margin_deg,
        "start_outside_plane_margin_m": (depth + body_radius) * (
            math.tan(math.radians(env.hfov_deg / 2 + env.edge_margin_deg)) -
            math.tan(math.radians(env.hfov_deg / 2))),
        "horizontal_endpoint_plane_margin_m": near_depth * math.tan(math.radians(env.hfov_deg / 2)) - abs(lateral) - body_radius,
        "vertical_endpoint_plane_margin_m": near_depth * math.tan(vfov_half) - vertical,
        "capture_camera_moves": False,
        "evidence_kind": "analytic_body_envelope_prediction",
        "native_pixel_truth_verified": False,
    }



def entry_timing(length_m: float, speed_m_s: float, *, stop_s: float = 7.6,
                 gait_extension_reserve_m: float = .65) -> dict:
    """Put the completed entry away from half-overlapping crop boundaries.

    An entry that stops just after t=5 can leave a crop containing only a slow
    deceleration sliver. Solve the start from length instead: the requested
    finish is near t=7.6, leaving room for complete-step fitting and a stable
    first-window entry witness, and real locomotion inside the t=5 crop.
    """
    if not math.isfinite(length_m) or length_m <= 0 or not .8 <= speed_m_s <= 1.4:
        raise ValueError("entry timing needs positive length and legal human speed")
    start = stop_s-length_m/speed_m_s-.25
    latest = stop_s+.02+gait_extension_reserve_m/speed_m_s+1/30.
    if start < .75 or latest > 9.:
        raise ValueError("entry route cannot fit an outside pause and stable first-window finish")
    return {"start_s": start, "requested_interval_s": [start, stop_s+.02],
            "nominal_stop_s": stop_s, "latest_reserved_stop_s": latest,
            "complete_step_time_is_revalidated_natively": True}

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
    if not rows or [r.get("frame_index") for r in rows] != list(range(len(rows))):
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
        facts = {"visibility": {"target": {str(i): dict(r) for i, r in enumerate(rows)}},
                 "time": {"frame_count": len(rows), "frame_rate_hz": fps}}
        for index in range(1, len(rows)):
            previous, current = rows[index-1], rows[index]
            if previous["state"] != "out_of_view" or current["state"] not in VISIBLE_STATES:
                continue
            outside_begin = index-1
            while outside_begin > 0 and rows[outside_begin-1]["state"] == "out_of_view":
                outside_begin -= 1
            stable = _entry_transition_window(facts, "target", index, center=center, dead_zone=dead_zone)
            if stable is None or not publishable_query_window(facts, stable, start_rounding="floor")["publishable"]:
                continue
            offset = float(current["target_centroid_xy_px"][0]) - center
            actual_side = "right" if offset > 0 else "left"
            stable_offsets = [SIDES[side] * (float(rows[j]["target_centroid_xy_px"][0])-center)
                              for j in range(*stable)]
            if (actual_side == side and (index-outside_begin)/fps >= minimum_outside_s
                    and (stable[1]-stable[0])/fps >= minimum_stable_s
                    and min(stable_offsets) >= dead_zone + pixel_margin_px):
                transitions.append({"entry_frame": index, "stable_frames": stable,
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
    length = float(distance[-1])
    expected = float(route["planning_length_m"])
    reserve = float(route.get("native_gait_extension_reserve_m", .65))
    if length < expected - .002 or length > expected + reserve:
        raise ValueError("QA-07 completed gait length exceeds its jointly certified route envelope")
    direction = np.array([route["inward_unit_xz"][0], 0., route["inward_unit_xz"][1]], float)
    if not np.isfinite(direction).all() or abs(float(np.linalg.norm(direction))-1.) > 1e-6:
        raise ValueError("QA-07 route direction must be a unit XZ vector")
    endpoint = np.asarray(route["endpoint_m"], float)
    start = endpoint - direction * length
    positions = start + distance[:, None] * direction
    positions[:, 1] = float(entry["applied_root_input"][1, 3])
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
    if wm is None or not wm.contains_many(positions[:, [0, 2]]).all() or not wm.segment_free(start[[0, 2]], endpoint[[0, 2]]):
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
    local_angle = math.atan2(float(local_forward[2]), float(local_forward[0]))
    heading = math.atan2(float(direction[2]), float(direction[0]))
    yield SimpleNamespace(
        positions=positions, yaws=np.full(len(positions), local_angle-heading),
        receipt={"route_kind": "joint QA-07 analytic side entry", "branch": route["branch"],
                 "root_samples": len(positions), "planning_length_m": expected,
                 "executed_complete_gait_length_m": length,
                 "gait_extension_m": length-expected,
                 "maximum_horizontal_snap_error_m": float(error.max()),
                 "native_geodesic_m": float(native_path.geodesic_distance),
                 "camera_changed": False, "proposal_clock_changed": False,
                 "native_full_pose_checks_still_required": True,
                 "native_entry_preflight_required": True})



def _native_rest_framing(video, entry):
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    from avengine.capture.v7_offscreen_layout import vertices_outside_camera
    mesh = indexed_static_mesh(entry["motion_pose_path"]) if entry.get("motion_episode") else entry["original_mesh"]
    transform = np.asarray(entry["applied_root_input"], float)
    vertices = np.asarray(mesh.vertices, float) @ transform[:3, :3].T + transform[:3, 3]
    sensor = video.sim._sensors["rgb"]._sensor_object
    projection = np.asarray(sensor.render_camera.projection_matrix, float)
    world_to_camera = np.linalg.inv(np.asarray(sensor.node.absolute_transformation(), float))
    if vertices_outside_camera(vertices, world_to_camera, projection):
        return {"state": "outside", "original_mesh_vertices": len(vertices)}
    clip = np.c_[vertices, np.ones(len(vertices))] @ world_to_camera.T @ projection.T
    if np.any(clip[:, 3] <= 0):
        return {"state": "edge", "reason": "actual_body_crosses_near_plane"}
    ndc = clip[:, :2] / clip[:, 3, None]
    margin = float(min((1-np.abs(ndc[:, 0])).min()*video.width/2,
                       (1-np.abs(ndc[:, 1])).min()*video.height/2))
    return {"state": "inside" if margin >= -1e-6 else "edge", "edge_margin_px": margin,
            "original_mesh_vertices": len(vertices)}

def native_entry_preflight(video, scene, entries):
    """Measure real adjacent modal/target-only masks before full media rendering.

    Capture the first 10 seconds of the SAME final native scene and gait. All
    controls and all moving actors are restored to frame zero on every exit.
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
    try:
        for index in range(150):
            for actor in entries:
                if actor.get("motion_episode"):
                    video._apply_motion_entry(actor, index*2)
            counts = video._native_pair_pixels(scene, entries)
            if index in (0, 149):
                rest_frames.append({"frame_index": index, "actors": {e["declaration"]["asset_id"]: _native_rest_framing(video, e) for e in entries}})
            modal = np.asarray(video.sim.get_sensor_observations()["semantic"]).copy()
            measured = stream.capture(index, modal, identities)
            rows.append(measured["per_instance"][tid]["frames"][0])
            if index == 0:
                initial_projection = native_projection_proof(video, scene, target,
                                                              counts[target["declaration"]["asset_id"]])
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
    result.update(status="pass" if not result["reasons"] else "fail",
                  evidence_kind="same_native_final_scene_modal_target_only_preflight",
                  native_pixel_truth_verified=True, target_asset_id=target["declaration"]["asset_id"],
                  target_instance_id=tid, peak_actual_visible_pixels=peak,
                  initial_projection=initial_projection, native_control_restoration=stream.receipt(),
                  measured_frames=rows, native_rest_framing=rest_frames, final_complete_stream_still_required=True)
    return result



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

class QA07Plugin:
    """Family-specific side-plane construction using the shared room/route core."""

    def __init__(self, bodies, sound_requirements=()):
        self.bodies = dict(bodies)
        self.sound_requirements = tuple(sound_requirements)

    def candidates(self, context, request):
        from shapely.geometry import LineString, Point
        from avengine.construction.joint import CameraPose, ScenePlan
        from avengine.construction.joint.geometry import framing
        from avengine.construction.joint.routes import construct_route, static_actor, ConstructionError

        if request.family != "QA-07" or request.branch not in SIDES or len(request.target_ids) != 1:
            raise ValueError("QA07Plugin requires a named target and a fixed left/right branch")
        from avengine.construction.joint.scheduling import schedule_sound_events
        schedule = schedule_sound_events(self.sound_requirements)
        if schedule.status != "solved":
            return
        target = request.target_ids[0]; body = self.bodies[target]
        points = np.asarray(context.points_m, float)
        points = points[context.walkable_map.contains_many(points[:, [0, 2]], "human")]
        floor = float(context.room["measured_floor_y_m"])
        points = points[np.abs(points[:, 1]-floor) <= .03].copy()
        points[:, 1] = floor
        if not len(points):
            return
        # Only family endpoint choices are enumerated. Yaw and start are solved,
        # not an independent grid of arbitrary start/heading pairs.
        radius = math.sqrt(2.)*(body.radius_m+body.margin_m)
        envelope = EntryEnvelope(radius_m=radius, height_m=body.height_m+2*body.margin_m)
        bystanders = [aid for aid in self.bodies if aid != target]
        speed = float(request.options.get("speed_m_s", 1.05))
        reserve = float(request.options.get("gait_extension_reserve_m", .65))
        emitted = 0
        seen_cameras = set()
        for original_camera in context.cameras:
            position = tuple(original_camera.position_m)
            if position in seen_cameras:
                continue
            seen_cameras.add(position)
            distances = np.linalg.norm(points[:, [0, 2]]-np.array(position)[[0, 2]], axis=1)
            endpoint_order = np.argsort(np.abs(distances-4.3), kind="stable")
            for index in endpoint_order[:min(len(points), 600)]:
                if not 3.3 <= distances[index] <= 7.:
                    continue
                endpoint = points[index]
                for theta in (26., 30., 22.):
                    try:
                        anchors = tangent_entry_anchors(position, endpoint-[0., body.margin_m, 0.],
                                                        request.branch, final_azimuth_deg=theta,
                                                        envelope=envelope)
                    except ValueError:
                        continue
                    anchors["endpoint_m"][1] = floor; anchors["start_m"][1] = floor
                    camera = CameraPose(position, anchors["camera_yaw_deg"], hfov_deg=original_camera.hfov_deg,
                                        width=original_camera.width, height=original_camera.height)
                    direction = np.array([anchors["inward_unit_xz"][0], 0., anchors["inward_unit_xz"][1]])
                    reserved = np.asarray(anchors["start_m"])-reserve*direction
                    if not context.walkable_map.segment_free(reserved[[0, 2]], endpoint[[0, 2]], "human"):
                        continue
                    if framing(camera, anchors["start_m"], body)[0] != "outside" or framing(camera, endpoint, body)[0] != "inside":
                        continue
                    line = LineString([reserved[[0, 2]], endpoint[[0, 2]]])
                    options = []
                    for point in points:
                        if (np.linalg.norm(point[[0, 2]]-np.array(position)[[0, 2]]) < 1.6
                                or line.distance(Point(*point[[0, 2]])) < 1.15):
                            continue
                        views = [framing(camera, point, self.bodies[aid])[0] for aid in bystanders]
                        if "edge" not in views:
                            delta = point-np.asarray(position)
                            angle = math.degrees(math.atan2(float(delta@camera.right), float(delta@camera.forward)))
                            # Only the requested moving target needs to enter.
                            # Keep every companion real, wholly in or out, and
                            # reserve its space before native motion compilation.
                            options.append(((0 if views[0] == "outside" else 1,
                                             abs(abs(angle)-105.) if views[0] == "outside" else abs(angle+SIDES[request.branch]*17.)), point))
                    options.sort(key=lambda item: item[0])
                    selected = []
                    for _, point in options:
                        if not all(np.linalg.norm((point-other)[[0, 2]]) >= 1.1 for other in selected):
                            continue
                        if selected:
                            a = np.asarray(anchors["start_m"])[[0, 2]]
                            b = selected[0][[0, 2]]
                            delta = point[[0, 2]]-a
                            baseline = b-a
                            area = abs(float(baseline[0]*delta[1]-baseline[1]*delta[0]))
                            if area/max(1e-9, float(np.linalg.norm(baseline))) < .4:
                                continue
                        selected.append(point)
                        if len(selected) == len(bystanders):
                            break
                    if len(selected) != len(bystanders):
                        continue
                    try:
                        timing = entry_timing(anchors["planning_length_m"], speed, gait_extension_reserve_m=reserve)
                    except ValueError:
                        continue
                    try:
                        mover = construct_route(context, [anchors["start_m"], anchors["endpoint_m"]],
                                                species="human", speed_m_s=speed, start_s=timing["start_s"],
                                                camera=camera, entity_id=target)
                    except ConstructionError:
                        continue
                    mover.body = body
                    actors = {target: mover}
                    for aid, point in zip(bystanders, selected):
                        actors[aid] = static_actor(aid, point, self.bodies[aid])
                    anchors.update(room_id=context.room.get("room_label", context.room.get("room_id")),
                                   walkable_map_path=getattr(context.walkable_map, "path", None),
                                   native_gait_extension_reserve_m=reserve,
                                   speed_m_s=speed, motion_interval_s=timing["requested_interval_s"],
                                   timing=timing, native_preflight_required=True)
                    yield ScenePlan(camera, actors, sounds=schedule.events, metadata={
                        "joint_entry_route": anchors,
                        "sound_schedule_starts_s": list(schedule.starts_s),
                        "sound_schedule_diagnostics": schedule.diagnostics,
                        "rest_views": {aid: framing(camera, actor.positions_m[0], actor.body)[0] for aid, actor in actors.items()},
                        "native_entry_preflight_required": True,
                        "planning_prediction_is_not_pixel_truth": True})
                    emitted += 1
                    if emitted >= int(request.options.get("candidate_limit", 32)):
                        return

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
        seen = {}
        for frame, root in enumerate(actor.positions_m[:900:2]):
            key = tuple(np.round(root, 8))
            if key not in seen:
                seen[key] = context.predict(camera, root, actor.body,
                    occluders=tuple((aid, other.positions_m[min(frame*2, 900)], other.body)
                                    for aid, other in plan.actors.items() if aid != target))
            prediction = seen[key]
            predictions.append(prediction)
            if framing(camera, root, actor.body)[0] == "outside":
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


def construct_qa07_scene(room, house, state, catalog, records, semantic, contract, runtime,
                         *, scene_id, seed, family="QA-07", attempt=0):
    """Translate a certified core proposal to the original phase-two producer."""
    from copy import deepcopy
    from dataclasses import asdict
    from avengine.construction.joint import BodySpec, FamilyRequest, solve
    from avengine.construction.joint.context import build_room_context
    from avengine.timeline.v7_family_constructors import construct_family_scene, room_context, ConstructionUnavailable, _request
    from avengine.timeline.v7_visibility_construction import walking_body, whole_walking_speech
    from avengine.timeline.v7_production_inputs import validate_recipe

    if family != "QA-07":
        raise ValueError("QA-07 constructor cannot serve a different family")
    if (house.get("split") != "train" or str(house["house"]).startswith("hm3d_val_")
            or house["house"] in set(contract.get("test_houses", []))):
        raise ConstructionUnavailable("QA-07 only constructs on the allowed non-val train inventory")
    base = construct_family_scene(room, house, state, catalog, records, semantic, contract, runtime,
                                  scene_id=scene_id, seed=seed, family="QA-03", attempt=attempt)
    row = base["row"]
    target = row["semantic_human_asset_ids"][0]
    row, target = walking_body(row, target, catalog, room, contract, state, seed)
    whole_walking_speech(row, target, catalog, records, semantic, seed)
    from avengine.timeline.v7_family_clock import build_constructed_scene
    _, sound_scene, sound_program, sound_assets = build_constructed_scene(row, house, records, catalog, semantic)
    sound_requirements = qa07_sound_requirements(sound_scene, sound_program, sound_assets)
    side = requested_side(state, seed)
    bodies = {aid: BodySpec("human", radius_m=.35, height_m=1.9, margin_m=.05) for aid in row["asset_ids"]}
    actual_room = deepcopy(room_context(room, house, attempt)["original_room"])
    request = FamilyRequest("QA-07", side, (target,), {"speed_m_s": 1.05, "candidate_limit": 32})
    with build_room_context(actual_room, house=house, runtime=runtime,
                            cache_dir=runtime.get("joint_cache_directory"),
                            camera_limit=72, grid_step_m=.3, camera_heights_m=[1.25],
                            width=1280, height=720) as context:
        solved = solve(context, request, QA07Plugin(bodies, sound_requirements), budget_s=120., max_candidates=32)
    if solved.status != "certified" or solved.plan is None:
        raise ConstructionUnavailable("QA-07 joint construction " + solved.status + ": " + "; ".join(solved.reasons))
    plan = solved.plan
    route = deepcopy(plan.metadata["joint_entry_route"])
    row["listener_m"] = list(plan.camera.position_m)
    row["camera_yaw_deg"] = plan.camera.yaw_deg
    row["camera_yaw_knots"] = [{"time_s": t, "yaw_deg": plan.camera.yaw_deg} for t in (0., 30.)]
    row["native_placement_preferences_m"] = {aid: actor.positions_m[0].tolist() for aid, actor in plan.actors.items()}
    row["native_attachment_supports"] = {
        aid: {"surface": "floor", "original_surface_point_m": actor.positions_m[0].tolist(),
              "normal": [0., 1., 0.], "floor_y": float(actor.positions_m[0, 1]),
              "_outward_clearance_m": 0.} for aid, actor in plan.actors.items()}
    start_delta = np.asarray(route["start_m"])-np.asarray(route["listener_m"])
    angle = abs(math.degrees(math.atan2(float(start_delta@np.asarray(route["camera_right"])),
                                        float(start_delta@np.asarray(route["camera_forward"])))))
    row["native_view_roles"] = {}
    for aid, actor in plan.actors.items():
        if aid == target:
            continue
        view = plan.metadata["rest_views"][aid]
        role = {"view": view, "static": True}
        if view == "outside":
            delta = actor.positions_m[0]-np.asarray(plan.camera.position_m)
            azimuth = math.degrees(math.atan2(float(delta@plan.camera.right), float(delta@plan.camera.forward)))
            role.update(minimum_abs_azimuth_deg=max(0., abs(azimuth)-1.),
                        preferred_azimuth_deg=azimuth, real_registered_drawable_required=True)
        row["native_view_roles"][aid] = role
    row["native_view_roles"][target] = {"view": "outside", "static": False, "required_visible_sometime": True,
        "minimum_abs_azimuth_deg": angle-1., "preferred_azimuth_deg": SIDES[side]*angle,
        "real_registered_drawable_required": True}
    speed = route["speed_m_s"]
    row["motion_requests"] = {target: {"action": "walk", "speed_m_s": speed,
        "intervals_s": [route["motion_interval_s"]], "minimum_listener_horizontal_distance_m": 1.5,
        "joint_entry_route": route,
        "camera_path_condition": {"kind": "enter", "camera_forward": route["camera_forward"],
            "required_entry_side": SIDES[side], "initial_outside_minimum_abs_deg": angle-1.,
            "final_inside_maximum_abs_deg": abs(route["final_azimuth_deg"])+.5}}}
    row["automatic_construction"].update(family="QA-07", constructor="construct_qa07_scene",
        revision=REVISION, requested_entry_side=side, core_certificate=asdict(solved.certificate),
        core_diagnostics=solved.diagnostics, native_entry_preflight_required=True,
        planned_sound_events=[asdict(event) for event in plan.sounds],
        sound_schedule_starts_s=plan.metadata["sound_schedule_starts_s"])
    _request(row, state, "QA-07")
    validate_recipe(row, catalog, contract)
    return {"profile": "phase2", "row": row}
