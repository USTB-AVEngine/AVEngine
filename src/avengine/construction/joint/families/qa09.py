"""QA-09 requests and preflight evidence; native QA remains the truth authority.

The family asks for an existential reappearance over a complete ten-second
window. A single visible frame after *any* full occlusion makes the original
answer yes. Construction therefore checks the whole window, especially for no,
not only two chosen probe times. Shadow and route geometry belongs to joint_core.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

from avengine.qa.observation_predicates import full_occlusion_frames, reappeared_frames


@dataclass(frozen=True)
class ReappearanceDemand:
    target_asset_id: str
    occluder_asset_id: str
    answer: str
    window_start_s: float = 0.0
    window_duration_s: float = 10.0
    frame_rate_hz: int = 15
    minimum_hidden_s: float = 1.0
    minimum_visible_s: float = 1.0
    minimum_visible_pixels: int = 100
    minimum_target_pixels: int = 100

    def __post_init__(self) -> None:
        if self.answer not in {"yes", "no"}:
            raise ValueError("QA-09 answer must be yes or no")
        if not self.target_asset_id or not self.occluder_asset_id or self.target_asset_id == self.occluder_asset_id:
            raise ValueError("QA-09 needs distinct registered target and occluder")
        if not isinstance(self.frame_rate_hz, int) or self.frame_rate_hz <= 0:
            raise ValueError("positive integer mask rate required")
        for name in ("window_duration_s", "minimum_hidden_s", "minimum_visible_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("positive finite QA-09 durations required")
        if not math.isfinite(self.window_start_s) or self.window_start_s < 0:
            raise ValueError("nonnegative finite window start required")
        if self.window_start_s + self.window_duration_s > 30.0:
            raise ValueError("QA-09 window must lie in the thirty-second parent")
        for seconds in (self.window_start_s, self.window_duration_s):
            if abs(seconds * self.frame_rate_hz - round(seconds * self.frame_rate_hz)) > 1e-8:
                raise ValueError("window must align with actual mask frames")
        if min(self.minimum_visible_pixels, self.minimum_target_pixels) < 1:
            raise ValueError("positive construction pixel margins required")

    @property
    def frame_count(self) -> int:
        return round(self.window_duration_s * self.frame_rate_hz)

    def requested_conditions(self) -> list[str]:
        """Use only registered pipeline condition keys; no is a branch value."""
        conditions = ["registered_full_occluder", "stable_mask_intervals"]
        if self.answer == "yes":
            conditions.append("later_visible_reappearance")
        return conditions

    def relation(self) -> dict[str, Any]:
        """Serialize the request for the native adapter; this is never truth."""
        return {
            "target_asset_id": self.target_asset_id,
            "occluder_asset_id": self.occluder_asset_id,
            "required_state": "fully_occluded",
            "minimum_occluded_fraction": 1.0,
            "minimum_stable_seconds": self.minimum_hidden_s,
            # Existing placement scoring needs a target root during the pause.
            # The complete-window probe below supplies the real certificate.
            "query_times_s": [self.window_start_s + 4.5 + i / self.frame_rate_hz
                              for i in range(math.ceil(self.minimum_hidden_s * self.frame_rate_hz))],
            "foreground_depth_fraction": .55,
            "preferred_lateral_m": 0.0,
            "native_mask_and_unique_registered_identity_required": True,
            "qa09_demand": asdict(self),
        }


def planning_admissible(certificate, *, native_sweep_pending=False):
    """A coarse-map miss may enter native verification, never certify itself."""
    from avengine.construction.joint.solver import _awaits_only_native_sweep
    return certificate.passed or (native_sweep_pending and _awaits_only_native_sweep(certificate.checks))


def requested_answer(state: Mapping[str, Any], seed: int = 0) -> str:
    """Balance actual deduplicated outputs, not admitted plans or retries."""
    tables = state.get("answer_counts") or state.get("family_answer_counts") or {}
    counts = tables.get("QA-09")
    if counts is None:
        counts = state.get("answer_branch_distribution", {}).get("families", {}).get("QA-09", {}).get("answers", {})
    yes, no = (int(counts.get(answer, 0)) for answer in ("yes", "no"))
    if yes < 0 or no < 0:
        raise ValueError("negative delivered QA-09 count")
    if yes == no:
        return "yes" if seed % 2 == 0 else "no"
    return "yes" if yes < no else "no"


def _runs(indices: Sequence[int], fps: int, minimum_s: float) -> list[list[int]]:
    result: list[list[int]] = []
    current: list[int] = []
    required = math.ceil(minimum_s * fps - 1e-9)
    for index in indices:
        if current and index != current[-1] + 1:
            if len(current) >= required:
                result.append([current[0], current[-1]])
            current = []
        current.append(index)
    if len(current) >= required:
        result.append([current[0], current[-1]])
    return result


def certify_mask_sequence(
    frames: Sequence[Mapping[str, Any]],
    occluder_frames: Sequence[Mapping[str, Any]],
    demand: ReappearanceDemand,
    *,
    target_instance_id: str,
    occluder_instance_id: str,
    registered_instance_ids: Sequence[str],
) -> dict[str, Any]:
    """Check actual target-only/modal summaries before expensive media/audio.

    Indices are relative to the requested window. Original QA predicates supply
    the measured branch; one-second intervals, 100 pixels and unique registered
    identity are additional construction margins, never edits to QA thresholds.
    """
    errors: list[str] = []
    ids = {str(x) for x in registered_instance_ids}
    tid, oid = str(target_instance_id), str(occluder_instance_id)
    if tid == oid or not {tid, oid} <= ids:
        errors.append("target_or_occluder_not_uniquely_registered")
    actual_indices = [r.get("frame_index") for r in frames]
    complete = actual_indices == list(range(demand.frame_count))
    if not complete:
        errors.append("incomplete_or_unordered_window")
    known_states = {"out_of_view", "visible_clear", "visible_occluded", "fully_occluded"}
    if any(r.get("state") not in known_states for r in frames):
        errors.append("unknown_visibility_state")
    # No sparse sample can establish absence. Return before interpreting a
    # malformed sequence, including duplicate frame indices.
    if errors:
        return {"status": "fail", "errors": errors, "observed_answer": None,
                "complete_window": complete, "demand": asdict(demand)}
    fully = [int(i) for i in full_occlusion_frames(frames)]
    after = [int(i) for i in reappeared_frames(frames, fully)]
    observed = ("yes" if after else "no") if fully else None
    if not fully:
        errors.append("no_fully_occluded_frame")
    if observed != demand.answer:
        errors.append("observed_branch_differs_from_requested")
    occ: dict[int, Mapping[str, Any]] = {}
    for r in occluder_frames:
        if str(r.get("target_instance_id")) != tid:
            continue
        index = r.get("frame_index")
        if index in occ:
            errors.append("duplicate_occluder_frame")
        occ[index] = r
    hidden: list[int] = []
    clear: list[int] = []
    visible: list[int] = []
    for r in frames:
        i = int(r["frame_index"])
        target_pixels, visible_pixels = r.get("target_pixels", -1), r.get("visible_pixels", -1)
        if target_pixels < 0 or visible_pixels < 0 or visible_pixels > target_pixels:
            errors.append("invalid_native_pixel_counts")
            continue
        if r["state"] in {"visible_clear", "visible_occluded"} and visible_pixels >= demand.minimum_visible_pixels:
            visible.append(i)
            if r["state"] == "visible_clear":
                clear.append(i)
        if r["state"] != "fully_occluded":
            continue
        witness = occ.get(i, {})
        if (target_pixels >= demand.minimum_target_pixels and visible_pixels == 0
                and witness.get("occluder_instance_ids") == [oid]
                and witness.get("unknown_occluder_pixels") == 0
                and witness.get("target_only_pixels") == target_pixels
                and witness.get("context_pixels") == 0
                and witness.get("occluded_pixels") == target_pixels):
            hidden.append(i)
    hidden_runs = _runs(hidden, demand.frame_rate_hz, demand.minimum_hidden_s)
    clear_runs = _runs(clear, demand.frame_rate_hz, demand.minimum_visible_s)
    visible_runs = _runs(visible, demand.frame_rate_hz, demand.minimum_visible_s)
    first_hidden = min(fully) if fully else demand.frame_count
    seen_before = [run for run in visible_runs if run[1] < first_hidden]
    seen_after = [run for run in visible_runs if any(run[0] > h[1] for h in hidden_runs)]
    if not hidden_runs:
        errors.append("no_stable_registered_full_occlusion")
    if not seen_before:
        errors.append("no_stable_visibility_before_first_occlusion")
    if demand.answer == "yes" and not seen_after:
        errors.append("no_stable_visible_reappearance")
    if demand.answer == "no" and fully and hidden != list(range(first_hidden, demand.frame_count)):
        errors.append("negative_branch_not_hidden_through_window_end")
    return {
        "status": "pass" if not errors else "fail",
        "errors": sorted(set(errors)), "observed_answer": observed,
        "complete_window": complete, "demand": asdict(demand),
        "fully_occluded_frames": fully, "reappeared_frames": after,
        "registered_hidden_intervals_frames": hidden_runs,
        "visible_before_intervals_frames": seen_before,
        "visible_after_intervals_frames": seen_after,
        "clear_before_intervals_frames": [r for r in clear_runs if r[1] < first_hidden],
        "clear_after_intervals_frames": [r for r in clear_runs if any(r[0] > h[1] for h in hidden_runs)],
        "scope": "construction preflight; final original QA independently evaluates native full stream",
    }


def probe_native_sequence(video: Any, scene: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
                          demand: ReappearanceDemand) -> dict[str, Any]:
    """Reversibly sample every requested native frame after actual motion setup.

    The integration-owned native relation dispatcher calls this only once both
    requested entries are placed. It must run before final video/audio rendering.
    No prediction, mask or QA answer is written into the production truth stream.
    """
    import numpy as np
    from avengine.capture.continuous_native_visibility import NativeVisibilityStream

    if scene.get("video_fps", 15) != demand.frame_rate_hz:
        raise ValueError("QA-09 preflight must use the actual native mask frame rate")
    by_asset = {e["declaration"]["asset_id"]: e for e in entries}
    target, occluder = (by_asset[aid] for aid in (demand.target_asset_id, demand.occluder_asset_id))
    tid, oid = (str(e["declaration"]["native_instance_id"]) for e in (target, occluder))
    identities = [{"endpoint_id": e["endpoint_id"], "actor_id": e["declaration"]["native_instance_id"]}
                  for e in video.actors]
    stream = NativeVisibilityStream(video)
    frames: list[Mapping[str, Any]] = []
    try:
        for index in range(demand.frame_count):
            at = demand.window_start_s + index / demand.frame_rate_hz
            for entry in entries:
                if entry.get("motion_episode"):
                    video._apply_motion_entry(entry, round(at * 30))
            video._native_pair_pixels(scene, entries)
            modal = np.asarray(video.sim.get_sensor_observations()["semantic"]).copy()
            result = stream.capture(index, modal, identities)
            frames.append(result["per_instance"][tid]["frames"][0])
        certificate = certify_mask_sequence(frames, stream.occluder_frames, demand,
            target_instance_id=tid, occluder_instance_id=oid,
            registered_instance_ids=[str(r["actor_id"]) for r in identities])
    finally:
        for entry in entries:
            if entry.get("motion_episode"):
                video._apply_motion_entry(entry, 0)
        video._native_pair_pixels(scene, entries)
        stream.close()
    certificate["native_control_restoration"] = stream.receipt()
    certificate["native_mask_frames"] = list(frames)
    certificate["native_occluder_frames"] = [r for r in stream.occluder_frames if r["target_instance_id"] == tid]
    return certificate


def clock_anchored_positions(distance: Any, route: Mapping[str, Any]) -> Any:
    """Fit the unchanged complete-step clock around the solved hidden stop.

    Gait quantization may change the first walk's length. Anchor that walk's
    *end*, then recertify the resulting start/end, rather than allowing the
    hidden stop to drift outside the narrow native silhouette shadow.
    """
    import numpy as np
    distance = np.asarray(distance, dtype=float)
    if distance.shape != (901,) or not np.isfinite(distance).all():
        raise ValueError("QA-09 needs the actual complete 901-frame gait distance clock")
    if abs(distance[0]) > 1e-8 or np.min(np.diff(distance)) < -1e-8:
        raise ValueError("QA-09 distance clock must start at zero and never run backwards")
    direction = np.asarray(route["cross_direction_xz"], dtype=float)
    shadow = np.asarray(route["shadow_root_m"], dtype=float)
    if (direction.shape != (2,) or shadow.shape != (3,) or not np.isfinite(direction).all()
            or not np.isfinite(shadow).all() or abs(np.linalg.norm(direction) - 1) > 1e-6):
        raise ValueError("QA-09 requires a finite shadow root and unit crossing direction")
    lo, hi = (float(v) for v in route.get("hidden_interval_s", [4.5, 5.5]))
    if not 0 <= lo < hi < 10 or hi - lo < 1.0 - 1e-8:
        raise ValueError("QA-09 solved hidden stop must last at least one second in the first window")
    first, last = math.ceil(lo * 30), math.floor(hi * 30)
    anchor = float(distance[first])
    if anchor <= 0 or np.ptp(distance[first:last + 1]) > 1e-8:
        raise ValueError("complete-step gait did not stop in the solved hidden interval")
    answer = route["answer"]
    if answer == "no":
        if np.ptp(distance[first:]) > 1e-8:
            raise ValueError("negative QA-09 request must stay at the hidden stop")
    elif answer == "yes":
        if distance[299] <= anchor + 1e-8:
            raise ValueError("positive QA-09 request never leaves the hidden stop in its window")
    else:
        raise ValueError("QA-09 route has no selected yes/no answer")
    vector = np.array([direction[0], 0., direction[1]])
    return shadow + (distance - anchor)[:, None] * vector


def native_qa09_route_proposals(video: Any, entry: Mapping[str, Any], scene: Mapping[str, Any],
                                distance: Any, local_forward: Any, request: Mapping[str, Any]):
    """Motion-owner hook: one solved route, recertified on the actual gait clock."""
    import numpy as np
    from types import SimpleNamespace
    from avengine.construction.joint import (ActorPlan, BodySpec, CameraPose, ScenePlan,
        build_room_context, certify_plan, static_actor)

    solved = request["joint_qa09_route"]
    planned_distance = np.asarray(solved["planned_distance_m"], float)
    positions = np.asarray(solved["positions_m"], float).copy()
    if (planned_distance.shape != (901,) or positions.shape != (901, 3)
            or not np.isfinite(positions).all()
            or not np.allclose(np.asarray(distance), planned_distance, atol=1e-8, rtol=0)):
        raise ValueError("native gait clock differs from the solved QA-09 clock; return conflict to solver")
    if not np.allclose(np.asarray(entry["applied_root_input"])[:3, 3], positions[0], atol=.0002, rtol=0):
        raise ValueError("native standing placement differs from the solved QA-09 start; no execution relocation")
    camera = CameraPose(**solved["camera"])
    if not np.allclose(camera.position_m, scene["listener_m"], atol=1e-6, rtol=0):
        raise ValueError("native camera moved away from the QA-09 solved camera")
    if not np.allclose(camera.forward, scene["camera_forward"], atol=1e-6, rtol=0):
        raise ValueError("native camera yaw differs from QA-09 solved yaw")
    aid = entry["declaration"]["asset_id"]
    body = BodySpec(**solved["target_body"])
    active = np.flatnonzero(np.diff(np.asarray(distance)) > 1e-8)
    chunks = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1) if len(active) else []
    intervals = [[float(c[0]) / 30, float(c[-1] + 1) / 30] for c in chunks if len(c)]
    actors = {aid: ActorPlan(aid, positions, body, metadata={"walk_intervals_s": intervals, "ramp_s": .3})}
    for other_id, placement in solved["static_actors"].items():
        actors[other_id] = static_actor(other_id, placement["root_m"], BodySpec(**placement["body"]))
    room = dict(video.motion_house["room"])
    room["walkable_map_path"] = solved["walkable_map_path"]
    room.setdefault("room_label", solved["room_id"])
    allow_fallback = bool(solved.get("allow_native_map_fallback", False))
    raycaster = None
    if allow_fallback:
        from avengine.assets.v7_phase2_floor import original_native_stage_mesh
        from avengine.rooms.walkable_map_builder import stage_path
        from avengine.construction.joint import MeshRaycaster
        mesh, evidence = original_native_stage_mesh(video, stage_path(video.motion_house))
        raycaster = MeshRaycaster(mesh, evidence=evidence)
    with build_room_context(room, pathfinder=video.pathfinder, raycaster=raycaster, camera_limit=1,
                            allow_native_map_fallback=allow_fallback,
                            width=camera.width, height=camera.height) as context:
        certificate = certify_plan(context, ScenePlan(camera, actors))
    if not planning_admissible(certificate, native_sweep_pending=allow_fallback):
        reasons = [c.name + ":" + c.reason for c in certificate.checks if c.status != "pass"]
        raise ValueError("executed QA-09 clock invalidates solved geometry: " + "; ".join(reasons))
    direction = np.asarray(solved["cross_direction_xz"])
    heading = math.atan2(float(direction[1]), float(direction[0]))
    local = np.asarray(local_forward)
    yaw = math.atan2(float(local[2]), float(local[0])) - heading
    planned_yaws = np.asarray(solved.get("native_yaws_rad", np.full(901, yaw)), float)
    if planned_yaws.shape != (901,) or not np.isfinite(planned_yaws).all():
        raise ValueError("QA-09 requires all planned native yaw samples")
    yield SimpleNamespace(positions=positions, yaws=planned_yaws.copy(), receipt={
        "route_kind": "joint QA-09 crossing anchored at registered full-shadow stop",
        "answer_requested": solved["answer"], "root_samples": 901,
        "hidden_interval_s": list(solved.get("hidden_interval_s", [4.5, 5.5])),
        "actual_initial_root_m": positions[0].tolist(),
        "shadow_root_m": list(solved["shadow_root_m"]),
        "proposal_clock_changed": False, "camera_changed": False, "execution_replanning": False,
        "geometry_certificate": [asdict(c) for c in certificate.checks],
        "native_full_pose_checks_still_required": True,
        "complete_native_mask_preflight_still_required": True,
    })


TURNAROUND_INTERVAL_S = (5.6, 6.6)


def gait_intervals(answer, positive_route_kind="arc"):
    if answer not in {"yes", "no"} or positive_route_kind not in {"arc", "out_and_back"}:
        raise ValueError("unknown QA-09 gait branch or route kind")
    if answer == "no":
        return [[1.5, 3.3]]
    return [[1.5, 3.3], [6.8, 8.6] if positive_route_kind == "out_and_back" else [6., 7.8]]


def prepare_gait_geometry(asset: Mapping[str, Any], runtime: Mapping[str, Any],
                          house: Mapping[str, Any], answer: str) -> dict[str, Any]:
    """Reuse the admitted gait clock and IK on a declared flat planning floor.

    This supplies actual posed visual triangles to joint_core, not a static
    standing silhouette. Native compilation must recheck the real floor and
    the complete final mask stream. A cache is an ordinary derived asset cache.
    """
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    from avengine.assets.roster_motion.episode import admitted_master, triangle_mesh
    from avengine.assets.roster_motion.contact import FootPlant
    from avengine.construction.joint import admitted_human_clock_source, complete_human_gait_clock

    if answer not in {"yes", "no"}:
        raise ValueError("unknown QA-09 gait branch")
    route_kind = str(runtime.get("qa09_positive_route_kind", "arc")) if answer == "yes" else "arc"
    intervals = gait_intervals(answer, route_kind)
    arc_radius = float(runtime.get("qa09_arc_radius_m", .8))
    if not math.isfinite(arc_radius) or not .4 <= arc_radius <= 1.2:
        raise ValueError("QA-09 arc radius must be between .4 and 1.2 metres")
    d, b, action, lineage = admitted_master(
        {"declaration": asset, "prepared_glb": asset["habitat_glb"]}, runtime, house)
    key_data = {"revision": "qa09-arc-gait-2" if answer == "yes" else "qa09-flat-gait-1", "asset_id": asset["asset_id"],
                "master_sha256": lineage["master_sha256"], "clip": lineage["admitted_clip"],
                "intervals_s": intervals, "speed_m_s": 1.0}
    if answer == "yes" and route_kind == "out_and_back":
        key_data.update(revision="qa09-out-and-back-gait-1", route_kind="out_and_back",
                        stationary_turn_interval_s=list(TURNAROUND_INTERVAL_S))
    if answer == "yes" and route_kind == "arc" and arc_radius != .8:
        key_data.update(revision="qa09-arc-gait-3", turn_radius_m=arc_radius)
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:24]
    directory = runtime.get("joint_cache_directory")
    path = Path(directory) / ("qa09_gait_" + key + ".npz") if directory else None
    if path is not None and path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return {"vertices": data["vertices"], "faces": data["faces"], "distance": data["distance"],
                    "metadata": json.loads(str(data["metadata"].item()))}
    source = admitted_human_clock_source(asset, runtime, house)
    if source["source_lineage"] != lineage:
        raise ValueError("QA-09 clock and pose source must share the same admitted lineage")
    fitted = complete_human_gait_clock(source, 1., intervals)
    ts = source["source_times_s"]
    times, distance, clock = fitted["times_s"], fitted["distance_m"], fitted["phase_s"]
    receipt = fitted["certificate"]
    from avengine.assets.roster_motion.straight_route import _shape_xy
    anchor = float(distance[135])
    if answer == "yes" and route_kind == "out_and_back":
        from avengine.construction.joint import out_and_back_shape
        positions, headings = out_and_back_shape([-anchor, 0., 0.], [1., 0.], distance, anchor,
                                                  turn_interval_s=TURNAROUND_INTERVAL_S)
    else:
        corner = (anchor, arc_radius, (float(distance[-1])-anchor)/arc_radius) if answer == "yes" else None
        xz, headings = _shape_xy(distance, np.array([-anchor, 0.]), 0., corner)
        positions = np.column_stack((xz[:, 0], np.zeros(901), xz[:, 1]))
    yaws = -math.pi/2-headings

    class FlatPlanningFloor:
        # Match native phase2's numerical sole-switch policy while explicitly
        # retaining a flat-floor assumption in the resulting planning receipt.
        original_ray = True
        def __call__(self, point):
            return 0.0
        def many(self, points):
            return np.zeros(len(points))

    solver = FootPlant(d, b, action, clock.tolist(), positions.tolist(), yaws.tolist(), 1.,
                      float(ts[0]), float(ts[-1] - ts[0]), FlatPlanningFloor())
    solver.times = times
    vertices = []
    faces = None
    for index in range(300):
        _, parts, _ = solver.frame(index)
        if index % 2 == 0:
            v, faces = triangle_mesh(d, b, parts)
            vertices.append(v.astype(np.float32))  # Original rigid_glb also stores float32 positions.
    result = {"vertices": np.stack(vertices), "faces": np.asarray(faces, np.int32), "distance": distance,
              "metadata": {**key_data, "lineage": lineage, "clock": receipt,
                           "frame_indices_30hz": list(range(0, 300, 2)),
                           "authority": "admitted_pose_plus_original_IK_on_flat_planning_floor",
                           "native_room_admission": False}}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile, os
        fd, temporary = tempfile.mkstemp(prefix=".qa09-gait-", suffix=".npz", dir=path.parent)
        os.close(fd)
        try:
            np.savez_compressed(temporary, vertices=result["vertices"], faces=result["faces"],
                                distance=distance, metadata=np.array(json.dumps(result["metadata"])))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return result


def _mesh_instance(asset_id, raycaster, root, yaw):
    import numpy as np
    from avengine.capture.v7_visible_attachment import rotation
    from avengine.construction.joint import MeshInstance
    matrix = np.eye(4)
    matrix[:3, :3] = rotation(yaw)
    matrix[:3, 3] = root
    return MeshInstance(asset_id, raycaster, matrix)


def select_roles(room, catalog, contract, state, seed):
    """Select measured small walker / large foreground, preserving real identities."""
    from avengine.timeline.v7_family_constructors import eligible_humans, stable_seed, ConstructionUnavailable
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    import numpy as np
    pool = eligible_humans(catalog, room, contract)
    groups = {}
    for asset in pool:
        groups.setdefault(asset["body_identity_id"], []).append(asset)
    dimensions = {}
    for body, variants in groups.items():
        dimensions[body] = np.ptp(indexed_static_mesh(variants[0]["habitat_glb"]).vertices, axis=0)
    small = sorted((a for a in pool if a.get("walk_allowed")), key=lambda a: (
        dimensions[a["body_identity_id"]][1], stable_seed(seed, "target", a["asset_id"])))
    if not small:
        raise ConstructionUnavailable("QA09 has no legal admitted walker")
    target = small[0]
    large = [a for a in pool if a["body_identity_id"] != target["body_identity_id"] and a["color"] != target["color"]]
    large.sort(key=lambda a: (-dimensions[a["body_identity_id"]][1], -dimensions[a["body_identity_id"]][0],
                              stable_seed(seed, "foreground", a["asset_id"])))
    if not large:
        raise ConstructionUnavailable("QA09 has no distinct registered foreground person")
    foreground = large[0]
    return target, foreground


def prediction_certificate(predictions, demand, foreground_id):
    """Family temporal obligation over geometric predictions, never native truth."""
    from avengine.construction.joint import CheckResult
    if len(predictions) != demand.frame_count:
        return CheckResult("QA09:temporal", "fail", "incomplete_geometric_prediction_window")
    states = [{"frame_index": i, "state": p.state} for i, p in enumerate(predictions)]
    fully = [int(i) for i in full_occlusion_frames(states)]
    later = reappeared_frames(states, fully)
    predicted = ("yes" if later else "no") if fully else None
    hidden = [i for i, p in enumerate(predictions) if p.state == "fully_occluded"
              and p.visible_pixels_upper == 0 and p.depth_margin_m >= .04
              and p.occluder_ids == (foreground_id,)
              and p.evidence.get("unknown_occluder_pixels") == 0]
    clear = [i for i, p in enumerate(predictions) if p.state in {"visible_clear", "visible_occluded"}
             and p.visible_pixels_lower >= 300 and p.edge_margin_px >= 4]
    hr = _runs(hidden, demand.frame_rate_hz, demand.minimum_hidden_s)
    vr = _runs(clear, demand.frame_rate_hz, demand.minimum_visible_s)
    before = bool(fully and any(r[1] < min(fully) for r in vr))
    after = any(v[0] > h[1] for v in vr for h in hr)
    errors = []
    if predicted != demand.answer:
        errors.append("geometric_branch_mismatch")
    if not hr:
        errors.append("no_stable_exact_mesh_shadow")
    if not before:
        errors.append("no_stable_initial_appearance")
    if demand.answer == "yes" and not after:
        errors.append("no_stable_mesh_reappearance")
    if demand.answer == "no" and (not fully or hidden != list(range(min(fully), demand.frame_count))):
        errors.append("negative_prediction_not_hidden_until_end")
    if any(p.state == "unknown" for p in predictions):
        errors.append("unknown_visual_geometry")
    return CheckResult("QA09:temporal", "fail" if errors else "pass", ";".join(errors), evidence={
        "predicted_answer": predicted, "hidden_intervals_frames": hr, "visible_intervals_frames": vr,
        "prediction_frames": len(predictions), "native_truth": False,
        "pose_floor_assumption": "flat planning floor; actual native floor and masks must requalify",
        "minimum_depth_margin_m": .04, "required_native_visible_pixels": demand.minimum_visible_pixels,
    })


class QA09Plugin:
    """Place a solved crossing behind a registered standing person's real mesh."""
    def __init__(self, target, foreground, companion, gait, *, seed=0, geometry_budget_s=120., sounds=()):
        import numpy as np
        import trimesh
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        from avengine.construction.joint import MeshRaycaster, BodySpec
        self.target, self.foreground, self.companion = target, foreground, companion
        self.gait, self.seed, self.geometry_budget_s = gait, seed, geometry_budget_s
        self.sounds = tuple(sounds)
        selected_assets = [a for a in (target, foreground, companion) if a is not None]
        self.supporters = [a for a in (foreground, companion) if a is not None]
        self.meshes = {a["asset_id"]: indexed_static_mesh(a["habitat_glb"]) for a in selected_assets}
        self.rays = {aid: MeshRaycaster(mesh, evidence={"registered_asset_id": aid,
                       "visual_mesh": next(a["habitat_glb"] for a in selected_assets if a["asset_id"] == aid)})
                     for aid, mesh in self.meshes.items()}
        self.target_rays = {}
        self.bodies = {}
        for aid, mesh in self.meshes.items():
            vertices = np.asarray(gait["vertices"]).reshape(-1, 3) if aid == target["asset_id"] else np.asarray(mesh.vertices)
            # The full registered/posed radial footprint, with no asset scaling.
            radius = float(np.linalg.norm(vertices[:, [0, 2]], axis=1).max())
            self.bodies[aid] = BodySpec("human", radius, float(vertices[:, 1].max()), .03)
        self.diagnostics = {"geometric_layouts": 0, "common_rejections": {}, "mesh_rejections": {}}

    def _target_ray(self, frame):
        import trimesh
        from avengine.construction.joint import MeshRaycaster
        # Reuse exactly identical held poses (memoization only, not an admission identity).
        vertices = self.gait["vertices"][frame]
        key = vertices.tobytes()
        if key not in self.target_rays:
            self.target_rays[key] = MeshRaycaster(trimesh.Trimesh(vertices, self.gait["faces"], process=False),
                evidence={"asset_id": self.target["asset_id"], "pose_source": self.gait["metadata"]["lineage"],
                          "pose_floor_assumption": "flat planning floor"})
        return self.target_rays[key]

    def _predict(self, context, plan, demand):
        import numpy as np
        from avengine.construction.joint import build_depth_buffer, predict_mesh_visibility
        aid = self.target["asset_id"]
        # One depth buffer at a time avoids retaining many full-resolution ID images.
        depth = build_depth_buffer(plan.camera, context.raycaster)
        yaws = plan.metadata["body_yaws_rad"]
        occluders = [_mesh_instance(a["asset_id"], self.rays[a["asset_id"]],
                    plan.actors[a["asset_id"]].positions_m[0], yaws[a["asset_id"]])
                    for a in self.supporters]
        predictions = []
        cache = {}
        for frame in range(demand.frame_count):
            position = plan.actors[aid].positions_m[frame * 2]
            ray = self._target_ray(frame)
            key = (id(ray), tuple(position), float(plan.actors[aid].metadata["native_yaw_rad_per_frame"][frame*2]))
            if key not in cache:
                cache[key] = predict_mesh_visibility(plan.camera,
                    _mesh_instance(aid, ray, position, float(plan.actors[aid].metadata["native_yaw_rad_per_frame"][frame*2])), scene_depth=depth, occluders=occluders)
            predictions.append(cache[key])
        return predictions

    def candidates(self, context, request):
        import time
        import numpy as np
        from avengine.construction.joint import CameraPose, ActorPlan, ScenePlan, static_actor, certify_plan, framing, bind_distance_clock, straight_then_arc, out_and_back, ConstructionError
        from avengine.timeline.v7_family_constructors import stable_seed
        demand = ReappearanceDemand(self.target["asset_id"], self.foreground["asset_id"], request.branch)
        started = time.monotonic()
        seen_cameras = set()
        points = context.points_m
        self.diagnostics.update(camera_candidates=len(context.cameras), floor_points=len(points),
                                body_envelopes={k: asdict(v) for k, v in self.bodies.items()})
        def rejected(reason):
            table = self.diagnostics.setdefault("prefilter_rejections", {})
            table[reason] = table.get(reason, 0) + 1
        tids = [self.target["asset_id"], self.foreground["asset_id"]]
        if self.companion is not None:
            tids.append(self.companion["asset_id"])
        distance = self.gait["distance"]
        active = np.flatnonzero(np.diff(distance) > 1e-8)
        chunks = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
        intervals = [[float(c[0])/30, float(c[-1]+1)/30] for c in chunks if len(c)]
        for original in context.cameras:
            pos = tuple(original.position_m)
            if pos in seen_cameras:
                continue
            seen_cameras.add(pos)
            delta = points[:, [0, 2]] - np.asarray(pos)[[0, 2]]
            radii = np.linalg.norm(delta, axis=1)
            possible = np.flatnonzero((radii >= 4.9) & (radii <= 6.0))
            possible = sorted(possible, key=lambda i: (abs(radii[i]-5.0), float(points[i, 0]), float(points[i, 2])))
            for point_index in possible[:48]:
                shadow = points[point_index].copy()
                f = delta[point_index] / radii[point_index]
                right = np.array([-f[1], f[0]])
                yaw = math.degrees(math.atan2(f[0], -f[1]))
                camera = CameraPose(pos, yaw, width=original.width, height=original.height)
                for side in ((1.,) if request.branch == "yes" else (1., -1.)):
                    # Diagonal passage makes the final planted gait narrower in
                    # projection than a broadside walk, without changing its mesh.
                    direction = .5 * side * right + math.sqrt(.75) * f
                    body_yaw = -math.pi/2-math.atan2(direction[1], direction[0])
                    route_kind = self.gait["metadata"].get("route_kind", "arc" if request.branch == "yes" else "straight")
                    route = {"answer": request.branch, "route_kind": route_kind, "shadow_root_m": shadow.tolist(),
                             "cross_direction_xz": direction.tolist(), "hidden_interval_s": [4.5, 5.5]}
                    anchor = float(distance[135])
                    origin = shadow.copy(); origin[[0, 2]] -= anchor*direction
                    # Ask the shared original-visual-mesh raycaster for the actual
                    # start floor, rather than borrowing the distant shadow cell Y.
                    if context.raycaster is None:
                        rejected("missing_original_visual_floor"); continue
                    ray_origin = origin + [0., .2, 0.]
                    hit, _ = context.raycaster.first_hit(ray_origin, np.array([[0., -1., 0.]]))
                    if not np.isfinite(hit[0]) or not .12 <= hit[0] <= .28:
                        rejected("start_physical_floor_level"); continue
                    origin[1] = shadow[1] = ray_origin[1] - hit[0]
                    route["shadow_root_m"] = shadow.tolist()
                    try:
                        if request.branch == "yes" and route_kind == "out_and_back":
                            moving_actor = out_and_back(context, origin, direction, distance, anchor,
                                turn_interval_s=TURNAROUND_INTERVAL_S, camera=camera, entity_id=tids[0],
                                body=self.bodies[tids[0]], ramp_s=.3)
                        elif request.branch == "yes":
                            moving_actor = straight_then_arc(context, shadow, direction, distance, anchor,
                                turn_radius_m=float(self.gait["metadata"].get("turn_radius_m", .8)),
                                turn_sign=1, camera=camera, entity_id=tids[0],
                                body=self.bodies[tids[0]], ramp_s=.3)
                        else:
                            moving_actor = bind_distance_clock(context, distance, origin_m=origin,
                                heading_rad=math.atan2(direction[1], direction[0]), camera=camera,
                                entity_id=tids[0], body=self.bodies[tids[0]], ramp_s=.3)
                    except ConstructionError as exc:
                        rejected("core_route:" + exc.reason); continue
                    track = moving_actor.positions_m
                    native_yaws = -math.pi/2-np.asarray(moving_actor.headings_rad)
                    body_yaw = float(native_yaws[0])
                    moving_actor.metadata.update(native_yaw_rad=body_yaw,
                        native_yaw_rad_per_frame=native_yaws.tolist())
                    if not all(framing(camera, track[i], self.bodies[tids[0]])[0] == "inside" for i in (0, 135, 299, 900)):
                        rejected("target_rest_framing"); continue
                    for foredepth in (2.8, 3.0):
                        foreground = np.array([pos[0]+f[0]*foredepth, shadow[1], pos[2]+f[1]*foredepth])
                        fore_origin = foreground + [0., .2, 0.]
                        fore_hit, _ = context.raycaster.first_hit(fore_origin, np.array([[0., -1., 0.]]))
                        if not np.isfinite(fore_hit[0]) or not .12 <= fore_hit[0] <= .28:
                            rejected("foreground_physical_floor_level"); continue
                        foreground[1] = fore_origin[1] - fore_hit[0]
                        if not context.walkable_map.contains(foreground[[0, 2]], "human"):
                            rejected("foreground_map"); continue
                        if framing(camera, foreground, self.bodies[tids[1]])[0] != "inside":
                            rejected("foreground_framing"); continue
                        if radii[point_index] - foredepth < 2.1 - 1e-8:
                            rejected("foreground_target_separation_margin"); continue
                        actors = {tids[0]: moving_actor,
                                  tids[1]: static_actor(tids[1], foreground, self.bodies[tids[1]])}
                        yaws = {tids[0]: body_yaw, tids[1]: body_yaw}
                        actors[tids[1]].metadata["native_yaw_rad"] = body_yaw
                        if self.companion is not None:
                            preferred = np.array([pos[0], pos[2]]) + f*3.1 + side*right*1.7
                            ordering = np.argsort(np.linalg.norm(points[:, [0, 2]]-preferred, axis=1))
                            companion = None
                            for ci in ordering[:64]:
                                q = points[ci]
                                if (np.linalg.norm(q[[0, 2]]-foreground[[0, 2]]) >= 1.05
                                    and np.min(np.linalg.norm(track[:, [0, 2]]-q[[0, 2]], axis=1)) >= 1.05
                                    and framing(camera, q, self.bodies[tids[2]])[0] == "inside"):
                                    point_plan = ScenePlan(camera, {tids[2]: static_actor(tids[2], q, self.bodies[tids[2]], duration_s=1, fps=1)},
                                                           frame_rate_hz=1, duration_s=1)
                                    if certify_plan(context, point_plan).passed:
                                        companion = q.copy(); break
                            if companion is None:
                                rejected("no_companion_site"); continue
                            actors[tids[2]] = static_actor(tids[2], companion, self.bodies[tids[2]])
                            yaws[tids[2]] = math.radians(stable_seed(self.seed, "companion_yaw") % 360)
                            actors[tids[2]].metadata["native_yaw_rad"] = yaws[tids[2]]
                        plan = ScenePlan(camera, actors, sounds=self.sounds, metadata={"body_yaws_rad": yaws, "qa09_route": route,
                                         "qa09_demand": asdict(demand),
                                         "allow_native_map_fallback": context.allow_native_map_fallback})
                        self.diagnostics["geometric_layouts"] += 1
                        common = certify_plan(context, plan)
                        if not planning_admissible(common, native_sweep_pending=bool(request.options.get("require_native_preflight"))):
                            for check in common.checks:
                                if check.status != "pass":
                                    key = check.name+":"+check.reason
                                    self.diagnostics["common_rejections"][key] = self.diagnostics["common_rejections"].get(key, 0)+1
                            continue
                        plan.metadata["qa09_native_sweep_pending"] = any(
                            c.status == "unknown" and c.reason == "native_full_pose_required_for_map_fallback"
                            for c in common.checks)
                        predictions = self._predict(context, plan, demand)
                        family = prediction_certificate(predictions, demand, tids[1])
                        if family.status != "pass":
                            self.diagnostics["mesh_rejections"][family.reason] = self.diagnostics["mesh_rejections"].get(family.reason, 0)+1
                        else:
                            plan.metadata["qa09_prediction_check"] = asdict(family)
                            plan.metadata["qa09_prediction_states"] = [p.state for p in predictions]
                            yielded_at = time.monotonic()
                            yield plan
                            started += time.monotonic()-yielded_at  # Core owns the native preparation budget.
                        if time.monotonic()-started >= self.geometry_budget_s:
                            return
                if time.monotonic()-started >= self.geometry_budget_s:
                    return

    def verify(self, context, request, plan):
        from avengine.construction.joint import CheckResult
        value = plan.metadata.get("qa09_prediction_check")
        if value is None:
            yield CheckResult("QA09:temporal", "unknown", "missing_exact_mesh_prediction")
        else:
            yield CheckResult(**value)


def build_qa09_row(room, house, state, catalog, contract, plan, certificate, roles, authored, gait,
                   walkable_map_path, *, scene_id, seed, native_sweep_pending=False):
    from copy import deepcopy
    from avengine.timeline.v7_family_constructors import _request
    from avengine.timeline.v7_production_inputs import validate_recipe
    from avengine.construction.joint import strict_native_bindings
    from avengine.assets.roster_motion.navmesh_execution import REVISION as EXECUTION_REVISION
    if not planning_admissible(certificate, native_sweep_pending=native_sweep_pending):
        raise ValueError("QA-09 row requires the joint planning certificate")
    target, foreground = roles[:2]
    ids = [a["asset_id"] for a in roles]
    speech_id = ids[-1]
    demand = ReappearanceDemand(**plan.metadata["qa09_demand"])
    yaws = plan.metadata["body_yaws_rad"]
    route = deepcopy(plan.metadata["qa09_route"])
    route.update(allow_native_map_fallback=bool(plan.metadata.get("allow_native_map_fallback", False)),
        planned_distance_m=gait["distance"].tolist(), positions_m=plan.actors[ids[0]].positions_m.tolist(),
        native_yaws_rad=plan.actors[ids[0]].metadata["native_yaw_rad_per_frame"],
        camera=asdict(plan.camera), target_body=asdict(plan.actors[ids[0]].body),
        room_id=room["room_id"], walkable_map_path=str(walkable_map_path),
        static_actors={aid: {"root_m": plan.actors[aid].positions_m[0].tolist(),
                            "body": asdict(plan.actors[aid].body)} for aid in ids[1:]})
    row = {"schema": "v7-visible-registered-pilot-phase2-v1", "scene_id": scene_id,
        "independent_parent_scene_id": scene_id, "scene_seed": seed, "house": room["house"],
        "room_id": room["room_id"], "split": "train", "family": room["family"],
        "listener_m": list(plan.camera.position_m), "camera_yaw_deg": float(plan.camera.yaw_deg),
        "camera_yaw_knots": [{"time_s": t, "yaw_deg": float(plan.camera.yaw_deg)} for t in (0., 30.)],
        "asset_ids": ids, "silent_asset_ids": [aid for aid in ids if aid != speech_id], "source_classes": {speech_id: ["speech_playback"]},
        "semantic_scenario_id": authored["scenario_id"], "semantic_sound_ids": [authored["sound_asset_id"]],
        "semantic_human_asset_ids": [speech_id], "semantic_start_offsets_s": [0.],
        "phase2_authored_speech_schedule_s": {speech_id: [0., 10., 20.]},
        "motion_requests": {ids[0]: {"action": "walk", "speed_m_s": 1.,
            "intervals_s": deepcopy(gait["metadata"]["intervals_s"]),
            "minimum_listener_horizontal_distance_m": 1.5, "scene_speed_range_m_s": [.8, 1.4],
            "motion_execution_revision": EXECUTION_REVISION, "joint_qa09_route": route}},
        "motion_execution_revision": EXECUTION_REVISION,
        "native_view_roles": {aid: {"view": "inside", "static": aid != ids[0],
                                    "required_visible_sometime": True} for aid in ids},
        "native_dynamic_occlusion_relations": [demand.relation()],
        "members": ["v0_a0"], "trial_only": True, "acceptance_allow_partial_families": True,
        "window_count": 5, "questions_per_window": 14, "question_quota_base": 0,
        "acceptance_window_starts_s": [0., 5., 10., 15., 20.],
        "automatic_construction": {"revision": "joint-qa09-gait-shadow-v1", "family": "QA-09",
            "constructor": "construct_qa09_scene", "candidate_index": 0, "room_key": room["key"],
            "required_role_count": len(ids), "scene_template_files_read": [], "requested_answer": demand.answer,
            "joint_geometry_certificate": asdict(certificate), "gait_prediction": gait["metadata"],
            "predicted_states_first_window": plan.metadata["qa09_prediction_states"],
            "native_preflight_required": True},
        "owner_social_layout": {"minimum_human_center_distance_m": 1.,
            "minimum_human_camera_distance_m": 1.5, "human_orientation": "seed-selected before joint geometry certification",
            "distribution": "passing target and registered foreground speaker"}}
    row.update(strict_native_bindings(plan, roles))
    _request(row, state, "QA-09")
    row["scene_demand"]["requested_answer"] = demand.answer
    row["scene_demand"]["requested_conditions"] = demand.requested_conditions()
    validate_recipe(row, catalog, contract)
    return {"profile": "phase2", "row": row}


def construct_qa09_scene(room, house, state, catalog, records, semantic, contract, runtime,
                         *, scene_id, seed, family="QA-09", attempt=0):
    """Joint camera, actual gait silhouette, registered foreground, and answer."""
    import os
    from pathlib import Path
    from avengine.construction.joint import build_room_context, solve, FamilyRequest, SoundEvent, Certificate, CheckResult, certify_plan, prepare_native
    from avengine.rooms.walkable_map import room_map_path
    from avengine.timeline.v7_family_constructors import room_context, choose_authored, ConstructionUnavailable
    if family != "QA-09":
        raise ConstructionUnavailable("QA09 plugin received another family")
    if room["family"] not in {"hm3d", "mp3d"}:
        raise ConstructionUnavailable("QA09 plugin requires the admitted native Habitat route")
    geometric_only = bool(runtime.get("qa09_geometric_only", False))
    if not geometric_only and not runtime.get("qa09_oracle_only"):
        import inspect
        from avengine.timeline.v7_phase2_generation import generate_phase2_parent
        if "prepared_native" not in inspect.signature(generate_phase2_parent).parameters:
            raise ConstructionUnavailable("QA09 formal generation requires the shared prepared_native consumer API")
    target, foreground = roles = select_roles(room, catalog, contract, state, seed)
    companion = foreground
    answer = str(runtime.get("qa09_requested_answer") or requested_answer(state, seed))
    gait = prepare_gait_geometry(target, runtime, house, answer)
    minimum_planning_skin_y = float(gait["vertices"][:, :, 1].min())
    if minimum_planning_skin_y < -.01:
        raise ConstructionUnavailable(
            f"QA09 canonical pose penetrates the declared flat planning floor by {-minimum_planning_skin_y:.6f}m; "
            "do not spend native preparation on this gait")
    authored = choose_authored(semantic, [companion], state, seed, attempt=attempt)[0][0]
    duration = float(authored["synthesis_readback"]["duration_s"])
    sounds = tuple(SoundEvent(companion["asset_id"], authored["sound_asset_id"], "speech_playback", t, t+duration)
                   for t in (0., 10., 20.))
    actual = room_context(room, house)["original_room"]
    map_root = runtime.get("walkable_map_root") or os.environ.get("AVENGINE_WALKABLE_MAP_ROOT")
    if not map_root:
        raise ConstructionUnavailable("QA09 requires the selected v2 walkable-map root")
    path = room_map_path(map_root, room["room_id"])
    actual = dict(actual, walkable_map_path=str(path))
    plugin = QA09Plugin(target, foreground, None, gait, seed=seed, sounds=sounds,
                       geometry_budget_s=float(runtime.get("qa09_geometry_budget_s", 120.)))
    cache = runtime.get("joint_cache_directory")
    with build_room_context(actual, house=house, runtime=runtime,
            cache_dir=Path(cache)/room["room_id"] if cache else None,
            camera_limit=int(runtime.get("qa09_camera_limit", 72)), grid_step_m=.3,
            camera_heights_m=[1.1, 1.25], width=1280, height=720,
            allow_native_map_fallback=bool(runtime.get("qa09_allow_native_map_fallback", False)) and not geometric_only) as context:
        def make_row(plan):
            cert = Certificate(certify_plan(context, plan).checks + tuple(plugin.verify(context, request, plan)))
            result = build_qa09_row(room, house, state, catalog, contract, plan, cert,
                                   roles, authored, gait, path, scene_id=scene_id, seed=seed,
                                   native_sweep_pending=not geometric_only)
            from avengine.assets.roster_motion.navmesh_execution import prepare_scene_motion
            return prepare_scene_motion(result["row"], catalog)

        def family_native_checks(video, scene, entries, *, common_sweep_required=False):
            report = getattr(video, "family_native_preflights", {}).get("QA-09", {})
            certificates = [r.get("native_complete_window_certificate", {}) for r in report.get("relations", [])
                            if r.get("target_asset_id") == target["asset_id"]]
            passed = (report.get("status") == "pass" and len(certificates) == 1
                      and certificates[0].get("status") == "pass"
                      and certificates[0].get("observed_answer") == answer)
            yield CheckResult("native:QA09:complete_window", "pass" if passed else "fail",
                "original_predicates_on_complete_native_mask_stream",
                evidence={"requested_answer": answer, "certificates": certificates})
            if passed and not common_sweep_required:
                from avengine.construction.joint import native_full_mesh_sweep
                yield from native_full_mesh_sweep(video, entries)

        request = FamilyRequest("QA-09", answer, (target["asset_id"], foreground["asset_id"]),
                                {"require_native_preflight": not geometric_only})
        def native_oracle(ctx, req, plan):
            if not cache:
                raise ConstructionUnavailable("QA09 native preparation requires its owned cache directory")
            return prepare_native(ctx, req, plan, row_factory=make_row, house=house, catalog=catalog,
                records=records, semantic=semantic, runtime=dict(runtime, phase2=True, integration_pilot=True),
                output_dir=Path(cache)/"qa09_native"/scene_id,
                family_check=lambda v, s, e: family_native_checks(v, s, e,
                    common_sweep_required=bool(plan.metadata.get("qa09_native_sweep_pending"))),
                roles={a["asset_id"]: "inside" for a in roles},
                minimum_visible_pixels={target["asset_id"]: 100, foreground["asset_id"]: 100},
                appearance_ids=(target["asset_id"],))
        result = solve(context, request, plugin,
            budget_s=float(runtime.get("qa09_geometry_budget_s", 120.))+5 if geometric_only
                     else float(runtime.get("qa09_solver_budget_s", 1200.)),
            max_candidates=32, prepare_and_verify=None if geometric_only else native_oracle,
            max_native_preparations=int(runtime.get("qa09_max_native_preparations", 8)))
        if result.status != "certified":
            import json
            raise ConstructionUnavailable("QA09 joint geometry " + result.status + ": " +
                json.dumps({"reasons": result.reasons, "diagnostics": plugin.diagnostics,
                            "solver_diagnostics": result.diagnostics}, sort_keys=True))
        if result.prepared is not None:
            payload = result.prepared.payload
            payload.row["automatic_construction"]["joint_geometry_certificate"] = asdict(result.certificate)
            return {"profile": "phase2", "row": payload.row, "prepared_native": payload}
        row = make_row(result.plan)
        row["automatic_construction"]["prototype_validation_only"] = True
        return {"profile": "phase2", "row": row}

