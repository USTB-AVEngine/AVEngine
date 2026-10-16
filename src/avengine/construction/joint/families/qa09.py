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
    minimum_hidden_s: float = 1.0 / 15
    minimum_visible_s: float = 1.0 / 15
    minimum_visible_pixels: int = 1
    minimum_target_pixels: int = 1

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

    def relation(self, query_times_s=None) -> dict[str, Any]:
        """Serialize the request for the native adapter; this is never truth."""
        return {
            "target_asset_id": self.target_asset_id,
            "occluder_asset_id": self.occluder_asset_id,
            "required_state": "fully_occluded",
            "minimum_occluded_fraction": 1.0,
            "minimum_stable_seconds": self.minimum_hidden_s,
            # Existing placement scoring needs a target root during the pause.
            # The complete-window probe below supplies the real certificate.
            "query_times_s": list(query_times_s or [self.window_start_s + self.window_duration_s / 2]),
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
    the measured branch. The default accepts their frame-level existence witness;
    complete coverage and actual registered identity remain explicit evidence.
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
    turn_sign = float(runtime.get("qa09_arc_turn_sign", 1))
    if turn_sign not in (-1., 1.):
        raise ValueError("QA-09 arc turn sign must be -1 or 1")
    turn_sign = int(turn_sign)
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
    if answer == "yes" and route_kind == "arc" and turn_sign != 1:
        key_data.update(revision="qa09-arc-gait-4", turn_radius_m=arc_radius, turn_sign=turn_sign)
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
        corner = (anchor, arc_radius, turn_sign*(float(distance[-1])-anchor)/arc_radius) if answer == "yes" else None
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
              and p.visible_pixels_upper == 0
              and p.occluder_ids == (foreground_id,)
              and p.evidence.get("unknown_occluder_pixels") == 0]
    clear = [i for i, p in enumerate(predictions) if p.state in {"visible_clear", "visible_occluded"}
             and p.visible_pixels_upper >= demand.minimum_visible_pixels]
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
    if any(p.state == "unknown" for p in predictions):
        errors.append("unknown_visual_geometry")
    return CheckResult("QA09:temporal", "fail" if errors else "pass", ";".join(errors), evidence={
        "predicted_answer": predicted, "hidden_intervals_frames": hr, "visible_intervals_frames": vr,
        "prediction_frames": len(predictions), "native_truth": False,
        "pose_floor_assumption": "flat planning floor; actual native floor and masks must requalify",
        "required_native_visible_pixels": demand.minimum_visible_pixels,
        "measured_minimum_depth_margin_m": min((p.depth_margin_m for p in predictions),default=None),
    })


def endpoint_event_route(context, start, hidden, end, *, entity_id, body, answer,
                         window_start_s=0., window_duration_s=10., camera=None):
    """Declare A/H/B obligations; the shared core owns paths, rounding and time."""
    import numpy as np
    from avengine.construction.joint import route_between, construct_route, ConstructionError
    a = route_between(context, start, hidden, species=body.species)
    if answer == "yes" or not np.allclose(hidden, end):
        b = route_between(context, hidden, end, species=body.species)
        points = np.concatenate((a, b[1:]))
        hidden_index = len(a)-1
        legs = 2
    else:
        points, hidden_index, legs = a, len(a)-1, 1
    lengths = np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1)
    total = float(lengths.sum())
    # The existing core starts/stops with .3s ramps. Remaining time is assigned
    # to initial visibility, the hidden waypoint and final visibility/rest.
    frame = 1/15
    available = window_duration_s - legs*.3 - (legs+1)*frame
    if available <= 0 or total > 1.4*available:
        raise ConstructionError("qa09_route_exceeds_answer_window")
    from avengine.timeline.v7_motion_speed import SPEED_RANGES
    lower, upper = SPEED_RANGES[body.species]
    feasible_lower = max(lower, total/max(available, frame))
    # Maximise clearance to the existing speed limits. Choosing the lower
    # boundary made float32 native readback fall below .8 by 16 micrometres/s.
    # This constructs an interior solution; no admission threshold is changed.
    speed = float((feasible_lower + upper)/2)
    spare = window_duration_s-total/speed-legs*.3
    dwell = np.zeros(len(points))
    initial = spare/(legs+1)
    if legs == 2:
        dwell[hidden_index] = initial
    # Radius follows actual waypoints, without a fixed preferred turn or angle.
    radius = float(lengths.min()/2) if len(lengths) else 0.
    actor = construct_route(context, points, species=body.species, speed_m_s=speed,
        start_s=window_start_s+initial, dwell_s=dwell, corner_radius_m=radius,
        camera=camera, entity_id=entity_id, body=body)
    intervals = actor.metadata["walk_intervals_s"]
    hidden_start = intervals[0][1]
    hidden_end = intervals[1][0] if legs == 2 else window_start_s+window_duration_s
    actor.metadata.update(native_yaw_rad=float(-actor.headings_rad[0]-math.pi/2),
        native_yaw_rad_per_frame=(-actor.headings_rad-math.pi/2).tolist(),
        qa09_hidden_interval_s=[hidden_start, hidden_end],
        qa09_event_window_s=[window_start_s, window_start_s+window_duration_s],
        qa09_answer=answer, qa09_hidden_root_m=list(map(float, hidden)))
    return actor


class EndpointPoseSampler:
    """Actual admitted skin/IK on the declared planning floor; native requalifies."""
    def __init__(self, asset, actor, runtime, house):
        import numpy as np
        from avengine.assets.roster_motion.episode import admitted_master, endpoint_route_from_plan
        from avengine.assets.roster_motion.contact import FootPlant
        from avengine.construction.joint import admitted_human_clock_source
        self.document, self.binary, action, self.lineage = admitted_master(
            {"declaration":asset, "prepared_glb":asset["habitat_glb"]}, runtime, house)
        source = admitted_human_clock_source(asset, runtime, house)
        phase = min(source["double_support_s"], default=float(source["source_times_s"][0]))
        route = endpoint_route_from_plan({"positions_m": actor.positions_m,
            "headings_rad": actor.headings_rad, "metadata": actor.metadata}, source_phase_s=phase)
        ground = float(actor.positions_m[0,1])
        class Floor:
            original_ray = True
            def __call__(self, point): return ground
            def many(self, points): return np.full(len(points), ground)
        ts=source["source_times_s"]
        self.solver=FootPlant(self.document,self.binary,action,route.clock.tolist(),
            actor.positions_m.tolist(),route.yaws.tolist(),float(actor.metadata["speed_m_s"]),
            float(ts[0]),float(ts[-1]-ts[0]),Floor())
        self.solver.times=np.arange(901)/30
        self.cache={}
        self.minimum_skin_y=float("inf")
    def ray(self,index):
        import numpy as np
        import trimesh
        from avengine.assets.roster_motion.episode import triangle_mesh
        from avengine.construction.joint import MeshRaycaster
        if index not in self.cache:
            _,parts,_=self.solver.frame(index)
            v,f=triangle_mesh(self.document,self.binary,parts)
            self.minimum_skin_y=min(self.minimum_skin_y,float(np.asarray(v)[:,1].min()))
            self.cache[index]=MeshRaycaster(trimesh.Trimesh(v,f,process=False),
                evidence={"pose_source":self.lineage,"floor_assumption":"flat planning floor; native room requalification required"})
        return self.cache[index]


class QA09Plugin:
    """Endpoint roles and exact registered shadows with a jointly solved camera."""
    def __init__(self, target, foreground, second, runtime, house, *, seed=0,
                 geometry_budget_s=120., sounds=()):
        import numpy as np
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        from avengine.construction.joint import MeshRaycaster, BodySpec
        self.target,self.foreground,self.second=target,foreground,second
        self.runtime,self.house,self.seed=runtime,house,seed
        self.geometry_budget_s,self.sounds=geometry_budget_s,tuple(sounds)
        self.assets={a["asset_id"]:a for a in (target,foreground,second) if a is not None}
        self.meshes={k:indexed_static_mesh(a["habitat_glb"]) for k,a in self.assets.items()}
        self.rays={k:MeshRaycaster(m,evidence={"registered_asset_id":k}) for k,m in self.meshes.items()}
        self.bodies={k:BodySpec("human",float(np.linalg.norm(np.asarray(m.vertices)[:,[0,2]],axis=1).max()),
                               float(np.asarray(m.vertices)[:,1].max()),0.) for k,m in self.meshes.items()}
        self.diagnostics={"candidate_pairs":0,"paths_solved":0,"geometry_certified":0,
                          "native_rendered_pass":0,"rejections":{},"multi_event_attempts":0}
        self.samplers={}
        self._seen_plans=set()
    def reject(self,reason):
        d=self.diagnostics["rejections"];d[reason]=d.get(reason,0)+1
    def _predict(self,context,plan,demand,samplers,deadline=None):
        import time
        from avengine.construction.joint import predict_mesh_visibility
        depth=context.depth(plan.camera)
        predictions=[]
        for i in range(demand.frame_count):
            if deadline is not None and time.monotonic()>=deadline:return None
            frame=round((demand.window_start_s+i/demand.frame_rate_hz)*30)
            instances={}
            for aid,actor in plan.actors.items():
                ray=samplers[aid].ray(frame) if aid in samplers else self.rays[aid]
                yaw=(float(-actor.headings_rad[frame]-math.pi/2) if actor.headings_rad is not None
                     else actor.metadata["native_yaw_rad"])
                instances[aid]=_mesh_instance(aid,ray,actor.positions_m[frame],yaw)
            predictions.append(predict_mesh_visibility(plan.camera,instances.pop(demand.target_asset_id),
                scene_depth=depth,occluders=tuple(instances.values()),depth_margin_m=0.))
        return predictions
    def _events(self,context,camera,foreground_actor,target,answer,offset,*,existing=None,deadline=None):
        import time
        import numpy as np
        from avengine.construction.joint import ScenePlan,certify_plan,ConstructionError
        from avengine.construction.joint.routes import minimum_track_distance
        from avengine.construction.joint.camera_solve import PointRole,RouteRole
        aid,fid=target["asset_id"],self.foreground["asset_id"]
        body=self.bodies[aid]
        fore=_mesh_instance(fid,self.rays[fid],foreground_actor.positions_m[0],
                            foreground_actor.metadata["native_yaw_rad"])
        shadow_context=context.with_occluders(fore)
        grid=shadow_context.grid(camera,body)
        points=context.points_m
        legal=context.walkable_map.contains_many(points[:,[0,2]],"human") & (grid.distances_m>=1.5)
        legal &= np.linalg.norm((points-foreground_actor.positions_m[0])[:,[0,2]],axis=1)>=1.
        inside=np.flatnonzero(legal & (grid.frustum_state=="inside"))
        rests=np.flatnonzero(legal & (grid.frustum_state!="edge"))
        self.diagnostics["available_endpoint_pairs"]=self.diagnostics.get("available_endpoint_pairs",0)+len(inside)*len(rests)
        # Envelope shadow is only ranking. A human is not a filled cylinder;
        # only the original posed mesh and the complete native stream certify.
        hidden=sorted(inside,key=lambda i:(float(np.nan_to_num(grid.visible_fraction[i],nan=1.)),
                                           -int(grid.shadow_mask[i]),int(i)))
        for h in hidden:
            if deadline is not None and time.monotonic()>=deadline:return
            starts=sorted((int(i) for i in inside if i!=h),key=lambda i:(
                -float(np.nan_to_num(grid.visible_fraction[i],nan=0.)),
                float(np.linalg.norm((points[i]-points[h])[[0,2]])),i))
            ends=[h] if answer=="no" else starts
            for a in starts:
                # A no branch may end in the shadow, or leave the frame after it.
                choices=ends if answer=="yes" else [h]+[int(i) for i in rests if grid.frustum_state[i]=="outside"]
                for b in choices:
                    if b==a:continue
                    if deadline is not None and time.monotonic()>=deadline:return
                    self.diagnostics["candidate_pairs"]+=1
                    try:
                        actor=endpoint_event_route(context,points[a],points[h],points[b],entity_id=aid,
                            body=body,answer=answer,window_start_s=offset,camera=camera)
                    except ConstructionError as exc:self.reject("route:"+exc.reason);continue
                    self.diagnostics["paths_solved"]+=1
                    if minimum_track_distance(actor.positions_m,foreground_actor.positions_m)<1.:
                        self.reject("owner_foreground_separation");continue
                    if existing and any(minimum_track_distance(actor.positions_m,v.positions_m)<1.
                                        for k,v in existing.actors.items() if k!=fid):
                        self.reject("owner_other_target_separation");continue
                    actors={fid:foreground_actor,aid:actor}
                    if existing:actors.update(existing.actors)
                    roles=[PointRole("inside_full_body",tuple(foreground_actor.positions_m[0]),self.bodies[fid],"occluder")]
                    for key,entry in actors.items():
                        if key==fid:continue
                        event_answer=entry.metadata["qa09_answer"]
                        roles.append(RouteRole(tuple(map(tuple,entry.metadata["waypoints_m"])),
                            "inside_full_body","inside_full_body" if event_answer=="yes" else "rest_full_body",
                            entry.body,name=key))
                        roles.append(PointRole("inside_full_body",tuple(entry.metadata["qa09_hidden_root_m"]),entry.body,key+".hidden"))
                    # Re-solve position, height and continuous horizontal yaw for
                    # all endpoints. Keep this shadow's site first; the full camera
                    # set is still searched by the outer generator.
                    solved=context.solve_cameras(roles)
                    cameras=sorted(solved.cameras,key=lambda c:(sum((x-y)**2 for x,y in zip(c.position_m,camera.position_m)),
                                                              abs(c.yaw_deg-camera.yaw_deg)))
                    self.diagnostics["camera_solve"]=solved.diagnostics
                    if existing:
                        cameras=[c for c in cameras if np.allclose(c.position_m,camera.position_m)]
                    demand=ReappearanceDemand(aid,fid,answer,window_start_s=offset)
                    demands=list(existing.metadata["qa09_demands"]) if existing else []
                    demands.append(asdict(demand))
                    sampler=EndpointPoseSampler(target,actor,self.runtime,self.house)
                    samplers=dict(self.samplers) if existing else {}
                    samplers[aid]=sampler
                    for fixed in cameras:
                        if deadline is not None and time.monotonic()>=deadline:return
                        plan=ScenePlan(fixed,actors,sounds=self.sounds,metadata={"qa09_demands":demands,
                            "body_yaws_rad":{k:v.metadata["native_yaw_rad"] for k,v in actors.items()},
                            "allow_native_map_fallback":context.allow_native_map_fallback,
                            "construction":"endpoint_shadow_camera_roles_v2"})
                        import hashlib
                        identity=hashlib.sha256(repr((fixed,demands,plan.metadata["body_yaws_rad"])).encode())
                        for key in sorted(actors):
                            identity.update(key.encode());identity.update(actors[key].positions_m.tobytes())
                            if actors[key].headings_rad is not None:identity.update(actors[key].headings_rad.tobytes())
                        key=identity.hexdigest()
                        if key in self._seen_plans:
                            self.reject("duplicate_solved_plan");continue
                        self._seen_plans.add(key)
                        common=certify_plan(context,plan)
                        if not planning_admissible(common,native_sweep_pending=True):
                            for c in common.checks:
                                if c.status not in {"pass","unknown"}:self.reject(c.name+":"+c.reason)
                            continue
                        checks=[];states={}
                        for raw in demands:
                            obligation=ReappearanceDemand(**raw)
                            prediction=self._predict(context,plan,obligation,samplers,deadline)
                            if prediction is None:
                                self.reject("prediction_computation_budget");return
                            check=prediction_certificate(prediction,obligation,fid)
                            checks.append(asdict(check));states[obligation.target_asset_id]=[p.state for p in prediction]
                            if check.status!="pass":self.reject(check.reason);break
                        if len(checks)!=len(demands) or any(c["status"]!="pass" for c in checks):continue
                        plan.metadata.update(qa09_prediction_checks=checks,qa09_prediction_states=states,
                            qa09_native_sweep_pending=any(c.status=="unknown" for c in common.checks),
                            contact_quality_observation={k:{"minimum_planning_skin_y_m":v.minimum_skin_y,
                                "admission_effect":"record_only","authority":"PHASE2 A"} for k,v in samplers.items()})
                        self.samplers=samplers
                        self.diagnostics["geometry_certified"]+=1
                        yield plan
    def candidates(self,context,request):
        import time
        import numpy as np
        from avengine.construction.joint import static_actor
        from avengine.construction.joint.camera_solve import PointRole
        from avengine.timeline.v7_family_constructors import stable_seed
        started=time.monotonic();deadline=started+self.geometry_budget_s
        fid=self.foreground["asset_id"]
        points=context.points_m
        legal=context.walkable_map.contains_many(points[:,[0,2]],"human")
        # Full domain, deterministic spacing rank, no camera/foreground distance ring.
        order=np.flatnonzero(legal)
        center=np.mean(points[order],axis=0) if len(order) else np.zeros(3)
        order=sorted(order,key=lambda i:(float(np.linalg.norm(points[i]-center)),int(i)))
        for i in order:
            if time.monotonic()>=deadline:return
            fore=static_actor(fid,points[i],self.bodies[fid])
            fore.metadata["native_yaw_rad"]=math.radians(stable_seed(self.seed,"foreground_heading",int(i))%360)
            cameras=context.solve_cameras((PointRole("inside_full_body",tuple(points[i]),self.bodies[fid],fid),))
            self.diagnostics["initial_camera_domain"]=cameras.diagnostics
            for solution in cameras.solutions:
                if time.monotonic()>=deadline:return
                for one in self._events(context,solution.camera,fore,self.target,request.branch,0.,deadline=deadline):
                    # Try the independent opposite branch in the last crop. If it
                    # cannot fit, preserve the already certified first event.
                    if self.second is not None:
                        self.diagnostics["multi_event_attempts"]+=1
                        other="no" if request.branch=="yes" else "yes"
                        second_deadline=min(deadline,time.monotonic()+max(0.,deadline-time.monotonic())/2)
                        paired=next(self._events(context,one.camera,fore,self.second,other,20.,existing=one,
                                                 deadline=second_deadline),None)
                        if paired is not None:
                            emitted=time.monotonic();yield paired
                            pause=time.monotonic()-emitted;deadline+=pause
                            one.metadata["multi_event_downgrade"]="paired_native_candidate_failed; retain single event"
                        else:one.metadata["multi_event_downgrade"]="no jointly certified second event in computation budget"
                    emitted=time.monotonic();yield one
                    pause=time.monotonic()-emitted;deadline+=pause
    def verify(self,context,request,plan):
        from avengine.construction.joint import CheckResult
        for i,value in enumerate(plan.metadata.get("qa09_prediction_checks",[])):
            yield CheckResult(**dict(value,name="QA09:temporal:"+str(i)))


def build_qa09_row(room,house,state,catalog,contract,plan,certificate,roles,authored,
                   walkable_map_path,*,scene_id,seed,native_sweep_pending=False):
    from copy import deepcopy
    from avengine.timeline.v7_family_constructors import _request
    from avengine.timeline.v7_production_inputs import validate_recipe
    from avengine.construction.joint import strict_native_bindings
    from avengine.assets.roster_motion.navmesh_execution import REVISION
    if not planning_admissible(certificate,native_sweep_pending=native_sweep_pending):
        raise ValueError("QA09 requires its planning certificate")
    assets=[a for a in roles if a["asset_id"] in plan.actors]
    ids=[a["asset_id"] for a in assets]
    demands=[ReappearanceDemand(**r) for r in plan.metadata["qa09_demands"]]
    speech_id=demands[0].occluder_asset_id
    motions={};relations=[]
    for demand in demands:
        actor=plan.actors[demand.target_asset_id]
        endpoint={"positions_m":actor.positions_m.tolist(),"headings_rad":actor.headings_rad.tolist(),
                  "metadata":deepcopy(actor.metadata)}
        motions[demand.target_asset_id]={"action":"walk","speed_m_s":actor.metadata["speed_m_s"],
            "intervals_s":deepcopy(actor.metadata["walk_intervals_s"]),
            "minimum_listener_horizontal_distance_m":1.5,"scene_speed_range_m_s":[.8,1.4],
            "motion_execution_revision":REVISION,"joint_endpoint_route":endpoint,
            # Compatibility marker preserves solved speed in the existing shared
            # prepare_scene_motion adapter. gate_policy consumes endpoint first.
            "joint_qa09_route":{"execution_api":"joint_endpoint_route"}}
        states=plan.metadata["qa09_prediction_states"][demand.target_asset_id]
        query=[demand.window_start_s+i/demand.frame_rate_hz for i,v in enumerate(states) if v=="fully_occluded"]
        relations.append(demand.relation(query))
    row={"schema":"v7-visible-registered-pilot-phase2-v1","scene_id":scene_id,
        "independent_parent_scene_id":scene_id,"scene_seed":seed,"house":room["house"],
        "room_id":room["room_id"],"split":"train","family":room["family"],
        "listener_m":list(plan.camera.position_m),"camera_yaw_deg":float(plan.camera.yaw_deg),
        "camera_yaw_knots":[{"time_s":t,"yaw_deg":float(plan.camera.yaw_deg)} for t in (0.,30.)],
        "asset_ids":ids,"silent_asset_ids":[i for i in ids if i!=speech_id],
        "source_classes":{speech_id:["speech_playback"]},
        "semantic_scenario_id":authored["scenario_id"],"semantic_sound_ids":[authored["sound_asset_id"]],
        "semantic_human_asset_ids":[speech_id],"semantic_start_offsets_s":[0.],
        "phase2_authored_speech_schedule_s":{speech_id:[0.,10.,20.]},
        "motion_requests":motions,"motion_execution_revision":REVISION,
        "native_view_roles":{aid:{"view":"inside","static":aid not in motions,
                                  "required_visible_sometime":True} for aid in ids},
        "native_dynamic_occlusion_relations":relations,"members":["v0_a0"],
        "trial_only":True,"acceptance_allow_partial_families":True,
        "window_count":5,"questions_per_window":14,"question_quota_base":0,
        "acceptance_window_starts_s":[0.,5.,10.,15.,20.],
        "automatic_construction":{"revision":"joint-qa09-endpoint-camera-v2","family":"QA-09",
            "constructor":"construct_qa09_scene","candidate_index":0,"room_key":room["key"],
            "required_role_count":len(ids),"scene_template_files_read":[],
            "requested_answer":demands[0].answer,"requested_events":[asdict(d) for d in demands],
            "joint_geometry_certificate":asdict(certificate),"native_preflight_required":True,
            "predicted_event_states":plan.metadata["qa09_prediction_states"],
            "multi_event_downgrade":plan.metadata.get("multi_event_downgrade"),
            "contact_quality_observation":plan.metadata.get("contact_quality_observation",{})},
        "owner_social_layout":{"minimum_human_center_distance_m":1.,
            "minimum_human_camera_distance_m":1.5,
            "human_orientation":"seed-selected foreground; route tangents for walkers",
            "distribution":"independent endpoint events with registered occluder"}}
    row.update(strict_native_bindings(plan,assets))
    _request(row,state,"QA-09")
    row["scene_demand"]["requested_answer"]=demands[0].answer
    row["scene_demand"]["requested_conditions"]=sorted(set(c for d in demands for c in d.requested_conditions()))
    validate_recipe(row,catalog,contract)
    return {"profile":"phase2","row":row}


def construct_qa09_scene(room,house,state,catalog,records,semantic,contract,runtime,
                         *,scene_id,seed,family="QA-09",attempt=0):
    """Joint endpoints, registered occluder and fixed camera, with two event roles."""
    import os,sys,json
    from pathlib import Path
    from avengine.construction.joint import (build_room_context,solve,FamilyRequest,SoundEvent,
        Certificate,CheckResult,certify_plan,prepare_native)
    from avengine.rooms.walkable_map import room_map_path
    from avengine.timeline.v7_family_constructors import (room_context,choose_authored,
        ConstructionUnavailable,eligible_humans,stable_seed)
    if family!="QA-09" or room["family"] not in {"hm3d","mp3d"}:
        raise ConstructionUnavailable("QA09 requires its registered Habitat family")
    geometric_only=bool(runtime.get("qa09_geometric_only",False))
    target,foreground=select_roles(room,catalog,contract,state,seed)
    # Independent physical identities preserve event diversity under original dedup.
    pool=[a for a in eligible_humans(catalog,room,contract) if a.get("walk_allowed")
          and a["asset_id"] not in {target["asset_id"],foreground["asset_id"]}]
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    pool.sort(key=lambda a:(a["color"] in {target["color"],foreground["color"]},
                            a["body_identity_id"] in {target["body_identity_id"],foreground["body_identity_id"]},
                            float(indexed_static_mesh(a["habitat_glb"]).vertices[:,1].max()),
                            stable_seed(seed,"second_target",a["asset_id"])))
    second=pool[0] if pool else None
    roles=[target,foreground]+([second] if second is not None else [])
    answer=str(runtime.get("qa09_requested_answer") or requested_answer(state,seed))
    authored=choose_authored(semantic,[foreground],state,seed,attempt=attempt)[0][0]
    duration=float(authored["synthesis_readback"]["duration_s"])
    sounds=tuple(SoundEvent(foreground["asset_id"],authored["sound_asset_id"],"speech_playback",t,t+duration)
                 for t in (0.,10.,20.))
    actual=dict(room_context(room,house)["original_room"])
    map_root=runtime.get("walkable_map_root") or os.environ.get("AVENGINE_WALKABLE_MAP_ROOT")
    if not map_root:raise ConstructionUnavailable("QA09 needs the admitted walkable-map root")
    path=room_map_path(map_root,room["room_id"]);actual["walkable_map_path"]=str(path)
    plugin=QA09Plugin(target,foreground,second,runtime,house,seed=seed,sounds=sounds,
                     geometry_budget_s=float(runtime.get("qa09_geometry_budget_s",600.)))
    cache=runtime.get("joint_cache_directory")
    evidence=runtime.get("qa09_evidence_directory")
    def write_diagnostics(extra=None):
        if evidence:
            destination=Path(evidence);destination.mkdir(parents=True,exist_ok=True)
            (destination/(scene_id+"_funnel.json")).write_text(json.dumps(
                {"scene_id":scene_id,"house":room["house"],"diagnostics":plugin.diagnostics,
                 "result":extra,"formal_questions":0},indent=2,default=str)+"\n")
    # camera_limit affects only the legacy context.cameras list. This plugin's
    # domain is camera_solve's full legal floor grid and existing height set.
    with build_room_context(actual,house=house,runtime=runtime,
            cache_dir=Path(cache)/room["room_id"] if cache else None,
            camera_heights_m=runtime.get("joint_camera_heights_m"),width=1280,height=720,
            allow_native_map_fallback=bool(runtime.get("qa09_allow_native_map_fallback",False))) as context:
        request=FamilyRequest("QA-09",answer,(target["asset_id"],foreground["asset_id"]),
                              {"require_native_preflight":not geometric_only})
        def make_row(plan):
            cert=Certificate(certify_plan(context,plan).checks+tuple(plugin.verify(context,request,plan)))
            result=build_qa09_row(room,house,state,catalog,contract,plan,cert,roles,authored,path,
                scene_id=scene_id,seed=seed,native_sweep_pending=not geometric_only)
            from avengine.assets.roster_motion.navmesh_execution import prepare_scene_motion
            return prepare_scene_motion(result["row"],catalog)
        def native_oracle(ctx,req,plan):
            if not cache:raise ConstructionUnavailable("QA09 requires its owned cache directory")
            expected=[ReappearanceDemand(**v) for v in plan.metadata["qa09_demands"]]
            def family_checks(video,scene,entries):
                report=getattr(video,"family_native_preflights",{}).get("QA-09",{})
                for demand in expected:
                    certificates=[r.get("native_complete_window_certificate",{}) for r in report.get("relations",[])
                                  if r.get("target_asset_id")==demand.target_asset_id]
                    passed=(len(certificates)==1 and certificates[0].get("status")=="pass"
                            and certificates[0].get("observed_answer")==demand.answer)
                    yield CheckResult("native:QA09:"+demand.target_asset_id,"pass" if passed else "fail",
                        "original_predicates_on_complete_native_mask_stream",
                        evidence={"demand":asdict(demand),"certificates":certificates})
                if not plan.metadata.get("qa09_native_sweep_pending"):
                    from avengine.construction.joint import native_full_mesh_sweep
                    yield from native_full_mesh_sweep(video,entries)
            write_diagnostics({"stage":"before_native_preparation","events":len(expected)})
            return prepare_native(ctx,req,plan,row_factory=make_row,house=house,catalog=catalog,
                records=records,semantic=semantic,runtime=dict(runtime,phase2=True,integration_pilot=True),
                output_dir=Path(cache)/"qa09_native"/scene_id,family_check=family_checks,
                roles={aid:"inside" for aid in plan.actors},
                appearance_ids=tuple(d.target_asset_id for d in expected))
        result=solve(context,request,plugin,
            budget_s=float(runtime.get("qa09_geometry_budget_s",600.))+5 if geometric_only
                else float(runtime.get("qa09_solver_budget_s",1800.)),
            max_candidates=sys.maxsize,prepare_and_verify=None if geometric_only else native_oracle,
            max_native_preparations=int(runtime.get("qa09_max_native_preparations",8)))
        write_diagnostics({"status":result.status,"solver":result.diagnostics,"reasons":result.reasons})
        if result.status!="certified":
            raise ConstructionUnavailable("QA09 joint endpoints "+result.status+": "+json.dumps(
                {"reasons":result.reasons,"diagnostics":plugin.diagnostics,"solver_diagnostics":result.diagnostics},sort_keys=True))
        if result.prepared is not None:
            payload=result.prepared.payload
            payload.row["joint_prepared_scene_required"]=True
            payload.row["automatic_construction"]["joint_geometry_certificate"]=asdict(result.certificate)
            return {"profile":"phase2","row":payload.row,"prepared_native":payload}
        row=make_row(result.plan);row["automatic_construction"]["prototype_validation_only"]=True
        return {"profile":"phase2","row":row}
