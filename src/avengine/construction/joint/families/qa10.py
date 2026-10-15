"""Demand-driven identity of a registered, visible foreground occluder.

Geometry and one-frame native certificates are construction evidence only.
The unmodified catalogue still derives answers from its complete native stream.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

REVISION = "qa10-registered-partial-occluder-v1"


@dataclass(frozen=True)
class OcclusionDemand:
    target_asset_id: str
    occluder_asset_id: str
    minimum_occluded_fraction: float = .25
    maximum_occluded_fraction: float = .65
    minimum_visible_pixels: int = 300
    minimum_stable_seconds: float = 1.

    def __post_init__(self):
        if (not self.target_asset_id or not self.occluder_asset_id
                or self.target_asset_id == self.occluder_asset_id):
            raise ValueError("QA-10 needs distinct registered target and occluder")
        if not (0 < self.minimum_occluded_fraction
                < self.maximum_occluded_fraction < 1):
            raise ValueError("QA-10 needs a strictly partial occlusion band")
        if self.minimum_visible_pixels < 100 or self.minimum_stable_seconds < .5:
            raise ValueError("QA-10 construction needs visible and stable evidence")

    def native_request(self):
        return {
            "target_asset_id": self.target_asset_id,
            "occluder_asset_id": self.occluder_asset_id,
            "minimum_stable_seconds": self.minimum_stable_seconds,
            "minimum_occluded_fraction": self.minimum_occluded_fraction,
            "relation": "unique registered foreground entity partially occludes the target",
        }


def certify_native_relation(witness: Mapping, demand: OcclusionDemand,
                            expected_occluder_instance_id: str):
    """Check actual modal/OBJECTS_ONLY measurements, never invent mask truth."""
    matches = [r for r in witness.get("relations", [])
               if r.get("target_asset_id") == demand.target_asset_id
               and r.get("occluder_asset_id") == demand.occluder_asset_id]
    errors = []
    if len(matches) != 1:
        return {"status": "fail", "errors": ["native_relation_missing_or_duplicated"]}
    relation = matches[0]
    metrics = relation.get("native_mask_metrics", {})
    if not relation.get("satisfies_request"):
        errors.append("native_partial_relation_not_observed")
    if relation.get("native_state") != "visible_occluded":
        errors.append("target_not_partly_visible")
    if relation.get("occluder_instance_ids") != [str(expected_occluder_instance_id)]:
        errors.append("occluder_identity_not_unique_or_wrong")
    if relation.get("unknown_occluder_pixels") != 0:
        errors.append("unknown_native_occluder_pixels")
    fraction = float(metrics.get("occlusion_fraction", math.nan))
    if not demand.minimum_occluded_fraction <= fraction <= demand.maximum_occluded_fraction:
        errors.append("occluded_fraction_outside_construction_band")
    if int(relation.get("context_pixels", 0)) < demand.minimum_visible_pixels:
        errors.append("insufficient_visible_target_pixels")
    restoration = witness.get("native_control_restoration", {})
    if any(restoration.get(k) != 0 for k in
           ("root_max_error", "joint_max_error", "camera_max_error", "link_max_error")):
        errors.append("native_control_restoration_not_exact")
    return {"status": "fail" if errors else "pass", "errors": errors,
            "occluded_fraction": fraction,
            "visible_target_pixels": int(relation.get("context_pixels", 0)),
            "occluder_instance_id": str(expected_occluder_instance_id),
            "scope": "one native static layout probe; complete stream still required"}


def full_body_in_view(camera_m, yaw_deg, root_m, *, radius_m=.40,
                      bottom_m=0., top_m=1.85, margin_deg=2.):
    """Conservative body box under the actual 85deg,16:9 camera.

    This rejects clipped construction candidates, including the old near
    foreground layout whose feet lie outside the vertical frustum.
    """
    camera = np.asarray(camera_m, float)
    root = np.asarray(root_m, float)
    if camera.shape != (3,) or root.shape != (3,) or not np.isfinite([camera, root]).all():
        raise ValueError("finite 3D camera and body root required")
    angle = math.radians(float(yaw_deg))
    forward = np.array([math.sin(angle), 0., -math.cos(angle)])
    right = np.cross(forward, [0., 1., 0.])
    horizontal = math.radians(42.5 - margin_deg)
    vertical = math.atan(math.tan(math.radians(42.5)) * 9 / 16) - math.radians(margin_deg)
    corners = np.array([[x, y, z] for x in (-radius_m, radius_m)
                        for y in (bottom_m, top_m) for z in (-radius_m, radius_m)]) + root
    delta = corners - camera
    depth = delta @ forward
    return bool(np.all(depth > 0)
                and np.all(np.abs(delta @ right) <= depth * math.tan(horizontal))
                and np.all(np.abs(delta[:, 1]) <= depth * math.tan(vertical)))


def projected_lower_body_coverage(camera_height_m, target_depth_m,
                                  occluder_depth_m, occluder_height_m):
    """Invert a floor object's shadow top edge onto the target's vertical plane.

    This geometric height is not a pixel occlusion fraction. The native
    target-only mask still has to measure the requested fraction.
    """
    values = [camera_height_m, target_depth_m, occluder_depth_m, occluder_height_m]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("positive finite dimensions required")
    if target_depth_m <= occluder_depth_m:
        raise ValueError("occluder must be in front of target")
    return camera_height_m + (occluder_height_m - camera_height_m) * target_depth_m / occluder_depth_m


def _colour(asset):
    return str(asset.get("color") or asset.get("registered_label_color")
               or asset.get("label_color") or "")


def _distinct_colours(assets):
    from avengine.qa.unified_catalog import _appearance_family
    families = [_appearance_family(_colour(a)) for a in assets]
    return bool(all(families) and len(set(families)) == len(families))


def select_roles(room, catalog, contract, state, seed):
    """Choose an answer and all referents before the geometric search."""
    from avengine.timeline.v7_family_constructors import (
        ConstructionUnavailable, eligible_humans, stable_seed,
    )
    from avengine.timeline.v7_production_inputs import room_eligible_catalog
    excluded = contract.get("policy", {}).get("temporarily_excluded_asset_ids", {})
    legal = room_eligible_catalog(catalog, room)
    occluders = [a for a in legal if a.get("kind") == "object"
                 and a.get("attachment_surface") == "floor"
                 and a.get("object_type") == "floorstanding_speaker"
                 and a["asset_id"] not in excluded
                 and _colour(a) in {"white", "black"}
                 and .90 <= a.get("geometry", {}).get("dimensions_m", [0, 0, 0])[1] <= 1.1]
    occluders.sort(key=lambda a: (_colour(a) != "white",
                                  -a["geometry"]["dimensions_m"][1],
                                  stable_seed(seed, a["asset_id"])))
    humans = eligible_humans(catalog, room, contract)
    humans.sort(key=lambda a: (state.get("source_entity_usage_counts", {}).get(a["asset_id"], 0),
                               stable_seed(seed, a["asset_id"])))
    for occluder in occluders:
        for target in humans:
            for companion in humans:
                if _distinct_colours([target, occluder, companion]):
                    return target, occluder, companion
    raise ConstructionUnavailable("QA-10 needs a legal floor occluder and two distinguishable human appearances")


def build_qa10_row(room, house, state, catalog, semantic, contract, plan,
                   certificate, target, occluder, companion, *, scene_id, seed):
    """Translate a solved fixed plan to existing native phase-two input fields."""
    from copy import deepcopy
    from dataclasses import asdict, is_dataclass
    from avengine.timeline.v7_family_constructors import choose_authored, _request
    from avengine.timeline.v7_production_inputs import validate_recipe
    if not _distinct_colours([target, occluder, companion]):
        raise ValueError("QA-10 target and answer options must have distinct colour families")
    ids = [target["asset_id"], companion["asset_id"], occluder["asset_id"]]
    if set(ids) != set(plan.actors):
        raise ValueError("QA-10 plan must retain every selected registered body")
    if not certificate.passed:
        raise ValueError("QA-10 needs a passing geometry certificate before native preparation")
    for actor in plan.actors.values():
        positions = np.asarray(actor.positions_m, float)
        if positions.shape != (901, 3) or not np.isfinite(positions).all():
            raise ValueError("QA-10 requires a complete static 30 Hz root plan")
        if not np.array_equal(positions, np.repeat(positions[:1], 901, axis=0)):
            raise ValueError("this QA-10 constructor requires stationary participants")
    actual = next(r for r in house["checked_rooms"]
                  if r.get("room_label", r.get("room_id")) == room["room_id"])
    floor = float(actual["measured_floor_y_m"])
    camera = [float(v) for v in plan.camera.position_m]
    yaw = float(plan.camera.yaw_deg)
    demand = OcclusionDemand(target["asset_id"], occluder["asset_id"])
    authored = choose_authored(semantic, [companion], state, seed)[0][0]
    speaker_events = tuple(plan.sounds)
    if (not speaker_events
            or any(e.entity_id != companion["asset_id"]
                   or e.sound_id != authored["sound_asset_id"]
                   or e.sound_class != "speech_playback" for e in speaker_events)):
        raise ValueError("QA-10 native program must retain the solver's real whole-utterance bindings")
    starts = [float(e.start_s) for e in speaker_events]
    if starts != sorted(set(starts)):
        raise ValueError("QA-10 solver event starts must be unique and ordered")
    cert = asdict(certificate) if is_dataclass(certificate) else {
        "authority": "planning_geometry_only", "passed": bool(certificate.passed)}
    row = {
        "schema": "v7-visible-registered-pilot-phase2-v1",
        "scene_id": scene_id, "independent_parent_scene_id": scene_id, "scene_seed": seed,
        "split": "train", "house": room["house"], "room_id": room["room_id"], "family": room["family"],
        "listener_m": camera, "camera_yaw_deg": yaw,
        "camera_yaw_knots": [{"time_s": t, "yaw_deg": yaw} for t in (0., 30.)],
        "asset_ids": ids, "silent_asset_ids": [target["asset_id"], occluder["asset_id"]],
        "source_classes": {companion["asset_id"]: ["speech_playback"]},
        "motion_requests": {}, "members": ["v0_a0"], "trial_only": True,
        "acceptance_allow_partial_families": True, "window_count": 5,
        "acceptance_window_starts_s": [0., 5., 10., 15., 20.],
        "questions_per_window": 14, "question_quota_base": 0,
        "semantic_scenario_id": authored["scenario_id"],
        "semantic_sound_ids": [authored["sound_asset_id"]],
        "semantic_human_asset_ids": [companion["asset_id"]],
        "semantic_start_offsets_s": [starts[0]],
        "phase2_authored_speech_schedule_s": {companion["asset_id"]: starts},
        "native_occlusion_relations": [demand.native_request()],
        "native_view_roles": {aid: {"view": "inside", "static": True,
                                   "required_visible_sometime": True} for aid in ids},
        "native_placement_preferences_m": {
            aid: np.asarray(plan.actors[aid].positions_m[0], float).tolist() for aid in ids},
        "native_attachment_supports": {},
        "owner_social_layout": {"minimum_human_center_distance_m": 1.,
                                "minimum_human_camera_distance_m": 1.5,
                                "human_orientation": "varied natural initial headings"},
        "automatic_construction": {
            "revision": REVISION, "family": "QA-10", "candidate_index": 0,
            "room_key": room["key"], "required_role_count": 3, "scene_template_files_read": [],
            "requested_branch": "visible_registered_partial", "target_asset_id": target["asset_id"],
            "occluder_asset_id": occluder["asset_id"], "planning_certificate": cert,
            "native_identity_certification_required": True,
        },
    }
    for aid in ids:
        root = np.asarray(plan.actors[aid].positions_m[0], float)
        row["native_attachment_supports"][aid] = {
            "surface": "floor", "original_surface_point_m": root.tolist(),
            "normal": [0., 1., 0.], "floor_y": floor, "_outward_clearance_m": .002,
        }
    _request(row, state, "QA-10")
    row["scene_demand"]["required_independent_family_counts"] = {"QA-10": 1}
    row["scene_demand"]["joint_static_answer"] = {
        "target_asset_id": target["asset_id"], "occluder_asset_id": occluder["asset_id"],
        "branch": "visible_registered_partial", "native_truth_required": True}
    validate_recipe(row, catalog, contract)
    return {"profile": "phase2", "row": deepcopy(row)}


class QA10Plugin:
    """Family constraints over the core's shared static geometry proposals."""
    def __init__(self, bodies, demand, *, sound_requirements=()):
        self.bodies = bodies
        self.demand = demand
        self.sound_requirements = tuple(sound_requirements)

    def candidates(self, context, request):
        from avengine.construction.joint import enumerate_static_layouts, schedule_sound_events
        from avengine.timeline.v7_family_constructors import ConstructionUnavailable
        schedule = schedule_sound_events(self.sound_requirements, start_grid_hz=10)
        if schedule.status != "solved":
            raise ConstructionUnavailable("QA-10 core sound schedule: " + str(schedule.reasons))
        roles = {aid: "inside" for aid in self.bodies}
        pair = (self.demand.target_asset_id, self.demand.occluder_asset_id,
                self.demand.minimum_occluded_fraction, self.demand.maximum_occluded_fraction)
        for plan in enumerate_static_layouts(context, self.bodies, roles, occlusion_pair=pair):
            plan.sounds = schedule.events
            plan.metadata["sound_schedule"] = {
                "status": schedule.status, "starts_s": list(schedule.starts_s),
                "diagnostics": schedule.diagnostics,
                "activity_authority": "conservative whole-admitted-file emission bounds; final activity measured independently"}
            plan.actors[self.demand.occluder_asset_id].metadata.update(
                native_entity_kind="object", motion_action="static",
                map_profile="slide_floor_envelope_only")
            plan.metadata.update(qa10_target=self.demand.target_asset_id,
                                 qa10_occluder=self.demand.occluder_asset_id,
                                 native_mask_preflight_required=True)
            yield plan

    def verify(self, context, request, plan):
        from avengine.construction.joint import CheckResult
        target = plan.actors.get(self.demand.target_asset_id)
        occluder = plan.actors.get(self.demand.occluder_asset_id)
        if target is None or occluder is None:
            yield CheckResult("qa10_registered_roles", "fail", "selected target/occluder absent")
            return
        prediction = context.predict(
            plan.camera, target.positions_m[0], target.body,
            occluders=((occluder.entity_id, occluder.positions_m[0], occluder.body),))
        passed = (prediction.state == "visible_occluded"
                  and tuple(prediction.occluder_ids) == (occluder.entity_id,)
                  and self.demand.minimum_occluded_fraction <= prediction.occluded_fraction
                      <= self.demand.maximum_occluded_fraction)
        yield CheckResult(
            "qa10_partial_registered_pair_geometry", "pass" if passed else "fail",
            "" if passed else "requested unique partial foreground relation not predicted",
            evidence={"state": prediction.state, "occluder_ids": list(prediction.occluder_ids),
                      "occlusion_fraction_proxy": prediction.occluded_fraction,
                      "native_identity_certification_required": True})


def construct_qa10_scene(room, house, state, catalog, records, semantic, contract, runtime,
                        *, scene_id, seed, family="QA-10", attempt=0):
    """Existing constructor signature; all geometry comes from joint_core."""
    from pathlib import Path
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable
    from avengine.construction.joint import (
        BodySpec, FamilyRequest, SoundRequirement, build_room_context, solve,
    )
    if family != "QA-10":
        raise ConstructionUnavailable("QA-10 plugin cannot construct another family")
    if room["house"] != house["house"] or room["room_id"] not in house["checked_room_ids"]:
        raise ConstructionUnavailable("QA-10 room/house authority differs")
    target, occluder, companion = select_roles(room, catalog, contract, state, seed)
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    bodies = {}
    for asset in (target, occluder, companion):
        vertices = np.asarray(indexed_static_mesh(asset["habitat_glb"]).vertices, float)
        radius = float(np.linalg.norm(vertices[:, [0, 2]], axis=1).max())
        bodies[asset["asset_id"]] = BodySpec(
            # Core's slide profile is its existing floor-object envelope layer.
            # Native entity kind stays object and all 901 frames stay static.
            species="slide" if asset["kind"] == "object" else "human",
            radius_m=radius, height_m=float(np.ptp(vertices[:, 1])), margin_m=.06)
    actual = next(r for r in house["checked_rooms"]
                  if r.get("room_label", r.get("room_id")) == room["room_id"])
    cache = runtime.get("joint_cache_directory")
    if not cache:
        raise ConstructionUnavailable("QA-10 needs a caller-owned joint_cache_directory")
    demand = OcclusionDemand(target["asset_id"], occluder["asset_id"])
    from avengine.timeline.v7_family_constructors import choose_authored
    authored = choose_authored(semantic, [companion], state, seed)[0][0]
    duration = math.ceil(float(authored["synthesis_readback"]["duration_s"]) * 10) / 10
    sound_requirements = tuple(SoundRequirement(
        companion["asset_id"], authored["sound_asset_id"], "speech_playback",
        ((0., duration),), (start, start), ((start, start + 9.),), wet_tail_s=.5)
        for start in (0., 10., 20.))
    request = FamilyRequest("QA-10", "visible_registered_partial", (target["asset_id"],),
                            {"occluder_asset_id": occluder["asset_id"]})
    with build_room_context(actual, house=house, runtime=runtime,
                            cache_dir=Path(cache) / "qa10", camera_heights_m=(1.1,),
                            width=1280, height=720) as context:
        solved = solve(context, request, QA10Plugin(bodies, demand, sound_requirements=sound_requirements),
                       budget_s=float(runtime.get("qa10_planning_budget_s", 120.)),
                       max_candidates=int(runtime.get("qa10_max_candidates", 256)))
    if solved.status != "certified":
        raise ConstructionUnavailable("QA-10 joint construction " + solved.status + ": " + str(solved.reasons))
    proposal = build_qa10_row(room, house, state, catalog, semantic, contract, solved.plan,
                               solved.certificate, target, occluder, companion,
                               scene_id=scene_id, seed=seed)
    proposal["row"] = certify_native_static_layout(
        proposal["row"], house, catalog, records, semantic, runtime,
        output_dir=Path(cache) / "qa10" / "native_preflight" / scene_id)
    return proposal


def certify_native_static_layout(row, house, catalog, records, semantic, runtime, *, output_dir):
    """Real static layout/identity/colour preflight, with no audio/video render.

    Exact native support sites are returned to the ordinary producer. It still
    rechecks placement and independently measures all 450 native frames.
    """
    from copy import deepcopy
    from pathlib import Path
    from itertools import combinations
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable
    from avengine.timeline.v7_family_clock import build_constructed_scene
    from avengine.capture.v7_phase2_native import NativePhase2Video
    from avengine.capture.v7_occlusion_demand import measure_requested_relations
    from avengine.timeline.continuous_pilot import write_json
    from PIL import Image

    result = deepcopy(row)
    if row.get("motion_requests"):
        raise ConstructionUnavailable("QA-10 static preflight cannot certify moving bodies")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    working, scene, _program, assets = build_constructed_scene(row, house, records, catalog, semantic)
    preflight_runtime = deepcopy(runtime)
    # Preparation may solve a nearby real support before it is certified.
    # Formal rendering receives the accepted exact site and uses one proposal.
    preflight_runtime["visible_placement_max_proposals_per_asset"] = int(
        runtime.get("qa10_preflight_support_proposal_limit", 121))
    video = NativePhase2Video(
        working, scene, preflight_runtime, assets,
        source_material_cache_directory=runtime["source_material_cache_directory"])
    try:
        checks = video.qualify_standing(scene, output_dir / "standing")
        witness = measure_requested_relations(video, scene, video.actors)
        byasset = {a["asset_id"]: a for a in assets}
        certificates = []
        for relation in row["native_occlusion_relations"]:
            demand = OcclusionDemand(
                relation["target_asset_id"], relation["occluder_asset_id"],
                minimum_occluded_fraction=relation["minimum_occluded_fraction"],
                minimum_stable_seconds=relation["minimum_stable_seconds"])
            oid = byasset[demand.occluder_asset_id]["native_instance_id"]
            certificates.append(certify_native_relation(witness, demand, str(oid)))

        video._native_pair_pixels(scene, video.actors)
        obs = {k: np.asarray(v).copy() for k, v in video.sim.get_sensor_observations().items()}
        video._check_source_frame(obs["rgb"][:, :, :3], obs["semantic"], 0)
        colour = {c["asset_id"]: c for c in video.color_checks}
        errors = [error for c in certificates for error in c["errors"]]
        for asset in assets:
            if not colour.get(asset["asset_id"], {}).get("matched"):
                errors.append("native_appearance_label_unreviewable:" + asset["asset_id"])

        sensor = video.sim._sensors["rgb"]._sensor_object
        projection = np.asarray(sensor.render_camera.projection_matrix, float)
        world_to_camera = np.linalg.inv(np.asarray(sensor.node.absolute_transformation(), float))
        body_projection = []
        for entry in video.actors:
            asset = entry["declaration"]
            transform = np.asarray(entry["applied_root_input"], float)
            local = np.asarray(entry["original_mesh"].vertices, float)
            vertices = local @ transform[:3, :3].T + transform[:3, 3]
            clip = np.column_stack([vertices, np.ones(len(vertices))]) @ world_to_camera.T @ projection.T
            forward = clip[:, 3]
            visible = bool(np.all(forward > 0))
            xy = clip[:, :2] / np.maximum(forward[:, None], 1e-12)
            margins = (1 - np.abs(xy).max(axis=0)) * np.array([video.width, video.height]) / 2
            in_view = bool(visible and np.min(margins) >= 8.)
            body_projection.append({
                "asset_id": asset["asset_id"], "full_mesh_vertices": len(vertices),
                "entire_body_inside_native_image": in_view,
                "minimum_horizontal_margin_px": float(margins[0]),
                "minimum_vertical_margin_px": float(margins[1]),
            })
            if not in_view:
                errors.append("native_full_body_clipped:" + asset["asset_id"])
        people = [e for e in video.actors if e["declaration"]["kind"] in {"human", "performer"}]
        roots = {e["declaration"]["asset_id"]: np.asarray(e["applied_root_input"], float)[:3, 3]
                 for e in video.actors}
        camera = np.asarray(scene["listener_m"], float)
        for entry in people:
            aid = entry["declaration"]["asset_id"]
            if np.linalg.norm((roots[aid] - camera)[[0, 2]]) < 1.5 - 1e-6:
                errors.append("native_person_camera_distance:" + aid)
        for first, second in combinations(people, 2):
            a, b = first["declaration"]["asset_id"], second["declaration"]["asset_id"]
            if np.linalg.norm((roots[a] - roots[b])[[0, 2]]) < 1. - 1e-6:
                errors.append("native_people_spacing:" + a + ":" + b)
        certificate = {
            "status": "fail" if errors else "pass", "errors": errors,
            "authority": "actual_native_static_layout_preflight",
            "one_frame_only": True, "complete_450_frame_stream_still_required": True,
            "relations": certificates, "full_body_projection": body_projection,
            "actual_roots_m": {aid: point.tolist() for aid, point in roots.items()},
            "actual_yaw_rad": {e["declaration"]["asset_id"]: float(e["resolved_yaw_rad"])
                               for e in video.actors},
            "native_colour_checks": video.color_checks,
            "native_control_restoration": witness.get("native_control_restoration"),
            "qa_predicates_truth_scorer_changed": False,
        }
        write_json(output_dir / "certificate.json", certificate)
        write_json(output_dir / "native_occlusion_witness.json", witness)
        Image.fromarray(obs["rgb"][:, :, :3]).save(output_dir / "preview.png")
        if errors:
            raise ConstructionUnavailable("QA-10 native preflight: " + ";".join(errors))
        for check in checks:
            aid = check["asset_id"]
            support = check["placement"]["support"]
            result["native_attachment_supports"][aid] = {
                key: deepcopy(support[key]) for key in
                ("surface", "original_surface_point_m", "normal", "floor_y",
                 "_outward_clearance_m", "_body_yaw_retry") if key in support}
            result["native_placement_preferences_m"][aid] = roots[aid].tolist()
        result["automatic_construction"]["native_preflight"] = {
            "certificate_path": str(output_dir / "certificate.json"),
            "actual_roots_m": certificate["actual_roots_m"],
            "actual_yaw_rad": certificate["actual_yaw_rad"],
            "formal_render_support_proposal_limit": 1,
            "status": "pass", "expected_occluder_asset_ids":
                [r["occluder_asset_id"] for r in row["native_occlusion_relations"]],
            "scope": "actual native static layout; final stream remains independent",
        }
        write_json(output_dir / "certified_row.json", result)
        return result
    finally:
        video.close()
