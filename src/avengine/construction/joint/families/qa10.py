"""Demand-driven identity of a registered, visible foreground occluder.

Geometry and one-frame native certificates are construction evidence only.
The unmodified catalogue still derives answers from its complete native stream.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

REVISION = "qa10-registered-partial-occluder-v2-joint-native"


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
        headings = getattr(actor, "headings_rad", None)
        if headings is not None and not np.all(np.asarray(headings) == headings[0]):
            raise ValueError("QA-10 static proof requires constant planned headings")
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
    starts = [float(t) for t in plan.metadata.get("source_playback_starts_s", {}).get(
        companion["asset_id"], [])]
    if not starts:
        raise ValueError("QA-10 solver must retain whole-file playback starts separately from measured activity")
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
        "joint_prepared_scene_required": True,
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
    bindings = deepcopy(plan.metadata.get("native_bindings", {}))
    pins = bindings.get("native_attachment_supports", {})
    if set(pins) != set(ids) or not all(pins[aid].get("joint_solved_pose") for aid in ids):
        raise ValueError("QA-10 requires exact static bindings from joint_core")
    for aid in ids:
        expected = np.asarray(pins[aid]["joint_expected_root_m"], float)
        if not np.allclose(expected, plan.actors[aid].positions_m[0], rtol=0., atol=1e-9):
            raise ValueError("QA-10 binding poses differ from the solved plan")
    row.update(bindings)
    _request(row, state, "QA-10")
    row["scene_demand"]["required_independent_family_counts"] = {"QA-10": 1}
    row["scene_demand"]["joint_static_answer"] = {
        "target_asset_id": target["asset_id"], "occluder_asset_id": occluder["asset_id"],
        "branch": "visible_registered_partial", "native_truth_required": True}
    validate_recipe(row, catalog, contract)
    return {"profile": "phase2", "row": deepcopy(row)}


class QA10Plugin:
    """Family constraints over the core's shared static geometry proposals."""
    def __init__(self, bodies, demand, *, assets, sound_requirements=()):
        self.bodies = bodies
        self.demand = demand
        self.assets = assets
        self.sound_requirements = tuple(sound_requirements)

    def candidates(self, context, request):
        from avengine.construction.joint import enumerate_static_layouts, schedule_sound_events, static_actor
        from avengine.construction.joint.native import strict_native_bindings
        from avengine.timeline.v7_family_constructors import ConstructionUnavailable
        schedule = schedule_sound_events(self.sound_requirements, start_grid_hz=10)
        if schedule.status != "solved":
            raise ConstructionUnavailable("QA-10 core sound schedule: " + str(schedule.reasons))
        roles = {aid: "inside" for aid in self.bodies}
        pair = (self.demand.target_asset_id, self.demand.occluder_asset_id,
                self.demand.minimum_occluded_fraction, self.demand.maximum_occluded_fraction)
        for plan in enumerate_static_layouts(
                context, self.bodies, roles, occlusion_pair=pair,
                floor_clearance_m=float(request.options.get("floor_clearance_m", .005))):
            plan.sounds = schedule.events
            plan.metadata["sound_schedule"] = {
                "status": schedule.status, "starts_s": list(schedule.starts_s),
                "diagnostics": schedule.diagnostics,
                "activity_authority": "source_normalization.measured_activity; final native activity independently read"}
            speaker = self.sound_requirements[0].entity_id
            plan.metadata["source_playback_starts_s"] = {speaker: list(schedule.starts_s)}
            # Answer identity should be readable: request the device's front
            # toward the viewer, inside solve's candidate stage.
            aid = self.demand.occluder_asset_id
            root = plan.actors[aid].positions_m[0]
            delta = np.asarray(plan.camera.position_m) - root
            plan.actors[aid] = static_actor(aid, root, self.bodies[aid],
                facing_rad=math.atan2(float(delta[2]), float(delta[0])))
            plan.actors[self.demand.occluder_asset_id].metadata.update(
                native_entity_kind="object", motion_action="static",
                map_profile="slide_floor_envelope_only")
            plan.metadata.update(qa10_target=self.demand.target_asset_id,
                                 qa10_occluder=self.demand.occluder_asset_id,
                                 native_mask_preflight_required=True)
            plan.metadata["native_bindings"] = strict_native_bindings(plan, self.assets)
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
            occluders=tuple((a.entity_id, a.positions_m[0], a.body)
                            for a in plan.actors.values() if a is not target))
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
        BodySpec, FamilyRequest, SoundRequirement, build_room_context, solve, certify_plan,
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
    activity = authored.get("source_normalization", {}).get("measured_activity", {})
    intervals = activity.get("source_activity_intervals_samples")
    rate = float(activity.get("source_rate_hz", 0))
    if not intervals or rate <= 0:
        raise ConstructionUnavailable("QA-10 needs measured activity for its admitted whole utterance")
    active_seconds = tuple((float(a)/rate, float(b)/rate) for a, b in intervals)
    sound_requirements = tuple(SoundRequirement(
        companion["asset_id"], authored["sound_asset_id"], "speech_playback",
        active_seconds, (start, start+1.), ((start, start + 9.),), wet_tail_s=.5)
        for start in (0., 10., 20.))
    request = FamilyRequest("QA-10", "visible_registered_partial", (target["asset_id"],),
                            {"occluder_asset_id": occluder["asset_id"],
                             "require_native_preflight": True,
                             "floor_clearance_m": float(runtime.get("qa10_floor_clearance_m", .005))})
    preparations = 0
    def prepare(context, request, plan):
        nonlocal preparations
        preparations += 1
        proposal = build_qa10_row(
            room, house, state, catalog, semantic, contract, plan,
            certify_plan(context, plan), target, occluder, companion,
            scene_id=scene_id, seed=seed)
        directory = Path(cache) / "qa10" / "native_preflight" / scene_id / (
            "candidate_" + str(preparations).zfill(3))
        return prepare_qa10_native_candidate(
            plan, proposal["row"], house, catalog, records, semantic, runtime,
            context=context, request=request, demand=demand, output_dir=directory)
    with build_room_context(actual, house=house, runtime=runtime,
                            cache_dir=Path(cache) / "qa10", camera_heights_m=(1.1,),
                            width=1280, height=720) as context:
        solved = solve(
            context, request, QA10Plugin(bodies, demand, assets=[target, occluder, companion],
                                         sound_requirements=sound_requirements),
            budget_s=float(runtime.get("qa10_planning_budget_s", 180.)),
            max_candidates=int(runtime.get("qa10_max_candidates", 256)),
            prepare_and_verify=prepare,
            max_native_preparations=int(runtime.get("qa10_max_native_preparations", 6)))
    if solved.status != "certified":
        raise ConstructionUnavailable("QA-10 joint construction " + solved.status + ": " + str(solved.reasons))
    return finalize_qa10_solution(solved, request)


def finalize_qa10_solution(solved, request):
    """Transfer the exact accepted live world to its renderer."""
    from dataclasses import asdict
    from avengine.construction.joint import PreparedCandidate
    prepared = solved.prepared
    payload = prepared.payload if isinstance(prepared, PreparedCandidate) else prepared
    row = payload.row
    row["automatic_construction"]["planning_certificate"] = asdict(solved.certificate)
    row["automatic_construction"]["joint_solver"] = {
        "status": solved.status, "diagnostics": solved.diagnostics,
        "branch": request.branch, "native_execution_must_not_replan": True}
    # solve() already bound the immutable core provenance to this payload.
    provenance = row.get("joint_solution", {})
    if (provenance.get("solver_entry") != "avengine.construction.joint.solve"
            or provenance.get("geometry_certificate_status") != "pass"
            or provenance.get("native_preflight_status") != "pass"):
        raise ValueError("QA-10 prepared world lacks core-owned solution provenance")
    return {"profile": "phase2", "row": row, "prepared_native": payload}


def prepare_qa10_native_candidate(plan, row, house, catalog, records, semantic, runtime,
                                  *, context, request, demand, output_dir):
    """QA-10 native predicate adapter; the core owns preparation and geometry."""
    from pathlib import Path
    from avengine.construction.joint import CheckResult
    from avengine.construction.joint.native import prepare_static_native
    from avengine.capture.v7_occlusion_demand import measure_requested_relations
    from avengine.timeline.continuous_pilot import write_json
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    def family_check(video, scene, entries):
        witness = video.attachment_checks[-1].get("requested_occlusion_witness")
        if witness is None:
            witness = measure_requested_relations(video, scene, entries)
        occluder_id = next(e["declaration"]["native_instance_id"] for e in entries
                           if e["declaration"]["asset_id"] == demand.occluder_asset_id)
        relation = certify_native_relation(witness, demand, str(occluder_id))
        write_json(directory/"native_occlusion_witness.json", witness)
        yield CheckResult("qa10:native_unique_partial_occluder", relation["status"],
                          ";".join(relation["errors"]), evidence=relation)
    minimum = {aid: 100 for aid in plan.actors}
    minimum[demand.target_asset_id] = demand.minimum_visible_pixels
    prepared = prepare_static_native(
        context, request, plan, row_factory=lambda unused: row,
        house=house, catalog=catalog, records=records, semantic=semantic,
        runtime=runtime, output_dir=directory/"core",
        family_check=family_check, roles={aid:"inside" for aid in plan.actors},
        minimum_visible_pixels=minimum, appearance_ids=tuple(plan.actors),
        acoustic_preflight=True)
    if prepared.payload is not None:
        payload = prepared.payload
        payload.row["automatic_construction"]["native_preflight"] = {
            "status": "pass", "certificate_path": str(payload.directory/"native_certificate.json"),
            "formal_render_support_proposal_limit": 1, "native_execution_must_not_replan": True,
            "scope": "same live prepared scene; final 450-frame stream remains required"}
    return prepared
