"""Common planning checks and witness-scoped calls to the original motion verifier."""
from __future__ import annotations
from itertools import combinations
import math
import numpy as np
from .types import Certificate, CheckResult
from .context import camera_legal
from .geometry import framing
from .routes import ConstructionError, SPEED_BANDS, minimum_track_distance, validate_track


def motion_witness_checks(condition_plan, positions_by_asset, listener_m):
    """Do not AND unrelated offsets into one family boolean.

    Each result retains plan index and crop offset. The caller selects promised
    witnesses; failed unpromised witnesses can be dropped without losing others.
    """
    from avengine.timeline.v7_motion_during_conditions import verify_planned_trajectories
    from copy import deepcopy
    checks = []
    for index, item in enumerate(condition_plan.get("plans", [])):
        for offset in item["offsets_s"]:
            one = deepcopy(item)
            one["offsets_s"] = [offset]
            verdicts = verify_planned_trajectories({"plans": [one]}, positions_by_asset, listener_m)
            for family, passed in verdicts.items():
                checks.append(CheckResult("motion:%d:%s:%g" % (index, family, offset), "pass" if passed else "fail",
                    "original_planned_trajectory_verifier", evidence={"family": family, "offset_s": offset,
                    "plan_index": index, "native_root_or_emitter_readback": False,
                    "predicate": "avengine.timeline.v7_motion_during_conditions.verify_planned_trajectories"}))
    return tuple(checks)


def certify_plan(context, plan):
    checks = []
    def check(name, ok, reason="", margin=None, **evidence):
        checks.append(CheckResult(name, "pass" if ok else "fail", reason, margin, evidence))
    clock_ok = (math.isfinite(plan.duration_s) and math.isfinite(plan.frame_rate_hz)
                and plan.duration_s > 0 and plan.frame_rate_hz > 0)
    expected = round(plan.duration_s*plan.frame_rate_hz)+1 if clock_ok else 0
    check("parent_clock", expected > 1 and clock_ok,
          frames=expected, fps=plan.frame_rate_hz, duration_s=plan.duration_s)
    legal, clearance = camera_legal(context.room, plan.camera)
    check("fixed_camera", legal, "room_interior_and_wall_clearance", clearance-.5,
          wall_clearance_m=clearance, camera_pose_count=1)
    if context.pathfinder is None:
        checks.append(CheckResult("native_navigation", "unknown", "missing_navmesh"))
    if not plan.actors:
        checks.append(CheckResult("actors", "fail", "no_real_entities"))
    valid = {}
    for key, actor in plan.actors.items():
        p = np.asarray(actor.positions_m, float)
        finite = p.shape == (expected, 3) and np.isfinite(p).all()
        check(key+":clock", finite and actor.entity_id == key, "finite_common_root_clock")
        if not finite:
            continue
        valid[key] = p
        fallback=None
        try:
            _, report = validate_track(context, p, actor.body.species, camera=plan.camera, snap=False)
            fallback=report.get("native_map_fallback")
            if fallback is not None:
                checks.append(CheckResult(key+":route","unknown","native_full_pose_required_for_map_fallback",
                                          evidence={"entity_id":key,**report}))
            else:
                check(key+":route", True, **report)
        except ConstructionError as exc:
            check(key+":route", False, exc.reason, **exc.details)
        if actor.headings_rad is not None:
            check(key+":headings", np.shape(actor.headings_rad) == (expected,) and np.isfinite(actor.headings_rad).all())
        # The existing map already includes its species radius. Only additional
        # actual width requires further clearance; never erode the same radius twice.
        from avengine.rooms.walkable_map import SPECIES_RADII_M
        extra = max(0., actor.body.radius_m - SPECIES_RADII_M[actor.body.species])
        if extra and fallback is None:
            angles = np.arange(16)*2*math.pi/16
            ok = all(context.walkable_map.contains_many(p[:, [0, 2]]+extra*np.array([math.cos(a), math.sin(a)]), actor.body.species).all()
                     for a in angles)
            if not ok and getattr(context,"allow_native_map_fallback",False):
                from .routes import _native_map_fallback
                try:
                    screen=_native_map_fallback(context,p,actor.body.species)
                    checks.append(CheckResult(key+":extra_body_width","unknown",
                        "native_full_pose_required_for_map_fallback",
                        evidence={"entity_id":key,"extra_radius_m":extra,"coarse_rejection":"circular_extra_body_width",**screen}))
                except ConstructionError as exc:
                    check(key+":extra_body_width",False,exc.reason,**exc.details)
            else:
                check(key+":extra_body_width", ok, additional_radius_m=extra)
        speed = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1)*plan.frame_rate_hz
        lo, hi = SPEED_BANDS[actor.body.species]
        mid = (np.arange(len(speed))+.5)/plan.frame_rate_hz
        ramp = float(actor.metadata.get("ramp_s", 0.))
        intervals = actor.metadata.get("walk_intervals_s", [])
        ramp_mask = np.zeros(len(speed), bool)
        valid_ramps = 0 <= ramp <= .5 and all(0 <= a < b <= plan.duration_s for a, b in intervals)
        if valid_ramps:
            for a, b in intervals:
                ramp_mask |= ((mid >= a-1/plan.frame_rate_hz) & (mid <= a+ramp+1/plan.frame_rate_hz))
                ramp_mask |= ((mid >= b-ramp-1/plan.frame_rate_hz) & (mid <= b+1/plan.frame_rate_hz))
        check(key+":speed", bool(valid_ramps and np.all(speed <= hi+1e-5)
              and np.all((speed < .02) | (speed >= lo-.02) | ramp_mask)),
              "species_speed_with_bounded_start_stop_ramps", max_speed_m_s=float(speed.max(initial=0)), band_m_s=(lo, hi))
        still = np.r_[True, speed < 1e-6] & np.r_[speed < 1e-6, True]
        rests = np.unique(p[still], axis=0)
        frames = [framing(plan.camera, q, actor.body)[0] for q in rests]
        check(key+":rest_framing", all(f != "edge" for f in frames), "whole_body_inside_or_outside_at_rest", states=frames)
    for a, b in combinations(valid, 2):
        ba, bb = plan.actors[a].body, plan.actors[b].body
        required = 1. if "human" in {ba.species,bb.species} and {ba.species,bb.species}<={"human","dog","cat"} else ba.radius_m+bb.radius_m
        minimum = minimum_track_distance(valid[a], valid[b])
        check(a+":"+b+":separation", minimum >= required-1e-7, "continuous_piecewise_linear_distance",
              minimum-required, minimum_m=minimum, required_m=required)
    classes = {}
    for index, event in enumerate(plan.sounds):
        finite = np.isfinite([event.start_s, event.end_s]).all()
        binding = event.entity_id in plan.actors and bool(event.sound_id) and bool(event.sound_class)
        check("sound:%d:binding" % index, binding and finite and 0 <= event.start_s < event.end_s <= plan.duration_s)
        wet = event.end_s+.5 if event.wet_end_s is None else event.wet_end_s
        check("sound:%d:wet_tail" % index, math.isfinite(wet) and event.end_s <= wet <= event.end_s+.5+1e-8,
              "declared_cap_requires_renderer_enforcement", wet_end_s=wet, measured_audio=False)
        classes.setdefault(event.entity_id, set()).add(event.sound_class)
    for key, values in classes.items():
        if key in plan.actors and plan.actors[key].body.species == "human":
            check(key+":one_sound_class", len(values) == 1, sound_classes=sorted(values))
    conditions = plan.metadata.get("motion_condition_plan")
    if conditions and len(valid) == len(plan.actors):
        # The existing adapter's source clock is specifically 901 at 30 Hz.
        if plan.frame_rate_hz != 30. or plan.duration_s != 30.:
            check("motion_clock", False, "original_motion_verifier_requires_30Hz_30s")
        else:
            tracks = plan.metadata.get("planned_emitter_positions_by_asset", valid)
            witness = motion_witness_checks(conditions, tracks, plan.camera.position_m)
            promised = plan.metadata.get("required_motion_witnesses")
            chosen = tuple(c for c in witness if promised is None or c.name in promised)
            if promised is not None and set(promised)-{c.name for c in witness}:
                check("motion_witness_selection", False, "unknown_required_motion_witness")
            checks.extend(chosen)
    return Certificate(tuple(checks))


def track_framing_check(camera, actor, *, required="outside", frame_indices=None):
    """Check every selected planned body envelope, including moving frames.

    This is conditional on BodySpec enclosing the actual poses. It is not
    native mask truth or a certificate of the supplied mesh envelope itself.
    """
    from collections import Counter
    if required not in {"inside","outside"}:
        raise ValueError("track framing requires inside or outside")
    p=np.asarray(actor.positions_m,float)
    name=actor.entity_id+":track_framing"
    if p.ndim!=2 or p.shape[1:]!=(3,)or not len(p)or not np.isfinite(p).all():
        return CheckResult(name,"fail","invalid_framing_root_clock")
    indices=np.arange(len(p))if frame_indices is None else np.asarray(frame_indices)
    if not indices.size:
        return CheckResult(name,"unknown","no_framing_frames_selected")
    if (indices.ndim!=1 or indices.dtype.kind not in "iu"
            or np.any(indices<0)or np.any(indices>=len(p))):
        return CheckResult(name,"fail","invalid_framing_frame_indices")
    indices=np.unique(indices)
    roots,inverse=np.unique(p[indices],axis=0,return_inverse=True)
    views=[framing(camera,point,actor.body)for point in roots]
    states=np.asarray([state for state,_ in views])[inverse]
    failed=indices[states!=required]
    margins=[margin for state,margin in views if state=="inside"]
    return CheckResult(name,"fail"if len(failed)else"pass","expanded_planned_body_envelope_all_selected_frames",
        evidence={"required_state":required,"checked_frames":len(indices),"total_track_frames":len(p),
            "unique_roots_checked":len(roots),"state_counts":dict(Counter(states.tolist())),
            "failed_frame_count":len(failed),"first_failed_frame_indices":failed[:12].tolist(),
            "minimum_inside_margin_px":min(margins)if margins else None,
            "authority":"planning_geometry_only","native_pixels_verified":False,
            "actual_posed_mesh_envelope_verified":False})
