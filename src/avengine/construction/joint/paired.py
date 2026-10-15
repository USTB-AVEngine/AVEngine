"""Shared parallel-lane proposals for two already fixed motion clocks."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import math

import numpy as np

from .certification import certify_plan
from .context import camera_legal
from .geometry import framing
from .routes import ConstructionError, minimum_track_distance
from .types import ActorPlan, ScenePlan


def _translated_actor(actor, offset):
    metadata=deepcopy(actor.metadata)
    # Old room evidence cannot certify a translated proposal.
    metadata.pop("validation",None)
    for name in ("fixed_clock_geometry","prototype_fixed_clock_geometry"):
        if name in metadata and "origin_m" in metadata[name]:
            metadata[name]["origin_m"]=(np.asarray(metadata[name]["origin_m"],float)+offset).tolist()
    if "waypoints_m" in metadata:
        metadata["waypoints_m"]=(np.asarray(metadata["waypoints_m"],float)+offset).tolist()
    metadata["parallel_lane_translation_m"]=offset.tolist()
    return ActorPlan(actor.entity_id,np.asarray(actor.positions_m,float).copy()+offset,actor.body,
                     None if actor.headings_rad is None else np.asarray(actor.headings_rad,float).copy(),metadata)


def paired_parallel_routes(context, first, second, *, camera, fps=30.,
                           separations_m=(1.25,1.4,1.6,1.8), sides=(1,-1),
                           minimum_pair_distance_m=1.2, yaw_search_deg=30., diagnostics=None):
    """Yield ScenePlan proposals on separated parallel lines, without retiming.

    Both supplied ActorPlans must already have their actual clocks on the same
    straight centreline. The family determines direction, sound and witness
    times. Only second is translated; the fixed camera position is retained and
    its yaw is jointly selected for all four complete body envelopes.

    Common geometry is screened here, then checked again by solve. Explicit
    native-map fallback stays UNKNOWN until the actual native full-pose proof.
    This helper does not certify radial answers, native gait or negative audio
    controls; the family and original native gates must do so.
    """
    p=np.asarray(first.positions_m,float);q=np.asarray(second.positions_m,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<2 or p.shape!=q.shape
            or not np.isfinite(p).all() or not np.isfinite(q).all()
            or first.entity_id==second.entity_id or not math.isfinite(fps) or fps<=0):
        raise ConstructionError("invalid_parallel_actor_clocks")
    if (not math.isfinite(minimum_pair_distance_m) or minimum_pair_distance_m<0
            or not math.isfinite(yaw_search_deg) or not 0<=yaw_search_deg<=180):
        raise ConstructionError("invalid_parallel_lane_limits")
    separations=tuple(float(x)for x in separations_m);sides=tuple(sides)
    if (not separations or not np.isfinite(separations).all() or min(separations)<=0
            or not sides or any(x not in {-1,1}for x in sides)):
        raise ConstructionError("invalid_parallel_lane_domain")
    direction=p[-1,[0,2]]-p[0,[0,2]]
    length=float(np.linalg.norm(direction))
    if length<1e-7:
        raise ConstructionError("parallel_lane_requires_a_moving_seed")
    direction/=length;lateral=np.array([-direction[1],direction[0]])
    for track in (p,q):
        relative=track[:,[0,2]]-p[0,[0,2]]
        progress=relative@direction
        step=np.diff(progress)
        if (np.max(np.abs(relative@lateral))>1e-6
                or abs(progress[-1]-progress[0])<1e-7
                or not (np.all(step>=-1e-8)or np.all(step<=1e-8))):
            raise ConstructionError("paired_lanes_require_monotone_shared_straight_tracks")
    stats=diagnostics if diagnostics is not None else {}
    stats.setdefault("lane_candidates",0);stats.setdefault("yielded",0)
    failures=stats.setdefault("rejections",{})
    def reject(reason):
        failures[reason]=failures.get(reason,0)+1
    angles=sorted(set([0.,-yaw_search_deg,yaw_search_deg]+
                      list(np.arange(-math.floor(yaw_search_deg),math.floor(yaw_search_deg)+1))),key=abs)
    for separation in separations:
        for side in sides:
            stats["lane_candidates"]+=1
            offset=np.array([side*separation*lateral[0],0.,side*separation*lateral[1]])
            a=_translated_actor(first,np.zeros(3));b=_translated_actor(second,offset)
            actual_distance=minimum_track_distance(a.positions_m,b.positions_m)
            if actual_distance<minimum_pair_distance_m-1e-8:
                reject("parallel_pair_distance");continue
            endpoints=[(actor,point)for actor in (a,b)for point in actor.positions_m[[0,-1]]]
            centre=np.mean([point for _,point in endpoints],axis=0)-np.asarray(camera.position_m)
            if np.linalg.norm(centre[[0,2]])<1e-9:
                reject("parallel_camera_target_degenerate");continue
            nominal=math.degrees(math.atan2(centre[0],-centre[2]))
            best=None
            for shift in angles:
                candidate=replace(camera,yaw_deg=nominal+shift)
                if not camera_legal(context.room,candidate)[0]:
                    continue
                views=[framing(candidate,point,actor.body)for actor,point in endpoints]
                if any(state!="inside"for state,_ in views):
                    continue
                score=(min(margin for _,margin in views),-abs(shift))
                if best is None or score>best[0]:
                    best=(score,candidate,views,shift)
            if best is None:
                reject("parallel_complete_body_framing");continue
            score,selected,views,shift=best
            plan=ScenePlan(selected,{a.entity_id:a,b.entity_id:b},frame_rate_hz=fps,
                duration_s=(len(p)-1)/fps,metadata={
                    "construction":"shared_parallel_fixed_clocks",
                    "lane_separation_m":separation,"side":side,
                    "minimum_pair_distance_m":actual_distance,
                    "rest_framing":{"states":[state for state,_ in views],
                        "minimum_margin_px":score[0],"nominal_yaw_deg":nominal,"yaw_shift_deg":shift},
                    "native_full_pose_required":True,"family_radial_predicates_required":True})
            checks=certify_plan(context,plan).checks
            failed=[c for c in checks if c.status=="fail"]
            if failed:
                for c in failed:
                    reject(c.name+":"+c.reason)
                continue
            plan.metadata["common_screen"]=[asdict(c)for c in checks]
            stats["yielded"]+=1
            yield plan
