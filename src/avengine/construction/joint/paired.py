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



def _joint_rest_camera(context, camera, actors, roles, yaw_search_deg):
    endpoints=[(actor,point)for actor in actors for point in actor.positions_m[[0,-1]]]
    centre=np.mean([point for _,point in endpoints],axis=0)-np.asarray(camera.position_m)
    if np.linalg.norm(centre[[0,2]])<1e-9:
        return None
    nominal=math.degrees(math.atan2(centre[0],-centre[2]))
    aims=[nominal]
    if any(roles[a.entity_id]!="inside"for a in actors):
        aims.append(camera.yaw_deg)
        for actor in actors:
            v=np.mean(actor.positions_m[[0,-1]],axis=0)-camera.position_m
            if np.linalg.norm(v[[0,2]])>1e-9:
                aims.append(math.degrees(math.atan2(v[0],-v[2])))
    angles=sorted(set([0.,-yaw_search_deg,yaw_search_deg]+
                      list(np.arange(-math.floor(yaw_search_deg),math.floor(yaw_search_deg)+1))),
                  key=lambda x:(abs(x),x))
    best=None
    for aim in aims:
        for shift in angles:
            candidate=replace(camera,yaw_deg=float(aim+shift))
            if not camera_legal(context.room,candidate)[0]:
                continue
            views=[framing(candidate,point,actor.body)for actor,point in endpoints]
            if any(state=="edge"or(roles[actor.entity_id]!="any"and state!=roles[actor.entity_id])
                   for (actor,_),(state,_)in zip(endpoints,views)):
                continue
            inside=[margin for state,margin in views if state=="inside"]
            score=(len(inside),min(inside)if inside else 0.,-abs(shift))
            if best is None or score>best[0]:
                best=(score,candidate,views,shift,aim)
    if best is None:
        return None
    score,selected,views,shift,aim=best
    return selected,{"states":[state for state,_ in views],
        "minimum_margin_px":score[1]if score[0]else None,"nominal_yaw_deg":aim,"yaw_shift_deg":float(shift)}

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
    for separation in separations:
        for side in sides:
            stats["lane_candidates"]+=1
            offset=np.array([side*separation*lateral[0],0.,side*separation*lateral[1]])
            a=_translated_actor(first,np.zeros(3));b=_translated_actor(second,offset)
            actual_distance=minimum_track_distance(a.positions_m,b.positions_m)
            if actual_distance<minimum_pair_distance_m-1e-8:
                reject("parallel_pair_distance");continue
            view=_joint_rest_camera(context,camera,(a,b),{a.entity_id:"inside",b.entity_id:"inside"},yaw_search_deg)
            if view is None:
                reject("parallel_complete_body_framing");continue
            selected,rest_framing=view
            plan=ScenePlan(selected,{a.entity_id:a,b.entity_id:b},frame_rate_hz=fps,
                duration_s=(len(p)-1)/fps,metadata={
                    "construction":"shared_parallel_fixed_clocks",
                    "lane_separation_m":separation,"side":side,
                    "minimum_pair_distance_m":actual_distance,
                    "rest_framing":rest_framing,
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


def _straight_shape(actor):
    p=np.asarray(actor.positions_m,float)
    if p.ndim!=2 or p.shape[1:]!=(3,)or len(p)<2 or not np.isfinite(p).all():
        raise ConstructionError("invalid_independent_actor_track")
    direction=p[-1,[0,2]]-p[0,[0,2]]
    length=float(np.linalg.norm(direction))
    if length<1e-7:
        raise ConstructionError("independent_pair_requires_moving_tracks")
    direction/=length
    relative=p[:,[0,2]]-p[0,[0,2]]
    if (np.max(np.abs(relative@np.array([-direction[1],direction[0]])))>1e-6
            or np.any(np.diff(relative@direction)<-1e-8)):
        raise ConstructionError("independent_pair_requires_monotone_straight_tracks")
    return p,direction,length


def _placed_straight_actor(actor, origin, heading):
    p,direction,_=_straight_shape(actor)
    old_heading=math.atan2(direction[1],direction[0])
    turn=heading-old_heading
    cosine,sine=math.cos(turn),math.sin(turn)
    rotation=np.array([[cosine,-sine],[sine,cosine]])
    def transform(points):
        x=np.asarray(points,float).copy()
        x[..., [0,2]]=(x[..., [0,2]]-p[0,[0,2]])@rotation.T+origin[[0,2]]
        x[...,1]+=origin[1]-p[0,1]
        return x
    metadata=deepcopy(actor.metadata)
    metadata.pop("validation",None)
    for name in ("fixed_clock_geometry","prototype_fixed_clock_geometry"):
        if name in metadata:
            if "origin_m"in metadata[name]:
                metadata[name]["origin_m"]=transform(metadata[name]["origin_m"]).tolist()
            if "heading_rad"in metadata[name]:
                metadata[name]["heading_rad"]+=turn
    if "waypoints_m"in metadata:
        metadata["waypoints_m"]=transform(metadata["waypoints_m"]).tolist()
    metadata["independent_straight_placement"]={"original_root_m":p[0].tolist(),
        "planned_root_m":origin.tolist(),"yaw_rotation_rad":turn,"distance_clock_rescaled":False}
    return ActorPlan(actor.entity_id,transform(p),actor.body,
        None if actor.headings_rad is None else np.asarray(actor.headings_rad,float).copy()+turn,metadata)


def independent_straight_pair_routes(context, first, second, *, camera, second_branch,
        roles=None, fps=30., origins_m=None, heading_offsets_deg=(0.,-15.,15.,-30.,30.,-60.,60.),
        minimum_pair_distance_m=1.05, minimum_radial_change_m=.5, yaw_search_deg=30.,
        max_origins=96, max_evaluations=1024, max_candidates=24, budget_s=20., diagnostics=None):
    """Solve the second straight route and common fixed view for supplied clocks.

    first stays at its exact solved positions. second is a canonical monotone
    line whose original distance and heading time series are rigidly placed,
    never retimed. Its requested full-track radial sign is screened; original
    family witness/utterance predicates still run in solve.verify.

    Default roots combine following seeds and diverse room-grid points; explicit
    origins_m is a caller-restricted world-root domain. Complete-body rest roles
    are inside/outside/any (default inside for both). This finite proposal search
    does not prove continuous-room infeasibility. Diagnostics distinguish its
    own compute budget, finite evaluations and yielded candidates.
    """
    from time import perf_counter
    from .layouts import _rank_points
    p,direction,_=_straight_shape(first)
    q,_,length=_straight_shape(second)
    if (p.shape!=q.shape or first.entity_id==second.entity_id or not math.isfinite(fps)or fps<=0
            or second_branch not in {"nearer","farther"}):
        raise ConstructionError("invalid_independent_pair_request")
    roles=dict(roles)if roles is not None else {first.entity_id:"inside",second.entity_id:"inside"}
    if set(roles)!={first.entity_id,second.entity_id}or any(v not in {"inside","outside","any"}for v in roles.values()):
        raise ConstructionError("invalid_independent_rest_roles")
    limits=[minimum_pair_distance_m,minimum_radial_change_m,yaw_search_deg,budget_s]
    offsets=tuple(float(x)for x in heading_offsets_deg)
    if (not np.isfinite(limits).all() or min(limits[:2])<0 or not 0<=yaw_search_deg<=180
            or budget_s<=0 or min(max_origins,max_evaluations,max_candidates)<1
            or not offsets or not np.isfinite(offsets).all()):
        raise ConstructionError("invalid_independent_pair_domain")
    if origins_m is None:
        seeds=[q[0].copy()]
        for endpoint in p[[0,-1]]:
            for gap in (max(1.05,minimum_pair_distance_m),1.25,1.5):
                for sign in (-1,1):
                    root=endpoint.copy();root[[0,2]]+=sign*gap*direction;seeds.append(root)
        preferred=length+3. if second_branch=="nearer"else 3.
        grid=_rank_points(context.points_m,camera,preferred=preferred,limit=max_origins)
        raw=np.concatenate((np.asarray(seeds),grid),axis=0)
    else:
        raw=np.asarray(origins_m,float)
    if raw.ndim!=2 or raw.shape[1:]!=(3,)or not np.isfinite(raw).all():
        raise ConstructionError("invalid_independent_origin_domain")
    roots=[];seen=set()
    for root in raw:
        key=tuple(np.round(root,6))
        if key not in seen:
            seen.add(key);roots.append(root.copy())
        if len(roots)>=max_origins:break
    stats=diagnostics if diagnostics is not None else {}
    stats.update(origin_domain=len(roots),evaluated=0,yielded=0,stop_reason="search_in_progress")
    failures=stats.setdefault("rejections",{})
    def reject(reason):
        failures[reason]=failures.get(reason,0)+1
    active_start=perf_counter();spent=0.
    radial_sign=-1 if second_branch=="nearer"else 1
    for origin in roots:
        radial=radial_sign*(origin[[0,2]]-np.asarray(camera.position_m)[[0,2]])
        if np.linalg.norm(radial)<1e-8:
            reject("independent_origin_at_camera");continue
        nominal=math.atan2(radial[1],radial[0])
        seed_heading=math.atan2(direction[1],direction[0])
        headings=[nominal+math.radians(x)for x in offsets]+[seed_heading,seed_heading+math.pi]
        unique=set()
        for heading in headings:
            key=round((heading+math.pi)%(2*math.pi)-math.pi,10)
            if key in unique:continue
            unique.add(key)
            if spent+perf_counter()-active_start>=budget_s:
                stats["stop_reason"]="compute_budget_exhausted";return
            if stats["evaluated"]>=max_evaluations:
                stats["stop_reason"]="evaluation_budget_exhausted";return
            stats["evaluated"]+=1
            b=_placed_straight_actor(second,origin,heading)
            radius=np.linalg.norm(b.positions_m[:,[0,2]]-np.asarray(camera.position_m)[[0,2]],axis=1)
            signed=radial_sign*np.diff(radius)
            if (radial_sign*(radius[-1]-radius[0])<minimum_radial_change_m
                    or np.any(signed<-.005)):
                reject("independent_requested_radial_branch");continue
            actual_distance=minimum_track_distance(p,b.positions_m)
            if actual_distance<minimum_pair_distance_m-1e-8:
                reject("independent_continuous_pair_distance");continue
            if (not getattr(context,"allow_native_map_fallback",False)
                    and not context.walkable_map.contains_many(b.positions_m[:,[0,2]],b.body.species).all()):
                reject("independent_map_cells");continue
            a=_translated_actor(first,np.zeros(3))
            view=_joint_rest_camera(context,camera,(a,b),roles,yaw_search_deg)
            if view is None:
                reject("independent_complete_body_framing");continue
            selected,rest_framing=view
            plan=ScenePlan(selected,{a.entity_id:a,b.entity_id:b},frame_rate_hz=fps,
                duration_s=(len(p)-1)/fps,metadata={"construction":"shared_independent_straight_fixed_clocks",
                    "second_branch":second_branch,"second_origin_m":origin.tolist(),"second_heading_rad":heading,
                    "minimum_pair_distance_m":actual_distance,"rest_framing":rest_framing,"roles":roles,
                    "second_full_track_radial_change_m":float(radius[-1]-radius[0]),
                    "native_full_pose_required":True,"family_radial_predicates_required":True})
            checks=certify_plan(context,plan).checks
            failed=[c for c in checks if c.status=="fail"]
            if failed:
                for c in failed:reject(c.name+":"+c.reason)
                continue
            plan.metadata["common_screen"]=[asdict(c)for c in checks]
            stats["yielded"]+=1
            spent+=perf_counter()-active_start
            yield plan
            active_start=perf_counter()
            if stats["yielded"]>=max_candidates:
                stats["stop_reason"]="candidate_budget_reached";return
    stats["stop_reason"]="finite_domain_exhausted"
