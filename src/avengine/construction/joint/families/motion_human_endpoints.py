"""Human motion demands: frame endpoints first, route with the shared core.

This module does not search camera positions or directions. The bootstrap uses
all legal floor sites and continuous yaw through the core camera_solve API.
All results remain proposals until the unchanged family/native checks pass.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from copy import copy
import heapq
import math
import numpy as np

FAMILIES = ("QA-06", "QA-15", "QA-16", "QA-17")


@dataclass(frozen=True)
class HumanEndpointDemand:
    entity_id: str
    branch: str
    minimum_during_change_m: float = .5

    def __post_init__(self):
        if self.branch not in {"nearer", "farther"}:
            raise ValueError("human endpoint demand needs an explicit radial answer")

    def roles(self):
        from avengine.qa.unified_catalog import POST_SOUND_DISTANCE_MARGIN_M
        points = {name: {"entity_id": self.entity_id, "framing_alternatives":
                  ["inside_full_body", "outside_full_body"],
                  "distance_range": {"minimum_m": 1.5}}
                  for name in ("start", "end")}
        return {"points": points, "route": {"kind": self.branch,
                "start": "start", "end": "end",
                "minimum_distance_difference_m":
                    self.minimum_during_change_m + POST_SOUND_DISTANCE_MARGIN_M},
                "camera": {"fixed_during_episode": True, "pitch_rad": 0.},
                "ranking": "maximum original-answer margin; no extra radial ring"}


@dataclass(frozen=True)
class EndpointPair:
    start_index: int
    end_index: int
    radial_difference_m: float


def floor_bounded_context(context):
    """Intersect the original map with the original floor BEFORE route search.

    Whole raster cells are kept so A* supercover links stay inside the room.
    The original v2 map and source geometry are never edited.
    """
    from shapely.geometry import shape
    import shapely
    from avengine.rooms.walkable_map import WalkableMap
    cached = getattr(context, "_human_floor_bounded", None)
    if cached is not None:
        return cached
    original = context.walkable_map
    polygon = shape(context.room["floor_polygon"])
    z, x = np.indices(original.shape)
    lo_x = original.origin[0] + x * original.resolution
    lo_z = original.origin[1] + z * original.resolution
    boxes = shapely.box(lo_x, lo_z, lo_x + original.resolution,
                        lo_z + original.resolution)
    within = np.asarray(shapely.covers(polygon, boxes), bool)
    layers = {key: values & within for key, values in original.layers.items()}
    derived = WalkableMap(original.origin.copy(), original.resolution,
                         original.floor_reference, layers,
                         {**original.metadata, "human_endpoint_domain":
                          "original v2 cells intersect original room floor polygon"},
                         original.floor_y)
    result = copy(context)
    result.walkable_map = derived
    context._human_floor_bounded = result
    return result


def endpoint_pairs(context, camera, demand, *, body=None, funnel=None):
    """Enumerate ALL eligible pairs by radial margin, with an O(N) heap."""
    from avengine.construction.joint import BodySpec
    from avengine.qa.unified_catalog import POST_SOUND_DISTANCE_MARGIN_M
    body = body or BodySpec.for_species("human")
    counts = funnel if funnel is not None else Counter()
    bounded = floor_bounded_context(context)
    grid = context.grid(camera, body)
    points = np.asarray(context.points_m, float)
    allowed = bounded.walkable_map.contains_many(points[:, [0, 2]], "human")
    allowed &= (np.asarray(grid.frustum_state) != "edge")
    allowed &= np.asarray(grid.distances_m) >= 1.5
    indices = np.flatnonzero(allowed)
    counts["camera_grids"] += 1
    counts["endpoint_domain_points"] += len(indices)
    if not len(indices):
        return
    distance = np.asarray(grid.distances_m, float)
    # In both branches transformed distance decreases from start to end.
    signed = distance if demand.branch == "nearer" else -distance
    ordered = indices[np.argsort(signed[indices], kind="stable")]
    values = signed[ordered]
    required = demand.minimum_during_change_m + POST_SOUND_DISTANCE_MARGIN_M
    valid_ends = np.searchsorted(values, signed[indices] - required, side="right")
    counts["candidate_pair_domain"] += int(valid_ends.sum())
    heap = []
    for start, n in zip(indices, valid_ends):
        if n:
            heap.append((float(values[0] - signed[start]), int(start), 0, int(n)))
    heapq.heapify(heap)
    while heap:
        negative_margin, start, j, n = heapq.heappop(heap)
        end = int(ordered[j])
        counts["candidate_pairs_yielded"] += 1
        yield EndpointPair(start, end, -negative_margin)
        j += 1
        if j < n:
            heapq.heappush(heap, (float(values[j] - signed[start]), start, j, n))


def ranked_endpoint_pairs(context, demand, *, body=None, funnel=None, cameras=None):
    """Merge per-camera streams; no fixed first-camera or top-k shortcut."""
    counts = funnel if funnel is not None else Counter()
    streams = []
    heap = []
    for index, camera in enumerate(context.cameras if cameras is None else cameras):
        counts["camera_candidates"] += 1
        stream = iter(endpoint_pairs(context, camera, demand, body=body, funnel=counts))
        streams.append((camera, stream))
        try:
            pair = next(stream)
        except StopIteration:
            continue
        heapq.heappush(heap, (-pair.radial_difference_m, index, pair))
    while heap:
        _, index, pair = heapq.heappop(heap)
        camera, stream = streams[index]
        yield camera, pair
        try:
            following = next(stream)
        except StopIteration:
            continue
        heapq.heappush(heap, (-following.radial_difference_m, index, following))


def route_for_pair(context, camera, pair, demand, *, speed_m_s, start_s=0.,
                   duration_s=30., body=None, funnel=None):
    """Call core A* and core rounded route/time construction, never a private path."""
    from avengine.construction.joint import (route_between, construct_route, BodySpec)
    counts = funnel if funnel is not None else Counter()
    bounded = floor_bounded_context(context)
    points = context.points_m
    cache=getattr(bounded,"_human_endpoint_paths",None)
    if cache is None:
        cache={};bounded._human_endpoint_paths=cache
    key=(pair.start_index,pair.end_index)
    if key not in cache:
        cache[key]=route_between(bounded, points[pair.start_index],
                                points[pair.end_index], species="human")
    else:
        counts["shared_route_reused"]+=1
    waypoints=cache[key]
    counts["route_between_pass"] += 1
    segments = np.linalg.norm(np.diff(waypoints[:, [0, 2]], axis=0), axis=1)
    radius = float(segments.min() / 2.) if len(segments) else 0.
    # Native scene-node translations are float32. Reserve the worst two-sample
    # XZ rounding displacement at 30Hz, without changing the declared speed band.
    coordinate_scale=np.float32(max(1.,float(np.abs(waypoints).max())))
    native_speed_guard=float(np.spacing(coordinate_scale))*60.*math.sqrt(2.)
    speed_m_s=float(np.clip(speed_m_s,.85+native_speed_guard,1.35-native_speed_guard))
    # The species map already includes the human footprint; subtracting its
    # radius again here collapsed every fillet to zero on valid NavMesh cells.
    # Core construct_route validates the actual rounded path against that map.
    from avengine.construction.joint.routes import ConstructionError,minimum_track_distance
    from copy import deepcopy
    strict=copy(bounded);strict.allow_native_map_fallback=False
    actor_cache=getattr(bounded,"_human_endpoint_actors",None)
    if actor_cache is None:
        actor_cache={};bounded._human_endpoint_actors=actor_cache
    key=(pair.start_index,pair.end_index,speed_m_s,start_s,duration_s,body)
    if key in actor_cache:
        actor=deepcopy(actor_cache[key])
        counts["core_route_clock_reused"]+=1
    else:
        import time
        while True:
            if time.monotonic()>getattr(context,"_human_endpoint_search_deadline",math.inf):
                raise ConstructionError("geometry_search_compute_budget")
            try:
                actor=construct_route(strict,waypoints,species="human",
                    speed_m_s=speed_m_s,start_s=start_s,duration_s=duration_s,
                    corner_radius_m=radius,camera=None,entity_id=demand.entity_id,body=body)
                speed=np.linalg.norm(np.diff(actor.positions_m[:,[0,2]],axis=0),axis=1)*30.
                mid=(np.arange(len(speed))+.5)/30.
                ramp=float(actor.metadata["ramp_s"]);steady=np.ones(len(speed),bool)
                for lo_s,hi_s in actor.metadata["walk_intervals_s"]:
                    steady&=~(((mid>=lo_s-1/30.)&(mid<=lo_s+ramp+1/30.))|
                              ((mid>=hi_s-ramp-1/30.)&(mid<=hi_s+1/30.)))
                observed=speed[steady&(speed>=.02)]
                if len(observed)and observed.min()<.8-.02:
                    # Solve the clip speed inside the existing owner band;
                    # short curved chords can be slower than their arc clock.
                    next_speed=min(1.35-native_speed_guard,speed_m_s*.8/float(observed.min()))
                    if next_speed<=speed_m_s:
                        raise ConstructionError("species_speed_with_bounded_start_stop_ramps")
                    speed_m_s=next_speed
                    counts["owner_speed_domain_refinements"]+=1
                    continue
                break
            except ConstructionError as exc:
                if exc.reason=="route_exceeds_parent_duration"and start_s>0:
                    start_s=max(0.,duration_s-(float(exc.details["end_s"])-start_s))
                    continue
                if exc.reason in {"walkable_map_cells","walkable_map_segment",
                                  "snapped_route_outside_map","snapped_segment_outside_map"}:
                    previous=radius;radius/=2.
                    counts["rounded_radius_domain_refinements"]+=1
                    if radius>np.spacing(max(1.,float(np.abs(waypoints).max()))) and radius<previous:continue
                raise
        actor.metadata["rounded_core_radius_m"]=radius
        actor.metadata["native_float32_speed_guard_m_s"]=native_speed_guard
        actor_cache[key]=deepcopy(actor)
    minimum=minimum_track_distance(actor.positions_m,np.asarray(camera.position_m))
    if minimum<1.5-1e-8:
        raise ConstructionError("camera_standoff",{"minimum_m":minimum,"required_m":1.5})
    counts["construct_route_pass"] += 1
    actor.entity_id=demand.entity_id
    actor.metadata["human_endpoint_demand"] = demand.roles()
    actor.metadata["endpoint_indices"] = [pair.start_index, pair.end_index]
    actor.metadata["endpoint_radial_difference_m"] = pair.radial_difference_m
    return actor


def native_endpoint_route_proposals(video, entry, scene, distance, local_forward, request):
    """Consume the solved original route and exact clock without relocating it."""
    from types import SimpleNamespace
    solved = request["joint_endpoint_route"]
    positions = np.asarray(solved["positions_m"], float)
    planned_distance = np.asarray(solved["planned_distance_m"], float)
    observed_distance = np.asarray(distance, float)
    if (positions.shape != (901, 3) or planned_distance.shape != (901,) or
            observed_distance.shape != (901,) or
            not np.allclose(observed_distance, planned_distance, atol=1e-8, rtol=0)):
        raise ValueError("native gait clock differs from solved endpoint route")
    applied = np.asarray(entry["applied_root_input"], float)[:3, 3]
    if not np.allclose(applied[[0, 2]], positions[0, [0, 2]], atol=.002, rtol=0):
        raise ValueError("native endpoint start XZ changed after joint solving")
    from avengine.construction.joint import CameraPose
    camera = CameraPose(**solved["camera"])
    if not np.allclose(camera.position_m, scene["listener_m"], atol=1e-6, rtol=0):
        raise ValueError("native camera differs from solved endpoint camera")
    if not np.allclose(camera.forward, scene["camera_forward"], atol=1e-6, rtol=0):
        raise ValueError("native camera yaw differs from solved endpoint camera")
    local = np.asarray(local_forward, float)
    headings = np.asarray(solved["headings_rad"], float)
    yaws = math.atan2(float(local[2]), float(local[0])) - headings
    yield SimpleNamespace(positions=positions.copy(), yaws=yaws,
                          receipt={"route_kind": "core endpoint route_between/construct_route",
                                   "root_samples": 901, "execution_replanning": False,
                                   "clock_changed": False, "camera_changed": False,
                                   "standing_y_delta_m": float(applied[1] - positions[0, 1]),
                                   "native_full_pose_checks_still_required": True})


def spatial_endpoint_domain(context,demand,*,funnel=None):
    """All floor endpoint pairs before choosing the shared camera."""
    from avengine.qa.unified_catalog import POST_SOUND_DISTANCE_MARGIN_M
    counts=funnel if funnel is not None else Counter()
    bounded=floor_bounded_context(context)
    points=np.asarray(context.points_m,float)
    ids=np.flatnonzero(bounded.walkable_map.contains_many(points[:,[0,2]],"human"))
    counts["spatial_endpoint_points"]+=len(ids)
    if len(ids)<2:return []
    xz=points[ids][:,[0,2]]
    distances=np.linalg.norm(xz[:,None,:]-xz[None,:,:],axis=2)
    required=demand.minimum_during_change_m+POST_SOUND_DISTANCE_MARGIN_M
    # Four-neighbour components exactly match reachability for the core's
    # eight-neighbour A* that forbids diagonal corner-cutting. No private path.
    from scipy.ndimage import label
    labels=getattr(bounded,"_human_component_labels",None)
    if labels is None:
        labels,_=label(bounded.walkable_map.layers["human"])
        bounded._human_component_labels=labels
    cells=np.floor((xz-bounded.walkable_map.origin)/bounded.walkable_map.resolution).astype(int)
    components=labels[cells[:,1],cells[:,0]]
    distance_ok=distances>=required
    connected=(components[:,None]==components[None,:])&(components[:,None]>0)
    counts["unconnected_endpoint_pairs_avoided"]+=int(np.count_nonzero(distance_ok&~connected))
    a,b=np.nonzero(distance_ok&connected)
    order=np.argsort(-distances[a,b],kind="stable")
    counts["spatial_candidate_pair_domain"]+=len(order)
    return [EndpointPair(int(ids[a[k]]),int(ids[b[k]]),float(distances[a[k],b[k]]))
            for k in order]


def joint_endpoint_groups(first,second,*,points_m=None):
    """All independent pairs; prefer separated rest endpoints, never drop any."""
    if not first or not second:return
    if points_m is None:
        heap=[(-(first[0].radial_difference_m+second[0].radial_difference_m),0,0)]
        seen={(0,0)}
        while heap:
            _,i,j=heapq.heappop(heap)
            yield first[i],second[j]
            for a,b in ((i+1,j),(i,j+1)):
                if a>=len(first)or b>=len(second)or(a,b)in seen:continue
                seen.add((a,b))
                heapq.heappush(heap,(-(first[a].radial_difference_m+second[b].radial_difference_m),a,b))
        return
    p=np.asarray(points_m)[:,[0,2]]
    spatial=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=2)
    starts=np.asarray([q.start_index for q in second])
    ends=np.asarray([q.end_index for q in second])
    orders={};heap=[]
    def add(i,j):
        pair=first[i]
        if i not in orders:
            clearance=np.minimum.reduce([spatial[pair.start_index,starts],
                spatial[pair.start_index,ends],spatial[pair.end_index,starts],
                spatial[pair.end_index,ends]])
            preferred=clearance>=1.
            orders[i]=np.r_[np.flatnonzero(preferred),np.flatnonzero(~preferred)]
        if j>=len(orders[i]):return
        k=int(orders[i][j]);other=second[k]
        clearance=min(spatial[a,b]for a in(pair.start_index,pair.end_index)
                       for b in(other.start_index,other.end_index))
        heapq.heappush(heap,(int(clearance<1.),
            -(pair.radial_difference_m+other.radial_difference_m),i,j,k))
    add(0,0);unseen=1
    while heap:
        _,_,i,j,k=heapq.heappop(heap)
        yield first[i],second[k]
        add(i,j+1)
        if unseen<len(first):
            add(unseen,0);unseen+=1


def solve_endpoint_cameras(context,pairs,demands,bodies,*,funnel=None):
    """Four endpoint roles, all legal sites/heights, continuous horizontal yaw."""
    from avengine.construction.joint.camera_solve import RouteRole
    from avengine.qa.unified_catalog import POST_SOUND_DISTANCE_MARGIN_M
    counts=funnel if funnel is not None else Counter()
    roles=[]
    for pair,demand,body in zip(pairs,demands,bodies):
        points=tuple(map(tuple,np.asarray(context.points_m)[[pair.start_index,pair.end_index]].tolist()))
        roles.append(RouteRole(points,start_role="rest_full_body",end_role="rest_full_body",
                     body=body,name=demand.entity_id,distance_change=demand.branch,
                     min_distance_change_m=demand.minimum_during_change_m+POST_SOUND_DISTANCE_MARGIN_M))
    solved=context.solve_cameras(roles,limit=None)
    counts["joint_camera_role_groups"]+=1
    counts["camera_position_height_tests"]+=solved.diagnostics["position_height_pairs"]
    counts["camera_role_solutions"]+=len(solved.solutions)
    return solved


def retained_actor_sets(context,actors,pairs,bodies,ids,failures,seed):
    """Keep valid independent movement; only the failing element stands."""
    from avengine.construction.joint import ActorPlan
    from avengine.construction.joint.routes import minimum_track_distance
    if all(a is not None for a in actors):
        if minimum_track_distance(actors[0].positions_m,actors[1].positions_m)>=1.-1e-8:
            repeated=repeat_first_event(context,actors[0])
            if repeated is not None and minimum_track_distance(
                    repeated.positions_m,actors[1].positions_m)>=1.-1e-8:
                yield [repeated,actors[1]]
            yield actors
            return
        failures=["independent routes violate simultaneous owner spacing"]*2
    for keep in range(2):
        if actors[keep] is None:continue
        other=1-keep
        pair=pairs[other]
        # Both endpoint roles were solved with this exact fixed camera.
        for index in dict.fromkeys((pair.start_index,pair.end_index)):
            root=np.asarray(context.points_m[index],float)
            positions=np.repeat(root[None,:],len(actors[keep].positions_m),axis=0)
            if minimum_track_distance(actors[keep].positions_m,positions)<1.-1e-8:continue
            heading=float(np.random.default_rng(seed+other+index).uniform(-math.pi,math.pi))
            static=ActorPlan(ids[other],positions,bodies[other],
                np.full(len(positions),heading),
                {"native_yaw_rad":-heading-math.pi/2,
                 "walk_intervals_s":[],"ramp_s":0.,
                 "element_degradation":{"entity_id":ids[other],"from":"walk","to":"stand_and_speak",
                     "reason":failures[other]or"independent route cannot retain owner spacing",
                     "dropped_families":list(FAMILIES)}})
            result=list(actors);result[other]=static
            if keep==0:
                repeated=repeat_first_event(context,actors[keep])
                if repeated is not None and minimum_track_distance(
                        repeated.positions_m,static.positions_m)>=1.-1e-8:
                    yield [repeated,static]
            yield result


def repeat_first_event(context,actor):
    """Try a reversed second event on the same core route in the middle crop."""
    from avengine.construction.joint import repeat_bouts,ConstructionError
    if "human_endpoint_demand"not in actor.metadata:return None
    intervals=actor.metadata.get("walk_intervals_s",[])
    if len(intervals)!=1 or intervals[0][1]>=10.:return None
    gap=10.+intervals[0][0]-intervals[0][1]
    if gap<=2/30.:return None
    try:
        repeated=repeat_bouts(actor,offsets_s=(0.,10.),
                source_window_s=(0.,10.),turn_duration_s=(gap-1/30.)/2.,
                context=floor_bounded_context(context))
    except ConstructionError:
        return None
    branch=actor.metadata["human_endpoint_demand"]["route"]["kind"]
    repeated.metadata["extra_motion_events"]=[{
        "crop_offset_s":10.,"branch":"farther"if branch=="nearer"else"nearer"}]
    return repeated


def stationary_speech_slot(record,crop,companion_duration):
    duration=math.ceil(record["header"]["sample_count"]/1600.)/10.
    reply=math.floor((10.-companion_duration-.5-1/15.)*10.+1e-9)/10.
    return {"start_s":crop,"local_start_s":0.,"reply_local_s":reply,
            "duration_s":duration,"queries":[],"rank":(0,0,0),
            "predicted_families":dict.fromkeys(FAMILIES,False)}


class EndpointHumanPlugin:
    """Joint role demands, shared-core paths, independent people and original QA."""
    def __init__(self,room,house,catalog,records,semantic,contract,state,scene_id,seed,runtime):
        self.room,self.house=room,house
        self.catalog,self.records,self.semantic=catalog,records,semantic
        self.contract,self.state=contract,state
        self.scene_id,self.seed,self.runtime=scene_id,int(seed),runtime
        self.funnel=Counter()
        self.failures=Counter()
        self.best_partial=None
        self.best_partial_score=(-1,-1)

    def _build_candidate(self,context,camera,actors,assets,pair,measured,slots):
        from copy import deepcopy
        from avengine.construction.joint import ScenePlan,SoundEvent,certify_plan
        from avengine.timeline.v7_human_endpoint_program import (
            base_row,attach_routes,verify_actor_families,combine_obligations)
        from avengine.timeline.v7_family_clock import build_constructed_scene
        from avengine.timeline.v7_family_constructors import _request
        from avengine.timeline.v7_production_inputs import attach_runtime_contract,validate_recipe
        from avengine.construction.joint import strict_native_bindings
        plan=ScenePlan(camera,actors,metadata={"registered_assets":assets,
                       "roles":{a["asset_id"]:"any"for a in assets}})
        row=base_row(self.room,assets,pair,self.scene_id,self.seed,camera,slots)
        row=attach_routes(row,plan,assets)
        original_plans=[];answers={};failures={}
        for index,asset in enumerate(assets):
            aid=asset["asset_id"]
            if aid not in row["motion_requests"]:
                continue
            branch=actors[aid].metadata["human_endpoint_demand"]["route"]["kind"]
            event_slots=[(branch,slots[aid])]+[
                (extra["branch"],extra)for extra in slots[aid].get("additional_events",[])]
            for event_branch,event_slot in event_slots:
                key=aid if event_slot is slots[aid]else aid+"@"+str(event_slot["crop_offset_s"])
                plans,verdicts,rejections=verify_actor_families(
                    row,plan,assets,self.records+measured,aid,event_branch,event_slot,funnel=self.funnel)
                original_plans.extend(plans);answers[key]=verdicts;failures[key]=rejections
        if not original_plans:
            self.failures["no_original_family_witness"]+=1
            return None
        row=combine_obligations(row,plan,original_plans,answers)
        row.update(strict_native_bindings(plan,assets))
        row["native_view_roles"]={a["asset_id"]:{"view":"unconstrained_motion","static":a["asset_id"]not in row["motion_requests"],
                                                "required_visible_sometime":False}for a in assets}
        row["human_endpoint_family_failures"]=failures
        row["joint_prepared_scene_required"]=True
        _request(row,self.state,"QA-06")
        retained=sorted({q for values in answers.values()for q,ok in values.items()if ok})
        row["scene_demand"]["condition_families"]={"motion_human":retained}
        row["scene_demand"]["required_actual_candidate_families"]=retained
        row["scene_demand"]["element_degradations"]=[
            a.metadata["element_degradation"] for a in actors.values()
            if "element_degradation" in a.metadata]
        row["scene_demand"]["required_independent_family_counts"]={
            qa:sum(bool(values.get(qa))for values in answers.values())for qa in retained}
        row["human_motion_event_count"]=len(answers)
        row["primary_motion_speech_anchors_s"]={
            aid:t for aid,t in row["primary_motion_speech_anchors_s"].items()
            if aid in row["motion_requests"]}
        row=attach_runtime_contract(row,self.runtime)
        validate_recipe(row,self.catalog,self.contract)
        _,scene,program,actual_assets=build_constructed_scene(
            row,self.house,self.records,self.catalog,self.semantic)
        bindings={e["event_id"]:e for e in program["events"]}
        plan.sounds=tuple(SoundEvent(scene["sources"][e["endpoint_id"]]["asset_id"],
                 bindings[e["event_id"]]["sound_asset_id"],e["sound_class"],
                 e["start_tick"]/10.,e["end_tick"]/10.,min(30.,e["end_tick"]/10.+.5))
                 for e in scene["events"])
        plan.metadata.update(family_row=deepcopy(row),
                            original_family_results=answers,
                            endpoint_funnel=dict(self.funnel))
        certificate=certify_plan(context,plan)
        bad=[c for c in certificate.checks if c.status=="fail"]
        if bad:
            for c in bad:self.failures[c.name+":"+c.reason]+=1
            return None
        self.funnel["common_and_original_family_candidates"]+=1
        return plan

    def candidates(self,context,request):
        import time
        from copy import deepcopy
        from avengine.construction.joint import BodySpec,ConstructionError
        from avengine.construction.joint.routes import minimum_track_distance,SPEED_BANDS
        from avengine.qa.unified_catalog import POST_SOUND_DISTANCE_MARGIN_M
        from avengine.timeline.v7_human_endpoint_program import (
            complete_pairs,human_roles,measured_record,speech_slots)
        start=time.monotonic()
        budget=float(self.runtime.get("human_endpoint_geometry_budget_s",
                     float(self.runtime.get("human_endpoint_solve_budget_s",900.))/2.))
        context._human_endpoint_search_deadline=start+budget
        all_pairs=complete_pairs(self.semantic)
        for voice_pair in all_pairs:
            if time.monotonic()-start>budget:
                self.funnel["geometry_time_budget_exhausted"]+=1
                if self.best_partial is not None:
                    best=self.best_partial;self.best_partial=None
                    yield best
                return
            try:
                assets=human_roles(self.catalog,voice_pair,self.seed)
                measured=[measured_record(s,a)for s,a in zip(voice_pair,assets)]
            except ValueError as exc:
                self.failures[str(exc)]+=1;continue
            ids=[a["asset_id"]for a in assets]
            spans=[(r["screens"]["automatic_activity"]["activity_intervals_samples"][-1][1]-
                    r["screens"]["automatic_activity"]["activity_intervals_samples"][0][0])/16000.
                   for r in measured]
            low,high=.85,1.35  # owner 12:15: construct inside the tolerated [.8,1.4] band
            if any(span*high<.5 for span in spans):
                self.failures["whole_recording_cannot_cover_QA15_at_owner_max_speed"]+=1
                continue
            branches=["nearer"if self.seed%2 else"farther",
                      "farther"if self.seed%2 else"nearer"]
            demands=[HumanEndpointDemand(aid,branch)for aid,branch in zip(ids,branches)]
            bodies=[BodySpec.for_species("human")for _ in assets]
            first=spatial_endpoint_domain(context,demands[0],funnel=self.funnel)
            second=spatial_endpoint_domain(context,demands[1],funnel=self.funnel)
            for pa,pb in joint_endpoint_groups(first,second,points_m=context.points_m):
                if time.monotonic()-start>budget:
                    self.funnel["geometry_time_budget_exhausted"]+=1
                    if self.best_partial is not None:
                        best=self.best_partial;self.best_partial=None
                        yield best
                    return
                self.funnel["joint_endpoint_groups"]+=1
                p=np.asarray(context.points_m)
                # At clip start both roots occupy their initial endpoints; at
                # clip end both occupy their final endpoints. These are owner distances.
                if (np.linalg.norm((p[pa.start_index]-p[pb.start_index])[[0,2]])<1. or
                    np.linalg.norm((p[pa.end_index]-p[pb.end_index])[[0,2]])<1.):
                    self.failures["original_simultaneous_endpoint_spacing"]+=1;continue
                solved=solve_endpoint_cameras(context,(pa,pb),demands,bodies,funnel=self.funnel)
                if not solved.solutions:
                    self.failures["joint_camera_role_domain_empty"]+=1;continue
                for solution in solved.solutions:
                    if time.monotonic()-start>budget:
                        self.funnel["geometry_time_budget_exhausted"]+=1
                        if self.best_partial is not None:
                            best=self.best_partial;self.best_partial=None
                            yield best
                        return
                    camera=solution.camera
                    actors=[];actual_pairs=[];route_failures=[]
                    for index,pair in enumerate((pa,pb)):
                        d=np.linalg.norm(p[[pair.start_index,pair.end_index]][:,[0,2]]-
                                         np.asarray(camera.position_m)[[0,2]],axis=1)
                        radial=(d[0]-d[1])if branches[index]=="nearer"else(d[1]-d[0])
                        actual_pair=EndpointPair(pair.start_index,pair.end_index,float(radial))
                        speed=float(np.clip((radial+.5-POST_SOUND_DISTANCE_MARGIN_M)/(2*spans[index]+.5),low,high))
                        try:
                            actor=route_for_pair(context,camera,actual_pair,demands[index],
                                  speed_m_s=speed,start_s=0. if index==0 else 20.,
                                  body=bodies[index],funnel=self.funnel)
                            route_failures.append(None)
                        except ConstructionError as exc:
                            self.failures[f"actor_{index}_route:"+exc.reason]+=1
                            actor=None;route_failures.append(exc.reason)
                        actors.append(actor);actual_pairs.append(actual_pair)
                    durations=[math.ceil(r["header"]["sample_count"]/1600.)/10. for r in measured]
                    for placed in retained_actor_sets(context,actors,(pa,pb),bodies,ids,
                                                      route_failures,self.seed):
                        moving=sum("element_degradation"not in a.metadata for a in placed)
                        self.funnel["two_independent_routes_spacing_pass"if moving==2
                                    else"single_element_standing_fallback"]+=1
                        options=[speech_slots(actor,camera,record,branch,crop_offset=offset,
                                 companion_duration_s=durations[1-index])
                                 if"element_degradation"not in actor.metadata else
                                 [stationary_speech_slot(record,offset,durations[1-index])]
                                 for index,(actor,record,branch,offset)in enumerate(
                                      zip(placed,measured,branches,(0.,20.)))]
                        if not all(options):
                            self.failures["speech_clock_domain_empty"]+=1;continue
                        for index,actor in enumerate(placed):
                            for extra in actor.metadata.get("extra_motion_events",[]):
                                additional=speech_slots(actor,camera,measured[index],extra["branch"],
                                    crop_offset=extra["crop_offset_s"],
                                    companion_duration_s=durations[1-index])
                                if additional:
                                    for slot in options[index]:
                                        slot["additional_events"]=[{**additional[0],**extra}]
                        stop_slots=False
                        for sa in options[0]:
                            for sb in options[1]:
                                if time.monotonic()-start>budget:
                                    self.funnel["geometry_time_budget_exhausted"]+=1
                                    if self.best_partial is not None:
                                        best=self.best_partial;self.best_partial=None
                                        yield best
                                    return
                                candidate=self._build_candidate(context,camera,
                                    {ids[0]:deepcopy(placed[0]),ids[1]:deepcopy(placed[1])},
                                    assets,voice_pair,measured,{ids[0]:sa,ids[1]:sb})
                                if candidate is not None:
                                    candidate.metadata["camera_solve"]=solution.evidence
                                    candidate.metadata["camera_role_margins"]=solution.role_margins
                                    self.funnel["ranked_candidate_plans"]+=1
                                    values=candidate.metadata["original_family_results"]
                                    score=(sum(sum(v.values())for v in values.values()),len(values))
                                    if score>self.best_partial_score:
                                        self.best_partial=candidate;self.best_partial_score=score
                                    if len(values)>=2 and all(all(v.values())for v in values.values()):
                                        self.best_partial=None
                                        paused=time.monotonic()
                                        yield candidate
                                        start+=time.monotonic()-paused
                                        context._human_endpoint_search_deadline=start+budget
                                    if sum(sum(v.values())for v in values.values())==4*len(values):
                                        stop_slots=True
                                        break
                            if stop_slots:break
        if self.best_partial is not None:
            best=self.best_partial;self.best_partial=None
            yield best

    def verify(self,context,request,plan):
        from avengine.construction.joint.families.motion_human import motion_witness_checks
        yield from motion_witness_checks(plan.metadata["family_row"]["motion_during_condition_plan"],
                plan.metadata["planned_emitter_positions_by_asset"],plan.camera.position_m)


def construct_human_endpoint_scene(room,house,state,catalog,records,semantic,contract,runtime,
        *,scene_id,seed,family="QA-06",attempt=0):
    """Phase-two entry; only core owns route and camera implementations."""
    from pathlib import Path
    from dataclasses import asdict
    import json
    from avengine.construction.joint import (build_room_context,FamilyRequest,solve,prepare_native,
                                            certify_plan)
    from avengine.construction.joint.families.motion_human import native_motion_obligations
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable
    original=next(r for r in house["checked_rooms"]if r["room_label"]==room["room_id"])
    context=build_room_context(original,house=house,runtime=runtime,
          cache_dir=runtime.get("human_endpoint_context_cache"),camera_limit=int(runtime.get("human_endpoint_camera_compute_budget",512)),
          width=1280,height=720,allow_native_map_fallback=True)
    plugin=EndpointHumanPlugin(room,house,catalog,records,semantic,contract,state,scene_id,seed,runtime)
    request=FamilyRequest(family,"independent_endpoint_"+("nearer_farther"if seed%2 else"farther_nearer"),
                          (),{"desired_families":list(FAMILIES),"per_element_degradation":True,
                              "require_native_preflight":True})
    root=Path(runtime["human_endpoint_evidence_root"])/scene_id;root.mkdir(parents=True,exist_ok=True)
    def save_funnel():
        (root/"endpoint_funnel.json").write_text(json.dumps({
             "source":"joint endpoint roles, all legal floor camera sites/continuous yaw; core route_between/construct_route",
             "funnel":dict(plugin.funnel),"rejections":dict(plugin.failures),
             "native_rendered_counts":{}},ensure_ascii=False,indent=2)+"\n")
    def oracle(ctx,req,plan):
        from avengine.timeline.v7_family_clock import build_constructed_scene
        from .motion_human_native import retain_native_witnesses
        rows=[]
        def scene_factory(row,*args):
            rows.append(row)
            result=build_constructed_scene(row,*args)
            rows.append(result[0])
            return result
        return prepare_native(ctx,req,plan,row_factory=lambda p:p.metadata["family_row"],
             house=house,catalog=catalog,records=records,semantic=semantic,runtime=runtime,
             output_dir=root,roles=plan.metadata["roles"],minimum_visible_pixels={},
             scene_factory=scene_factory,
             family_check=lambda video,scene,entries:retain_native_witnesses(
                   video,scene,entries,plan,rows))
    try:
        if runtime.get("human_endpoint_geometric_only"):
            plan=next(plugin.candidates(context,request),None)
            if plan is None:raise ConstructionUnavailable("endpoint geometry domain exhausted; see "+str(root))
            certificate=certify_plan(context,plan)
            (root/"geometric_plan.json").write_text(json.dumps(asdict(plan),
                default=lambda x:x.tolist()if hasattr(x,"tolist")else str(x),indent=2)+"\n")
            return {"profile":"phase2","row":plan.metadata["family_row"],
                    "geometric_only":True,"joint_plan":plan,"certificate":certificate}
        solved=solve(context,request,plugin,prepare_and_verify=oracle,
                     budget_s=float(runtime.get("human_endpoint_solve_budget_s",900.)))
        detail={"status":solved.status,"reasons":list(solved.reasons),
                "diagnostics":solved.diagnostics,
                "checks":[asdict(c)for c in solved.certificate.checks]if solved.certificate else[]}
        (root/("joint_solution.json"if solved.status in ("certified","degraded")else"joint_failure.json")).write_text(
            json.dumps(detail,default=lambda x:x.tolist()if hasattr(x,"tolist")else str(x),indent=2)+"\n")
        if solved.status not in ("certified","degraded")or solved.prepared is None:
            raise ConstructionUnavailable("endpoint joint solve "+solved.status+"; see "+str(root))
        prepared=solved.prepared.payload
        prepared.row["joint_common_geometry_status"]="certified_by_common_solve_and_same_native_prepared_world"
        prepared.row["scene_demand"]["motion_human_joint_certificate"]=detail
        prepared.row["joint_prepared_scene_required"]=True
        prepared.row.setdefault("joint_solution",{})["native_retained_family_counts"]=(
                solved.plan.metadata.get("native_retained_family_counts",{}))
        return {"profile":"phase2","row":prepared.row,"prepared_native":prepared,
                "prepared":prepared,"joint_solver_certificate":solved.certificate}
    finally:
        save_funnel();context.close()
