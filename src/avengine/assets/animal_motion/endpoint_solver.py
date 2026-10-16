"""Answer-defined animal endpoints; all navigation is delegated to joint core.

The supplied grid is the domain. There are no radial rings, headings, camera
position deduplication, or fixed limits on pairs. A caller's deadline is a
compute observation, never an infeasibility claim. Native certification remains
mandatory after these proposals.
"""
from __future__ import annotations
import heapq
import time
import numpy as np


def endpoint_pairs(context, camera, body, *, species, branch, minimum_net_change_m=.5,
                   maximum_path_length_m=None, diagnostics=None, deadline=None):
    """Rank every inside far/near pair by its answer's distance margin.

    maximum_path_length_m, when supplied, is a necessary straight-distance bound
    derived from the legal species speed and the requested event window. It is
    not a hard-coded room/route range. A* and final native motion may still fail.
    Counts distinguish the complete combinatorial domain from pairs inspected.
    """
    if branch not in {'nearer','farther'}:
        raise ValueError('choose the requested distance answer before endpoints')
    d=diagnostics if diagnostics is not None else {}
    points=np.asarray(context.points_m,float)
    grid=context.grid(camera,body)
    legal=context.walkable_map.contains_many(points[:,[0,2]],species)
    indices=np.flatnonzero(legal & (np.asarray(grid.frustum_state)=='inside'))
    distances=np.asarray(grid.distances_m,float)
    order=indices[np.argsort(distances[indices],kind='stable')]
    near_dist=distances[order]
    d['inside_endpoint_cells']=d.get('inside_endpoint_cells',0)+len(order)
    heap=[]
    for far in order[::-1]:
        count=int(np.searchsorted(near_dist,distances[far]-minimum_net_change_m,side='right'))
        if not count:continue
        d['endpoint_pairs_in_domain']=d.get('endpoint_pairs_in_domain',0)+count
        near=int(order[0]);delta=float(distances[far]-distances[near])
        heapq.heappush(heap,(-delta,int(far),0,count))
    while heap:
        if deadline is not None and time.monotonic()>=deadline:
            d['compute_budget_exhausted']=True
            return
        negative,far,rank,count=heapq.heappop(heap);near=int(order[rank])
        if rank+1<count:
            nxt=int(order[rank+1])
            heapq.heappush(heap,(-float(distances[far]-distances[nxt]),far,rank+1,count))
        d['endpoint_pairs_inspected']=d.get('endpoint_pairs_inspected',0)+1
        a,b=(far,near) if branch=='nearer' else (near,far)
        if maximum_path_length_m is not None and np.linalg.norm(points[a,[0,2]]-points[b,[0,2]])>maximum_path_length_m+1e-9:
            d['pairs_exceed_event_clock_bound']=d.get('pairs_exceed_event_clock_bound',0)+1
            continue
        evidence={'answer':branch,'start_cell':int(a),'end_cell':int(b),
            'start_role':'inside_full_body','end_role':'inside_full_body',
            'start_distance_m':float(distances[a]),'end_distance_m':float(distances[b]),
            'distance_margin_m':float(-negative-minimum_net_change_m),
            'camera_position_m':list(camera.position_m),'camera_yaw_deg':float(camera.yaw_deg),
            'domain':'all supplied species-layer inside grid cells',
            'native_truth_verified':False}
        yield points[a].copy(),points[b].copy(),evidence


def endpoint_routes(context, *, body, species, branch, entity_id, speed_m_s,
                    start_s, event_end_s, diagnostics=None, deadline=None):
    """Obtain a core route between answer endpoints, then fit legal speed/time.

    The event_end_s bound comes from the declared nonoverlapping 10-second
    target event and any retained native stationary turn. Shorter paths retain
    the requested legal speed; longer ones can use another speed in the same
    owner interval. No trajectory is rescaled or replanned after certification.
    """
    from avengine.construction.joint.routes import route_between,construct_route,SPEED_BANDS,ConstructionError
    from avengine.construction.joint.context import camera_legal
    d=diagnostics if diagnostics is not None else {}
    lo,hi=SPEED_BANDS[species]
    if not lo<=speed_m_s<=hi:raise ValueError('requested species speed is illegal')
    usable=event_end_s-start_s-.3  # the shared constructor's unchanged ramp clock
    if usable<=0:return
    for camera in context.cameras:
        if deadline is not None and time.monotonic()>=deadline:
            d['compute_budget_exhausted']=True
            return
        d['cameras_inspected']=d.get('cameras_inspected',0)+1
        if not camera_legal(context.room,camera)[0]:continue
        for a,b,evidence in endpoint_pairs(context,camera,body,species=species,branch=branch,
                maximum_path_length_m=hi*usable,diagnostics=d,deadline=deadline):
            try:
                path=route_between(context,a,b,species=species)
                d['routes_found']=d.get('routes_found',0)+1
                path_length=float(np.linalg.norm(np.diff(path[:,[0,2]],axis=0),axis=1).sum())
                chosen=max(float(speed_m_s),path_length/usable)
                if chosen>hi+1e-9:
                    d['routed_paths_exceed_event_clock']=d.get('routed_paths_exceed_event_clock',0)+1
                    continue
                chosen=min(hi,float(np.nextafter(chosen,float("inf"))))
                actor=construct_route(context,path,species=species,speed_m_s=chosen,start_s=start_s,
                    camera=camera,entity_id=entity_id,body=body)
                moving=np.flatnonzero(np.linalg.norm(np.diff(actor.positions_m[:,[0,2]],axis=0),axis=1)>1e-8)
                actual_end_s=float(moving[-1]+1)/30. if len(moving) else 0.
                if max(actor.metadata['walk_intervals_s'][-1][1],actual_end_s)>event_end_s+1e-8:
                    d['constructed_clock_exceeds_event']=d.get('constructed_clock_exceeds_event',0)+1
                    continue
            except ConstructionError as error:
                failures=d.setdefault('route_conflicts',{})
                failures[error.reason]=failures.get(error.reason,0)+1
                continue
            evidence.update(path_waypoints_m=np.asarray(path).tolist(),speed_m_s=chosen,
                walk_intervals_s=actor.metadata['walk_intervals_s'],
                solver='context.grid -> route_between -> construct_route')
            actor.metadata['animal_endpoint_solution']=evidence
            d['constructed_routes']=d.get('constructed_routes',0)+1
            yield camera,actor,evidence


def physical_endpoint_pairs(context, *, species, maximum_path_length_m, diagnostics=None, deadline=None):
    """All species-floor endpoint pairs, before choosing any camera.

    Each row is scanned lazily with linear temporary memory. The heap stores
    one pair per start cell, not an N-by-N matrix. Both trajectory directions
    are retained; camera_solve will enforce the requested distance answer.
    """
    d=diagnostics if diagnostics is not None else {}
    points=np.asarray(context.points_m,float)
    indices=np.flatnonzero(context.walkable_map.contains_many(points[:,[0,2]],species))
    xy=points[:,[0,2]];heap=[]
    def next_for(a,previous=None):
        squared=np.sum((xy[indices]-xy[a])**2,axis=1)
        mask=(indices>a)&(squared>=.5**2)&(squared<=maximum_path_length_m**2+1e-9)
        if previous is None:
            d['physical_pairs_in_domain']=d.get('physical_pairs_in_domain',0)+2*int(mask.sum())
        else:
            old,b=previous
            mask &= (squared<old)|((squared==old)&(indices>b))
        legal=np.flatnonzero(mask)
        if not len(legal):return None
        j=int(legal[np.argmax(squared[legal])]);return float(squared[j]),int(indices[j])
    for a in indices:
        if deadline is not None and time.monotonic()>=deadline:
            d['compute_budget_exhausted']=True;return
        result=next_for(int(a))
        if result is not None:heapq.heappush(heap,(-result[0],int(a),result[1]))
    while heap:
        if deadline is not None and time.monotonic()>=deadline:
            d['compute_budget_exhausted']=True;return
        negative,a,b=heapq.heappop(heap)
        nxt=next_for(a,(-negative,b))
        if nxt is not None:heapq.heappush(heap,(-nxt[0],a,nxt[1]))
        for first,last in ((a,b),(b,a)):
            d['physical_pairs_inspected']=d.get('physical_pairs_inspected',0)+1
            yield points[first].copy(),points[last].copy(),{'start_cell':first,'end_cell':last,'endpoint_separation_m':float(np.sqrt(-negative))}


def joint_camera_endpoint_routes(context, *, body, species, branch, entity_id, speed_m_s,
                                  start_s, event_end_s, diagnostics=None, deadline=None):
    """Choose physical endpoints, route them, jointly solve camera x/y/z/yaw.

    No context.cameras seed or fixed heading is required. Each solved camera
    comes from the owner's entire legal floor-grid/height/yaw role domain.
    Existing height values and horizontal pitch are inherited unchanged.
    """
    from avengine.construction.joint.camera_solve import roles_for_endpoints
    from avengine.construction.joint.routes import route_between,construct_route,SPEED_BANDS,ConstructionError
    d=diagnostics if diagnostics is not None else {};lo,hi=SPEED_BANDS[species]
    if branch not in {'nearer','farther'} or not lo<=speed_m_s<=hi:
        raise ValueError('choose a legal speed and explicit answer first')
    usable=event_end_s-start_s-.3
    if usable<=0:return
    for a,b,endpoint in physical_endpoint_pairs(context,species=species,maximum_path_length_m=hi*usable,diagnostics=d,deadline=deadline):
        try:
            path=route_between(context,a,b,species=species);d['routes_found']=d.get('routes_found',0)+1
            length=float(np.linalg.norm(np.diff(path[:,[0,2]],axis=0),axis=1).sum())
            speed=max(float(speed_m_s),length/usable)
            if speed>hi+1e-9:continue
            speed=min(hi,float(np.nextafter(speed,float('inf'))))
            actor=construct_route(context,path,species=species,speed_m_s=speed,start_s=start_s,entity_id=entity_id,body=body)
            moving=np.flatnonzero(np.linalg.norm(np.diff(actor.positions_m[:,[0,2]],axis=0),axis=1)>1e-8)
            actual_end=float(moving[-1]+1)/30. if len(moving) else 0.
            if max(actor.metadata['walk_intervals_s'][-1][1],actual_end)>event_end_s+1e-8:continue
        except ConstructionError as error:
            failures=d.setdefault('route_conflicts',{});failures[error.reason]=failures.get(error.reason,0)+1
            continue
        roles=roles_for_endpoints('approach' if branch=='nearer' else 'recede',a,b,body=body,
            min_distance_change_m=.5,waypoints_m=path,name=entity_id)
        solved=context.solve_cameras(roles,limit=None)
        d['camera_role_solves']=d.get('camera_role_solves',0)+1
        d['role_camera_solutions']=d.get('role_camera_solutions',0)+len(solved.solutions)
        if not solved.solutions:continue
        for solution in solved.solutions:
            if deadline is not None and time.monotonic()>=deadline:
                d['compute_budget_exhausted']=True;return
            from copy import deepcopy
            selected=deepcopy(actor)
            evidence={**endpoint,'answer':branch,'start_role':'inside_full_body','end_role':'inside_full_body',
                'path_waypoints_m':np.asarray(path).tolist(),'speed_m_s':speed,
                'camera_position_m':list(solution.camera.position_m),'camera_yaw_deg':solution.camera.yaw_deg,
                'camera_yaw_margin_deg':solution.yaw_margin_deg,'role_margins':solution.role_margins,
                'camera_solve_diagnostics':solved.diagnostics,'native_truth_verified':False,
                'solver':'physical endpoints -> route_between -> construct_route -> camera_solve'}
            selected.metadata['animal_endpoint_solution']=evidence
            d['constructed_routes']=d.get('constructed_routes',0)+1
            yield solution.camera,selected,evidence
