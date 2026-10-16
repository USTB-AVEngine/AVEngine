"""Plan unfrozen ordinary motion parents through the shared endpoint/camera core."""
from copy import copy, deepcopy
from pathlib import Path
import math
import time
import numpy as np


def endpoint_camera_routes(context, body, request, *, distance=None, diagnostics=None,
                           retained_roles=(), retained_routes=(), source_phase_s=0.):
    """Yield a camera and exact core route; endpoints precede continuous camera solving."""
    from avengine.construction.joint import route_between
    from avengine.construction.joint.routes import ConstructionError
    from avengine.construction.joint.camera_solve import PointRole, RouteRole
    from avengine.assets.roster_motion.episode import endpoint_route_proposals
    from avengine.assets.roster_motion.demand import matches_distance_request
    stats={} if diagnostics is None else diagnostics
    stats.update(endpoint_pairs=0,paths_solved=0,camera_solves=0,cameras_considered=0,
                 full_gate_attempted=0,rejections={})
    def reject(reason):
        stats['rejections'][reason]=stats['rejections'].get(reason,0)+1
    points=np.asarray(context.points_m)
    points=points[context.walkable_map.contains_many(points[:,[0,2]],'human')]
    if len(points)<2:return
    a,b=np.meshgrid(np.arange(len(points)),np.arange(len(points)),indexing='ij')
    a,b=a.ravel(),b.ravel()
    chords=np.linalg.norm((points[b]-points[a])[:,[0,2]],axis=1)
    keep=a!=b
    # Bounds come from the authored clock and the unchanged species speed band.
    if distance is not None:
        d=np.asarray(distance,float)
        edges=np.diff(np.r_[False,np.diff(d)>1e-8,False].astype(int))
        bouts=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
        lengths=[]
        for left,right in bouts:
            v=float(np.median(np.diff(d[left:right+1])*30.))
            if v>0:lengths.append((d[right]-d[left])*1.4/v)
        if lengths:keep &= chords<=min(lengths)+1e-6
    pairs=np.flatnonzero(keep)
    pairs=pairs[np.argsort(-chords[pairs],kind='stable')]
    stats['endpoint_pairs']=len(pairs)
    condition=request.get('camera_path_condition',{})
    trend=request.get('distance_trend')
    started=time.process_time()
    name='motion_'+str(len(retained_routes))
    for index in pairs:
        if time.process_time()-started>1800.:
            stats['search_status']='existing_cpu_budget_exhausted';return
        A,B=points[a[index]],points[b[index]]
        try:waypoints=route_between(context,A,B,species='human')
        except ConstructionError as exc:reject(exc.reason);continue
        stats['paths_solved']+=1
        start=end='rest_full_body'
        extra=()
        if condition.get('kind')=='enter':
            start='outside_left' if int(condition['required_entry_side'])<0 else 'outside_right'
            end='inside_full_body'
        elif condition.get('kind')=='cross_view':
            start=end='inside_full_body'
            extra=(PointRole('azimuth_sector',A,body,name+'.left',azimuth_deg=(-180.,0.)),
                   PointRole('azimuth_sector',B,body,name+'.right',azimuth_deg=(0.,180.)))
        minimum=float(request.get('minimum_window_distance_change_m',
                                  request.get('minimum_distance_change_m',.5))) if trend else 0.
        roles=(RouteRole(tuple(map(tuple,waypoints)),start,end,body,name,
            distance_change={'approaching':'nearer','receding':'farther'}.get(trend),
            min_distance_change_m=minimum,waypoint_roles=extra),)
        solved=context.solve_cameras((*retained_roles,*roles),limit=None)
        stats['camera_solves']+=1
        # Prefer answer margin and then the camera owner's continuous-yaw margin.
        solutions=sorted(solved.solutions,key=lambda s:(
            -s.role_margins.get(name,{}).get('distance_change_slack_m',0.),-s.yaw_margin_deg))
        for solution in solutions:
            camera=solution.camera
            stats['cameras_considered']+=1
            if any(not matches_distance_request(r.positions,np.arange(901)/30.,camera.position_m,
                    {k:v for k,v in req.items() if k!='camera_path_condition'})
                   for r,req in retained_routes):continue
            # The context owns the full camera grid. Only the route subquery is
            # restricted to this already selected endpoint pair; map/pathfinder
            # and original visual raycaster are the very same objects.
            pair_context=copy(context);pair_context.points_m=np.array([A,B])
            pair_context._grids={};pair_context._depth={};pair_context._key_cache={}
            pair_context.cache_dir=None
            funnel={}
            for route in endpoint_route_proposals(pair_context,camera,body,request,
                    distance=distance,source_phase_s=source_phase_s,diagnostics=funnel):
                # Preserve selected endpoint order (especially entry/exit roles).
                if not np.allclose(route.receipt['endpoint_a_m'],A):continue
                route.receipt['joint_camera_roles']={
                    'solver':'avengine.construction.joint.camera_solve',
                    'camera_position_m':list(camera.position_m),'camera_yaw_deg':camera.yaw_deg,
                    'yaw_margin_deg':solution.yaw_margin_deg,'role_margins':solution.role_margins,
                    'endpoint_first':True,'planning_only':True}
                yield camera,route,roles
            for reason,count in funnel.get('rejections',{}).items():
                stats['rejections'][reason]=stats['rejections'].get(reason,0)+count


def _apply_fields(row,working,scene,fields):
    row.update(deepcopy(fields))
    if fields.get('listener_m') is not None:
        for container in (scene,scene.setdefault('production_scenario',{})):
            container['listener_m']=deepcopy(fields['listener_m'])
            container['camera_yaw_knots']=deepcopy(fields['camera_yaw_knots'])
        yaw=math.radians(fields['camera_yaw_knots'][0]['yaw_deg'])
        scene['camera_forward']=[math.sin(yaw),0.,-math.cos(yaw)]
        row['camera_forward']=list(scene['camera_forward'])
        working['room'].setdefault('placement',{})['camera_m']=deepcopy(fields['listener_m'])
    by_asset={a['asset_id']:a['endpoint_id']for a in scene.get('assets',[])}
    # Builders expose endpoint-keyed motion requests. Match their original keys
    # via the explicit mapping retained by this producer.
    by_asset.update(fields.get('motion_plan_endpoint_ids',{}))
    requests={}
    for aid,req in fields.get('motion_requests',{}).items():
        item=deepcopy(req)
        if item.get('camera_path_condition'):
            item['camera_path_condition']['camera_forward']=scene['camera_forward']
        requests[by_asset.get(aid,aid)]=item
    scene.setdefault('production_scenario',{})['motion_requests']=requests
    for name in ('element_degradation','prepared_degradation','joint_motion_camera_plan'):
        if name in fields:scene[name]=deepcopy(fields[name])
    preferences=fields.get('native_placement_preferences_m',{})
    scene.setdefault('native_placement_preferences_m',{}).update(deepcopy(preferences))


def plan_parent_motion(row,working,scene,program,assets,inputs,root):
    """Called before input_recipe/native/RIR freeze. Existing worlds are immutable."""
    import json
    from avengine.timeline.continuous_pilot import write_json
    def read_json(path):return json.loads(Path(path).read_text())
    root=Path(root);receipt=root/'prefreeze_motion_plan.json'
    frozen=any((root/name).exists()for name in
        ('input_recipe.json','visible_native_completed.json','audio_v0_a0_completed.json','rir_cache'))
    if frozen:
        if receipt.is_file():
            saved=read_json(receipt)
            # A resume may supply the original producer row or the saved row.
            current=row.get('motion_requests',{})
            if current not in (saved['input_motion_requests'],saved['fields']['motion_requests']):
                raise ValueError('frozen motion request differs from its prefreeze plan')
            from avengine.construction.contact_policy import degrade_motion_to_standing, reconcile_degraded_program
            by_id={a['asset_id']:a for a in assets}
            for record in saved['fields'].get('element_degradation',[]):
                if record.get('replacement')=='stand':
                    asset=by_id[record['asset_id']]
                    degrade_motion_to_standing(scene,{'declaration':asset,'endpoint_id':asset['endpoint_id']},
                                               record['original_plan'],record['reason'])
            reconcile_degraded_program(scene,program,row)
            _apply_fields(row,working,scene,saved['fields'])
        return {'status':'frozen_world_retained'}
    if row.get('reuse_qualified_phase2_native_parent') or row.get('reuse_completed_native_parent'):
        return {'status':'frozen_world_retained'}
    requests=row.get('motion_requests',{})
    humans={a['asset_id']:a for a in assets if a.get('kind')in {'human','performer'}}
    wanted={k:v for k,v in requests.items()if k in humans and v.get('action','walk')=='walk'}
    if not wanted:return {'status':'no_human_motion_request'}
    from avengine.rooms.walkable_map_builder import native_static_video
    from avengine.assets.roster_motion.episode import native_endpoint_context,admitted_master,PhysicalFloor
    from avengine.assets.roster_motion.pose import pose,animation_times
    from avengine.assets.roster_motion.contact import foot_masks,foot_min
    from avengine.assets.roster_motion.navmesh_execution import human_distance_driven_clock
    from avengine.capture.v7_registered_native import loaded_mesh
    from avengine.construction.joint import BodySpec
    from avengine.construction.contact_policy import degrade_motion_to_standing
    video=None
    original=deepcopy(requests)
    fields={'motion_requests':deepcopy(requests),'motion_plan_endpoint_ids':
            {a['asset_id']:a['endpoint_id']for a in assets},
            'native_placement_preferences_m':deepcopy(row.get('native_placement_preferences_m',{}))}
    report={'status':'planned','schema':'joint_motion_camera_plan_v1','actors':{},'native_full_render_verified':False}
    retained_roles=[];retained_routes=[]
    try:
        video=native_static_video(working,inputs);video.width=1280;video.height=720
        context=native_endpoint_context(video,scene,working,inputs,root/'motion_planning')
        # Height is solved alongside positions/yaw using the camera owner's API.
        floor=working['room']['measured_floor_y_m']
        known=[row['listener_m'],working['room'].get('preferred_listener_m'),
               working['room'].get('placement',{}).get('camera_m')]
        context.camera_heights_m=tuple(sorted({float(point[1])-floor for point in known if point is not None}))
        for aid,request in wanted.items():
            asset=humans[aid];entry={'declaration':asset,'endpoint_id':asset['endpoint_id'],
                                    'prepared_glb':asset['habitat_glb']}
            d,b,action,_=admitted_master(entry,inputs,working)
            ts=animation_times(d,b,action);masks=foot_masks(d,b);double=[]
            for t in ts:
                parts,_=pose(d,b,action,float(t),ground=False)
                if max(abs(foot_min(parts,masks,s)[0])for s in ('L','R'))<.025:double.append(float(t))
            if not double:double=[float(ts[0])]
            _,_,_,distance,_,_=human_distance_driven_clock(ts,double,float(request['speed_m_s']),
                request.get('intervals_s',[[2.,6.],[16.,20.]]))
            local=np.asarray(loaded_mesh(asset['habitat_glb']).vertices)
            body=BodySpec('human',float(np.ptp(local[:,[0,2]],axis=0).max()/2),
                          float(np.ptp(local[:,1])),0.)
            stats={};report['actors'][aid]=stats
            for camera,route,roles in endpoint_camera_routes(context,body,request,distance=distance,
                    diagnostics=stats,retained_roles=retained_roles,retained_routes=retained_routes,
                    source_phase_s=min(double)):
                stats['full_gate_attempted']+=1
                try:
                    ground=PhysicalFloor(video,floor)
                    positions=ground.ground_roots(route.positions)
                    snapped=np.asarray([video.pathfinder.snap_point(p)for p in positions])
                    error=float(np.max(np.linalg.norm((snapped-positions)[:,[0,2]],axis=1)))
                    if not np.isfinite(error) or error>.002:continue
                except ValueError as exc:
                    stats.setdefault('native_rejections',[]).append(str(exc));continue
                fields['listener_m']=list(camera.position_m)
                fields['camera_yaw_knots']=[{'time_s':t,'yaw_deg':camera.yaw_deg}for t in (0.,30.)]
                fields['motion_requests'][aid]['joint_endpoint_route']={
                    'positions_m':positions.tolist(),'headings_rad':(-route.yaws-math.pi/2).tolist(),
                    'metadata':{**route.receipt,'contact_quality_observation':ground.observation()}}
                fields['native_placement_preferences_m'][aid]=positions[0].tolist()
                retained_roles.extend(roles);retained_routes.append((route,request))
                stats.update(status='route_found',maximum_navmesh_horizontal_error_m=error,
                    camera_position_m=list(camera.position_m),camera_yaw_deg=camera.yaw_deg,
                    endpoint_a_m=route.receipt['endpoint_a_m'],endpoint_b_m=route.receipt['endpoint_b_m'])
                break
            else:
                # Existing element/family deferral records, without inventing motion.
                record=degrade_motion_to_standing(scene,entry,request,'endpoint_camera_domain_exhausted')
                fields['motion_requests'].pop(aid,None)
                stats.update(status='standing_degradation',element_degradation=record)
        if scene.get('element_degradation'):
            fields['element_degradation']=deepcopy(scene['element_degradation'])
        from avengine.construction.contact_policy import reconcile_degraded_program
        reconcile_degraded_program(scene,program,row)
        for key in ('source_classes','silent_asset_ids','phase2_include_footsteps_asset_ids'):
            if key in row:fields[key]=deepcopy(row[key])
        fields['joint_motion_camera_plan']=report
        _apply_fields(row,working,scene,fields)
        write_json(receipt,{'input_motion_requests':original,'fields':fields})
        return report
    finally:
        if video is not None:video.sim.close()
