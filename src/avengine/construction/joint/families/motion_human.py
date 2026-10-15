"""Typed motion witnesses for the two-speaker native-NavMesh family adapter.

The common cached-map certificate remains distinct: a cached map rejection is
not relabelled as a native pass. The stock native floor/skin stage still owns
physical admission of every walking body.
"""
from dataclasses import asdict
import numpy as np
from avengine.construction.joint.types import ActorPlan,BodySpec,CameraPose,Certificate,ScenePlan,SoundEvent
from avengine.construction.joint.certification import motion_witness_checks


def scene_plan_for_row(row,room):
    camera=CameraPose(tuple(row['listener_m']),float(row['camera_yaw_deg']),width=1280,height=720)
    actors={};emitters={}
    for aid,request in row['motion_requests'].items():
        route=request['motion_quartet_route'];xz=np.asarray(route['positions_xz_m'],float)
        roots=np.column_stack((xz[:,0],np.full(len(xz),float(room['measured_floor_y_m'])),xz[:,1]))
        window=route['short_window_walk']
        actors[aid]=ActorPlan(aid,roots,BodySpec.for_species('human'),np.asarray(route['travel_yaws_rad']),
            metadata={'walk_intervals_s':[[window['walk_start_s'],window['walk_end_s']]],'ramp_s':window.get('ramp_s',.3),
                      'navigation_authority':'actual registered native NavMesh; cached map preference may be bypassed'})
        emitters[aid]=np.column_stack((xz[:,0],np.full(len(xz),camera.position_m[1]+.3),xz[:,1]))
    control=row.get('motion_wrong_emitter_control')
    if control:
        aid=control['asset_id'];roots=np.tile(np.asarray(control['root_m'],float),(901,1))
        actors[aid]=ActorPlan(aid,roots,BodySpec.for_species('human'),metadata={'silent_negative_control':True})
        emitters[aid]=roots
    for aid in row['asset_ids']:
        if aid in actors:continue
        roots=np.tile(np.asarray(row['native_placement_preferences_m'][aid],float),(901,1))
        actors[aid]=ActorPlan(aid,roots,BodySpec.for_species('human'),metadata={'static_authored_companion':True})
        emitters[aid]=roots.copy();emitters[aid][:,1]=camera.position_m[1]+.3
    sounds=[]
    durations=dict(zip(row['semantic_human_asset_ids'],row['motion_quartet_construction']['utterance_durations_s']))
    for aid,starts in row['phase2_source_event_schedule_s'].items():
        for start in starts:
            end=float(start)+float(durations[aid])
            sounds.append(SoundEvent(aid,row['proposed_sound_asset_ids'][aid],'speech_playback',float(start),end,end+.5))
    return ScenePlan(camera,actors,tuple(sounds),metadata={'motion_condition_plan':row['motion_during_condition_plan'],
        'planned_emitter_positions_by_asset':emitters,'native_readback_required':True})


def motion_certificate(condition_plan,emitter_tracks,listener_m):
    return Certificate(motion_witness_checks(condition_plan,emitter_tracks,listener_m))


def certificate_record(certificate):
    return {'authority':certificate.authority,'passed':certificate.passed,
            'checks':[asdict(c)for c in certificate.checks],
            'scope':'original motion witnesses only; common map certificate and native skin/floor/pixels are separate'}

# The production adapter below keeps proposal geometry separate from proof.
# solve owns common checks, strict preparation and every candidate decision.

def core_corridor_proposals(context, seed_route, room, pathfinder, *, first_branch='nearer',
        second_duration_s=1.6, require_negative_control=True, pair_mode='opposing',second_offset_s=15.):
    """Family radial witnesses around shared-core fixed-clock route proposals."""
    from copy import deepcopy
    from itertools import chain
    from avengine.timeline.v7_two_human_motion import single_bout_at,negative_control_bystander
    from avengine.construction.joint import paired_parallel_routes,independent_straight_pair_routes
    local_start=round(float(seed_route['emission_slot']['start_s'])-float(seed_route['emission_slot']['offset_s']),1)
    first=single_bout_at(seed_route,start_s=local_start,offset_s=0.,
        reverse=seed_route['emission_slot']['branch']!=first_branch)
    second=single_bout_at(first,start_s=float(second_offset_s)+local_start,offset_s=float(second_offset_s),
        reverse=pair_mode=='opposing',duration_s=second_duration_s)
    second_branch=second['emission_slot']['branch']
    def actor(aid,route):
        xz=np.asarray(route['positions_xz_m'],float);w=route['short_window_walk'];floor=float(room['measured_floor_y_m'])
        ray=getattr(context,'floor_raycaster',None)
        if ray is not None:
            distance,_=ray.first_hit([[xz[0,0],floor+.2,xz[0,1]]],[[0.,-1.,0.]])
            if np.isfinite(distance[0])and 0<=distance[0]<=.85:floor=floor+.2-float(distance[0])
        return ActorPlan(aid,np.column_stack((xz[:,0],np.full(901,floor),xz[:,1])),
            BodySpec.for_species('human'),np.asarray(route['travel_yaws_rad'],float),
            metadata={'walk_intervals_s':[[w['walk_start_s'],w['walk_end_s']]],'ramp_s':w.get('ramp_s',.3)})
    a,b=actor('first',first),actor('second',second)
    centre=np.mean(a.positions_m[[0,-1]],axis=0)-np.asarray(seed_route['listener_m'])
    seed_yaw=seed_route.get('camera_yaw_deg',float(np.degrees(np.arctan2(centre[0],-centre[2]))))
    camera=CameraPose(tuple(seed_route['listener_m']),seed_yaw,width=1280,height=720)
    parallel=paired_parallel_routes(context,a,b,camera=camera)if pair_mode=='opposing'else iter(())
    diagnostics={}
    independent=independent_straight_pair_routes(context,a,b,camera=camera,second_branch=second_branch,
        minimum_pair_distance_m=1.2,max_origins=48,max_evaluations=384,max_candidates=16,budget_s=20.,
        diagnostics=diagnostics)
    for planned in chain(parallel,independent):
        converted=[];valid=True
        for key,original,branch in [('first',first,first_branch),('second',second,second_branch)]:
            target=planned.actors[key];route=deepcopy(original)
            route['positions_xz_m']=np.asarray(target.positions_m)[:,[0,2]].tolist()
            route['travel_yaws_rad']=np.asarray(target.headings_rad).tolist()
            route['short_window_walk']['heading_rad']=float(target.headings_rad[0])
            route['camera_yaw_deg']=float(planned.camera.yaw_deg)
            route['minimum_camera_distance_m']=1.5
            route['rest_pose_framing']=deepcopy(planned.metadata['rest_framing'])
            route['core_route_construction']=planned.metadata['construction']
            slot=route['emission_slot'];slot['branch']=branch
            radii=np.linalg.norm(np.asarray(route['positions_xz_m'])-np.asarray(camera.position_m)[[0,2]],axis=1)
            i,j=round(slot['start_s']*30),round(slot['end_s']*30);trim=round(.2*(j-i))
            values=radii[i+trim:j-trim+1];sign=-1 if branch=='nearer'else 1
            if sign*(values[-1]-values[0])<.5 or np.any(sign*np.diff(values)<-.005):
                valid=False;break
            slot['radial_change_m']=float(values[-1]-values[0]);converted.append(route)
        if not valid:continue
        control=negative_control_bystander(room,pathfinder,seed_route['listener_m'],
            planned.camera.yaw_deg,converted)if require_negative_control else None
        if require_negative_control and control is None:continue
        yield *converted,{'pair_mode':pair_mode,'minimum_pair_distance_m':planned.metadata['minimum_pair_distance_m'],
            'rest_framing':planned.metadata['rest_framing'],'negative_control':control,
            'core_route_construction':planned.metadata['construction'],
            'core_geometry_screen':deepcopy(planned.metadata['common_screen']),
            'core_proposal_diagnostics':deepcopy(diagnostics)}

def bystander_native_yaw(row, root, listener):
    seed=int(row['scene_seed'])
    yaw=(seed%6283)/1000.-np.pi
    if (row.get('human_motion_pair_mode')=='following' or row.get('human_single_walker_fallback')) and seed%2:
        # Both approaching walkers can face the camera; solve the quiet person's
        # random orientation in a side/back domain before physical preparation.
        away=np.asarray(root,float)[[0,2]]-np.asarray(listener,float)[[0,2]]
        yaw=-np.arctan2(away[1],away[0])-np.pi/2+((seed%10001)/10000.-.5)*2*np.pi/3
    return float(yaw)



def single_walker_fallback_row(row):
    """Keep both real voices and every person; promise only A's four witnesses."""
    from copy import deepcopy
    result=deepcopy(row);first,second=result['semantic_human_asset_ids']
    result['motion_requests'].pop(second)
    result['motion_during_condition_requests']=[p for p in result['motion_during_condition_requests']if p['target_asset_id']==first]
    result['motion_during_condition_plan']['plans']=[p for p in result['motion_during_condition_plan']['plans']if p['target_asset_id']==first]
    result['primary_motion_speech_anchors_s']={first:result['primary_motion_speech_anchors_s'][first]}
    result['human_motion_event_planned_verification']=[p for p in result['human_motion_event_planned_verification']if p['asset_id']==first]
    result.update(human_single_walker_fallback=True,human_motion_event_count=1,human_motion_strategy='one_walker_real_static_speaking_companion')
    result['motion_quartet_construction']['routes'].pop(second)
    result['motion_quartet_construction']['independent_walking_speakers']=1
    result['motion_quartet_construction']['planned_verification']=result['human_motion_event_planned_verification']
    result['motion_quartet_construction']['fallback_scope']='one walking-speaking target per family; static companion answers are not added to motion quotas'
    result['motion_quartet_construction']['revision']+='-single-walker'
    return result


class HumanMotionPlugin:
    def __init__(self, base_row, state, catalog, semantic, records, contract, house, attempt=0):
        self.base_row=base_row;self.state=state;self.catalog=catalog
        self.semantic=semantic;self.records=records;self.contract=contract
        self.house=house;self.attempt=attempt

    def candidates(self, context, request):
        from avengine.timeline.v7_motion_quartet_planning import plan_room_motion_quartet
        from avengine.timeline.v7_family_clock import build_constructed_scene
        from avengine.construction.joint import SoundEvent
        from copy import deepcopy
        from avengine.timeline.v7_two_human_motion import two_human_motion_template
        if self.base_row.get('human_motion_seed_hint_root'):
            from pathlib import Path
            import json
            for path in sorted(Path(self.base_row['human_motion_seed_hint_root']).glob('*.json')):
                saved=json.loads(path.read_text())
                if(saved.get('house')==self.base_row['house']and saved.get('room_id')==self.base_row['room_id']and saved.get('scene_seed')==self.base_row['scene_seed']):
                    key=saved['semantic_human_asset_ids'][0]
                    self.base_row['precomputed_human_radial_seed']=saved['motion_requests'][key]['motion_quartet_route']
                    self.base_row['geometric_seed_provenance']={'path':str(path),'scope':'proposal only; common/family/native checks rerun'}
                    break
        def shared_constructor(*args,**kwargs):
            return two_human_motion_template(*args,joint_context=context,**kwargs)
        row=plan_room_motion_quartet(self.base_row,self.state,context.room,self.catalog,
            self.semantic,self.records,context.pathfinder,attempt=self.attempt,contract=self.contract,constructor=shared_constructor)
        if self.base_row.get('human_single_walker_fallback'):row=single_walker_fallback_row(row)
        plan=scene_plan_for_row(row,context.room)
        row['joint_motion_witness_certificate']=certificate_record(motion_certificate(row['motion_during_condition_plan'],plan.metadata['planned_emitter_positions_by_asset'],plan.camera.position_m))
        by_id={a['asset_id']:a for a in self.catalog}
        # compile_walking uses this original scan vertical ray at the route's
        # first root, then holds that root Y. Solve that Y before native execution.
        ray=context.floor_raycaster
        for aid,actor in plan.actors.items():
            if aid in row['motion_requests']:
                first=np.asarray(actor.positions_m[0],float)
                distance,_=ray.first_hit([first+[0,.2,0]],[[0.,-1.,0.]])
                if not np.isfinite(distance[0]) or not 0<=distance[0]<=.85:
                    continue
                actor.positions_m[:,1]=first[1]+.2-float(distance[0])
                actor.metadata['native_yaw_rad']=-float(actor.headings_rad[0])-np.pi/2
                actor.metadata['original_scan_root_floor_m']=float(actor.positions_m[0,1])
            else:
                # The silent real person's orientation is fixed before prepare.
                actor.metadata['native_yaw_rad']=bystander_native_yaw(row,actor.positions_m[0],plan.camera.position_m)
                from avengine.construction.joint import floor_contact_variants
                from avengine.capture.v7_visible_mesh import indexed_static_mesh
                try:
                    plan.actors[aid]=next(floor_contact_variants(context,actor,
                        indexed_static_mesh(by_id[aid]['habitat_glb']),asset=by_id[aid]))
                except (ValueError,StopIteration):
                    pass
        # Use the unchanged source compiler's exact ticks/bindings rather than
        # the floating TTS header duration as the declared sound support.
        _,scene,program,assets=build_constructed_scene(row,self.house,self.records,self.catalog,self.semantic)
        bindings={e['event_id']:e for e in program['events']}
        sounds=[]
        for event in scene['events']:
            aid=scene['sources'][event['endpoint_id']]['asset_id']
            sounds.append(SoundEvent(aid,bindings[event['event_id']]['sound_asset_id'],
                event['sound_class'],event['start_tick']/10.,event['end_tick']/10.,
                min(30.,event['end_tick']/10.+.5)))
        plan.sounds=tuple(sounds)
        plan.metadata.update(family_row=deepcopy(row),
            roles={aid:('inside' if aid in row['semantic_human_asset_ids'] else 'any')for aid in plan.actors})
        yield plan

    def verify(self, context, request, plan):
        from avengine.construction.joint import CheckResult
        row=plan.metadata['family_row']
        checks=motion_witness_checks(row['motion_during_condition_plan'],
            plan.metadata['planned_emitter_positions_by_asset'],plan.camera.position_m)
        yield from checks
        expected='nearer'if int(self.base_row['scene_seed'])%2 else'farther'
        actual=row['motion_requests'][row['semantic_human_asset_ids'][0]]['motion_quartet_route']['emission_slot']['branch']
        second=(row['motion_requests'][row['semantic_human_asset_ids'][1]]['motion_quartet_route']['emission_slot']['branch']if len(row['motion_requests'])==2 else None)
        wanted_second=expected if self.base_row.get('human_motion_pair_mode')=='following'else('farther'if expected=='nearer'else'nearer')
        yield CheckResult('motion_human:requested_radial_branch','pass'if actual==expected and (second is None or second==wanted_second)else'fail',
            evidence={'requested':[expected,wanted_second],'planned':[actual,second]})
        yield CheckResult('motion_human:complete_requested_motion_witness_sets',
            'pass'if len(checks)==4*len(row['motion_requests'])and all(c.status=='pass'for c in checks)else'fail')


def native_motion_obligations(video, scene, entries, condition_plan=None):
    """Re-run the original native verifier on all actual mouth/root readbacks."""
    from copy import deepcopy
    from avengine.construction.joint import CheckResult, native_full_mesh_sweep
    from avengine.timeline.v7_motion_during_conditions import verify_motion_during_native
    actual=deepcopy(scene);admissions={}
    for entry in entries:
        ep=entry['endpoint_id'];record=entry.get('motion_episode')
        if record:
            frames=record.get('native_30Hz_readbacks',[])
            if len(frames)!=901:
                yield CheckResult('native:motion_human:mouth_readbacks','fail','missing_actual_901_mouth_samples')
                return
            actual['sources'][ep]['trajectory']=[{'time_s':i/30.,'position_m':r['emitter_position_m']}for i,r in enumerate(frames)]
            admissions[ep]=record
        elif ep in actual['sources']:
            point=video._apply_entry(entry,scene,frame_index=0)['emitter_position_m']
            actual['sources'][ep]['trajectory']=[{'time_s':0.,'position_m':list(point)},{'time_s':30.,'position_m':list(point)}]
    plan=condition_plan or scene.get('motion_during_condition_plan') or (scene.get('production_scenario')or{}).get('motion_during_condition_plan')
    if not plan:
        yield CheckResult('native:motion_human:original_witnesses','unknown','missing_actual_motion_condition_plan')
        return
    observed=verify_motion_during_native({'motion_during_condition_plan':plan},actual,
        {'native_motion_admissions':admissions})
    video.receipt['motion_human_joint_original_witnesses']=observed
    yield CheckResult('native:motion_human:original_witnesses',observed['status'],
        'unchanged_verify_motion_during_native_on_actual_mouths_and_roots',evidence=observed)
    if observed['status']=='pass':
        yield from native_full_mesh_sweep(video,entries)

def native_view_roles_for_motion(row):
    roles={aid:{'view':'inside','static':aid not in row['motion_requests'],'required_visible_sometime':True}for aid in row['semantic_human_asset_ids']}
    control=row['motion_wrong_emitter_control'];aid=control['asset_id'];view=control['frame_state']
    if view not in {'inside','outside'}:raise ValueError('quiet human must be entirely inside or outside')
    roles[aid]={'view':view,'static':True,'required_visible_sometime':view=='inside',
        'real_geometry_and_native_projection_required':True}
    return roles


def construct_motion_human_scene(room,house,state,catalog,records,semantic,contract,runtime,
        *,scene_id,seed,family='QA-06',attempt=0):
    """Formal single-entry solve and same-prepared delivery, with no legacy fallback."""
    from pathlib import Path
    from copy import deepcopy
    import json
    from dataclasses import asdict
    from avengine.construction.joint import build_room_context,FamilyRequest,solve,prepare_native
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable
    from avengine.timeline.v7_production_inputs import attach_runtime_contract,validate_recipe
    original=next(r for r in house['checked_rooms']if r['room_label']==room['room_id'])
    base={'schema':'v7-visible-registered-pilot-phase2-v1','split':'train','house':house['house'],
        'room_id':room['room_id'],'scene_seed':seed,'scene_id':scene_id,
        'motion_source_type':'human_speech','human_motion_event_count':2,'human_motion_strategy':'two_walkers',
        'human_motion_pair_mode':runtime.get('motion_human_pair_mode','opposing'),
        'human_second_motion_offset_s':float(runtime.get('motion_human_second_offset_s',20.)),
        'human_single_walker_fallback':int(runtime.get('motion_human_walker_count',2))==1,
        'human_motion_seed_hint_root':runtime.get('motion_human_seed_hint_root'),
        'motion_requested_families':['QA-06','QA-15','QA-16','QA-17'],
        'question_quota_base':10,'questions_per_window':18,'window_count':5,'members':['v0_a0'],'trial_only':True}
    root=runtime.get('motion_human_native_oracle_root')
    if not root:raise ConstructionUnavailable('motion_human requires caller-owned native oracle output')
    context=build_room_context(original,house=house,runtime=runtime,
        cache_dir=runtime.get('motion_human_joint_cache_dir'),width=1280,height=720,
        camera_heights_m=(1.1,),allow_native_map_fallback=True)
    plugin=HumanMotionPlugin(base,state,catalog,semantic,records,contract,house,attempt)
    def make_row(plan):
        row=deepcopy(plan.metadata['family_row'])
        row['native_view_roles']=native_view_roles_for_motion(row)
        row['joint_common_geometry_status']='awaiting_common_core_and_native_preflight'
        row=attach_runtime_contract(row,runtime);validate_recipe(row,catalog,contract)
        return row
    def oracle(context,request,plan):
        return prepare_native(context,request,plan,row_factory=make_row,house=house,catalog=catalog,
            records=records,semantic=semantic,runtime=runtime,output_dir=Path(root)/scene_id,
            roles=plan.metadata['roles'],minimum_visible_pixels={aid:100 for aid in plan.metadata['family_row']['motion_requests']},
            family_check=lambda video,scene,entries:native_motion_obligations(video,scene,entries,plan.metadata['family_row']['motion_during_condition_plan']))
    try:
        request=FamilyRequest(family,('one_walker_'if base['human_single_walker_fallback']else'two_walkers_')+base['human_motion_pair_mode']+'_'+('nearer'if seed%2 else'farther'),
            (),{'require_native_preflight':True})
        solved=solve(context,request,plugin,prepare_and_verify=oracle,
            budget_s=float(runtime.get('motion_human_joint_budget_s',900.)),
            max_candidates=1,max_native_preparations=1)
    finally:context.close()
    if solved.status!='certified':
        detail={'status':solved.status,'reasons':list(solved.reasons),'diagnostics':solved.diagnostics,
            'checks':[asdict(c)for c in solved.certificate.checks]if solved.certificate else[]}
        destination=Path(root)/scene_id/'joint_failure.json';destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(detail,default=lambda x:x.tolist()if hasattr(x,'tolist')else str(x),indent=2)+'\n')
        raise ConstructionUnavailable('motion_human joint solve failed; '+str(destination))
    if solved.prepared is None or solved.prepared.payload is None:
        raise RuntimeError('motion_human certified without its live prepared world')
    prepared=solved.prepared.payload;row=prepared.row
    row['joint_prepared_scene_required']=True
    row['joint_common_geometry_status']='certified_by_common_solve_and_same_native_prepared_world'
    row['scene_demand']['motion_human_joint_certificate']=json.loads(json.dumps(asdict(solved.certificate),
        default=lambda x:x.tolist()if hasattr(x,'tolist')else x.item()))
    return {'profile':'phase2','row':row,'joint_plan':solved.plan,
        'joint_solver_certificate':solved.certificate,'prepared_native':prepared,
        'prepared':prepared,'prepared_candidate':solved.prepared}
