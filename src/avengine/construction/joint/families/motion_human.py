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
        row=plan_room_motion_quartet(self.base_row,self.state,context.room,self.catalog,
            self.semantic,self.records,context.pathfinder,attempt=self.attempt,contract=self.contract)
        plan=scene_plan_for_row(row,context.room)
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
                actor.metadata['native_yaw_rad']=((int(row['scene_seed'])%6283)/1000)-np.pi
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
            roles={aid:('inside' if aid in row['motion_requests'] else 'any')for aid in plan.actors})
        yield plan

    def verify(self, context, request, plan):
        from avengine.construction.joint import CheckResult
        row=plan.metadata['family_row']
        checks=motion_witness_checks(row['motion_during_condition_plan'],
            plan.metadata['planned_emitter_positions_by_asset'],plan.camera.position_m)
        yield from checks
        expected='nearer'if int(self.base_row['scene_seed'])%2 else'farther'
        actual=row['motion_requests'][row['semantic_human_asset_ids'][0]]['motion_quartet_route']['emission_slot']['branch']
        second=row['motion_requests'][row['semantic_human_asset_ids'][1]]['motion_quartet_route']['emission_slot']['branch']
        wanted_second=expected if self.base_row.get('human_motion_pair_mode')=='following'else('farther'if expected=='nearer'else'nearer')
        yield CheckResult('motion_human:requested_radial_branch','pass'if actual==expected and second==wanted_second else'fail',
            evidence={'requested':[expected,wanted_second],'planned':[actual,second]})
        yield CheckResult('motion_human:two_complete_motion_witness_sets',
            'pass'if len(checks)==8 and all(c.status=='pass'for c in checks)else'fail')


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
        row=attach_runtime_contract(row,runtime);validate_recipe(row,catalog,contract)
        return row
    def oracle(context,request,plan):
        return prepare_native(context,request,plan,row_factory=make_row,house=house,catalog=catalog,
            records=records,semantic=semantic,runtime=runtime,output_dir=Path(root)/scene_id,
            roles=plan.metadata['roles'],minimum_visible_pixels={aid:100 for aid in plan.metadata['family_row']['motion_requests']},
            family_check=lambda video,scene,entries:native_motion_obligations(video,scene,entries,plan.metadata['family_row']['motion_during_condition_plan']))
    try:
        request=FamilyRequest(family,'two_walkers_'+base['human_motion_pair_mode']+'_'+('nearer'if seed%2 else'farther'),
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
