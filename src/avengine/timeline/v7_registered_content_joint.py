"""Compile a current-room authored quartet through the shared joint solver.

Only the original room, registrations and TTS manifest are inputs. No previous
parent recipe, media or hand-written scene coordinates are read.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import math
import numpy as np


def select_content_people(room,catalog,semantic,contract,state,seed,attempt=0):
    from avengine.timeline.v7_family_constructors import eligible_humans,choose_authored,stable_seed,ConstructionUnavailable
    from avengine.qa.v7_visual_admission import COLOR_HUE_DEGREES
    pool=eligible_humans(catalog,room,contract)
    pool.sort(key=lambda a:(state.get('source_entity_usage_counts',{}).get(a['asset_id'],0),stable_seed(seed,a['asset_id'],attempt)))
    groups=defaultdict(list)
    for a in pool:
        if a['color']in COLOR_HUE_DEGREES:groups[(a.get('body_identity_id',a['visual_gender']),a['visual_gender'])].append(a)
    for key in sorted(groups,key=lambda k:stable_seed(seed,'content_body',*k,attempt)):
        for left in groups[key]:
            for right in groups[key]:
                if left['color']==right['color']:continue
                for hidden in pool:
                    if hidden['color']in {left['color'],right['color']}:continue
                    quiet=next((a for a in pool if a['color']not in {left['color'],right['color'],hidden['color']}),None)
                    if quiet is None:continue
                    try:epochs=choose_authored(semantic,[left,right,hidden],state,seed,epoch_count=3,attempt=attempt,whole_window_s=10.)
                    except ConstructionUnavailable:continue
                    # The original registered clock has three epochs. Repeated
                    # admitted utterances remain explicit and dedup still applies.
                    epochs=[deepcopy(epochs[i%len(epochs)])for i in range(3)]
                    return deepcopy([left,right,hidden,quiet]),epochs
    raise ConstructionUnavailable('no legal counterfactual body pair, real offscreen speaker and silent visual witness support whole fixed-voice authored roles')


def content_role_offsets(epochs):
    """Fit all unchanged complete recordings to the original three 10s epochs."""
    ticks=[max(math.ceil(e[i]['synthesis_readback']['sample_count']/1600)for e in epochs)for i in range(3)]
    if any(n<=0 or n>100 for n in ticks):raise ValueError('complete authored role does not fit a 10 second window')
    # Put the real offscreen voice first when sequential whole clips fit. Longer
    # roles overlap naturally; their own physical source and words stay intact.
    offsets=[0.,0.,0.];order=(2,0,1)
    if sum(ticks)<=100:
        slack=(100-sum(ticks))//4;cursor=slack
        for i in order:
            offsets[i]=min(cursor,100-ticks[i],70)/10;cursor+=ticks[i]+slack
    else:
        offsets=[min((100-n)//2,70)/10 for n in ticks]
    return offsets


def content_row(room,house,state,catalog,semantic,contract,context,plan,people,epochs,*,scene_id,seed):
    from avengine.timeline.v7_family_constructors import stable_seed,_request
    from avengine.timeline.v7_demand import plan_registered_content_demand
    from avengine.timeline.v7_production_inputs import validate_recipe
    from avengine.construction.joint import registered_row_for_plan
    ids=[a['asset_id']for a in people];offsets=content_role_offsets(epochs)
    row={'schema':'v7-recorded-registered-standing-v1','scene_id':scene_id,'independent_parent_scene_id':scene_id,
        'scene_seed':seed,'house':room['house'],'room_id':room['room_id'],'split':'train','family':room['family'],
        'duration_s':30,'assignment_status':'admitted','trial_only':True,'questions_per_window':14,'question_quota_base':0,
        'asset_ids':ids,'silent_asset_ids':[ids[3]],'source_classes':{a:['speech_playback']for a in ids[:3]},'motion_requests':{},
        'semantic_human_asset_ids':ids[:3],'semantic_scenario_id':epochs[0][0]['scenario_id'],
        'semantic_sound_ids':[s['sound_asset_id']for s in epochs[0]],'semantic_start_offsets_s':offsets,
        'seeds':{k:stable_seed(seed,k)for k in ('appearance','audio','camera','trajectory','event_schedule','pose','qa')},
        'production_scenario':{'speaker_count':3,'gender':people[0]['visual_gender'][0].upper(),'asset_ids':ids,
            'non_speech_class':None,'category_focus_class':None,'camera_policy':'hold',
            'semantic_scenario_id':epochs[0][0]['scenario_id'],'semantic_sound_ids':[s['sound_asset_id']for s in epochs[0]],
            'semantic_sound_subset_ids':[s['sound_asset_id']for s in epochs[0]],
            'semantic_epoch_sound_ids':[[s['sound_asset_id']for s in epoch]for epoch in epochs],
            'semantic_role_start_offsets_s':offsets,'environment_sound_classes':[]},
        'automatic_construction':{'revision':'content-joint-current-room-v1','family':'V7-CONTENT-BIND',
            'constructor':'construct_registered_content_joint','scene_template_files_read':[],
            'required_role_count':4,'distinct_authored_scenarios':len({e[0]['scenario_id']for e in epochs})},
        'offscreen_construction':{'offscreen_asset_ids':[ids[2]],'visible_asset_ids':[ids[0],ids[1],ids[3]],
            'all_waveforms_author_tts':True,'all_waveforms_complete':True,'audio_only_points':0},
        'native_minimum_endpoint_distance_gaps':[{'endpoints':['voice1','voice2'],'minimum_difference_m':.6}],
        'native_require_complete_visible_bodies':True,'joint_prepared_scene_required':True}
    _request(row,state,'V7-CONTENT-BIND');row=plan_registered_content_demand(row,state,people,semantic)
    row['scene_demand'].update(target_family='V7-CONTENT-BIND',required_independent_family_counts={'V7-CONTENT-BIND':3},required_distinct_semantic_direction_sectors=False)
    row=registered_row_for_plan(context,row,plan,people,record_only_distance_bands=True,record_only_offscreen_angle=True)
    validate_recipe(row,catalog,contract)
    return row


def construct_registered_content_joint(room,house,state,catalog,records,semantic,contract,runtime,*,scene_id,seed,family='V7-CONTENT-BIND',attempt=0):
    from avengine.construction.joint import BodySpec,FamilyRequest,PreparedCandidate,CheckResult,build_room_context,solve,enumerate_joint_static_layouts,prepare_registered_bundle,static_actor
    from avengine.construction.joint.native import native_yaw
    from avengine.capture.v7_visible_attachment import rotation
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable,stable_seed
    from avengine.timeline.continuous_pilot import write_json
    if family!='V7-CONTENT-BIND':raise ConstructionUnavailable('registered quartet plugin owns only content binding')
    cache=runtime.get('joint_cache_directory')
    if not cache:raise ConstructionUnavailable('registered quartet needs a caller-owned joint_cache_directory')
    cache=Path(cache)/'registered_content';people,epochs=select_content_people(room,catalog,semantic,contract,state,seed,attempt)
    ids=[a['asset_id']for a in people];meshes={a['asset_id']:indexed_static_mesh(a['habitat_glb'])for a in people}
    bodies={aid:BodySpec('human',float(np.linalg.norm(np.asarray(mesh.vertices)[:,[0,2]],axis=1).max()),float(np.ptp(np.asarray(mesh.vertices)[:,1])))for aid,mesh in meshes.items()}
    headings={aid:2*math.pi*(stable_seed(seed,'content_heading',aid)%2**32)/2**32 for aid in ids}
    offsets={a['asset_id']:rotation(native_yaw(static_actor(a['asset_id'],[0.,0.,0.],bodies[a['asset_id']],facing_rad=headings[a['asset_id']]),a))@np.asarray(a['mouth_local_glb_m'],float)for a in people}
    roles={aid:('outside'if aid==ids[2]else'inside')for aid in ids};diagnostics={}
    class Plugin:
        def candidates(self,context,request):
            for plan in enumerate_joint_static_layouts(context,bodies,roles,minimum_distance_gaps=[(ids[0],ids[1],.6)],distance_offsets_m=offsets,budget_s=float(runtime.get('content_planning_budget_s',600.)),diagnostics=diagnostics):
                for aid,actor in plan.actors.items():actor.headings_rad[:]=headings[aid]
                yield plan
        def verify(self,context,request,plan):
            return (CheckResult('content:original_authored_roles','pass',evidence={'source_scenario_ids':[e[0]['scenario_id']for e in epochs],'original_QA_group_check_still_required':True}),)
    def group_check(bundle):
        checks=[]
        for name,payload in bundle.visuals.items():
            actual={entry['declaration']['asset_id']:payload.video._apply_entry(entry,payload.scene,frame_index=0)['emitter_position_m']for entry in payload.video.actors}
            distances=[np.linalg.norm((np.asarray(actual[aid])-np.asarray(payload.scene['listener_m']))[[0,2]])for aid in ids[:2]];gap=float(abs(distances[0]-distances[1]))
            checks.append(CheckResult('content:'+name+':actual_mouth_distance_witness','pass'if gap>=.5 else'fail',evidence={'gap_m':gap,'actual_mouth_distances_m':[float(d)for d in distances],'full_render_and_original_content_QA_required':True}))
        return checks
    def prepare(context,request,plan):
        row=content_row(room,house,state,catalog,semantic,contract,context,plan,people,epochs,scene_id=scene_id,seed=seed)
        return prepare_registered_bundle(context,request,plan,row_factory=lambda _:row,house=house,catalog=people,records=records,semantic=semantic,runtime=runtime,output_dir=cache/'native_preflight'/scene_id,roles_by_visual={v:roles for v in ('v0','v1')},appearance_ids=ids[:2],group_check=group_check)
    request=FamilyRequest(family,'whole_authored_four_members',tuple(ids[:2]),{'require_native_preflight':True})
    actual=next(r for r in house['checked_rooms']if r['room_label']==room['room_id'])
    with build_room_context(actual,house=house,runtime=runtime,cache_dir=cache/'geometry',width=1280,height=720)as context:
        result=solve(context,request,Plugin(),prepare_and_verify=prepare,budget_s=float(runtime.get('content_planning_budget_s',runtime.get('speech_planning_budget_s',180.))),max_candidates=int(runtime.get('content_max_candidates',256)),max_native_preparations=int(runtime.get('content_max_native_preparations',8)))
    (cache/'planning_results').mkdir(parents=True,exist_ok=True)
    write_json(cache/'planning_results'/(scene_id+'.json'),{'status':result.status,'reasons':result.reasons,'diagnostics':result.diagnostics,'layout_diagnostics':diagnostics,'certificate':asdict(result.certificate)if result.certificate else None})
    if result.status!='certified':raise ConstructionUnavailable('registered content joint construction '+result.status+': '+str(result.reasons))
    bundle=result.prepared.payload if isinstance(result.prepared,PreparedCandidate)else result.prepared
    bundle.row['joint_prepared_scene_required']=True
    return {'profile':'registered','row':bundle.row,'prepared_native':bundle}
