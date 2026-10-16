"""QA-14: solve two visible distances and a real offscreen TTS source jointly."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import math
import numpy as np


def native_distance_witness(video,scene,entries,pair):
    """Original QA-14 needs two visible appearances separated by 0.5m."""
    from avengine.construction.joint import CheckResult
    actual={e['declaration']['asset_id']:video._apply_entry(e,scene,frame_index=0)['emitter_position_m']for e in entries}
    distances=[float(np.linalg.norm(np.asarray(actual[aid])-np.asarray(scene['listener_m'])))for aid in pair]
    gap=abs(distances[0]-distances[1])
    return (CheckResult('qa14:native_original_distance_margin','pass'if gap>=.5 else'fail',evidence={'actual_distances_m':distances,'difference_m':gap,'original_minimum_m':.5}),)


def visual_distance_row(room,plan,people,epochs,schedule,state,contract,catalog,*,scene_id,seed):
    from avengine.timeline.v7_family_constructors import _request
    from avengine.timeline.v7_production_inputs import validate_recipe
    ids=[a['asset_id']for a in people];speaker=ids[2];yaw=float(plan.camera.yaw_deg)
    row={'schema':'v7-visible-registered-pilot-phase2-v1','scene_id':scene_id,'independent_parent_scene_id':scene_id,'scene_seed':seed,
        'house':room['house'],'room_id':room['room_id'],'split':'train','family':room['family'],
        'listener_m':list(plan.camera.position_m),'camera_yaw_deg':yaw,'camera_yaw_knots':[{'time_s':t,'yaw_deg':yaw}for t in (0.,30.)],
        'asset_ids':ids,'silent_asset_ids':ids[:2],'source_classes':{speaker:['speech_playback']},'motion_requests':{},
        'members':['v0_a0'],'trial_only':True,'joint_prepared_scene_required':True,'acceptance_allow_partial_families':True,
        'window_count':5,'acceptance_window_starts_s':[0.,5.,10.,15.,20.],'questions_per_window':14,'question_quota_base':0,
        'semantic_scenario_id':epochs[0][0]['scenario_id'],'semantic_sound_ids':[epochs[0][0]['sound_asset_id']],
        'semantic_human_asset_ids':[speaker],'semantic_start_offsets_s':[schedule[0]['start_tick']/10],
        'speech_schedule':deepcopy(schedule),'speech_epoch_sound_ids':[[e[0]['sound_asset_id']]for e in epochs],
        'native_view_roles':{aid:{'view':'outside'if aid==speaker else'inside','static':True,'required_visible_sometime':aid!=speaker}for aid in ids},
        'automatic_construction':{'revision':'qa14-joint-current-room-v1','family':'QA-14','constructor':'construct_visual_distance_joint',
            'room_key':room.get('key'),'scene_template_files_read':[],'required_role_count':3,'visible_distance_pair_asset_ids':ids[:2],
            'old_absolute_distance_bands_removed':True,'real_offscreen_source_asset_ids':[speaker]},
        'owner_social_layout':{'minimum_human_center_distance_m':1.,'minimum_human_camera_distance_m':1.5,'human_orientation':'seeded varied initial headings'}}
    _request(row,state,'QA-14');validate_recipe(row,catalog,contract);return row


def construct_visual_distance_joint(room,house,state,catalog,records,semantic,contract,runtime,*,scene_id,seed,family='QA-14',attempt=0):
    from avengine.construction.joint import BodySpec,FamilyRequest,PreparedCandidate,SoundEvent,build_room_context,enumerate_joint_static_layouts,solve,static_actor
    from avengine.construction.joint.native import native_yaw,prepare_static_native
    from avengine.capture.v7_visible_attachment import rotation
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    from avengine.timeline.v7_family_constructors import select_people,choose_authored,stable_seed,ConstructionUnavailable
    from avengine.timeline.v7_speech_construction import schedule_epochs,build_speech_scene
    from avengine.timeline.continuous_pilot import write_json
    if family!='QA-14':raise ConstructionUnavailable('visual distance plugin owns QA-14')
    cache=runtime.get('joint_cache_directory')
    if not cache:raise ConstructionUnavailable('visual distance needs caller-owned joint_cache_directory')
    cache=Path(cache)/'visual_distance';allowed=[a for a in catalog if a.get('kind')!='human'or a.get('color')in {'blue','yellow','green','white','black'}]
    people=select_people(allowed,room,contract,seed,attempt=attempt);ids=[a['asset_id']for a in people]
    epochs=choose_authored(semantic,people[2:],state,seed,epoch_count=3,attempt=attempt,whole_window_s=10.)
    epochs=[deepcopy(epochs[i%len(epochs)])for i in range(3)];schedule=schedule_epochs(epochs,ids[2:])
    meshes={a['asset_id']:indexed_static_mesh(a['habitat_glb'])for a in people}
    bodies={aid:BodySpec('human',float(np.linalg.norm(np.asarray(mesh.vertices)[:,[0,2]],axis=1).max()),float(np.ptp(np.asarray(mesh.vertices)[:,1])))for aid,mesh in meshes.items()}
    headings={aid:2*math.pi*(stable_seed(seed,'visual_distance_heading',aid)%2**32)/2**32 for aid in ids}
    offsets={a['asset_id']:rotation(native_yaw(static_actor(a['asset_id'],[0.,0.,0.],bodies[a['asset_id']],facing_rad=headings[a['asset_id']]),a))@np.asarray(a['emitter_local_m'],float)for a in people}
    roles={aid:('outside'if aid==ids[2]else'inside')for aid in ids};diagnostics={}
    class Plugin:
        def candidates(self,context,request):
            for plan in enumerate_joint_static_layouts(context,bodies,roles,minimum_distance_gaps=[(ids[0],ids[1],.6)],distance_offsets_m=offsets,budget_s=float(runtime.get('visual_distance_planning_budget_s',600.)),diagnostics=diagnostics):
                for aid,actor in plan.actors.items():actor.headings_rad[:]=headings[aid]
                plan.sounds=tuple(SoundEvent(ids[2],s['sound_asset_id'],'speech_playback',s['start_tick']/10,s['end_tick']/10,min(30.,s['end_tick']/10+.5))for s in schedule)
                yield plan
        def verify(self,context,request,plan):return ()
    def prepare(context,request,plan):
        row=visual_distance_row(room,plan,people,epochs,schedule,state,contract,catalog,scene_id=scene_id,seed=seed)
        return prepare_static_native(context,request,plan,row_factory=lambda _:row,house=house,catalog=catalog,records=records,semantic=semantic,runtime=runtime,
            output_dir=cache/'native_preflight'/scene_id,roles=roles,appearance_ids=ids[:2],scene_factory=build_speech_scene,
            family_check=lambda video,scene,entries:native_distance_witness(video,scene,entries,ids[:2]))
    request=FamilyRequest('QA-14','two_visible_distances_and_real_offscreen_source',tuple(ids[:2]),{'require_native_preflight':True});actual=next(r for r in house['checked_rooms']if r['room_label']==room['room_id'])
    with build_room_context(actual,house=house,runtime=runtime,cache_dir=cache/'geometry',width=1280,height=720)as context:
        result=solve(context,request,Plugin(),prepare_and_verify=prepare,budget_s=float(runtime.get('visual_distance_planning_budget_s',600.)),max_candidates=256,max_native_preparations=8)
    (cache/'planning_results').mkdir(parents=True,exist_ok=True)
    write_json(cache/'planning_results'/(scene_id+'.json'),{'status':result.status,'reasons':result.reasons,'diagnostics':result.diagnostics,'layout_diagnostics':diagnostics,'certificate':asdict(result.certificate)if result.certificate else None})
    if result.status!='certified':raise ConstructionUnavailable('visual distance joint construction '+result.status+': '+str(result.reasons))
    payload=result.prepared.payload if isinstance(result.prepared,PreparedCandidate)else result.prepared
    return {'profile':'phase2','row':payload.row,'prepared_native':payload}
