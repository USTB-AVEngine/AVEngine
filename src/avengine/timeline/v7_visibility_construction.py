"""Construct fixed-camera entry and real-entity occlusion using native routes."""
from copy import deepcopy
import math

VISIBILITY_FAMILIES=('QA-07','QA-09','QA-11','QA-22')


def occlusion_construction_distances(camera_minimum_m=1.5,human_separation_m=1.0):
    """Derive a compact proposal from retained standoff and body spacing.

    The old 4.2m/5.3m radii were search preferences, not QA requirements.
    Full native masks and depth still decide whether occlusion happened.
    """
    fraction=.55
    minimum=math.ceil(max(camera_minimum_m/fraction,human_separation_m/(1-fraction))*10)/10
    return {'minimum_target_radius_m':minimum,'preferred_target_radius_m':max(3.5,minimum),
        'foreground_depth_fraction':fraction}


def _replace_id(value,old,new):
    if isinstance(value,dict):return {(new if k==old else k):_replace_id(v,old,new)for k,v in value.items()}
    if isinstance(value,list):return [_replace_id(v,old,new)for v in value]
    return new if isinstance(value,str)and value==old else value


def walking_body(row,target,catalog,room,contract,state,seed):
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable,stable_seed
    from avengine.timeline.v7_production_inputs import asset_category
    assets={a['asset_id']:a for a in catalog};original=assets[target]
    if original.get('walk_allowed'):return row,target
    used_colors={assets[aid].get('color')for aid in row['asset_ids']if aid!=target and assets[aid].get('controlled_shirt')}
    excluded=contract.get('policy',{}).get('temporarily_excluded_asset_ids',{})
    candidates=[a for a in catalog if a.get('kind')=='human'and a.get('walk_allowed')
        and a.get('visual_gender')==original.get('visual_gender')and a['asset_id']not in row['asset_ids']
        and a['asset_id']not in excluded and(not a.get('controlled_shirt')or a.get('color')not in used_colors)
        and asset_category(a,{},room['family'])in room['categories']]
    if not candidates:raise ConstructionUnavailable('required real human needs another admitted walking individual in this legal room cell')
    chosen=min(candidates,key=lambda a:(state.get('entity_use_counts',{}).get(a['asset_id'],0),stable_seed(seed,target,a['asset_id'])))
    return _replace_id(row,target,chosen['asset_id']),chosen['asset_id']


def whole_walking_speech(row,target,catalog,records,semantic,seed):
    from avengine.timeline.v7_family_constructors import whole_recorded_triplet
    assets={a['asset_id']:a for a in catalog};sounds={s['sound_asset_id']:s for s in semantic['sounds']}
    original_authored=list(row.get('semantic_human_asset_ids',[]))
    excluded={sounds[sid]['voice_reference']['speaker_id']for aid,sid in zip(original_authored,row.get('semantic_sound_ids',[]))if aid!=target}
    byrecord={r['sound_asset_id']:r for r in records}
    excluded.update(byrecord[sid]['speaker_id']for aid,sid in row.get('proposed_sound_asset_ids',{}).items()
        if aid!=target and sid in byrecord and byrecord[sid].get('speaker_id'))
    available=[r for r in records if r.get('speaker_id')not in excluded and r.get('sound_class')=='speech_playback'
        and .5<=r.get('duration_s',0)<=2.5]
    gender='F'if assets[target]['visual_gender']=='female'else'M'
    clips=whole_recorded_triplet(available,gender,seed)
    if target in original_authored:
        i=original_authored.index(target)
        for key in ('semantic_human_asset_ids','semantic_sound_ids','semantic_start_offsets_s'):row[key].pop(i)
        row['phase2_authored_speech_schedule_s'].pop(target,None)
    row['source_classes'][target]=['speech_playback']
    row.setdefault('proposed_sound_asset_ids',{})[target]=clips[0]['sound_asset_id']
    row.setdefault('phase2_recorded_speech_schedule_s',{})[target]=[0.,10.,20.]
    row.setdefault('phase2_recorded_sound_ids',{})[target]=[r['sound_asset_id']for r in clips]
    row['explicit_whole_walking_speech']={'target_asset_id':target,'speaker_id':clips[0]['speaker_id'],
        'sound_asset_ids':[r['sound_asset_id']for r in clips],'every_clip_complete':True,'footsteps_added':False}


def construct_visibility_scene(room,house,state,catalog,records,semantic,contract,runtime,
        *,scene_id,seed,family,attempt=0):
    from avengine.timeline.v7_offscreen_construction import construct_offscreen_scene
    from avengine.timeline.v7_family_constructors import _request,ConstructionUnavailable
    from avengine.timeline.v7_production_inputs import validate_recipe
    if family not in VISIBILITY_FAMILIES:raise ConstructionUnavailable('no explicit visibility route constructor for '+family)
    plan=construct_offscreen_scene(room,house,state,catalog,records,semantic,contract,runtime,
        scene_id=scene_id,seed=seed,family='QA-08',attempt=attempt,offscreen_people=1,
        first_sounding_view='outside'if family in {'QA-07','QA-22'}else'inside',
        visible_people=2 if family in {'QA-07','QA-22'}else 3)
    row=plan['row'];enter=family in {'QA-07','QA-22'}
    target=row['offscreen_construction']['offscreen_asset_ids'][0]if enter else row['semantic_human_asset_ids'][0]
    row,target=walking_body(row,target,catalog,room,contract,state,seed);plan['row']=row
    whole_walking_speech(row,target,catalog,records,semantic,seed)
    yaw=row['camera_yaw_deg'];angle=math.radians(yaw)
    view={'kind':'enter'if enter else'cross_view','camera_forward':[math.sin(angle),0.,-math.cos(angle)],
        'initial_outside_minimum_abs_deg':50.,'final_inside_maximum_abs_deg':35.,
        'minimum_angular_span_deg':20.,'maximum_crossing_abs_deg':32.,
        'minimum_crossing_radius_m':4.2,'required_entry_side':(-1 if seed%2 else 1)}
    compact=occlusion_construction_distances()
    if not enter:view['minimum_crossing_radius_m']=compact['minimum_target_radius_m']
    intervals=[[3.,4.5]]if enter else[[3.,4.],[6.,7.]]
    row['motion_requests']={target:{'action':'walk','speed_m_s':1.,'intervals_s':intervals,
        'minimum_listener_horizontal_distance_m':1.5,'camera_path_condition':view}}
    if enter:
        row['native_view_roles'][target].update(static=False,required_visible_sometime=True,minimum_abs_azimuth_deg=50.,
            preferred_azimuth_deg=view['required_entry_side']*75.)
        row['native_placement_preferences_m'].pop(target,None)
        row['explicit_entry']={'target_asset_id':target,'side':view['required_entry_side'],
            'outside_speech_before_motion':True,'real_body_enters_by_s':5.,'original_native_mask_entry_required':True,
            'QA22_all_speakers_visible_at_least_once_in_the_first_window':True,'original_QA22_truth_changed':False}
    else:
        occluder=row['silent_asset_ids'][0]
        row['native_dynamic_occlusion_relations']=[{'target_asset_id':target,'occluder_asset_id':occluder,
            'required_state':'fully_occluded'if family=='QA-09'else'visible_occluded',
            'query_times_s':[5.+i/15 for i in range(9)],'later_clear_time_s':8.,
            'minimum_stable_seconds':.5,'minimum_occluded_fraction':1. if family=='QA-09'else .25,
            'foreground_depth_fraction':compact['foreground_depth_fraction'],'preferred_lateral_m':0. if family=='QA-09'else .20,
            'native_mask_and_unique_registered_identity_required':True}]
        row['native_facing_offsets_deg']={occluder:0.}
        row['native_placement_preferences_m'].pop(occluder,None)
        row['native_attachment_distance_targets_m']={target:compact['preferred_target_radius_m']}
        row['native_distance_bands_m']={target:[compact['minimum_target_radius_m'],max(7.,compact['preferred_target_radius_m'])]}
        row['explicit_occlusion_motion']={'target_asset_id':target,'occluder_asset_id':occluder,
            'pause_while_blocked_s':[5.,6.],'clear_after_s':8.,'camera_moves':False,
            'every_entity_retained':True,'native_full_stream_is_the_truth':True}
    row['automatic_construction'].update(family=family,constructor='construct_visibility_scene')
    _request(row,state,family);validate_recipe(row,catalog,contract)
    return plan
