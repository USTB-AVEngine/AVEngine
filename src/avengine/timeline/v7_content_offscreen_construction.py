"""Four native members, three whole authored voices, one real offscreen body."""
from copy import deepcopy
import math


def construct_pure_tts_offscreen_quartet(row,people,state,semantic,ctx,attempt,catalog,room,contract):
    from avengine.timeline.v7_family_constructors import eligible_humans,choose_authored,stable_seed,ConstructionUnavailable,_request
    from avengine.timeline.v7_offscreen_construction import joint_offscreen_floor
    from avengine.timeline.v7_demand import plan_registered_content_demand
    from avengine.timeline.v7_production_inputs import validate_recipe
    used={a['asset_id']for a in people};colors={a['color']for a in people}
    candidates=[a for a in eligible_humans(catalog,room,contract)if a['asset_id']not in used and a['color']not in colors]
    candidates.sort(key=lambda a:(state.get('entity_use_counts',{}).get(a['asset_id'],0),stable_seed(row['scene_seed'],a['asset_id'],attempt)))
    for extra in candidates:
        actual=deepcopy([*people[:2],extra,people[2]])
        try:epochs=choose_authored(semantic,actual[:3],state,row['scene_seed'],epoch_count=3,attempt=attempt)
        except ConstructionUnavailable:continue
        break
    else:raise ConstructionUnavailable('pure authored quartet needs a fourth legal registered body and three complete distinct fixed voices across three scenarios; no person or member is removed')
    if len(ctx['source_proposals'])<2:raise ConstructionUnavailable('whole native quartet needs the original legal room height proposals')
    ctx,yaw,inside,outside=joint_offscreen_floor(ctx,1,row['scene_seed']+attempt)
    floor_height=sum(p[1]for p in ctx['source_proposals'])/len(ctx['source_proposals'])
    points=[inside[0],inside[1],outside[0],inside[2]]
    positions=[[p[0],floor_height,p[1]]for p in points];ids=[a['asset_id']for a in actual];extra_id=ids[2]
    valid=[]
    for index,position in enumerate(positions):
        look=positions[1-index]if index<2 else[2*position[i]-ctx['listener_m'][i]for i in range(3)]
        delta=[look[i]-position[i]for i in range(3)]
        valid.append({'position_m':position,'body_yaw_deg':math.degrees(math.atan2(delta[0],-delta[2])),
            'status':'automatic floor proposal; actual native complete-mesh contact and pixels remain required'})
    row=deepcopy(row);row.update(schema='v7-recorded-registered-standing-v1',duration_s=30,assignment_status='admitted',
        listener_m=list(ctx['listener_m']),asset_ids=ids,silent_asset_ids=[ids[3]],source_classes={aid:['speech_playback']for aid in ids[:3]},
        semantic_human_asset_ids=ids[:3],semantic_scenario_id=epochs[0][0]['scenario_id'],
        semantic_sound_ids=[s['sound_asset_id']for s in epochs[0]],semantic_start_offsets_s=[3.,6.5,0.],
        camera_yaw_deg=yaw,camera_yaw_knots=[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}],
        seeds={key:stable_seed(row['scene_seed'],key)for key in ('appearance','audio','camera','trajectory','event_schedule','pose','qa')},
        measured_static_layout={'listener_m':ctx['listener_m'],'valid':valid,'room_floor_polygon':ctx['room_floor_polygon'],
            'admission_status':'unqualified; all four bodies require fresh native admission'},
        production_scenario={'speaker_count':3,'gender':actual[0]['visual_gender'][0].upper(),'asset_ids':ids,
            'non_speech_class':None,'category_focus_class':None,'camera_yaw_deg':yaw,'camera_policy':'hold',
            'semantic_scenario_id':epochs[0][0]['scenario_id'],'semantic_sound_ids':[s['sound_asset_id']for s in epochs[0]],
            'semantic_sound_subset_ids':[s['sound_asset_id']for s in epochs[0]],
            'semantic_epoch_sound_ids':[[s['sound_asset_id']for s in values]for values in epochs],
            'semantic_role_start_offsets_s':[3.,6.5,0.],'environment_sound_classes':[]},
        native_view_roles={extra_id:{'view':'outside','static':True,'minimum_abs_azimuth_deg':60.,'real_registered_drawable_required':True}},
        offscreen_construction={'revision':'pure-authored-offscreen-quartet-v1','offscreen_asset_ids':[extra_id],
            'visible_asset_ids':[ids[0],ids[1],ids[3]],'all_waveforms_author_tts':True,'all_waveforms_complete':True,
            'speaker_count_whole_world':3,'audio_only_points':0,'first_sounding_view_requested':'outside'})
    row['automatic_construction'].update(family='V7-CONTENT-BIND',constructor='construct_pure_tts_offscreen_quartet',required_role_count=4)
    _request(row,state,'V7-CONTENT-BIND')
    row=plan_registered_content_demand(row,state,actual,semantic)
    row['production_scenario']['camera_policy']='hold'
    row['production_scenario']['binding_camera_yaw_offset_deg']=0.
    row['scene_demand'].update(target_family='V7-CONTENT-BIND',required_independent_family_counts={'V7-CONTENT-BIND':3},
        required_distinct_semantic_direction_sectors=False)
    validate_recipe(row,catalog,contract)
    return {'profile':'registered','row':row}
