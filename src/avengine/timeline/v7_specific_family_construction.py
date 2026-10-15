"""Explicit identification, overlap, distance and counting scene requests.

Every branch binds whole waveforms to real bodies. No answer or mask state is
assigned; the existing native producer and original QA predicates decide.
"""
from __future__ import annotations
import math

SPECIFIC_FAMILIES=('QA-02','QA-05','QA-12','QA-14','QA-18','QA-23','STARSS-CLASS13','STARSS-SPEAKERS')


def distance_comparison_proposals(row,ctx):
    from shapely.geometry import shape,Point
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable,stable_seed
    polygon=shape(ctx['room_floor_polygon']);x0,z0,x1,z1=polygon.bounds
    listener=row['listener_m'];yaw=row['camera_yaw_deg'];step=max(.25,math.sqrt(max(1.,polygon.area)/1000))
    points=[];x=x0+step/3
    while x<x1:
        z=z0+step/3
        while z<z1:
            if polygon.covers(Point(x,z)):
                distance=math.hypot(x-listener[0],z-listener[2])
                angle=(math.degrees(math.atan2(x-listener[0],-(z-listener[2])))-yaw+180)%360-180
                if distance>=1.5 and abs(angle)<=32:points.append((x,z,distance,angle))
            z+=step
        x+=step
    people=row['offscreen_construction']['visible_asset_ids']
    outside=row['offscreen_construction']['offscreen_asset_ids']
    chosen=[(row['native_placement_preferences_m'][aid][0],row['native_placement_preferences_m'][aid][2])for aid in outside]
    bands=((1.8,2.4),(4.2,6.5),(2.7,4.0));angles=(-25.,25.,0.)
    for aid,band,angle in zip(people,bands,angles,strict=True):
        candidates=[p for p in points if band[0]<=p[2]<=band[1]
            and all(math.hypot(p[0]-q[0],p[1]-q[1])>=1.05 for q in chosen)]
        if not candidates:raise ConstructionUnavailable('distance comparison needs another legal floor area/room for every near/far person; no participant is removed')
        p=min(candidates,key=lambda p:(abs(p[3]-angle)/15+abs(p[2]-sum(band)/2),stable_seed(row['scene_seed'],aid,p[0],p[1])))
        row['native_placement_preferences_m'][aid]=[p[0],listener[1]-1.5,p[1]]
        row.setdefault('native_distance_bands_m',{})[aid]=list(band);chosen.append(p[:2])


def construct_specific_scene(room,house,state,catalog,records,semantic,contract,runtime,
        *,scene_id,seed,family,attempt=0):
    from avengine.timeline.v7_offscreen_construction import construct_offscreen_scene
    from avengine.timeline.v7_family_constructors import _request,room_context,ConstructionUnavailable
    from avengine.timeline.v7_production_inputs import validate_recipe
    if family not in SPECIFIC_FAMILIES:raise ConstructionUnavailable('no explicit static constructor for '+family)
    count=1
    if family=='STARSS-SPEAKERS':
        counts=state.get('live_speaker_count_counts',{})
        requested=min((3,4),key=lambda n:(counts.get(str(n),counts.get(n,0)),(n-seed)%2))
        count=requested-2
    plan=construct_offscreen_scene(room,house,state,catalog,records,semantic,contract,runtime,
        scene_id=scene_id,seed=seed,family='QA-08',attempt=attempt,offscreen_people=count,first_sounding_view='inside',visible_people=3 if family=='QA-14'else 2)
    row=plan['row'];people=row['semantic_human_asset_ids'];lookup={s['sound_asset_id']:s for s in semantic['sounds']}
    utterances=[lookup[sid]for sid in row['semantic_sound_ids']]
    if family in {'QA-02','QA-12'}:
        texts=[r['transcript'].strip().casefold()for r in utterances]
        if len(texts)!=2 or len(set(texts))!=2:raise ConstructionUnavailable('identification needs distinct complete words from the two visible people')
        row['explicit_identification']={'appearance_asset_ids':list(people),'whole_sound_ids':list(row['semantic_sound_ids']),
            'native_appearance_and_content_binding_required':True,'outside_competitor_asset_ids':row['offscreen_construction']['offscreen_asset_ids']}
    elif family=='QA-05':
        # A and B overlap on purpose. The third voice occurs later; all
        # source/wet activity is remeasured before choosing either answer.
        row['semantic_start_offsets_s']=[0.,.1]
        row['phase2_authored_speech_schedule_s']={aid:[epoch+offset for epoch in (0.,10.,20.)]
            for aid,offset in zip(people,(0.,.1),strict=True)}
        row['explicit_overlap']={'overlapping_asset_ids':list(people),
            'later_disjoint_candidate_asset_ids':row['offscreen_construction']['offscreen_asset_ids'],
            'authority':'native wet support; no dry schedule is assigned as truth'}
    elif family=='QA-14':distance_comparison_proposals(row,room_context(room,house,attempt))
    elif family=='QA-18':
        row['explicit_activity_bins']={'early_speaker':people[0],'later_speaker':people[1],
            'silent_asset_ids':list(row['silent_asset_ids']),'actual_native_integer_second_activity_required':True}
    elif family in {'QA-23','STARSS-CLASS13','STARSS-SPEAKERS'}:
        if any(kinds!=['speech_playback']for kinds in row['source_classes'].values()):
            raise ConstructionUnavailable('count/class13 constructor requires all actual sources inside its original class domain')
        row['explicit_class13_count']={'live_speakers':2+count,'includes_offscreen_speakers':True,
            'required_sounding_asset_ids':[*people,*row['offscreen_construction']['offscreen_asset_ids']],
            'silent_asset_ids':list(row['silent_asset_ids']),'every_person_has_one_sound_kind':True,
            'actual_native_support_and_original_count_required':True}
    row['automatic_construction'].update(family=family,constructor='construct_specific_scene')
    _request(row,state,family);validate_recipe(row,catalog,contract)
    return plan
