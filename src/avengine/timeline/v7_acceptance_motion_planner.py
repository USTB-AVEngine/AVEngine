"""Choose actual motion deficits and construct proposals with the shipped v3 solver."""
from collections import defaultdict
from copy import deepcopy
import random
from avengine.timeline.v7_demand import demand_snapshot

MOTION_FAMILIES=('QA-06','QA-15','QA-16','QA-17','QA-13')


def choose_motion_family(state,seed):
    snapshot=demand_snapshot(state,next_questions=70);counts=state.get('family_counts',{})
    missing=[f for f in MOTION_FAMILIES if counts.get(f,0)<3]
    candidates=missing or list(MOTION_FAMILIES)
    if missing:
        least=min(counts.get(f,0)for f in missing)
        tied=[f for f in missing if counts.get(f,0)==least]
    else:
        gaps=snapshot['family_deficits'];maximum=max(gaps[f]for f in candidates)
        tied=[f for f in candidates if abs(gaps[f]-maximum)<1e-9]
    return random.Random(int(seed)).choice(tied)


def prepare_conditioned_motion_gap(row,state,catalog,records,inputs,house,semantic):
    """Route post-sound deficits to the independent post-sound adapter.

    The moving speaker keeps real complete recordings; its standing companion
    gets a whole authored utterance. The native producer still admits all poses,
    routes, retiming, RIRs, wet support and QA answers.
    """
    if row.get('conditioned_post_motion') or row.get('motion_during_condition_plan'):return row
    family=choose_motion_family(state,row['scene_seed'])
    if family in {'QA-06','QA-15'}:return row
    from avengine.timeline.v7_conditioned_post_motion import registered_human_motion_registry,plan_post_sound_motion
    from avengine.timeline.v7_phase2_sources import build_phase2_scene
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    from avengine.timeline.v7_audio_sources import filter_admission
    from avengine.timeline.v7_environmental_sources import recording_header
    row=deepcopy(row);byid={a['asset_id']:a for a in catalog}
    humans=[byid[aid]for aid in row['asset_ids']if byid[aid]['kind']=='human'and aid not in row.get('silent_asset_ids',[])]
    target=next(a for a in humans if a.get('walk_allowed'))
    companion=next(a for a in humans if a['asset_id']!=target['asset_id']);target_id=target['asset_id'];other_id=companion['asset_id']
    admitted,rejected=filter_admission(production_audio_record_views(records))
    if rejected:raise ValueError('post-sound planning contains rejected original records')
    voices=defaultdict(dict);gender='F'if target['visual_gender']=='female'else'M'
    for r in admitted:
        if r.get('dataset')=='VCTK-Corpus 0.92'and r.get('gender')==gender and r.get('legacy_library_split')=='train'and r.get('transcript')and .8<=float(r.get('duration_s',0))<=2.3:
            voices[r['speaker_id']][r['transcript']]=r
    eligible=sorted(voice for voice,lines in voices.items()if len(lines)>=3)
    if not eligible:raise ValueError('post-sound request needs three distinct whole recordings by one actual voice')
    voice=eligible[row['scene_seed']%len(eligible)];clips=sorted(voices[voice].values(),key=lambda r:(r['duration_s'],r['sound_asset_id']))[:3]
    authored=[s for s in semantic['sounds']if s['voice_reference']['gender']==companion['visual_gender']and .8<=float(s['synthesis_readback']['duration_s'])<=2.3]
    sound=min(authored,key=lambda s:(s['synthesis_readback']['duration_s'],s['sound_asset_id']))
    row.update(semantic_scenario_id=sound['scenario_id'],semantic_sound_ids=[sound['sound_asset_id']],semantic_human_asset_ids=[other_id],semantic_start_offsets_s=[0.])
    row.pop('motion_during_condition_requests',None);row.pop('motion_during_condition_plan',None)
    row.get('phase2_include_footsteps_asset_ids',[]).clear()
    row.setdefault('source_classes',{})[target_id]=['speech_playback'];row['source_classes'][other_id]=['speech_playback']
    row.setdefault('proposed_sound_asset_ids',{})[target_id]=clips[0]['sound_asset_id'];row['proposed_sound_asset_ids'].pop(other_id,None)
    row.setdefault('phase2_recorded_speech_schedule_s',{})[target_id]=[.5,10.5,20.5]
    row.setdefault('phase2_recorded_sound_ids',{})[target_id]=[r['sound_asset_id']for r in clips]
    for field in ('phase2_recorded_speech_schedule_s','phase2_recorded_sound_ids'):row[field].pop(other_id,None)
    row['phase2_authored_speech_schedule_s']={other_id:[0.,10.,20.]}
    yaw=float(row['camera_yaw_deg']);row['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}];row.pop('semantic_camera_sector_targets_deg',None)
    speed=float(row.get('motion_requests',{}).get(target_id,{}).get('speed_m_s',.15))
    row['motion_requests']={target_id:{'action':'walk','speed_m_s':speed,'intervals_s':[]}}
    _,scene,program,assets=build_phase2_scene(row,house,records,catalog,semantic)
    lookup={r['sound_asset_id']:r for r in program['records']};actual_companion=next(r for r in program['records']if r.get('source_type')=='author_tts')
    records_by_asset={target_id:[lookup.get(r['sound_asset_id'])or recording_header(r)for r in clips],other_id:[actual_companion]*3}
    registry=registered_human_motion_registry(humans,inputs,house)
    planned=plan_post_sound_motion(row,qa_id=family,target=target_id,companion=other_id,assets=assets,registry=registry,records_by_asset=records_by_asset,speed_m_s=speed)
    planned['scene_demand']['automatic_motion_gap_family']=family
    planned['scene_demand']['automatic_motion_gap_counts']=deepcopy(state.get('family_counts',{}))
    return planned
