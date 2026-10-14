"""Close the production feedback loop using selected logical questions only.

Requests guide real asset/timing/route proposals; they never certify QA truth.
The default target shares are configurable production goals, not measured facts.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json,math,os,random

DEMAND_REVISION = 'v7-demand-driven-20261015-v1'
# Priority order follows the owner; vision retains approximately 30% of items.
TARGET_GROUPS = (
    ('binding', .28, 5, ('QA-01','QA-02','QA-03','QA-12','QA-24','QA-26','QA-27','QA-28','V7-CONTENT-BIND')),
    ('direction', .16, 4, ('QA-04','QA-25','STARSS-FRUSTUM')),
    ('offscreen', .08, 3, ('QA-08','QA-09')),
    ('vision', .30, 2, ('QA-05','QA-07','QA-10','QA-11','QA-14','QA-20')),
    ('motion_time', .13, 1, ('QA-06','QA-13','QA-15','QA-16','QA-17','QA-18','QA-19','QA-23','STARSS-SPEAKERS')),
    ('taxonomy', .05, 4, ('QA-21','QA-22','STARSS-CLASS13','V7-CLASS34')),
)
CONDITION_FAMILIES = {
    'walking_human': ['QA-06','QA-13','QA-15','QA-16','QA-17'],
    'distance_trend_during_sound': ['QA-15'],
    'moving_after_source_and_wet_tail_stop': ['QA-13','QA-16','QA-17'],
    'moving_animal': ['QA-06','QA-13','QA-15','QA-16','QA-17'],
    'same_source_repeated': ['QA-19','QA-23','STARSS-SPEAKERS'],
    'staggered_sources': ['QA-03','QA-18','QA-23','STARSS-SPEAKERS'],
    'countable_class13_activity': ['QA-23','STARSS-SPEAKERS'],
    'silent_unique_bystander': ['QA-01','QA-02','QA-14'],
    'registered_occluder_relation': ['QA-09','QA-10','QA-11'],
    'authored_native_quartet': ['QA-26','QA-27','QA-28','V7-CONTENT-BIND'],
    'admitted_source_taxonomy': ['QA-21','STARSS-CLASS13','V7-CLASS34'],
    'source_direction_and_frustum': ['QA-04','QA-25','STARSS-FRUSTUM','QA-08','QA-09'],
}

FAMILY_SCENE_REQUIREMENTS = {'QA-01': {'title_zh': '外观→是否发声',
           'conditions': ['unique_reviewed_appearance', 'silent_same_type_bystander', 'speaking_actor']},
 'QA-02': {'title_zh': '声音/台词→外观',
           'conditions': ['unique_sound_or_whole_transcript',
                          'visible_referable_speaker',
                          'different_appearance_alternative']},
 'QA-03': {'title_zh': '谁先发声',
           'conditions': ['two_speaking_actors',
                          'unique_staggered_first_onset',
                          'distinct_candidate_labels']},
 'QA-04': {'title_zh': '发声者左右',
           'conditions': ['actual_camera_listener_frames',
                          'left_right_source_separation',
                          'bound_sound_event']},
 'QA-05': {'title_zh': '发声区间重叠',
           'conditions': ['source_measured_activity', 'overlapping_and_disjoint_event_pairs']},
 'QA-06': {'title_zh': '发声期间是否移动',
           'conditions': ['moving_and_static_speaking_bodies', 'stable_native_motion_during_event']},
 'QA-07': {'title_zh': '从哪侧入画',
           'conditions': ['actual_left_or_right_entry',
                          'camera_sweep_or_source_motion',
                          'modal_target_pixel_stream']},
 'QA-08': {'title_zh': '发声时可见状态',
           'conditions': ['in_frame_and_offscreen_events', 'clear_and_occluded_pixel_states']},
 'QA-09': {'title_zh': '完全遮挡后是否重现',
           'conditions': ['registered_full_occluder', 'later_visible_reappearance', 'stable_mask_intervals']},
 'QA-10': {'title_zh': '遮挡物身份',
           'conditions': ['registered_visible_occluder',
                          'explicit_target_occluder_relation',
                          'complete_native_occluder_ids']},
 'QA-11': {'title_zh': '部分遮挡→完全可见',
           'conditions': ['registered_partial_occluder', 'later_clear_visibility', 'stable_mask_intervals']},
 'QA-12': {'title_zh': '外观→说了什么',
           'conditions': ['unique_reviewed_appearance',
                          'complete_whole_transcript',
                          'different_whole_transcript_alternative']},
 'QA-13': {'title_zh': '发声结束后的方位',
           'conditions': ['complete_native_wet_support',
                          'post_source_stop_visible_query',
                          'separated_post_sound_directions']},
 'QA-14': {'title_zh': '给定时刻距离比较',
           'conditions': ['three_visible_unique_actors',
                          'near_far_separated_distances',
                          'stable_legal_query_window']},
 'QA-15': {'title_zh': '发声期间靠近/远离',
           'conditions': ['explicit_approaching_or_receding_route',
                          'net_distance_margin_during_whole_utterance']},
 'QA-16': {'title_zh': '声停后距离变化',
           'conditions': ['complete_native_wet_support',
                          'motion_continues_after_emission',
                          'post_stop_distance_change_above_original_margin']},
 'QA-17': {'title_zh': '声停后是否移动',
           'conditions': ['complete_native_wet_support', 'post_stop_moving_and_static_alternatives']},
 'QA-18': {'title_zh': '指定时刻谁发声',
           'conditions': ['complete_native_wet_support',
                          'source_activity_integer_second_bins',
                          'staggered_speakers_and_silent_bins']},
 'QA-19': {'title_zh': '目标首次发声时刻',
           'conditions': ['unique_reviewed_target',
                          'repeated_source_events',
                          'distinct_first_onset_seconds']},
 'QA-20': {'title_zh': '可见候选谁发声/都不是',
           'conditions': ['visible_unique_candidates',
                          'bound_speaker_or_offscreen_speaker',
                          'native_candidate_visibility']},
 'QA-21': {'title_zh': '外观→声音类别',
           'conditions': ['admitted_class34_source',
                          'unique_reviewed_appearance',
                          'actual_capable_emission_mechanism']},
 'QA-22': {'title_zh': '实体数/发声个体数',
           'conditions': ['stable_native_actor_ids',
                          'silent_and_speaking_distinct_entities',
                          'complete_activity_scope']},
 'QA-23': {'title_zh': '发声事件次数',
           'conditions': ['all_window_audio_within_STARSS13',
                          'measured_activity_segments',
                          'repeated_and_staggered_events']},
 'QA-24': {'title_zh': '先发声者片尾可见状态',
           'conditions': ['unique_earliest_source', 'late_camera_or_source_visibility_change']},
 'QA-25': {'title_zh': '实例连续水平角度',
           'conditions': ['eight_direction_source_proposals',
                          'actual_camera_listener_frames',
                          'unchanged_native_RIR_direction_gate']},
 'QA-26': {'title_zh': '外观→话语内容',
           'conditions': ['authored_whole_TTS', 'unique_reviewed_appearance', 'native_dialogue_binding']},
 'QA-27': {'title_zh': '话语内容→外观',
           'conditions': ['authored_whole_TTS', 'unique_role_content', 'native_dialogue_binding']},
 'QA-28': {'title_zh': '话语内容→发声方位',
           'conditions': ['authored_whole_TTS', 'separated_source_sectors', 'native_dialogue_binding']},
 'STARSS-SPEAKERS': {'title_zh': '独立说话人数',
                     'conditions': ['physical_actor_identity',
                                    'repeated_same_speaker_and_multiple_speakers',
                                    'complete_live_speech_activity']},
 'STARSS-FRUSTUM': {'title_zh': '声音在画内/画外',
                    'conditions': ['in_and_out_frustum_source_slots',
                                   'actual_camera_track',
                                   'measured_direction']},
 'STARSS-CLASS13': {'title_zh': 'STARSS13 类别',
                    'conditions': ['admitted_STARSS13_source_classes', 'correct_live_or_device_mechanism']},
 'V7-CLASS34': {'title_zh': 'QA21 类别域',
                'conditions': ['admitted_class34_source_classes', 'correct_device_or_animal_mechanism']},
 'V7-CONTENT-BIND': {'title_zh': '内容声画绑定',
                     'conditions': ['complete_native_four_member_group',
                                    'all_program_records_authored_TTS',
                                    'observed_counterfactual_content_witnesses']}}


def default_targets():
    return {f: share/len(families) for _,share,_,families in TARGET_GROUPS for f in families}


def demand_snapshot(state, *, next_questions=14):
    counts=Counter(state.get('family_counts',{}));total=int(state.get('logical_questions',sum(counts.values())))
    if total<0 or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in counts.values()) or sum(counts.values())!=total:
        raise ValueError('demand feedback must count actual selected logical items exactly once')
    targets=state.get('target_shares') or default_targets()
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in targets.values()) or abs(sum(targets.values())-1)>1e-8:
        raise ValueError('demand target shares must be finite, nonnegative and sum to one')
    horizon=total+int(next_questions)
    deficits={f:max(0.,horizon*share-counts[f]) for f,share in targets.items()}
    priority={f:p for _,_,p,families in TARGET_GROUPS for f in families}
    # Feed the original selector its supported family_counts input: a weighted
    # relative deficit is a negative count. One real selected item raises it
    # by one, so a large gap cannot monopolize a crop with repeated stems.
    selection={f:-min(1.,gap/max(1.,horizon*targets[f]))*(1.+.1*(priority.get(f,1)-1))
               for f,gap in deficits.items()}
    return {'revision':DEMAND_REVISION,'observed_logical_questions':total,'observed_family_counts':dict(counts),
            'next_question_horizon':horizon,'target_shares':dict(targets),'family_deficits':deficits,
            'selection_counts':selection,'condition_families':deepcopy(CONDITION_FAMILIES),
            'family_scene_requirements':deepcopy(FAMILY_SCENE_REQUIREMENTS)}


def read_demand_state(path):
    p=Path(path)
    state=json.loads(p.read_text()) if p.exists() else {'revision':DEMAND_REVISION,'logical_questions':0,'family_counts':{},'parents':{}}
    demand_snapshot(state)
    return state


def record_selected_parent(path,result):
    if result.get('status') not in {'integration_pilot_admitted','formal_alpha_admitted'}:
        return
    # A task-owned state can be updated by multiple completed workers. The lock
    # and atomic replace cover read/modify/write; resume replaces one parent's
    # actual count rather than counting the same parent again.
    import fcntl
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_suffix(p.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read_demand_state(p);counts=Counter()
        for w in result['windows']:
            if sum(w['family_counts'].values())!=w['logical_questions']:
                raise ValueError('parent feedback does not match its selected logical question count')
            counts.update(w['family_counts'])
        n=sum(counts.values());key=result.get('requested_scene_id',result['scene_id'])
        old=state.setdefault('parents',{}).get(key,{'logical_questions':0,'family_counts':{}})
        aggregate=Counter(state['family_counts']);aggregate.subtract(old['family_counts']);aggregate.update(counts)
        state.update(logical_questions=state['logical_questions']-old['logical_questions']+n,
                     family_counts={k:v for k,v in aggregate.items() if v},revision=DEMAND_REVISION)
        state['parents'][key]={'scene_id':result['scene_id'],'logical_questions':n,'family_counts':dict(counts),
                               'delivery_directory':result.get('delivery_directory'),'raw_logical_questions':result.get('raw_logical_questions',n),
                               'dedup_dropped':result.get('dedup_dropped',0),'dedup_rule_version':result.get('dedup_rule_version')}
        demand_snapshot(state)
        tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
        with tmp.open('w') as stream:
            json.dump(state,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,p)


def add_scene_demand(row,state,*,focus=None):
    row=deepcopy(row)
    if row.get('schema')=='v7-recorded-registered-standing-v1':
        row['window_count']=5;row['window_start_ticks']=[0,50,100,150,200]
    request=demand_snapshot(state,next_questions=row.get('window_count',2)*row.get('questions_per_window',14)*len(row.get('members',['v0_a0'])))
    gaps=request['family_deficits'];rng=random.Random(row['scene_seed'])
    choices=['binding','human_motion','count','animal_motion','occlusion','taxonomy','content_binding']
    groups=[('QA-01','QA-02','QA-03','QA-12','QA-28'),('QA-06','QA-13','QA-15','QA-16','QA-17'),('QA-18','QA-19','QA-23','STARSS-SPEAKERS'),('QA-06','QA-15'),('QA-09','QA-10','QA-11'),('QA-21','STARSS-CLASS13','V7-CLASS34'),('QA-26','QA-27','QA-28','V7-CONTENT-BIND')]
    weights=[sum(gaps.get(f,0) for f in group)*(5 if name=='binding' else 1) for name,group in zip(choices,groups)]
    chosen=focus or rng.choices(choices,weights=weights if any(weights) else None,k=1)[0]
    if chosen not in choices:raise ValueError('unsupported demand focus')
    request.update(focus=chosen,requested_conditions=['same_source_repeated','staggered_sources','source_direction_and_frustum'],
                   status='requested_only; original native and QA gates determine realization')
    if chosen in {'human_motion','animal_motion'}:
        request['requested_conditions']+=['walking_human' if chosen=='human_motion' else 'moving_animal',
                                         'distance_trend_during_sound','moving_after_source_and_wet_tail_stop']
    if chosen=='count':request['requested_conditions'].append('countable_class13_activity')
    if chosen in {'binding','occlusion'}:request['requested_conditions'].append('silent_unique_bystander')
    if chosen=='occlusion':request['requested_conditions'].append('registered_occluder_relation')
    if chosen=='taxonomy':request['requested_conditions'].append('admitted_source_taxonomy')
    if chosen=='content_binding':request['requested_conditions'].append('authored_native_quartet')
    row['scene_demand']=request
    return row


def plan_phase2_demand(row,state,catalog,records,contract,*,focus=None):
    """Compile concrete demands through the existing registered phase2 intake."""
    from avengine.timeline.v7_production_inputs import bind_animal_motion,validate_recipe
    row=add_scene_demand(row,state,focus=focus);request=row['scene_demand'];focus=request['focus']
    row['acceptance_allow_partial_families']=False
    if row.get('schema')=='v7-visible-registered-pilot-phase2-v1':
        row['trial_only']=True
        if row.get('members') not in (None,['v0_a0']):raise ValueError('phase2 motion producer requires a single native member')
        row['members']=['v0_a0']
    byid={a['asset_id']:a for a in catalog};rng=random.Random(row['scene_seed'])
    if row.get('family')=='kujiale':
        request['unresolved_conditions']=['phase2 motion unavailable in the registered Kujiale producer']
        return row
    humans=[byid[aid] for aid in row['asset_ids'] if byid[aid].get('kind')=='human' and byid[aid].get('controlled_shirt')]
    if focus=='content_binding':
        request['required_producer_profile']='registered_authored_quartet'
        request['unresolved_conditions']=['phase2 single-member row must be assigned an original registered authored quartet template']
    elif focus in {'human_motion','count'}:
        if len(humans)<2:raise ValueError('demand needs at least two existing registered humans')
        # Keep the exact selected bodies and choose admitted whole recorded
        # speech; authored standing-only TTS is never moved or relabelled.
        row['asset_ids']=[a['asset_id'] for a in humans]
        for key in list(row):
            if key.startswith(('semantic_','reuse_')):row.pop(key)
        row['source_classes']={};row['proposed_sound_asset_ids']={};row['motion_requests']={};row['silent_asset_ids']=[]
        row['phase2_recorded_speech_schedule_s']={}
        selected=humans[:2]
        for i,a in enumerate(selected):
            gender='F' if a.get('visual_gender')=='female' else 'M'
            pool=[r for r in records if r.get('dataset')=='VCTK-Corpus 0.92' and r.get('source_type','recorded')=='recorded'
                  and r.get('gender')==gender and r.get('legacy_library_split')=='train'
                  and r.get('transcript') and .8<=r.get('duration_s',0)<=2.5]
            if not pool:raise ValueError('demand lacks admitted short whole recorded speech for '+gender)
            shortest=min(r['duration_s'] for r in pool);pool=[r for r in pool if r['duration_s']<=shortest+.1]
            record=rng.choice(sorted(pool,key=lambda r:r['sound_asset_id']));aid=a['asset_id']
            row['source_classes'][aid]=['speech_playback'];row['proposed_sound_asset_ids'][aid]=record['sound_asset_id']
            row['phase2_recorded_speech_schedule_s'][aid]=[start+offset for start in (0.,10.,20.) for offset in ((.3,2.8) if i==0 else (0.,))]
        for a in humans[2:]:row['silent_asset_ids'].append(a['asset_id'])
        if focus=='human_motion':
            walker=selected[0]
            if not walker.get('walk_allowed'):raise ValueError('demand selected body has no registered walk')
            trend='receding' if row['scene_seed']%2 else 'approaching'
            row['motion_requests'][walker['asset_id']]={'action':'walk','speed_m_s':.15,'intervals_s':[[0.,2.5],[8.,9.9],[10.,12.5],[18.,19.9],[20.,22.5],[28.,29.9]],'distance_trend':trend}
            request['distance_trend']=trend
        request['count_scope']='all actual human source activity in STARSS class13; no mixed-domain count'
    elif focus=='animal_motion':
        animals=[byid[aid] for aid in row['asset_ids'] if aid in contract.get('animal_motion',{})]
        if not animals:
            request['unresolved_conditions']=['no selected animal with final registered motion authority']
        else:
            animal=rng.choice(animals);aid=animal['asset_id']
            row.setdefault('motion_requests',{})[aid]=bind_animal_motion({'speed_m_s':.03},aid,contract)
            row['phase2_single_animal_utterance_crop']=True
            row['phase2_animal_event_limit']={aid:3}
            # Actual complete calls stop while the admitted body keeps translating.
            row['phase2_effect_start_offsets_s']=[.5,10.5,20.5,5.2,15.2,25.2]
    else:
        used={a.get('color') for a in humans};silent=[a for a in catalog if a.get('kind')=='human' and a.get('controlled_shirt')
                and a.get('color') not in used and a['asset_id'] not in row['asset_ids']
                and a['asset_id'] not in contract.get('policy',{}).get('temporarily_excluded_asset_ids',{})]
        if silent:
            a=rng.choice(sorted(silent,key=lambda a:a['asset_id']));row['asset_ids'].append(a['asset_id']);row.setdefault('silent_asset_ids',[]).append(a['asset_id'])
        else:request['unresolved_conditions']=['no additional uniquely colored registered silent bystander']
    # Five half-overlapping crops reuse one native 30s world. They keep the parent
    # identity and never claim six independent scenes or a counterfactual group.
    if row.get('trial_only') is True and row.get('members')==['v0_a0']:
        row['acceptance_window_starts_s']=[0.,5.,10.,15.,20.];row['window_count']=5
    request.update(demand_snapshot(state,next_questions=row.get('window_count',2)*row.get('questions_per_window',14)*len(row.get('members',['v0_a0']))))
    if focus=='occlusion':
        audible=[a for a in humans if a['asset_id'] not in row.get('silent_asset_ids',[])]
        silent=[byid[aid] for aid in row.get('silent_asset_ids',[]) if byid.get(aid,{}).get('kind')=='human']
        if audible and silent:
            target=audible[0]['asset_id'];occluder=silent[-1]['asset_id']
            row.setdefault('semantic_azimuth_targets_deg',{}).update({target:10.,occluder:0.})
            row.setdefault('native_attachment_distance_targets_m',{}).update({target:3.,occluder:1.5})
            row['native_occlusion_relations']=[{'target_asset_id':target,'occluder_asset_id':occluder,
                'relation':'registered silent foreground human partially occludes the farther speaker',
                'minimum_stable_seconds':1.,'status':'proposal_requires_original_native_mask_and_occluder_ID_admission'}]
        else:request.setdefault('unresolved_conditions',[]).append('registered target and uniquely referable silent occluder missing')
    if focus=='taxonomy':
        active=[cl for classes in row.get('source_classes',{}).values() for cl in classes if cl!='speech_playback']
        if active:row['category_focus_class']=rng.choice(sorted(set(active)))
        else:request.setdefault('unresolved_conditions',[]).append('no selected non-speech class34-capable physical source')
    request.update(proposed_motion_requests=deepcopy(row.get('motion_requests',{})),
                   proposed_silent_asset_ids=list(row.get('silent_asset_ids',[])))
    validate_recipe(row,catalog,contract)
    return row
