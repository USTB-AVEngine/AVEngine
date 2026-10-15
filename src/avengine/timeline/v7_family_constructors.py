"""Room-based construction requests. No saved scene recipe is an input.

A request never certifies a route, mask, wet tail, answer or native admission.
The motion helper owns QA06/15; these constructors own the taxonomy gaps.
"""
from __future__ import annotations
from copy import deepcopy
from collections import defaultdict
import hashlib,math,random,itertools

REVISION='v7-room-family-constructors-20261015-v1'
FAMILIES=('QA-16','QA-17','QA-13','QA-10','QA-28','QA-21','V7-CLASS34',
          'V7-CONTENT-BIND','QA-01','QA-03','QA-19','QA-26','QA-27')
POST_FAMILIES={'QA-13','QA-16','QA-17'}

class ConstructionUnavailable(ValueError):
    """Keep the requested family and try another legal candidate; never drop it."""


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:7],'big')


def next_family(state,families=FAMILIES):
    counts=state.get('family_counts',{})
    # The minimum-three requirement comes before the proportional goals.
    return min(families,key=lambda f:(min(counts.get(f,0),3),counts.get(f,0),families.index(f)))


def room_context(room,house,attempt=0):
    if room['house']!=house['house'] or room['room_id']not in house['checked_room_ids']:
        raise ConstructionUnavailable('original room/house acoustic authority does not match')
    actual=next(r for r in house['checked_rooms'] if r.get('room_label',r.get('room_id'))==room['room_id'])
    placement=actual.get('placement',actual.get('native_placement',{}))
    listener=room.get('listener_m') or placement.get('camera_m') or actual.get('listener_m')
    if listener is None:raise ConstructionUnavailable('room needs an actual inherited listener proposal')
    sources=[placement[k] for k in ('source_1_m','source_2_m') if placement.get(k)]
    if sources:
        x=sum(p[0]for p in sources)/len(sources)-listener[0]
        z=sum(p[2]for p in sources)/len(sources)-listener[2]
        yaw=math.degrees(math.atan2(x,-z))
    else:yaw=0.
    yaw+= (0,15,-15,30,-30,60,-60,90,-90,135,-135,180)[attempt%12]
    polygon=actual.get('floor_polygon',actual.get('room_floor_polygon'))
    return {'listener_m':list(listener),'camera_yaw_deg':yaw,'source_proposals':sources,
        'room_floor_polygon':deepcopy(polygon),'original_room':actual}


def eligible_humans(catalog,room,contract):
    from avengine.timeline.v7_production_inputs import asset_category
    excluded=contract.get('policy',{}).get('temporarily_excluded_asset_ids',{})
    return sorted((a for a in catalog if a.get('kind')=='human' and a.get('controlled_shirt')
        and a.get('dialogue_voice_binding_eligible',True) and a['asset_id']not in excluded
        and asset_category(a,{},room['family'])in room['categories']),key=lambda a:a['asset_id'])


def select_people(catalog,room,contract,seed,*,moving=False,attempt=0,speaker_genders=None):
    pool=eligible_humans(catalog,room,contract);rng=random.Random(stable_seed(seed,'bodies',attempt))
    if len({a['color']for a in pool})<3:
        raise ConstructionUnavailable('three distinct registered human appearances are required')
    candidates=[a for a in pool if (not moving or a.get('walk_allowed'))and(not speaker_genders or a['visual_gender']==speaker_genders[0])]
    if not candidates:raise ConstructionUnavailable('no registered walking human in this legal cell')
    first=rng.choice(candidates);others=[a for a in pool if a['color']!=first['color']]
    second_pool=[a for a in others if not speaker_genders or a['visual_gender']==speaker_genders[1]]
    if not second_pool:raise ConstructionUnavailable('no legal distinct-color body for the authored gender combination')
    second=rng.choice(second_pool);quiet=rng.choice([a for a in others if a['color']!=second['color']])
    return deepcopy([first,second,quiet])


def choose_authored(manifest,people,state,seed,*,epoch_count=1,attempt=0,excluded_voices=()):
    """Whole roles, fixed distinct real voices; rotate over the admitted directory."""
    groups=defaultdict(list)
    from avengine.qa.v7_question_wording import contains_internal_wording
    for sound in manifest['sounds']:
        if contains_internal_wording(sound.get('transcript','')):continue
        if .65<=sound['synthesis_readback']['duration_s']<=2.8:
            groups[sound['scenario_id']].append(sound)
    usage=state.get('authored_voice_counts',{});combinations=defaultdict(dict)
    for scenario,rows in sorted(groups.items()):
        roles=sorted({s['role_id']for s in rows})
        for order in itertools.permutations(roles,len(people)):
            pools=[]
            for role,person in zip(order,people):
                gender=person['visual_gender']
                pools.append([r for r in rows if r['role_id']==role
                    and r['voice_reference']['gender'].lower()in {gender,gender[:1]}
                    and r['voice_reference']['speaker_id']not in excluded_voices])
            for current in itertools.product(*pools):
                voices=tuple(r['voice_reference']['speaker_id']for r in current)
                if len(set(voices))!=len(people):continue
                combinations[voices].setdefault(scenario,current)
    compatible=[(voices,choices)for voices,choices in combinations.items()if len(choices)>=epoch_count]
    if not compatible:raise ConstructionUnavailable('not enough distinct complete authored scenarios for fixed physical voices')
    voices,choices=min(compatible,key=lambda item:(sum(usage.get(v,0)for v in item[0]),
        max(usage.get(v,0)for v in item[0]),stable_seed(seed,'voice',*item[0],attempt)))
    ordered=[choices[key]for key in sorted(choices)]
    offset=stable_seed(seed,'scenario',attempt)%len(ordered)
    ordered=ordered[offset:]+ordered[:offset]
    return deepcopy(ordered[:epoch_count])


def choose_taxonomy_emitter(catalog,people,room,contract,records,state,seed,attempt):
    """Jointly choose a legal registered source and an underproduced real class."""
    from avengine.timeline.v7_production_inputs import asset_category
    from avengine.timeline.v7_taxonomy_demand import choose_unseen_class
    excluded=contract.get('policy',{}).get('temporarily_excluded_asset_ids',{})
    available=[a for a in catalog if a.get('kind')not in {'human','performer','chair'}]
    # The authored companion keeps exactly one speaking source. The remaining
    # two real humans can exchange the audible/quiet role without being removed.
    available.extend([people[0],people[2]])
    options=[]
    for asset in available:
        if asset['asset_id']in excluded:continue
        try:
            if asset_category(asset,{},room['family'])not in room['categories']:continue
            cls,proof=choose_unseen_class(asset,records,state.get('class34_positive_counts',{}))
        except ValueError:continue
        surface={'floor':0,'table':1,'wall':2,'ceiling':3}.get(asset.get('attachment_surface'),4)
        score=(state.get('class34_positive_counts',{}).get(cls,0),0 if asset in people else 1,surface,
            stable_seed(seed,asset['asset_id'],cls,attempt))
        options.append((score,asset,cls,proof))
    if not options:raise ConstructionUnavailable('no registered source in the original legal cell can emit an admitted CLASS34 candidate')
    _,asset,cls,proof=min(options,key=lambda item:item[0])
    return deepcopy(asset),cls,proof


def _request(row,state,family):
    from avengine.timeline.v7_demand import demand_snapshot,FAMILY_SCENE_REQUIREMENTS
    request=demand_snapshot(state,next_questions=70)
    request.update(focus='automatic_family',target_family=family,
        requested_conditions=list(FAMILY_SCENE_REQUIREMENTS[family]['conditions']),
        required_asset_ids=list(row['asset_ids']),required_independent_family_counts={family:3},
        status='automatic construction proposal; original native and QA gates still required')
    row['scene_demand']=request


def _camera_knots(yaw,**unused):
    # Owner 15:0x: fixed camera for the whole world, including final seconds.
    return [{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}]


def construct_family_scene(room,house,state,catalog,records,semantic,contract,runtime,
        *,scene_id,seed,family=None,attempt=0):
    """Accept any original legal room; never load a hand-authored scene template."""
    from avengine.timeline.v7_production_inputs import validate_recipe
    family=family or next_family(state)
    if family not in FAMILIES:raise ConstructionUnavailable('family belongs to the separate motion/UE constructor: '+family)
    if room['family']=='kujiale':raise ConstructionUnavailable('Kujiale automatic construction is owned by its registered UE producer')
    people=select_people(catalog,room,contract,seed,moving=family in POST_FAMILIES,attempt=attempt)
    if family=='V7-CONTENT-BIND':
        # Color counterfactuals share a registered body and gender. Choose that
        # compatibility from the actual roster, then solve whole fixed voices.
        from avengine.qa.v7_visual_admission import COLOR_HUE_DEGREES
        pool=eligible_humans(catalog,room,contract);groups=defaultdict(list)
        for person in pool:
            # This producer uses the original hue reviewer, unlike phase-2's
            # CIELAB reviewer. Select compatible registered color variants;
            # retain every required person and counterfactual member.
            if person['color']in COLOR_HUE_DEGREES:
                groups[(person.get('body_identity_id',person['visual_gender']),person['visual_gender'])].append(person)
        keys=sorted(groups,key=lambda key:stable_seed(seed,'quartet body',*key,attempt))
        for key in keys:
            variants=groups[key];colors=sorted({a['color']for a in variants})
            if len(colors)<2:continue
            colors.sort(key=lambda color:stable_seed(seed,key[0],color,attempt))
            pair=[next(a for a in variants if a['color']==color)for color in colors[:2]]
            quiet=[a for a in pool if a['color']not in colors[:2]]
            if not quiet:continue
            candidate=deepcopy([*pair,quiet[stable_seed(seed,'quiet',attempt)%len(quiet)]])
            try:choose_authored(semantic,candidate[:2],state,seed,epoch_count=3,attempt=attempt)
            except ConstructionUnavailable:continue
            people=candidate;break
        else:raise ConstructionUnavailable('no legal shared-body color pair and fixed voices support three whole authored scenarios')
    ids=[a['asset_id']for a in people];ctx=room_context(room,house,attempt)
    row={'schema':'v7-visible-registered-pilot-phase2-v1','scene_id':scene_id,
        'independent_parent_scene_id':scene_id,'scene_seed':seed,'house':room['house'],
        'room_id':room['room_id'],'split':'train','family':room['family'],
        'listener_m':ctx['listener_m'],'camera_yaw_deg':ctx['camera_yaw_deg'],
        'camera_yaw_knots':[{'time_s':0.,'yaw_deg':ctx['camera_yaw_deg']},{'time_s':30.,'yaw_deg':ctx['camera_yaw_deg']}],
        'asset_ids':ids,'silent_asset_ids':[ids[2]],'source_classes':{},'motion_requests':{},
        'members':['v0_a0'],'trial_only':True,'window_count':5,'questions_per_window':14,
        'acceptance_window_starts_s':[0.,5.,10.,15.,20.],'question_quota_base':0,
        'automatic_construction':{'revision':REVISION,'family':family,'candidate_index':attempt,
            'room_key':room['key'],'required_role_count':3,'scene_template_files_read':[]},
        'owner_social_layout':{'minimum_human_center_distance_m':1.,'minimum_human_animal_distance_m':1.,
            'human_orientation':'varied side-facing, conversational or facing away','distribution':'different floor regions; preserve every required participant'}}
    _request(row,state,family)
    if family in POST_FAMILIES:
        return _construct_post(row,people,state,catalog,records,semantic,contract,runtime,house,family)
    taxonomy=family in {'QA-21','V7-CLASS34'}
    tts_people=people[1:2]if taxonomy else people[:2]
    tts_ids=[a['asset_id']for a in tts_people]
    authored=choose_authored(semantic,tts_people,state,seed,attempt=attempt)[0]
    offsets=[0.]if taxonomy else[0.,max(2.5,math.ceil(authored[0]['synthesis_readback']['duration_s']*10)/10+.5)]
    row.update(semantic_scenario_id=authored[0]['scenario_id'],
        semantic_sound_ids=[r['sound_asset_id']for r in authored],semantic_human_asset_ids=tts_ids,
        semantic_start_offsets_s=offsets)
    row['source_classes']={aid:['speech_playback']for aid in tts_ids}
    row['phase2_authored_speech_schedule_s']={aid:[start+epoch for epoch in (0.,10.,20.)]
        for aid,start in zip(tts_ids,offsets)}
    if family=='QA-10':
        # Rotate logical roles, not manually chosen colors or native support coordinates.
        target=ids[attempt%2];occluder=ids[2]
        row['native_occlusion_relations']=[{'target_asset_id':target,'occluder_asset_id':occluder,
            'minimum_stable_seconds':.5,'minimum_occluded_fraction':.25,
            'relation':'unique registered foreground entity partially occludes the target'}]
        row['semantic_azimuth_targets_deg']={target:0.}
        row['native_attachment_distance_targets_m']={target:3.2,occluder:1.8}
    elif family=='QA-28':
        # Both speakers are at least 15 degrees inside the front sector,
        # visible to this fixed camera. Distinct colours and depths retain
        # independent identity; angular distractors/QA domains are unchanged.
        row['semantic_azimuth_targets_deg']={ids[0]:-5.,ids[1]:5.}
        row['native_attachment_distance_targets_m']={ids[0]:2.3,ids[1]:3.8}
        row['camera_yaw_knots']=_camera_knots(ctx['camera_yaw_deg'])
        row['scene_demand']['required_sector_boundary_margin_deg']=15.
    elif family in {'QA-21','V7-CLASS34'}:
        emitter,cls,proof=choose_taxonomy_emitter(catalog,people,room,contract,records,state,seed,attempt)
        aid=emitter['asset_id']
        if aid not in row['asset_ids']:row['asset_ids'].append(aid)
        row['silent_asset_ids']=[person['asset_id']for person in people if person['asset_id']not in {ids[1],aid}]
        row['source_classes'][aid]=[cls];row['category_focus_class']=cls
        row['automatic_taxonomy_clock']={'asset_id':aid,'after_authored_complete_utterance':True}
        row['scene_demand']['class34_construction']=proof
        row['scene_demand']['required_asset_ids']=list(row['asset_ids'])
        row['automatic_construction']['required_role_count']=len(row['asset_ids'])
    elif family=='V7-CONTENT-BIND':
        return _construct_content(row,people,state,semantic,ctx,attempt)
    validate_recipe(row,catalog,contract)
    return {'profile':'phase2','row':row}


def whole_recorded_triplet(records,gender,seed):
    """Prefer complete short utterances; no recording is cropped or relabelled."""
    voices=defaultdict(dict)
    from avengine.qa.v7_question_wording import contains_internal_wording
    for record in records:
        if contains_internal_wording(record.get('transcript','')):continue
        if (record.get('dataset')!='VCTK-Corpus 0.92'or record.get('source_type','recorded')!='recorded'
                or record.get('gender')!=gender or record.get('legacy_library_split')!='train'
                or not record.get('transcript')or record.get('duration_s',0)<=0):continue
        previous=voices[record['speaker_id']].get(record['transcript'])
        if previous is None or record['duration_s']<previous['duration_s']:
            voices[record['speaker_id']][record['transcript']]=record
    choices=[]
    for voice,unique in voices.items():
        if len(unique)<3:continue
        whole=sorted(unique.values(),key=lambda r:(len(r['transcript'].split())<3,r['duration_s'],r['sound_asset_id']))[:3]
        choices.append(whole)
    if not choices:raise ConstructionUnavailable('no actual same-voice three-utterance recording group matches the body')
    return min(choices,key=lambda group:(sum(len(r['transcript'].split())<3 for r in group),
        max(r['duration_s']for r in group),sum(r['duration_s']for r in group),stable_seed(seed,group[0]['speaker_id'])))


def _construct_post(row,people,state,catalog,records,semantic,contract,runtime,house,family):
    from avengine.timeline.v7_environmental_sources import recording_header
    from avengine.timeline.v7_registered_sampling import semantic_record
    from avengine.timeline.v7_conditioned_post_motion import registered_human_motion_registry,plan_post_sound_motion
    from avengine.timeline.v7_family_clock import build_constructed_scene
    target,companion=(a['asset_id']for a in people[:2]);gender='F'if people[0]['visual_gender']=='female'else'M'
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    from avengine.timeline.v7_audio_sources import filter_admission
    admitted,rejected=filter_admission(production_audio_record_views(records))
    if rejected:raise ConstructionUnavailable('the original source admission rejects records; no source may be invented')
    clips=whole_recorded_triplet(admitted,gender,row['scene_seed']);voice=clips[0]['speaker_id']
    row['source_classes']={target:['speech_playback'],companion:['speech_playback']}
    row['motion_requests']={};row['proposed_sound_asset_ids']={target:clips[0]['sound_asset_id']}
    row['phase2_recorded_sound_ids']={target:[r['sound_asset_id']for r in clips]}
    row['scene_demand']['distance_trend']=('approaching','receding')[(len(state.get('parents',{}))+row['automatic_construction']['candidate_index'])%2]
    registry=registered_human_motion_registry(people[:2],runtime,house)
    target_records=[recording_header(r)for r in clips]
    other_gender=people[1]['visual_gender'];usage=state.get('authored_voice_counts',{})
    candidates=[s for s in semantic['sounds']if s['voice_reference']['gender'].lower()in {other_gender,other_gender[:1]}
        and s['voice_reference']['speaker_id']!=voice]
    candidates.sort(key=lambda r:(usage.get(r['voice_reference']['speaker_id'],0),r['synthesis_readback']['duration_s'],
        stable_seed(row['scene_seed'],r['sound_asset_id'])))
    rejected=[]
    for speech in candidates:
        candidate=deepcopy(row)
        candidate.update(semantic_scenario_id=speech['scenario_id'],semantic_sound_ids=[speech['sound_asset_id']],
            semantic_human_asset_ids=[companion],semantic_start_offsets_s=[0.])
        try:
            reference=semantic_record(speech,'F'if other_gender=='female'else'M')
            candidate=plan_post_sound_motion(candidate,qa_id=family,target=target,companion=companion,
                assets=people,registry=registry,records_by_asset={target:target_records,companion:[reference]},
                speed_m_s=.8+.6*(stable_seed(row['scene_seed'],target,'speed')%10001)/10000)
            # This is only source-program compilation. Native render and wet-tail
            # gates still run afresh; the original solver retains its 3s reserve.
            build_constructed_scene(candidate,house,records,catalog,semantic)
        except ValueError as error:
            rejected.append({'sound_asset_id':speech['sound_asset_id'],'reason':str(error)[:300]});continue
        candidate['automatic_construction']['joint_source_timing_rejections']=rejected
        candidate['automatic_construction']['whole_recorded_target_ids']=[r['sound_asset_id']for r in clips]
        return {'profile':'phase2','row':candidate}
    raise ConstructionUnavailable('no unchanged v3 joint source/timing solution for complete admitted speech: '+repr(rejected[-3:]))


def registered_layout_proposals(ctx,people,attempt):
    """Enumerate new floor/grid candidates; original native controls admit them."""
    from shapely.geometry import shape,Point
    polygon=shape(ctx['room_floor_polygon']);x0,z0,x1,z1=polygon.bounds
    step=.35 if attempt<8 else .2;listener=ctx['listener_m'];yaw=ctx['camera_yaw_deg']
    floor_height=sum(p[1]for p in ctx['source_proposals'])/len(ctx['source_proposals'])
    points=[];x=x0+.1
    while x<x1:
        z=z0+.1
        while z<z1:
            if polygon.covers(Point(x,z)):
                distance=math.hypot(x-listener[0],z-listener[2])
                if distance>.5:
                    angle=(math.degrees(math.atan2(x-listener[0],-(z-listener[2])))-yaw+180)%360-180
                    points.append((x,z,distance,angle))
            z+=step
        x+=step
    if not points:raise ConstructionUnavailable('room floor proposal enumeration is empty; original native admission remains required')
    chosen=[]
    for index,person in enumerate(people):
        target=(-28.,28.,5.)[index];distance_target=2.8 if index<2 else 1.8
        ranked=sorted(points,key=lambda p:(any(math.hypot(p[0]-q[0],p[1]-q[1])<.65 for q in chosen),
            abs(p[3]-target)/15+abs(p[2]-distance_target),p[0],p[1]))
        from avengine.capture.v7_social_layout import distributed_point
        ranked=[p for p in ranked if distributed_point(p,chosen,minimum=1.)]
        if not ranked:raise ConstructionUnavailable('no distinct registered body proposal remains; no element may be dropped')
        selected=ranked[(attempt//4)%len(ranked)];chosen.append(selected)
    return [[p[0],floor_height,p[1]]for p in chosen]


def _construct_content(row,people,state,semantic,ctx,attempt):
    from avengine.timeline.v7_demand import plan_registered_content_demand
    epoch=choose_authored(semantic,people[:2],state,row['scene_seed'],epoch_count=3,attempt=attempt)
    if len(ctx['source_proposals'])<2 or ctx['room_floor_polygon']is None:
        raise ConstructionUnavailable('registered quartet requires actual room floor and two unqualified source proposals')
    positions=registered_layout_proposals(ctx,people,attempt)
    valid=[]
    for index,position in enumerate(positions):
        look=positions[1-index]if index<2 else[2*position[i]-ctx['listener_m'][i]for i in range(3)]
        d=[look[i]-position[i]for i in range(3)]
        valid.append({'position_m':list(position),'body_yaw_deg':math.degrees(math.atan2(d[0],-d[2])),
            'status':'inherited room proposal only; complete registered native body admission required'})
    row.update(schema='v7-recorded-registered-standing-v1',duration_s=30,assignment_status='admitted',
        seeds={key:stable_seed(row['scene_seed'],key)for key in ('appearance','audio','camera','trajectory','event_schedule','pose','qa')},
        measured_static_layout={'listener_m':ctx['listener_m'],'valid':valid,'room_floor_polygon':ctx['room_floor_polygon'],
            'admission_status':'unqualified candidate; no body or answer truth inherited'},
        production_scenario={'speaker_count':2,'gender':people[0]['visual_gender'][0].upper(),
            'asset_ids':row['asset_ids'],'non_speech_class':None,'category_focus_class':None,
            'camera_yaw_deg':ctx['camera_yaw_deg'],'camera_policy':'hold','camera_sweep_peak_deg':0,
            'semantic_scenario_id':epoch[0][0]['scenario_id'],'semantic_sound_ids':[s['sound_asset_id']for s in epoch[0]],
            'semantic_sound_subset_ids':[s['sound_asset_id']for s in epoch[0]],
            'semantic_epoch_sound_ids':[[s['sound_asset_id']for s in sounds]for sounds in epoch],
            'semantic_role_start_offsets_s':row['semantic_start_offsets_s'],'environment_sound_classes':[]})
    planned=plan_registered_content_demand(row,state,people,semantic)
    # This is a content quartet. Direction gets a separate center-sector scene;
    # do not impose an unrelated fixed camera offset on the quartet.
    angles=[math.atan2(p[0]-ctx['listener_m'][0],-(p[2]-ctx['listener_m'][2]))for p in positions[:2]]
    center=math.degrees(math.atan2(sum(math.sin(a)for a in angles),sum(math.cos(a)for a in angles)))
    planned['production_scenario']['binding_camera_yaw_offset_deg']=(center-ctx['camera_yaw_deg']+180)%360-180
    planned['scene_demand'].update(target_family='V7-CONTENT-BIND',required_independent_family_counts={'V7-CONTENT-BIND':3},required_distinct_semantic_direction_sectors=False)
    return {'profile':'registered','row':planned}
