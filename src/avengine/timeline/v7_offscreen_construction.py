"""Real off-camera people and complete speech; requests never override QA truth."""
from copy import deepcopy
import math

OFFSCREEN_FAMILIES=('QA-04','QA-08','QA-20','QA-24','QA-25','STARSS-FRUSTUM')
ANSWER_TARGETS={'QA-08':('out_of_view',.25,.35),'STARSS-FRUSTUM':('out_of_frame',.30,.40),
    'QA-20':('none_of_visible',.20,.30),'QA-24':('out_of_view',.15,1.)}


def answer_branch_demands(state):
    result={}
    tables=state.get('answer_counts',state.get('family_answer_counts',{}))
    if not tables:tables={k:v.get('answers',{})for k,v in state.get('answer_branch_distribution',{}).get('families',{}).items()}
    for family,(answer,lower,upper)in ANSWER_TARGETS.items():
        counts=tables.get(family,{});total=sum(counts.values());outside=int(counts.get(answer,0))
        share=outside/total if total else None
        result[family]={'answer':answer,'lower_share':lower,'upper_share':upper,
            'actual_outside_count':outside,'actual_total':total,'actual_share':share,
            'requested_branch':('outside'if not total or share<lower else'inside'if share>upper else'either'),
            'authority':'actual deduplicated selected questions; construction is not counted'}
    return result


def _azimuth(point,listener,yaw):
    return (math.degrees(math.atan2(point[0]-listener[0],-(point[1]-listener[2])))-yaw+180)%360-180


def plan_offscreen_floor(ctx,off_count,seed):
    """Choose initial fixed yaw and unqualified floor proposals in any room."""
    from shapely.geometry import shape,Point
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable,stable_seed
    polygon=shape(ctx['room_floor_polygon']);x0,z0,x1,z1=polygon.bounds;listener=ctx['listener_m']
    step=max(.20,math.sqrt(max(polygon.area,1.)/900));points=[];x=x0+step/3
    while x<x1:
        z=z0+step/3
        while z<z1:
            distance=math.hypot(x-listener[0],z-listener[2])
            if distance>=1.5 and polygon.covers(Point(x,z)):points.append((x,z,distance))
            z+=step
        x+=step
    points.sort(key=lambda p:(abs(p[2]-2.8),stable_seed(seed,p[0],p[1])))
    def separate(p,chosen):return all(math.hypot(p[0]-q[0],p[1]-q[1])>=1.05 for q in chosen)
    sectors=[90.,-90.,135.,-135.,180.];offset=seed%len(sectors);sectors=sectors[offset:]+sectors[:offset]
    for target in sectors:
        for anchor in points[:120]:
            bearing=math.degrees(math.atan2(anchor[0]-listener[0],-(anchor[1]-listener[2])))
            yaw=(bearing-target+180)%360-180;outside=[anchor]
            for index in range(1,off_count):
                options=[p for p in points if abs(_azimuth(p,listener,yaw))>=65 and separate(p,outside)]
                options.sort(key=lambda p:(abs(abs(_azimuth(p,listener,yaw))-90)/15,abs(p[2]-3.8)))
                if not options:break
                outside.append(options[0])
            if len(outside)!=off_count:continue
            inside=[]
            for angle,distance in [(-26.,2.8),(26.,3.4),(5.,1.8)]:
                options=[p for p in points if abs(_azimuth(p,listener,yaw))<=33 and separate(p,[*outside,*inside])]
                if len(inside)>=2:
                    a,b=inside[:2];dx,dz=b[0]-a[0],b[1]-a[1]
                    options=[p for p in options if abs(dx*(p[1]-a[1])-dz*(p[0]-a[0]))/max(1e-9,math.hypot(dx,dz))>=.35]
                options.sort(key=lambda p:(abs(_azimuth(p,listener,yaw)-angle)/15+abs(p[2]-distance)))
                if not options:break
                inside.append(options[0])
            if len(inside)==3:return yaw,inside,outside
    raise ConstructionUnavailable('same legal room has no measured-floor proposal for three visible plus requested offscreen people; retry another area/room, never remove a person')


def joint_offscreen_floor(ctx,off_count,seed):
    """Select a new initial legal camera when the inherited point cannot fit.

    This is construction before capture, never a camera trajectory. Native
    support, body contacts, projection and RIR still remeasure the new world.
    """
    from shapely.geometry import shape,Point
    from avengine.timeline.v7_family_constructors import ConstructionUnavailable,stable_seed
    try:
        yaw,inside,outside=plan_offscreen_floor(ctx,off_count,seed)
        return ctx,yaw,inside,outside
    except ConstructionUnavailable as original:
        reason=str(original)
    polygon=shape(ctx['room_floor_polygon']);free=polygon.buffer(-.35)
    if free.is_empty:raise ConstructionUnavailable(reason+'; no native-floor camera proposal with 35cm room-boundary clearance')
    x0,z0,x1,z1=free.bounds;points=[free.representative_point()]
    for ix in range(9):
        for iz in range(9):
            p=Point(x0+(x1-x0)*(ix+.5)/9,z0+(z1-z0)*(iz+.5)/9)
            if free.covers(p):points.append(p)
    chosen=[];ordered=[]
    while points and len(ordered)<32:
        p=max(points,key=lambda p:(min((p.distance(q)for q in chosen),default=free.boundary.distance(p)),
            stable_seed(seed,p.x,p.y)))
        points.remove(p);chosen.append(p);ordered.append(p)
    for index,p in enumerate(ordered):
        candidate=deepcopy(ctx);candidate['listener_m']=[p.x,ctx['listener_m'][1],p.y]
        try:yaw,inside,outside=plan_offscreen_floor(candidate,off_count,seed)
        except ConstructionUnavailable:continue
        candidate['initial_camera_construction']={'method':'legal same-room floor grid; spread candidates before capture',
            'inherited_camera_m':ctx['listener_m'],'selected_camera_m':candidate['listener_m'],
            'candidate_index':index,'capture_camera_moves':False,'physical_native_requalification_required':True}
        return candidate,yaw,inside,outside
    raise ConstructionUnavailable(reason+'; inherited plus '+str(len(ordered))+' alternate legal initial cameras tried without dropping any person')


def construct_offscreen_scene(room,house,state,catalog,records,semantic,contract,runtime,
        *,scene_id,seed,family='QA-08',attempt=0,offscreen_people=None,first_sounding_view=None):
    from avengine.timeline.v7_family_constructors import construct_family_scene,room_context,stable_seed,ConstructionUnavailable,_request
    from avengine.timeline.v7_production_inputs import asset_category,validate_recipe
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    plan=construct_family_scene(room,house,state,catalog,records,semantic,contract,runtime,
        scene_id=scene_id,seed=seed,family='QA-03',attempt=attempt)
    row=plan['row'];ctx=room_context(room,house,attempt);ids=list(row['asset_ids'])
    population=state.get('offscreen_population_request',row['scene_demand'].get('offscreen_population_request',{}))
    if offscreen_people is None:offscreen_people=int(population.get('required_offscreen_sounding_people',1))
    if offscreen_people not in {1,2}:raise ValueError('one or two real offscreen people required')
    if first_sounding_view not in {None,'inside','outside'}:raise ValueError('unsupported first sounding view')
    requested=answer_branch_demands(state).get(family,{}).get('requested_branch','outside')
    outside_first=(first_sounding_view=='outside'if first_sounding_view else requested!='inside')
    assets={a['asset_id']:a for a in catalog};sounds={s['sound_asset_id']:s for s in semantic['sounds']}
    voices={sounds[sid]['voice_reference']['speaker_id']for sid in row['semantic_sound_ids']}
    pool=production_audio_record_views(records);usage=state.get('recording_use_counts',{});entity_usage=state.get('entity_use_counts',{})
    extras=[];chosen_records={};excluded=contract.get('policy',{}).get('temporarily_excluded_asset_ids',{})
    for index in range(offscreen_people):
        options=[]
        for person in catalog:
            if person.get('kind')!='human'or person['asset_id']in {*ids,*extras}or person['asset_id']in excluded:continue
            used_colors={assets[aid].get('color',assets[aid].get('registered_label_color'))
                for aid in [*ids,*extras]if assets[aid].get('controlled_shirt')}
            if person.get('controlled_shirt')and(person.get('color',person.get('registered_label_color'))in used_colors
                    or person.get('color',person.get('registered_label_color'))not in {'black','blue','green','white','yellow'}):continue
            if 'speech_playback'not in person.get('allowed_classes',[])or person.get('visual_gender')not in {'male','female'}:continue
            if asset_category(person,{},room['family'])not in room['categories']:continue
            gender='F'if person['visual_gender']=='female'else'M'
            for record in pool:
                if (record.get('dataset')!='VCTK-Corpus 0.92'or record.get('legacy_library_split')!='train'
                    or record.get('source_type','recorded')!='recorded'or record.get('sound_class')!='speech_playback'
                    or record.get('gender')!=gender or not .8<=record.get('duration_s',0)<=2.5
                    or record.get('speaker_id')in voices or not record.get('transcript')):continue
                from avengine.qa.v7_question_wording import contains_internal_wording
                if contains_internal_wording(record['transcript']):continue
                score=(entity_usage.get(person['asset_id'],0),usage.get(record['sound_asset_id'],0),
                    abs(record['duration_s']-1.),stable_seed(seed,attempt,index,person['asset_id'],record['sound_asset_id']))
                options.append((score,person,record))
        if not options:raise ConstructionUnavailable('no additional real registered person and complete distinct admitted speaking voice in the legal room cell')
        _,person,record=min(options,key=lambda x:x[0]);extras.append(person['asset_id']);chosen_records[person['asset_id']]=record;voices.add(record['speaker_id'])
    ctx,yaw,inside,outside=joint_offscreen_floor(ctx,offscreen_people,seed+attempt)
    row['listener_m']=list(ctx['listener_m'])
    if ctx.get('initial_camera_construction'):row['automatic_construction']['initial_camera_search']=ctx['initial_camera_construction']
    row['asset_ids']=[*ids,*extras];row['camera_yaw_deg']=yaw
    row['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}]
    # Three complete non-overlapping dry utterances per epoch. Wet activity is
    # remeasured natively; it is never assumed to end with these dry intervals.
    offsets=(3.,6.5)if outside_first else(0.,3.)
    row['semantic_start_offsets_s']=list(offsets)
    row['phase2_authored_speech_schedule_s']={aid:[offset+epoch for epoch in (0.,10.,20.)]
        for aid,offset in zip(row['semantic_human_asset_ids'],offsets)}
    row['native_view_roles']={aid:{'view':'outside','static':True,'preferred_azimuth_deg':_azimuth(p,ctx['listener_m'],yaw),
        'minimum_abs_azimuth_deg':60.,'real_registered_drawable_required':True}for aid,p in zip(extras,outside)}
    row['native_placement_preferences_m']={aid:[p[0],ctx['listener_m'][1]-1.5,p[1]]for aid,p in zip([*ids,*extras],[*inside,*outside])}
    for index,aid in enumerate(extras):
        row['source_classes'][aid]=['speech_playback'];row.setdefault('proposed_sound_asset_ids',{})[aid]=chosen_records[aid]['sound_asset_id']
        starts=[epoch+(index if outside_first else 7.+index*.1)for epoch in (0.,10.,20.)]
        row.setdefault('phase2_recorded_speech_schedule_s',{})[aid]=starts
        row.setdefault('phase2_recorded_sound_ids',{})[aid]=[chosen_records[aid]['sound_asset_id']]*len(starts)
    row['automatic_construction'].update(family=family,constructor='construct_offscreen_scene',required_role_count=3+offscreen_people)
    row['offscreen_construction']={'revision':'real-offscreen-people-v1','offscreen_asset_ids':extras,
        'visible_asset_ids':ids,'speaker_count_whole_world':2+offscreen_people,
        'first_sounding_view_requested':'outside'if outside_first else'inside','audio_only_points':0,
        'all_waveforms_complete':True,'all_people_have_one_sound_kind':True,'actual_mask_and_RIR_admission_required':True,
        'QA22_scope':'original visible-entities speaking subset; total live speakers use SPEAKERS; crossing needed for both to coincide'}
    _request(row,state,family);row['scene_demand']['answer_branch_targets']=answer_branch_demands(state)
    row['scene_demand']['required_asset_ids']=list(row['asset_ids'])
    validate_recipe(row,catalog,contract)
    return {'profile':'phase2','row':row}
