"""Registered drawable source compilation for the stationary integration pilot.

All classes retain their admitted recording labels. No audio-only node, moving
source, proxy animal, colour relabel or alternative class is admitted here.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib,json,math
import numpy as np

SCHEMA='v7-visible-registered-pilot-v1'
EXCLUDED_ASSET='v7_object_mantel_clock_wide_bracket_analog_wood_brown_20261012'
PHASE2_CLASSES={'footsteps'}
LIVE_CLASSES={'clapping','coughing','flute_live','guitar_live','hand_drum_live','piano_live','saxophone_live','violin_live'}


def normalized_visible_classes(classes):
    """Map legacy phone permits into the v13 production class, once per entity.

    Source registrations and the published STARSS taxonomy remain unchanged.
    The active answer domain still comes from the admitted recording input.
    """
    aliases={'telephone':'telephone_bell_ringing','phone_ring':'telephone_bell_ringing'}
    return list(dict.fromkeys(aliases.get(value,value) for value in classes))
ANCHOR_CLASS={'hand_clap':'clapping','clapping':'clapping','cough':'coughing','flute':'flute_live','acoustic_guitar':'guitar_live','hand_drum':'hand_drum_live','piano':'piano_live','saxophone':'saxophone_live','violin':'violin_live'}

def production_sound_class(value):
    """Map legacy phone permissions into the admitted v13 production domain.

    Registration bytes and the published STARSS 13-class taxonomy are retained.
    Only live registered-pilot permissions and explicit recipe requests use this
    mapping; actual admitted recording labels remain authoritative.
    """
    return 'telephone_bell_ringing' if value in {'telephone','phone_ring'} else value


def production_audio_record_views(records):
    """Expose the existing calibrated v13 screen under its legacy input key.

    This is a read-only schema view. Keep the BEATs model, actual score,
    owner-fixed threshold, status and SHA untouched; never claim new ASR.
    The original source blocker still validates the actual waveform and
    rejects a failed screen or one bound to a different audio SHA.
    """
    result=[]
    for record in records:
        if (record.get('source_type')=='generated_audio'
                and 'automatic_no_speech' not in record
                and 'human_speech_screen' in record):
            record=deepcopy(record)
            record['automatic_no_speech']=deepcopy(record['human_speech_screen'])
            record['production_screen_schema_mapping']='human_speech_screen -> automatic_no_speech; existing calibrated evidence, no new transcription'
        result.append(record)
    return result

def validate_english_source_admission(source):
    """Reject accidental use of the pre-English v8 pool or its removed clips."""
    policy=source.get('scope_change_5') or {}
    removed=policy.get('removed_sound_asset_ids')
    if not policy.get('evidence') or not isinstance(removed,list) or not removed:
        raise ValueError('integration pilot requires the v9 English language audit')
    ids=[r['sound_asset_id'] for r in source['records']]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate admitted sound identity')
    restored=set(ids).intersection(removed)
    if restored:raise ValueError('removed non-English recording restored: '+','.join(sorted(restored)))
    if policy.get('records_after')!=len(ids):
        generated=[r for r in source['records'] if r.get('source_type')=='generated_audio']
        supplement=source.get('sao_topup',{})
        if policy.get('records_after')+len(generated)!=len(ids) or supplement.get('added_records')!=len(generated):
            raise ValueError('English admission record count differs from audited base plus generated supplements')
        from avengine.timeline.v7_audio_sources import filter_admission
        _,rejected=filter_admission(production_audio_record_views(generated))
        if rejected:raise ValueError('generated supplement admission failed: '+str(rejected))
    return {'status':'pass','admitted_record_count':len(ids),'removed_non_English_records_absent':len(removed),
            'language_audit':policy['evidence'],'new_ASR_run':False}

def speech_class_id(record,visual_gender):
    gender=record.get('gender')
    if gender not in {'F','M'}:raise ValueError('recorded speech needs an admitted voice gender')
    if visual_gender in {'female','male'} and gender!=('F' if visual_gender=='female' else 'M'):
        raise ValueError('recorded voice differs from the registered visual gender')
    return 0 if gender=='F' else 1

def semantic_schedule(offsets,role_index,role_count,duration_s):
    if offsets is None:return [1.+(role_index+1)*.8,16.+(role_index+1)*.8]
    if len(offsets)!=role_count:raise ValueError('semantic timing requires every authored role')
    value=float(offsets[role_index]);ticks=math.ceil(float(duration_s)*10)
    if not math.isfinite(value) or value<0 or abs(value*10-round(value*10))>1e-8 or round(value*10)+ticks>100:
        raise ValueError('whole English authored utterance must fit each 10s window at 100ms ticks')
    return [value,15.+value]

def point(value):
    p=np.asarray(value,dtype=float)
    if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('registered emitter must be a finite metre point')
    return p.tolist()

def _base(asset_id,path,registry,kind):
    path=Path(path)
    if not path.is_file():raise ValueError('registered drawable is missing: '+str(path))
    return {'asset_id':asset_id,'habitat_glb':str(path),'kind':kind,'source_registry':str(registry),
            'actual_artifact_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()},
            'source_registry_sha256':hashlib.sha256(Path(registry).read_bytes()).hexdigest(),
            'standing_registered_qualified':True,'allowed_pose_for_this_task':'stand',
            'attachment_surface':'floor','registered_label_color':None,'label_color':None,
            'silent_decor':False,'entity_class':'static_object','species_id':None}

def human_catalog(registry):
    registry=Path(registry);d=json.loads(registry.read_text());root=Path(d['artifact_root']);rows=[]
    if d.get('schema')!='avengine_v7_people_animals_seats_v1':raise ValueError('expected the registered people/animals/seats roster')
    from avengine.qa.v7_roster_intake import intake_roster
    human_registry=root/'registrations/human_roster.json'
    if json.loads(human_registry.read_text())!=d['humans']:raise ValueError('combined and formal human rosters differ')
    intake=intake_roster(human_registry);formal={a['asset_id']:a for a in intake['records']}
    for major in d['humans']['majors']:
        for v in major['minor_variants']:
            if v['controlled_shirt'] and v['color']=='red':continue
            if not v.get('dual_backend_static_usable') or v['habitat']['status']!='pass':raise ValueError('human static registration is not admitted')
            a=_base(v['variant_id'],root/v['habitat']['glb'],human_registry,'performer' if major['source']=='lx' else 'human')
            if not formal[v['variant_id']]['standing_registered_qualified']:raise ValueError('formal human intake blocked: '+str(formal[v['variant_id']]['static_blockers']))
            a.update(source_asset_roster=str(registry),source_asset_roster_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),formal_intake_warnings=formal[v['variant_id']]['upstream_warnings'])
            a.update(body_identity_id=major['major_id'],visual_gender=major['gender'],name_zh=major['name_zh'],
                     mouth_local_glb_m=point(v['mouth_local_glb_m']),controlled_shirt=bool(v['controlled_shirt']),
                     color=v['color'],license_authority='owner formal static registration',
                     source=major['source'],registered_label_color=v['color'] if v['controlled_shirt'] else None,
                     label_color=v['color'] if v['controlled_shirt'] else None)
            anchor=major.get('sound_anchor') or v.get('primary_sound_anchor')
            if anchor and anchor.get('sound_class') in ANCHOR_CLASS:
                if not anchor.get('status','').startswith('pass'):raise ValueError('performer anchor not admitted')
                a.update(emitter_local_m=point(anchor['primary_local_glb_m']),emitter_part=anchor['primary_part'],
                         allowed_classes=[ANCHOR_CLASS[anchor['sound_class']]],anchor_registration=deepcopy(anchor))
            else:
                a.update(emitter_local_m=a['mouth_local_glb_m'],emitter_part='mouth',allowed_classes=['speech_playback','laughter','coughing'])
            a['entity_class']='articulated_human';a['species_id']='human';rows.append(a)
    return rows

def object_catalog(registry):
    registry=Path(registry);d=json.loads(registry.read_text());rows=[];excluded=[]
    if d.get('schema')!='avengine_v7_static_object_roster_v1':raise ValueError('unsupported object registration')
    for r in d['assets']:
        if r['asset_id']==EXCLUDED_ASSET:
            excluded.append({'asset_id':r['asset_id'],'reason':'owner exclusion: measured highlight 5.49% exceeds 2%'});continue
        if r.get('status')!='pass':raise ValueError('object registration is not admitted')
        a=_base(r['asset_id'],r['derived_glb'],registry,'object')
        a.update(name_zh=r.get('object_type','object'),object_type=r['object_type'],body_identity_id=r['asset_id'],
                 form_factor=r.get('form_factor'),attachment_surface=r['attachment_surface'],
                 allowed_classes=normalized_visible_classes(r['allowed_v7_classes_candidate']),silent_decor=r.get('silent_decor',False),
                 registered_label_color=r.get('label_color'),label_color=r.get('label_color'),
                 geometry=deepcopy(r['geometry_checks']),asset_metadata=r['asset_metadata'],
                 license_authority=deepcopy(r.get('license')),material_receipt=deepcopy(r.get('material_receipt')))
        if a['attachment_surface'] not in {'floor','table','wall','ceiling'}:raise ValueError('unsupported attachment surface')
        if a['silent_decor']:
            if r.get('emitter') is not None or a['allowed_classes']:raise ValueError('silent decoration has an emitter or sound class')
            a.update(emitter_local_m=None,emitter_part=None)
        else:
            if not r.get('emitter',{}).get('active') or not a['allowed_classes']:raise ValueError('active object emitter is missing')
            a.update(emitter_local_m=point(r['emitter']['local_m']),emitter_part=r['emitter']['physical_rationale'])
        if a['object_type']=='smoke_detector' and a['attachment_surface']=='ceiling':
            a['label_color']=None;a['color_question_policy']='owner forbids ceiling smoke-detector colour labels'
        if {'music_playback','speech_playback'}.intersection(a['allowed_classes']):a['entity_class']='loudspeaker'
        rows.append(a)
    return rows,excluded

def validate_scene_bindings(scene,assets):
    by_ep={a['endpoint_id']:a for a in assets}
    if len(by_ep)!=len(assets) or len({a['native_instance_id'] for a in assets})!=len(assets):raise ValueError('duplicate physical entity identity')
    if set(scene['sources'])!=set(by_ep):raise ValueError('source lacks a registered drawable entity')
    errors=[]
    for event in scene['events']:
        a=by_ep.get(event['endpoint_id']);spec=scene['sources'].get(event['endpoint_id'],{})
        cls=event.get('sound_class',spec.get('sound_class'))
        if cls in PHASE2_CLASSES:errors.append('footsteps require second-stage moving people')
        if a is None or a.get('silent_decor') or cls not in a.get('allowed_classes',[]):errors.append('sound class has no permitted registered emitter')
        if spec.get('native_visual_kind')=='audio_only' or spec.get('asset_id')!=(a or {}).get('asset_id'):errors.append('invisible or mismatched source')
        if a and spec.get('actor_id')!=a['native_instance_id']:errors.append('source/entity identity differs')
    if errors:raise ValueError('; '.join(sorted(set(errors))))
    return {'status':'pass','events':len(scene['events']),'registered_entities':len(assets),'invisible_sources':0,'footsteps_events':0}

def audit_native_bindings(scene,assets,visual):
    validate_scene_bindings(scene,assets);by_ep={a['endpoint_id']:a for a in assets};active={e['endpoint_id'] for e in scene['events']}
    errors=[];worst=0.;readbacks=0;appearance_counts={a['asset_id']:0 for a in assets}
    for frame in visual['frames']:
        actual={r['endpoint_id']:r for r in frame['actors']}
        if len(actual)!=len(assets) or set(actual)!=set(by_ep):errors.append('native entity disappeared')
        for ep,a in by_ep.items():
            r=actual.get(ep)
            if r is None:continue
            if r['asset_id']!=a['asset_id'] or r['actor_id']!=a['native_instance_id']:errors.append('native entity changed registration')
            if r.get('visible_pixels',0)>=100:appearance_counts[a['asset_id']]+=1
            if ep not in active:continue
            expected=(np.asarray(r['body_to_world_m'])@np.r_[a['emitter_local_m'],1.])[:3]
            err=float(np.linalg.norm(np.asarray(r['emitter_position_m'])-expected));worst=max(worst,err);readbacks+=1
            if err>.0001:errors.append('emitter readback exceeds 0.1mm')
            if r.get('native_object_id') is None:errors.append('emitter has no drawable native object')
    invisible=sum(ep not in {r['endpoint_id'] for r in visual['frames'][0]['actors']} for ep in active)
    from avengine.capture.v7_offscreen_layout import coverage_errors
    errors.extend(coverage_errors(scene,assets,visual,appearance_counts))
    return {'status':'pass' if not errors else 'fail','invisible_sources':invisible,'emitter_readbacks':readbacks,
            'emitter_worst_error_m':worst,'asset_visible_frame_counts':appearance_counts,'errors':sorted(set(errors))}


def conditioned_recorded_duration_fits(record,asset_id,row):
    duration=float(record.get('duration_s',0))
    if .8<=duration<=2.5:return True
    # This is constructor scheduling, not source admission. The untouched v3
    # solver has already fitted this exact whole admitted recording and tail.
    if duration<.8:return False
    for item in row.get('motion_during_condition_plan',{}).get('plans',[]):
        chosen=item.get('config',{}).get('sounds',{}).get(asset_id,{})
        if item.get('solution',{}).get('status')=='solved' and chosen.get('sound_asset_id')==record['sound_asset_id']:
            return True
    return False


def select_proposed_record(pool, asset_id, row, rng):
    """Resolve an explicit candidate clip inside the already admitted pool."""
    proposed=row.get('proposed_sound_asset_ids',{}).get(asset_id)
    if proposed is not None:
        matches=[r for r in pool if r['sound_asset_id']==proposed]
        if len(matches)!=1:raise ValueError('proposed recording is absent from the exact admitted class/voice pool: '+asset_id+'/'+proposed)
        return matches[0]
    return pool[int(rng.integers(len(pool)))]


def recorded_event_layout(row,asset,record,human_speech,fallback):
    """Keep a whole admitted animal recording on an explicit physical clock."""
    starts=row.get('phase2_whole_source_recordings',{}).get(asset['asset_id'])
    if starts is not None:
        if record.get('sound_class')=='footsteps' or asset['kind']not in {'animal','human'}:raise ValueError('whole source schedule requires a registered physical non-footstep emitter')
        if any(not math.isfinite(float(t))or float(t)<0 or float(t)+float(record['duration_s'])>30 for t in starts):raise ValueError('whole source schedule exceeds parent')
        return [float(t)for t in starts],True
    starts=row.get('phase2_sound_program_schedule_s',{}).get(asset['asset_id'])
    if starts is not None:return [float(t)for t in starts],False
    starts=row.get('phase2_whole_animal_recordings',{}).get(asset['asset_id'])
    if starts is None:return fallback,human_speech
    if asset['kind']!='animal' or asset['asset_id'] not in set(row.get('motion_requests',{}))|set(row.get('phase2_moving_asset_ids',[])):
        raise ValueError('whole animal recording requires a registered moving animal')
    if record.get('source_type')=='author_tts':raise ValueError('animal cannot emit authored human speech')
    starts=[float(t)for t in starts]
    if not starts or any(not math.isfinite(t) or t<0 or t+float(record['duration_s'])>30 for t in starts):
        raise ValueError('whole animal recording schedule does not fit the parent')
    return starts,True


def build_visible_scene(row,house,records,catalog,semantic_manifest=None):
    from avengine.timeline.v7_audio_sources import PROGRAM_SCHEMA,filter_admission
    from avengine.timeline.v7_environmental_sources import recording_header,STRICT_CLASS13
    from avengine.timeline.v7_registered_sampling import semantic_record
    if row.get('schema')!=SCHEMA or row['split']!='train' or house['split']!='train' or row['house']!=house['house']:
        raise ValueError('integration pilot requires a training-house row')
    if house['acoustic_status']!='pass' or row['room_id'] not in house['checked_room_ids']:raise ValueError('room lacks existing acoustic admission')
    if any(name in row['house'] for name in ('y9hTuugGdiq','4ok3usBNeis','q5QZSEeHe5g','QaLdnwvtxbs','eF36g7L6Z9M','qyAac8rV8Zk','wcojb4TFT35','zt1RVoi7PcG','CrMo8WxCyVb','VBzV5z6i1WS','svBbv1Pavdk','6s7QHgap2fW')):
        raise ValueError('owner prohibited testing house')
    if row.get('motion_requests'):raise ValueError('walking/sitting/moving RIR belongs to stage two')
    admitted,rejected=filter_admission(production_audio_record_views(records))
    if rejected:raise ValueError('v9 source pool contains unadmitted records: '+str(rejected[:3]))
    by_id={a['asset_id']:a for a in catalog};assets=[deepcopy(by_id[aid]) for aid in row['asset_ids']]
    if len(assets)>16:raise ValueError('pilot entity capacity exceeded')
    working=deepcopy(house);room=next(r for r in working['checked_rooms'] if r['room_label']==row['room_id']);working['room']=deepcopy(room)
    listener=point(row['listener_m']);yaw=float(row.get('camera_yaw_deg',0));angle=math.radians(yaw)
    working['room'].setdefault('placement',{})['camera_m']=list(listener)
    scene={'scene_id':row['scene_id'],'independent_parent_scene_id':row['scene_id'],'seed':row['scene_seed'],
        'split':'train','duration_s':30.,'video_fps':15,'sample_rate_hz':16000,'listener_m':listener,
        'camera_forward':[math.sin(angle),0.,-math.cos(angle)],
        'camera_yaw_knots':[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}],
        'window_starts_s':[0.,15.],'sources':{},'events':[],
        'production_scenario':{'speaker_count':0,'non_speech_class':None},
        'admitted_sound_class_domain':sorted({r['sound_class'] for r in admitted}-PHASE2_CLASSES),
        'native_placement_candidates':[], 'source_motion_policy':'stationary original registered entities',
        'sampling_limitations':['pilot only; no train-table export','walking/moving RIR/external seats deferred to stage two']}
    if row.get('native_attachment_supports'):scene['native_attachment_supports']=deepcopy(row['native_attachment_supports'])
    for key in ('native_view_roles','native_placement_preferences_m','offscreen_construction','owner_social_layout','automatic_construction',
                'native_distance_bands_m','native_dynamic_occlusion_relations','native_facing_offsets_deg'):
        if row.get(key):scene[key]=deepcopy(row[key])
    program={'schema':PROGRAM_SCHEMA,'records':[],'events':[]};used={};rng=np.random.default_rng(row['scene_seed']);used_speakers=set()
    authored=[]
    if row.get('semantic_scenario_id'):
        if not semantic_manifest:raise ValueError('authored semantic manifest missing')
        wanted=row.get('semantic_sound_ids');authored=[s for s in semantic_manifest['sounds'] if s['scenario_id']==row['semantic_scenario_id'] and (not wanted or s['sound_asset_id'] in wanted)]
        if wanted:authored=sorted(authored,key=lambda s:wanted.index(s['sound_asset_id']))
        authored_ids=row.get('semantic_human_asset_ids')
        if authored_ids is None:authored_ids=[a['asset_id'] for a in assets if a['kind']=='human']
        humans={a['asset_id'] for a in assets if a['kind']=='human'}
        if len(authored_ids)!=len(set(authored_ids)) or not set(authored_ids)<=humans:
            raise ValueError('authored roles must bind distinct actual registered human bodies')
        if len(authored)!=len(authored_ids):raise ValueError('complete authored roles need matching humans')
        scene['production_scenario'].update(semantic_scenario_id=row['semantic_scenario_id'],semantic_sound_ids=[s['sound_asset_id'] for s in authored])
    human_index=0
    for i,a in enumerate(assets):
        ep='voice'+str(i+1);a.update(endpoint_id=ep,native_instance_id=i+1)
        classes=[c for c in a['allowed_classes'] if c not in PHASE2_CLASSES]
        requested=row.get('source_classes',{}).get(a['asset_id'])
        if requested is not None:classes=normalized_visible_classes(requested)
        elif a['kind']=='human':classes=['speech_playback']
        elif a['kind']=='performer' and len(classes)>1:classes=['speech_playback']
        if not a['silent_decor'] and not classes:raise ValueError('registered asset has no stage-one sound mechanism')
        human_speech=a['kind'] in {'human','performer'} and classes==['speech_playback']
        mechanism='human_live_recorded' if human_speech else ('human_live_recorded_'+classes[0] if a['kind'] in {'human','performer'} else ('animal_live_recorded' if a['kind']=='animal' else 'loudspeaker_playback' if a['entity_class']=='loudspeaker' else 'device_recorded_event'))
        source={'actor_id':i+1,'asset_id':a['asset_id'],'entity_class':a['entity_class'],'species_id':a['species_id'],
            'class_id':(0 if a['visual_gender']=='female' else 1) if human_speech and a.get('visual_gender') in {'female','male'} else STRICT_CLASS13.get(classes[0]) if classes else None,
            'content_kind':'speech' if human_speech else classes[0] if classes else 'silent',
            'mechanism':mechanism,'sound_class':classes[0] if len(classes)==1 else None,'silent_decor':a['silent_decor'],'native_visual_kind':'registered_drawable','dry_gain':.6,
            'trajectory':[{'time_s':0.,'position_m':listener},{'time_s':30.,'position_m':listener}]}
        scene['sources'][ep]=source
        if a['silent_decor']:continue
        if human_speech:scene['production_scenario']['speaker_count']+=1
        choices={}
        if a['kind']=='human' and authored and a['asset_id'] in authored_ids:
            s=authored[human_index];g='F' if a['visual_gender']=='female' else 'M';record=semantic_record(s,g);choices['speech_playback']=record;human_index+=1
        else:
            for cls in classes:
                pool=[r for r in admitted if r['sound_class']==cls]
                if human_speech:
                    pool=[r for r in pool if r.get('dataset')=='VCTK-Corpus 0.92' and r.get('legacy_library_split')=='train' and r.get('gender')==('F' if a.get('visual_gender')=='female' else 'M') and r.get('speaker_id') not in used_speakers and conditioned_recorded_duration_fits(r,a['asset_id'],row)]
                elif cls=='speech_playback':
                    pool=[r for r in pool if r.get('dataset')!='VCTK-Corpus 0.92'] or pool
                if not pool:raise ValueError('no exact admitted recording for '+a['asset_id']+'/'+cls)
                record=recording_header(select_proposed_record(pool,a['asset_id'],row,rng));choices[cls]=record
                if human_speech:used_speakers.add(record['speaker_id'])
        if human_speech:
            source['class_id']=speech_class_id(choices['speech_playback'],a.get('visual_gender'))
            source['voice_gender']=choices['speech_playback']['gender']
            source['visual_gender_status']='registered' if a.get('visual_gender') in {'female','male'} else 'unknown_not_inferred_from_voice'
        schedule=[.2+.1*i+5*j for j in range(6)]
        if a['kind']=='human' and authored and a['asset_id'] in authored_ids:schedule=semantic_schedule(row.get('semantic_start_offsets_s'),human_index-1,len(authored),choices['speech_playback']['duration_s'])
        if human_speech and not authored and a['asset_id'] in row.get('phase2_recorded_speech_schedule_s',{}):
            schedule=[float(t)for t in row['phase2_recorded_speech_schedule_s'][a['asset_id']]]
            if not schedule or any(not math.isfinite(t) or t<0 or t>=30 for t in schedule):
                raise ValueError('explicit whole recorded speech schedule is invalid')
        if a['asset_id'] in row.get('phase2_whole_animal_recordings',{}) and len(choices)!=1:
            raise ValueError('explicit whole animal recording needs one admitted sound class')
        schedule,whole=recorded_event_layout(row,a,next(iter(choices.values())),human_speech,schedule)
        for index,start in enumerate(schedule):
            cls=classes[index%len(classes)];record=choices[cls]
            duration=float(record['duration_s']);ticks=math.ceil(duration*10) if whole else max(1,math.floor(min(2.,duration)*10))
            measured_program=row.get('phase2_animal_program_samples',{}).get(a['asset_id'])
            if measured_program is not None:
                n=int(measured_program['sample_count']);begin=int(measured_program.get('crop_start_sample',0))
                if (a['kind']!='animal' and not (a['kind']=='human' and record['sound_class']in {'laughter','coughing'})) or n<=0 or n%1600 or begin<0 or begin+n>record['header']['sample_count']:
                    raise ValueError('invalid complete measured animal activity program')
                ticks=n//1600;whole=False
            lo=round(start*10);hi=lo+ticks
            if hi>300:raise ValueError('complete utterance does not fit the continuous world')
            activity=record.get('screens',{}).get('automatic_activity',{}).get('activity_intervals_samples') or record.get('source_normalization',{}).get('measured_activity',{}).get('source_activity_intervals_samples')
            if not activity:raise ValueError('source has no measured activity')
            crop=0 if whole else max(0,min(int(activity[0][0]),record['header']['sample_count']-ticks*1600))
            if measured_program is not None:
                crop=int(measured_program.get('crop_start_sample',0))
                if measured_program.get('tail_muted'):
                    from .v7_tail_muting import read_tail_muted_program
                    _,processed_activity=read_tail_muted_program(record,measured_program)
                    if not processed_activity['activity_intervals_samples']:raise ValueError('tail-muted program has no actual activity')
                else:
                    required=measured_program.get('required_activity_intervals_samples',activity)
                    if measured_program.get('selected_activity_index')is not None and required!=[activity[int(measured_program['selected_activity_index'])]]:raise ValueError('selected activity must match the original measured complete call')
                    if any(int(lo)<crop or int(hi)>crop+ticks*1600 for lo,hi in required):
                        raise ValueError('measured animal program would remove observed animal activity')
            eid=f'event_{len(scene["events"]):04d}';cid=source['class_id'] if human_speech else STRICT_CLASS13.get(cls)
            scene['events'].append({'event_id':eid,'endpoint_id':ep,'source_id':i+1,'class_id':cid,'sound_class':cls,'start_tick':lo,'end_tick':hi,'mechanism':mechanism})
            program['events'].append({'event_id':eid,'sound_asset_id':record['sound_asset_id'],'crop_start_sample':crop,
                **({'preserve_full_recorded_clip':True} if whole and record.get('source_type')!='author_tts' else {}),
                **({'tail_muting':deepcopy(measured_program)} if measured_program is not None and measured_program.get('tail_muted') else {})})
            used[record['sound_asset_id']]=record
    program['records']=list(used.values());working['room']['placement']={**working['room']['placement'],'camera_m':listener}
    validate_scene_bindings(scene,assets)
    return working,scene,program,assets




def animal_catalog(registry):
    registry=Path(registry);d=json.loads(registry.read_text());rows=[]
    if d.get('schema')!='avengine_v7_animal_variants_20261012_v1':raise ValueError('unsupported final animal registration')
    for r in d['animals']:
        if r['status']!='admitted' or r['emit_head_status']!='pass':raise ValueError('animal static body or mouth not admitted')
        for family in ('hm3d','mp3d'):
            if r['native_qa'][family]['status']!='pass':raise ValueError('animal dual-scene static registration failed')
        a=_base(r['asset_id'],r['habitat_glb'],registry,'animal')
        if a['actual_artifact_sha256'][a['habitat_glb']]!=r['runtime_sha256']:raise ValueError('animal registered GLB changed')
        category=r['category'];bird_family=r.get('sound_appearance_family')
        cls='dog_bark' if category=='dog' else 'cat_meow' if category=='cat' else 'pet_bird_canary' if bird_family=='canary' else 'pet_bird_parrot' if bird_family=='parrot' else None
        if cls is None:raise ValueError('animal body lacks an exact stage-one call class')
        a.update(name_zh=r['body_name_zh'],body_identity_id=r['body_id'],species_id=category,
            entity_class='articulated_animal',allowed_classes=[cls],emitter_local_m=point(r['emit_point_local_m']),emitter_part='registered mouth/beak',
            mouth_local_glb_m=point(r['emit_point_local_m']),registered_label_color=r['label_color'],label_color=r['label_color'],
            body_only_glb=r.get('body_only_glb'),body_floor_y_m=r.get('body_floor_y_m',0.),
            landmarks_local_m=deepcopy(r.get('landmarks_local_m',{})),dimensions_m=r['dimensions_m'],
            license_authority=deepcopy(r['license']),asset_metadata=r['asset_metadata'],coordinates=deepcopy(r.get('coordinates',{})),
            source_mode=r['mode'],upstream_walk_allowed=r['walk_allowed'],pilot_walk_allowed=False,
            color_calibration=deepcopy(d['color_calibration']),static_native_receipts={k:r['native_qa'][k]['receipt'] for k in ('hm3d','mp3d')})
        rows.append(a)
    return rows
