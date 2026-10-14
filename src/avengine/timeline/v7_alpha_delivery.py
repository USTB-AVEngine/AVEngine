"""Portable alpha delivery and measured rejection accounting.

Only immutable qualified parent seals are exported. Natural-language training
labels are checked by the existing scorer; private canonicals stay unchanged.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv,json,os,shutil,statistics
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_production_contract import validate_question_counts,question_allocation
from avengine.qa.unified_scoring import score_unified_item
from avengine.timeline.v7_english_export import english_public_question,validate_english_model_text
from avengine.timeline.v7_question_independence import independence_key,INDEPENDENCE_REVISION


def load(path):return json.loads(Path(path).read_text())


def _copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise ValueError('export must not replace an existing artifact')
    from avengine.timeline.v7_storage_budget import ENV_KEY,check_write
    policy=json.loads(os.environ[ENV_KEY]) if os.environ.get(ENV_KEY) else {}
    owner_root=Path(policy['root']).resolve() if policy.get('root') else None
    afs=os.environ.get('AVENGINE_V7_AFS_OWN_ROOT')
    if afs:
        afsroot=Path(afs).resolve()
        if not str(afsroot).startswith('/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_'):raise ValueError('invalid private AFS artifact root')
        owner_root=afsroot
    own_pair=(owner_root is not None and Path(source).resolve().is_relative_to(owner_root)
              and Path(target).resolve().is_relative_to(owner_root))
    if own_pair:
        try:os.link(source,target);return
        except OSError:pass
    # Original datasets and other deliveries stay read-only, including their
    # inode/link metadata. Only our own immutable artifacts are hardlinked.
    check_write(target,expected_bytes=Path(source).stat().st_size)
    shutil.copyfile(source,target)


def _answer_text(value):
    if isinstance(value,(dict,list,tuple)):return json.dumps(value,ensure_ascii=False)
    return str(value)


def readable_training_answer(question,gold,authority):
    """Translate an internal ID only through the retained original label map."""
    if gold.get('authority_question_id'):
        item=authority[gold['authority_question_id']]
        label=gold.get('authority_truth',{}).get('label',gold['canonical_answer'])
        options=[_answer_text(label),_answer_text(gold['canonical_answer'])]
        if question['form']=='mcq':
            form=item['forms']['mcq'];index=form['gold']['correct_index'];choice=form['options'][index]
            options.insert(0,_answer_text(choice.get('label_en',choice.get('label_zh',''))))
        for value in dict.fromkeys(options):
            score=score_unified_item(item,value,form=question['form'])
            if score.get('status')=='scored' and score.get('score')==1.:
                return value,{'status':'pass','method':'unchanged_v6_scorer','result':score}
        raise ValueError('readable training target is not accepted by the original scorer')
    if gold.get('authority')=='v7_content_binding_v1':
        from avengine.qa.v7_content_binding import content_truth,admission_blockers
        evidence=gold['evidence'];actual=content_truth(evidence['scene'],evidence['query'])['answer']
        if actual!=gold['canonical_answer'] or admission_blockers(evidence['scene'],evidence['query'],binding_witness=evidence['native_witness']):raise ValueError('original content-binding authority or actual pair admission failed')
        from avengine.qa.v7_english_surface import content_answer_en
        return content_answer_en(actual),{'status':'pass','method':'unchanged_v7_content_binding_authority_and_native_pairs','canonical_preserved':True}
    return _answer_text(gold['canonical_answer']),{'status':'canonical_preserved','method':'original_v0.2_protocol_canonical'}


def _csv(path,rows,fields):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def export_alpha(seeds,seal_roots,output,expected_parents=None,sealed_parent_map=None):
    rows=[json.loads(line) for line in Path(seeds).read_text().splitlines() if line.strip()]
    if expected_parents is not None and len(rows)!=expected_parents:raise ValueError('seed count differs from the release request')
    if len({r['scene_id'] for r in rows})!=len(rows):raise ValueError('duplicate parent seeds')
    target=Path(output)
    if target.exists():raise ValueError('select a fresh immutable export directory')
    target.mkdir(parents=True)
    preferred=load(sealed_parent_map)['parents'] if sealed_parent_map else {}
    if preferred and set(preferred)!={r['scene_id'] for r in rows}:raise ValueError('explicit seal selection differs from the release seeds')
    chosen={};parent_rows=[];window_rows=[];training=[];family=Counter();modality=Counter();rejects=Counter();candidate_rejects=Counter();protocol_rejects=Counter()
    material_rows=[];directions=[];activity_counts=Counter();speaker_counts=Counter();panel_family=Counter();temporal=Counter();used_source_ids=set();used_speakers=set();native_checks=[];native_visual_cpu=[];sources={};reference_cpu=[];profiles=set();all_gold={};public_by_window={};classes=Counter();sound_classes=Counter();genders=Counter();mechanisms=Counter();humans={};moving_windows=0;tts_event_count=0
    for row in rows:
        candidates=[]
        search=[Path(preferred[row['scene_id']])] if preferred else [Path(root)/'parents'/row['scene_id'] for root in seal_roots]
        for parent in search:
            pointer=parent/'alpha_seal_completed.json'
            if pointer.is_file():candidates.append((parent,load(pointer)))
        if len(candidates)!=1:raise ValueError(f'{row["scene_id"]} has {len(candidates)} qualified seals; explicit unique input required')
        parent,pointer=candidates[0];seal=parent/pointer['attempt'];result=pointer['result'];profiles.add(result['profile'])
        for asset in pointer['recipe']['input_recipe']['assets']:humans[asset['asset_id']]=asset
        if result['status']!='formal_alpha_admitted' or result['full_v7_generation_ready']:raise ValueError('wrong alpha release status')
        if pointer['recipe']['input_recipe']['seed_row']!=row:raise ValueError('qualified seed differs from the requested seed')
        for name in pointer['required_files']:
            p=seal/name
            if not p.is_file():raise ValueError(f'qualified seal lacks {name}')
            if Path(name).name!='questions.json':_copy(p,target/'parents'/row['scene_id']/name)
        delivery=target/'parents'/row['scene_id']
        chosen[row['scene_id']]={'source_parent_directory':str(parent),'source_seal_directory':str(seal),'portable_delivery_directory':str(delivery.relative_to(target))}
        times=[]
        for root in seal_roots:
            for shard in Path(root).glob('shard_*.json'):
                times.extend(r for r in load(shard)['parents'] if r['scene_id']==row['scene_id'] and r['status']=='formal_alpha_admitted' and not r.get('resumed_sealed_delivery') and Path(r['delivery_directory']).resolve()==seal.resolve())
        if len(times)!=1:raise ValueError('full successful attempt timing is missing or ambiguous')
        measured=times[0]
        parent_rows.append({'scene_id':row['scene_id'],'split':row['split'],'house':row['house'],'room_id':row['room_id'],
            'formal_windows':result['alpha_formal_windows'],'logical_questions':result['alpha_formal_logical_questions'],
            'wall_seconds':measured['wall_seconds'],'cpu_seconds_including_children':measured['cpu_seconds_including_children'],
            'RIR_calls':result['measured_RIR_calls'],'native_visual_worlds':len({m.split('_')[0] for m in row['members']})})
        reference_cpu.append(measured['cpu_seconds_including_children'])
        native_visual_cpu.extend(t['cpu_seconds_including_children'] for t in measured['stage_timings'] if t['stage'].startswith('visual_') and not t['resumed'])
        program=load(seal/'private/dry_program.json');receipt=load(seal/'private/dry_emission_receipt.json')
        used_source_ids.update(r['original_id'] for r in program['records']);used_speakers.update(x for x in receipt['endpoint_speaker_ids'].values() if isinstance(x,str) and x)
        for source in program['records']:sources[source['sound_asset_id']]=source
        # Retain the exact persistent RIR payloads used for this parent. The
        # production process is finished before export; database files are proof,
        # never a mutable shared AFS cache for concurrent producers.
        cache=parent/'rir_cache'
        if cache.is_dir():
            for p in cache.rglob('*'):
                if p.is_file() and not p.name.startswith('.') and not p.name.endswith(('-wal','-shm','.lock')):
                    _copy(p,delivery/'private/rir_cache'/p.relative_to(cache))
        for v in sorted({m.split('_')[0] for m in row['members']}):
            native=seal/'private/native'/v
            visual=load(native/('native_registered_visual_evidence.json' if (native/'native_registered_visual_evidence.json').is_file() else 'native_alpha_visual_evidence.json'))
            # UE retains native BaseColor-highlight metrics; withdrawn raw
            # texture-sRGB color errors are absent, never manufactured as zero.
            for observation in visual['source_material_checks']['checks']:
                if 'metrics' in observation:
                    metric=observation['metrics']
                    material_rows.append({**observation,'status':metric['status'],
                        'highlight_ratio':metric.get('highlight_ratio'),
                        'texture_chromaticity_error_p50':None,
                        'texture_chromaticity_error_p95':None})
                else:material_rows.append(observation)
            native_checks.append({'scene_id':row['scene_id'],'visual_variant':v,'frame_count':visual['frame_count'],
                'alignment':visual['alignment'],'standing_bodies':len(visual['standing_admission']),
                'known_bad_contacts_detected':all(r['known_bad_detected'] for r in visual['standing_admission']),
                'contact_tolerance_m':visual.get('native_contact_tolerance_m',.02 if 'UE' in str(visual.get('renderer',{}).get('backend','')) else .015),'material_status':visual['source_material_checks']['status']})
        for factor in sorted({m.split('_')[1] for m in row['members']}):
            audio=load(seal/'private/native'/(factor+'_audio_evidence.json'))
            directions.extend(audio['directions']);
        for win in result['windows']:
            wid=win['window_id'];q=load(seal/'windows'/wid/'questions.json');gold=load(seal/'private'/(wid+'_answers.json'));by_id={g['question_id']:g for g in gold};all_gold.update(by_id);public_by_window[(row['scene_id'],wid)]=[]
            quota=row['question_quota_base']+row['members'].index(win['member'])*row['window_count']+int(wid.rsplit('w',1)[1])
            validate_question_counts(q,gold,expected_allocation=question_allocation(quota,14))
            selected=load(parent/('selected_windows_'+win['member']+'_completed.json'))
            raw_dir=Path(selected['recipe']['pool_directory'])/wid;audit=load(raw_dir/'candidate_audit.json');authority={x['question_id']:x for x in audit['v6_items']}
            _copy(raw_dir/'candidate_audit.json',delivery/'private'/(wid+'_candidate_audit.json'))
            for rejected in audit['deferred']:rejects[(rejected.get('qa_id','unknown'),rejected.get('code',rejected.get('reason','unknown')))]+=1
            for attempt in audit['v6_generation'].get('candidate_attempts',[]):
                if attempt.get('status')=='candidate_rejected':candidate_rejects[(attempt.get('qa_id','unknown'),attempt.get('code',attempt.get('reason','unknown')))]+=1
            for excluded in audit['protocol']['exclusions']:
                for qa in excluded.get('qa_ids',[excluded.get('qa_id','unspecified_candidate')]):protocol_rejects[(qa,excluded.get('reason','unknown'))]+=1
            evidence=load(seal/'private'/(wid+'_evidence.json'));segments=audit['protocol']['segments']
            classes.update(s['class_id'] for s in segments)
            event_bindings={r['event_id']:r for r in receipt['events']};record_by_id={r['sound_asset_id']:r for r in program['records']}
            lo=round(win['start_s']*16000);hi=lo+160000
            scene=evidence['scene']
            moving_windows+=int(any(any(k['position_m']!=spec['trajectory'][0]['position_m'] for k in spec['trajectory']) for spec in scene['sources'].values()))
            for event in scene['events']:
                b=event_bindings[event['event_id']]
                if b['output_start_sample']>=hi or b['output_end_sample_exclusive']<=lo:continue
                record=record_by_id[b['sound_asset_id']]
                sound_classes[record.get('sound_class','unknown')]+=1;mechanisms[event.get('mechanism','unknown')]+=1
                if event['class_id'] in {0,1}:genders['F' if event['class_id']==0 else 'M']+=1
                tts_event_count+=int(record.get('source_type')=='author_tts')
            activity_counts[len(segments)]+=1;speaker_counts[len({s['source_id'] for s in segments if s['class_id'] in {0,1}})]+=1
            temporal['carry_in_windows']+=int(any(s.get('carry_in') for s in segments))
            temporal['more_than_6_activity_segments_windows']+=int(len(segments)>6)
            temporal['same_actor_repeated_windows']+=int(max(Counter(s['source_id'] for s in segments).values(),default=0)>=2)
            for question in q:
                if question['qa_id']=='QA-14' and '这两个个体中' not in question['question_zh']:
                    raise ValueError('QA-14 uses the obsolete unnamed-pair public stem; replay this parent with the current code')
                answer,check=readable_training_answer(question,by_id[question['question_id']],authority)
                folder=Path('parents')/row['scene_id']/'windows'/wid
                # This is explicitly supervised train/dev data. Runtime E/S
                # predictions are not manufactured from private scene truth.
                english=english_public_question(question)
                public_by_window[(row['scene_id'],wid)].append(english)
                validate_english_model_text({**english,'model_answer':answer})
                training.append({'question_id':question['question_id'],'logical_question_id':question['logical_question_id'],
                    'independent_parent_scene_id':row['scene_id'],'window_id':wid,'window_start_s':win['start_s'],'split':row['split'],'house':row['house'],
                    'qa_id':question['qa_id'],'protocol_panel':question['protocol_panel'],'modality':question['modality'],
                    'question_en':english['question_en'],'choices':english.get('choices',[]),
                    'video':str(folder/'video.mp4'),'audio_foa':str(folder/'audio_foa.wav'),'audio_mono':str(folder/'audio_mono.wav'),
                    'foa_format':'ACN_WYZX_SN3D_camera','model_answer':answer,
                    'canonical_answer':by_id[question['question_id']]['canonical_answer'],
                    'private_evidence':str(Path('parents')/row['scene_id']/'private'/(wid+'_evidence.json')),
                    'target_verification':check,'E_S_input_source':'deployment_prediction_not_supplied'})
                family[question['qa_id']]+=1;panel_family[(question['protocol_panel'],question['qa_id'])]+=1;modality[question['modality']]+=1
            window_rows.append({'scene_id':row['scene_id'],'window_id':wid,'house':row['house'],'member':win['member'],
                'start_s':win['start_s'],'logical_questions':win['logical_questions'],'V':win['modality_counts'].get('vision',0),
                'A':win['modality_counts'].get('audio',0),'AV':win['modality_counts'].get('audiovisual',0),
                'binding':win['binding_logical_questions'],'family_counts_json':json.dumps(win['family_counts'],sort_keys=True)})
    license_files={}
    for asset_id,record in sources.items():
        _copy(Path(record.get('normalized_path') or record['path']),target/'sources'/asset_id/'clip.wav')
        packaged=dict(record,packaged_wav=str(Path('sources')/asset_id/'clip.wav'))
        lic=record.get('screens',{}).get('license',{});path=lic.get('retained_text')
        if path:
            path=Path(path)
            if str(path) not in license_files:
                name=f'license_{len(license_files):02d}.txt';_copy(path,target/'licenses'/name);license_files[str(path)]=str(Path('licenses')/name)
            packaged['packaged_license_text']=license_files[str(path)]
        write_json(target/'sources'/asset_id/'source.json',packaged)
    for parent in rows:
        licenses=load(target/'parents'/parent['scene_id']/'private/human_license_admission.json')
        packaged_assets=[]
        for human in licenses:
            packaged_asset={**human,'license':dict(human['license'])};extras=[]
            for extra in human.get('additional_licenses',[]):
                original=Path(extra['retained_text']);digest=extra['sha256']
                import hashlib
                if not original.is_file()or hashlib.sha256(original.read_bytes()).hexdigest()!=digest:
                    raise ValueError('additional source license changed before export')
                if str(original)not in license_files:
                    dest=target/'licenses'/(digest[:16]+'_'+original.name);_copy(original,dest);license_files[str(original)]=str(dest.relative_to(target))
                extras.append({**extra,'packaged_license_text':license_files[str(original)]})
            packaged_asset['additional_licenses']=extras;packaged_assets.append(packaged_asset)
            path=Path(human['license']['retained_text'])
            if str(path) not in license_files:
                name=f'license_{len(license_files):02d}.txt';_copy(path,target/'licenses'/name);license_files[str(path)]=str(Path('licenses')/name)
        write_json(target/'parents'/parent['scene_id']/'private/packaged_asset_licenses.json',packaged_assets)
    raw_export_questions=len(training);kept={}
    for item in sorted(training,key=lambda r:(r['window_start_s'],r['window_id'])):
        key,_=independence_key(item['independent_parent_scene_id'],item,all_gold[item['question_id']])
        kept.setdefault(key,item)
    training=list(kept.values());kept_ids={r['question_id']for r in training}
    family=Counter(r['qa_id']for r in training);modality=Counter(r['modality']for r in training)
    panel_family=Counter((r['protocol_panel'],r['qa_id'])for r in training)
    for (scene_id,wid),public in public_by_window.items():
        write_json(target/'parents'/scene_id/'windows'/wid/'questions.json',[q for q in public if q['question_id']in kept_ids])
    for parent in parent_rows:
        parent['raw_export_questions']=parent['logical_questions']
        parent['logical_questions']=sum(r['independent_parent_scene_id']==parent['scene_id']for r in training)
        parent['dedup_dropped']=parent['raw_export_questions']-parent['logical_questions']
        parent['dedup_rule_version']=INDEPENDENCE_REVISION
    for window in window_rows:
        window['raw_export_questions']=window['logical_questions']
        window['logical_questions']=sum(r['window_id']==window['window_id']for r in training)
    for split in sorted({r['split'] for r in rows}):
        with (target/(split+'.jsonl')).open('x') as f:
            for record in training:
                if record['split']==split:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    _csv(target/'parents.csv',parent_rows,list(parent_rows[0]))
    _csv(target/'windows.csv',window_rows,list(window_rows[0]))
    def ranked(counter):return [{'qa_id':qa,'reason':reason,'count':count} for (qa,reason),count in counter.most_common()]
    rejection={'denominator':'qualified window candidate pools, including counterfactual members; not independent worlds',
        'window_pools':len(window_rows),'family_deferrals':ranked(rejects),'candidate_attempt_rejections':ranked(candidate_rejects),'protocol_candidate_exclusions':ranked(protocol_rejects),
        'alpha_skippable_families':sorted({qa for qa,_ in rejects}-set(family)),
        'cannot_skip':['recorded_source_license_and_origin','native_static_contact_material_masks','FOA_camera_direction','14_selected_questions_and_modality_counts','counterfactual_answer_invariance_or_flip','complete_media_clocks']}
    write_json(target/'rejections.json',rejection)
    def describe(values):
        ordered=sorted(float(v) for v in values)
        if not ordered:return {'n':0,'median':None,'p95':None,'min':None,'max':None}
        return {'n':len(ordered),'median':statistics.median(ordered),'p95':ordered[min(len(ordered)-1,int(.95*len(ordered)))],'min':ordered[0],'max':ordered[-1]}
    bindings=sum(all_gold[r['question_id']].get('counterfactual_role')=='binding' for r in training)
    summary={'status':'formal_alpha_admitted','profile':next(iter(profiles)) if len(profiles)==1 else sorted(profiles),'independent_parent_scenes':len(rows),
        'base_time_windows':sum(r['window_count'] for r in rows),'formal_windows':len(window_rows),'formal_logical_questions':len(training),'raw_export_questions':raw_export_questions,
        'independent_questions':len(training),'dedup_dropped':raw_export_questions-len(training),'dedup_rule_version':INDEPENDENCE_REVISION,
        'questions_per_window':dict(Counter(w['logical_questions'] for w in window_rows)),
        'family_counts':dict(family),'panel_family_counts':{panel+'/'+qa:n for (panel,qa),n in panel_family.items()},'actual_temporal_window_counts':dict(temporal),'family_zero_counts':{f'QA-{i:02d}':0 for i in range(1,29) if f'QA-{i:02d}' not in family},
        'modality_counts':dict(modality),'modality_percent':{k:100*v/len(training) for k,v in modality.items()},
        'binding_questions':bindings,'binding_percent':100*bindings/len(training),
        'parent_wall_seconds':describe([r['wall_seconds'] for r in parent_rows]),'parent_cpu_seconds_including_children':describe(reference_cpu),
        'native_visual_world_cpu_seconds':describe(native_visual_cpu),'successful_parent_cpu_seconds_total':sum(reference_cpu),
        'native_RIR_calls':sum(r['RIR_calls'] for r in parent_rows),'used_recorded_original_ids':sorted(used_source_ids),'used_vctk_speakers':sorted(used_speakers),
        'actual_non_speech_classes':len(set(classes)-{0,1}),'actual_class13_ids':sorted(classes),'actual_class13_segment_counts':dict(classes),'actual_sound_class_emission_counts':dict(sound_classes),'actual_gender_labels':sorted(genders),'actual_speech_gender_emission_counts':dict(genders),'actual_emission_mechanism_counts':dict(mechanisms),'actual_moving_source_windows':moving_windows,'actual_natural_TTS_emissions':tts_event_count,
        'actual_speakers_per_window':dict(speaker_counts),'actual_activity_segments_per_window':dict(activity_counts),
        'full_v5_quota_realized':False,'full_v7_formal_windows':0,'full_v7_generation_ready':False,
        'human_asset_body_templates_used':len({a.get('body_identity_id',a.get('asset_id')) for a in humans.values()if a.get('entity_class','articulated_human')=='articulated_human'}),
        'human_asset_color_variants_used':sum(a.get('entity_class','articulated_human')=='articulated_human'for a in humans.values()),
        'animal_assets_used':sum(a.get('entity_class')=='articulated_animal'for a in humans.values()),
        'human_motion_scope':'actions only as recorded in actual native_visual_checks; no static-to-motion promotion',
        'TTS_license_status':'actual_admitted_CosyVoice2_source_chain_and_native_episode' if tts_event_count else 'no_TTS_emissions_in_this_release','E_S_predictions':'not_emitted; deployment predictor required by downstream training',
        'native_visual_checks':native_checks,'source_material_checks':{'observations':len(material_rows),'all_status_pass':all(r['status']=='pass' for r in material_rows),
            'max_highlight_ratio':max((r['highlight_ratio'] for r in material_rows if r.get('highlight_ratio') is not None),default=None),
            'max_chromaticity_error_p50':max((r['texture_chromaticity_error_p50'] for r in material_rows if r.get('texture_chromaticity_error_p50') is not None),default=None),
            'max_chromaticity_error_p95':max((r['texture_chromaticity_error_p95'] for r in material_rows if r.get('texture_chromaticity_error_p95') is not None),default=None)},
        'FOA_direction_checks':directions,'chosen_seals':chosen,'packaged_sources':len(sources),'packaged_license_texts':license_files,
        'claim_boundary':'only actual qualified standing recorded/TTS native windows; no STARSS evaluation or full-v7 quota/family completion claim'}
    write_json(target/'summary.json',summary)
    return summary
