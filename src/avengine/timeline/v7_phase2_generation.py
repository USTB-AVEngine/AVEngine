"""CPU registered integration pilot with complete native/QA evidence and no export."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json,os,resource,subprocess,time
import numpy as np
import soundfile as sf
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_candidate_generation import stage,read_json,parent_dry,member_scene
from avengine.timeline.v7_registered_audio import render_alpha_audio,source_qa_evidence,phase2_ray_multiplier
from avengine.timeline.v7_registered_generation import _verify_window
from avengine.timeline.v7_full_question_delivery import collect_native_pool,finalize_parent_question_pools
from avengine.timeline.v7_phase2_sources import build_phase2_scene as build_visible_scene,audit_phase2_bindings as audit_native_bindings,resolve_footstep_events
from avengine.timeline.v7_storage_budget import check_write
from avengine.spatial_audio.audio import write_float32_wav

PILOT_PROFILE='v7-integration-pilot-phase2-visible-v1'

def cpu_seconds():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime

def native_checks(scene,assets,visual,audio=None,*,pending_feet=False):
    bind=audit_native_bindings(scene,assets,visual,allow_pending_feet=pending_feet);attachments=visual['visible_attachment_checks'];mats=visual['source_material_checks']['checks']
    measured=[r for r in mats if r['status'] in {'pass','fail'}];fail=[r for r in measured if r['status']=='fail']
    by_asset={a['asset_id']:[r for r in measured if r['asset_id']==a['asset_id']] for a in assets}
    missing=[aid for aid,r in by_asset.items() if not r];alignment=visual['alignment']
    errors=list(bind['errors'])
    # Owner 2026-10-14 23:3x SGT: material quality is recorded, not an admission error.
    quality_warnings=[]
    if fail:quality_warnings.append('native source highlight/appearance failure')
    # Borrowed object references qualify only material quality. Actual missing
    # measurements remain missing, and no appearance/position/answer is copied.
    borrowed=visual.get('owner_material_quality_fallbacks',[])
    quality_ids={r['asset_id']for r in borrowed if r.get('status')=='borrowed_measured_object_material_quality'
        and r.get('same_registered_function_and_attachment_class')is True
        and r.get('actual_asset_was_measured')is False
        and r.get('target_appearance_color_position_and_answer_truth_borrowed')is False}
    required_missing=[aid for aid in missing if aid not in quality_ids]
    if required_missing:quality_warnings.append('source material never measured: '+','.join(required_missing))
    if any(a['attachment_surface']!=a['support_type'] or a['attachment_worst_error_m']>.015 or not a['skin_pass'] for a in attachments):errors.append('attachment or collision failure')
    if any(alignment[k]!=0 for k in ['root_max_error','joint_max_error','camera_max_error','link_max_error']) or alignment['restored_modal_RGB_depth_semantic_frames']!=450:errors.append('native mask restoration or frame alignment failed')
    from avengine.timeline.v7_phase2_checks import motion_checks
    motion=motion_checks(visual)
    if motion['status']!='pass':errors.extend(motion['errors'])
    chair_pixels=visual.get('phase2_chair_pixel_checks',[])
    for chair in visual.get('phase2_chairs',[]):
        if max((r['visible_pixels']for r in chair_pixels if r['asset_id']==chair['asset_id']),default=0)<100:
            errors.append('registered seat never reaches actual 100 visible pixels')
    directions=audio.get('directions',[]) if audio else []
    return {'status':'pass' if not errors else 'fail','errors':errors,
        'material_quality_policy':'owner_record_only_20261014',
        'quality_warnings':quality_warnings,'bindings':bind,
        'attachments':{'passed':sum(a['status']=='pass' for a in attachments),'checked':len(attachments),
            'worst_error_m':max((a['attachment_worst_error_m'] for a in attachments),default=0),
            'worst_penetration_m':min((c['contact_distance_m'] for a in attachments for c in a['contacts']),default=0)},
        'highlights':{'passed':sum(r['status']=='pass' for r in measured),'failed':len(fail),'checked':len(measured),
            'not_run':len(mats)-len(measured),'worst_ratio':max((r.get('highlight_ratio',0) for r in measured),default=0),'unmeasured_asset_ids':missing,'borrowed_object_quality_references':borrowed},
        'RIR_direction':{'status':'pass' if audio and all(r['passed_5deg'] for r in directions) else 'not_run',
            'passed':sum(r['passed_5deg'] for r in directions),'checked':len(directions),
            'worst_error_deg':max((r['angular_error_deg'] for r in directions),default=None)},
        'native_frames':visual['frame_count'],'alignment':alignment,'motion':motion,
        'color_observations':{'matched':sum(r['matched'] for r in visual['visible_color_checks']),
            'measured':len(visual['visible_color_checks']),'engineering_measured':sum(r['engineering_only'] for r in visual['visible_color_checks'])}}



def exact_phase2_visual_identity(scene):
    """Only logical scene names and an explicit reuse locator are nonphysical."""
    if isinstance(scene,dict):
        return {k:exact_phase2_visual_identity(v) for k,v in scene.items()
            if k not in ('scene_id','independent_parent_scene_id','reuse_qualified_phase2_native_parent')}
    if isinstance(scene,list):return [exact_phase2_visual_identity(v)for v in scene]
    return scene



def exact_phase2_visual_reuse(old_scene,new_scene):
    old=exact_phase2_visual_identity(old_scene);new=exact_phase2_visual_identity(new_scene)
    if old==new:return True
    old_events=old.pop('events',[]);new_events=new.pop('events',[])
    if old!=new or not new_events:return False
    # Only a subset of exactly identical animal call events may be omitted.
    # Camera/TTS/footstep schedules, source positions, actions and all other
    # visual fields remain exact. Every originally active owner remains active.
    animal_eps={ep for ep,spec in old.get('sources',{}).items()if spec.get('species_id')in {'dog','cat','bird'}}
    if {e['endpoint_id']for e in old_events}!={e['endpoint_id']for e in new_events}:return False
    if [e for e in old_events if e['endpoint_id']not in animal_eps]!=[e for e in new_events if e['endpoint_id']not in animal_eps]:return False
    return all(e in old_events for e in new_events)


def acceptance_crop_starts(row,scene):
    """Deliver actual 10s crops without changing the original 30s native world."""
    values=row.get('acceptance_window_starts_s')
    if values is None:return list(scene['window_starts_s'])
    if row.get('trial_only') is not True or row.get('members')!=['v0_a0']:
        raise ValueError('extra acceptance crops require explicit trial-only single CPU member')
    if not isinstance(values,list) or not values:raise ValueError('empty acceptance crop schedule')
    starts=[float(v) for v in values]
    if any(not np.isfinite(v) or not 0.<=v<=float(scene['duration_s'])-10.
           or abs(v*15-round(v*15))>1e-8 for v in starts):
        raise ValueError('acceptance crop must be a full 10s at original 15fps inside the native world')
    if starts!=sorted(set(starts)) or row.get('window_count')!=len(starts):
        raise ValueError('acceptance crop schedule must be unique, ordered and match declared window_count')
    return starts


def generate_phase2_parent(row,house,inputs,catalog,source_admission,human_admission,output):
    from avengine.timeline.v7_production_inputs import attach_runtime_contract
    row=attach_runtime_contract(row,inputs)
    from avengine.capture.v7_phase2_native import NativePhase2Video as NativeVisibleVideo,PHASE2_VISUAL_REVISION as VISUAL_REVISION
    from avengine.timeline.v7_position_admission import PositionAdmission
    from avengine.qa.v7_validation import validate_v6_items,validate_content_items
    from avengine.qa.v7_visible_truth import validate_additional_items
    root=Path(output)/'parents'/row['scene_id'];check_write(root,expected_bytes=1_000_000_000);root.mkdir(parents=True,exist_ok=True)
    wall=time.perf_counter();cpu=cpu_seconds();semantic=read_json(inputs['semantic_manifest']) if row.get('semantic_scenario_id') else None
    working,base,program,assets=build_visible_scene(row,house,source_admission['records'],catalog,semantic)
    from avengine.timeline.v7_production_inputs import bind_semantic_registry
    semantic=bind_semantic_registry(semantic,assets)
    # Preserve an already qualified exact native stage for audio-only resume.
    # A changed row, inputs, assets or program still fails recipe comparison.
    if (root/'visible_native_completed.json').is_file() and (root/'input_recipe.json').is_file():
        saved_native=read_json(root/'visible_native_completed.json')
        saved_recipe=read_json(root/'input_recipe.json')
        if saved_recipe.get('visual_revision')==saved_native['recipe'].get('revision'):
            VISUAL_REVISION=saved_recipe['visual_revision']
    recipe={'profile':PILOT_PROFILE,'visual_revision':VISUAL_REVISION,'row':row,'house':working,'inputs':inputs,'assets':assets,'program':program}
    if (root/'input_recipe.json').exists():
        if read_json(root/'input_recipe.json')!=recipe:raise ValueError('pilot recipe changed; use a fresh scene ID')
    else:write_json(root/'input_recipe.json',recipe)
    if (root/'integration_seal_completed.json').exists():
        saved=read_json(root/'integration_seal_completed.json');directory=root/saved['attempt']
        if saved['recipe']!={'input_recipe':recipe} or any(not (directory/f).is_file() for f in saved['required_files']):raise ValueError('pilot seal incomplete')
        from avengine.timeline.v7_question_independence import deduplicate_parent_delivery
        delivered=deduplicate_parent_delivery(row,saved['result'],root,directory)
        return {**delivered,'resumed':True}
    def render(attempt):
        from avengine.timeline.v7_material_quality_fallback import recover_complete_native_attempt,qualify_missing_materials
        recovered=recover_complete_native_attempt(root,attempt,base,assets,native_checks)
        if recovered is not None:return recovered
        if row.get('reuse_qualified_phase2_native_parent'):
            import hashlib
            from shutil import copytree
            source_root=Path(row['reuse_qualified_phase2_native_parent']).resolve()
            if source_root.parent!=root.parent.resolve():raise ValueError('phase2 visual reuse must be another exact parent inside this pilot')
            pointer_path=source_root/'visible_native_completed.json'
            pointer=read_json(pointer_path);old_recipe=pointer['recipe'];source_dir=source_root/pointer['attempt']
            if (not exact_phase2_visual_reuse(old_recipe['scene'],base)
                    or old_recipe['assets']!=assets or old_recipe['inputs']!=inputs):
                raise ValueError('phase2 reused visual input differs physically')
            if any(not(source_dir/f).is_file()for f in pointer['required_files']):raise ValueError('qualified native source incomplete')
            visual=read_json(source_dir/'native_visible_visual_evidence_phase2.json')
            if (source_dir/'owner_material_quality_references.json').is_file():
                visual['owner_material_quality_fallbacks']=read_json(source_dir/'owner_material_quality_references.json')['references']
            checks=native_checks(base,assets,visual,pending_feet=True)
            if checks['status']!='pass':raise ValueError('exact phase2 visual reuse fails fresh native gate')
            # Link original immutable evidence bytes, including diagnostic PNGs.
            copytree(source_dir,attempt,copy_function=os.link,dirs_exist_ok=True)
            reuse={'status':'pass_exact_phase2_native_reuse','source_parent':str(source_root),
                'source_pointer_sha256':hashlib.sha256(pointer_path.read_bytes()).hexdigest(),
                'physical_scene_inputs_assets_exactly_equal':True,'animal_call_schedule_subset_only':old_recipe['scene']['events']!=base['events'],'native_checks_fresh':checks,
                'original_scene_id':old_recipe['scene']['scene_id'],'new_scene_id':base['scene_id'],
                'source_files_rewritten':False,'phase1_reuse':False}
            write_json(attempt/'exact_phase2_native_reuse.json',reuse)
            return {'status':'pass','native_checks':checks,'exact_phase2_reuse':reuse},[*pointer['required_files'],'exact_phase2_native_reuse.json']
        if row.get('reuse_completed_native_parent'):
            raise ValueError('phase-two original poses require their own exact native admission; phase-one reuse forbidden')
        check_write(attempt,expected_bytes=200_000_000)
        material_cache=Path(inputs.get('source_material_cache_directory',root/'source_material_cache'))
        check_write(material_cache,expected_bytes=64_000_000)
        native_scene=deepcopy(base)
        video=NativeVisibleVideo(working,native_scene,inputs,assets,source_material_cache_directory=material_cache)
        gate=PositionAdmission(base,working,inputs,root/'rir_cache',attempt/'acoustic_placement_admission.json');video.acoustic_position_admission=gate.check
        try:visual=video.encode(native_scene,attempt/'scene_video.mp4',collect_pixel_visibility=True)
        finally:gate.close();video.close()
        del gate,video
        from avengine.capture.v7_native_context_recycling import release_unused_native_memory
        release_unused_native_memory()
        visual=qualify_missing_materials(root,assets,visual)
        if visual.get('owner_material_quality_fallbacks'):
            write_json(attempt/'owner_material_quality_references.json',{'references':visual['owner_material_quality_fallbacks'],'actual_original_measurements_retained':True})
        checks=native_checks(base,assets,visual,pending_feet=True);write_json(attempt/'native_checks.json',checks)
        if checks['status']!='pass':raise ValueError('native integration gate failed: '+str(checks['errors']))
        return {'status':'pass','native_checks':checks},['scene_video.mp4','native_visible_visual_evidence_phase2.json','native_color_checks.json','native_checks.json','source_material_checks.json','native_visibility_evidence.json','acoustic_placement_admission.json','standing/standing_admission.json','standing/native_support_survey.json']+(['owner_material_quality_references.json']if visual.get('owner_material_quality_fallbacks')else [])
    vd,vr,_=stage(root,'visible_native',{'scene':base,'assets':assets,'inputs':inputs,'revision':VISUAL_REVISION},render)
    visual=read_json(vd/'native_visible_visual_evidence_phase2.json')
    if (vd/'owner_material_quality_references.json').is_file():
        visual['owner_material_quality_fallbacks']=read_json(vd/'owner_material_quality_references.json')['references']
    from avengine.assets.animal_motion.production_episode import realized_animal_motion
    realized_animal_motion(base,visual)
    if visual.get('realized_phase2_camera'):
        realized=visual['realized_phase2_camera']
        base['camera_yaw_knots']=deepcopy(realized['camera_yaw_knots'])
        base['camera_forward']=deepcopy(realized['camera_forward'])
    from avengine.assets.v7_visible_color import rebuild_appearance_review,COLOR_EVIDENCE_REVISION
    visual['native_appearance_review'],color_eligibility=rebuild_appearance_review(
        visual['visible_color_checks'],assets,(inputs.get('animal_color_thresholds') or {}).get('min_pixels',200))
    resolved=resolve_footstep_events(base,program,visual)
    foot_receipt=root/'resolved_footstep_contacts.json'
    if foot_receipt.exists():
        if read_json(foot_receipt)!=resolved:raise ValueError('actual footstep contact plan changed')
    else:write_json(foot_receipt,resolved)
    dry,receipt=parent_dry(root,base,program)
    for ep,voice in receipt['endpoint_speaker_ids'].items():base['sources'][ep]['voice_identity_id']=voice
    scene=member_scene(base,visual,'v0_a0')
    from avengine.timeline.v7_motion_during_conditions import verify_motion_during_native
    motion_condition_report=verify_motion_during_native(row,scene,visual)
    if motion_condition_report is not None:
        write_json(root/'conditioned_motion_native_verification.json',motion_condition_report)
        if motion_condition_report['status']!='pass':raise ValueError('original conditioned_motion native verification failed; see conditioned_motion_native_verification.json')
    def audio(attempt):return render_alpha_audio(scene,working,inputs,root/'rir_cache',attempt,dry,receipt,visual['camera_frame_readbacks'])
    from avengine.timeline.v7_wet_support import WET_SUPPORT_REVISION
    audio_recipe={'scene':scene,'inputs':inputs,'wet_support_revision':WET_SUPPORT_REVISION}
    if any(np.any(dry[ep]) and any(not np.array_equal(k['position_m'],spec['trajectory'][0]['position_m'])for k in spec['trajectory']) for ep,spec in scene['sources'].items()):
        from avengine.timeline.v7_phase2_motion_audio import REVISION
        from avengine.timeline.v7_phase2_fresh_rir import REVISION as CONTEXT_REVISION
        audio_recipe.update(moving_interpolation_revision=REVISION,
            native_sampling_revision=f'same-native-engine-rays-x{phase2_ray_multiplier(scene,dry)}-all-reference-gates-v2',
            moving_native_context_revision=CONTEXT_REVISION)
    ad,ar,_=stage(root,'audio_v0_a0',audio_recipe,audio)
    checks=native_checks(scene,assets,visual,ar)
    delivery_starts=acceptance_crop_starts(row,scene)
    def windows(attempt):
        samples,sr=sf.read(ad/'scene_foa_full_tail.wav',dtype='float64',always_2d=True);required=[];wins=[]
        for wi,start in enumerate(delivery_starts):
            check_write(attempt,expected_bytes=32_000_000);wid=f'{row["scene_id"]}__v0_a0__w{wi:02d}';dest=attempt/wid;dest.mkdir();first=round(start*15)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-threads','1','-i',str(vd/'scene_video.mp4'),
                '-vf',f'select=between(n\\,{first}\\,{first+149}),setpts=N/(15*TB)','-frames:v','150','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(dest/'video.mp4')],check=True)
            crop=samples[round(start*16000):round(start*16000)+160000]
            if sr!=16000 or crop.shape!=(160000,4):raise ValueError('pilot FOA/media clock differs')
            write_float32_wav(dest/'audio_foa.wav',crop,16000,channel_axis=1);write_float32_wav(dest/'audio_mono.wav',crop[:,:1],16000,channel_axis=1)
            media={k:str(dest/f) for k,f in [('video','video.mp4'),('audio_foa','audio_foa.wav'),('audio_mono','audio_mono.wav')]}
            proof=source_qa_evidence(scene,program,receipt,visual,dest/'audio_foa.wav',ar,question_episode_id=wid,window_start_s=start)
            if any(not e['activity_intervals_samples_global'] for e in proof['events'].values()):raise ValueError('scheduled event lacks actual measured emitted activity')
            if semantic:
                from avengine.timeline.v7_visible_semantics import visible_dialogue_evidence
                proof.update(visible_dialogue_evidence(scene,program,receipt,visual,semantic,start,str(vd/'native_visible_visual_evidence_phase2.json')))
            q,g,audit=collect_native_pool(scene,start,wid,visual,media,proof,row['question_quota_base']+wi,14)
            parts=[validate_v6_items(q,g,audit,seed=str(scene['seed'])),validate_additional_items(q,g,proof),validate_content_items(q,g)]
            replay={'status':'pass','checked_forms':sum(p['checked_forms'] for p in parts),'passed_forms':sum(p['passed_forms'] for p in parts),'mismatches':[r for p in parts for r in p['mismatches']],'panels':parts}
            if replay['checked_forms']!=len(q):raise ValueError('independent replay did not cover every question form')
            if replay['mismatches']:replay['status']='fail'
            write_json(dest/'independent_v6_truth.json',replay)
            if replay['mismatches']:raise ValueError('independent original-family truth mismatch')
            write_json(dest/'questions.json',q);write_json(dest/'private_answers.json',g);audit['native_color_eligibility']=color_eligibility
            write_json(dest/'candidate_audit.json',audit)
            write_json(dest/'evidence.json',{'profile':PILOT_PROFILE,'seed_row':row,'scene':scene,'source_qa_evidence':proof,
                'native_visual_evidence':str(vd/'native_visible_visual_evidence_phase2.json'),'native_audio_evidence':str(ad/'audio_evidence.json'),
                'machine_checks':checks,'independent_v6_truth':replay,'formal_dataset_count':0})
            wins.append({'window_id':wid});required.extend(wid+'/'+f for f in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json','private_answers.json','candidate_audit.json','evidence.json','independent_v6_truth.json'])
        return {'windows':wins},required
    window_recipe={'scene':scene,'receipt':receipt,'profile':PILOT_PROFILE,'color_evidence_revision':COLOR_EVIDENCE_REVISION}
    if row.get('acceptance_window_starts_s') is not None:window_recipe['acceptance_window_starts_s']=delivery_starts
    wd,wr,_=stage(root,'windows_v0_a0',window_recipe,windows)
    selection_row={**deepcopy(row),'members':['v0_a0'],'window_count':len(wr['windows']),'questions_per_window':14,'production_scenario':deepcopy(scene['production_scenario'])}
    members,selection=finalize_parent_question_pools(selection_row,[{'member':'v0_a0','windows_directory':str(wd),'windows':wr['windows'],'audio_directory':str(ad),'visual_directory':str(vd)}],root)
    def seal(attempt):
        required=[];stats=[];source=Path(members[0]['windows_directory']);private=attempt/'private';private.mkdir()
        for win in members[0]['windows']:
            wid=win['window_id'];dest=attempt/'windows'/wid;dest.mkdir(parents=True)
            for f in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json']:os.link(source/wid/f,dest/f);required.append(str((dest/f).relative_to(attempt)))
            for old,new in [('private_answers.json',wid+'_answers.json'),('evidence.json',wid+'_evidence.json')]:os.link(source/wid/old,private/new);required.append('private/'+new)
            qs=read_json(dest/'questions.json');stats.append({'window_id':wid,'media_readback':_verify_window(dest),'logical_questions':len({q['logical_question_id'] for q in qs}),'family_counts':dict(Counter(q['qa_id'] for q in {q['logical_question_id']:q for q in qs}.values()))})
        write_json(private/'machine_checks.json',checks);write_json(private/'assets.json',assets);write_json(private/'dry_program.json',program);write_json(private/'dry_receipt.json',receipt)
        required.extend('private/'+f for f in ['machine_checks.json','assets.json','dry_program.json','dry_receipt.json'])
        result={'status':'integration_pilot_admitted','profile':PILOT_PROFILE,'scene_id':row['scene_id'],'house':row['house'],
            'source_admission_version':inputs.get('source_admission_version'),'source_admission_sha256':inputs.get('source_admission_sha256'),
            'family':house['family'],'room_id':row['room_id'],'windows':stats,'machine_checks':checks,
            'asset_counts':dict(Counter([a['asset_id'] for a in assets]+[a['asset_id']for a in visual.get('phase2_chairs',[])])),
            'chair_instances':visual.get('phase2_chairs',[]),'footstep_contact_receipt':str(foot_receipt),'class_event_counts':dict(Counter(e['sound_class'] for e in scene['events'])),
            'pilot_windows':len(stats),'pilot_logical_questions':sum(w['logical_questions'] for w in stats),
            'delivery_window_starts_s':delivery_starts,'delivery_crops_share_one_independent_30s_parent':True,
            'acceptance_allow_partial_families':row.get('acceptance_allow_partial_families') is True,
            'alpha_formal_windows':0,'alpha_formal_logical_questions':0,'train_table_exported':False,
            'wall_seconds':time.perf_counter()-wall,'cpu_seconds_including_children':cpu_seconds()-cpu}
        write_json(attempt/'pilot_manifest.json',result);required.append('pilot_manifest.json');return result,required
    directory,result,_=stage(root,'integration_seal',{'input_recipe':recipe},seal)
    from avengine.timeline.v7_question_independence import deduplicate_parent_delivery
    return deduplicate_parent_delivery(row,result,root,directory)
