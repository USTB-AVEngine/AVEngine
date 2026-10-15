"""Requalify unchanged single-member native worlds with current camera evidence.

This preserves original physical stage versions. It recomputes question truth
from observed poses, never edits an old seal or relabels old visuals as new.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import json, os, shutil, time
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_candidate_generation import stage,read_json,audit_parent_group
from avengine.timeline.v7_registered_generation import _verify_window,_cpu,qualify_licenses
from avengine.timeline.v7_registered_audio import source_qa_evidence
from avengine.timeline.v7_full_question_delivery import collect_native_pool,finalize_parent_question_pools
from avengine.timeline.v7_production_contract import question_allocation,validate_question_counts
from avengine.timeline.v7_storage_budget import check_write

REQUALIFICATION_PROFILE='v7-S-native-camera-requalified-standing-14-v1'


def require_single_training_or_dev(row):
    if row['split'] not in {'train','dev'} or row['members']!=['v0_a0']:
        raise ValueError('requalification accepts only declared single train/dev worlds')


def requalify_native_single_parent(source,output,human_admission,family_counts=None):
    start=time.perf_counter();cpu=_cpu();source=Path(source)
    old=read_json(source/'alpha_seal_completed.json');prior=source/old['attempt'];original=old['recipe']['input_recipe'];row=original['seed_row']
    require_single_training_or_dev(row)
    if old['result']['status']!='formal_alpha_admitted' or any(not (prior/f).is_file() for f in old['required_files']):
        raise ValueError('original physical seal incomplete')
    root=Path(output)/'parents'/row['scene_id'];check_write(root,expected_bytes=64_000_000);root.mkdir(parents=True,exist_ok=False)
    recipe=deepcopy(original);recipe['profile']=REQUALIFICATION_PROFILE
    recipe['requalification']={'source_parent':str(source),'original_visual_revision':original['visual_revision'],
        'physical_media_reused_unchanged':True,'current_listener_binding':'complete native camera frames',
        'original_physical_CPU_seconds':old['result']['cpu_seconds_before_seal_including_children'],
        'original_physical_wall_seconds':old['result']['wall_seconds_before_seal'],
        'new_renderer_claim':False}
    write_json(root/'input_recipe.json',recipe)
    for pointer in source.glob('*_completed.json'):
        if not pointer.name.startswith(('visual_','audio_','dry_')):continue
        x=read_json(pointer);os.link(pointer,root/pointer.name)
        shutil.copytree(source/x['attempt'],root/x['attempt'],copy_function=os.link)
    visual_marker=read_json(root/'visual_v0_completed.json');vd=root/visual_marker['attempt'];visual=read_json(vd/'native_alpha_visual_evidence.json')
    visual['native_appearance_review']=read_json(vd/'native_appearance_review.json')['review']
    if visual['frame_count']!=450 or len(visual['camera_frame_readbacks'])!=450 or visual['source_material_checks']['status']!='pass':
        raise ValueError('full native world/material proof missing')
    if any(visual['alignment'][k]!=0 for k in ['root_max_error','joint_max_error','camera_max_error','link_max_error']) or visual['alignment']['restored_modal_RGB_depth_semantic_frames']!=450:
        raise ValueError('native rendering/mask restoration failed')
    if any(a['status']!='pass' or a['known_bad_detected'] is not True for a in visual['standing_admission']):
        raise ValueError('original native standing proof incomplete')
    licenses=qualify_licenses(original['assets'],human_admission)
    program=original['recorded_program'];receipt=read_json(prior/'private/dry_emission_receipt.json');audio=read_json(prior/'private/native/a0_audio_evidence.json')
    if any(not r.get('recorded_boundary_mismatch') and (not r['passed_5deg'] or r['angular_error_deg']>5.) for r in audio['directions']):raise ValueError('original physical RIR direction failed')
    wd=root/'camera_requalified_candidates';wd.mkdir();windows=[];base=None;differences=[]
    for wi,win in enumerate(old['result']['windows']):
        wid=win['window_id'];src=prior/'windows'/wid;d=wd/wid;d.mkdir();check_write(d,expected_bytes=16_000_000)
        _verify_window(src)
        for name in ['video.mp4','audio_foa.wav','audio_mono.wav']:os.link(src/name,d/name)
        ev=read_json(prior/'private'/(wid+'_evidence.json'));scene=ev['scene'];base=scene
        proof=source_qa_evidence(scene,program,receipt,visual,d/'audio_foa.wav',audio,question_episode_id=wid,window_start_s=win['start_s'])
        media={k:str(d/n) for k,n in [('video','video.mp4'),('audio_foa','audio_foa.wav'),('audio_mono','audio_mono.wav')]}
        q,g,audit=collect_native_pool(scene,win['start_s'],wid,visual,media,proof,row['question_quota_base']+wi,14)
        observed=audit['v6_generation']['input_facts']['listener']['yaw_deg'];native=[r['yaw_deg'] for r in audit['native_bundle']['frame_readbacks']['camera']]
        if observed!=native:raise ValueError('QA listener differs from actual native camera')
        old_yaw=[r['yaw_deg'] for r in audit['native_bundle']['frame_readbacks']['camera']]
        differences.append({'window':wid,'native_yaw_span_deg':max(old_yaw)-min(old_yaw),'new_QA_equals_native_all_frames':True})
        write_json(d/'questions.json',q);write_json(d/'private_answers.json',g);write_json(d/'candidate_audit.json',audit)
        ev.update(release_profile=REQUALIFICATION_PROFILE,source_qa_evidence=proof,
            candidate_family_counts=dict(Counter(x['qa_id'] for x in q if x['form']=='open')),
            requalification=recipe['requalification'],alpha_selection='pending')
        write_json(d/'evidence.json',ev);windows.append({'window_id':wid,'logical_candidates':len({x['logical_question_id'] for x in q})})
    members=[{'member':'v0_a0','windows_directory':str(wd),'windows':windows,'audio_directory':str(root/read_json(root/'audio_v0_a0_completed.json')['attempt']),'visual_directory':str(vd)}]
    members,selection=finalize_parent_question_pools(row,members,root,binding_items_per_window=5,family_counts=family_counts or {})
    group=audit_parent_group(row,base,members);write_json(root/'counterfactual_alpha_audit.json',group);write_json(root/'native_listener_requalification.json',differences)
    elapsed=time.perf_counter()-start;used_cpu=_cpu()-cpu
    def seal(attempt):
        required=[];private=attempt/'private';private.mkdir();shared=private/'native';shared.mkdir()
        for src in (prior/'private/native').rglob('*'):
            if src.is_file():
                dest=shared/src.relative_to(prior/'private/native');dest.parent.mkdir(parents=True,exist_ok=True);os.link(src,dest);required.append(str(dest.relative_to(attempt)))
        for name in ['human_license_admission.json','dry_program.json','dry_emission_receipt.json']:
            src=prior/'private'/name;os.link(src,private/name);required.append('private/'+name)
        write_json(private/'counterfactual_audit.json',group);required.append('private/counterfactual_audit.json');stats=[]
        for wi,w in enumerate(members[0]['windows']):
            wid=w['window_id'];src=Path(members[0]['windows_directory'])/wid;dest=attempt/'windows'/wid;dest.mkdir(parents=True)
            for name in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json']:os.link(src/name,dest/name);required.append(str((dest/name).relative_to(attempt)))
            q=read_json(src/'questions.json');g=read_json(src/'private_answers.json');ev=read_json(src/'evidence.json');allocation=question_allocation(row['question_quota_base']+wi,14);check=validate_question_counts(q,g,expected_allocation=allocation)
            ev['alpha_admission']={'status':'admitted','profile':REQUALIFICATION_PROFILE,'media_clock':_verify_window(dest),'question_counts':check,'native_contact_material_masks_direction':'pass','counterfactual_answers':'not_applicable_declared_single','full_v7_generation_ready':False}
            for suffix,data in [('answers',g),('evidence',ev)]:
                f=private/(wid+'_'+suffix+'.json');write_json(f,data);required.append(str(f.relative_to(attempt)))
            stats.append({'window_id':wid,'member':'v0_a0','start_s':old['result']['windows'][wi]['start_s'],'logical_questions':len({x['logical_question_id'] for x in q}),'question_rows':len(q),'family_counts':dict(Counter(x['qa_id'] for x in q if x['form']=='open')),'modality_counts':dict(Counter(x['modality'] for x in q if x['form']=='open')),'binding_logical_questions':0,'public_question_path':str((dest/'questions.json').relative_to(attempt))})
        result={'status':'formal_alpha_admitted','profile':REQUALIFICATION_PROFILE,'scene_id':row['scene_id'],'house':row['house'],'room_id':row['room_id'],'independent_parent_scenes':1,'windows':stats,'alpha_formal_windows':len(stats),'alpha_formal_logical_questions':sum(w['logical_questions'] for w in stats),'full_v7_formal_windows':0,'full_v7_generation_ready':False,'alpha_limitations':base['sampling_limitations'],'wall_seconds_before_seal':elapsed,'cpu_seconds_before_seal_including_children':used_cpu,'stage_timings':[{'stage':'native_camera_requalification','wall_seconds':elapsed,'cpu_seconds_including_children':used_cpu,'resumed':False}],'used_dry_original_ids':sorted(r['original_id'] for r in program['records']),'measured_RIR_calls':0,'public_windows_directory':'windows','private_answers_directory':'private','requalification':recipe['requalification'],'E_S_training_inputs':'deployment predictions only; not emitted by this renderer'}
        write_json(attempt/'delivery_manifest.json',result);required.append('delivery_manifest.json');return result,required
    directory,result,_=stage(root,'alpha_seal',{'input_recipe':recipe},seal)
    return {**result,'delivery_directory':str(directory),'wall_seconds':time.perf_counter()-start,'cpu_seconds_including_children':_cpu()-cpu}
