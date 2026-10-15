"""Formal static/recorded v7-S alpha with unchanged native QA predicates.

Full-v7 coverage remains a separate unreleased scope. The alpha qualification
requires actual sound admission, static contact, RGB/masks, direction, exact
modality counts and measured counterfactual answers before immutable delivery.
"""
from __future__ import annotations
from avengine.timeline.v7_wet_support import WET_SUPPORT_REVISION
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,time
import numpy as np
import soundfile as sf
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_candidate_generation import stage,read_json,parent_dry,member_scene,audit_parent_group
from avengine.timeline.v7_registered_sampling import build_registered_scene,REGISTERED_INPUT_SCHEMA,choose_assets
from avengine.timeline.v7_registered_audio import render_alpha_audio,source_qa_evidence
from avengine.timeline.v7_audio_sources import rebind_receipt,source_blockers
from avengine.timeline.v7_full_question_delivery import collect_native_pool,derive_native_appearance_evidence,finalize_parent_question_pools
from avengine.timeline.v7_production_contract import validate_question_counts,question_allocation
from avengine.spatial_audio.audio import write_float32_wav

ALPHA_RELEASE_PROFILE='v7-S-recorded-registered-native-14-v6-critical-coverage'
PRODUCER_QA_REVISION='registered-native-r17-UE-physical-emitter-v1'


def _cpu():
    self_usage=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    return self_usage.ru_utime+self_usage.ru_stime+child.ru_utime+child.ru_stime


def qualify_licenses(assets,admission):
    by_id={r['asset_id']:r for r in admission['records']};checks=[]
    if any(not a['asset_id'].startswith('v7_') or 'lx_' in a['asset_id'] for a in assets):raise ValueError('standing profile license path is Rocketbox-only; use explicit registered capability intake for lx')
    mit=admission['license_receipts']['rocketbox_mit']
    if not mit['local_binding_valid']:raise ValueError('registered Rocketbox MIT binding failed')
    for asset in assets:
        row=by_id[asset['asset_id']]
        if not row['standing_registered_qualified']:raise ValueError('static registered authority missing')
        for path,digest in row['actual_artifact_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('registered input changed')
        additional=[]
        for filename in asset.get('additional_license_texts',[]):
            path=Path(filename)
            if not path.is_file()or not path.read_bytes():raise ValueError('additional asset license text missing')
            text=path.read_text()
            if 'CC0'not in text or 'Universal'not in text:raise ValueError('unrecognized canine animation license text')
            additional.append({'status':'pass','license':'CC0-1.0','retained_text':str(path),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_uri':'https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt',
                'declaration_uri':'https://quaternius.com/packs/ultimateanimatedanimals.html',
                'scope':'original admitted canine Walking_Contact animation; original Rocketbox body uses MIT'})
        motion_scope='registered stand; any requested action requires complete native episode contact and visibility'
        if asset.get('entity_class')=='articulated_animal':
            authority=Path(asset['static_pose_authority']);records=json.loads(authority.read_text())
            record=next((x for x in records if x.get('breed')==asset.get('animal_variant')or x.get('static_glb')==asset['habitat_glb']),None)
            if record is None or record.get('status')!='pass'or record.get('static_glb')!=asset['habitat_glb']:
                raise ValueError('original canine static pose authority does not match this GLB')
            if not additional:raise ValueError('canine motion requires retained CC0 license')
            motion_scope='registered original static canine; continuous admitted contact walk only; no planted pause claimed'
        checks.append({'asset_id':asset['asset_id'],'entity_class':asset.get('entity_class','articulated_human'),
            'license':{'status':'pass','license':mit['license'],'retained_text':str(Path(row['source_registry']).parent.parent/mit['retained_copy']),'sha256':mit['sha256']},
            'additional_licenses':additional,'source_chain_verified':True,'motion_scope':motion_scope})
    return checks


def _verify_window(folder):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,width,height,r_frame_rate,duration','-of','json',str(folder/'video.mp4')]))['streams'][0]
    if int(info['nb_read_frames'])!=150 or info['r_frame_rate']!='15/1' or abs(float(info['duration'])-10)>1e-6:
        raise ValueError('sealed video clock differs from 150 native frames')
    data,sr=sf.read(folder/'audio_foa.wav',always_2d=True,dtype='float32')
    if sr!=16000 or data.shape!=(160000,4) or not np.isfinite(data).all():raise ValueError('sealed FOA clock or channels invalid')
    mono,msr=sf.read(folder/'audio_mono.wav',always_2d=True,dtype='float32')
    if msr!=sr or not np.array_equal(mono,data[:,:1]):raise ValueError('derived mono differs from FOA W')
    return {'video':info,'FOA':{'sample_rate_hz':sr,'samples':len(data),'channels':4,'format':'ACN_WYZX_SN3D_camera'},'mono_matches_W':True}


def generate_registered_parent(row,house,inputs,assets,source_admission,human_admission,output):
    from avengine.timeline.v7_production_inputs import attach_runtime_contract
    row=attach_runtime_contract(row,inputs)
    if inputs.get('render_backend')=='ue_spear':
        from avengine.capture.v7_registered_ue import NativeRegisteredUEVideo as NativeAlphaVideo,REGISTERED_UE_VISUAL_REVISION as ALPHA_VISUAL_REVISION
    else:
        from avengine.capture.v7_registered_native import NativeRegisteredVideo as NativeAlphaVideo,REGISTERED_VISUAL_REVISION as ALPHA_VISUAL_REVISION
    from avengine.assets.source_materials import MATERIAL_REVISION
    from avengine.capture.continuous_native_visibility import VISIBILITY_REVISION
    start_wall=time.perf_counter();start_cpu=_cpu()
    from avengine.timeline.v7_storage_budget import check_write
    root=Path(output)/'parents'/row['scene_id']
    render_ready=all((root/('visual_'+v+'_completed.json')).is_file() for v in {m.split('_')[0] for m in row['members']})
    check_write(root,expected_bytes=0 if (root/'alpha_seal_completed.json').is_file() else (64_000_000 if render_ready else 1_000_000_000))
    root.mkdir(parents=True,exist_ok=True)
    assets=choose_assets(assets,row)
    from avengine.timeline.v7_production_inputs import validate_registered_sampling_row
    production_intake=validate_registered_sampling_row(row,assets,inputs)
    licenses=qualify_licenses(assets,human_admission)
    semantic=read_json(inputs['semantic_manifest']) if inputs.get('semantic_manifest') else None
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    working,base,program=build_registered_scene(row,house,production_audio_record_views(source_admission['records']),assets,semantic,camera_evidence_policy=inputs.get('registered_camera_evidence_policy'))
    if production_intake:base['production_sampling_intake']=production_intake
    authored_ids={r['sound_asset_id'] for r in program['records'] if r.get('source_type')=='author_tts'}
    if authored_ids:
        from avengine.timeline.v7_production_inputs import bind_semantic_registry
        semantic=bind_semantic_registry(semantic,assets)
        full=semantic;semantic=deepcopy(full)
        semantic['sounds']=[r for r in full['sounds'] if r['sound_asset_id'] in authored_ids]
        actual_scenarios={r['scenario_id'] for r in semantic['sounds']}
        semantic['dialogues']=deepcopy(full['dialogues'])
        semantic['dialogues']['scenarios']=[r for r in full['dialogues']['scenarios'] if r.get('id',r.get('scenario_id')) in actual_scenarios]
        semantic['retained_batch_manifest']={'path':inputs['semantic_manifest'],
            'sha256':hashlib.sha256(Path(inputs['semantic_manifest']).read_bytes()).hexdigest(),
            'scope':'only actual parent sound IDs and authored scenarios copied into each private window'}
    else:semantic=None
    recipe={'profile':ALPHA_RELEASE_PROFILE,'producer_qa_revision':PRODUCER_QA_REVISION,'seed_row':row,'house':working,'runtime_inputs':inputs,
        'assets':assets,'recorded_program':program,'visual_revision':ALPHA_VISUAL_REVISION,
        'source_material_revision':MATERIAL_REVISION,'pixel_visibility_revision':VISIBILITY_REVISION}
    existing=root/'input_recipe.json'
    if existing.exists():
        if read_json(existing)!=recipe:raise ValueError('alpha parent recipe changed; select a fresh output')
    else:write_json(existing,recipe)
    pointer=root/'alpha_seal_completed.json'
    if pointer.exists():
        saved=read_json(pointer);directory=root/saved['attempt']
        if saved['recipe']!={'input_recipe':recipe} or any(not (directory/p).is_file() for p in saved['required_files']):
            raise ValueError('sealed alpha delivery is incomplete or recipe changed')
        return {**saved['result'],'delivery_directory':str(directory),'resumed_sealed_delivery':True}
    write_json(root/f'worker_{os.getpid()}_{time.time_ns()}.json',{'pid':os.getpid(),'parent_pid':os.getppid(),
        'scene_id':row['scene_id'],'compute_device':'CPU','release_profile':ALPHA_RELEASE_PROFILE})
    dry,receipt=parent_dry(root,base,program)
    for ep,speaker in receipt['endpoint_speaker_ids'].items():base['sources'][ep]['voice_identity_id']=speaker
    visual_dirs={};visuals={};variants={};audio_factors={};members=[];timings=[];clip_paths={}
    visual_names=sorted({m.split('_')[0] for m in row['members']})
    for v in visual_names:
        variant=deepcopy(assets)
        if v=='v1':
            endpoints=[a['endpoint_id'] for a in variant[:2]];variant[:2]=variant[:2][::-1]
            for asset,ep in zip(variant[:2],endpoints,strict=True):asset['endpoint_id']=ep
        def render(attempt):
            check_write(attempt,expected_bytes=200_000_000)
            video=NativeAlphaVideo(working,base,inputs,variant,source_material_cache_directory=root/'source_material_cache')
            from avengine.timeline.v7_position_admission import PositionAdmission
            position_gate=PositionAdmission(base,working,inputs,root/'rir_cache',attempt/'acoustic_placement_admission.json')
            video.acoustic_position_admission=position_gate.check
            try:visual=video.encode(base,attempt/'scene_video.mp4',collect_pixel_visibility=True)
            finally:
                position_gate.close();video.close()
            appearance=derive_native_appearance_evidence(attempt,visual,variant)
            write_json(attempt/'native_appearance_review.json',appearance)
            return {'wall_seconds':visual['wall_seconds'],'appearance_status':appearance['status']},[
                'scene_video.mp4','native_registered_visual_evidence.json','native_visual_evidence.json',
                'source_materials.json','source_material_checks.json','native_visibility_evidence.json','native_appearance_review.json','native_environmental_emitters.json','standing/standing_admission.json','acoustic_placement_admission.json']
        vc=_cpu();vw=time.perf_counter()
        from avengine.timeline.v7_native_stage_resume import compatible_static_native_recipe
        visual_recipe=compatible_static_native_recipe(root,'visual_'+v,{'scene':base,'inputs':inputs,'assets':variant,
            'visual_revision':ALPHA_VISUAL_REVISION,'source_material_revision':MATERIAL_REVISION},row)
        vd,vr,resumed=stage(root,'visual_'+v,visual_recipe,render)
        timings.append({'stage':'visual_'+v,'wall_seconds':time.perf_counter()-vw,'cpu_seconds_including_children':_cpu()-vc,'resumed':resumed})
        visual=read_json(vd/'native_registered_visual_evidence.json');appearance=read_json(vd/'native_appearance_review.json')
        # Reconstruct the producer state also when v0 was resumed. Its actual
        # SDK standing roots originally drove the camera recentering; leaving
        # the prototype here would change the exact v1 stage recipe on restart.
        if base.get('camera_evidence_policy'):
            from avengine.timeline.v7_camera_evidence import recenter_on_actual_roots
            replay_entries=[]
            for admitted in visual['standing_admission']:
                matrix=np.eye(4);matrix[:3,3]=admitted['resolved_body_root_world_m']
                replay_entries.append({'endpoint_id':admitted['endpoint_id'],'applied_root_input':matrix.tolist()})
            recenter_on_actual_roots(base,replay_entries)
        if appearance['status']!='pass' or len(appearance['review']['actors'])<2:raise ValueError('at least two observed native appearance reviews required')
        visual['native_appearance_review']=appearance['review']
        if visual['frame_count']!=450 or len(visual['camera_frame_readbacks'])!=450:raise ValueError('incomplete native world')
        if inputs.get('render_backend')=='ue_spear':
            from avengine.capture.v7_alpha_ue import assert_ue_alpha_visual
            assert_ue_alpha_visual(visual)
        elif any(visual['alignment'][k]!=0 for k in ['root_max_error','joint_max_error','camera_max_error','link_max_error']) or visual['alignment']['restored_modal_RGB_depth_semantic_frames']!=450:
            raise ValueError('native mask restoration qualification failed')
        for a in visual['standing_admission']:
            if a['status']!='pass' or a['known_bad_detected'] is not True:raise ValueError('native static contact admission incomplete')
        # Disjoint native full-skin AABBs certify that isolated placement probes
        # remain clear when the four static bodies are present together.
        for i,a in enumerate(visual['standing_admission']):
            alo,ahi=np.asarray(a['native_full_skin_bounds_m'])
            for b in visual['standing_admission'][i+1:]:
                blo,bhi=np.asarray(b['native_full_skin_bounds_m'])
                if np.all(np.minimum(ahi,bhi)>np.maximum(alo,blo)):raise ValueError('native static body envelopes overlap')
        if visuals and (visual['observed_mouth_trajectories']!=visuals['v0']['observed_mouth_trajectories']
                or visual['camera_frame_readbacks']!=visuals['v0']['camera_frame_readbacks']):
            raise ValueError('appearance counterfactual changed native geometry or camera')
        visual_dirs[v]=vd;visuals[v]=visual;variants[v]=variant
    for member in row['members']:
        v,factor=member.split('_');visual=visuals[v];vd=visual_dirs[v]
        scene=member_scene(base,visual,member)
        local_dry=dry;local_receipt=receipt
        if factor=='a1':
            mapping={'voice1':'voice2','voice2':'voice1'}
            local_dry={ep:dry[mapping.get(ep,ep)] for ep in dry};local_receipt=rebind_receipt(receipt,mapping)
        def audio(attempt):
            check_write(attempt,expected_bytes=16_000_000)
            if factor in audio_factors:
                source,result=audio_factors[factor]
                for name in ['scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json']:os.link(source/name,attempt/name)
                return result,['scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json']
            return render_alpha_audio(scene,working,inputs,root/'rir_cache',attempt,local_dry,local_receipt,visual['camera_frame_readbacks'])
        audio_recipe={'scene':scene,'inputs':inputs,'wet_support_revision':WET_SUPPORT_REVISION}
        if any(not np.array_equal(k['position_m'],source['trajectory'][0]['position_m']) for source in scene['sources'].values() for k in source['trajectory']):
            audio_recipe['moving_native_step_revision']='registered-fixed-.2s-bisection-min-.1s-v1'
        ac=_cpu();aw=time.perf_counter();ad,ar,resumed=stage(root,'audio_'+member,audio_recipe,audio)
        timings.append({'stage':'audio_'+member,'wall_seconds':time.perf_counter()-aw,'cpu_seconds_including_children':_cpu()-ac,'resumed':resumed})
        audio_factors.setdefault(factor,(ad,ar))
        def windows(attempt):
            samples,sr=sf.read(ad/'scene_foa_full_tail.wav',dtype='float64',always_2d=True)
            required=[];records=[]
            for wi,start in enumerate(scene['window_starts_s']):
                check_write(attempt,expected_bytes=32_000_000)
                wid=f'{row["scene_id"]}__{member}__w{wi:02d}';dest=attempt/wid;dest.mkdir();first=round(start*15)
                if (v,wi) in clip_paths:os.link(clip_paths[(v,wi)],dest/'video.mp4')
                else:
                    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-threads','1',
                        '-i',str(vd/'scene_video.mp4'),'-vf',f'select=between(n\\,{first}\\,{first+149}),setpts=N/(15*TB)',
                        '-frames:v','150','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','18',
                        '-pix_fmt','yuv420p',str(dest/'video.mp4')],check=True)
                    clip_paths[(v,wi)]=dest/'video.mp4'
                lo=round(start*16000);cropped=samples[lo:lo+160000]
                if sr!=16000 or cropped.shape!=(160000,4):raise ValueError('alpha crop lacks complete FOA history')
                write_float32_wav(dest/'audio_foa.wav',cropped,16000,channel_axis=1)
                write_float32_wav(dest/'audio_mono.wav',cropped[:,:1],16000,channel_axis=1)
                media={role:str(dest/name) for role,name in [('video','video.mp4'),('audio_foa','audio_foa.wav'),('audio_mono','audio_mono.wav')]}
                proof=source_qa_evidence(scene,program,local_receipt,visual,dest/'audio_foa.wav',ar,question_episode_id=wid,window_start_s=start)
                if semantic:
                    from avengine.timeline.v7_registered_semantics import dialogue_evidence
                    proof.update(dialogue_evidence(scene,program,local_receipt,visual,semantic,start,str(vd/'native_registered_visual_evidence.json')))
                quota=row['question_quota_base']+row['members'].index(member)*row['window_count']+wi
                q,a,audit=collect_native_pool(scene,start,wid,visual,media,proof,quota,row['questions_per_window'])
                if base.get('require_current_native_environment'):
                    from avengine.qa.kujiale_current_color import filter_current_scene_colors
                    q,a,color_audit=filter_current_scene_colors(q,a,audit,visual,start)
                    audit['current_scene_colour_gate']=color_audit
                excluded=set();keep={x['question_id'] for x in q if x['qa_id'] not in excluded}
                q=[x for x in q if x['question_id'] in keep];a=[x for x in a if x['question_id'] in keep]
                write_json(dest/'questions.json',q);write_json(dest/'private_answers.json',a)
                write_json(dest/'candidate_audit.json',audit)
                ev={'release_profile':ALPHA_RELEASE_PROFILE,'seed_row':row,'member':member,'start_s':start,
                    'scene':scene,'native_visual_evidence':str(vd/'native_registered_visual_evidence.json'),
                    'native_audio_evidence':str(ad/'audio_evidence.json'),'source_qa_evidence':proof,
                    'candidate_family_counts':dict(Counter(x['qa_id'] for x in q if x['form']=='open')),
                    'full_v7_formal_count':0,'alpha_selection':'pending'}
                write_json(dest/'evidence.json',ev)
                records.append({'window_id':wid,'logical_candidates':len({x['logical_question_id'] for x in q})})
                required.extend(f'{wid}/{n}' for n in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json','private_answers.json','evidence.json','candidate_audit.json'])
            return {'windows':records},required
        wc=_cpu();ww=time.perf_counter();wd,wr,resumed=stage(root,'windows_'+member,{'scene':scene,'source_receipt':local_receipt,
            'profile':ALPHA_RELEASE_PROFILE,'producer_qa_revision':PRODUCER_QA_REVISION,'quota_index':row['question_quota_base']},windows)
        timings.append({'stage':'windows_'+member,'wall_seconds':time.perf_counter()-ww,'cpu_seconds_including_children':_cpu()-wc,'resumed':resumed})
        for wi,win in enumerate(wr['windows']):clip_paths.setdefault((v,wi),wd/win['window_id']/'video.mp4')
        members.append({'member':member,'windows_directory':str(wd),'windows':wr['windows'],'audio_directory':str(ad),'visual_directory':str(vd)})
    content_audit={'status':'not_applicable_no_authored_quartet'}
    if semantic and len(members)==4:
        from avengine.timeline.v7_registered_content import augment_content_binding
        members,content_audit=augment_content_binding(row,members,root)
    family_counts=Counter()
    for path in row.get('selection_coverage_seals',[]):
        observed=read_json(Path(path)/'alpha_seal_completed.json')['result']
        if observed['status']!='formal_alpha_admitted':raise ValueError('selection coverage baseline is not an actual qualified seal')
        for window in observed['windows']:family_counts.update(window['family_counts'])
    members,selection=finalize_parent_question_pools(row,members,root,binding_items_per_window=5,family_counts=family_counts)
    group=audit_parent_group(row,base,members)
    group_path=root/'counterfactual_alpha_audit.json'
    if group_path.exists():
        if read_json(group_path)!=group:raise ValueError('retained counterfactual audit changed')
    else:write_json(group_path,group)
    elapsed=time.perf_counter()-start_wall;cpu=_cpu()-start_cpu
    def seal(attempt):
        required=[];stats=[];private=attempt/'private';private.mkdir();shared=private/'native';shared.mkdir()
        for v in visual_names:
            target=shared/v;target.mkdir()
            for name in ['native_registered_visual_evidence.json','source_materials.json','source_material_checks.json','native_visibility_evidence.json','native_appearance_review.json','native_environmental_emitters.json','acoustic_placement_admission.json']:
                os.link(visual_dirs[v]/name,target/name);required.append(str((target/name).relative_to(attempt)))
        for factor,(directory,result) in audio_factors.items():
            for name in ['audio_evidence.json','native_RLR_setup.json']:
                target=shared/(factor+'_'+name);os.link(directory/name,target);required.append(str(target.relative_to(attempt)))
        write_json(private/'human_license_admission.json',licenses);write_json(private/'dry_program.json',program)
        write_json(private/'dry_emission_receipt.json',receipt);write_json(private/'counterfactual_audit.json',group)
        for name in ['human_license_admission.json','dry_program.json','dry_emission_receipt.json','counterfactual_audit.json']:required.append('private/'+name)
        if (root/'content_binding').is_dir():
            for original in sorted((root/'content_binding').glob('*.json')):
                target=private/'content_binding'/original.name;target.parent.mkdir(exist_ok=True);os.link(original,target);required.append(str(target.relative_to(attempt)))
        for m in members:
            for wi,win in enumerate(m['windows']):
                wid=win['window_id'];source=Path(m['windows_directory'])/wid;dest=attempt/'windows'/wid;dest.mkdir(parents=True)
                for name in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json']:os.link(source/name,dest/name)
                gold=read_json(source/'private_answers.json');q=read_json(source/'questions.json')
                allocation=question_allocation(row['question_quota_base']+row['members'].index(m['member'])*row['window_count']+wi,14)
                count=validate_question_counts(q,gold,expected_allocation=allocation)
                measured=_verify_window(dest);evidence=read_json(source/'evidence.json')
                evidence['alpha_admission']={'status':'admitted','profile':ALPHA_RELEASE_PROFILE,
                    'media_clock':measured,'question_counts':count,'native_contact_material_masks_direction':'pass',
                    'counterfactual_answers':'pass' if len(row['members'])==4 else 'not_applicable_declared_single','full_v7_generation_ready':False}
                answer_path=private/(wid+'_answers.json');ev_path=private/(wid+'_evidence.json')
                write_json(answer_path,gold);write_json(ev_path,evidence)
                required.extend(str((dest/name).relative_to(attempt)) for name in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json'])
                required.extend([str(answer_path.relative_to(attempt)),str(ev_path.relative_to(attempt))])
                bindings=sum(x.get('counterfactual_role')=='binding' for x in gold if x.get('form','open')=='open')
                stats.append({'window_id':wid,'member':m['member'],'start_s':base['window_starts_s'][wi],
                    'logical_questions':len({x['logical_question_id'] for x in q}),'question_rows':len(q),
                    'family_counts':dict(Counter(x['qa_id'] for x in q if x['form']=='open')),
                    'modality_counts':dict(Counter(x['modality'] for x in q if x['form']=='open')),
                    'binding_logical_questions':len({x['logical_question_id'] for x in gold if x.get('counterfactual_role')=='binding'}),
                    'public_question_path':str((dest/'questions.json').relative_to(attempt))})
        result={'status':'formal_alpha_admitted','profile':ALPHA_RELEASE_PROFILE,'scene_id':row['scene_id'],
            'house':row['house'],'room_id':row['room_id'],'independent_parent_scenes':1,'windows':stats,
            'content_binding_native_candidates':content_audit,'alpha_formal_windows':len(stats),'alpha_formal_logical_questions':sum(w['logical_questions'] for w in stats),
            'material_quality_policy':'owner_record_only_20261014','full_v7_formal_windows':0,'full_v7_generation_ready':False,'alpha_limitations':base['sampling_limitations'],
            'wall_seconds_before_seal':elapsed,'cpu_seconds_before_seal_including_children':cpu,'stage_timings':timings,
            'used_dry_original_ids':sorted(r['original_id'] for r in program['records']),
            'measured_RIR_calls':sum(result['RIR_cache']['new_rirs'] for _,result in audio_factors.values())+sum(read_json(visual_dirs[v]/'acoustic_placement_admission.json')['cache_stats']['new_rirs'] for v in visual_names),
            'public_windows_directory':'windows','private_answers_directory':'private','E_S_training_inputs':'deployment predictions only; not emitted by this renderer'}
        write_json(attempt/'delivery_manifest.json',result);required.append('delivery_manifest.json')
        return result,required
    directory,result,resumed=stage(root,'alpha_seal',{'input_recipe':recipe},seal)
    from avengine.timeline.v7_question_independence import deduplicate_parent_delivery
    delivered=deduplicate_parent_delivery(row,result,root,directory)
    return {**delivered,'wall_seconds':time.perf_counter()-start_wall,
        'cpu_seconds_including_children':_cpu()-start_cpu,'seal_resumed':resumed}
