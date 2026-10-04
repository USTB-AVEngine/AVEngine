"""Formal static/recorded v7-S alpha with unchanged native QA predicates.

Full-v7 coverage remains a separate unreleased scope. The alpha qualification
requires actual sound admission, static contact, RGB/masks, direction, exact
modality counts and measured counterfactual answers before immutable delivery.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,time
import numpy as np
import soundfile as sf
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_candidate_generation import stage,read_json,parent_dry,member_scene,audit_parent_group
from avengine.timeline.v7_alpha_sampling import build_alpha_scene,ALPHA_INPUT_SCHEMA
from avengine.timeline.v7_alpha_audio import render_alpha_audio,source_qa_evidence
from avengine.timeline.v7_audio_sources import rebind_receipt,source_blockers
from avengine.timeline.v7_full_question_delivery import collect_native_pool,derive_native_appearance_evidence,finalize_parent_question_pools
from avengine.timeline.v7_production_contract import validate_question_counts,question_allocation
from avengine.spatial_audio.audio import write_float32_wav

ALPHA_RELEASE_PROFILE='v7-S-alpha-recorded-static-native-14-v1'


def _cpu():
    self_usage=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    return self_usage.ru_utime+self_usage.ru_stime+child.ru_utime+child.ru_stime


def qualify_licenses(assets,admission):
    by_id={r['asset_id']:r for r in admission['records']};checks=[];hashed={}
    for asset in assets:
        row=by_id.get(asset['asset_id'])
        if row is None or row['license']['status']!='pass':raise ValueError('alpha human license admission missing')
        license=row['license'];path=Path(license['retained_text'])
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=license['sha256']:raise ValueError('retained human license changed')
        for source in row.get('local_source_bindings',[]):
            p=source['read_path']
            if p not in hashed:hashed[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
            if hashed[p]!=source['expected_sha256']:raise ValueError('licensed native body source changed')
        checks.append({'asset_id':asset['asset_id'],'license':license,'source_chain_verified':True,
            'motion_scope':'idle_pose0; each placement requires native full-skin contact qualification'})
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


def generate_alpha_parent(row,house,inputs,assets,source_admission,human_admission,output):
    if house['input'].get('visual_backend') == 'ue_spear':
        from avengine.capture.v7_alpha_ue import NativeUEAlphaVideo as NativeAlphaVideo,ALPHA_UE_VISUAL_REVISION as ALPHA_VISUAL_REVISION
    else:
        from avengine.capture.v7_alpha_native import NativeAlphaVideo,ALPHA_VISUAL_REVISION
    from avengine.assets.source_materials import MATERIAL_REVISION
    from avengine.capture.continuous_native_visibility import VISIBILITY_REVISION
    start_wall=time.perf_counter();start_cpu=_cpu()
    root=Path(output)/'parents'/row['scene_id'];root.mkdir(parents=True,exist_ok=True)
    licenses=qualify_licenses(assets,human_admission)
    working,base,program=build_alpha_scene(row,house,source_admission['records'])
    recipe={'profile':ALPHA_RELEASE_PROFILE,'seed_row':row,'house':working,'runtime_inputs':inputs,
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
    visual_variants = sorted({m.split('_')[0] for m in row['members']})
    for v in visual_variants:
        variant=deepcopy(assets)
        if v=='v1':
            endpoints=[a['endpoint_id'] for a in variant[:2]];variant[:2]=variant[:2][::-1]
            for asset,ep in zip(variant[:2],endpoints,strict=True):asset['endpoint_id']=ep
        def render(attempt):
            video=NativeAlphaVideo(working,base,inputs,variant,source_material_cache_directory=attempt/'source_material_cache')
            try:visual=video.encode(base,attempt/'scene_video.mp4',collect_pixel_visibility=True)
            finally:video.close()
            if house['input'].get('visual_backend') == 'ue_spear':
                from avengine.timeline.v7_ue_appearance import derive_ue_appearance_evidence
                appearance=derive_ue_appearance_evidence(attempt,visual,variant)
            else:
                appearance=derive_native_appearance_evidence(attempt,visual,variant)
            write_json(attempt/'native_appearance_review.json',appearance)
            return {'wall_seconds':visual['wall_seconds'],'appearance_status':appearance['status']},[
                'scene_video.mp4','native_alpha_visual_evidence.json','native_visual_evidence.json',
                'source_materials.json','source_material_checks.json','native_visibility_evidence.json','native_appearance_review.json','standing/standing_admission.json']
        vc=_cpu();vw=time.perf_counter()
        vd,vr,resumed=stage(root,'visual_'+v,{'scene':base,'inputs':inputs,'assets':variant,
            'visual_revision':ALPHA_VISUAL_REVISION,'source_material_revision':MATERIAL_REVISION},render)
        timings.append({'stage':'visual_'+v,'wall_seconds':time.perf_counter()-vw,'cpu_seconds_including_children':_cpu()-vc,'resumed':resumed})
        visual=read_json(vd/'native_alpha_visual_evidence.json');appearance=read_json(vd/'native_appearance_review.json')
        if appearance['status']!='pass' or len(appearance['review']['actors'])!=4:raise ValueError('four observed native appearance reviews required')
        visual['native_appearance_review']=appearance['review']
        if visual['frame_count']!=450 or len(visual['camera_frame_readbacks'])!=450:raise ValueError('incomplete native world')
        if house['input'].get('visual_backend') == 'ue_spear':
            from avengine.capture.v7_alpha_ue import assert_ue_alpha_visual
            assert_ue_alpha_visual(visual)
        elif visual['source_material_checks']['status']!='pass' or any(visual['alignment'][k]!=0 for k in ['root_max_error','joint_max_error','camera_max_error','link_max_error']) or visual['alignment']['restored_modal_RGB_depth_semantic_frames']!=450:
            raise ValueError('native material or mask restoration qualification failed')
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
            if factor in audio_factors:
                source,result=audio_factors[factor]
                for name in ['scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json']:os.link(source/name,attempt/name)
                return result,['scene_foa_full_tail.wav','native_RLR_setup.json','audio_evidence.json']
            return render_alpha_audio(scene,working,inputs,root/'rir_cache',attempt,local_dry,local_receipt,visual['camera_frame_readbacks'])
        ac=_cpu();aw=time.perf_counter();ad,ar,resumed=stage(root,'audio_'+member,{'scene':scene,'inputs':inputs},audio)
        timings.append({'stage':'audio_'+member,'wall_seconds':time.perf_counter()-aw,'cpu_seconds_including_children':_cpu()-ac,'resumed':resumed})
        audio_factors.setdefault(factor,(ad,ar))
        def windows(attempt):
            samples,sr=sf.read(ad/'scene_foa_full_tail.wav',dtype='float64',always_2d=True)
            required=[];records=[]
            for wi,start in enumerate(scene['window_starts_s']):
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
                quota=row['question_quota_base']+row['members'].index(member)*row['window_count']+wi
                q,a,audit=collect_native_pool(scene,start,wid,visual,media,proof,quota,row['questions_per_window'])
                excluded={'QA-26','QA-27','QA-28'};keep={x['question_id'] for x in q if x['qa_id'] not in excluded}
                q=[x for x in q if x['question_id'] in keep];a=[x for x in a if x['question_id'] in keep]
                write_json(dest/'questions.json',q);write_json(dest/'private_answers.json',a)
                write_json(dest/'candidate_audit.json',audit)
                ev={'release_profile':ALPHA_RELEASE_PROFILE,'seed_row':row,'member':member,'start_s':start,
                    'scene':scene,'native_visual_evidence':str(vd/'native_alpha_visual_evidence.json'),
                    'native_audio_evidence':str(ad/'audio_evidence.json'),'source_qa_evidence':proof,
                    'candidate_family_counts':dict(Counter(x['qa_id'] for x in q if x['form']=='open')),
                    'full_v7_formal_count':0,'alpha_selection':'pending'}
                write_json(dest/'evidence.json',ev)
                records.append({'window_id':wid,'logical_candidates':len({x['logical_question_id'] for x in q})})
                required.extend(f'{wid}/{n}' for n in ['video.mp4','audio_foa.wav','audio_mono.wav','questions.json','private_answers.json','evidence.json','candidate_audit.json'])
            return {'windows':records},required
        wc=_cpu();ww=time.perf_counter();wd,wr,resumed=stage(root,'windows_'+member,{'scene':scene,'source_receipt':local_receipt,
            'profile':ALPHA_RELEASE_PROFILE,'quota_index':row['question_quota_base']},windows)
        timings.append({'stage':'windows_'+member,'wall_seconds':time.perf_counter()-ww,'cpu_seconds_including_children':_cpu()-wc,'resumed':resumed})
        for wi,win in enumerate(wr['windows']):clip_paths.setdefault((v,wi),wd/win['window_id']/'video.mp4')
        members.append({'member':member,'windows_directory':str(wd),'windows':wr['windows'],'audio_directory':str(ad),'visual_directory':str(vd)})
    members,selection=finalize_parent_question_pools(row,members,root,binding_items_per_window=5)
    group=audit_parent_group(row,base,members)
    write_json(root/'counterfactual_alpha_audit.json',group)
    elapsed=time.perf_counter()-start_wall;cpu=_cpu()-start_cpu
    def seal(attempt):
        required=[];stats=[];private=attempt/'private';private.mkdir();shared=private/'native';shared.mkdir()
        for v in visual_variants:
            target=shared/v;target.mkdir()
            for name in ['native_alpha_visual_evidence.json','source_materials.json','source_material_checks.json','native_visibility_evidence.json','native_appearance_review.json']:
                shutil.copyfile(visual_dirs[v]/name,target/name);required.append(str((target/name).relative_to(attempt)))
        for factor,(directory,result) in audio_factors.items():
            for name in ['audio_evidence.json','native_RLR_setup.json']:
                target=shared/(factor+'_'+name);shutil.copyfile(directory/name,target);required.append(str(target.relative_to(attempt)))
        write_json(private/'human_license_admission.json',licenses);write_json(private/'dry_program.json',program)
        write_json(private/'dry_emission_receipt.json',receipt);write_json(private/'counterfactual_audit.json',group)
        for name in ['human_license_admission.json','dry_program.json','dry_emission_receipt.json','counterfactual_audit.json']:required.append('private/'+name)
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
                    'counterfactual_answers':'pass','full_v7_generation_ready':False}
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
            'alpha_formal_windows':len(stats),'alpha_formal_logical_questions':sum(w['logical_questions'] for w in stats),
            'full_v7_formal_windows':0,'full_v7_generation_ready':False,'alpha_limitations':base['sampling_limitations'],
            'wall_seconds_before_seal':elapsed,'cpu_seconds_before_seal_including_children':cpu,'stage_timings':timings,
            'used_dry_original_ids':sorted(r['original_id'] for r in program['records']),
            'measured_RIR_calls':sum(result['RIR_cache']['new_rirs'] for _,result in audio_factors.values()),
            'public_windows_directory':'windows','private_answers_directory':'private','E_S_training_inputs':'deployment predictions only; not emitted by this renderer'}
        write_json(attempt/'delivery_manifest.json',result);required.append('delivery_manifest.json')
        return result,required
    directory,result,resumed=stage(root,'alpha_seal',{'input_recipe':recipe},seal)
    return {**result,'delivery_directory':str(directory),'wall_seconds':time.perf_counter()-start_wall,
        'cpu_seconds_including_children':_cpu()-start_cpu,'seal_resumed':resumed}
