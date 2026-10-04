#!/usr/bin/env python3
"""Current v7 native components, isolated media diagnostics; no training release.

Render and single-threaded RLR run in separate processes. Only admitted TRAIN
houses and legacy v3 canary recipes are used; v4 scientific gates stay intact.
Question failures are measured explicitly, never replaced with invented truth.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def records(paths):return [{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
def inventory(attempt,result):
    paths=[p for p in sorted(attempt.rglob('*')) if p.is_file() and p.name!='stage.json']
    write(attempt/'stage.json',{'scope':'infrastructure_trial','training_eligible':False,'formal_dataset_count':0,'result':result,'files':records(paths)})
def check_request(req):
    if req.get('scope')!='infrastructure_trial' or req.get('training_eligible') is not False:raise ValueError('missing no-ingest contract')
    row=req['row'];house=req['house']
    if row['split']!='train' or house['split']!='train' or not row['scene_id'].startswith('cci_infra_r4_'):raise ValueError('trial contains a held-out house or production ID')
    if row.get('schema')!='v7-production-inputs-20261003-v3' or row['members']!=['v0_a0']:raise ValueError('trial needs an explicit single-member legacy v3 canary; no v4 fallback')
    from avengine.timeline.v7_candidate_generation import seeded_scene
    return seeded_scene(row,house)

def run(a):
    req=read(a.request);house,base=check_request(req);attempt=Path(a.attempt);attempt.mkdir(parents=True,exist_ok=True)
    from avengine.timeline.v7_candidate_generation import member_scene,audio_stage,parent_dry
    if a.phase=='plan':
        write(attempt/'plan.json',{'working_house':house,'base_scene':base,'request_sha256':sha(a.request),'scope':'infrastructure_trial','training_eligible':False})
        inventory(attempt,{'planned':True});return
    if a.phase=='render':
        if not a.gpu or ',' in a.gpu or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu:raise ValueError('missing leased renderer GPU')
        os.environ.update(CCI_INFRA_RENDER_GPU=a.gpu,CCI_INFRASTRUCTURE_TRIAL='1')
        from avengine.capture.continuous_native_video import NativeContinuousVideo
        scene=read(Path(a.plan_root)/'plan.json')['base_scene'];assets=req['assets']
        start=time.monotonic();video=NativeContinuousVideo(house,scene,req['runtime_inputs'],assets,source_material_cache_directory=attempt/'source_material_cache')
        try:result=video.encode(scene,attempt/'scene_video.mp4')
        finally:video.close()
        inventory(attempt,{'wall_seconds':time.monotonic()-start,'renderer':result['renderer'],'native_frames':450,'infrastructure_trial_only':True});return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('acoustics and delivery must have no CUDA lease')
    visual=read(Path(a.render_root)/'native_visual_evidence.json')
    scene=member_scene(base,visual,'v0_a0')
    if a.phase=='audio':
        parent=Path(a.scratch_parent)/'persistent_audio';parent.mkdir(parents=True,exist_ok=True)
        dry,dry_receipt=parent_dry(parent,base)
        result,_=audio_stage(scene,house,req['runtime_inputs'],parent/'rir_cache',attempt,dry_override=dry,dry_receipt=dry_receipt)
        setup=read(attempt/'native_RLR_setup.json');actual=setup.get('configuration_readback',{})
        if actual.get('thread_count')!=1:raise ValueError('native RLR thread_count readback is not 1')
        inventory(attempt,result);return
    if a.phase!='deliver':raise ValueError('unsupported phase')
    import soundfile as sf
    import numpy as np
    from avengine.spatial_audio.audio import write_float32_wav
    from avengine.timeline.v7_native_questions import make_questions
    from avengine.timeline.v7_production_contract import validate_question_counts
    world_root=Path(a.world_root);world_root.mkdir(parents=True,exist_ok=True)
    sealed=world_root/'ARTIFACTS.json'
    if sealed.exists():
        saved=read(sealed)
        for item in saved['files']:
            p=Path(item['path'])
            if not p.resolve().is_relative_to(world_root.resolve()) or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
                raise ValueError('infrastructure delivery readback failed')
        inventory(attempt,{'resumed_sealed_delivery':True});return
    root=world_root/('payload_'+attempt.name);root.mkdir()
    audio,sr=sf.read(Path(a.audio_root)/'scene_foa_full_tail.wav',dtype='float64',always_2d=True)
    if sr!=16000 or audio.shape[1]!=4 or not np.isfinite(audio).all():raise ValueError('invalid native FOA')
    appearances={base['sources'][x['endpoint_id']]['actor_id']:x.get('appearance_zh',{'blue':'蓝色上衣的人','yellow':'黄色上衣的人'}[x['appearance']['top_color']]) for x in req['assets']}
    windows=[];question_errors=[]
    for i,start in enumerate(scene['window_starts_s']):
        wid=f"{req['row']['scene_id']}__v0_a0__w{i:02d}";dest=root/wid;dest.mkdir(parents=True,exist_ok=True)
        first=round(start*15)
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-threads','1','-i',str(Path(a.render_root)/'scene_video.mp4'),
             '-vf',rf'select=between(n\,{first}\,{first+149}),setpts=N/(15*TB)','-frames:v','150','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(dest/'video.mp4')]
        subprocess.run(cmd,check=True)
        samples=audio[round(start*16000):round(start*16000)+160000]
        if samples.shape!=(160000,4):raise ValueError('window lacks full audio history')
        write_float32_wav(dest/'audio_foa.wav',samples,16000,channel_axis=1)
        media={'video':'video.mp4','audio_foa':'audio_foa.wav','foa_format':'ACN_WYZX_SN3D_camera'}
        try:
            questions,answers,check=make_questions(scene,start,wid,visual,appearances,media,quota_index=req['row']['question_quota_base']+i,count=req['row']['questions_per_window'])
            counts=validate_question_counts(questions,answers)
            write(dest/'questions.json',questions);write(dest/'private_answers.json',answers)
            question_status={'status':'generated_research_only','check':check,'counts':counts,'logical_questions':len(questions)}
        except ValueError as error:
            question_status={'status':'native_question_generation_failed','error':str(error),'logical_questions':0}
            question_errors.append({'window':wid,'error':str(error)})
        write(dest/'question_generation.json',question_status)
        # Decode actual delivered H264 frames; hashes are measured, not inferred.
        decoded=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-threads','1','-i',str(dest/'video.mp4'),'-f','framehash','-hash','sha256','-'],capture_output=True,text=True,check=True)
        if len([l for l in decoded.stdout.splitlines() if l and not l.startswith('#')])!=150:raise ValueError('delivered video does not contain exactly 150 decoded frames')
        write(dest/'frame_hashes.json',{'command':'ffmpeg -threads 1 -i video.mp4 -f framehash -hash sha256 -','framehash_sha256':decoded.stdout,'frames':len([l for l in decoded.stdout.splitlines() if l and not l.startswith('#')])})
        windows.append({'window_id':wid,'question_generation':question_status})
    evidence=root/'diagnostics';evidence.mkdir(exist_ok=True)
    for directory,names in ((Path(a.render_root),('native_visual_evidence.json','source_materials.json','source_material_checks.json','native_visibility_evidence.json')),(Path(a.audio_root),('native_RLR_setup.json','audio_evidence.json'))):
        for name in names:shutil.copyfile(directory/name,evidence/name)
    write(root/'NO_TRAINING_INGEST.json',{'scope':'infrastructure_trial','training_eligible':False,'formal_dataset_count':0,'scientific_acceptance':'not_admitted','generator_profile_sha256':req['generator_profile_sha256'],'row':req['row'],'windows':windows,'question_errors':question_errors})
    files=records([p for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('ARTIFACTS.json','COMPLETE.json')])
    write(world_root/'ARTIFACTS.json',{'payload_directory':root.name,'scope':'infrastructure_trial','training_eligible':False,'files':files,'native_media_windows':len(windows),'question_errors':question_errors})
    inventory(attempt,{'delivery_artifacts':str(world_root/'ARTIFACTS.json'),'formal_dataset_count':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--request',required=True);p.add_argument('--attempt',required=True)
    for name in ('plan-root','render-root','audio-root','world-root','scratch-parent','gpu'):p.add_argument('--'+name)
    args=p.parse_args();run(args)
