#!/usr/bin/env python3
"""Resume native-mouth CPU audio and complete 10 s qualification candidates."""
from __future__ import annotations
import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import soundfile as sf
from avengine.timeline.v7_candidate_generation import stage,audio_stage,FORMAL_BLOCKERS,read_json
from avengine.timeline.continuous_pilot import write_json
from avengine.spatial_audio.audio import write_float32_wav


def work(task):
    inputs_path,windows_root,output_root,house_id=task
    inputs=read_json(inputs_path);house=next(x for x in inputs['houses'] if x['house']==house_id)
    scene=read_json(Path(windows_root)/(house_id+'__observed_scene.json'))
    root=Path(output_root)/house_id;root.mkdir(parents=True,exist_ok=True)
    legacy=root/'candidate_completed.json'
    if legacy.is_file():
        receipt=read_json(legacy)
        if receipt['scene']!=scene:raise ValueError('completed candidate scene changed')
        if any(not (root/p).is_file() for p in receipt['required_files']):raise ValueError('completed candidate lost a file')
        return receipt
    audio_dir,ar,audio_resumed=stage(root,'audio',{'scene':scene,'inputs':inputs},
        lambda attempt:audio_stage(scene,house,inputs,root/'rir_cache',attempt))
    def export(attempt):
        audio,sr=sf.read(audio_dir/'scene_foa_full_tail.wav',dtype='float64',always_2d=True)
        if sr!=16000 or audio.shape[1]!=4:raise ValueError('native history is not FOA 16 kHz')
        windows=[];required=[]
        for i,start_s in enumerate(scene['window_starts_s']):
            start_s=round(start_s*5)/5;wid=scene['scene_id']+f'__w{i:02d}'
            src=Path(windows_root)/wid;dest=attempt/wid;dest.mkdir()
            for name in ('video.mp4','private_answers.json','question_checks.json'):os.link(src/name,dest/name)
            lo=round(start_s*16000);samples=audio[lo:lo+160000]
            if samples.shape!=(160000,4):raise ValueError('native history ended before crop')
            write_float32_wav(dest/'audio_foa.wav',samples,16000,channel_axis=1)
            write_float32_wav(dest/'audio_mono.wav',samples[:,:1],16000,channel_axis=1)
            questions=read_json(src/'public_questions.json')
            for q in questions:q['media'].pop('audio_status',None)
            write_json(dest/'public_questions.json',questions)
            evidence={'window_id':wid,'scene':scene,'start_s':start_s,'frames':150,'samples':160000,
                'FOA_channels':4,'format':'ACN_WYZX_SN3D_camera','logical_questions':len(questions),
                'native_mouth_audio':True,'formal_dataset_count':0,'status':'candidate_only',
                'blocking_reasons':list(FORMAL_BLOCKERS),'audio_evidence':str(audio_dir/'audio_evidence.json')}
            write_json(dest/'evidence.json',evidence);windows.append(evidence)
            required += [str(p.relative_to(attempt)) for p in dest.iterdir() if p.is_file() and not p.name.startswith('.')]
        return {'windows':windows,'formal_dataset_count':0},required
    directory,result,resumed=stage(root,'windows',{'scene':scene,'windows_root':str(Path(windows_root).resolve())},export)
    return {'scene':scene,'status':'candidate_only','formal_dataset_count':0,
            'windows':result['windows'],'windows_directory':str(directory),
            'resumed_stages':{'audio':audio_resumed,'windows':resumed},'audio':ar}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True)
    p.add_argument('--windows-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not 1<=a.workers<=64:raise ValueError('48g worker ceiling is 64')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only synthesis requires CUDA disabled')
    inputs=read_json(a.inputs);a.output.mkdir(parents=True,exist_ok=True)
    for h in inputs['houses']:
        if not (a.native_root/h['house']/'native_visual_evidence.json').is_file():raise ValueError('native visual evidence missing')
    tasks=[(str(a.inputs),str(a.windows_root),str(a.output),h['house']) for h in inputs['houses']]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(work,tasks))
    path=a.output/f'summary_{time.time_ns()}.json'
    write_json(path,{'status':'candidate_only','scenes':results,'formal_dataset_count':0})
    print(str(path),flush=True)

if __name__=='__main__':main()
