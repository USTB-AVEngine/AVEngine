#!/usr/bin/env python3
"""Export checked native video windows and balanced pilot question candidates."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from avengine.timeline.v7_native_questions import make_questions
from avengine.timeline.continuous_pilot import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for case in sorted(a.native_root.iterdir()):
        if not (case/'native_visual_evidence.json').is_file():continue
        visual=json.loads((case/'native_visual_evidence.json').read_text());scene=json.loads((case/'scene_plan_input.json').read_text())
        # Both registered figures are male; use actor identity rather than
        # pretending the former f3 proxy certified a female native person.
        for k in ('voice1','voice2'):scene['sources'][k]['class_id']=1
        for e in scene['events']:
            if e['class_id']==0:e['class_id']=1
        # The idle/walk M2 actors have no clap animation. This production
        # candidate deliberately uses speech-only activity, still >6 segments.
        scene['events']=[x for x in scene['events'] if x['endpoint_id']!='clap1']
        scene['sources'].pop('clap1')
        for endpoint,trajectory in visual['observed_mouth_trajectories'].items():scene['sources'][endpoint]['trajectory']=trajectory
        scene['sources']['voice1']['voice']='en-us+m1';scene['sources']['voice2']['voice']='en-us+m3'
        scene['visual_status']='Habitat_native_M2_CPU_qualification'
        write_json(a.output/(case.name+'__observed_scene.json'),scene)
        for index,start in enumerate(scene['window_starts_s']):
            start=round(start*5)/5;wid=scene['scene_id']+f'__w{index:02d}'
            folder=a.output/wid;folder.mkdir(exist_ok=True)
            first=round(start*15)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-threads','1','-i',str(case/'scene_video.mp4'),
                '-vf',f'select=between(n\\,{first}\\,{first+149}),setpts=N/(15*TB)','-frames:v','150','-an',
                '-c:v','libx264','-threads','1','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',str(folder/'video.mp4')],check=True)
            media={'video':'video.mp4','audio_foa':'audio_foa.wav','foa_format':'ACN_WYZX_SN3D_camera','audio_status':'not_generated_in_video_stage'}
            q,g,check=make_questions(scene,start,wid,visual,{1:'蓝色上衣的人',2:'黄色上衣的人'},media,quota_index=len(rows))
            write_json(folder/'public_questions.json',q);write_json(folder/'private_answers.json',g);write_json(folder/'question_checks.json',check)
            row={'window_id':wid,'house':case.name,'start_s':start,'logical_questions':len(q),
                'modality_counts':check['question_check']['modality_counts'],'qualification':'not_formal_until_audio_and_full_family_admission'}
            rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    write_json(a.output/'summary.json',{'windows':rows,'formal_dataset_count':0})

if __name__=='__main__':main()
