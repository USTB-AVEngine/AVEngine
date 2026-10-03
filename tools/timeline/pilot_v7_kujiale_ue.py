#!/usr/bin/env python3
"""Run one measured Kujiale UE/SPEAR research pilot through unchanged v7 stages.

Only automated physical room checks allow a pilot. Formal v7 blockers are
preserved by the shared generator; this tool never publishes training release.
"""
from __future__ import annotations
import argparse, json, resource, time
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--house',type=Path,required=True)
    p.add_argument('--seed-row',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--visual-canary-seconds',type=float)
    args=p.parse_args()
    house,row,inputs=read(args.house),read(args.seed_row),read(args.inputs)
    assets=read(args.assets)['assets']
    if house['input'].get('visual_backend')!='ue_spear':raise ValueError('native UE backend must be explicit')
    if house['split']!='train' or row['split']!='train':raise ValueError('Kujiale is training only')
    qa=read(house['native_qa_result'])
    measured=next(r for r in qa['rooms'] if r['room_id']==row['room_id'])
    if measured['automated_room_status']!='pass':raise ValueError('room has not passed automated native visual/floor checks')
    if len(measured['selected_witnesses'])<3:raise ValueError('room needs three frozen legal native witnesses')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();self_before=resource.getrusage(resource.RUSAGE_SELF);child_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    from avengine.timeline.v7_candidate_generation import seeded_scene,generate_parent,FORMAL_BLOCKERS
    from avengine.timeline.continuous_pilot import write_json
    receipt={'schema':'kujiale_v7_native_research_pilot_v1','house':house['house'],'room_id':row['room_id'],
        'house_input':str(args.house),'seed_row':str(args.seed_row),'render_path':'ue_spear',
        'formal_dataset_count':0,'blocking_reasons':list(FORMAL_BLOCKERS),'status':'started'}
    try:
        from avengine.acoustics.runtime import load_habitat_runtime
        from avengine.capture.native_ue_trajectory import check_native_trajectories
        working,scene=seeded_scene(row,house)
        hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
        pf=hs.PathFinder()
        if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('CPU navmesh preflight load failed')
        plan=read(house['input']['ue_config']['acceptance_plan_path']);params=plan['frozen_parameters']
        check=check_native_trajectories(scene,pf,source_height_m=params['source_height_m'],
                                        source_clearance_m=params['placement_source_clearance_m'])
        receipt['continuous_navigation_preflight']=check
        if check['status']!='pass':raise ValueError('unchanged seeded native motion fails frozen source clearance/navigation preflight')
        if args.visual_canary_seconds:
            from avengine.capture.native_continuous_backend import create_native_video
            scene['duration_s']=args.visual_canary_seconds
            video=create_native_video(working,scene,inputs,assets)
            try:
                visual=video.encode(scene,args.output/'scene_video.mp4',review_frames=[0,round(args.visual_canary_seconds*15)-1])
            finally:video.close()
            receipt.update(status='visual_canary_measured',visual_summary={k:visual[k] for k in ['frame_count','wall_seconds','native_source_material_status','native_body_foot_status']})
        else:
            result=generate_parent(row,house,inputs,assets,args.output,compact=False)
            write_json(args.output/'parent_result.json',result)
            windows=[]
            for member in result['members']:
                for w in member['windows']:
                    directory=Path(member['windows_directory'])/w['window_id']
                    files=[f for f in directory.rglob('*') if f.is_file()]
                    windows.append({**w,'directory':str(directory),'bytes':sum(f.stat().st_size for f in files),
                        'file_count':len(files),'files':[{'name':str(f.relative_to(directory)),'bytes':f.stat().st_size} for f in files]})
            receipt.update(status='complete_research_candidate',result=result,windows=windows)
    except BaseException as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        now=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(elapsed_seconds=time.perf_counter()-start,
            process_cpu_seconds=now.ru_utime+now.ru_stime-self_before.ru_utime-self_before.ru_stime,
            waited_child_cpu_seconds=child.ru_utime+child.ru_stime-child_before.ru_utime-child_before.ru_stime,
            logical_output_bytes=sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file() and not f.is_symlink()),
            cpu_source='resource.getrusage SELF and waited CHILDREN excludes detached UE; updated guard records owned-tree CPU lower bound when present')
        write(args.output/'result.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['house','room_id','status','elapsed_seconds','logical_output_bytes']},ensure_ascii=False))


if __name__=='__main__':main()
