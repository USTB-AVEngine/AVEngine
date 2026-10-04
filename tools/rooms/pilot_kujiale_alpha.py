#!/usr/bin/env python3
"""Run the canonical recorded alpha question/selection/seal pipeline with UE."""
from __future__ import annotations
import argparse,json,time,traceback
from pathlib import Path
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_alpha_generation import generate_alpha_parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['house','seed-row','inputs','assets','source-admission','human-admission','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--layout',type=Path);a=p.parse_args();read=lambda p:json.loads(p.read_text());house=read(a.house);row=read(a.seed_row);inputs=read(a.inputs);assets=read(a.assets)['assets']
    if a.layout is not None:
        from tools.rooms.bind_kujiale_formal_seed import bind_formal_seed
        row=bind_formal_seed(row,house,read(a.layout))
    elif 'measured_static_layout' not in row:
        raise ValueError('room-budget seed needs its matching --layout; no inherited canary pose')
    if house['family']!='kujiale' or house['split']!='train' or house['input']['visual_backend']!='ue_spear':raise ValueError('training-only native Kujiale input required')
    a.output.mkdir(parents=True,exist_ok=a.resume);start=time.perf_counter()
    result={'status':'started','scene_id':row['scene_id'],'house':house['house'],'render_path':'ue_spear','alpha_formal_windows':0,'alpha_formal_logical_questions':0}
    try:
        result=generate_alpha_parent(row,house,inputs,assets,read(a.source_admission),read(a.human_admission),a.output)
    except BaseException as error:
        result.update(status='failed_not_formal',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        raise
    finally:
        result['whole_pipeline_wall_seconds']=time.perf_counter()-start;write_json(a.output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','scene_id','alpha_formal_windows','alpha_formal_logical_questions','whole_pipeline_wall_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
