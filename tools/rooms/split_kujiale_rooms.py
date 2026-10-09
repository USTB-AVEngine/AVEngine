#!/usr/bin/env python3
"""Select InteriorAgent CAD rooms and build real CPU overhead reviews."""
from __future__ import annotations
import argparse,os,sys
from pathlib import Path


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--stage',required=True,choices=['prepare','render','stairs','select','assemble','review'])
    a.add_argument('--root',required=True);a.add_argument('--old-root');a.add_argument('--dataset-root')
    a.add_argument('--workers',type=int,default=4);a.add_argument('--houses',nargs='*')
    v=a.parse_args();root=Path(v.root)
    if v.stage=='prepare':
        if not v.old_root or not v.dataset_root:a.error('prepare requires --old-root and --dataset-root')
        from tools.rooms.room_split_kujiale.adapter import plan_inputs,coordinate_evidence,dump
        from tools.rooms.room_split_auto.pipeline import load_native
        plan=plan_inputs(v.old_root,v.dataset_root,root);hs=load_native()
        recs=[coordinate_evidence(h,hs) for h in plan['houses']]
        dump(root/'evidence/coordinate_alignment_v1.json',dict(houses=recs,samples=sum(x['samples'] for x in recs),
             inside_cad_union_count=sum(x['inside_cad_union_count'] for x in recs),cpu_only=True,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0)))
        if any(x['status']!='aligned_numeric' for x in recs):raise RuntimeError('Coordinate mismatch: stop without guessing')
    elif v.stage=='render':
        from tools.rooms.room_split_kujiale.prepare_scene import main as render_main
        sys.argv=[sys.argv[0],'--root',str(root),'--workers',str(v.workers)]+(['--houses']+v.houses if v.houses else [])
        render_main()
    elif v.stage=='stairs':
        from tools.rooms.room_split_kujiale.stairs import prepare
        prepare(root)
    elif v.stage=='select':
        from tools.rooms.room_split_kujiale.pipeline import run
        run(root,v.workers)
    elif v.stage=='assemble':
        from tools.rooms.room_split_kujiale.delivery import assemble
        assemble(root)
    else:
        from tools.rooms.room_split_kujiale.review import build
        build(root)

if __name__=='__main__':main()
