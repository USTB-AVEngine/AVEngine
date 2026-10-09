#!/usr/bin/env python3
"""Split oversized HM3D semantic regions on CPU and compare frozen manual references."""
import argparse
from pathlib import Path
from tools.rooms.room_split_auto import pipeline

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','select','evaluate','review','validate','render-overheads'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection-base',type=Path)
    parser.add_argument('--training-csv',type=Path)
    parser.add_argument('--scope',choices=['tune','holdout','reference','all'],default='all')
    parser.add_argument('--run-name',default='all_v1')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--reuse-reference-run',help='Reuse the recorded 33 reference geometries in all scope')
    args=parser.parse_args()
    if not 1<=args.workers<=16:parser.error('workers must be in [1,16]: aggregate process-tree address-space ceiling <=108 GiB')
    if args.command=='prepare':pipeline.prepare(args.output,args.selection_base,args.training_csv)
    elif args.command=='run':pipeline.run(args.output,args.scope,args.run_name,args.workers,args.reuse_reference_run)
    elif args.command=='render-overheads':
        from tools.rooms.room_split_auto import software_render
        software_render.run(args.output,min(args.workers,4))
    else:
        from tools.rooms.room_split_auto import evaluation,review
        if args.command=='select':evaluation.select(args.output,args.run_name)
        elif args.command=='evaluate':evaluation.evaluate(args.output,args.run_name,args.scope)
        elif args.command=='review':review.publish(args.output,args.run_name,args.scope)
        elif args.command=='validate':review.validate(args.output,args.run_name)
if __name__=='__main__':main()
