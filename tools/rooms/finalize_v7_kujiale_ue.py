#!/usr/bin/env python3
"""Finish native room QA from measured RGB/depth and actual UE asset bindings."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from avengine.capture.native_ue_qa import depth_geometry_check,source_materials


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--editor-audit',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start=time.monotonic()
    plan=json.loads(args.plan.read_text());audit=json.loads(args.editor_audit.read_text())
    capture=json.loads((args.capture/'result.json').read_text())
    if capture.get('status')!='native_pixels_and_floor_measured' or not capture.get('rooms'):
        raise ValueError('failed or empty native capture cannot produce completed room QA')
    result={'schema':'kujiale_native_ue_automated_room_qa_v1','house':plan['house'],'rooms':[],
            'render_path':'ue_spear','source_materials':list(source_materials(audit).values()),
            'inputs':{'plan':str(args.plan),'editor_audit':str(args.editor_audit),'capture':str(args.capture)}}
    for measured in capture['rooms']:
        room=next(r for r in plan['rooms'] if r['room_id']==measured['room_id'])
        views=[]
        for view in measured['views']:
            view_start=time.monotonic()
            depth=np.load(args.capture/(view['view']['view_id']+'_depth.npz'),allow_pickle=False)['depth_m']
            q=depth_geometry_check(audit,view['view'],depth,room);q.pop('source_material_records',None)
            views.append({**view,'native_geometry_material_checks':q,'cpu_check_seconds':time.monotonic()-view_start})
            print(plan['house'],view['view']['view_id'],q['status'],q['failures'],flush=True)
        failures=[]
        if measured['floor_reference']['status']!='pass':failures.append('NATIVE_FLOOR_OR_CAPSULE_PROBE_FAILED')
        if measured['native_pose_alignment_status']!='pass':failures.append('NATIVE_POSE_ALIGNMENT_FAILED')
        if len(views)!=4:failures.append('FEWER_THAN_THREE_VALID_WITNESS_VIEWS')
        for view in views:
            if view['metrics']['status']!='pixel_checks_pass':failures.append(view['view']['view_id']+':PIXEL_CHECK_FAILED')
            if view['native_geometry_material_checks']['status']!='pass':failures.extend(view['view']['view_id']+':'+s for s in view['native_geometry_material_checks']['failures'])
        row={**measured,'views':views,'automated_room_status':'fail' if failures else 'pass',
             'failures':failures,'native_human_foot_contact':'pending_asset_motion_check'}
        result['rooms'].append(row)
        (args.output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    result.update(elapsed_seconds=time.monotonic()-start,
                  room_geometry_visual_floor_pass_count=sum(r['automated_room_status']=='pass' for r in result['rooms']),
                  formal_training_admission_count=0,
                  formal_training_admission_status='pending_native_source_asset_motion_and_shared_generator_admission')
    (args.output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['house','elapsed_seconds','room_geometry_visual_floor_pass_count','formal_training_admission_count']}))


if __name__=='__main__':main()
