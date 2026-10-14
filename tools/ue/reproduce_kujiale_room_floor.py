#!/usr/bin/env python3
"""Replay retained standing-on-bed coordinates against the real native floor."""
import argparse,json
from pathlib import Path
import numpy as np
from avengine.capture.native_floor_reference import NativeFloorReference
from tools.rooms.check_kujiale_native_capsule_contact import query_capsule


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--houses',type=Path,required=True);p.add_argument('--witnesses',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=json.loads(a.houses.read_text());houses=v['houses'] if isinstance(v,dict) else v;house=next(x for x in houses if x['house']=='kujiale_0003');room=next(x for x in house['checked_rooms'] if x['room_label']=='R1')
    floor=NativeFloorReference(house['input']['ue_config']['native_floor_surface']);validation=floor.validate_probes(room['floor_reference']);rows=[]
    for index in range(4,8):
        sid=f'v7_kujiale_train_{index:06d}'
        for source in a.witnesses.glob('*/'+sid+'/visual_*/standing/standing_admission.json'):
            old=json.loads(source.read_text());check=next(x for x in old['checks'] if x['endpoint_id']=='voice2');point=np.asarray(check['native_capsule_static_geometry']['center_m'])-[0,.75,0]
            rec={'scene_id':sid,'source_receipt':str(source),'asset_id':check['asset_id'],'old_status':check['status'],'old_contact_gap_m':check['native_sole_depth']['gap_m'],'old_root_m':point.tolist(),'new_original_position_status':'not_run'}
            try:rec['sampled_original_m']=floor.sample(point);rec['new_original_position_status']='pass'
            except ValueError as error:rec.update(new_original_position_status='rejected',rejection_reason=str(error))
            candidates=[p['nav_point_m'] for p in room['floor_reference']['probes'] if p['role'].startswith('source')]
            for candidate in candidates:
                try:root=np.asarray(floor.sample(candidate))
                except ValueError:continue
                capsule=query_capsule(floor.mesh,{'witness_index':0,'role':'voice2','capsule':{'radius_cm':20.,'half_height_cm':75.,'center_ue_cm':[100*root[0],100*root[2],100*(root[1]+.75)]}})
                if capsule['status']=='pass':
                    rec.update(corrected_root_m=root.tolist(),corrected_floor_support=floor.last_support_readback,corrected_capsule=capsule);break
            rec['status']='pass_cpu_native_geometry_reproduction' if rec['new_original_position_status']=='rejected' and 'corrected_root_m' in rec else 'fail';rows.append(rec)
    out={'status':'pass_cpu_native_geometry_reproduction' if len(rows)==8 and all(x['status']=='pass_cpu_native_geometry_reproduction' for x in rows) else 'fail','house':'kujiale_0003','room_id':'R1','floor_reference_validation':validation,'native_map_path':house['input']['ue_map_path'],'results':rows,'ue_render_reproduction':'not_run'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':out['status'],'cases':len(rows),'floor_y_m':floor.room_floor_y_m,'old_heights_m':sorted(set(round(x['old_root_m'][1],6) for x in rows))},ensure_ascii=False))

if __name__=='__main__':main()
