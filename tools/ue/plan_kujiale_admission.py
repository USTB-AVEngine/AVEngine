#!/usr/bin/env python3
"""Plan admitted native assets on measured floor/table/wall/ceiling surfaces."""
from __future__ import annotations
import argparse,json,math,time,traceback
from pathlib import Path
import numpy as np
from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry,import_vertices,camera_candidates
from avengine.qa.kujiale_asset_paws import landmark_contacts


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--houses',type=Path,required=True);p.add_argument('--only',nargs='*');p.add_argument('--house-id',default='kujiale_0020');a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());revision=manifest['revision'];hvalue=json.loads(a.houses.read_text());houses=hvalue['houses'] if isinstance(hvalue,dict) else hvalue
    house=next(x for x in houses if x['house']==a.house_id);room=house['checked_rooms'][0]
    geometry=NativeSupportGeometry(house['input']['ue_config']['native_floor_surface']);catalog,polygon=geometry.support_catalog(house,room)
    write(a.root/'evidence'/f'support_catalog_{a.house_id}.json',{'house':a.house_id,'room_id':room['room_label'],'catalog':catalog,'floor_y_m':geometry.floor.room_floor_y_m,'map':house['input']['ue_map_path']})
    rows=[]
    for row in manifest['assets']:
        if a.only and row['asset_id'] not in a.only:continue
        ev=a.root/'evidence/plans'/revision/a.house_id/(row['asset_id']+'.json')
        if ev.exists() and json.loads(ev.read_text()).get('status')=='pass_geometry_plan':rows.append(json.loads(ev.read_text()));continue
        imported=a.root/'evidence/import'/revision/(row['asset_id']+'.json')
        if not imported.exists():continue
        receipt=json.loads(imported.read_text());rec={'asset_id':row['asset_id'],'house':a.house_id,'room_id':room['room_label'],'status':'fail','source_manifest':str(a.manifest),'placements':[]};started=time.time()
        try:
            if receipt['status']!='pass':raise RuntimeError('native import did not pass')
            vertices=import_vertices(receipt);lo,hi=vertices.min(axis=0),vertices.max(axis=0)
            expected=np.asarray(row['dimensions_m'],float);actual=hi-lo;errors=np.abs(actual-expected)/np.maximum(expected,1e-5)
            rec['native_dimensions_m']=actual.tolist();rec['dimensions_relative_error']=errors.tolist();rec['size_status']='pass' if errors.max()<=.15 else 'fail'
            if rec['size_status']!='pass':raise RuntimeError('native LOD0 size differs from registration beyond 15%')
            surface={'tabletop':'table'}.get(row['attachment_surface'],row['attachment_surface']);supports=catalog[surface];tolerance=.012 if row['category'] in ['dog','cat','bird'] else .015
            for support in supports:
                if row.get('object_type')=='air_conditioner' and surface=='wall' and support['point_m'][1]-support['floor_y_m']<2.2:continue
                n=np.asarray(support['normal'])
                angles=[math.atan2(-n[2],n[0])] if surface=='wall' else [0.,math.pi/2,math.pi,-math.pi/2]
                for yaw in angles:
                    placement=geometry.placement(vertices,support,yaw,tolerance)
                    if placement is None:continue
                    placement['paw_contacts']=landmark_contacts(geometry,row,vertices,placement)
                    if placement['paw_contacts']['status']=='fail':continue
                    views=camera_candidates(vertices,placement,polygon)
                    # Precheck cameras against the same native room triangles.
                    center=np.asarray(views[0]['target_m']) if views else None;clear=[]
                    for view in views:
                        eye=np.asarray(view['position_m']);d=center-eye;distance=float(np.linalg.norm(d));hit=geometry.ray(eye,d/distance,distance-.02)
                        if hit is None:clear.append(view)
                    if len(clear)<3:continue
                    placement['camera_candidates']=clear;rec['placements'].append(placement)
                    if len(rec['placements'])>=4:break
                if len(rec['placements'])>=4:break
            if not rec['placements']:raise RuntimeError('no native attachment/clearance/three-camera placement')
            rec['status']='pass_geometry_plan'
        except Exception:rec['error']=traceback.format_exc()
        rec['elapsed_seconds']=time.time()-started;write(ev,rec);rows.append(rec);write(a.root/'evidence'/f'plans_{revision}_{a.house_id}.json',rows)
        print('PLAN',a.house_id,row['asset_id'],rec['status'],len(rec['placements']),flush=True)
    write(a.root/'evidence'/f'plans_{revision}_{a.house_id}.json',rows)

if __name__=='__main__':main()
