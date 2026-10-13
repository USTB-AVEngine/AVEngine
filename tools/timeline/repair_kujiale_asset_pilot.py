#!/usr/bin/env python3
"""Prepare fresh failed-parent recipes with native view and RIR-qualified supports."""
from pathlib import Path
from copy import deepcopy
import argparse,json,math,time
import numpy as np
from shapely.geometry import shape
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_registered_sampling import build_registered_scene,choose_assets
from avengine.timeline.v7_position_admission import PositionAdmission
from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry,import_vertices,rotation,camera_candidates
from avengine.qa.kujiale_asset_paws import landmark_contacts
from tools.timeline.prepare_kujiale_asset_pilot import separated

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--indices',default='0,5,6,7,8,9,10,11');ap.add_argument('--name-suffix',default='v1')
    args=ap.parse_args();root=args.root.resolve()
    if root.name!='ue_round21_pilot':raise ValueError('unexpected pilot root')
    rows=[json.loads(l) for l in (root/'inputs/pilot_seeds_v2.jsonl').read_text().splitlines()]
    houses={h['house']:h for h in json.loads((root/'inputs/houses.json').read_text())['houses']}
    assets=json.loads((root/'inputs/assets.json').read_text())['assets']
    physical={r['asset_id']:r for r in json.loads((root/'inputs/assets_selected.json').read_text())['assets']}
    inputs=json.loads((root/'inputs/runtime.json').read_text())
    sounds=json.loads((root/'inputs/source_admission.json').read_text())['records']
    failures=[int(x) for x in args.indices.split(',')]
    for previous in root.glob('repair_plans/*/seed.json'):
        saved=json.loads(previous.read_text())
        index=saved['repair_provenance']['logical_parent_index']
        if index not in failures:rows[index]=saved
    summary=[]
    for index in failures:
        row=rows[index];original=row['scene_id'];row['scene_id']=original+'_repair1';row['independent_parent_scene_id']=row['scene_id']
        row['repair_provenance']={'original_parent':original,'reason':'retained native geometry/acoustic/visibility/binding rejection','logical_parent_index':index}
        row['production_scenario']['camera_policy']='pilot_binding_coverage'
        row['production_scenario']['binding_camera_yaw_offset_deg']=0.
        working,scene,_=build_registered_scene(row,houses[row['house']],sounds,choose_assets(assets,row),None)
        directory=root/'repair_plans'/row['scene_id'];directory.mkdir(parents=True,exist_ok=True)
        gate=PositionAdmission(scene,working,inputs,directory/'rir_cache',directory/'native_rir_admission.json')
        geometry=NativeSupportGeometry(working['input']['ue_config']['native_floor_surface'])
        catalog=json.loads((root/'evidence'/('support_catalog_'+row['house']+'.json')).read_text())['catalog']
        polygon=shape(row['measured_static_layout']['room_floor_polygon'])
        listener=np.array(scene['listener_m']);envelopes=[]
        for pos in row['measured_static_layout']['valid'][:3]:
            point=np.array(pos.get('root_m') or [pos['position_m'][0],pos['position_m'][1]-1.3,pos['position_m'][2]])
            envelopes.append((point+[-.4,0,-.4],point+[.4,1.9,.4]))
        updated=[];rejections=[]
        try:
            for order,binding in enumerate(row['production_scenario']['native_environment']):
                reg=physical[binding['asset_id']];vertices=import_vertices(json.loads(Path(binding['import_receipt']).read_text()))
                tol=.015 if reg['category']=='object' else .012
                supports=sorted(catalog[reg['attachment_surface']],key=lambda s:abs(np.linalg.norm((np.array(s['point_m'])-listener)[[0,2]])-3.1))
                accepted=None
                for support in supports:
                    if reg.get('object_type')=='air_conditioner' and reg['attachment_surface']=='wall' and support['point_m'][1]-support['floor_y_m']<2.2:continue
                    normal=np.array(support['normal']);angles=[math.atan2(-normal[2],normal[0])] if reg['attachment_surface']=='wall' else [0.,math.pi/2,math.pi,-math.pi/2]
                    for yaw in angles:
                        rotation_m=rotation(yaw);v=vertices@rotation_m.T
                        rough=np.array(support['point_m'])-(v.min(axis=0)+v.max(axis=0))/2
                        if not separated(v.min(axis=0)+rough,v.max(axis=0)+rough,envelopes):continue
                        place=geometry.placement(vertices,support,yaw,tol)
                        if place is None:continue
                        world=v+np.array(place['root_m']);low,high=world.min(axis=0),world.max(axis=0)
                        if not separated(low,high,envelopes):continue
                        center=(low+high)/2;delta=center-listener;distance=float(np.linalg.norm(delta))
                        horizontal=float(np.linalg.norm(delta[[0,2]]))
                        if horizontal<.2 or abs(math.degrees(math.atan2(delta[1],horizontal)))>23.:continue
                        if geometry.ray(listener,delta/distance,distance-.04) is not None:continue
                        paws=landmark_contacts(geometry,reg,vertices,place)
                        if paws['status']=='fail':continue
                        point=np.array(place['root_m'])+rotation_m@np.array(reg.get('emitter_local_m') or [0,0,0])
                        endpoint='environment'+str(sum(not b['silent_decor'] for b in updated)+1)
                        acoustic={'status':'not_applicable_silent_decor'} if reg['silent_decor'] else gate.check(endpoint,point)
                        if acoustic['status']=='blocked':
                            rejections.append({'asset_id':reg['asset_id'],'point_m':point.tolist(),'angle_deg':acoustic['direction']['check']['angle_deg']});continue
                        clear=[]
                        for view in camera_candidates(vertices,place,polygon):
                            eye=np.array(view['position_m']);delta=np.array(view['target_m'])-eye;length=float(np.linalg.norm(delta))
                            if geometry.ray(eye,delta/length,length-.02) is None:clear.append(view)
                        if len(clear)<3:continue
                        place.update(paw_contacts=paws,camera_candidates=clear,native_world_bounds_m=[low.tolist(),high.tolist()])
                        accepted=deepcopy(binding);accepted.update(placement=place,expected_emitter_m=point.tolist(),preproduction_rir=acoustic,
                            production_view_geometry={'status':'clear_native_LOD_ray','listener_m':listener.tolist(),'center_m':center.tolist()})
                        envelopes.append((low,high));break
                    if accepted:break
                if accepted is None:raise ValueError('no unchanged-gate visible/RIR support for '+reg['asset_id'])
                updated.append(accepted);print('REPAIRED_SUPPORT',index,reg['asset_id'],accepted['placement']['root_m'],flush=True)
            row['production_scenario']['native_environment']=updated
            write_json(directory/'seed.json',row);write_json(directory/'rejected_positions.json',rejections)
            summary.append({'logical_parent_index':index,'scene_id':row['scene_id'],'status':'planned','assets':len(updated),'native_rir_rejected_candidates':len(rejections)})
            print('REPAIRED_PARENT',index,row['scene_id'],flush=True)
        finally:
            gate.close()
    target=root/'inputs'/('pilot_seeds_repair_'+args.name_suffix+'.jsonl')
    with target.open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json(root/'evidence'/('repair_planning_summary_'+args.name_suffix+'.json'),{'status':'pass','parents':summary,'seed_path':str(target),'labels_modified':False,'thresholds_unchanged':True})
    with (root/'PROGRESS.md').open('a') as f:f.write('- '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'：修复批次支持面、生产相机视线和原生 RIR 方向规划完成，原失败产物全部保留。\n')
if __name__=='__main__':main()
