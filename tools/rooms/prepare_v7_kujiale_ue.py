#!/usr/bin/env python3
"""Prepare isolated native UE requests with three frozen-rule pose witnesses."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np


def canonical_to_ue_cm(point):
    x, y, z = map(float, point)
    return [100*x, 100*z, 100*y]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--thresholds',type=Path,required=True)
    parser.add_argument('--baseline-manifest',type=Path,required=True)
    parser.add_argument('--houses',nargs='+',required=True)
    parser.add_argument('--version',type=int,default=1)
    parser.add_argument('--witness-count',type=int,default=3)
    parser.add_argument('--existing-map-version',type=int)
    args=parser.parse_args()
    import yaml, trimesh
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    baseline=json.loads(args.baseline_manifest.read_text())
    hs=prepare_installed_habitat_runtime(**baseline['runtime']).habitat_sim
    from shapely.geometry import shape
    from tools.rooms.room_selection.navigation import sample_navigation,placement
    params=yaml.safe_load(args.thresholds.read_text())
    if 'parameters' in params:params=params['parameters']
    if 'thresholds' in params:params=params['thresholds']
    params={key:(value['value'] if isinstance(value,dict) and 'value' in value else value) for key,value in params.items()}
    selection=json.loads((args.output_root/'evidence/build_selection.json').read_text())
    results=[]
    for house in args.houses:
        raw=selection.get(house) if isinstance(selection,dict) else None
        if isinstance(raw,dict):raw=raw.get('build') or raw.get('build_directory') or raw.get('selected_build')
        if raw:
            build=Path(raw);build=build if build.is_absolute() else args.output_root/'houses'/house/build
        else:
            build=args.output_root/'houses'/house/('build_v5' if house=='kujiale_0024' else 'build_v4')
        output=args.output_root/'ue_v1/houses'/house
        output.mkdir(parents=True,exist_ok=True)
        target=output/f'acceptance_plan_v{args.version}.json'
        if target.exists():raise FileExistsError(target)
        rooms=json.loads((build/'rooms_screened.json').read_text())
        vertices=np.load(build/'surface/vertices.npy',allow_pickle=False)
        faces=np.load(build/'surface/triangles.npy',allow_pickle=False)
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        pf=hs.PathFinder()
        if not pf.load_nav_mesh(str(build/'navigation/navigation.navmesh')):raise RuntimeError('navmesh load failed')
        selected=[]
        for room in rooms:
            if not room.get('geometry_eligible'):continue
            geometry=shape(room['floor_geometry'])
            sampled,points,clearance,adj,components=sample_navigation(pf,hs,geometry,room['floor_height_m'],params)
            witnesses=[room['metrics']['placement']]
            allowed=set(range(len(points)))
            for repeat in range(args.witness_count-1):
                previous=np.asarray(witnesses[-1]['camera_m'])-[0,params['camera_height_m'],0]
                distances=np.linalg.norm(points[:,[0,2]]-previous[[0,2]],axis=1)
                allowed.difference_update(np.flatnonzero(distances<.01).tolist())
                witness=placement(mesh,points,clearance,adj,params,allowed=allowed)
                if not witness.get('found'):break
                witnesses.append(witness)
            views=[]
            for i,witness in enumerate(witnesses):
                camera=np.asarray(witness['camera_m'])
                midpoint=(np.asarray(witness['source_1_m'])+witness['source_2_m'])/2
                forward=midpoint-camera;forward[1]=0;forward/=np.linalg.norm(forward)
                views.append({'view_id':f'{room["room_label"]}_camera_{i+1}','kind':'perspective',
                              'position_m':camera.tolist(),'position_ue_cm':canonical_to_ue_cm(camera),
                              'yaw_deg':math.degrees(math.atan2(forward[2],forward[0])),
                              'pitch_deg':0.,'witness':witness})
            minx,minz,maxx,maxz=geometry.bounds
            # Below the ceiling, looking down. The orthographic footprint is
            # declared; no ceiling/mesh is deleted for this diagnostic view.
            ortho_width=max(maxx-minx,(maxz-minz)*16/9)*100*1.03
            views.insert(0,{'view_id':f'{room["room_label"]}_top','kind':'orthographic',
                            'position_m':[(minx+maxx)/2,room['floor_height_m']+2.4,(minz+maxz)/2],
                            'position_ue_cm':canonical_to_ue_cm([(minx+maxx)/2,room['floor_height_m']+2.4,(minz+maxz)/2]),
                            'yaw_deg':0.,'pitch_deg':-90.,'ortho_width_cm':ortho_width})
            selected.append({'room_id':room['room_label'],'room_type':room['room_type'],
                             'floor_area_m2':room['floor_area_m2'],'floor_polygon':room['floor_geometry'],
                             'source_floor_y_m':room['floor_height_m'],'witnesses':witnesses,
                             'required_witness_count':3,'witness_count_status':'pass' if len(witnesses)>=3 else 'fail',
                             'views':views})
        source=json.loads((build/'native_ue_pending/import_request.json').read_text())
        source['result_path']=str(output/f'import_result_v{args.version}.json')
        source['map_path']=f'/Game/AVEngine/V7Kujiale20261003/NativeV{args.existing_map_version or args.version}/{house}'
        source['destination_path']=f'/Game/AVEngine/V7Kujiale20261003/ImportedNativeV{args.version}/{house}'
        source['acceptance_plan_path']=str(target)
        source['native_editor_audit_path']=str(output/f'native_editor_audit_v{args.version}.json')
        source['world_obj_path']=str(output/f'ue_world_geometry_v{args.version}.obj')
        plan={'schema':'kujiale_native_ue_acceptance_plan_v1','house':house,'build_directory':str(build),
              'render_backend':'ue_spear','map_path':source['map_path'],'rooms':selected,
              'thresholds_path':str(args.thresholds),'thresholds_sha256':hashlib.sha256(args.thresholds.read_bytes()).hexdigest(),
              'frozen_parameters':params,'capture_width':2560,'capture_height':1440,
              'delivery_width':1280,'delivery_height':720,'hfov_deg':85.,'fps':15,
              'source_geometry_unchanged':True,'original_import_request':str(build/'native_ue_pending/import_request.json')}
        target.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
        (output/f'import_request_v{args.version}.json').write_text(json.dumps(source,indent=2,ensure_ascii=False)+'\n')
        results.append({'house':house,'rooms':len(selected),'witnesses':[len(r['witnesses']) for r in selected]})
    print(json.dumps(results))


if __name__=='__main__':main()
