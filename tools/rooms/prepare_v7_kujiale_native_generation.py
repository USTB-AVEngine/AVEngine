#!/usr/bin/env python3
"""Prepare explicit native UE inputs from already measured physical room QA."""
from __future__ import annotations
import argparse,json,hashlib
from copy import deepcopy
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def geometry_support_alignment(plan,audit,rooms):
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    b=Path(plan['build_directory']).resolve()
    index=read(b/'engine_asset_index.json')
    manifest_path=(b/index['acoustic_package']).resolve()
    if b not in manifest_path.parents:raise ValueError('acoustic package escapes the derived build')
    m=read(manifest_path)
    acoustic=np.load(manifest_path.parent/m['arrays']['vertices']['path'],allow_pickle=False)
    source=np.load(b/'surface/vertices.npy',allow_pickle=False)
    if source.shape!=acoustic.shape:raise ValueError('acoustic/source vertex shape differs')
    error=float(np.max(np.abs(source.astype(float)-acoustic.astype(float))))
    if error>.0005:raise ValueError('acoustic/source coordinates disagree; no fitting is allowed')
    af=np.load(manifest_path.parent/m['arrays']['triangles']['path'],allow_pickle=False)
    ng=audit['native_render_geometry'];nv=np.fromfile(ng['world_vertices_ue_cm_f32'],dtype='<f4').reshape(-1,3)[:,[0,2,1]]/100
    nf=np.fromfile(ng['triangles_u32'],dtype='<u4').reshape(-1,3)
    ar=RayMeshIntersector(trimesh.Trimesh(vertices=acoustic,faces=af,process=False));nr=RayMeshIntersector(trimesh.Trimesh(vertices=nv,faces=nf,process=False))
    directions=np.array([v for v in np.ndindex(3,3,3) if v!=(1,1,1)],float)-1
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    checks=[]
    for room in rooms:
        for witness in room['selected_witnesses'][:3]:
            for role in ('camera','source_1','source_2'):
                point=np.asarray(witness['placement'][role+'_m'])
                origins=np.repeat(point[None],len(directions),axis=0)
                def distances(ray):
                    loc,ids,_=ray.intersects_location(origins,directions,multiple_hits=False)
                    values=np.full(len(directions),np.inf);values[ids]=np.linalg.norm(loc-origins[ids],axis=1);return values
                a,n=distances(ar),distances(nr);both=np.isfinite(a)&np.isfinite(n)
                mismatch=int(np.count_nonzero(np.isfinite(a)!=np.isfinite(n)))
                max_error=float(np.max(np.abs(a[both]-n[both]))) if both.any() else 0.
                checks.append({'room_id':room['room_id'],'witness_index':witness['index'],'role':role,
                    'point_m':point.tolist(),'direction_count':len(directions),'matched_surface_ray_count':int(both.sum()),
                    'native_acoustic_hit_presence_mismatch':mismatch,'max_surface_distance_error_m':max_error,
                    'status':'pass' if mismatch==0 and max_error<=.0005 else 'fail'})
    return {'status':'pass' if checks and all(x['status']=='pass' for x in checks) else 'fail',
        'source_acoustic_vertex_max_error_m':error,'alignment_fitting_used':False,'tolerance_m':.0005,
        'sampling':'three native witnesses per room, camera/two emitters, 26 surrounding rays each; checks openings and solid surfaces',
        'acoustic_manifest':str(manifest_path),'checks':checks}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--editor-audit',type=Path,required=True)
    p.add_argument('--qa',type=Path,required=True);p.add_argument('--ue-config',type=Path,required=True)
    p.add_argument('--seed-template',type=Path,required=True);p.add_argument('--source-material-inventory',type=Path,required=True)
    p.add_argument('--scene-index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    plan,audit,qa=read(args.plan),read(args.editor_audit),read(args.qa)
    rooms=[r for r in qa['rooms'] if r['automated_room_status']=='pass']
    if not rooms:raise ValueError('no native room passed physical visual/floor QA')
    alignment=geometry_support_alignment(plan,audit,rooms);write(args.output/'native_acoustic_alignment.json',alignment)
    if alignment['status']!='pass':raise ValueError('native/acoustic support geometry disagrees')
    ue=read(args.ue_config);ue['source_material_inventory']=str(args.source_material_inventory)
    ue['acceptance_plan_path']=str(args.plan)
    checked=[]
    for r in rooms:
        selected=r['selected_witnesses'][0]['placement']
        checked.append({'room_label':r['room_id'],'room_type':r['room_type'],'placement':selected,'floor_reference':r['floor_reference'],
            'native_witnesses':r['selected_witnesses'],'native_qa_status':'pass'})
    build=Path(plan['build_directory'])
    nav=build/'navigation/scene.navmesh'
    if not nav.exists():
        navs=list((build/'navigation').glob('*.navmesh'))
        if len(navs)!=1:raise ValueError('exact native navigation path must be unambiguous')
        nav=navs[0]
    house={'schema':'v7-admitted-physical-native-house-inputs-v1','house':plan['house'],'split':'train','family':'kujiale',
        'known_exposure':True,'acoustic_status':'pass','checked_room_ids':[r['room_label'] for r in checked],
        'checked_rooms':checked,'native_qa_result':str(args.qa),'native_rgb_status':'automated_native_ue_pass',
        'acoustic_package_manifest':alignment['acoustic_manifest'],
        'input':{'house':plan['house'],'family':'kujiale','visual_backend':'ue_spear','ue_map_path':plan['map_path'],'ue_config':ue,
            'navmesh_source':str(nav),'semantic_source':str(build/'rooms_screened.json')},
        'formal_dataset_count':0,'qualification_scope':'physical native room only; shared v7 formal generator blockers remain'}
    # Keep the actual v7 30 s / two 10 s windows / 14 logical questions contract.
    template=next(json.loads(s) for s in args.seed_template.read_text().splitlines() if json.loads(s)['members']==['v0_a0'])
    from avengine.timeline.v7_production_contract import derived_seed
    row=deepcopy(template);row.update(house=plan['house'],room_id=checked[0]['room_label'],scene_index=args.scene_index,
        scene_id='v7_kujiale_native_pilot_'+plan['house'].removeprefix('kujiale_'),scene_seed=derived_seed('train',args.scene_index,'scene'))
    row['seeds']={key:derived_seed('train',args.scene_index,key) for key in row['seeds']}
    write(args.output/'house.json',house);write(args.output/'seed_row.json',row)
    print(json.dumps({'house':plan['house'],'room_count':len(checked),'native_acoustic_alignment':alignment['status'],'output':str(args.output)}))


if __name__=='__main__':main()
