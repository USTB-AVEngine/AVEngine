#!/usr/bin/env python3
"""Copy admitted animal/object GLBs into a task-owned UE import staging area."""
from __future__ import annotations
import argparse,datetime,json,shutil
from pathlib import Path
from avengine.assets.gltf_static_bake import bake_static_glb


def admitted_rows(path: Path):
    value=json.loads(path.read_text())
    for key in ('animals','assets'):
        for row in value.get(key,[]):
            if not isinstance(row,dict):continue
            if 'minor_variants' in row:
                for child in row['minor_variants']:
                    if child.get('dual_backend_static_usable') or child.get('dual_backend_usable'):
                        yield {**child,'category':row['major_id'],'status':'admitted','source_root':value['artifact_root']}
            elif row.get('status') in ('admitted','pass'):
                yield {**row,'source_root':value.get('artifact_root',str(path.parent.parent))}


def prepare(root: Path,registrations: list[Path],revision: str):
    records=[];seen=set()
    for registration in registrations:
        for row in admitted_rows(registration):
            aid=row['asset_id']
            if aid in seen:raise ValueError('duplicate admitted asset_id: '+aid)
            seen.add(aid)
            meta={}
            if row.get('asset_metadata'):
                meta=json.loads(Path(row['asset_metadata']).read_text())
            elif row.get('source_static_metadata'):meta=row['source_static_metadata']
            source=Path(row.get('runtime_glb') or row.get('derived_glb') or row.get('habitat_glb') or meta['runtime_glb'])
            if not source.is_absolute():source=Path(row['source_root'])/source
            dest=root/'derived'/revision/aid;dest.mkdir(parents=True,exist_ok=True)
            bake=bake_static_glb(source,dest/'static.glb')
            emit=row.get('emit_point_local_m') or meta.get('emit_point_local_m') or meta.get('mouth_m') or row.get('beak_local_glb_m')
            if row.get('emitter'):emit=row['emitter']['local_m']
            category=row.get('category','object');source_walk=bool(row.get('walk_allowed',False));walk=source_walk and category=='dog'
            motion=row.get('motion_runtime_glb') or meta.get('motion_runtime_glb') or row.get('motion_master_glb')
            if walk:
                if not motion:raise ValueError('admitted walker has no declared motion master: '+aid)
                motion=Path(motion)
                if not motion.is_absolute():motion=Path(row['source_root'])/motion
                shutil.copy2(motion,dest/'motion.glb')
            lo=row.get('bounds_min_m') or meta.get('bounds_min_m') or row.get('geometry_checks',{}).get('bbox_m',[None,None])[0]
            hi=row.get('bounds_max_m') or meta.get('bounds_max_m') or row.get('geometry_checks',{}).get('bbox_m',[None,None])[1]
            lo=lo or bake['bounds_min_m'];hi=hi or bake['bounds_max_m']
            dimensions=row.get('dimensions_m') or meta.get('dimensions_m') or ([b-a for a,b in zip(lo,hi)] if lo and hi else None)
            records.append({'asset_id':aid,'category':category,'body_id':row.get('body_id',meta.get('body_id',row.get('object_type') or ('beagle' if 'beagle' in aid else 'german_shepherd' if 'german_shepherd' in aid else 'sparrow' if 'sparrow' in aid else 'parrot' if 'parrot' in aid else aid))),
                'object_type':row.get('object_type'),'form_factor':row.get('form_factor'),'functional_front_axis':row.get('functional_front_axis',meta.get('functional_front_axis')),'perch':row.get('perch',meta.get('perch',(row.get('source_metadata') or {}).get('perch'))),'source_registration':str(registration),'source_glb':str(source),'static_glb':str(dest/'static.glb'),
                'motion_glb':str(dest/'motion.glb') if walk else None,'walk_allowed':walk,'source_walk_allowed':source_walk,'source_walk_delivery_form':row.get('walk_delivery_form'),'source_walking_frame_cache':row.get('walking_frame_cache'),'bake_receipt':bake,'attachment_surface':row.get('attachment_surface','floor'),
                'emitter_local_m':emit,'silent_decor':bool(row.get('silent_decor',False)),'label_color':row.get('label_color',row.get('color')),
                'bounds_min_m':lo,'bounds_max_m':hi,'dimensions_m':dimensions,'landmarks_local_m':row.get('landmarks_local_m',meta.get('landmarks_local_m',{})),
                'destination_path':f'/Game/V7KujialeAssets20261012/{revision}/{aid}','source_status':row['status'],'source_metadata':row.get('asset_metadata'),
                'license':row.get('license',row.get('source_license'))})
    value={'schema':'kujiale_ue_admission_inputs_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':revision,
           'registration_paths':[str(p) for p in registrations],'assets':records,'source_glbs_modified':False}
    target=root/'inputs'/f'assets_{revision}.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');return target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--registrations',type=Path,nargs='+',required=True);p.add_argument('--revision',required=True);a=p.parse_args();print(prepare(a.root,a.registrations,a.revision))

if __name__=='__main__':main()
