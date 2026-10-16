"""Audit original mesh, old animation surfaces and inverse binds without modifying assets."""
import argparse,json
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,json_write
from v7_human_roster_pose import pose,accessor,animation_times
def run(T):
 out=[]
 for row in json.loads((T/'registrations/selection.json').read_text())['avatars']:
  aid=row['asset_id'];od,ob=read_glb(T/'assets/metric'/aid/'runtime.glb');nd,nb=read_glb(T/'round2/assets/humans'/aid/'final_ue_root.glb');fail=[];attrs=0
  for mi,mesh in enumerate(od['meshes']):
   for pi,p in enumerate(mesh['primitives']):
    npart=nd['meshes'][mi]['primitives'][pi]
    for name,ai in p['attributes'].items():
     attrs+=1
     if not np.array_equal(accessor(od,ob,ai),accessor(nd,nb,npart['attributes'][name])):fail.append('attribute '+name)
    if 'indices'in p and not np.array_equal(accessor(od,ob,p['indices']),accessor(nd,nb,npart['indices'])):fail.append('indices')
  for i,skin in enumerate(od['skins']):
   old=accessor(od,ob,skin['inverseBindMatrices']);new=accessor(nd,nb,nd['skins'][i]['inverseBindMatrices'])
   if not np.array_equal(old,new[:len(old)]):fail.append('inverse binds')
   if skin['joints']!=nd['skins'][i]['joints'][:len(skin['joints'])]:fail.append('joint order')
  maxerr=0.
  for ac in od['animations']:
   nac=next(a for a in nd['animations']if a['name']==ac['name']);ts=animation_times(od,ob,ac)
   for t in [float(ts[0]),float((ts[0]+ts[-1])/2),float(ts[-1])]:
    op,_=pose(od,ob,ac,t);np_,_=pose(nd,nb,nac,t);maxerr=max(maxerr,max(float(np.linalg.norm(a['positions']-b['positions'],axis=1).max())for a,b in zip(op,np_)))
  if maxerr>1e-6:fail.append('old animated surface')
  out.append({'asset_id':aid,'status':'pass'if not fail else'fail','equal_original_attribute_arrays':attrs,'maximum_old_animation_surface_error_m':maxerr,'failures':fail,'original':str(T/'assets/metric'/aid/'runtime.glb'),'new_master':str(T/'round2/assets/humans'/aid/'final_ue_root.glb')})
 json_write(T/'round3/evidence/source_geometry_invariance.json',out);print('SOURCE_REST',len(out),sum(r['status']=='pass'for r in out),max(r['maximum_old_animation_surface_error_m']for r in out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
