"""Restore byte-identical baseline mesh and source actions after Blender adds clips."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import accessor
def run(T):
 out=[]
 for row in json.loads((T/'registrations/selection.json').read_text())['avatars']:
  aid=row['asset_id'];D=T/'round2/assets/humans'/aid;old,ob=read_glb(T/'assets/metric'/aid/'runtime.glb');d,b=read_glb(D/'runtime.glb');binary=bytearray(b)
  if [n.get('name')for n in old['nodes']]!=[n.get('name')for n in d['nodes']]:raise ValueError('Node order differs')
  def add(ai):
   a=copy.deepcopy(old['accessors'][ai]);v=old['bufferViews'][a['bufferView']];offset=v.get('byteOffset',0);data=ob[offset:offset+v['byteLength']]
   binary.extend(bytes((-len(binary))%4));nv=copy.deepcopy(v);nv['byteOffset']=len(binary);nv['buffer']=0;vi=len(d['bufferViews']);d['bufferViews'].append(nv);binary.extend(data);a['bufferView']=vi;idx=len(d['accessors']);d['accessors'].append(a);return idx
  for m,n in zip(old['meshes'],d['meshes']):
   if len(m['primitives'])!=len(n['primitives']):raise ValueError('Primitive count differs')
   for p,r in zip(m['primitives'],n['primitives']):
    if not np.array_equal(accessor(old,ob,p['indices']),accessor(d,b,r['indices'])):raise ValueError('Topology differs')
    for k in ['POSITION','NORMAL','TANGENT','JOINTS_0','WEIGHTS_0','TEXCOORD_0','COLOR_0','COLOR_1']:
     if k in p['attributes']:r['attributes'][k]=add(p['attributes'][k])
  for sk,nsk in zip(old['skins'],d['skins']):
   if sk['joints']!=nsk['joints']:raise ValueError('Joint order differs')
   nsk['inverseBindMatrices']=add(sk['inverseBindMatrices'])
  for n,nn in zip(old['nodes'],d['nodes']):
   for k in ['translation','rotation','scale','matrix']:
    nn.pop(k,None)
    if k in n:nn[k]=copy.deepcopy(n[k])
  for ac in old.get('animations',[]):
   name=ac['name']
   if name not in ['Walking','Standing_Idle','Sitting_Idle']:continue
   nac=copy.deepcopy(ac)
   for sampler in nac['samplers']:
    sampler['input']=add(sampler['input']);sampler['output']=add(sampler['output'])
   d['animations']=[nac if a['name']==name else a for a in d['animations']]
  d['buffers'][0]['byteLength']=len(binary);write_glb(D/'runtime.glb',d,bytes(binary))
  out.append({'asset_id':aid,'baseline':str(T/'assets/metric'/aid/'runtime.glb'),'status':'pass','mesh_and_skin_byte_identical':True,'baseline_clips_restored':['Walking','Standing_Idle','Sitting_Idle']})
 json_write(T/'round2/evidence/baseline_mesh_restoration.json',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
