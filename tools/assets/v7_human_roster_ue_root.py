"""Include animated non-joint armature roots in derived UE skins without changing weights."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,write_glb,json_write
from v7_human_roster_pose import accessor,globals_at
def promote(d,b):
 d=copy.deepcopy(d);payload=bytearray(b);records=[]
 for si,skin in enumerate(d['skins']):
  joints=set(skin['joints']);parents={child:i for i,n in enumerate(d['nodes'])for child in n.get('children',[])};animated={c['target']['node']for a in d.get('animations',[])for c in a['channels']};extra=set()
  for j in joints:
   p=parents.get(j)
   while p is not None:
    if p in animated and p not in joints:extra.add(p)
    p=parents.get(p)
  if not extra:continue
  old=accessor(d,bytes(payload),skin['inverseBindMatrices']);g=globals_at(d,bytes(payload));matrices=[np.linalg.inv(g[i]).T.reshape(16)for i in sorted(extra)];arr=np.concatenate([old,np.array(matrices)],axis=0).astype(np.float32);payload.extend(bytes((-len(payload))%4));vi=len(d['bufferViews']);d['bufferViews'].append({'buffer':0,'byteOffset':len(payload),'byteLength':arr.nbytes});payload.extend(arr.tobytes());ai=len(d['accessors']);d['accessors'].append({'bufferView':vi,'componentType':5126,'count':len(arr),'type':'MAT4'});skin['inverseBindMatrices']=ai;skin['joints'].extend(sorted(extra));records.append({'skin':si,'original_joint_count':len(joints),'promoted_nodes':[{'node':i,'name':d['nodes'][i].get('name')}for i in sorted(extra)],'joint_count':len(skin['joints'])})
 d['buffers'][0]['byteLength']=len(payload)
 return d,bytes(payload),records
def run(T):
 rows=[]
 for p in sorted((T/'assets/metric').glob('*/runtime.glb')):
  d,b=read_glb(p);nd,nb,info=promote(d,b);aid=p.parent.name;dest=T/'assets/ue_joint_root'/aid/'runtime.glb';write_glb(dest,nd,nb)
  # Adding an unused joint cannot change LBS: all old matrices and all indexed weights remain identical.
  for ni,node in enumerate(d['nodes']):
   if 'skin'not in node:continue
   count=len(d['skins'][node['skin']]['joints'])
   for prim in d['meshes'][node['mesh']]['primitives']:assert int(accessor(d,b,prim['attributes']['JOINTS_0']).max())<count
  for si,skin in enumerate(d['skins']):
   old=accessor(d,b,skin['inverseBindMatrices']);new=accessor(nd,nb,nd['skins'][si]['inverseBindMatrices']);assert np.array_equal(old,new[:len(old)])
  assert d['nodes']==nd['nodes']and d['animations']==nd['animations']and nb[:len(b)]==b
  rows.append({'asset_id':aid,'source':str(p.relative_to(T)),'derived':str(dest.relative_to(T)),'promotion':info,'all_original_buffer_bytes_preserved':True,'node_and_animation_channels_preserved':True,'original_ibms_preserved':True,'status':'pass'})
 json_write(T/'evidence/ue_joint_root_promotion.json',rows);print('PROMOTED_ROOTS',len(rows))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
