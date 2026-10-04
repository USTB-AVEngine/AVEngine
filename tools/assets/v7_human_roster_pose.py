"""glTF linear skinning, metric pose GLBs and resumable Habitat motion cache.

Only materialized rigid poses are passed to Habitat. A glTF animation file
alone does not make a Habitat rigid object walk. No source asset is modified.
"""
import argparse,copy,hashlib,io,json,sys
from PIL import Image
from pathlib import Path
import numpy as np
from v7_human_roster_materials import read_glb,write_glb,viewbytes,json_write

DTYPES={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
WIDTH={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}

def accessor(d,b,i):
 a=d['accessors'][i];v=d['bufferViews'][a['bufferView']];dt=np.dtype(DTYPES[a['componentType']]);n=WIDTH[a['type']];off=v.get('byteOffset',0)+a.get('byteOffset',0);stride=v.get('byteStride',n*dt.itemsize)
 x=np.ndarray((a['count'],n),dt,buffer=b,offset=off,strides=(stride,dt.itemsize)).copy()
 if a.get('normalized') and a['componentType']!=5126:
  lim=np.iinfo(dt);x=x.astype(float)/lim.max
  if lim.min<0:x=np.maximum(x,-1)
 return x

def quat_matrix(q):
 q=np.asarray(q,float);q=q/np.linalg.norm(q);x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def slerp(a,b,t):
 a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);dot=float(a@b)
 if dot<0:b=-b;dot=-dot
 if dot>.9995:r=a*(1-t)+b*t;return r/np.linalg.norm(r)
 angle=np.arccos(np.clip(dot,-1,1));return (a*np.sin((1-t)*angle)+b*np.sin(t*angle))/np.sin(angle)

def animation_times(d,b,action):
 # A constant root channel may have one key; other bones carry the real motion.
 return np.unique(np.concatenate([accessor(d,b,s['input']).ravel()for s in action['samplers']]))

def globals_at(d,b,action=None,t=0):
 nodes=copy.deepcopy(d['nodes'])
 if action is not None:
  for c in action['channels']:
   target=c['target'];sam=action['samplers'][c['sampler']];times=accessor(d,b,sam['input']).ravel();values=accessor(d,b,sam['output'])
   assert sam.get('interpolation','LINEAR') in ('LINEAR','STEP'),sam
   j=int(np.searchsorted(times,t,side='right')-1);j=max(0,min(j,len(times)-1));k=min(j+1,len(times)-1);u=0 if k==j else float((t-times[j])/(times[k]-times[j]));u=np.clip(u,0,1)
   if sam.get('interpolation')=='STEP':u=0
   v=slerp(values[j],values[k],u) if target['path']=='rotation' else values[j]*(1-u)+values[k]*u
   nodes[target['node']][target['path']]=v.tolist();nodes[target['node']].pop('matrix',None)
 parent={}
 for i,n in enumerate(nodes):
  for j in n.get('children',[]):parent[j]=i
 # Unreferenced export helper nodes must not parent the actual scene root.
 roots=d['scenes'][d.get('scene',0)]['nodes']
 for i in roots:parent.pop(i,None)
 result={}
 def calc(i):
  if i in result:return result[i]
  n=nodes[i]
  if 'matrix'in n:m=np.array(n['matrix']).reshape(4,4).T
  else:
   m=np.eye(4);m[:3,:3]=quat_matrix(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1]));m[:3,3]=n.get('translation',[0,0,0])
  result[i]=calc(parent[i])@m if i in parent else m;return result[i]
 for i in range(len(nodes)):calc(i)
 return result

def pose(d,b,action=None,t=0,canonical_flip=True):
 g=globals_at(d,b,action,t);out=[]
 for ni,node in enumerate(d['nodes']):
  if 'mesh'not in node:continue
  for pi,p in enumerate(d['meshes'][node['mesh']]['primitives']):
   pos=accessor(d,b,p['attributes']['POSITION']).astype(float);norm=accessor(d,b,p['attributes']['NORMAL']).astype(float) if 'NORMAL'in p['attributes']else None
   if 'skin'in node:
    skin=d['skins'][node['skin']];ibm=accessor(d,b,skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
    matrices=np.stack([g[j]@ib for j,ib in zip(skin['joints'],ibm)])
    js=accessor(d,b,p['attributes']['JOINTS_0']).astype(int);ws=accessor(d,b,p['attributes']['WEIGHTS_0']).astype(float);ws/=ws.sum(axis=1,keepdims=True)
    trans=np.einsum('vk,vkij->vij',ws,matrices[js]);pos=np.einsum('vij,vj->vi',trans[:,:3,:3],pos)+trans[:,:3,3]
    if norm is not None:norm=np.einsum('vij,vj->vi',trans[:,:3,:3],norm)
   else:
    m=g[ni];pos=pos@m[:3,:3].T+m[:3,3]
    if norm is not None:norm=norm@np.linalg.inv(m[:3,:3])
   if canonical_flip:pos[:,[0,2]]*=-1
   if norm is not None:
    if canonical_flip:norm[:,[0,2]]*=-1
    norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-9)
   out.append({'node':ni,'primitive':pi,'positions':pos,'normals':norm})
 floor=min(float(p['positions'][:,1].min()) for p in out);allpos=np.concatenate([p['positions'] for p in out])
 mouthids=[i for i,n in enumerate(d['nodes']) if n.get('name','').endswith('MUpperLip')]
 mouth=g[mouthids[0]][:3,3].copy() if mouthids else None
 if mouth is not None:
  if canonical_flip:mouth[[0,2]]*=-1
  mouth[1]-=floor
 for p in out:p['positions'][:,1]-=floor
 allpos[:,1]-=floor
 evidence={'time_s':t,'pre_ground_floor_y_m':floor,'ground_correction_m':-floor,'bounds_min_m':allpos.min(axis=0).tolist(),'bounds_max_m':allpos.max(axis=0).tolist(),'mouth_m':None if mouth is None else mouth.tolist(),'mouth_nearest_surface_m':None if mouth is None else float(np.linalg.norm(allpos-mouth,axis=1).min()),'canonical_front':'-Z' if canonical_flip else 'source','floor_y_m':float(allpos[:,1].min())}
 return out,evidence

def rigid_glb(d,b,posed,dest):
 nd={k:copy.deepcopy(d[k]) for k in ['asset','materials','textures','samplers','images','extensionsUsed','extensionsRequired'] if k in d}
 nd.update(buffers=[{'byteLength':0}],bufferViews=[],accessors=[],meshes=[],nodes=[],scenes=[{'nodes':[]}],scene=0)
 binary=bytearray()
 def add_payload(payload,target=None):
  binary.extend(b'\0'*((-len(binary))%4));i=len(nd['bufferViews']);v={'buffer':0,'byteOffset':len(binary),'byteLength':len(payload)}
  if target:v['target']=target
  nd['bufferViews'].append(v);binary.extend(payload);return i
 def add_ac(x,atype,component=5126,normalized=False):
  x=np.asarray(x,dtype=DTYPES[component]);i=len(nd['accessors']);v=add_payload(x.tobytes(),34963 if atype=='SCALAR' and component!=5126 else 34962);a={'bufferView':v,'componentType':component,'count':len(x),'type':atype}
  if normalized:a['normalized']=True
  if atype=='VEC3':a.update(min=x.min(axis=0).tolist(),max=x.max(axis=0).tolist())
  nd['accessors'].append(a);return i
 for im in nd.get('images',[]):
  assert 'bufferView'in im;payload=viewbytes(d,b,im['bufferView'])
  if im.get('mimeType')=='image/webp' or payload[:4]==b'RIFF':
   image=Image.open(io.BytesIO(payload));buf=io.BytesIO();image.save(buf,format='PNG');payload=buf.getvalue();im['mimeType']='image/png'
  im['bufferView']=add_payload(payload)
 for p in posed:
  old=d['meshes'][d['nodes'][p['node']]['mesh']]['primitives'][p['primitive']];new={k:copy.deepcopy(v) for k,v in old.items() if k not in ('attributes','indices','targets')}
  attrs={'POSITION':add_ac(p['positions'],'VEC3')}
  if p['normals']is not None:attrs['NORMAL']=add_ac(p['normals'],'VEC3')
  for key,i in old['attributes'].items():
   if key.startswith('TEXCOORD'):
    a=d['accessors'][i];attrs[key]=add_ac(accessor(d,b,i),a['type'],a['componentType'],False)
   elif key.startswith('COLOR'):
    a=d['accessors'][i];values=accessor(d,b,i)
    # Omit artist-default white colours; preserve any actual colour as floats.
    if not np.allclose(values,1):attrs[key]=add_ac(values,a['type'],5126,False)
  new['attributes']=attrs
  if 'indices'in old:
   a=d['accessors'][old['indices']];new['indices']=add_ac(accessor(d,b,old['indices']),a['type'],a['componentType'])
  mi=len(nd['meshes']);nd['meshes'].append({'name':f'pose_mesh_{mi}','primitives':[new]});nd['nodes'].append({'name':f'pose_mesh_{mi}','mesh':mi});nd['scenes'][0]['nodes'].append(mi)
 for tex in nd.get('textures',[]):
  ext=tex.get('extensions',{})
  if 'EXT_texture_webp'in ext:
   tex['source']=ext.pop('EXT_texture_webp')['source']
   if not ext:tex.pop('extensions',None)
 for key in ['extensionsUsed','extensionsRequired']:
  if key in nd:nd[key]=[x for x in nd[key]if x!='EXT_texture_webp']
 nd['buffers'][0]['byteLength']=len(binary);write_glb(dest,nd,bytes(binary))

def normalize(T,legacy):
 sys.path.insert(0,str(legacy/'tools'))
 from normalize_rocketbox_glb_for_ue import normalize_in_place_grounded_metric_glb_bytes
 for p in sorted((T/'assets/rocketbox').glob('*/runtime.glb')):
  dest=T/'assets/metric'/p.parent.name/'runtime.glb'
  if dest.exists():continue
  d,b=read_glb(p);r=next(i for i,n in enumerate(d['nodes']) if n.get('name')=='HumanRosterRoot');root=d['nodes'][r];arm=root['children'][0];n=d['nodes'][arm]
  n['translation']=[x+y for x,y in zip(n.get('translation',[0,0,0]),root.get('translation',[0,0,0]))];root['children']=[];d['scenes'][d.get('scene',0)]['nodes']=[arm]
  temp=dest.parent/'pre_metric.glb';write_glb(temp,d,b);raw,e=normalize_in_place_grounded_metric_glb_bytes(temp.read_bytes());dest.write_bytes(raw);json_write(dest.with_name('normalization.json'),e)
  print('NORMALIZED',p.parent.name,flush=True)

def build_poses(T):
 results=[]
 for base in sorted((T/'assets/runtime').glob('*/*/runtime.glb')):
  standing=base.with_name('habitat.glb');evpath=base.with_name('poses.json')
  if standing.exists() and evpath.exists():results.append(json.loads(evpath.read_text()));continue
  d,b=read_glb(base);actions={a['name']:a for a in d.get('animations',[])}
  info={'asset_id':base.parent.parent.name,'color':base.parent.name,'poses':{},'habitat_glb':str(standing.relative_to(T)),'actions':list(actions)}
  for family,aname in [('stand','Standing_Idle'),('sit','Sitting_Idle')]:
   if aname not in actions:
    if family=='sit':continue
    aname=next(iter(actions))
   ac=actions[aname];times=accessor(d,b,ac['samplers'][0]['input']).ravel();t=float(times[0]);ps,ev=pose(d,b,ac,t)
   dest=standing if family=='stand'else base.with_name('sitting.glb');rigid_glb(d,b,ps,dest);info['poses'][family]=ev
  # Geometry-only cache shared by all shirt variants.
  cache=T/'assets/motion'/base.parent.parent.name
  if not (cache/'motion.json').exists():
   cache.mkdir(parents=True,exist_ok=True);motion={}
   for family,aname in [('walk','Walking'),('stand','Standing_Idle'),('sit','Sitting_Idle')]:
    if aname not in actions:continue
    ac=actions[aname];sam=ac['samplers'][0];times=accessor(d,b,sam['input']).ravel()
    # Every source frame, including the last endpoint; no synthetic motion.
    sampled=animation_times(d,b,ac) if family=='walk' else np.array([times[0]]);records=[];data={}
    for fi,t in enumerate(sampled):
     ps,ev=pose(d,b,ac,float(t));records.append(ev)
     for pi,p in enumerate(ps):
      data[f'p_{fi}_{pi}']=p['positions'].astype('float32')
      if p['normals']is not None:data[f'n_{fi}_{pi}']=p['normals'].astype('float32')
    np.savez_compressed(cache/(family+'.npz'),**data);motion[family]={'source_action':aname,'frame_count':len(records),'times_s':sampled.tolist(),'frames':records,'cache':str((cache/(family+'.npz')).relative_to(T))}
   json_write(cache/'motion.json',motion)
  info['motion_manifest']=str((cache/'motion.json').relative_to(T));json_write(evpath,info);results.append(info);print('POSED',base.parent.parent.name,base.parent.name,flush=True)
 json_write(T/'evidence/pose_results.json',results)

def materialize(base,motion,family,frame,dest):
 d,b=read_glb(base);meta=json.loads((motion/'motion.json').read_text())[family];fi=frame%meta['frame_count'];data=np.load(motion/(family+'.npz'))
 layout,_=pose(d,b,d['animations'][0],0)
 for i,p in enumerate(layout):p['positions']=data[f'p_{fi}_{i}'];p['normals']=data[f'n_{fi}_{i}'] if f'n_{fi}_{i}'in data else None
 rigid_glb(d,b,layout,dest);return meta['frames'][fi]

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['normalize','poses','materialize']);a.add_argument('--task-root',type=Path,required=True);a.add_argument('--legacy-root',type=Path);a.add_argument('--base',type=Path);a.add_argument('--motion',type=Path);a.add_argument('--family',default='walk');a.add_argument('--frame',type=int,default=0);a.add_argument('--output',type=Path);x=a.parse_args()
 if x.mode=='normalize':normalize(x.task_root,x.legacy_root)
 elif x.mode=='poses':build_poses(x.task_root)
 else:json_write(x.output.with_suffix('.pose.json'),materialize(x.base,x.motion,x.family,x.frame,x.output))
