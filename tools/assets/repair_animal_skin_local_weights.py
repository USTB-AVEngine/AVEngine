"""Repair local skin-weight discontinuities in a derived copy; keep all geometry.

Seed the repair from observed stretched triangles, expand only through original
triangle adjacency, average coincident seam weights, then diffuse weights with
fixed unmarked boundary values. This is an asset operation, never a gate change.
"""
import argparse,copy,json,pathlib,hashlib
import numpy as np
from scipy import sparse
from avengine.assets.roster_motion.io import read_glb,write_glb
from avengine.assets.roster_motion.pose import accessor


def repair(source,seed_pose,output,*,rings=16,iterations=180):
 d,b=read_glb(source);sd,sb=read_glb(seed_pose);out=copy.deepcopy(d);payload=bytearray(b);rows=[]
 source_parts=[(ni,pi,p) for ni,n in enumerate(d['nodes']) if 'mesh' in n for pi,p in enumerate(d['meshes'][n['mesh']]['primitives'])]
 seed_parts=[p for n in sd['nodes'] if 'mesh' in n for p in sd['meshes'][n['mesh']]['primitives']]
 assert len(source_parts)==len(seed_parts)
 def append_array(array,component,type_):
  payload.extend(b'\0'*((-len(payload))%4));bv=len(out['bufferViews']);out['bufferViews'].append({'buffer':0,'byteOffset':len(payload),'byteLength':array.nbytes});payload.extend(array.tobytes());idx=len(out['accessors']);out['accessors'].append({'bufferView':bv,'componentType':component,'count':len(array),'type':type_});return idx
 for (ni,pi,p),sp in zip(source_parts,seed_parts):
  v=accessor(d,b,p['attributes']['POSITION']).astype(float);f=accessor(d,b,p['indices']).astype(int).reshape(-1,3);observed=accessor(sd,sb,sp['attributes']['POSITION']).astype(float)
  assert len(v)==len(observed) and np.array_equal(f,accessor(sd,sb,sp['indices']).reshape(-1,3))
  # Global registered scale is estimated robustly from the unchanged majority;
  # it affects only patch seed selection, never the validation baseline.
  def longest(x):
   t=x[f];return np.linalg.norm(t-np.roll(t,1,axis=1),axis=2).max(1)
  raw=longest(v);seen=longest(observed);scale=float(np.median(seen/np.maximum(raw,1e-12)));bad=seen/np.maximum(raw*scale,1e-12)>1.5
  _,weld=np.unique(np.round(v,7),axis=0,return_inverse=True);count=int(weld.max())+1;centers=np.zeros((count,3));np.add.at(centers,weld,v);counts=np.bincount(weld);centers/=counts[:,None]
  edges=np.concatenate([weld[f[:,[0,1]]],weld[f[:,[1,2]]],weld[f[:,[2,0]]]]);edges=np.unique(np.sort(edges,axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
  distance=np.linalg.norm(centers[edges[:,0]]-centers[edges[:,1]],axis=1);weight=1/np.maximum(distance,1e-8)
  graph=sparse.coo_matrix((np.r_[weight,weight],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(count,count)).tocsr();degree=np.asarray(graph.sum(1)).ravel();transition=sparse.diags(1/np.maximum(degree,1e-12))@graph
  selected=np.zeros(count,bool);selected[np.unique(weld[f[bad]])]=True;seed_count=int(selected.sum())
  for _ in range(rings):selected|=np.asarray(graph@selected.astype(float)).ravel()>0
  skin=d['skins'][d['nodes'][ni]['skin']];joint=accessor(d,b,p['attributes']['JOINTS_0']).astype(int);weights=accessor(d,b,p['attributes']['WEIGHTS_0']).astype(float);weights/=weights.sum(1,keepdims=True);dense=np.zeros((len(v),len(skin['joints'])))
  np.add.at(dense,(np.arange(len(v))[:,None],joint),weights);original=np.zeros((count,len(skin['joints'])));np.add.at(original,weld,dense);original/=counts[:,None];current=original.copy()
  for _ in range(iterations):
   proposed=.5*current+.5*(transition@current);current[selected]=proposed[selected]
  result=dense.copy();result[selected[weld]]=current[weld[selected[weld]]]
  order=np.argsort(result,axis=1)[:,-4:][:,::-1];value=np.take_along_axis(result,order,axis=1);value/=value.sum(1,keepdims=True)
  # Preserve all unselected original influences exactly, including their order.
  order[~selected[weld]]=joint[~selected[weld]];value[~selected[weld]]=weights[~selected[weld]]
  prim=out['meshes'][out['nodes'][ni]['mesh']]['primitives'][pi];prim['attributes']['JOINTS_0']=append_array(order.astype('<u2'),5123,'VEC4');prim['attributes']['WEIGHTS_0']=append_array(value.astype('<f4'),5126,'VEC4')
  rows.append({'node':ni,'primitive':pi,'vertices':len(v),'welded_vertices':count,'seed_stretched_faces':int(bad.sum()),'seed_welded_vertices':seed_count,'repaired_vertices':int(selected[weld].sum()),'total_vertices':len(v),'selection_scale_only':scale,'rings':rings,'iterations':iterations})
 out['buffers'][0]['byteLength']=len(payload);output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=False);write_glb(output,out,bytes(payload));dd,bb=read_glb(output)
 # All source payload stays intact; only two skin accessors per primitive differ.
 assert bb[:len(b)]==b
 for (ni,pi,p) in source_parts:
  p2=dd['meshes'][dd['nodes'][ni]['mesh']]['primitives'][pi]
  assert p['indices']==p2['indices']
  assert all(k in ('JOINTS_0','WEIGHTS_0') or i==p2['attributes'][k] for k,i in p['attributes'].items())
 for key in ('nodes','skins','animations','images','materials','textures','samplers'):
  assert d.get(key)==dd.get(key)
 receipt={'source':str(source),'derived_glb':str(output),'seed_pose':str(seed_pose),'method':'coincident seam averaging and local original-triangle harmonic weight diffusion with fixed external boundary; top four influences','source_sha256':hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest(),'derived_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_binary_prefix_unchanged':True,'positions_uv_topology_materials_images_nodes_bones_animations_unchanged':True,'thresholds_changed':False,'parts':rows}
 output.with_suffix('.repair.json').write_text(json.dumps(receipt,indent=2));return receipt


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--seed-pose',required=True);p.add_argument('--output',required=True);p.add_argument('--rings',type=int,default=16);p.add_argument('--iterations',type=int,default=180);a=p.parse_args();print(json.dumps(repair(a.source,a.seed_pose,a.output,rings=a.rings,iterations=a.iterations)),flush=True)
if __name__=='__main__':main()
