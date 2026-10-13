"""Restore the registered source scale of an existing owned Blender rig."""
import argparse,copy,json,pathlib,importlib.util
import numpy as np
from scipy.spatial import cKDTree
from avengine.assets.roster_motion.io import read_glb,write_glb
from avengine.assets.roster_motion.pose import pose
from avengine.assets.roster_motion.episode import triangle_mesh
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,required=True);p.add_argument('--registration',type=pathlib.Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 old=json.loads((a.case/'rigging.json').read_text());row=next(x for x in json.loads(a.registration.read_text())['assets'] if x['asset_id']==a.asset)
 d,b=read_glb(a.case/'motion.glb');sd,sb=read_glb(row['runtime_glb'])
 parts,_=pose(d,b,None,ground=False,canonical_flip=False);v,f=triangle_mesh(d,b,parts)
 source,_=pose(sd,sb,None,ground=False,canonical_flip=False);sv,sf=triangle_mesh(sd,sb,source)
 T=np.asarray(old['support_plane']['matrix']);sv=sv@T[:3,:3].T+T[:3,3]
 ratios=np.ptp(sv,axis=0)/np.ptp(v,axis=0);scale=float(np.mean(ratios))
 if np.ptp(ratios)>2e-5:raise ValueError('source_rig_scale_mismatch_is_not_uniform:'+str(ratios))
 translation=(sv.min(0)+sv.max(0)-scale*(v.min(0)+v.max(0)))/2
 normalized=v*scale+translation
 if len(sf)!=len(f):raise ValueError('source_triangle_count_changed')
 st=sv[sf];dt=normalized[f];tree=cKDTree(st.mean(1));distance,order=tree.query(dt.mean(1),k=1)
 if float(distance.max())>2e-6:raise ValueError('rest_surface_correspondence_failed:'+str(distance.max()))
 # Corner matching independent of vertex splitting and cyclic corner order.
 delta=np.linalg.norm(dt[:,:,None,:]-st[order,None,:,:],axis=3)
 error=float(max(delta.min(2).max(),delta.min(1).max()))
 if error>2e-6:raise ValueError('source_rest_shape_changed:'+str(error))
 spec=importlib.util.spec_from_file_location('preserve',pathlib.Path(__file__).with_name('check_quadruped_source_preservation.py'))
 preserve=importlib.util.module_from_spec(spec);spec.loader.exec_module(preserve)
 images1,images2=preserve.pixels(sd,sb),preserve.pixels(d,b)
 equal_images=len(images1)==len(images2) and all(np.array_equal(x,y) for x,y in zip(images1,images2))
 def canonical(m):
  x=copy.deepcopy(m);x.setdefault('doubleSided',False);p=x.setdefault('pbrMetallicRoughness',{})
  p.setdefault('roughnessFactor',1.0);p.setdefault('metallicFactor',1.0);p.setdefault('baseColorFactor',[1.0]*4);return x
 equal_pbr=[canonical(m) for m in sd.get('materials',[])]==[canonical(m) for m in d.get('materials',[])]
 if not equal_images or not equal_pbr:raise ValueError('source_color_or_highlight_preservation_failed')
 root=len(d['nodes']);d['nodes'].append({'name':'RegisteredSourceScaleCorrection','scale':[scale]*3,'translation':translation.tolist(),'children':list(d['scenes'][d.get('scene',0)]['nodes'])})
 d['scenes'][d.get('scene',0)]['nodes']=[root]
 from avengine.assets.roster_motion.pose import globals_at,animation_times
 from avengine.assets.roster_motion.contact import serialize_action
 action=next(x for x in d['animations'] if x['name']=='Walking')
 idle=next(x for x in d['animations'] if x['name']=='Idle')
 reference=globals_at(d,b,idle,0.);body=next(i for i,n in enumerate(d['nodes']) if n.get('name')=='Body')
 upper=[i for i,n in enumerate(d['nodes']) if n.get('name') in ('Body','Spine','Chest','Neck','Head') or n.get('name','').startswith('Tail')]
 times=animation_times(d,b,action);frames=[];source_inverse=np.linalg.inv(reference[body])
 for t in times:
  g=globals_at(d,b,action,float(t));fixed=reference[body].copy();fixed[:3,3]=g[body][:3,3];placement=fixed@source_inverse
  for i in upper:g[i]=placement@reference[i]
  frames.append(g)
 d['animations']=[x for x in d['animations'] if x['name']!='Walking']
 d,b=serialize_action(d,b,'Walking',times,frames)
 path=a.output/'motion.glb';write_glb(path,d,b)
 candidate=json.loads((a.case/'candidate.json').read_text())
 from avengine.assets.animal_motion.profiles import Rig
 record=copy.deepcopy(json.loads((a.case/'inventory.json').read_text())['assets'][0])
 candidate['source_package_manifest']=str(a.case/'manifest.json')
 # Resolve actual new Idle head before deriving the current registered emitter.
 cp=a.output/'candidate.json';cp.write_text(json.dumps(candidate,indent=2))
 record['walking_candidate']=str(cp)
 for key in ['motion_runtime_glb','motion_master_glb']:record['files'][key]={'path':str(path)}
 rig=Rig.load(record);g=rig.globals(0,'Idle')
 emit=T[:3,:3]@np.array(row['emit_point_local_m'])+T[:3,3]
 candidate['emitter_head_local_m']=(np.linalg.inv(g[rig.head])@np.r_[emit,1])[:3].tolist();cp.write_text(json.dumps(candidate,indent=2))
 (a.output/'inventory.json').write_text(json.dumps({'assets':[record]},indent=2))
 current=copy.deepcopy(row)
 def transformed(point):return (T[:3,:3]@np.asarray(point)+T[:3,3]).tolist()
 current['emit_point_local_m']=transformed(row['emit_point_local_m']);current['head_center_local_m']=transformed(row['head_center_local_m'])
 for k,value in list(current['landmarks_local_m'].items()):
  if isinstance(value,list) and len(value)==3:current['landmarks_local_m'][k]=transformed(value)
 lo=np.array(row['landmarks_local_m']['head_bbox_min']);hi=np.array(row['landmarks_local_m']['head_bbox_max'])
 corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])@T[:3,:3].T+T[:3,3]
 current['landmarks_local_m']['head_bbox_min']=corners.min(0).tolist();current['landmarks_local_m']['head_bbox_max']=corners.max(0).tolist()
 (a.output/'request.json').write_text(json.dumps({'registration':current},indent=2))
 corrected=copy.deepcopy(old);corrected.update(source_scale_correction=scale,source_translation_correction_m=translation.tolist(),motion_runtime_glb=str(path),
 actual_authored_speed_m_s=old['root_speed_m_s']*scale,actual_authored_lift_m=.008*scale,upper_body_head_tail_stable=True,upper_body_motion_method='fixed own Idle relative upper body; original target-owned Walking limb world transforms retained')
 (a.output/'rigging.json').write_text(json.dumps(corrected,indent=2))
 result={'status':'pass','source':row['runtime_glb'],'original_owned_rig':str(a.case/'motion.glb'),'derived_glb':str(path),
 'uniform_scale':scale,'axis_scale_ratios':ratios.tolist(),'translation_m':translation.tolist(),'rest_triangle_corner_max_error_m':error,
 'source_and_derived_triangles':len(sf),'decoded_texture_pixels_equal':equal_images,'semantic_pbr_values_equal':equal_pbr,
 'vertex_splitting_retained':True,'surface_repair_used':False,'source_modified':False}
 (a.output/'source_preservation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
