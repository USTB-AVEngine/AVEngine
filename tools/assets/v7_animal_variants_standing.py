"""Owner-authorized v6 static Idle derivatives, bounded grounding, and size measurements."""
import argparse,copy,json,math,sys,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from v7_animal_variants_v6_candidates import semantic_masks,descendants,sha,dump
from v7_human_roster_pose import read_glb,pose,animation_times,globals_at,rigid_glb

BODY={'abyssinian':('cat','abyssinian','阿比西尼亚猫'),'labrador':('dog','labrador','拉布拉多'),'shiba_inu':('dog','shiba_inu','柴犬'),'british_shorthair':('cat','british_shorthair','英国短毛猫'),'burmese':('cat','burmese','缅甸猫'),'jack_russell_terrier':('dog','jack_russell_terrier','杰克罗素梗'),'siamese':('cat','siamese','暹罗猫'),'border_collie':('dog','border_collie','边境牧羊犬'),'pembroke_welsh_corgi':('dog','pembroke_welsh_corgi','彭布罗克威尔士柯基')}
PAWS=['paw_front_left','paw_front_right','paw_hind_left','paw_hind_right']

def tilt_matrix(pitch,roll):
 x,z=np.radians([pitch,roll]);cx,sx,cz,sz=np.cos(x),np.sin(x),np.cos(z),np.sin(z)
 return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

def support_measure(parts,masks):
 feet={}
 for key in PAWS:
  v=np.concatenate([p['positions'][m[key]] for p,m in zip(parts,masks) if m[key].any()]);bottom=v[v[:,1]<=v[:,1].min()+.003];q=np.median(bottom,axis=0);q[1]=v[:,1].min();feet[key]=q
 head=np.concatenate([p['positions'][m['head']] for p,m in zip(parts,masks) if m['head'].any()]);v=np.concatenate([p['positions'] for p in parts]);return feet,head,v

def size_metrics(v,feet,category,perch=None):
 fs=np.array(list(feet.values()));height=float(np.quantile(v[:,1],.998)-v[:,1].min());out={'robust_top_height_m':height,'extent_height_m':float(np.ptp(v[:,1]))}
 if category!='bird':
  zz=np.sort(fs[:,2]);lo,hi=zz[:2].mean(),zz[2:].mean();core=v[(v[:,2]>lo+(hi-lo)*.22)&(v[:,2]<lo+(hi-lo)*.72)]
  out.update(fore_hind_paw_span_m=float(hi-lo),shoulder_proxy_m=float(np.quantile(core[:,1],.95)-np.median(fs[:,1])))
 else:out.update(body_length_m=float(np.ptp(v[:,2])),body_height_without_perch_m=float(np.ptp(v[:,1])))
 return out

def transform_parts(parts,rotation,translation):
 out=copy.deepcopy(parts)
 for p in out:
  p['positions']=p['positions']@rotation.T+translation
  if p['normals'] is not None:p['normals']=p['normals']@rotation.T
 return out

def evaluated_pose(d,b,action,t,masks,yaw,pitch=0,roll=0,joint_edits=None):
 g=globals_at(d,b,action,float(t));joint_edits=joint_edits or []
 for edit in joint_edits:
  for n in descendants(d,edit['node_index']):g[n]=g[n].copy();g[n][:3,3]+=np.array(edit['source_world_translation_m'])
 parts,_=pose(d,b,action,float(t),canonical_flip=False,ground=False,globals_override=g);rotation=tilt_matrix(pitch,roll)@yaw;parts=transform_parts(parts,rotation,np.zeros(3));feet,head,v=support_measure(parts,masks);translation=np.array([0,-v[:,1].min(),0]);parts=transform_parts(parts,np.eye(3),translation);feet,head,v=support_measure(parts,masks);maximum=max(abs(x[1]) for x in feet.values());return parts,feet,head,v,maximum,rotation,translation

def grounding_choices(frames):
 ranked=sorted(frames,key=lambda f:(np.ptp([x['support_local_m'][1] for x in f['feet'].values()]),f['frame']))
 out=[]
 for row in ranked:
  fs=np.array([row['feet'][k]['support_local_m'] for k in PAWS]);grid=[]
  for pitch in np.linspace(-3,3,25):
   for roll in np.linspace(-3,3,25):
    ys=(fs@tilt_matrix(pitch,roll).T)[:,1];grid.append((float(np.ptp(ys)),abs(pitch)+abs(roll),float(pitch),float(roll)))
  grid.sort();out.append((row,[(0.,0.)]+[(x[2],x[3]) for x in grid[:6] if x[2]!=0 or x[3]!=0]))
 return out

def build(T):
 P=T/'append_20261012';A=json.loads((T/'evidence/A_provenance.json').read_text());rows=[]
 for a in A['animals']:
  aid=a['asset_id'];orig=T/'C_candidates'/aid
  if not (orig/'candidate.json').exists():orig=P/'C_first_four'/aid
  D=P/'v6_standing'/aid;D.mkdir(parents=True,exist_ok=True)
  try:
   cm=json.loads((orig/'candidate.json').read_text());d,b=read_glb(cm['motion_runtime_glb']);manifest=json.loads(Path(cm['source_package_manifest']).read_text());masks,head_node=semantic_masks(d,b,manifest);idle=next(x for x in d['animations'] if x['name']=='Idle');yaw=np.array(cm['rotation']);frames=cm['CPU_actions'][0]['frames'];prior=[]
   for house in ['mp3d','hm3d']:
    rp=T/'evidence/C_native'/house/(aid+'_Idle.json')
    if not rp.exists():rp=P/'evidence/C_first_four'/house/(aid+'_Idle.json')
    prior.append(json.loads(rp.read_text()) if rp.exists() else {'status':'not_run'})
   original_pass=all(x['status']=='pass' for x in prior);choices=[(frames[0],[(0.,0.)])] if original_pass else grounding_choices(frames);chosen=None;attempts=[]
   for row,tilts in choices:
    for pitch,roll in tilts:
     q=evaluated_pose(d,b,idle,row['time_s'],masks,yaw,pitch,roll);attempts.append({'frame':row['frame'],'time_s':row['time_s'],'pitch_deg':pitch,'roll_deg':roll,'maximum_paw_delta_m':q[4]})
     if q[4]<=.012+1e-8:chosen=(row,pitch,roll,q,[]);break
    if chosen:break
   if not chosen:
    # Last resort only: translate authored terminal paw subtree by <=3 cm.
    anchors={x['anchor_id']:x['joint_id'] for x in manifest['anchors']};nodes={x.get('name'):i for i,x in enumerate(d['nodes'])};row,tilts=choices[0];pitch,roll=min([(x['pitch_deg'],x['roll_deg']) for x in attempts if x['frame']==row['frame']],key=lambda pr:next(x['maximum_paw_delta_m'] for x in attempts if x['frame']==row['frame'] and (x['pitch_deg'],x['roll_deg'])==pr));q=evaluated_pose(d,b,idle,row['time_s'],masks,yaw,pitch,roll);edits=[]
    for key,pt in sorted(q[1].items(),key=lambda kv:kv[1][1],reverse=True)[:1]:
     if pt[1]>.0115:
      delta=float(pt[1]-.004)
      if delta>.03+1e-9:continue
      world=(q[5].T@np.array([0,-delta,0])).tolist();edits.append({'paw':key,'joint_name':anchors[key],'node_index':nodes[anchors[key]],'translation_norm_m':delta,'source_world_translation_m':world,'method':'terminal paw subtree world translation before linear skinning; no source animation mutation'})
    q=evaluated_pose(d,b,idle,row['time_s'],masks,yaw,pitch,roll,edits);attempts.append({'frame':row['frame'],'time_s':row['time_s'],'pitch_deg':pitch,'roll_deg':roll,'joint_edits':edits,'maximum_paw_delta_m':q[4]})
    if q[4]<=.012+1e-8:chosen=(row,pitch,roll,q,edits)
   dump(D/'grounding_attempts.json',{'attempts':attempts,'threshold_m':.012,'maximum_pitch_roll_deg':3,'maximum_terminal_translation_m':.03})
   if not chosen:raise RuntimeError('all Idle frames and bounded final endpoint correction fail 12 mm contact gate')
   row,pitch,roll,q,edits=chosen;parts,feet,head,v,maximum,rotation,translation=q;g=globals_at(d,b,idle,row['time_s']);mouth=rotation@(g[head_node][:3,:3]@np.array(cm['emitter_head_local_m'])+g[head_node][:3,3])+translation;distance=float(np.linalg.norm(head-mouth,axis=1).min());key=next(k for k in BODY if aid.startswith('generated_'+k+'_'));cat,body,name=BODY[key];runtime=D/'runtime.glb';rigid_glb(d,b,parts,runtime);body_only=D/'body_only.glb';body_only.write_bytes(runtime.read_bytes());landmarks={f'foot_{i}':feet[key].tolist() for i,key in enumerate(PAWS)};landmarks.update(mouth=mouth.tolist(),head_center=head.mean(0).tolist(),head_bbox_min=head.min(0).tolist(),head_bbox_max=head.max(0).tolist());tri=sum(len(p['positions']) for p in parts);dd,bb=read_glb(runtime)
   from v7_human_roster_pose import accessor
   triangles=sum(len(accessor(dd,bb,p['indices']))//3 for m in dd['meshes'] for p in m['primitives'])
   meta={'asset_id':aid,'category':cat,'body_id':body,'body_name_zh':name,'mode':'v6_idle_static_derivative','walk_allowed':False,'status':'pending_native_acceptance','runtime_glb':str(runtime),'body_only_glb':str(body_only),'runtime_sha256':sha(runtime),'dimensions_m':np.ptp(v,axis=0).tolist(),'bounds_min_m':v.min(0).tolist(),'bounds_max_m':v.max(0).tolist(),'landmarks_local_m':landmarks,'emit_point_local_m':mouth.tolist(),'head_center_local_m':head.mean(0).tolist(),'emit_head_distance_m':distance,'emit_head_radius_m':.08,'emit_head_status':'pass' if distance<.08 and np.all(mouth>=head.min(0)-.03) and np.all(mouth<=head.max(0)+.03) else 'fail','triangles':triangles,'glb_bytes':runtime.stat().st_size,'perch':None,'cross_requirement_exempt_companion':True,'intended_color_label':None,'label_color':None,'body_floor_y_m':float(v[:,1].min()),'source_provenance':a,'source_original_sha256':cm['source_sha256'],'source_original_glb':cm['source_glb'],'source_original_package_manifest':cm['source_package_manifest'],'source_original_package_sha256':cm['source_package_sha256'],'production_material_receipt':cm['production_material_receipt'],'license':{'source':a['licenses'],'owner_authorized_all_v6_reference_chains':True,'unverified_fields_preserved':a.get('notes',a['generator_3d']),'downstream_ai_training_admission':'pending_native_checks; owner-authorized provenance exception'},'idle_derivative':{'original_Idle_full_clip_native_pass':original_pass,'selected_frame':row['frame'],'selected_time_s':row['time_s'],'pitch_deg':pitch,'roll_deg':roll,'translation_m':translation.tolist(),'root_source_rotation':yaw.tolist(),'terminal_joint_edits':edits,'maximum_paw_delta_m':maximum,'method':'unchanged Idle frame baked static' if original_pass else 'best flat Idle frame baked, bounded rigid tilt/translation; endpoint adjustment only last resort'},'normalization':{'method':'v6 Idle static derivative; original package immutable','four_paw_height_range_m':float(np.ptp([pt[1] for pt in feet.values()])),'shoulder_height_m':size_metrics(v,feet,cat)['shoulder_proxy_m'],'maximum_contact_delta_m':maximum,'body_up_status':'pass' if head[:,1].mean()>np.mean([pt[1] for pt in feet.values()])+.05 else 'fail'},'size_measurements':size_metrics(v,feet,cat)}
   dump(D/'asset.json',meta);rows.append({'asset_id':aid,'status':'built','metadata':str(D/'asset.json'),**meta['idle_derivative']});print('V6_STATIC',aid,rows[-1],flush=True)
  except Exception as e:rows.append({'asset_id':aid,'status':'fail','error':repr(e),'traceback':traceback.format_exc()});print('V6_STATIC_FAIL',aid,repr(e),flush=True)
  dump(P/'reports/v6_standing_build.json',rows)
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);a=p.parse_args();rows=build(a.task_root);return int(any(x['status']!='built' for x in rows))
if __name__=='__main__':raise SystemExit(main())
