"""Validate each native UE dog animation key and its four paw / mouth transforms."""
import os,json,math,traceback,hashlib
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT'])
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def xyz(p):return [p.x,p.y,p.z]
def U(p):return [-100*p[2],100*p[0],100*p[1]]
def world(p,root,yaw):
 a=math.pi/2-yaw;c,s=math.cos(a),math.sin(a);b=U(root)
 return [b[0]+c*p[0]-s*p[1],b[1]+s*p[0]+c*p[1],b[2]+p[2]]
def error(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def norm(n):return n.lower().replace('_','').replace(' ','').replace('-','')
def run():
 out=[]
 for r in json.loads((T/'round3/evidence/ue_import_results.json').read_text()):
  if r['kind']!='dog_motion' or r['status']!='pass':continue
  aid=r['asset_id'];meshpath=next(a['path']for a in r['assets']if a['class']=='SkeletalMesh');mesh=unreal.load_asset(meshpath);opts=unreal.AnimPoseEvaluationOptions();opts.optional_skeletal_mesh=mesh
  for family in ['Idle','Walking']:
   plan=json.loads((T/'round2/assets/dogs'/aid/('kujiale_'+family+'.json')).read_text());rec={'asset_id':aid,'family':family,'status':'fail','frames':[],'method':'native UE AnimPose readback every authored 30Hz key; original geometry scene collision recorded separately'}
   try:
    cp=next(a['path']for a in r['assets']if a['class']=='AnimSequence' and a['name'].endswith(plan['clip']));clip=unreal.load_asset(cp)
    for i,f in enumerate(plan['frames']):
     p=unreal.AnimPoseExtensions.get_anim_pose_at_time(clip,i/30,opts);names=[str(n)for n in unreal.AnimPoseExtensions.get_bone_names(p)];mouth=next(n for n in names if norm(n).endswith('xtramouth'));root=plan['path_points_m'][i];yaw=plan['yaw_rad'][i]
     def point(n):return world(xyz(unreal.AnimPoseExtensions.get_bone_pose(p,n,unreal.AnimPoseSpaces.WORLD).translation),root,yaw)
     m=point(mouth);feet={}
     for side,suffix in [('FL','lhand'),('FR','rhand'),('HL','lfoot'),('HR','rfoot')]:
      n=next(n for n in names if norm(n).endswith(suffix));v=point(n);feet[side]={'bone':n,'world_cm':v,'error_cm':error(v,U(f['feet'][side]['ankle_world_m']))}
     rec['frames'].append({'frame':i,'mouth_bone':mouth,'mouth_world_cm':m,'mouth_error_cm':error(m,U(f['mouth_world_m'])),'feet':feet,'bone_count':len(names)})
    rec.update(animation=cp,skeletal_mesh=meshpath,frame_count=len(rec['frames']),source_geometric_status=plan['geometric_status'],maximum_mouth_error_cm=max(f['mouth_error_cm']for f in rec['frames']),maximum_paw_error_cm=max(v['error_cm']for f in rec['frames']for v in f['feet'].values()))
    rec['status']='pass'if rec['maximum_mouth_error_cm']<.3 and rec['maximum_paw_error_cm']<.3 else'fail'
   except Exception:rec['error']=traceback.format_exc()
   write(T/'round3/evidence/ue_dog_motion'/(aid+'_'+family+'.json'),rec);out.append(rec);unreal.log('V7_DOG_NATIVE '+aid+' '+family+' '+rec['status'])
 write(T/'round3/evidence/ue_dog_motion_results.json',out)
run()
