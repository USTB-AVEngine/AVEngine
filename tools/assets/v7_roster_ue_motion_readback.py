"""Read every imported Kujiale walk key with native UE AnimPose evaluation."""
import json,math,os,traceback,hashlib
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT'])
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def U(p):return [-100*p[2],100*p[0],100*p[1]]
def world(p,root,yaw):
 a=-yaw-math.pi/2;c,s=math.cos(a),math.sin(a);base=U(root)
 return [base[0]+c*p[0]-s*p[1],base[1]+s*p[0]+c*p[1],base[2]+p[2]]
def xyz(p):return [p.x,p.y,p.z]
def error(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def run():
 imports=json.loads((T/'round3/evidence/ue_import_results.json').read_text());path=json.loads((T/'round2/registrations/path_kujiale_v3.json').read_text());out=[]
 for row in imports:
  if row['kind']!='human' or row['status']!='pass':continue
  aid=row['asset_id'];plan=json.loads((T/'round2/assets/humans'/aid/'walking_kujiale.json').read_text());pathfile=Path(plan.get('path_manifest')or T/'round2/registrations/path_kujiale_v3.json');path=json.loads(pathfile.read_text());pathhash=hashlib.sha256(pathfile.read_bytes()).hexdigest();proof=T/'round3/evidence/ue_motion'/(aid+'.json')
  if proof.exists()and json.loads(proof.read_text()).get('status')=='pass'and json.loads(proof.read_text()).get('path_manifest_sha256')==pathhash:out.append(json.loads(proof.read_text()));continue
  rec={'path_manifest_sha256':pathhash,'path_manifest':str(pathfile),'asset_id':aid,'status':'fail','frames':[],'method':'native UE AnimPoseExtensions every 30Hz key, world actor transform; scene renderer and scene collision have separate receipts','native_physics_collision':'not_claimed'}
  try:
   meshpath=next(a['path']for a in row['assets']if a['class']=='SkeletalMesh');clippath=next(a['path']for a in row['assets']if a['class']=='AnimSequence'and a['name'].endswith('Walking_Contact_kujiale'));mesh=unreal.load_asset(meshpath);clip=unreal.load_asset(clippath);opts=unreal.AnimPoseEvaluationOptions();opts.optional_skeletal_mesh=mesh;plan=json.loads((T/'round2/assets/humans'/aid/'walking_kujiale.json').read_text())
   for i,t in enumerate(path['time_s']):
    p=unreal.AnimPoseExtensions.get_anim_pose_at_time(clip,t,opts);names=[str(n)for n in unreal.AnimPoseExtensions.get_bone_names(p)];mouth=next(n for n in names if n.endswith('MUpperLip'));actual=world(xyz(unreal.AnimPoseExtensions.get_bone_pose(p,mouth,unreal.AnimPoseSpaces.WORLD).translation),path['points_m'][i],path['yaw_rad'][i]);expected=U(plan['frames'][i]['mouth_world_m']);feet={}
    for side in ['L','R']:
     fn=next(n for n in names if n.lower().replace('_','').replace(' ','').replace('-','').endswith(side.lower()+'foot'));av=world(xyz(unreal.AnimPoseExtensions.get_bone_pose(p,fn,unreal.AnimPoseSpaces.WORLD).translation),path['points_m'][i],path['yaw_rad'][i]);ex=U(plan['frames'][i]['feet'][side]['ankle_world_m']);feet[side]={'bone':fn,'actual_world_cm':av,'expected_world_cm':ex,'error_cm':error(av,ex)}
    rec['frames'].append({'frame':i,'time_s':t,'mouth_bone':mouth,'mouth_world_cm':actual,'mouth_expected_world_cm':expected,'mouth_error_cm':error(actual,expected),'feet':feet,'bone_count':len(names)})
   rec.update(skeletal_mesh=meshpath,animation=clippath,frame_count=len(rec['frames']),source_geometric_status=plan['geometric_contact_status'],maximum_mouth_error_cm=max(f['mouth_error_cm']for f in rec['frames']),maximum_ankle_error_cm=max(v['error_cm']for f in rec['frames']for v in f['feet'].values()))
   rec['status']='pass'if rec['maximum_mouth_error_cm']<.3 and rec['maximum_ankle_error_cm']<.3 and all(f['bone_count']>=81 for f in rec['frames'])else'fail'
  except Exception:rec['error']=traceback.format_exc()
  write(proof,rec);out.append(rec);unreal.log('V7_NATIVE_MOTION '+aid+' '+rec['status'])
 write(T/'round3/evidence/ue_motion_results.json',out)
run()
