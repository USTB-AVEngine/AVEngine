"""Read native UE speech/turn bone positions at every baked action key."""
import os,json,math,traceback
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT'])
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def xyz(p):return [p.x,p.y,p.z]
def U(p):return [-100*p[2],100*p[0],100*p[1]]
def error(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def run():
 out=[]
 for r in json.loads((T/'round3/evidence/ue_import_results.json').read_text()):
  if r['kind']!='human' or r['status']!='pass':continue
  aid=r['asset_id'];meshpath=next(a['path']for a in r['assets']if a['class']=='SkeletalMesh');mesh=unreal.load_asset(meshpath);opts=unreal.AnimPoseEvaluationOptions();opts.optional_skeletal_mesh=mesh
  for family in ['Standing_Talk','Turning_Left90','Sitting_Talk']:
   plan=json.loads((T/'round3/evidence/extra_actions'/aid/(family+'.json')).read_text());rec={'asset_id':aid,'family':family,'source_geometric_status':plan['geometric_status'],'status':'fail','frames':[]}
   try:
    cp=next(a['path']for a in r['assets']if a['class']=='AnimSequence'and a['name'].endswith(plan['clip']));clip=unreal.load_asset(cp)
    for i,f in enumerate(plan['frames']):
     p=unreal.AnimPoseExtensions.get_anim_pose_at_time(clip,i/30,opts);names=[str(n)for n in unreal.AnimPoseExtensions.get_bone_names(p)];mouth=next(n for n in names if n.endswith('MUpperLip'))
     def pt(n):
      v=xyz(unreal.AnimPoseExtensions.get_bone_pose(p,n,unreal.AnimPoseSpaces.WORLD).translation);return [v[1],-v[0],v[2]]
     m=pt(mouth);feet={}
     if 'feet'in f:
      for side in ['L','R']:
       n=next(n for n in names if n.lower().replace('_','').replace(' ','').replace('-','').endswith(side.lower()+'foot'));v=pt(n);feet[side]={'bone':n,'world_cm':v,'error_cm':error(v,U(f['feet'][side]['ankle_world_m']))}
     rec['frames'].append({'frame':i,'time_s':i/30,'mouth_world_cm':m,'mouth_error_cm':error(m,U(f['mouth_m'])),'feet':feet,'bone_count':len(names)})
    rec.update(animation=cp,skeletal_mesh=meshpath,frame_count=len(rec['frames']),maximum_mouth_error_cm=max(f['mouth_error_cm']for f in rec['frames']),maximum_ankle_error_cm=max([v['error_cm']for f in rec['frames']for v in f['feet'].values()]or[0]),method='native UE AnimPose at each baked 30Hz key, canonical world yaw=-90deg, position zero')
    rec['status']='pass'if rec['maximum_mouth_error_cm']<.3 and rec['maximum_ankle_error_cm']<.3 else'fail'
   except Exception:rec['error']=traceback.format_exc()
   write(T/'round3/evidence/ue_extra_actions'/aid/(family+'.json'),rec);out.append(rec);unreal.log('V7_EXTRA_NATIVE '+aid+' '+family+' '+rec['status'])
 write(T/'round3/evidence/ue_extra_action_results.json',out)
run()
