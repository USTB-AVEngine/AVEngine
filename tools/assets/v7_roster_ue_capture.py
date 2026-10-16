"""Sample native UE scene and isolated JPEGs for motions and paired silent seats."""
import os,json,math,time,traceback
from pathlib import Path
import unreal
unreal.EditorPythonScripting.set_keep_python_script_alive(True)
T=Path(os.environ['V7_HUMAN_TASK_ROOT'])
base=T/'worktree/tools/assets/v7_human_roster_ue_review.py'
exec(compile(base.read_text().split('class Review:')[0],str(base),'exec'),globals())
def U(p):return [-100*p[2],100*p[0],100*p[1]]
def rotp(v,a):
 c,s=math.cos(a),math.sin(a);return [c*v[0]+s*v[2],v[1],-s*v[0]+c*v[2]]
class Capture:
 def __init__(self):
  self.world,self.comp,self.rt,self.pos,self.scene=setup();self.comp.capture_every_frame=True;self.review_light=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight,vector([0,0,250]));lc=self.review_light.point_light_component;lc.set_mobility(unreal.ComponentMobility.MOVABLE);lc.set_editor_property('intensity_units',unreal.LightUnits.CANDELAS);lc.set_intensity(1200);lc.set_attenuation_radius(600);lc.set_source_radius(60);lc.set_cast_shadows(False)
  rows=json.loads((T/'round3/evidence/ue_import_results.json').read_text());self.chairs={r['asset_id']:r for r in rows if r['kind']=='chair' and r['status']=='pass'}
  fits={r['asset_id']:r for r in json.loads((T/'round3/evidence/chair_body_compatibility.json').read_text())};self.items=[]
  for r in rows:
   if r['status']!='pass':continue
   if r['kind']=='human':
    p=json.loads((T/'round2/assets/humans'/r['asset_id']/'walking_kujiale.json').read_text());path=json.loads(Path(p.get('path_manifest')or T/'round2/registrations/path_kujiale_v3.json').read_text())
    for i in sorted(set([0,len(p['frames'])//3,2*len(p['frames'])//3,len(p['frames'])-1])):
     self.items.append(dict(row=r,family='Walking',clip='Walking_Contact_kujiale',t=i/30,root=path['points_m'][i],yaw=path['yaw_rad'][i],expected=U(p['frames'][i]['mouth_world_m']),frame=i))
    f=json.loads((T/'round3/evidence/extra_actions'/r['asset_id']/'Sitting_Talk.json').read_text())['chair_fit']
    cr=[self.pos[1]/100,self.pos[2]/100,-self.pos[0]/100];offset=f['human_local_offset_m'];hr=[cr[i]+offset[i]for i in range(3)]
    self.items.append(dict(row=r,family='Sitting',clip='Sitting_Contact_kujiale',t=0,root=hr,yaw=0,seat_root=cr,fit=f))
    self.items.append(dict(row=r,family='Sitting_Talk',clip='Sitting_Talk_Contact',t=1.5,root=hr,yaw=0,seat_root=cr,fit=f))
    for family in ['Standing_Talk','Turning_Left90']:
     self.items.append(dict(row=r,family=family,clip=family+'_Contact',t=.5,root=cr,yaw=0,qualification='full contact and source-scene BVH receipts are separate'))
   elif r['kind']=='dog_motion':
    for family in ['Idle','Walking']:
     p=json.loads((T/'round2/assets/dogs'/r['asset_id']/('kujiale_'+family+'.json')).read_text())
     for i in sorted(set([0,len(p['frames'])//2,len(p['frames'])-1])):
      self.items.append(dict(row=r,family=family,clip=p['clip'],t=i/30,root=p['path_points_m'][i],yaw=p['yaw_rad'][i],expected=U(p['frames'][i]['mouth_world_m']),frame=i))
   elif r['kind']in ['chair','bird']:
    cr=[self.pos[1]/100,self.pos[2]/100,-self.pos[0]/100];self.items.append(dict(row=r,family='Static',root=cr,yaw=0))
  self.cameras=json.loads((T/'round3/registrations/ue_camera_plan.json').read_text())['cameras'];self.index=0;self.wait=45;self.phase='spawn';self.actors=[];self.results=[];self.started=time.time();self.handle=unreal.register_slate_post_tick_callback(self.tick)
 def exp(self,name):
  unreal.ImageWriteBlueprintLibrary.export_to_disk(self.rt,str(self.dest/name),unreal.ImageWriteOptions(format=unreal.DesiredImageFormat.JPG,compression_quality=85,overwrite_file=True,async_=False))
 def target(self,only):
  self.comp.clear_show_only_components()
  if only:
   for a in self.actors:self.comp.show_only_actor_components(a)
  self.comp.primitive_render_mode=unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST if only else unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES
 def begin(self):
  if self.index==len(self.items):self.finish();return
  self.job=j=self.items[self.index];r=j['row'];kind=r['kind'];aid=r['asset_id'];key=j['family']+'_'+str(j.get('frame',0));self.dest=T/'round3/images/ue'/aid/key;self.dest.mkdir(parents=True,exist_ok=True)
  saved=self.dest/'native.json'
  if saved.exists():
   old=json.loads(saved.read_text());planned=self.cameras[aid+'/'+j['family']+'/'+str(j.get('frame',0))]
   if old.get('capture_revision')=='refresh_camera_fill_v2' and old.get('status')=='pass_capture' and old.get('root_m')==j['root'] and old.get('yaw_rad')==j['yaw'] and old.get('time_s')==j.get('t',0) and old.get('source_camera')==planned and all(Path(old.get(k,'MISSING')).is_file() for k in ['rgb','isolated']) and (kind!='human' or old.get('material_slot_compatibility') is True):
    old['resume_receipt']='native pose, camera, time, root and material-slot-compatible result reused; import source unchanged'
    self.results.append(old);self.index+=1;write(T/'round3/evidence/ue_native_capture_results.json',self.results);self.phase='spawn';self.wait=0;return
  self.rec={'asset_id':aid,'kind':kind,'family':j['family'],'frame':j.get('frame'),'time_s':j.get('t',0),'root_m':j['root'],'yaw_rad':j['yaw'],'status':'pending','capture_revision':'refresh_camera_fill_v2','review_light':'native source lighting plus camera-following 1200cd soft point fill, radius60cm, no shadow; original material unchanged','ue_texture_color_difference':'not_used','qualification':j.get('qualification'),'native_physics_collision':'not_claimed'}
  angle=90-math.degrees(j['yaw']);pos=U(j['root'])
  if kind in ['human','dog_motion']:
   mesh=unreal.load_asset(next(a['path']for a in r['assets']if a['class']=='SkeletalMesh'));cp=next(a['path']for a in r['assets']if a['class']=='AnimSequence'and a['name'].endswith(j['clip']));clip=unreal.load_asset(cp);a=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkeletalMeshActor,vector(pos),unreal.Rotator(yaw=angle-180 if kind=='human'else angle));c=a.skeletal_mesh_component;c.set_mobility(unreal.ComponentMobility.MOVABLE);c.set_skeletal_mesh(mesh);c.set_editor_property('visibility_based_anim_tick_option',unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES);c.set_update_animation_in_editor(True);c.play_animation(clip,False);c.set_play_rate(0);c.set_position(j['t'],False);self.component=c
   opts=unreal.AnimPoseEvaluationOptions();opts.optional_skeletal_mesh=mesh;ap=unreal.AnimPoseExtensions.get_anim_pose_at_time(clip,j['t'],opts);self.mouth=next(str(n)for n in unreal.AnimPoseExtensions.get_bone_names(ap)if str(n).lower().replace('_','').replace('-','').replace(' ','').endswith('mupperlip'if kind=='human'else'xtramouth'));self.rec['animation']=cp;self.rec['mouth_bone']=self.mouth
   if kind=='human':
    source=json.loads((T/'evidence/ue_skeletal_instance_materials.json').read_text());mr=next((x for x in source if x['asset_id']==aid and x['status']=='pass'and x['color']=='red'),next((x for x in source if x['asset_id']==aid and x['status']=='pass'),None))
    if mr:
     native_slots=[str(x.material_slot_name)for x in mesh.get_editor_property('materials')];normalize=lambda x:x.lower().replace('_','').replace('-','').replace(' ','');self.rec['material_slot_compatibility']=all(x['index']<len(native_slots)and normalize(native_slots[x['index']])==normalize(x['slot_name'])for x in mr['slots']);self.rec['material_overrides']=mr['slots'];self.rec['native_material_slots']=native_slots
     for slot in mr['slots']:c.set_material(slot['index'],unreal.load_asset(slot['override_material']))
   self.actors.append(a)
   if 'fit'in j:
    f=j['fit'];rchair=self.chairs[f['chair_id']];parts=[a['path']for a in rchair['assets']if a['class']=='StaticMesh'];ca=spawn(parts,U(j['seat_root']),angle)
    for a in ca:a.set_actor_scale3d(unreal.Vector(1,1,f['instance_scale_y']))
    self.actors+=ca;self.rec['chair_id']=f['chair_id'];self.rec['chair_scale_y']=f['instance_scale_y'];self.rec['contact_source']='round3/evidence/chair_body_compatibility.json; scene placement proof is separate'
  else:self.actors=spawn([a['path']for a in r['assets']if a['class']=='StaticMesh'],pos,angle)
  center=j.get('seat_root',j['root']);height=.95 if kind=='dog_motion'else 1.22;distance=2.4 if kind=='dog_motion'else 2.8;eye=[center[i]+rotp([0,height,-distance],j['yaw'])[i]for i in range(3)];target=[center[0],center[1]+(.40 if kind=='dog_motion'else .90),center[2]]
  if kind in ['chair','bird']:target[1]=center[1]+(.25 if kind=='bird'else .52)
  if kind=='bird':
   eye=[center[0],center[1]+.40,center[2]-.85];self.comp.fov_angle=45
  else:self.comp.fov_angle=50
  key=aid+'/'+j['family']+'/'+str(j.get('frame',0));planned=self.cameras[key]
  if not planned.get('source_clear_rays'):raise RuntimeError('No source-clear review camera')
  eye=planned['eye_glb_m'];target=planned['target_glb_m'];self.comp.fov_angle=planned['fov_degrees'];self.rec['source_camera']=planned
  self.review_light.set_actor_location(vector(U([eye[0],eye[1]+.4,eye[2]])),False,False);cam=self.comp.get_owner();cam.set_actor_location(vector(U(eye)),False,False);cam.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(vector(U(eye)),vector(U(target))),False);self.target(False);self.phase='rgb';self.wait=8
 def complete(self):
  j=self.job
  lo,hi=boxes(self.actors);self.rec['native_bounds_cm']={'min':lo,'max':hi};self.rec['height_m']=(hi[2]-lo[2])/100
  if j['row']['kind']in ['human','dog_motion']:
   v=xyz(self.component.get_socket_location(self.mouth));self.rec['mouth_world_cm']=v
   if 'expected'in j:self.rec['mouth_expected_world_cm']=j['expected'];self.rec['mouth_error_cm']=math.sqrt(sum((a-b)**2 for a,b in zip(v,j['expected'])))
  if j['row']['kind']=='bird':
   bmeta=json.loads((T/'round3/assets/bird_sparrow_fallback/source.json').read_text());point=bmeta['beak_local_glb_m'];worldpoint=[j['root'][i]+rotp(point,j['yaw'])[i]for i in range(3)];self.rec['beak_world_cm']=U(worldpoint);self.rec['mouth_bone']='static original library beak candidate'
  self.rec.update(rgb=str(self.dest/'rgb.jpg'),isolated=str(self.dest/'isolated.jpg'),status='pass_capture'if self.rec.get('mouth_error_cm',0)<.3 else'fail')
  write(self.dest/'native.json',self.rec);self.results.append(self.rec);unreal.log('V7_CAPTURE '+self.rec['asset_id']+' '+self.rec['family']+' '+self.rec['status'])
  for a in self.actors:unreal.EditorLevelLibrary.destroy_actor(a)
  self.actors=[];self.index+=1;write(T/'round3/evidence/ue_native_capture_results.json',self.results);self.phase='spawn';self.wait=2
 def tick(self,delta):
  try:
   if self.wait:self.wait-=1;return
   if self.phase=='spawn':self.begin()
   elif self.phase=='rgb':self.exp('rgb.jpg');self.target(True);self.phase='isolated';self.wait=4
   elif self.phase=='isolated':self.exp('isolated.jpg');self.complete()
   if time.time()-self.started>3600:raise RuntimeError('Capture one-hour watchdog')
  except Exception:write(T/'round3/evidence/ue_capture_fatal.json',{'error':traceback.format_exc(),'index':self.index,'job':self.items[self.index]if self.index<len(self.items)else None});self.finish()
 def finish(self):
  for a in self.actors:unreal.EditorLevelLibrary.destroy_actor(a)
  self.actors=[];unreal.unregister_slate_post_tick_callback(self.handle);write(T/'round3/evidence/ue_native_capture_results.json',self.results);unreal.SystemLibrary.quit_editor()
try:
 _job=Capture()
 unreal.log('V7_CAPTURE_READY '+str(len(_job.items)))
except Exception:
 write(T/'round3/evidence/ue_capture_fatal.json',{'error':traceback.format_exc(),'phase':'initialization'})
 unreal.SystemLibrary.quit_editor()
 raise
