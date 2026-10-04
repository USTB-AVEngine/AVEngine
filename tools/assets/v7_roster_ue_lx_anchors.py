"""Verify static LX primary sound anchors on their actual native UE Blueprint actors."""
import os,json,math,traceback
from pathlib import Path
import unreal
unreal.EditorPythonScripting.set_keep_python_script_alive(True)
T=Path(os.environ['V7_HUMAN_TASK_ROOT']);base=T/'worktree/tools/assets/v7_human_roster_ue_review.py'
exec(compile(base.read_text().split('class Review:')[0],str(base),'exec'),globals())
def U(p):return [-100*p[2],100*p[0],100*p[1]]
class Job:
 def __init__(self):
  self.world,self.comp,self.rt,self.pos,self.scene=setup();self.comp.capture_every_frame=True;self.rows=json.loads((T/'round3/registrations/lx_sound_anchors.json').read_text());self.humans={m['major_id']:m for m in json.loads((T/'registrations/human_roster.json').read_text())['majors']};self.i=0;self.phase='spawn';self.wait=45;self.out=[];self.body=None;self.emitter=None;self.handle=unreal.register_slate_post_tick_callback(self.tick)
 def begin(self):
  if self.i==len(self.rows):self.finish();return
  r=self.rows[self.i];self.row=r;v=self.humans[r['asset_id']]['minor_variants'][0];cls=unreal.load_class(None,v['ue']['generated_class_path']);yaw=self.scene['actor_yaw_degrees'];self.body=unreal.EditorLevelLibrary.spawn_actor_from_class(cls,vector(self.pos),unreal.Rotator(yaw=yaw));p=r['primary_local_glb_m'];a=math.radians(90-yaw);c,s=math.cos(a),math.sin(a);local=[c*p[0]+s*p[2],p[1],-s*p[0]+c*p[2]];delta=U(local);self.expected=[self.pos[i]+delta[i]for i in range(3)];self.emitter=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.TargetPoint,vector(self.expected));self.emitter.set_actor_hidden_in_game(True);self.emitter.attach_to_actor(self.body,'',unreal.AttachmentRule.KEEP_WORLD,unreal.AttachmentRule.KEEP_WORLD,unreal.AttachmentRule.KEEP_WORLD,False);actual=xyz(self.emitter.get_actor_location());err=math.dist(actual,self.expected);moved=[self.pos[0],self.pos[1]+10,self.pos[2]];self.body.set_actor_location(vector(moved),False,False);follow=xyz(self.emitter.get_actor_location());fe=math.dist(follow,[self.expected[0],self.expected[1]+10,self.expected[2]]);self.body.set_actor_location(vector(self.pos),False,False);localcam=self.comp.get_world_transform().inverse_transform_location(vector(self.expected));f=320/math.tan(math.radians(self.comp.fov_angle)/2);pixel=[320+localcam.y/localcam.x*f,320-localcam.z/localcam.x*f];self.rec={'asset_id':r['asset_id'],'primary_part':r['primary_part'],'blueprint_class':v['ue']['generated_class_path'],'primary_world_cm':actual,'expected_world_cm':self.expected,'native_readback_error_cm':err,'attached_translation_follow_error_cm':fe,'native_pixel':pixel,'actor_yaw_degrees':yaw,'status':'pass'if err<.01 and fe<.01 and 0<=pixel[0]<640 and 0<=pixel[1]<640 else'fail','qualification':'native TargetPoint attachment and projection; anchor originated from native Habitat/original triangle annotation, no UE texture-colour test'};self.phase='capture';self.wait=12
 def capture(self):
  folder=T/'round3/images/lx_emitters_ue';folder.mkdir(parents=True,exist_ok=True);p=folder/(self.row['asset_id']+'.jpg');unreal.ImageWriteBlueprintLibrary.export_to_disk(self.rt,str(p),unreal.ImageWriteOptions(format=unreal.DesiredImageFormat.JPG,compression_quality=85,overwrite_file=True,async_=False));self.rec['image']=str(p);self.out.append(self.rec);write(T/'round3/evidence/lx_primary_ue_results.json',self.out);unreal.log('LX_PRIMARY '+self.row['asset_id']+' '+self.rec['status']);unreal.EditorLevelLibrary.destroy_actor(self.emitter);unreal.EditorLevelLibrary.destroy_actor(self.body);self.emitter=self.body=None;self.i+=1;self.phase='spawn';self.wait=2
 def tick(self,dt):
  try:
   if self.wait:self.wait-=1;return
   self.begin()if self.phase=='spawn'else self.capture()
  except Exception:write(T/'round3/evidence/lx_primary_ue_fatal.json',{'index':self.i,'error':traceback.format_exc()});self.finish()
 def finish(self):
  for a in [self.body,self.emitter]:
   if a:unreal.EditorLevelLibrary.destroy_actor(a)
  unreal.unregister_slate_post_tick_callback(self.handle);write(T/'round3/evidence/lx_primary_ue_results.json',self.out);unreal.SystemLibrary.quit_editor()
try:_job=Job()
except Exception:
 write(T/'round3/evidence/lx_primary_ue_fatal.json',{'error':traceback.format_exc()});unreal.SystemLibrary.quit_editor();raise
