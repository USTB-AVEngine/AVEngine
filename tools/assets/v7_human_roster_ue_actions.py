"""Capture source-matched skeletal poses in native UE, with dynamic FOA readback."""
import json,math,os,time,traceback
from pathlib import Path
import unreal

T=Path(os.environ['V7_HUMAN_TASK_ROOT'])
helper=T/'worktree/tools/assets/v7_human_roster_ue_review.py'
source=helper.read_text().split('\ntry:main()',1)[0]
source=source.replace('ue_render_fatal.json','ue_action_render_fatal.json')
scope={'__name__':'v7_review_helpers'}
exec(compile(source,str(helper),'exec'),scope)
Base=scope['Review']
write=scope['write']
xyz=scope['xyz']
vector=scope['vector']
glb_to_ue=scope['glb_to_ue']

class SkeletalReview(Base):
 def __init__(self):
  self.world,self.comp,self.rt,self.pos,self.scene=scope['setup']()
  self.comp.capture_every_frame=True
  self.depth_rt=unreal.RenderingLibrary.create_render_target2d(self.world,640,640,unreal.TextureRenderTargetFormat.RTF_RGBA16F,unreal.LinearColor(0,0,0,0),False)
  self.skeletons={r['asset_id']:r for r in json.loads((T/'evidence/ue_skeletal_root_results.json').read_text())}
  self.materials={(r['asset_id'],r['color']):r for r in json.loads((T/'evidence/ue_skeletal_instance_materials.json').read_text())}
  self.bones={r['asset_id']:r['bone_names']for r in json.loads((T/'evidence/ue_animation_root_readback.json').read_text())}
  self.items=json.loads((T/'registrations/habitat_action_review_cases.json').read_text())
  only=os.environ.get('V7_UE_RENDER_ONLY')
  if only:self.items=[r for r in self.items if only in r['asset_id']]
  limit=int(os.environ.get('V7_UE_RENDER_LIMIT','0'))
  if limit:self.items=self.items[:limit]
  self.index=0;self.results=[];self.actors=[];self.wait=60;self.phase='spawn';self.started=time.time()
  self.handle=unreal.register_slate_post_tick_callback(self.tick)
 def begin(self):
  if self.index==len(self.items):self.finish();return
  row=self.items[self.index];aid=row['asset_id'];label=row['color'];family=row['review_family']
  self.dest=T/'renders/ue_actions'/aid/label;self.dest.mkdir(parents=True,exist_ok=True);self.proof=self.dest/'native.json'
  if self.proof.exists():
   saved=json.loads(self.proof.read_text())
   if saved.get('status')=='pass'and saved.get('capture_qa_status')=='pass'and saved.get('qa_revision')=='skeletal_native_pose_v1':
    self.results.append(saved);self.index+=1;return
  ev=row['poses']['stand'];entry=self.skeletons[aid]
  mesh_path=next(x['path']for x in entry['assets']if x['class']=='SkeletalMesh')
  clip_name='Walking'if family=='walk'else'Sitting_Idle'
  clip_path=next(x['path']for x in entry['assets']if x['class']=='AnimSequence'and x['path'].split('/')[-1].split('.')[0]==clip_name)
  yaw=self.scene['actor_yaw_degrees']-180
  actor=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkeletalMeshActor,vector(self.pos),unreal.Rotator(yaw=yaw,pitch=0,roll=0))
  self.actors=[actor];self.component=actor.skeletal_mesh_component
  comp=self.component;comp.set_mobility(unreal.ComponentMobility.MOVABLE)
  comp.set_skeletal_mesh(unreal.load_asset(mesh_path))
  for slot in self.materials[(aid,row['controlled_color_source'])]['slots']:comp.set_material(slot['index'],unreal.load_asset(slot['override_material']))
  comp.set_update_animation_in_editor(True)
  comp.play_animation(unreal.load_asset(clip_path),False);comp.set_play_rate(0)
  comp.set_position(ev['time_s'],False)
  actor.set_actor_location(vector([self.pos[0],self.pos[1],self.pos[2]+100*ev['ground_correction_m']]),False,False)
  self.mouth_bone=next(n for n in self.bones[aid]if n.endswith('MUpperLip'))
  expected=[a+b for a,b in zip(self.pos,glb_to_ue(ev['mouth_m'],self.scene['actor_yaw_degrees']))]
  self.rec={'asset_id':aid,'color':label,'source_color':row['controlled_color_source'],'motion_family':family,'frame_index':row['review_frame'],'time_s':ev['time_s'],**self.scene,'capture_directory':str(self.dest.relative_to(T)),'skeletal_mesh':mesh_path,'animation':clip_path,'skeletal_actor_yaw_degrees':yaw,'ground_correction_m':ev['ground_correction_m'],'source_grounded_floor_delta_m':ev['floor_y_m'],'mouth_expected_world_cm':expected,'mouth_world_cm':expected,'mouth_bone':self.mouth_bone,'mouth_status':'native_skeletal_bone_readback','rgb':str((self.dest/'rgb.png').relative_to(T)),'isolated':str((self.dest/'isolated.png').relative_to(T)),'qa_revision':'skeletal_native_pose_v1','ue_texture_color_comparison':'not_used','status':'not_run','sit_support':'requires_matched_seat'if family=='sit'else'not_applicable'}
  self.capture(False,False);self.phase='rgb';self.wait=30
 def complete(self):
  actual=xyz(self.component.get_socket_location(self.mouth_bone));expected=self.rec['mouth_expected_world_cm']
  error=math.sqrt(sum((a-b)**2 for a,b in zip(actual,expected)))
  self.rec.update(mouth_world_cm=actual,mouth_source_error_cm=error,status='pass'if error<.3 and abs(self.rec['source_grounded_floor_delta_m'])<1e-6 else'fail')
  Base.complete(self)

try:
 _job=SkeletalReview()
except Exception:
 write(T/'evidence/ue_action_render_fatal.json',{'error':traceback.format_exc()})
 unreal.SystemLibrary.quit_editor()
 raise
