"""Import and capture the licensed static parrot as one native UE Blueprint actor."""
import os,json,math,time,traceback
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT']);R=T/'round4'
unreal.EditorPythonScripting.set_keep_python_script_alive(True)
base=T/'worktree/tools/assets/v7_human_roster_ue_review.py'
exec(compile(base.read_text().split('class Review:')[0],str(base),'exec'),globals())
bpcode=T/'worktree/tools/assets/v7_human_roster_ue_blueprints.py'
exec(compile(bpcode.read_text().split('def run():')[0],str(bpcode),'exec'),globals())
def write(p,d):
 p=Path(p)
 if p.parent==T/'evidence':p=R/'evidence/ue_setup'/p.name
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def U(p):return [-100*p[2],100*p[0],100*p[1]]
class Capture:
 def __init__(self):
  self.meta=json.loads((R/'assets/parrot/source.json').read_text());self.plan=json.loads((R/'evidence/parrot_source_geometry.json').read_text())
  if self.plan['status']!='pass':raise RuntimeError('Source geometry/cameras not admitted')
  aid=self.meta['asset_id'];folder='/Game/V7RosterPetBirdsCCBY/'+aid
  task=unreal.AssetImportTask();task.filename=str(R/'assets/parrot/runtime.glb');task.destination_path=folder;task.automated=True;task.save=True;task.async_=True;task.replace_existing=True
  unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task]);assets=[{'path':o.get_path_name(),'name':o.get_name(),'class':o.get_class().get_name()}for o in task.get_objects()]
  parts=[a['path']for a in assets if a['class']=='StaticMesh']
  if len(parts)<3:raise RuntimeError('Static bird/perch parts missing')
  name='BP_'+aid;bp=unreal.load_asset(folder+'/'+name)
  if bp is None:
   factory=unreal.BlueprintFactory();factory.set_editor_property('parent_class',unreal.Actor);bp=unreal.AssetToolsHelpers.get_asset_tools().create_asset(name,folder,unreal.Blueprint,factory);sub=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);parent=sub.k2_gather_subobject_data_for_blueprint(bp)[-1]
   for i,path in enumerate(parts):
    h,c=add(sub,bp,parent,unreal.StaticMeshComponent,'BirdOrSilentPerch_'+str(i));c.set_static_mesh(unreal.load_asset(path));c.set_relative_rotation(unreal.Rotator(yaw=90),False,True);c.set_mobility(unreal.ComponentMobility.MOVABLE)
   h,c=add(sub,bp,parent,unreal.SceneComponent,'FOA_beak');c.set_relative_location(vector(U(self.meta['beak_local_glb_m'])),False,True)
  unreal.BlueprintEditorLibrary.compile_blueprint(bp);unreal.EditorAssetLibrary.save_loaded_asset(bp,False);unreal.EditorAssetLibrary.save_directory(folder,only_if_is_dirty=True,recursive=True)
  write(R/'evidence/ue_parrot_import.json',{'status':'pass','asset_id':aid,'source_glb':str(R/'assets/parrot/runtime.glb'),'assets':assets,'blueprint_path':bp.get_path_name(),'generated_class_path':bp.generated_class().get_path_name(),'source_license':'CC-BY-4.0','movement':'static','logical_actor_count':1})
  self.world,self.comp,self.rt,self.location,self.scene=setup();self.root_cm=U(self.plan['root_m']);self.actor=unreal.EditorLevelLibrary.spawn_actor_from_class(bp.generated_class(),vector(self.root_cm),unreal.Rotator())
  anchor=next(c for c in self.actor.get_components_by_class(unreal.SceneComponent)if c.get_name().startswith('FOA_beak'));actual=xyz(anchor.get_world_location());expected=[self.root_cm[i]+U(self.meta['beak_local_glb_m'])[i]for i in range(3)]
  self.actor.set_actor_location(vector([self.root_cm[0]+10,self.root_cm[1],self.root_cm[2]]),False,False);moved=xyz(anchor.get_world_location());follow_error=max(abs(moved[i]-actual[i]-(10 if i==0 else 0))for i in range(3));self.actor.set_actor_location(vector(self.root_cm),False,False)
  bounds_points=[]
  for c in self.actor.get_components_by_class(unreal.StaticMeshComponent):
   box=c.get_editor_property('static_mesh').get_bounding_box();transform=c.get_world_transform();bounds_points.extend(xyz(transform.transform_location(unreal.Vector(x,y,z)))for x in [box.min.x,box.max.x]for y in [box.min.y,box.max.y]for z in [box.min.z,box.max.z])
  lo=[min(p[i]for p in bounds_points)for i in range(3)];hi=[max(p[i]for p in bounds_points)for i in range(3)]
  self.rec={'status':'pending','asset_id':aid,'root_cm':self.root_cm,'beak_world_cm':actual,'expected_beak_world_cm':expected,'maximum_beak_readback_error_cm':max(abs(a-b)for a,b in zip(actual,expected)),'attached_translation_follow_error_cm':follow_error,'component_count':len(self.actor.get_components_by_class(unreal.StaticMeshComponent)),'bounds_min_cm':lo,'bounds_max_cm':hi,'height_m':(hi[2]-lo[2])/100,'base_floor_delta_cm':lo[2]-self.root_cm[2],'logical_actor_count':1,'native_bounds_method':'only real static mesh components; excludes editor-only default-root billboard bounds','actor_yaw_degrees':0,'component_yaw_degrees':90,'native_scene':self.scene,'native_ue_physics':'not_claimed; complete source geometry receipt separate','ue_texture_color_difference':'not_used','images':[]}
  self.comp.capture_every_frame=True;self.index=0;self.wait=35;self.phase='camera';self.started=time.time();self.light=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight,vector(self.root_cm));lc=self.light.point_light_component;lc.set_mobility(unreal.ComponentMobility.MOVABLE);lc.set_editor_property('intensity_units',unreal.LightUnits.CANDELAS);lc.set_intensity(1200);lc.set_attenuation_radius(600);lc.set_source_radius(60);lc.set_cast_shadows(False);self.handle=unreal.register_slate_post_tick_callback(self.tick)
 def export(self,name):
  p=R/'images/ue_parrot'/str(self.index)/(name+'.jpg');p.parent.mkdir(parents=True,exist_ok=True);unreal.ImageWriteBlueprintLibrary.export_to_disk(self.rt,str(p),unreal.ImageWriteOptions(format=unreal.DesiredImageFormat.JPG,compression_quality=85,overwrite_file=True,async_=False));return str(p)
 def tick(self,delta):
  try:
   if self.wait:self.wait-=1;return
   if self.phase=='camera':
    if self.index==len(self.plan['cameras']):self.finish();return
    cam=self.plan['cameras'][self.index];owner=self.comp.get_owner();eye=U(cam['eye_glb_m']);target=U(cam['target_glb_m']);owner.set_actor_location(vector(eye),False,False);owner.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(vector(eye),vector(target)),False);self.comp.fov_angle=cam['fov_degrees'];self.light.set_actor_location(vector([eye[0],eye[1],eye[2]+40]),False,False);self.comp.clear_show_only_components();self.comp.primitive_render_mode=unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES;self.image={'camera':cam};self.phase='rgb';self.wait=8
   elif self.phase=='rgb':
    self.image['rgb']=self.export('rgb');self.comp.clear_show_only_components();self.comp.show_only_actor_components(self.actor);self.comp.primitive_render_mode=unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST;self.phase='isolated';self.wait=4
   elif self.phase=='isolated':
    self.image['isolated']=self.export('isolated');self.rec['images'].append(self.image);self.index+=1;self.phase='camera';self.wait=2
   if time.time()-self.started>600:raise RuntimeError('Parrot native capture deadline')
  except Exception:
   write(R/'evidence/ue_parrot_fatal.json',{'error':traceback.format_exc(),'phase':self.phase});self.finish(failed=True)
 def finish(self,failed=False):
  self.rec['status']='fail'if failed else'pass_capture'if self.rec['maximum_beak_readback_error_cm']<.01 and self.rec['attached_translation_follow_error_cm']<.01 and abs(self.rec['base_floor_delta_cm'])<.2 and abs(self.rec['height_m']-.85)<.005 and len(self.rec['images'])==2 else'fail';write(R/'evidence/ue_parrot_capture.json',self.rec);unreal.EditorLevelLibrary.destroy_actor(self.actor);unreal.unregister_slate_post_tick_callback(self.handle);unreal.SystemLibrary.quit_editor()
try:_capture=Capture()
except Exception:
 write(R/'evidence/ue_parrot_fatal.json',{'error':traceback.format_exc(),'phase':'init'});unreal.SystemLibrary.quit_editor();raise
