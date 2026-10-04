"""UE-native image review and composite actor readback in the copied Kujiale map."""
import json,math,os,time,traceback
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT']);MAP='/Game/AVEngine/Optional/KujialeBakedLit/kujiale_0020_full_home_baked_lit_v1'
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def vector(v):return unreal.Vector(*v)
def xyz(v):return [v.x,v.y,v.z]
def glb_to_ue(p,yaw=90):
 th=math.radians(yaw);x,y,z=p[0]*100,p[2]*100,p[1]*100;return [math.cos(th)*x-math.sin(th)*y,math.sin(th)*x+math.cos(th)*y,z]
def boxes(actors):
 lo=[float('inf')]*3;hi=[float('-inf')]*3
 for actor in actors:
  center,ext=actor.get_actor_bounds(False)
  for i,(c,e)in enumerate(zip(xyz(center),xyz(ext))):lo[i]=min(lo[i],c-e);hi[i]=max(hi[i],c+e)
 return lo,hi
def spawn(parts,pos,yaw=90):
 actors=[]
 for path in parts:
  mesh=unreal.load_asset(path);actor=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor,vector(pos),unreal.Rotator(yaw=yaw,pitch=0,roll=0));component=actor.static_mesh_component;component.set_mobility(unreal.ComponentMobility.MOVABLE);component.set_static_mesh(mesh);actors.append(actor)
 return actors
def trace_floor(world,start,end,ignore):
 result=unreal.SystemLibrary.line_trace_single(world,vector(start),vector(end),unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,True,ignore,unreal.DrawDebugTrace.NONE,True)
 return result
def setup():
 if not unreal.EditorLevelLibrary.load_level(MAP):raise RuntimeError('Cannot load copied native Kujiale map')
 world=unreal.EditorLevelLibrary.get_editor_world();actors=unreal.EditorLevelLibrary.get_all_level_actors();survey=[]
 actor_rows=[]
 for actor in actors:
  components=actor.get_components_by_class(unreal.StaticMeshComponent)
  if components:actor.set_actor_enable_collision(True)
  actor_rows.append({'label':actor.get_actor_label(),'class':actor.get_class().get_name(),'static_mesh_components':len(components)})
  for component in components:
   mesh=component.get_editor_property('static_mesh')
   if mesh is None:continue
   component.set_collision_profile_name('BlockAll');component.set_collision_enabled(unreal.CollisionEnabled.QUERY_ONLY)
   box=mesh.get_bounding_box();transform=component.get_world_transform();points=[xyz(transform.transform_location(unreal.Vector(x,y,z)))for x in [box.min.x,box.max.x]for y in [box.min.y,box.max.y]for z in [box.min.z,box.max.z]];lo=[min(p[i]for p in points)for i in range(3)];hi=[max(p[i]for p in points)for i in range(3)]
   survey.append({'label':actor.get_actor_label()+'/'+component.get_name(),'center_cm':[(a+b)/2 for a,b in zip(lo,hi)],'extent_cm':[(b-a)/2 for a,b in zip(lo,hi)],'mesh':mesh.get_path_name(),'import_file':mesh.get_editor_property('asset_import_data').get_first_filename()})
 write(T/'evidence/ue_map_actor_survey.json',actor_rows)
 write(T/'evidence/ue_room_geometry.json',survey)
 # Select an actual floor mesh and then raycast a point on that floor.
 if not survey:raise RuntimeError('Native Kujiale map has no loaded mesh components')
 # This copied map has one full-home mesh, not separate floor actors.
 full=max(survey,key=lambda a:a['extent_cm'][0]*a['extent_cm'][1]);base=list(full['center_cm']);bottom=base[2]-full['extent_cm'][2]
 hits=[];diagnostics=[]
 for fx,fy in [(0,0),(-.3,0),(.3,0),(0,-.3),(0,.3),(-.3,-.3),(.3,.3),(-.5,.2),(.5,-.2)]:
  p=[base[0]+fx*full['extent_cm'][0],base[1]+fy*full['extent_cm'][1]]
  result=trace_floor(world,[p[0],p[1],bottom+260],[p[0],p[1],bottom-20],[])
  hit=result[-1] if isinstance(result,tuple)else result
  try:
   point=hit.get_editor_property('impact_point');normal=hit.get_editor_property('impact_normal');blocking=hit.get_editor_property('blocking_hit')
   diagnostics.append({'xy_cm':p,'blocking':blocking,'point_cm':xyz(point),'normal':xyz(normal)})
   if blocking and normal.z>.9:hits.append((point,p))
  except Exception:diagnostics.append({'xy_cm':p,'raw':str(result),'error':traceback.format_exc()})
 write(T/'evidence/ue_floor_ray_survey.json',diagnostics)
 plan=json.loads((T/'evidence/ue_source_floor_plan.json').read_text())
 native_lo=[base[i]-full['extent_cm'][i]for i in range(3)];native_hi=[base[i]+full['extent_cm'][i]for i in range(3)]
 bound_error=max(abs(a-b)for a,b in zip(native_lo+native_hi,plan['bounds_min_cm']+plan['bounds_max_cm']))
 if bound_error>.02:raise RuntimeError('Source/native complete-geometry bounds disagree: '+str(bound_error))
 location=plan['selected']['location_cm'];floor=full
 hit_info={'point_cm':location,'method':'actual composed source-USD triangle ray; full native mesh point bounds verified within 0.02cm; collision trace unavailable','native_bound_error_cm':bound_error,'source':plan['source'],'status':'pass_geometry','native_physics_collision_status':'unavailable'}
 eye=plan['selected']['camera_cm'];target=[location[0],location[1],location[2]+95];actor_yaw=plan['selected']['actor_yaw_deg'];direction=[(eye[i]-location[i])/330 for i in range(2)]
 cam=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D,vector(eye),unreal.MathLibrary.find_look_at_rotation(vector(eye),vector(target)))
 comp=cam.get_component_by_class(unreal.SceneCaptureComponent2D);comp.capture_every_frame=False;comp.capture_on_movement=False;comp.fov_angle=50
 pp=unreal.PostProcessSettings();pp.override_auto_exposure_method=True;pp.auto_exposure_method=unreal.AutoExposureMethod.AEM_MANUAL;pp.override_auto_exposure_bias=True;pp.auto_exposure_bias=-8;pp.override_auto_exposure_apply_physical_camera_exposure=True;pp.auto_exposure_apply_physical_camera_exposure=False;comp.post_process_settings=pp;comp.post_process_blend_weight=1
 rt=unreal.RenderingLibrary.create_render_target2d(world,640,640,unreal.TextureRenderTargetFormat.RTF_RGBA8,unreal.LinearColor(0,0,0,1),False);comp.texture_target=rt;comp.capture_source=unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
 # Keep the source map's lights, plus one explicit soft review light at the camera.
 light=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight,vector([location[0]+180*direction[0],location[1]+180*direction[1],location[2]+210]));lc=light.point_light_component;lc.set_mobility(unreal.ComponentMobility.MOVABLE);lc.set_editor_property('intensity_units',unreal.LightUnits.CANDELAS);lc.set_intensity(1500);lc.set_attenuation_radius(1000);lc.set_source_radius(80);lc.set_cast_shadows(True)
 info={'map':MAP,'object_position_cm':location,'camera_position_cm':eye,'floor_selection':floor,'floor_evidence':hit_info,'camera_target_cm':target,'resolution':[640,640],'fov_degrees':50,'actor_yaw_degrees':actor_yaw,'clear_review_volume':plan['selected'],'light':'native source lighting plus MOVABLE actor-side white soft point light, 1500cd, source radius 80cm','render_backend':'UE 5.5 Vulkan GPU1','exposure':'manual EV100=0, exposure compensation -8 EV; physical camera exposure disabled'}
 write(T/'evidence/ue_review_scene.json',info)
 return world,comp,rt,location,info

def native_material_overrides():
 imports=json.loads((T/'evidence/ue_import_results.json').read_text());skeletal=json.loads((T/'evidence/ue_skeletal_results.json').read_text());meshes={r['asset_id']:next((x['path']for x in r['assets']if x['class']=='SkeletalMesh'),None)for r in skeletal};rows=[]
 for entry in imports:
  aid=entry['asset_id']
  if aid not in meshes:continue
  rec={'asset_id':aid,'color':entry['color'],'skeletal_mesh':meshes[aid],'status':'fail','slots':[]}
  try:
   mesh=unreal.load_asset(meshes[aid]);materials={unreal.load_asset(x['path']).get_name():x['path']for x in entry['assets']if x['class']=='MaterialInstanceConstant'}
   for i,slot in enumerate(mesh.get_editor_property('materials')):
    name=str(slot.material_slot_name);source=slot.material_interface.get_name()if slot.material_interface else None;key=name if name in materials else source
    rec['slots'].append({'index':i,'slot_name':name,'original_material_name':source,'override_material':materials.get(key)})
   rec['status']='pass'if rec['slots']and all(x['override_material']for x in rec['slots'])else'fail'
  except Exception:rec['error']=traceback.format_exc()
  rows.append(rec)
 write(T/'evidence/ue_skeletal_instance_materials.json',rows)

def native_pose_readback():
 results=[]
 for p in sorted((T/'evidence/ue_skeletal').glob('*.json')):
  entry=json.loads(p.read_text());aid=entry['asset_id'];row={'asset_id':aid,'status':'not_run','actions':[]};bone=[x for x in entry['assets']if x['class']=='SkeletalMesh'];anims=[x for x in entry['assets']if x['class']=='AnimSequence']
  try:
   mesh=unreal.load_asset(bone[0]['path']);options=unreal.AnimPoseEvaluationOptions()
   if hasattr(options,'optional_skeletal_mesh'):options.optional_skeletal_mesh=mesh
   build=json.loads((T/'assets/rocketbox'/aid/'build.json').read_text());name=build['mouth_bone']
   for a in anims:
    seq=unreal.load_asset(a['path']);frames=[]
    length=seq.get_editor_property('sequence_length')
    for t in [0,length*.5,max(0,length-1/30)]:
     pose=unreal.AnimPoseExtensions.get_anim_pose_at_time(seq,t,options);names=unreal.AnimPoseExtensions.get_bone_names(pose);row['bone_names']=[str(n)for n in names];actual=next(str(n)for n in names if str(n).endswith('MUpperLip'));transform=unreal.AnimPoseExtensions.get_bone_pose(pose,actual,unreal.AnimPoseSpaces.WORLD)
     foot={}
     prefix=name.split()[0]
     for side in ['L','R']:
      foot_name=next(str(n)for n in names if str(n).lower().replace('_','').replace(' ','').replace('-','').endswith(side.lower()+'foot'));ft=unreal.AnimPoseExtensions.get_bone_pose(pose,foot_name,unreal.AnimPoseSpaces.WORLD);foot[side]=xyz(ft.translation)
     frames.append({'time_s':t,'mouth_cm':xyz(transform.translation),'bone_count':len(names),'ankles_cm':foot})
    row['actions'].append({'asset':a['path'],'sequence_length_s':length,'frames':frames})
   row['status']='pass' if len(row['actions'])>=3 and all(f['bone_count']>=80 and 30<f['mouth_cm'][2]<210 for a in row['actions']for f in a['frames'])else 'fail'
  except Exception:row['error']=traceback.format_exc()
  results.append(row)
 write(T/'evidence/ue_animation_readback.json',results)

class Review:
 def __init__(self):
  self.world,self.comp,self.rt,self.pos,self.scene=setup();self.comp.capture_every_frame=True
  self.depth_rt=unreal.RenderingLibrary.create_render_target2d(self.world,640,640,unreal.TextureRenderTargetFormat.RTF_RGBA16F,unreal.LinearColor(0,0,0,0),False)
  rows=json.loads((T/'evidence/ue_import_results.json').read_text());only=os.environ.get('V7_UE_RENDER_ONLY');limit=int(os.environ.get('V7_UE_RENDER_LIMIT','0'))
  self.items=[r for r in rows if r['status']=='pass'and(not only or any(token in r['asset_id']for token in only.split(',')))]
  if limit:self.items=self.items[:limit]
  self.index=0;self.results=[];self.actors=[];self.wait=60;self.phase='spawn';self.handle=unreal.register_slate_post_tick_callback(self.tick);self.started=time.time()
 def tick(self,delta):
  try:
   if self.wait:self.wait-=1;return
   if self.phase=='spawn':self.begin()
   elif self.phase=='rgb':self.export('rgb.png',self.rt);self.capture(True,False);self.phase='isolated';self.wait=4
   elif self.phase=='isolated':self.export('isolated.png',self.rt);self.capture(False,True);self.phase='depth';self.wait=4
   elif self.phase=='depth':self.export('depth.hdr',self.depth_rt);self.capture(True,True);self.phase='isolated_depth';self.wait=4
   elif self.phase=='isolated_depth':self.export('isolated_depth.hdr',self.depth_rt);self.complete();self.phase='spawn';self.wait=2
   if time.time()-self.started>3600:raise RuntimeError('Native UE capture run exceeded one-hour watchdog')
  except Exception:
   write(T/'evidence/ue_render_fatal.json',{'error':traceback.format_exc(),'phase':self.phase,'index':self.index})
   self.finish()
 def capture(self,only,depth):
  self.comp.clear_show_only_components()
  if only:
   for actor in self.actors:self.comp.show_only_actor_components(actor)
  self.comp.primitive_render_mode=unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST if only else unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES
  self.comp.texture_target=self.depth_rt if depth else self.rt
  self.comp.capture_source=unreal.SceneCaptureSource.SCS_SCENE_DEPTH if depth else unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
 def export(self,name,rt):
  unreal.RenderingLibrary.export_render_target(self.world,rt,str(self.dest),name)
 def begin(self):
  if self.index==len(self.items):self.finish();return
  row=self.items[self.index];aid=row['asset_id'];color=row['color'];self.dest=T/'renders/ue'/aid/color;self.dest.mkdir(parents=True,exist_ok=True);self.proof=self.dest/'native.json'
  if self.proof.exists():
   saved=json.loads(self.proof.read_text())
   if saved.get('status')=='pass' and saved.get('capture_qa_status')=='pass' and saved.get('qa_revision')=='native_basis_tick_v7':self.results.append(saved);self.index+=1;return
  self.rec={'asset_id':aid,'color':color,**self.scene,'status':'fail','qa_revision':'native_basis_tick_v7','ue_texture_color_comparison':'not_used'}
  parts=[r['path']for r in row['assets']if r['class']=='StaticMesh'];self.actors=spawn(parts,self.pos,self.scene['actor_yaw_degrees']);lo,hi=boxes(self.actors)
  meta=json.loads((T/'assets/runtime'/aid/color/'poses.json').read_text());mouth=meta['poses']['stand']['mouth_m'];ue_mouth=[a+b for a,b in zip(self.pos,glb_to_ue(mouth,self.scene['actor_yaw_degrees']))]
  self.rec.update(mesh_paths=parts,bounds_world_cm={'min':lo,'max':hi},height_m=(hi[2]-lo[2])/100,floor_delta_m=(lo[2]-self.pos[2])/100,mouth_world_cm=ue_mouth,mouth_status='native_composite_transform_pending_visual_review',rgb=str((self.dest/'rgb.png').relative_to(T)),isolated=str((self.dest/'isolated.png').relative_to(T)),status='pass' if abs(lo[2]-self.pos[2])<1.5 and(140<hi[2]-lo[2]<215 or aid.startswith('v7_lx_'))else'fail')
  self.capture(False,False);self.phase='rgb';self.wait=20
 def complete(self):
  write(self.proof,self.rec);self.results.append(self.rec);unreal.log('V7_UE_RENDER '+self.rec['asset_id']+' '+self.rec['color']+' '+self.rec['status']);self.index+=1
  for actor in self.actors:unreal.EditorLevelLibrary.destroy_actor(actor)
  self.actors=[];write(T/'evidence'/os.environ.get('V7_UE_RENDER_RESULT','ue_native_results.json'),self.results)
 def finish(self):
  if self.actors:
   for actor in self.actors:unreal.EditorLevelLibrary.destroy_actor(actor)
  self.actors=[]
  if hasattr(self,'handle'):unreal.unregister_slate_post_tick_callback(self.handle)
  write(T/'evidence'/os.environ.get('V7_UE_RENDER_RESULT','ue_native_results.json'),self.results)
  if self.index==len(self.items)and os.environ.get('V7_UE_RENDER_ANIM','0')=='1':native_pose_readback();native_material_overrides()
  unreal.SystemLibrary.quit_editor()
def main():
 if os.environ.get('V7_UE_REIMPORT_STATIC','0')=='1':
  os.environ['V7_UE_EDITOR_EXIT']='0';path=T/'worktree/tools/assets/v7_human_roster_ue_import.py';exec(compile(path.read_text(),str(path),'exec'),{'__name__':'v7_import_for_review'})
 global _job
 _job=Review()
try:main()
except Exception:write(T/'evidence/ue_render_fatal.json',{'error':traceback.format_exc()});unreal.SystemLibrary.quit_editor();raise
