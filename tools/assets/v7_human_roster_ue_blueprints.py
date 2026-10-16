"""Build one-actor static human Blueprints and audit their native component transforms."""
import os,json,traceback
from pathlib import Path
import unreal
T=Path(os.environ['V7_HUMAN_TASK_ROOT'])
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def xyz(v):return [v.x,v.y,v.z]
def add(sub,bp,parent,cls,name):
 h,fail=sub.add_new_subobject(unreal.AddNewSubobjectParams(parent_handle=parent,new_class=cls,blueprint_context=bp))
 if str(fail):raise RuntimeError(str(fail))
 sub.rename_subobject(h,unreal.Text(name));data=unreal.SubobjectDataBlueprintFunctionLibrary.get_data(h);obj=unreal.SubobjectDataBlueprintFunctionLibrary.get_object_for_blueprint(data,bp);return h,obj
def run():
 results=[];sub=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);tools=unreal.AssetToolsHelpers.get_asset_tools();rows=json.loads((T/'evidence/ue_import_results.json').read_text())
 for row in rows:
  aid=row['asset_id'];color=row['color'];rec={'asset_id':aid,'color':color,'status':'fail'};actor=None;dest=T/'evidence/ue_blueprints'/aid/(color+'.json')
  if dest.exists()and json.loads(dest.read_text()).get('status')=='pass':results.append(json.loads(dest.read_text()));continue
  try:
   name='BP_'+aid+'_'+color;folder='/Game/V7HumanActors/'+aid+'/'+color;path=folder+'/'+name;bp=unreal.load_asset(path)
   if bp is None:
    factory=unreal.BlueprintFactory();factory.set_editor_property('parent_class',unreal.Actor);bp=tools.create_asset(name,folder,unreal.Blueprint,factory);handles=sub.k2_gather_subobject_data_for_blueprint(bp);parent=handles[-1]
    parts=[x['path']for x in row['assets']if x['class']=='StaticMesh']
    for i,mesh in enumerate(parts):
     h,c=add(sub,bp,parent,unreal.StaticMeshComponent,'HumanPart_'+str(i));c.set_static_mesh(unreal.load_asset(mesh));c.set_relative_rotation(unreal.Rotator(yaw=90,pitch=0,roll=0),False,True);c.set_mobility(unreal.ComponentMobility.MOVABLE)
    meta=json.loads((T/'assets/runtime'/aid/color/'poses.json').read_text());p=meta['poses']['stand']['mouth_m'];mouth=[-p[2]*100,p[0]*100,p[1]*100];h,c=add(sub,bp,parent,unreal.SceneComponent,'FOA_mouth');c.set_relative_location(unreal.Vector(*mouth),False,True)
   unreal.BlueprintEditorLibrary.compile_blueprint(bp);unreal.EditorAssetLibrary.save_loaded_asset(bp,False)
   cls=bp.generated_class();actor=unreal.EditorLevelLibrary.spawn_actor_from_class(cls,unreal.Vector(0,0,0),unreal.Rotator());components=actor.get_components_by_class(unreal.StaticMeshComponent);anchors=[c for c in actor.get_components_by_class(unreal.SceneComponent)if c.get_name().startswith('FOA_mouth')];points=[]
   for component in components:
    box=component.get_editor_property('static_mesh').get_bounding_box();tr=component.get_world_transform();points.extend(xyz(tr.transform_location(unreal.Vector(x,y,z)))for x in [box.min.x,box.max.x]for y in [box.min.y,box.max.y]for z in [box.min.z,box.max.z])
   minz=min(p[2]for p in points);maxz=max(p[2]for p in points);meta=json.loads((T/'assets/runtime'/aid/color/'poses.json').read_text());p=meta['poses']['stand']['mouth_m'];expected=[-p[2]*100,p[0]*100,p[1]*100];actual=xyz(anchors[0].get_world_location());error=max(abs(a-b)for a,b in zip(expected,actual))
   rec.update(blueprint_path=bp.get_path_name(),generated_class_path=cls.get_path_name(),static_mesh_component_count=len(components),expected_mesh_part_count=sum(x['class']=='StaticMesh'for x in row['assets']),mouth_component=anchors[0].get_name(),mouth_local_ue_cm=actual,mouth_error_cm=error,floor_delta_cm=minz,height_m=(maxz-minz)/100,logical_actor_count=1,status='pass'if error<.01 and abs(minz)<1.5 and len(components)==sum(x['class']=='StaticMesh'for x in row['assets'])else'fail')
  except Exception:rec['error']=traceback.format_exc()
  finally:
   if actor:unreal.EditorLevelLibrary.destroy_actor(actor)
  write(dest,rec);results.append(rec);unreal.log('V7_HUMAN_BLUEPRINT '+aid+' '+color+' '+rec['status'])
 write(T/'evidence/ue_blueprint_results.json',results)
def pose_probe():
 entry=next(r for r in json.loads((T/'evidence/ue_skeletal_results.json').read_text())if r['asset_id']=='v7_female_adult_01');mesh=unreal.load_asset(next(x['path']for x in entry['assets']if x['class']=='SkeletalMesh'));out=[]
 for a in [x for x in entry['assets']if x['class']=='AnimSequence']:
  seq=unreal.load_asset(a['path']);t=seq.get_editor_property('sequence_length')*.5
  for raw in [False,True]:
   for retarget in [False,True]:
    op=unreal.AnimPoseEvaluationOptions();op.optional_skeletal_mesh=mesh;op.should_retarget=retarget;op.incorporate_root_motion_into_pose=True
    if raw:op.evaluation_type=unreal.AnimDataEvalType.RAW
    pose=unreal.AnimPoseExtensions.get_anim_pose_at_time(seq,t,op);names=unreal.AnimPoseExtensions.get_bone_names(pose);mouth=next(str(n)for n in names if str(n).endswith('MUpperLip'));foot=next(str(n)for n in names if str(n).endswith('L-Foot'))
    out.append({'asset':a['path'],'time_s':t,'raw':raw,'retarget':retarget,'mouth_cm':xyz(unreal.AnimPoseExtensions.get_bone_pose(pose,mouth,unreal.AnimPoseSpaces.WORLD).translation),'L':xyz(unreal.AnimPoseExtensions.get_bone_pose(pose,foot,unreal.AnimPoseSpaces.WORLD).translation)})
 write(T/'evidence/ue_animation_options_probe.json',out)
try:run()
except Exception:write(T/'evidence/ue_blueprint_fatal.json',{'error':traceback.format_exc()});raise
