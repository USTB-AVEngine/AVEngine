"""Import both generated birds and capture their native one-actor Kujiale placements."""
import os,json,time,traceback,math
from pathlib import Path
import unreal
T=Path(os.environ["V7_PET_BIRDS_TASK_ROOT"]);W=Path(os.environ["V7_PET_BIRDS_WORKTREE"])
os.environ["V7_HUMAN_TASK_ROOT"]=str(T)
base=W/"tools/assets/v7_human_roster_ue_review.py"
exec(compile(base.read_text().split("class Review:")[0],str(base),"exec"),globals())
base=W/"tools/assets/v7_human_roster_ue_blueprints.py"
exec(compile(base.read_text().split("def run():")[0],str(base),"exec"),globals())
def U(p):return [-100*p[2],100*p[0],100*p[1]]
class Capture:
    def __init__(self):
        unreal.EditorPythonScripting.set_keep_python_script_alive(True)
        plans=json.loads((T/"evidence/ue_source_geometry.json").read_text());self.items=[];imports=[]
        for kind in ["canary","budgerigar"]:
            meta=json.loads((T/"assets"/kind/"asset.json").read_text());plan=next(x for x in plans if x["kind"]==kind)
            if plan["status"]!="pass":raise RuntimeError("Unqualified source geometry: "+kind)
            aid=meta["asset_id"];folder="/Game/V7Hy3DPetBirds20261012/"+aid;task=unreal.AssetImportTask();task.filename=str(T/"assets"/kind/"runtime.glb");task.destination_path=folder;task.automated=True;task.save=True;task.async_=True;task.replace_existing=True
            unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task]);assets=[{"path":o.get_path_name(),"name":o.get_name(),"class":o.get_class().get_name()} for o in task.get_objects()]
            parts=[x["path"] for x in assets if x["class"]=="StaticMesh"]
            if len(parts)<3:raise RuntimeError("Bird/perch component mesh missing: "+kind)
            name="BP_"+aid;bp=unreal.load_asset(folder+"/"+name)
            if bp is None:
                factory=unreal.BlueprintFactory();factory.set_editor_property("parent_class",unreal.Actor);bp=unreal.AssetToolsHelpers.get_asset_tools().create_asset(name,folder,unreal.Blueprint,factory);sub=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);parent=sub.k2_gather_subobject_data_for_blueprint(bp)[-1]
                for i,path in enumerate(parts):
                    h,c=add(sub,bp,parent,unreal.StaticMeshComponent,"BirdOrSilentPerch_"+str(i));c.set_static_mesh(unreal.load_asset(path));c.set_relative_rotation(unreal.Rotator(yaw=90),False,True);c.set_mobility(unreal.ComponentMobility.MOVABLE)
                h,c=add(sub,bp,parent,unreal.SceneComponent,"FOA_beak");c.set_relative_location(vector(U(meta["emit_point_local_m"])),False,True)
            unreal.BlueprintEditorLibrary.compile_blueprint(bp);unreal.EditorAssetLibrary.save_loaded_asset(bp,False);unreal.EditorAssetLibrary.save_directory(folder,only_if_is_dirty=True,recursive=True)
            imports.append({"kind":kind,"asset_id":aid,"status":"pass","source_glb":task.filename,"assets":assets,"blueprint_path":bp.get_path_name(),"generated_class_path":bp.generated_class().get_path_name(),"logical_actor_count":1,"movement":"static"})
            self.items.append((kind,meta,plan,bp))
        write(T/"evidence/ue_import_results.json",imports)
        self.world,self.comp,self.rt,location,self.scene=setup();self.scene["render_backend"]="UE 5.5 Vulkan physical GPU3, isolated /dev/nvidia3, graphicsadapter0"
        self.scene["setup_reference"]={key:self.scene.pop(key) for key in ("camera_position_cm","camera_target_cm","actor_yaw_degrees")}
        self.scene["viewpoint_scope"]="actual camera and actor transforms are read back in each asset/image record; setup_reference describes the old shared review setup"
        self.comp.capture_every_frame=True;self.results=[];self.item_index=0;self.camera_index=0;self.actor=None;self.phase="spawn";self.wait=40;self.started=time.time();self.warm_until=time.time()+30
        self.light=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight,vector([0,0,0]));lc=self.light.point_light_component;lc.set_mobility(unreal.ComponentMobility.MOVABLE);lc.set_editor_property("intensity_units",unreal.LightUnits.CANDELAS);lc.set_attenuation_radius(600);lc.set_source_radius(60);lc.set_cast_shadows(False);self.handle=unreal.register_slate_post_tick_callback(self.tick)
    def begin(self):
        if self.item_index==len(self.items):self.finish();return
        self.kind,self.meta,self.plan,bp=self.items[self.item_index];self.root=U(self.plan["root_m"]);self.actor=unreal.EditorLevelLibrary.spawn_actor_from_class(bp.generated_class(),vector(self.root),unreal.Rotator())
        child=next(c for c in self.actor.get_components_by_class(unreal.SceneComponent) if c.get_name().startswith("FOA_beak"));actual=xyz(child.get_world_location());expected=[self.root[i]+U(self.meta["emit_point_local_m"])[i] for i in range(3)]
        self.actor.set_actor_location(vector([self.root[0]+10,self.root[1],self.root[2]]),False,False);moved=xyz(child.get_world_location());follow=max(abs(moved[i]-actual[i]-(10 if i==0 else 0)) for i in range(3));self.actor.set_actor_location(vector(self.root),False,False)
        points=[]
        for c in self.actor.get_components_by_class(unreal.StaticMeshComponent):
            box=c.get_editor_property("static_mesh").get_bounding_box();tr=c.get_world_transform();points.extend(xyz(tr.transform_location(unreal.Vector(x,y,z))) for x in [box.min.x,box.max.x] for y in [box.min.y,box.max.y] for z in [box.min.z,box.max.z])
        lo=[min(p[i] for p in points) for i in range(3)];hi=[max(p[i] for p in points) for i in range(3)]
        self.rec={"kind":self.kind,"asset_id":self.meta["asset_id"],"root_cm":self.root,"actor_rotation_degrees":{"pitch":self.actor.get_actor_rotation().pitch,"yaw":self.actor.get_actor_rotation().yaw,"roll":self.actor.get_actor_rotation().roll},"beak_world_cm":actual,"expected_beak_world_cm":expected,"emit_readback_error_cm":max(abs(x-y) for x,y in zip(actual,expected)),"translation_follow_error_cm":follow,"bounds_min_cm":lo,"bounds_max_cm":hi,"height_m":(hi[2]-lo[2])/100,"expected_height_m":self.meta["dimensions_m"][1],"base_floor_delta_cm":lo[2]-self.root[2],"component_count":len(self.actor.get_components_by_class(unreal.StaticMeshComponent)),"logical_actor_count":1,"native_scene":self.scene,"native_ue_physics":"not_claimed; source-triangle BVH evidence separate","native_scene_warmup_seconds":30,"images":[],"light_policy":"same inverse-square normalized illumination as existing 1.35m parrot review: 1200cd*(distance/1.35)^2; fixed before QA"}
        self.camera_index=0;self.phase="camera";self.wait=15
    def export(self,name):
        p=T/"assets"/self.kind/"screenshots/ue"/str(self.camera_index)/(name+".jpg");p.parent.mkdir(parents=True,exist_ok=True);unreal.ImageWriteBlueprintLibrary.export_to_disk(self.rt,str(p),unreal.ImageWriteOptions(format=unreal.DesiredImageFormat.JPG,compression_quality=90,overwrite_file=True,async_=False));return str(p)
    def tick(self,delta):
        try:
            if time.time()<self.warm_until:return
            if self.wait:self.wait-=1;return
            if self.phase=="spawn":self.begin()
            elif self.phase=="camera":
                if self.camera_index==len(self.plan["cameras"]):self.complete();return
                cam=self.plan["cameras"][self.camera_index];eye=U(cam["eye_glb_m"]);target=U(cam["target_glb_m"]);owner=self.comp.get_owner();owner.set_actor_location(vector(eye),False,False);owner.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(vector(eye),vector(target)),False);self.comp.fov_angle=cam["fov_degrees"];self.light.set_actor_location(vector([eye[0],eye[1],eye[2]+40]),False,False);self.light.point_light_component.set_intensity(1200*(cam["distance_m"]/1.35)**2)
                self.comp.clear_show_only_components();self.comp.primitive_render_mode=unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES;self.image={"camera":cam,"ue_camera_position_cm":xyz(owner.get_actor_location()),"ue_camera_target_cm":target,"ue_camera_rotation_degrees":{"pitch":owner.get_actor_rotation().pitch,"yaw":owner.get_actor_rotation().yaw,"roll":owner.get_actor_rotation().roll},"native_fov_degrees":self.comp.fov_angle,"camera_native_readback_error_cm":max(abs(x-y) for x,y in zip(xyz(owner.get_actor_location()),eye))};self.phase="rgb";self.wait=10
            elif self.phase=="rgb":
                self.image["rgb"]=self.export("rgb");self.comp.clear_show_only_components();self.comp.show_only_actor_components(self.actor);self.comp.primitive_render_mode=unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST;self.phase="isolated";self.wait=5
            elif self.phase=="isolated":
                self.image["isolated"]=self.export("isolated");self.rec["images"].append(self.image);self.camera_index+=1;self.phase="camera";self.wait=3
            if time.time()-self.started>900:raise RuntimeError("Owned native UE capture deadline")
        except Exception:
            write(T/"evidence/ue_fatal.json",{"error":traceback.format_exc(),"phase":self.phase,"kind":getattr(self,"kind",None)});self.finish(failed=True)
    def complete(self):
        r=self.rec;r["status"]="pass_capture" if r["emit_readback_error_cm"]<.01 and r["translation_follow_error_cm"]<.01 and abs(r["base_floor_delta_cm"])<.2 and abs(r["height_m"]-r["expected_height_m"])<.002 and len(r["images"])==2 and all(abs(x)<.001 for x in r["actor_rotation_degrees"].values()) and all(x["camera_native_readback_error_cm"]<.001 for x in r["images"]) else "fail";self.results.append(r);write(T/"evidence/ue_capture_progress.json",self.results)
        unreal.EditorLevelLibrary.destroy_actor(self.actor);self.actor=None;self.item_index+=1;self.phase="spawn";self.wait=3
    def finish(self,failed=False):
        if self.actor:unreal.EditorLevelLibrary.destroy_actor(self.actor)
        if hasattr(self,"handle"):unreal.unregister_slate_post_tick_callback(self.handle)
        write(T/"evidence/ue_capture_results.json",{"status":"fail" if failed else "pass_capture" if len(self.results)==2 and all(r["status"]=="pass_capture" for r in self.results) else "fail","assets":self.results})
        unreal.SystemLibrary.quit_editor()
try:_capture=Capture()
except Exception:
    write(T/"evidence/ue_fatal.json",{"error":traceback.format_exc(),"phase":"init"});unreal.SystemLibrary.quit_editor();raise
