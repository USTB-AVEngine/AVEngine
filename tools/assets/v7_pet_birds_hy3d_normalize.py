"""Scale the reviewed generated bird and repair only its wooden stand for floor placement."""
import argparse,json,math,shutil,sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from avengine.assets.source_materials import prepare_source_glb,inspect_source_glb,read_glb,texture_pixels
IDS={"canary":"v7_pet_canary_yellow_perch_hy3d_01","budgerigar":"v7_pet_budgerigar_perch_hy3d_01"}
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
def main():
    p=argparse.ArgumentParser();p.add_argument("--task-root",type=Path,required=True);p.add_argument("--kind",choices=IDS,required=True);p.add_argument("--plan",type=Path,required=True);p.add_argument("--landmarks",type=Path,required=True);a=p.parse_args();D=a.task_root/"assets"/a.kind
    plan=json.loads(a.plan.read_text());land=json.loads(a.landmarks.read_text())["landmarks"];marks={k:np.array(v["point_local_glb"],float) for k,v in land.items()}
    source=D/"generation/textured_pbr.glb";mesh=trimesh.load(source,force="mesh",process=False)
    raw_length=float(np.linalg.norm(marks["beak_tip"]-marks["tail_tip"]));scale=plan["bird_length_m"]/raw_length
    front=marks["beak_tip"]-marks["tail_tip"];theta=math.pi-math.atan2(front[0],front[2]);c,s=math.cos(theta),math.sin(theta);rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    center=np.array([*plan["raw_post_center_xz"]]);origin=np.array([center[0],plan["raw_rail_top_y"],center[1]])
    def transform(v):return (np.asarray(v)-origin)@rot.T*scale+np.array([0,plan["perch_top_m"],0])
    centroids=mesh.triangles_center;remove=np.zeros(len(mesh.faces),bool)
    def in_region(points,region):
        pts=np.asarray(points);cx,cz=region["center_xz"];rad=(pts[...,0]-cx)**2+(pts[...,2]-cz)**2
        return (rad<=region["radius"]**2)&(pts[...,1]>=region["y_min"])&(pts[...,1]<=region["y_max"])
    for region in plan["remove_regions"]:
        remove|=in_region(centroids,region)
        for name in ["beak_tip","tail_tip","left_foot","right_foot"]:
            if name in marks and bool(in_region(marks[name],region)):raise ValueError("Repair volume includes a reviewed bird landmark: "+name)
    before=len(mesh.faces);mesh.update_faces(~remove);mesh.remove_unreferenced_vertices();mesh.vertices=transform(mesh.vertices)
    scene=trimesh.Scene();scene.add_geometry(mesh,geom_name="HunyuanGeneratedBirdAndRail",node_name="HunyuanGeneratedBirdAndRail")
    wood=trimesh.visual.material.PBRMaterial(name="TaskAuthoredMatteWood",baseColorFactor=[150,103,61,255],metallicFactor=0,roughnessFactor=.9)
    def cylinder(name,radius,height,y):
        obj=trimesh.creation.cylinder(radius=radius,height=height,sections=64);r=np.array([[1,0,0],[0,0,1],[0,-1,0]],float);obj.vertices=obj.vertices@r.T+[0,y,0]
        obj.visual=trimesh.visual.texture.TextureVisuals(material=wood);scene.add_geometry(obj,geom_name=name,node_name=name);return obj
    base=plan["base_height_m"];cylinder("SilentWoodRoundBase",plan["base_radius_m"],base,base/2)
    bottom=transform(np.array([center[0],plan["raw_rail_bottom_y"],center[1]]))[1]
    post_top=bottom+.003;post_height=post_top-base;cylinder("SilentWoodVerticalPost",plan["post_radius_m"],post_height,base+post_height/2)
    raw=D/"normalized_pbr.glb";scene.export(raw)
    repaired,receipt=prepare_source_glb(raw,asset_id=IDS[a.kind],entity_class="articulated_animal",output_directory=D/"material_preparation")
    runtime=D/"runtime.glb";shutil.copy2(repaired,runtime);dump(D/"material_receipt.json",receipt)
    check=inspect_source_glb(runtime);doc,blob=read_glb(runtime);materials=[]
    for m in doc["materials"]:
        pbr=m["pbrMetallicRoughness"];rough=float(pbr["roughnessFactor"]);texture=pbr.get("metallicRoughnessTexture")
        minimum=rough if texture is None else float(texture_pixels(doc,blob,texture,runtime)[:,:,1].min()/255*rough)
        materials.append({"name":m.get("name"),"metallic":pbr["metallicFactor"],"effective_roughness_min":minimum,"status":"pass" if pbr["metallicFactor"]==0 and minimum>=.85-1e-6 else "fail"})
    assert all(x["status"]=="pass" for x in materials);assert receipt["geometry_skin_animation_unchanged"]
    dump(D/"material_check.json",{"status":"pass","production_module":"src/avengine/assets/source_materials.py","production_module_origin_commit":(a.task_root/"evidence/production_materials_commit.txt").read_text().strip(),"profiles":receipt["profiles"],"materials":materials,"images_embedded":all("bufferView" in i for i in doc["images"]),"geometry_unchanged_by_material_preparation":receipt["geometry_skin_animation_unchanged"],"highlights":"pending_native; <=2% required","ibl":"configured off by configure_source_pbr in native Habitat"})
    loaded=trimesh.load(runtime,force="scene",process=False);bounds=loaded.bounds;dimensions=np.ptp(bounds,axis=0);emit=transform(marks["beak_tip"])
    count=sum(len(g.faces) for g in loaded.geometry.values())
    meta={"asset_id":IDS[a.kind],"name_zh":"黄色家养金丝雀" if a.kind=="canary" else "黄绿虎皮鹦鹉","species":"Serinus canaria domestica" if a.kind=="canary" else "Melopsittacus undulatus","movement":"static","walk_allowed":False,"units":"metres","coordinates":{"up":"+Y","front":"-Z","origin":"round base center at floor y=0","heading_method":"yaw aligns horizontal reviewed beak-to-tail axis to -Z"},"dimensions_m":dimensions.tolist(),"bounds_min_m":bounds[0].tolist(),"bounds_max_m":bounds[1].tolist(),"bird_length_m":float(np.linalg.norm(transform(marks["beak_tip"])-transform(marks["tail_tip"]))),"bird_length_method":"approximate straight beak-tip to tail-tip distance from reviewed mesh-ray landmarks","emit_point_local_m":emit.tolist(),"emit_point_method":"near beak tip, reviewed static surface ray; no per-frame mouth precision claimed","triangles":count,"perch":{"perch_top_m":plan["perch_top_m"],"base_radius_m":plan["base_radius_m"],"base_height_m":base,"post_radius_m":plan["post_radius_m"],"post_top_m":post_top,"base_floor_y_m":float(bounds[0,1]),"produces_sound":False,"structure":"generated horizontal rail; task-authored straight wooden post and circular flat base"},"normalization":{"raw_length":raw_length,"uniform_scale":scale,"yaw_rad":theta,"removed_wood_triangles":int(remove.sum()),"input_triangles":before,"retained_generated_triangles":len(mesh.faces),"bird_geometry_source":"Hunyuan3D 2.1; no replacement bird mesh","repair_plan":str(a.plan),"landmarks":str(a.landmarks),"semantic_landmarks_local_m":{k:transform(v).tolist() for k,v in marks.items()}},"source":{"reference":str(D/"reference/candidate.png"),"reference_model":"black-forest-labs/FLUX.2-klein-4B","shape_and_pbr_model":"tencent/Hunyuan3D-2.1","weights":"/data/models/hunyuan3d-2.1/hunyuan3d-2.1","upstream_commit":(a.task_root/"evidence/hunyuan_upstream_commit.txt").read_text().strip(),"generated_pbr_glb":str(source),"modifications":"uniform bird scale, heading and raised rail; remove reviewed old wood support volumes; author flat base and straight post; apply unchanged production PBR preparation"},"license":{"flux_weights":"Apache-2.0","hunyuan_weights_and_code":"Tencent Hunyuan3D 2.1 Community License Agreement","license_originals_dir":str(a.task_root/"licenses"),"excluded_territories":["European Union","United Kingdom","South Korea"],"other_ai_model_improvement":"prohibited by section 5(b), including Outputs/results","output_rights":"Tencent claims no rights in generated Outputs; use restrictions still apply","downstream_ai_training_admission":"not_granted"},"status":"pending_native_acceptance","habitat_status":"not_run","ue_status":"not_run"}
    assert abs(meta["perch"]["base_floor_y_m"])<1e-6;assert .30<=plan["perch_top_m"]<=.45
    dump(D/"asset.json",meta);print(json.dumps({"asset_id":meta["asset_id"],"dimensions_m":dimensions.tolist(),"emit_point_local_m":emit.tolist(),"triangles":count,"removed_wood_triangles":int(remove.sum())}))
if __name__=="__main__":main()
