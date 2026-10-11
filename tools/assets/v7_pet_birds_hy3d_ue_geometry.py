"""Check static birds against the read-only composed Kujiale source mesh and plan UE cameras."""
import argparse,json,sys,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from v7_human_roster_materials import read_glb
from v7_human_roster_pose import pose,accessor
from v7_roster_pet_parrot_geometry import downward_hit
from v7_roster_ue_camera_plan import visible
from v7_roster_scene_collision import model
def main():
    p=argparse.ArgumentParser();p.add_argument("--task-root",type=Path,required=True);p.add_argument("--reference-root",type=Path,required=True);a=p.parse_args();T=a.task_root
    sys.path.insert(0,str(a.reference_root/"round3/pydeps"));import fcl
    raw=np.load(a.reference_root/"evidence/ue_source_triangles.npz")["triangles"].astype(float)/100
    scene=np.stack([raw[:,:,1],raw[:,:,2],-raw[:,:,0]],axis=2);lo=scene.min(1);hi=scene.max(1)
    cm=json.loads((T/"evidence/ue_source_floor_plan.json").read_text())["selected"]["location_cm"];root=np.array([cm[1],cm[2],-cm[0]])/100
    records=[]
    for kind in ["canary","budgerigar"]:
        D=T/"assets"/kind;meta=json.loads((D/"asset.json").read_text());doc,blob=read_glb(D/"runtime.glb");parts,_=pose(doc,blob,canonical_flip=False,ground=False)
        vertices=[];faces=[];offset=0
        for part in parts:
            v=part["positions"];vertices.append(v);pr=doc["meshes"][doc["nodes"][part["node"]]["mesh"]]["primitives"][part["primitive"]];faces.append(accessor(doc,blob,pr["indices"]).astype(int).reshape(-1,3)+offset);offset+=len(v)
        verts=np.concatenate(vertices);indices=np.concatenate(faces);height=meta["dimensions_m"][1]
        near=scene[np.all(hi>=root-[.8,.1,.8],1)&np.all(lo<=root+[.8,height+.1,.8],1)];normal=np.cross(near[:,1]-near[:,0],near[:,2]-near[:,0]);ny=abs(normal[:,1])/np.maximum(np.linalg.norm(normal,axis=1),1e-9)
        floor=(ny>.98)&(near[:,:,1].max(1)<root[1]+.02)&(near[:,:,1].min(1)>root[1]-.02);obstacles=near[~floor]
        body=model(verts+root,indices);env=model(obstacles.reshape(-1,3),np.arange(obstacles.size//3).reshape(-1,3));collision=bool(fcl.collide(body,env,fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult()))
        distance=float(fcl.distance(body,env,fcl.DistanceRequest(),fcl.DistanceResult()))
        radius=meta["perch"]["base_radius_m"]*.7;support=[downward_hit(root+[x,0,z],near) for x,z in [(0,0),(-radius,0),(radius,0),(0,-radius),(0,radius)]]
        points=[root+[0,0,0],root+[0,height,0],root+[0,height*.5,0],root+np.array(meta["emit_point_local_m"]),root+[meta["perch"]["base_radius_m"],.01,0],root+[-meta["perch"]["base_radius_m"],.01,0]]
        cameras=[]
        for angle in [0,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3]:
            if len(cameras)==2:break
            for dist in [.65,.55,.75,.85]:
                eye=root+np.array([math.sin(angle)*dist,height*.65,-math.cos(angle)*dist])
                if all(visible(eye,point,scene,lo,hi) for point in points):
                    cameras.append({"eye_glb_m":eye.tolist(),"target_glb_m":(root+[0,height*.5,0]).tolist(),"fov_degrees":50,"distance_m":dist,"source_clear_rays":True,"source_ray_count":len(points)});break
        passed=not collision and distance>.02 and all(x is not None and abs(x)<.005 for x in support) and len(cameras)==2
        rec={"kind":kind,"asset_id":meta["asset_id"],"status":"pass" if passed else "fail","root_m":root.tolist(),"source_collision":collision,"minimum_non_floor_distance_m":distance,"base_floor_support_deltas_m":support,"expected_floor_triangles_excluded":int(floor.sum()),"obstacle_triangles":len(obstacles),"cameras":cameras,"source_triangles":str(a.reference_root/"evidence/ue_source_triangles.npz"),"method":"complete exported asset triangle BVH vs composed source USD triangles; actual floor support and camera rays","native_ue_physics":"not_claimed"}
        records.append(rec);(T/"evidence/ue_source_geometry.json").write_text(json.dumps(records,indent=2)+"\n");print("BIRD_SOURCE_GEOMETRY",kind,rec["status"],distance,support,flush=True)
if __name__=="__main__":main()
