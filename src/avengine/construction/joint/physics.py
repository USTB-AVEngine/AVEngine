"""Original triangle verifier on every actual prepared pose; no sampled-proof promotion."""
from __future__ import annotations
from collections import OrderedDict
from types import SimpleNamespace
from itertools import combinations
import numpy as np
from .types import CheckResult


def native_full_mesh_sweep(video, entries=None):
    """Return per-entity native full-pose proofs without moving the live world.

    Uses the existing TriangleClearance instance with record_only=False/stride=1.
    Static bystanders are included by that original verifier. Multiple movers
    are additionally tested against each other at the SAME native frame, rather
    than treating another walker's first pose as a stationary obstacle.
    """
    from avengine.assets.roster_motion.episode import TriangleClearance, fcl_model
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    entries=list(video.actors if entries is None else entries)
    movers=[e for e in entries if e.get("motion_episode")]
    static=[e for e in entries if not e.get("motion_episode")]
    if not movers:
        return (CheckResult("native:full_mesh_sweep","unknown","no_prepared_motion_records"),)
    records={};verifiers={};minimum={};meshes=OrderedDict()
    for entry in movers:
        key=entry["declaration"]["asset_id"];record=entry["motion_episode"]
        matrices=np.asarray([r.get("body_to_world_m",[])for r in record.get("native_30Hz_readbacks",[])],float)
        paths=record.get("pose_glbs",[])
        if matrices.shape!=(901,4,4) or not np.isfinite(matrices).all() or len(paths)!=901:
            return (CheckResult("native:"+key+":full_pose_physics","fail","incomplete_actual_901_pose_sweep"),)
        proxy=SimpleNamespace(sim=video.sim,motion_inputs=video.motion_inputs,receipt=video.receipt,
            actors=[entry,*static],phase2_native_stage_geometry_cache=getattr(video,"phase2_native_stage_geometry_cache",{}),
            phase2_floor_collision_ray_cache=getattr(video,"phase2_floor_collision_ray_cache",{}))
        verifier=TriangleClearance(proxy,video.motion_house,record["physical_ground_y_m"],entry)
        verifier.record_only=False;verifier.stride=1
        records[key]=(paths,matrices);verifiers[key]=verifier;minimum[key]=float("inf")
    def mesh_at(path):
        if path not in meshes:
            meshes[path]=indexed_static_mesh(path)
            if len(meshes)>32:meshes.popitem(last=False)
        else:meshes.move_to_end(path)
        return meshes[path]
    for frame in range(901):
        world={};faces={};bounds={}
        for key,(paths,matrices)in records.items():
            mesh=mesh_at(paths[frame]);matrix=matrices[frame]
            vertices=np.asarray(mesh.vertices)@matrix[:3,:3].T+matrix[:3,3]
            world[key]=vertices;faces[key]=mesh.faces;bounds[key]=(vertices.min(0),vertices.max(0))
            try:
                value=verifiers[key].check_world(vertices,mesh.faces)
            except ValueError as exc:
                return (CheckResult("native:"+key+":full_pose_physics","fail",str(exc),
                    evidence={"failed_frame_30hz":frame,"checked_pose_frames":frame+1,"stride":1,"record_only":False}),)
            minimum[key]=min(minimum[key],value)
        dynamic_objects={}
        for a,b in combinations(records,2):
            required=max(verifiers[a].minimum_clearance_m,verifiers[b].minimum_clearance_m)
            lo_a,hi_a=bounds[a];lo_b,hi_b=bounds[b]
            separation=np.maximum(np.maximum(lo_a-hi_b,lo_b-hi_a),0.)
            if np.linalg.norm(separation)>required:
                continue
            for key in (a,b):
                if key not in dynamic_objects:dynamic_objects[key]=fcl_model(world[key],faces[key])
            fcl=verifiers[a].fcl;first,second=dynamic_objects[a],dynamic_objects[b]
            collided=fcl.collide(first,second,fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult())
            distance=float(fcl.distance(first,second,fcl.DistanceRequest(),fcl.DistanceResult()))
            if collided or distance<=required:
                return tuple(CheckResult("native:"+key+":full_pose_physics","fail","actual_synchronous_mover_collision_or_declared_clearance",
                    evidence={"actors":[a,b],"failed_frame_30hz":frame,"distance_m":distance,"required_m":required})for key in (a,b))
            minimum[a]=min(minimum[a],distance);minimum[b]=min(minimum[b],distance)
    return tuple(CheckResult("native:"+key+":full_pose_physics","pass","original_triangle_verifier_every_actual_pose",
        evidence={"checked_pose_frames":901,"stride":1,"record_only":False,"recorded_intersections":0,
                  "minimum_nonfloor_clearance_m":minimum[key],"floor":verifier.floor_certificate(),
                  "dynamic_pairs_checked_synchronously":len(movers)>1})for key,verifier in verifiers.items())
