"""CPU exact-triangle silhouette refinement for a prepared pose.

This predicts native masks from actual visual geometry, never consumes a target
mask. Material transparency, alpha cutout and renderer rasterization differences
remain calibration limits. Dynamic intervals require every relevant pose.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .geometry import (DepthBuffer, MeshRaycaster, VisibilityPrediction, pixel_rays,
                       project_points, camera_coordinates)


@dataclass
class MeshInstance:
    entity_id: str
    raycaster: MeshRaycaster
    transform_m: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        self.transform_m = np.asarray(self.transform_m, float)
        if (self.transform_m.shape != (4, 4) or not np.isfinite(self.transform_m).all()
                or not np.allclose(self.transform_m[3], [0, 0, 0, 1])
                or abs(np.linalg.det(self.transform_m[:3, :3])) < 1e-12):
            raise ValueError("mesh instance needs an invertible affine transform")


def _mesh_window(camera, instance):
    vertices = np.asarray(instance.raycaster.mesh.vertices)
    if not len(vertices):
        return None
    transform = instance.transform_m
    world = vertices@transform[:3,:3].T+transform[:3,3]
    xyz = camera_coordinates(camera,world)
    x,y,z = xyz.T
    th,tv = camera.width/(2*camera.focal_px),camera.height/(2*camera.focal_px)
    planes = np.column_stack((z-.01,z*th-x,z*th+x,z*tv-y,z*tv+y))
    if np.any(np.all(planes<0,axis=0)):
        return None
    if np.any(z<=.01):
        return (0,0,camera.width,camera.height)
    uv,_ = project_points(camera,world)
    x0,y0 = np.maximum(0,np.floor(uv.min(axis=0))-1).astype(int)
    x1,y1 = np.minimum([camera.width,camera.height],np.ceil(uv.max(axis=0))+1).astype(int)
    return (int(x0),int(y0),int(x1),int(y1)) if x1>x0 and y1>y0 else None


def _raster_window(camera, instance, window):
    x0,y0,x1,y1=window
    yy,xx=np.mgrid[y0:y1,x0:x1]
    directions=pixel_rays(camera,xx.ravel()+.5-camera.width/2,yy.ravel()+.5-camera.height/2)
    inverse=np.linalg.inv(instance.transform_m)
    local_origin=inverse[:3,:3]@np.asarray(camera.position_m)+inverse[:3,3]
    local_direction=directions@inverse[:3,:3].T
    ranges,_=instance.raycaster.first_hit(local_origin,local_direction)
    depth=ranges/np.linalg.norm(local_direction,axis=1)
    depth[depth<.01]=np.inf
    return depth.astype(np.float32).reshape(yy.shape)


def raster_mesh_depth(camera, instance):
    """Full image compatibility API; prediction below works only in the mesh ROI."""
    result=np.full((camera.height,camera.width),np.inf,dtype=np.float32)
    window=_mesh_window(camera,instance)
    if window is not None:
        x0,y0,x1,y1=window
        result[y0:y1,x0:x1]=_raster_window(camera,instance,window)
    return result


def predict_mesh_visibility(camera, target, *, scene_depth, occluders=(), depth_margin_m=.04):
    if scene_depth.camera!=camera:
        raise ValueError("scene depth belongs to another camera")
    window=_mesh_window(camera,target)
    evidence={"method":"exact_visual_triangles_cpu_pixel_center_rays","native_pixel_truth":False,
              "target_entity_id":target.entity_id,"material_alpha_not_modeled":True,
              "target_only_pixels":0,"geometry":target.raycaster.evidence}
    if window is None:
        return VisibilityPrediction("out_of_view",confidence="exact_mesh_prediction",evidence=evidence)
    x0,y0,x1,y1=window
    target_depth=_raster_window(camera,target,window)
    silhouette=np.isfinite(target_depth);total=int(silhouette.sum())
    evidence["target_only_pixels"]=total
    evidence["raster_roi_xyxy"]=list(window)
    if not total:
        return VisibilityPrediction("out_of_view",confidence="exact_mesh_prediction",evidence=evidence)
    scene=scene_depth.depth_m[y0:y1,x0:x1]
    if occluders:
        scene=scene.copy()
    ids=np.full(scene.shape,"",dtype=object)
    if scene_depth.object_ids is not None:
        ids[:]=scene_depth.object_ids[y0:y1,x0:x1]
    for instance in occluders:
        if instance.entity_id==target.entity_id:
            raise ValueError("target must not be included among its own occluders")
        other=_mesh_window(camera,instance)
        if other is None:
            continue
        a,b,c,d=max(x0,other[0]),max(y0,other[1]),min(x1,other[2]),min(y1,other[3])
        if c<=a or d<=b:
            continue
        patch=_raster_window(camera,instance,(a,b,c,d))
        region=scene[b-y0:d-y0,a-x0:c-x0];labels=ids[b-y0:d-y0,a-x0:c-x0]
        front=patch<region
        region[front],labels[front]=patch[front],instance.entity_id
    known=~np.isnan(scene)
    visible=silhouette&known&(target_depth<=scene+1e-5)
    blocked=silhouette&known&~visible
    count=int(visible.sum());fraction=float(blocked.sum()/total)
    blocker_ids=tuple(sorted(set(ids[blocked].tolist())-{ "" }))
    yy,xx=np.nonzero(silhouette);xx=xx+x0;yy=yy+y0
    edge=float(min(xx.min(),camera.width-1-xx.max(),yy.min(),camera.height-1-yy.max()))
    gaps=scene[silhouette&known]-target_depth[silhouette&known]
    finite=gaps[np.isfinite(gaps)]
    margin=float(np.abs(finite).min()) if len(finite) else float("inf")
    source_ok=(scene_depth.source==target.raycaster.source=="visual_mesh"
               and all(o.raycaster.source=="visual_mesh"for o in occluders))
    if np.any(silhouette&~known) or not source_ok:
        state="unknown"
    elif count==0:
        state="fully_occluded"
    elif count==total:
        state="visible_clear"
    else:
        state="visible_occluded"
    robust=int((silhouette&known&(target_depth+depth_margin_m<scene)).sum())
    upper=int((silhouette&(~known|(target_depth<=scene+depth_margin_m))).sum())
    evidence.update({"unknown_occluder_pixels":int((blocked&(ids=="")).sum()),
        "unknown_depth_pixels":int((silhouette&~known).sum()),
        "target_centroid_xy_px":[float(xx.mean()),float(yy.mean())],
        "robust_visible_mesh_pixels":robust,
        "visible_pixels_lower_authority":"exact_mesh_assuming_opaque_materials_and_final_pose",
        "required_depth_margin_m":depth_margin_m})
    return VisibilityPrediction(state,count,robust,upper,fraction,edge,margin,blocker_ids,
                                "exact_mesh_prediction",evidence)


class InstanceRaycaster:
    """World-space rays into one exact local mesh, retaining its registered ID."""
    def __init__(self, instance):
        self.instance = instance
        self.inverse = np.linalg.inv(instance.transform_m)
        self.source = instance.raycaster.source
        self.evidence = {"entity_id": instance.entity_id, "transform_m": instance.transform_m.tolist(),
                         "mesh": instance.raycaster.evidence}

    def first_hit(self, origins, directions):
        directions = np.asarray(directions, float).reshape(-1, 3)
        origins = np.broadcast_to(np.asarray(origins, float), directions.shape)
        local_origin = origins@self.inverse[:3, :3].T+self.inverse[:3, 3]
        local_direction = directions@self.inverse[:3, :3].T
        distance, _ = self.instance.raycaster.first_hit(local_origin, local_direction)
        distance *= np.linalg.norm(directions, axis=1)/np.linalg.norm(local_direction, axis=1)
        ids = np.full(len(distance), "", dtype=object)
        ids[np.isfinite(distance)] = self.instance.entity_id
        return distance, ids
