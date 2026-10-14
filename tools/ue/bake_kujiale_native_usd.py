#!/usr/bin/env python3
"""Bake composed USD mesh transforms for exact UE import, preserving geometry."""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdLux,Gf,Vt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.receipt.exists():raise FileExistsError('bake output must be new')
    start=time.monotonic();source=Usd.Stage.Open(str(args.source));cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    layer=source.Flatten();stage=Usd.Stage.Open(layer)
    # Flatten resolves composition without changing source files. Compute all
    # source world matrices before changing any mesh's authored transform.
    transforms={str(prim.GetPath()):cache.GetLocalToWorldTransform(prim) for prim in source.Traverse() if prim.IsA(UsdGeom.Mesh)}
    light_transforms={str(prim.GetPath()):cache.GetLocalToWorldTransform(prim) for prim in source.Traverse() if prim.HasAPI(UsdLux.LightAPI)}
    records=[];affine_error=0.
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim);matrix=transforms[str(prim.GetPath())]
        if mesh.GetPointsAttr().GetTimeSamples():raise ValueError('animated mesh requires a separate native import policy')
        points=mesh.GetPointsAttr().Get()
        array=np.asarray(points,dtype=np.float64);raw_matrix=np.asarray(matrix)
        changed_points=array@raw_matrix[:3,:3]+raw_matrix[3,:3]
        for index in sorted({0,len(array)//2,len(array)-1}):
            if index>=0:
                golden=np.asarray(matrix.Transform(Gf.Vec3d(*array[index])))
                affine_error=max(affine_error,float(np.max(np.abs(changed_points[index]-golden))))
        world=Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(changed_points,dtype=np.float32))
        mesh.GetPointsAttr().Set(world)
        normals=mesh.GetNormalsAttr().Get()
        if normals is not None and len(normals):
            nmatrix=matrix.GetInverse().GetTranspose()
            changed=np.asarray(normals,dtype=np.float64)@np.asarray(nmatrix)[:3,:3]
            lengths=np.linalg.norm(changed,axis=1)
            changed/=np.maximum(lengths[:,None],1e-12)
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(changed,dtype=np.float32)))
        xform=UsdGeom.Xformable(prim);xform.ClearXformOpOrder();xform.SetResetXformStack(False)
        raw=np.asarray(matrix)[:3,:3];length=np.linalg.norm(raw,axis=1)
        basis=raw/np.maximum(length[:,None],1e-15);shear=float(np.max(np.abs(basis@basis.T-np.eye(3))))
        records.append({'mesh':str(prim.GetPath()),'point_count':len(points),'world_matrix':np.asarray(matrix).tolist(),
                        'nonorthogonal_basis_max':shear,'reset_xform_stack':False})
    if not records:raise ValueError('source contains no mesh to bake')
    # UE's collapsed USD hierarchy does not reliably retain per-mesh reset
    # stacks. Every point is already in world coordinates, so remove all
    # ancestor transforms in the derived stage as well. UVs and faces stay.
    cleared=0
    for prim in stage.Traverse():
        xform=UsdGeom.Xformable(prim)
        if xform:
            xform.ClearXformOpOrder();xform.SetResetXformStack(False);cleared+=1
    # Light coordinates are not baked into mesh points. Restore their original
    # composed world transforms after clearing the hierarchy, including any
    # surviving parent light transform. This fixes the old lights-at-origin bug.
    retained_lights=[]
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if path not in light_transforms:continue
        parent=prim.GetParent();parent_world=Gf.Matrix4d(1.)
        while parent and not parent.IsPseudoRoot():
            if str(parent.GetPath()) in light_transforms:
                parent_world=light_transforms[str(parent.GetPath())];break
            parent=parent.GetParent()
        xform=UsdGeom.Xformable(prim)
        xform.MakeMatrixXform().Set(light_transforms[path]*parent_world.GetInverse())
        xform.SetResetXformStack(False)
        observed=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        error=float(np.max(np.abs(np.asarray(observed)-np.asarray(light_transforms[path]))))
        if error>1e-9:raise ValueError('authored light world transform was not preserved: '+path)
        retained_lights.append({'prim_path':path,'world_matrix':np.asarray(observed).tolist(),'max_error':error})
    if affine_error>1e-9:raise ValueError('NumPy affine transform differs from USD Gf golden vectors')
    args.output.parent.mkdir(parents=True,exist_ok=True);stage.GetRootLayer().Export(str(args.output))
    result={'schema':'kujiale_exact_native_transform_bake_v3','source':str(args.source),'output':str(args.output),
            'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'source_geometry_modified':False,'method':'full composed USD affine transform baked to mesh points and inverse-transpose normals; no fitting',
            'all_xformable_orders_cleared':cleared,'authored_light_transforms':retained_lights,'authored_light_count':len(retained_lights),'numpy_gf_max_error_m':affine_error,'mesh_count':len(records),'point_count':sum(r['point_count'] for r in records),
            'nonorthogonal_world_transform_count':sum(r['nonorthogonal_basis_max']>1e-5 for r in records),
            'meshes':records,'elapsed_seconds':time.monotonic()-start}
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['mesh_count','point_count','nonorthogonal_world_transform_count','elapsed_seconds']}))


if __name__=='__main__':main()
