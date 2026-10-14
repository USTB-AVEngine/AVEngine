#!/usr/bin/env python3
"""Bake authored USD world emitting dimensions for isolated UE light import."""
import argparse,json,time,math
import numpy as np
from pathlib import Path
from pxr import Usd,UsdGeom,UsdLux,UsdShade,Sdf,Gf

def world_light_dimensions(prim, world, meters_per_unit):
 """Bake emitting area for UE 5.5, whose USD translator ignores Xform scale.

 Sphere/ellipsoid uses an equal-area sphere (UE has no ellipsoid emitter).
 A sheared rectangle keeps its plane, width and area, with its shear recorded.
 Radiance stays I*2**E. Normalize=True would divide by world emitting m2.
 """
 linear=np.asarray(world,dtype=float)[:3,:3]
 scales=np.linalg.norm(linear,axis=1)
 if min(scales)<=0 or not np.isfinite(linear).all():raise ValueError('degenerate light world transform')
 axes=linear/scales[:,None]
 shear=float(np.max(np.abs(axes@axes.T-np.eye(3))))
 # Gram-Schmidt preserves the rectangle's emitting plane, not actor scale.
 # A mirrored fixture still has an emitting plane. Inverse-transpose
 # transports its local +Z normal; retaining the sign prevents front/back
 # inversion. Width/area stay positive, and UE receives a rigid right frame.
 x=axes[0];z=np.linalg.inv(linear)[:,2];z/=np.linalg.norm(z);y=np.cross(z,x);y/=np.linalg.norm(y)
 matrix=Gf.Matrix4d(1.0)
 for i,row in enumerate([x,y,z]):
  for j,value in enumerate(row):matrix[i,j]=float(value)
 matrix.SetTranslateOnly(world.ExtractTranslation())
 kind=prim.GetTypeName();area=None;attrs={};approximation=None
 if kind=='RectLight':
  light=UsdLux.RectLight(prim);width=float(light.GetWidthAttr().Get());height=float(light.GetHeightAttr().Get())
  area_scale=float(np.linalg.norm(np.cross(linear[0],linear[1])))
  attrs={'inputs:width':width*scales[0],'inputs:height':height*area_scale/scales[0]}
  area=attrs['inputs:width']*attrs['inputs:height']*meters_per_unit**2
  if shear>1e-5:approximation='equal-area rectangle in original emitting plane; USD shear unsupported by UE rect component'
 elif kind=='SphereLight':
  light=UsdLux.SphereLight(prim);radius=float(light.GetRadiusAttr().Get())
  # Integrate the area of the actual affinely transformed sphere, including
  # nonuniform scale/shear; no empirical lamp brightness multiplier.
  def integrate(n):
   u,w=np.polynomial.legendre.leggauss(n);phi=(np.arange(2*n)+.5)*math.pi/n
   normals=np.stack(np.broadcast_arrays(np.sqrt(1-u[:,None]**2)*np.cos(phi),np.sqrt(1-u[:,None]**2)*np.sin(phi),u[:,None]),axis=-1)
   jacobian=abs(np.linalg.det(linear))*np.linalg.norm(normals@np.linalg.inv(linear).T,axis=-1)
   return float(np.sum(jacobian*w[:,None])*math.pi/n*radius**2*meters_per_unit**2)
  area=integrate(64);coarse=integrate(32)
  attrs={'inputs:radius':math.sqrt(area/(4*math.pi))/meters_per_unit}
  if max(scales)/min(scales)>1.00001 or shear>1e-5:approximation='equal-area sphere for USD ellipsoid; UE point-light source-radius approximation'
 else:
  return world,{}, {'type':kind,'world_size_bake_required':False}
 api=UsdLux.LightAPI(prim);intensity=float(api.GetIntensityAttr().Get());exposure=float(api.GetExposureAttr().Get())
 normalized=bool(api.GetNormalizeAttr().Get())
 if normalized:
  if not area or area<=0:raise ValueError('normalized zero-area lamp unsupported')
  attrs['inputs:intensity']=intensity/area;attrs['inputs:normalize']=False
 radiance=float(attrs.get('inputs:intensity',intensity))*2**exposure
 expected_lumens=math.pi*area*radiance if kind=='RectLight' else 4*math.pi*area*radiance
 record={'type':kind,'world_size_bake_required':True,'original_scale':scales.tolist(),'original_shear':shear,'original_determinant':float(np.linalg.det(linear)),
         'mirrored_fixture':bool(np.linalg.det(linear)<0),'emitting_normal_world':(-z).tolist(),
         'source_intensity':intensity,'source_exposure':exposure,'source_normalize':normalized,
         'baked_inputs':{k:float(v) if not isinstance(v,bool) else v for k,v in attrs.items()},
         'world_emitting_area_m2':area,'expected_native_intensity_lumens':expected_lumens,
         'shape_approximation':approximation,'world_position_preserved':True,'source_length_cm':0.0}
 if kind=='SphereLight':record['area_quadrature_relative_difference']=abs(area-coarse)/max(area,1e-30)
 return matrix,attrs,record

def build_material_reference(source, flat, output):
 """Small QA-only copy of the real Preview Surface graph, not a mesh import.

 Native material prim paths remain the ones in the retained base map. The
 file stem must therefore retain its original imported USD namespace. Mesh
 geometry still comes only from the measured native LOD0 export.
 """
 if output.exists():raise FileExistsError('material reference must be new')
 result=Usd.Stage.CreateNew(str(output));count=0;relocations=[];unresolved=[]
 for prim in source.Traverse():
  if not (prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader)):continue
  path=prim.GetPath()
  for ancestor in list(path.GetPrefixes())[:-1]:
   if str(ancestor)=='/':continue
   if not result.GetPrimAtPath(ancestor):UsdGeom.Scope.Define(result,ancestor)
  Sdf.CopySpec(flat,path,result.GetRootLayer(),path);count+=1
 # CopySpec(Material) copied its shader descendants. Resolve each used image
 # asset against the original composed stage before moving the graph.
 for new in result.TraverseAll():
  old=source.GetPrimAtPath(new.GetPath())
  if not old:continue
  for attr in old.GetAttributes():
   value=attr.Get()
   if isinstance(value,Sdf.AssetPath) and value.path:
    if value.resolvedPath and new.HasAttribute(attr.GetName()):
     new.GetAttribute(attr.GetName()).Set(Sdf.AssetPath(value.resolvedPath));relocations.append({'prim':str(new.GetPath()),'attribute':attr.GetName(),'original':value.path,'resolved':value.resolvedPath})
    else:unresolved.append({'prim':str(new.GetPath()),'attribute':attr.GetName(),'original':value.path})
 if any(prim.IsA(UsdGeom.Mesh) for prim in result.TraverseAll()):raise ValueError('QA material graph unexpectedly contains geometry')
 result.GetRootLayer().Save()
 return {'source':source.GetRootLayer().realPath,'output':str(output),'material_or_shader_specs':count,'asset_relocations':relocations,'unresolved_declared_assets_preserved':unresolved,'qa_reference_only':True,'visual_assets_modified':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--material-output',type=Path);a=p.parse_args()
 if a.output.exists() or a.receipt.exists():raise FileExistsError('light sidecar outputs must be new')
 start=time.monotonic();source=Usd.Stage.Open(str(a.source));flat=source.Flatten();cache=UsdGeom.XformCache(Usd.TimeCode.Default())
 stage=Usd.Stage.CreateNew(str(a.output));root=UsdGeom.Xform.Define(stage,'/AuthoredLightsWorldSize').GetPrim();stage.SetDefaultPrim(root)
 UsdGeom.SetStageUpAxis(stage,UsdGeom.GetStageUpAxis(source));UsdGeom.SetStageMetersPerUnit(stage,UsdGeom.GetStageMetersPerUnit(source))
 material_reference=build_material_reference(source,flat,a.material_output) if a.material_output else None
 records=[]
 for old in source.Traverse():
  if not old.HasAPI(UsdLux.LightAPI):continue
  target=Sdf.Path('/AuthoredLightsWorldSize/L'+str(len(records)).zfill(4));Sdf.CopySpec(flat,str(old.GetPath()),stage.GetRootLayer(),target)
  new=stage.GetPrimAtPath(target);world=cache.GetLocalToWorldTransform(old)
  baked,dimensions,photometry=world_light_dimensions(old,world,UsdGeom.GetStageMetersPerUnit(source))
  for name,value in dimensions.items():new.GetAttribute(name).Set(value)
  x=UsdGeom.Xformable(new);x.ClearXformOpOrder();x.MakeMatrixXform().Set(baked);x.SetResetXformStack(True)
  assets=[]
  for attr in old.GetAttributes():
   value=attr.Get()
   if isinstance(value,Sdf.AssetPath) and value.path:
    if not value.resolvedPath:raise ValueError('unresolved authored lamp asset: '+str(old.GetPath())+'/'+attr.GetName())
    new.GetAttribute(attr.GetName()).Set(Sdf.AssetPath(value.resolvedPath));assets.append({'attribute':attr.GetName(),'resolved_path':value.resolvedPath})
  new.SetCustomDataByKey('avengineOriginalPrimPath',str(old.GetPath()))
  records.append({'source_prim':str(old.GetPath()),'sidecar_prim':str(target),'type':old.GetTypeName(),'world_matrix_usd':[list(row) for row in cache.GetLocalToWorldTransform(old)],'asset_relocations':assets,'photometry':photometry})
 if not records:raise ValueError('source has no authored lights')
 if any(p.IsA(UsdGeom.Mesh) for p in stage.Traverse()):raise ValueError('light sidecar unexpectedly contains geometry')
 stage.GetRootLayer().Save()
 a.receipt.write_text(json.dumps({'status':'pass','source':str(a.source),'output':str(a.output),'meters_per_unit':UsdGeom.GetStageMetersPerUnit(source),'light_count':len(records),'lights':records,'method':'authored light specs; world emitting dimensions baked into rigid USD transforms; equal-area UE sphere/rectangle approximation explicitly recorded; no empirical intensity tuning','source_modified':False,'material_reference':material_reference,'geometry_sidecar_contains_no_mesh':True,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
 print('AUTHORED_LIGHT_SIDECAR',len(records),time.monotonic()-start)
if __name__=='__main__':main()
