"""CPU checks against actual native UE geometry and source material bindings."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np


def view_rays(view,x,y,width,height):
    yaw=math.radians(view['yaw_deg']);pitch=math.radians(view['pitch_deg'])
    forward=np.array([math.cos(yaw)*math.cos(pitch),math.sin(pitch),math.sin(yaw)*math.cos(pitch)])
    right=np.array([-math.sin(yaw),0.,math.cos(yaw)])
    up=np.cross(right,forward)
    px=(x+.5)/width-.5;py=.5-(y+.5)/height
    if view.get('kind')=='orthographic':
        span=view['ortho_width_cm']/100
        origins=np.asarray(view['position_m'])+px[:,None]*span*right+py[:,None]*span*height/width*up
        directions=np.repeat(forward[None],len(x),axis=0)
        return origins,directions,np.ones(len(x))
    tan=math.tan(math.radians(85.)/2)
    ray=forward+2*px[:,None]*tan*right+2*py[:,None]*tan*height/width*up
    ray/=np.linalg.norm(ray,axis=1)[:,None]
    return np.repeat(np.asarray(view['position_m'])[None],len(x),axis=0),ray,ray@forward


def source_materials(audit):
    """Inspect the converted USD; do not infer missing textures from grey paint."""
    from pxr import Usd,UsdShade
    stage=Usd.Stage.Open(audit['source_usd'])
    if stage is None:raise ValueError('source USD cannot be opened')
    result={}
    for material in audit['materials']:
        record={'native_path':material['path'],'default_native_material':material['default_material'],
                'source_status':'unresolved','expected_basecolor_texture':None,'missing_expected_texture':False,
                'glass_hint':False,'expected_opacity':None,'native_opaque': 'BLEND_OPAQUE' in material.get('blend_mode','')}
        stage_folder=Path(audit['source_usd']).stem
        path='/'+material['path'].split('/'+stage_folder+'/',1)[-1].split('.',1)[0]
        candidates=[path,path.rsplit('/',1)[0]+'/'+path.rsplit('/',1)[1].removeprefix('MI_')]
        prim=next((stage.GetPrimAtPath(p) for p in candidates if stage.GetPrimAtPath(p)),None)
        if prim:
            usd_material=UsdShade.Material(prim)
            connected=usd_material.GetSurfaceOutput().GetConnectedSource()
            if connected:
                shader=UsdShade.Shader(connected[0].GetPrim())
                if shader.GetIdAttr().Get()=='UsdPreviewSurface':
                    record.update(source_status='preview_surface',source_material_path=str(prim.GetPath()))
                    opacity=shader.GetInput('opacity').Get()
                    record['expected_opacity']=float(opacity) if opacity is not None else 1.
                    record['glass_hint']= 'glass' in str(prim.GetPath()).lower() or record['expected_opacity']<.99
                    base=shader.GetInput('diffuseColor').GetConnectedSource()
                    if base:
                        texture_shader=UsdShade.Shader(base[0].GetPrim())
                        asset=texture_shader.GetInput('file').Get()
                        if asset is not None:
                            record['expected_basecolor_texture']=asset.path
                            native=[x['path'] for x in material['textures'] if x['parameter']=='BaseColorTexture']
                            record['missing_expected_texture']=not native or native[0] is None or native[0].startswith('/Engine/')
        result[material['path']]=record
    return result


def depth_geometry_check(audit,view,depth,room):
    """Sample native depth against exact exported render triangles.

    Check the camera depth definition globally using matched finite surfaces;
    retain both model residuals. This is a depth-channel convention check,
    never a pose/scale/geometry fit. Large disagreement fails either model.
    """
    import trimesh
    from shapely.geometry import shape
    import shapely
    geometry=audit['native_render_geometry']
    vertices=np.fromfile(geometry['world_vertices_ue_cm_f32'],dtype='<f4').reshape(-1,3)[:,[0,2,1]]/100
    faces=np.fromfile(geometry['triangles_u32'],dtype='<u4').reshape(-1,3)
    h4,w4=depth.shape;h,w=h4*4,w4*4
    # Every retained native-depth pixel; this uses task-local Embree on CPU.
    xs=np.arange(w4);ys=np.arange(h4)
    xx,yy=np.meshgrid(xs,ys);xx=xx.ravel();yy=yy.ravel()
    origins,rays,forward_factor=view_rays(view,xx*4,yy*4,w,h)
    observed=depth[yy,xx]
    if view.get('kind')=='orthographic':
        keep=shapely.contains_xy(shape(room['floor_polygon']),origins[:,0],origins[:,2])
        origins,rays,forward_factor,observed=[a[keep] for a in [origins,rays,forward_factor,observed]]
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    materials=source_materials(audit)
    section_index=np.repeat(np.arange(len(geometry['sections'])),[s['triangle_count'] for s in geometry['sections']])
    if len(section_index)!=len(faces):raise ValueError('native material sections and triangles disagree')
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    locations,ray_ids,face_ids=RayMeshIntersector(mesh).intersects_location(origins,rays,multiple_hits=False)
    expected=np.full(len(origins),np.nan);hit_faces=np.full(len(origins),-1)
    expected[ray_ids]=np.linalg.norm(locations-origins[ray_ids],axis=1);hit_faces[ray_ids]=face_ids
    # Scene depth is view-axis distance in most UE capture materials. Confirm
    # against native pixels rather than silently comparing radial distance.
    finite=np.isfinite(expected)&np.isfinite(observed)&(observed>0)&(observed<65500)
    valid_count=int(finite.sum())
    predictions={'ray_distance':expected,'view_axis_distance':expected*forward_factor}
    if view.get('kind')=='orthographic':
        # A post-process material that evaluates length(WorldPosition -
        # CameraPosition) reports distance to the fixed camera actor. In
        # orthographic projection the parallel ray origins are offset from it.
        predictions['camera_origin_distance']=np.linalg.norm(origins+rays*expected[:,None]-np.asarray(view['position_m']),axis=1)
    residuals={name:float(np.median(np.abs(observed[finite]-value[finite]))) if valid_count else None
               for name,value in predictions.items()}
    definition=min(residuals,key=residuals.get) if valid_count else 'ray_distance'
    prediction=predictions[definition]
    radial_median=residuals['ray_distance'];projected_median=residuals['view_axis_distance']
    tolerance=.01+.002*prediction # Existing v6 native-depth comparison tolerance.
    source_surface=np.isfinite(prediction)
    missing=source_surface&((~np.isfinite(observed))|(observed>=65500)|(observed-prediction>tolerance))
    penetration=source_surface&np.isfinite(observed)&(observed>0)&(prediction-observed>tolerance)
    fallback=np.zeros(len(origins),bool);glass=np.zeros(len(origins),bool);opaque_glass=np.zeros(len(origins),bool)
    unresolved=np.zeros(len(origins),bool)
    for index,face in enumerate(hit_faces):
        if face<0:continue
        section=geometry['sections'][section_index[face]];m=materials.get(section['material_path'])
        if m is None:unresolved[index]=True;continue
        fallback[index]=m['default_native_material'] or m['missing_expected_texture']
        glass[index]=m['glass_hint'];opaque_glass[index]=glass[index] and m['native_opaque']
        unresolved[index]=m['source_status']=='unresolved'
    # A real translucent surface may expose its second surface in depth. Such
    # pixels are separately classified; they cannot certify missing solid faces.
    solid=source_surface&~glass
    n=max(1,len(origins));ratio=lambda a:float(np.count_nonzero(a)/n)
    metrics={'sample_count':len(origins),'matched_surface_samples':valid_count,
        'depth_definition':definition,'physical_depth_model_median_errors_m':residuals,
        'radial_depth_median_error_m':radial_median,'projected_depth_median_error_m':projected_median,
        'missing_solid_face_fraction':ratio(missing&solid),'penetration_solid_fraction':ratio(penetration&solid),
        'default_or_missing_texture_fraction':ratio(fallback),'glass_visible_fraction':ratio(glass),
        'opaque_glass_fraction':ratio(opaque_glass),'unresolved_source_material_fraction':ratio(unresolved),
        'native_mesh_vertices':len(vertices),'native_mesh_faces':len(faces),
        'depth_tolerance':'0.01 m + 0.002 * expected depth (retained v6)',
        'source_material_records':list(materials.values()),'alignment_fitting_used':False}
    failures=[]
    for key in ['missing_solid_face_fraction','penetration_solid_fraction','default_or_missing_texture_fraction','opaque_glass_fraction','unresolved_source_material_fraction']:
        if metrics[key]>.15:failures.append(key.upper()+'_ABOVE_015')
    if len(origins)<100:failures.append('INSUFFICIENT_NATIVE_GEOMETRY_SAMPLES')
    if not valid_count:failures.append('NO_MATCHED_NATIVE_DEPTH')
    metrics.update(status='fail' if failures else 'pass',failures=failures)
    return metrics
