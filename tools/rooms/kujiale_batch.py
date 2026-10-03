#!/usr/bin/env python3
"""CPU InteriorAgent inventory and conversion; never grants native UE admission."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import sys
import time
import numpy as np
import yaml
from shapely.geometry import Polygon, mapping
from shapely import make_valid
from shapely.ops import unary_union

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / 'src')]

ROOM_TYPES = {'living room':'living', 'livingroom':'living', 'bedroom':'bedroom',
              'kitchen':'kitchen', 'bathroom':'bathroom', 'dining room':'dining',
              'study room':'office', 'study':'office', 'office':'office',
              'balcony':'outdoor', 'terrace':'outdoor', 'courtyard':'outdoor',
              'corridor':'corridor', 'hallway':'corridor', 'stairs':'stairs',
              'storage':'storage','storage room':'storage', 'garage':'garage', 'laundry room':'utility'}

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def progress(output, message):
    import datetime
    import fcntl
    with (Path(output)/'PROGRESS_zh.md').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(f"\n阶段记录 {datetime.datetime.now(datetime.timezone.utc).isoformat()}：{message}\n")

def write_existing_or_new(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Resume metadata differs: '+str(path))
    else:
        write(path,value)

def completed_build(output, house, preferred='build_v4'):
    selection_path=Path(output)/'evidence/build_selection.json'
    selected=json.loads(selection_path.read_text()).get(house,preferred) if selection_path.exists() else preferred
    if not selected.startswith('build_') or '/' in selected or '..' in selected:
        raise ValueError('Invalid task build name: '+selected)
    return Path(output)/'houses'/house/selected

def type_of(value):
    return ROOM_TYPES.get(str(value).strip().lower(), 'unknown')

def thresholds(path):
    data = yaml.safe_load(Path(path).read_text())
    return {key: row['value'] for key, row in data['parameters'].items()}

def world_geometry(prim, cache, UsdGeom):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    flat = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    if points.ndim != 2 or points.shape[1] != 3 or counts.sum() != len(flat):
        raise ValueError('Invalid Mesh arrays: ' + str(prim.GetPath()))
    if len(flat) and (flat.min() < 0 or flat.max() >= len(points)):
        raise ValueError('Invalid face indices: ' + str(prim.GetPath()))
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float)
    world = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
    if not np.isfinite(world).all():
        raise ValueError('Nonfinite transformed points')
    holes = set(mesh.GetHoleIndicesAttr().Get() or [])
    if len(counts) and np.all(counts==3) and not holes:
        return world, flat.reshape(-1,3)
    faces, offset = [], 0
    for index, count in enumerate(counts):
        face = flat[offset:offset+count]; offset += count
        if index in holes:
            continue
        if count < 3:
            raise ValueError('Face has fewer than three vertices')
        faces.extend((face[0],face[j],face[j+1]) for j in range(1,count-1))
    return world, np.asarray(faces, dtype=np.int64).reshape(-1,3)

def audit_one(source_dir, output, params):
    from pxr import Usd, UsdGeom, UsdShade, Sdf
    start = time.monotonic()
    source = source_dir / (source_dir.name + '.usda')
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError('USD stage cannot be composed')
    layers = stage.GetUsedLayers()
    assets, unresolved = set(), set()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if not value or not value.path:
                continue
            assets.add(value.path)
            if not value.resolvedPath:
                unresolved.add(value.path)
    unresolved = sorted(unresolved)
    result = {'house':source_dir.name, 'source':str(source), 'family':'kujiale',
              'source_revision':json.loads((source_dir.parent/'MANIFEST_hf_download_20261002.json').read_text())['revision'], 'room_metadata':str(source_dir/'rooms.json'),
              'stage_up_axis':str(UsdGeom.GetStageUpAxis(stage)),
              'stage_meters_per_unit':float(UsdGeom.GetStageMetersPerUnit(stage)),
              'composition_errors':[str(x) for x in stage.GetCompositionErrors()],
              'unresolved_assets':[str(x) for x in unresolved],
              'dependency_layers':len(layers),'dependency_assets':len(assets),
              'scope_names':[p.GetName() for p in stage.GetPrimAtPath('/Root/Meshes').GetChildren()],
              'usd_mesh_count':0,'visible_mesh_count':0,'triangles':0,'vertices':0,
              'zero_area_triangles':0,'mesh_errors':[], 'rooms':[]}
    floor_triangles = []
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        result['usd_mesh_count'] += 1
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        result['visible_mesh_count'] += 1
        try:
            verts, faces = world_geometry(prim, cache, UsdGeom)
            result['vertices'] += len(verts); result['triangles'] += len(faces)
            tri = verts[faces]
            area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            result['zero_area_triangles'] += int((area<=1e-12).sum())
            if str(prim.GetPath()).startswith('/Root/Meshes/floor/'):
                floor_triangles.append(tri)
        except Exception as exc:
            result['mesh_errors'].append({'prim':str(prim.GetPath()),'error':str(exc)})
    floor = np.concatenate(floor_triangles) if floor_triangles else np.empty((0,3,3))
    for index, raw in enumerate(json.loads((source_dir/'rooms.json').read_text())):
        original_polygon = Polygon(raw['polygon'])
        polygon = original_polygon if original_polygon.is_valid else make_valid(original_polygon)
        polygon = unary_union([g for g in (polygon.geoms if hasattr(polygon,'geoms') else [polygon]) if g.geom_type in {'Polygon','MultiPolygon'}])
        if polygon.is_empty:
            result['rooms'].append({'room_label':f'R{index}','region_id':str(index),'source_room_type':raw['room_type'],'room_type':type_of(raw['room_type']),'metadata_area_m2':0.,'metadata_short_side_m':0.,'geometric_precheck_pass':False,'precheck_reasons':['INVALID_EMPTY_ROOM_POLYGON'],'floor_reference_status':'not_measured','status':'review'})
            continue
        coords = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)
        short = float(np.linalg.norm(np.diff(coords,axis=0),axis=1).min())
        row = {'room_label':f'R{index}', 'region_id':str(index), 'source_room_type':raw['room_type'],
               'room_type':type_of(raw['room_type']), 'metadata_area_m2':float(polygon.area),
               'metadata_short_side_m':short, 'floor_polygon':raw['polygon'], 'floor_geometry':mapping(polygon), 'original_polygon_valid':original_polygon.is_valid, 'polygon_operation':'none' if original_polygon.is_valid else 'make_valid_keep_all_positive_area_components',
               'precheck_reasons':[], 'status':'review', 'floor_reference_status':'not_measured'}
        # Measure original geometry before proposing a floor offset or an adapter.
        pieces=[]
        for triangle in floor:
            if np.ptp(triangle[:,2])>0.02:
                continue
            clipped=Polygon(triangle[:,:2]).intersection(polygon)
            if clipped.area>1e-10:
                pieces.append((float(np.median(triangle[:,2])),clipped))
        if pieces:
            heights=sorted({round(h,3) for h,_ in pieces})
            groups=[]; assigned=set()
            # Cluster with the frozen maximum height span, avoiding bridge chains.
            while len(assigned)<len(heights):
                remaining=[h for h in heights if h not in assigned]
                options=[(unary_union([p for h,p in pieces if a<=h<=a+params['floor_height_separation_m']]).area,a) for a in remaining]
                _,lo=max(options)
                hs=[h for h in remaining if lo<=h<=lo+params['floor_height_separation_m']]
                assigned.update(hs)
                ps=[p for h,p in pieces if lo-0.0005<=h<=lo+params['floor_height_separation_m']+0.0005]
                group=unary_union(ps)
                actual=[(h,p.area) for h,p in pieces if lo-0.0005<=h<=lo+params['floor_height_separation_m']+0.0005]
                actual.sort();cum=np.cumsum([a for _,a in actual]);med=actual[int(np.searchsorted(cum,cum[-1]/2))][0]
                groups.append({'floor_height_m':med,'projected_union_area_m2':float(group.area)})
            groups.sort(key=lambda x:x['projected_union_area_m2'],reverse=True)
            row.update(floor_reference_status='measured_usd_geometry_pending_native_readback',floor_levels=groups,
                       floor_height_m=groups[0]['floor_height_m'],floor_area_m2=groups[0]['projected_union_area_m2'],
                       dominant_floor_area_fraction=groups[0]['projected_union_area_m2']/sum(g['projected_union_area_m2'] for g in groups),
                       native_ue_floor_offset_m=None)
        if row['room_type'] in params['excluded_types']:
            row['precheck_reasons'].append('EXCLUDED_ROOM_TYPE')
        if row['room_type']=='unknown':
            row['precheck_reasons'].append('UNKNOWN_DESIGN_SEMANTICS')
        if row['metadata_area_m2']<params['floor_area_min_m2']:
            row['precheck_reasons'].append('METADATA_AREA_BELOW_FROZEN_THRESHOLD')
        if short<params['short_side_min_m']:
            row['precheck_reasons'].append('METADATA_SHORT_SIDE_BELOW_FROZEN_THRESHOLD')
        row['geometric_precheck_pass']=not row['precheck_reasons']
        result['rooms'].append(row)
    regular=[p for p in source_dir.rglob('*') if p.is_file()]
    result.update(raw_bytes=sum(p.stat().st_size for p in regular), raw_files=len(regular),
                  metadata_room_count=len(result['rooms']),room_type_counts=dict(Counter(r['source_room_type'] for r in result['rooms'])),
                  metadata_area_sum_m2=sum(r['metadata_area_m2'] for r in result['rooms']),
                  measured_floor_levels_m=sorted({round(r['floor_height_m'],3) for r in result['rooms'] if 'floor_height_m' in r}),
                  estimated_surface_array_bytes=12*(result['vertices']+result['triangles']),
                  usd_complete=not(result['composition_errors'] or result['unresolved_assets'] or result['mesh_errors']),
                  audit_wall_seconds=time.monotonic()-start,
                  native_visual_status='not_run',navmesh_status='not_run',acoustic_status='not_run',qualification='research_only')
    write(output/'houses'/source_dir.name/'inventory.json',result)
    print(json.dumps({k:result[k] for k in ['house','metadata_room_count','usd_complete','zero_area_triangles','audit_wall_seconds']},ensure_ascii=False),flush=True)
    return result

def audit(args):
    results=[];params=thresholds(args.thresholds)
    for source in sorted(args.source_root.glob('kujiale_*')):
        try:
            existing=args.output/'houses'/source.name/'inventory.json'
            results.append(json.loads(existing.read_text()) if existing.exists() else audit_one(source,args.output,params))
        except Exception as exc:
            value={'house':source.name,'error':repr(exc),'usd_complete':False,'rooms':[]}
            write(args.output/'houses'/source.name/f'inventory_error_{time.time_ns()}.json',value);results.append(value)
            print(json.dumps(value),flush=True)
    write(args.output/'inventory.json',results)
    raw=sum(r.get('raw_bytes',0) for r in results)
    estimate=raw*3+2_000_000_000
    write(args.output/'evidence/storage_estimate.json',{'raw_bytes':raw,'derived_estimate_bytes':estimate,'limit_bytes':50_000_000_000,
          'method':'conservative planning allowance: 3 x downloaded bytes + 2 GB private logs/cache; recheck after pilot',
          'historical_size_source':'v7_rooms_20261002/evidence/kujiale_existing_asset_sizes.json',
          'is_measured_final_size':False,'within_limit':estimate<=50_000_000_000})
    if estimate>50_000_000_000:
        raise SystemExit('STOP: derived estimate exceeds 50 GB')

def portable_preview(adapter_path, output, preserve_preview_graphs=False):
    """Flatten composed geometry and retain only the portable Preview graph.

    Re-encode used image inputs, replace native UE material metadata, and check
    the resulting dependency closure. Original data are read-only.
    """
    from pxr import Usd, UsdGeom, UsdShade, Sdf, UsdUtils
    from PIL import Image
    output.mkdir(parents=True,exist_ok=False)
    textures=output/'textures';textures.mkdir()
    layer=Usd.Stage.Open(str(adapter_path)).Flatten()
    stage=Usd.Stage.Open(layer)
    removed_shaders=0; removed_engine_hints=0
    empty_meshes=[]
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim)
        counts=mesh.GetFaceVertexCountsAttr().Get()
        indices=mesh.GetFaceVertexIndicesAttr().Get()
        if (counts is None or len(counts)==0) and (indices is None or len(indices)==0):
            empty_meshes.append(str(prim.GetPath()))
            prim.SetTypeName('Xform')
    kept_shader_paths=set()
    def keep_graph(connectable):
        if not connectable:return
        path=str(connectable.GetPrim().GetPath())
        if path in kept_shader_paths:return
        if connectable.GetPrim().IsA(UsdShade.Shader):
            ident=UsdShade.Shader(connectable.GetPrim()).GetIdAttr().Get()
            if ident not in {'UsdPreviewSurface','UsdUVTexture','UsdPrimvarReader_float2','UsdTransform2d'}:
                raise ValueError('Unsupported original Preview graph shader: '+str(ident))
        kept_shader_paths.add(path)
        for inp in connectable.GetInputs()+connectable.GetOutputs():
            connected=inp.GetConnectedSource()
            if connected:keep_graph(connected[0])
    if preserve_preview_graphs:
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Material):continue
            surface=UsdShade.Material(prim).ComputeSurfaceSource()
            if surface and surface[0]:keep_graph(surface[0].ConnectableAPI())
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdShade.Shader) and not prim.GetName().startswith('AVEngine') and str(prim.GetPath()) not in kept_shader_paths:
            stage.RemovePrim(prim.GetPath());removed_shaders+=1
    images={}; records=[]; missing_texture_fallbacks=[]
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdShade.Material):
            for output_attr in UsdShade.Material(prim).GetOutputs():
                if output_attr.GetFullName() != 'outputs:surface':
                    prim.RemoveProperty(output_attr.GetFullName())
        for attr in list(prim.GetAttributes()):
            if attr.GetTypeName()!=Sdf.ValueTypeNames.Asset:
                continue
            value=attr.Get()
            if not value or not value.path:
                continue
            if value.path.startswith(('/Game/','/Engine/')):
                prim.RemoveProperty(attr.GetName());removed_engine_hints+=1;continue
            source=Path(value.resolvedPath or value.path)
            if not source.is_file():
                if preserve_preview_graphs and prim.IsA(UsdShade.Shader) and UsdShade.Shader(prim).GetIdAttr().Get()=='UsdUVTexture' and attr.GetName()=='inputs:file':
                    missing_texture_fallbacks.append({'prim':str(prim.GetPath()),'unresolved':value.path,'action':'clear file; retain authored UsdUVTexture fallback; native review required'})
                    attr.Set(Sdf.AssetPath(''))
                    continue
                raise ValueError('Portable stage input is unresolved: '+value.path)
            if str(source) not in images:
                stem=f'image_{len(images):05d}'
                if source.suffix.lower() in {'.hdr','.exr'}:
                    import os
                    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
                    import cv2
                    array=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
                    if array is None:
                        raise ValueError('HDR texture could not be decoded: '+str(source))
                    dest=textures/(stem+('.exr' if source.suffix.lower()=='.exr' else '.hdr'))
                    if not cv2.imwrite(str(dest),array):
                        raise ValueError('HDR texture encoding failed')
                    dimensions=list(array.shape[:2])
                else:
                    dest=textures/(stem+'.png')
                    with Image.open(source) as image:
                        image.load()
                        if image.mode not in {'RGB','RGBA','L','LA','I;16'}:
                            image=image.convert('RGBA')
                        dimensions=[image.height,image.width]
                        image.save(dest,compress_level=1)
                images[str(source)]=dest.relative_to(output).as_posix()
                records.append({'source':str(source),'derived':images[str(source)],'dimensions_hw':dimensions,'bytes':dest.stat().st_size,'operation':'decode_and_reencode_no_resizing'})
            attr.Set(Sdf.AssetPath(images[str(source)]))
    # Source identities remain in the evidence receipt, not as operational USD inputs.
    layer.customLayerData={'avengine:adapter':'portable_interioragent_preview_surface_v1',
                           'avengine:qualification':'research_only',
                           'avengine:redistribution':'prohibited_external_dataset_terms'}
    destination=output/'render.usdc'
    layer.Export(str(destination))
    _,dependencies,unresolved=UsdUtils.ComputeAllDependencies(str(destination))
    root=output.resolve()
    external=[str(p) for p in dependencies if not Path(p).resolve().is_relative_to(root)]
    if unresolved or external:
        raise ValueError('Portable USD still depends on external assets: '+str([external,unresolved]))
    bound=0;missing=0
    reopened=Usd.Stage.Open(str(destination))
    for prim in reopened.Traverse():
        if prim.GetTypeName()!='Mesh':continue
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            bound+=1
            shader=material.ComputeSurfaceSource()
            if not shader or not shader[0]:missing+=1
        else:missing+=1
    receipt={'status':'converted','qualification':'research_only','derived_stage':str(destination),
             'missing_texture_fallbacks':missing_texture_fallbacks,'original_preview_shader_paths_retained':sorted(kept_shader_paths),'empty_mesh_containers_preserved_as_xform':empty_meshes,'usd_dependencies_external':external,'unresolved_dependencies':list(unresolved),
             'source_mdl_shaders_removed':removed_shaders,'unreal_material_hints_removed':removed_engine_hints,
             'used_images_reencoded':len(images),'bound_meshes':bound,'mesh_materials_unresolved':missing,
             'images':records,'native_ue_pixels':'not_run','native_ue_material_bake':'not_run',
             'physical_acoustic_material_truth':False}
    write(output/'conversion.json',receipt)
    return receipt

def screen_geometry(inventory, vertices, faces, objects, surfaces, nav_path, runtime, params):
    """Adapt design polygons to the shared measured-floor/nav/placement rules."""
    import shapely
    import trimesh
    from types import SimpleNamespace
    from shapely.geometry import shape
    from tools.rooms.room_selection.geometry import floor_windows
    from tools.rooms.room_selection.protocol import scope_metrics,decide
    from tools.rooms.room_selection.navigation import sample_navigation,placement
    from tools.rooms.room_selection.measurements import navigation_geometry,layer_furniture
    from tools.rooms.room_screening.geometry import classify_category
    from tools.rooms.room_screening.compute_semantic_region_candidates import navmesh_triangles
    pf,nav_polygons,nav_heights=navmesh_triangles(runtime.habitat_sim,nav_path)
    semantic={s['source_material_name']:s['semantic_category'] for s in surfaces}
    floor_parts=[];instances=[]
    for index,obj in enumerate(objects):
        triangles=vertices[faces[obj['triangle_offset']:obj['triangle_offset']+obj['triangle_count']]]
        category=semantic[obj['source_material_name']]
        if category=='floor':floor_parts.append(triangles)
        elif classify_category(category.replace('_',' '))=='blocker':
            instances.append({'role':'blocker','region_id':-1,'instance_id':index,
                              'category':category.replace('_',' '),'triangles':triangles})
    floor=np.concatenate(floor_parts) if floor_parts else np.empty((0,3,3))
    # Design floors are closed slabs. Bottom faces are not walking floors.
    cross=np.cross(floor[:,1]-floor[:,0],floor[:,2]-floor[:,0])
    floor=floor[cross[:,1]>0]
    # Semantic-only face selection; navigation and acoustics retain every face.
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    results=[];furniture_cache={}
    for room in inventory['rooms']:
        row=dict(room)
        row['room_type']=type_of(row['source_room_type'])
        if 'floor_geometry' not in row:
            row.update(geometry_eligible=False,metrics={},reason_codes=['INVALID_ROOM_POLYGON'])
            results.append(row);continue
        design=shape(row['floor_geometry']); ground=[]
        for triangle in floor:
            polygon=Polygon(triangle[:,[0,2]])
            if polygon.area<=params['projection_min_triangle_area_m2']:continue
            clipped=polygon.intersection(design)
            if clipped.area>params['projection_min_triangle_area_m2']:
                ground.append({'polygon':clipped,'ys':float(triangle[:,1].mean())})
        levels=floor_windows(ground,params)
        if not levels:
            row.update(geometry_eligible=False,metrics={},reason_codes=['USD_FLOOR_NOT_FOUND'])
            results.append(row);continue
        for level in levels:level['area_m2']=level['geometry'].area
        selected=max(levels,key=lambda x:x['area_m2'])
        fraction=selected['area_m2']/sum(x['area_m2'] for x in levels)
        scope=selected['geometry'];height=selected['floor_y_m']
        if height not in furniture_cache:
            # All design instances are unassigned, so every region sees the same
            # exact-height footprints. Cache those unchanged shared measurements.
            furniture_cache[height]=layer_furniture(SimpleNamespace(instances=instances),-1,height,1.5)
        furniture=furniture_cache[height]
        metrics=scope_metrics(scope,furniture)
        sampled,points,clearance,adj,components=sample_navigation(pf,runtime.habitat_sim,scope,height,params)
        metrics.update(sampled)
        metrics.update(navigation_geometry(nav_polygons,nav_heights,height,scope,params,
                                          shape(sampled['nav_grid_main_polygon'])))
        metrics.update(floor_y_m=height,dominant_floor_area_fraction=fraction,
                       floor_count=len(levels),black_fraction=None)
        metrics['placement']=placement(mesh,points,clearance,adj,params)
        decision=decide(metrics,row['room_type'],params)
        reasons=list(decision['reason_codes'])
        if fraction<params['dominant_floor_area_fraction_min']:reasons.append('DOMINANT_FLOOR_BELOW_FROZEN_THRESHOLD')
        if row['room_type']=='unknown':reasons.append('UNKNOWN_DESIGN_SEMANTICS')
        # Native RGB is required for the design-scene visual metric; it is missing.
        eligible=(not decision['known_failures'] and
                  set(decision['missing_evidence'])<= {'BLACK_FRACTION_UNAVAILABLE'} and
                  fraction>=params['dominant_floor_area_fraction_min'] and row['room_type']!='unknown')
        row.update(metrics=metrics,geometry_eligible=eligible,status='review' if eligible else decision['status'],
                   reason_codes=reasons,visual_status='not_run',native_floor_readback_status='not_run',
                   floor_height_m=height,dominant_floor_area_fraction=fraction)
        results.append(row)
        print('ROOM',inventory['house'],row['room_label'],round(metrics['floor_area_m2'],3),
              round(metrics['nav_main_area_m2'],3),eligible,flush=True)
    return results

def convert_one(args,inventory):
    import traceback
    import trimesh
    from types import SimpleNamespace
    from pxr import Usd
    from tools.rooms.prepare_interioragent_kujiale_adapter import prepare_adapter
    from tools.acoustics.extract_usd_acoustic_snapshot import _extract_stage,extract
    from tools.acoustics.accept_houses import choose_rooms,compact_room
    from avengine.rooms.navigation_preparation import prepare_render_surface_navigation
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    from avengine.acoustics.compiler import compile_usd_snapshot_semantic_research_scene
    from avengine.acoustics.research_cleanup import derive_rlr_compatible_research_package,ResearchCleanupError
    house=inventory['house'];root=args.output/'houses'/house/args.build_name
    root.mkdir(exist_ok=args.resume)
    source=Path(inventory['source']);started=time.monotonic(); cpu_started=time.process_time()
    baseline=json.loads(args.baseline_manifest.read_text())
    params=thresholds(args.thresholds)
    views=[]
    candidates=[r for r in inventory['rooms'] if r.get('geometric_precheck_pass') and r.get('floor_geometry')]
    if not candidates:candidates=[r for r in inventory['rooms'] if r.get('floor_geometry')]
    for i in range(4):
        r=candidates[i%len(candidates)]
        from shapely.geometry import shape
        point=shape(r['floor_geometry']).representative_point()
        views.append({'view_id':f'proposal_{i}','position_m':[point.x,point.y,r.get('floor_height_m',0.)+1.5],
                      'yaw_deg':90.*i})
    profile={'schema_version':'avengine_optional_interioragent_kujiale_profile_v1',
             'dataset_id':'spatialverse/InteriorAgent','scene_id':house,
             'room_type':inventory['rooms'][0]['source_room_type'],
             'map_path':f'/Game/AVEngine/V7Kujiale/{house}',
             'selected_scopes':inventory['scope_names'],'include_rendering_scope':True,
             'camera_views':views,'review_lights':[]}
    adapter=root/'adapter';adapter.mkdir(exist_ok=args.resume)
    write_existing_or_new(adapter/'profile.json',profile)
    if args.resume and (adapter/'receipt.json').exists():
        result=json.loads((adapter/'receipt.json').read_text())
    else:
        result=prepare_adapter(source=source,output=adapter/'preview.usda',profile_path=adapter/'profile.json',
                               result_path=adapter/'receipt.json',replace=False,whole_stage=True)
    if args.resume and (root/'engine/conversion.json').exists():
        visual=json.loads((root/'engine/conversion.json').read_text())
    else:
        visual=portable_preview(adapter/'preview.usda',root/'engine')
    progress(args.output,f"{house}：Preview Surface 派生闭包完成；原生画面尚未核验。")
    verts,triangles,objects,surfaces,metadata=_extract_stage(source=root/'engine/render.usdc',
                       room_id=house,transform_profile='kujiale_z_up_y_back_to_habitat')
    # The reviewed x,z,y transform reflects the triangle winding once.
    canonical=verts[:,[0,2,1]].copy();faces=triangles[:,[0,2,1]].copy()
    surface=root/'surface';surface.mkdir(exist_ok=args.resume)
    if args.resume and (surface/'surface.glb').exists():
        if not (np.array_equal(np.load(surface/'vertices.npy'),canonical) and np.array_equal(np.load(surface/'triangles.npy'),faces)):
            raise ValueError('Resume surface arrays differ from composed USD')
    else:
        np.save(surface/'vertices.npy',canonical);np.save(surface/'triangles.npy',faces)
        trimesh.Trimesh(canonical,faces,process=False).export(surface/'surface.glb')
    geometry_manifest={'schema':'avengine_room_package_v1','room_id':house,
            'room_kind':'external_usd_real_surface','geometry_representation':'real_surface_mesh',
            'coordinate_system':{'handedness':'right','up_axis':'+Y','forward_axis':'-Z','linear_unit':'meter','quaternion_order':'xyzw'},
            'assets':[{'role':'render_surface_mesh','path':'surface/surface.glb','license':'InteriorAgent noncommercial research; no redistribution'}],
            'navigation':{'agent_height_m':1.5,'agent_radius_m':.2,'include_static_objects':True},
            'provenance':{'geometry':'composed portable USD transforms and visible Mesh triangles',
                          'source_stage':str(root/'engine/render.usdc'),'native_ue_export':False},
            'qualification':'research_only','native_ue_visual_status':'not_run'}
    write_existing_or_new(root/'geometry_room_manifest.json',geometry_manifest)
    if args.resume and (root/'navigation/build_result.json').exists():
        nav=json.loads((root/'navigation/build_result.json').read_text())
        if not Path(nav['navmesh']).is_file():raise ValueError('Resume navmesh missing')
    else:
        nav=prepare_render_surface_navigation(root/'geometry_room_manifest.json',root/'navigation',
                  static_vertices=surface/'vertices.npy',static_triangles=surface/'triangles.npy',**baseline['runtime'])
    progress(args.output,f"{house}：同一 USD 几何的 GLB 与 CPU navmesh 完成。")
    runtime=prepare_installed_habitat_runtime(**baseline['runtime'])
    if args.resume and (root/'rooms_screened.json').exists():
        rows=json.loads((root/'rooms_screened.json').read_text())
    else:
        rows=screen_geometry(inventory,canonical,faces,objects,surfaces,Path(nav['navmesh']),runtime,params)
        write(root/'rooms_screened.json',rows)
    progress(args.output,f"{house}：冻结几何门槛筛选完成，{sum(r['geometry_eligible'] for r in rows)} 间几何候选；正式训练准入仍未完成。")
    write_existing_or_new(root/'semantic_rooms.json',{'house':house,'source':'InteriorAgent rooms.json + upward-facing visible USD floor triangles; slab undersides excluded',
           'coordinate_system':'Habitat [x,z,y] from original USD; polygons in x,z metres',
           'rooms':[{'room_label':r['room_label'],'room_type':r['room_type'],'source_room_type':r['source_room_type'],
                     'floor_geometry':r.get('floor_geometry'),'floor_height_m':r.get('floor_height_m'),
                     'geometry_eligible':r['geometry_eligible']} for r in rows]})
    floor_readout={'house':house,'status':'measured_source_geometry_pending_native_ue',
                  'rooms':[{'room_label':r['room_label'],'floor_height_m':r.get('floor_height_m'),
                            'native_ue_floor_offset_m':None,'native_collision_or_depth_status':'not_run',
                            'actor_foot_contact_status':'not_run'} for r in rows],
                  'first_new_house_step':str(root.parent/'inventory.json'),
                  'claim':'USD floor geometry is measured before adapting; zero UE offset is never copied from 0020'}
    write_existing_or_new(root/'floor_reference.json',floor_readout)
    selected=choose_rooms([r for r in rows if r['geometry_eligible']],3)
    package=None
    if selected:
        origins=[r['metrics']['placement']['camera_m'] for r in selected]
        if len(origins)<2:origins.append(selected[0]['metrics']['placement']['source_1_m'])
        snapshot=root/'acoustic_snapshot'
        extract(source=root/'engine/render.usdc',output=snapshot,room_id=house,
                transform_profile='kujiale_z_up_y_back_to_habitat',interior_origins=origins,
                source_revision=inventory['source_revision'],dataset_id='spatialverse/InteriorAgent',
                source_license='InteriorAgent: noncommercial research, attribution, no redistribution')
        package,coverage=compile_usd_snapshot_semantic_research_scene(room_manifest=snapshot/'room_manifest.json',
                  material_rules=args.material_rules,output=root/'acoustic_package',seed=20260905)
        try:
            derived=derive_rlr_compatible_research_package(package,root/'acoustic_package_rlr')
        except ResearchCleanupError as exc:
            if 'no RLR-incompatible triangles' not in str(exc):raise
            derived=package
        package=derived
    house_entry={'house':house,'family':'kujiale','scan_id':'surface','original_split':'train',
        'scene_directory':str(surface),'semantic_source':str(root/'engine/render.usdc'),
        'annotation_source':str(root/'semantic_rooms.json'),'navmesh_source':str(root/'navigation/navigation.navmesh'),
        'candidate_room_count':sum(r['geometry_eligible'] for r in rows),
        'candidate_area_sum_m2':sum(r.get('metrics',{}).get('floor_area_m2',0.) for r in rows if r['geometry_eligible']),
        'room_types':sorted({r['room_type'] for r in rows if r['geometry_eligible']}),
        'known_exposure':house=='kujiale_0020','known_exposure_source':'task: v6 kujiale_0020_full_home_v1',
        'acoustic_rooms':[compact_room(r) for r in selected], 'acoustic_package_manifest':str(package) if package else None,
        'geometry_authority':'portable composed USD; native UE export consistency remains unverified',
        'training_only':True,'registry_source':str(root/'rooms_screened.json')}
    write(root/'acceptance_house.json',house_entry)
    measured_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    receipt={'house':house,'status':'converted_cpu_research_assets',
             'wall_seconds':time.monotonic()-started,'driver_cpu_seconds':time.process_time()-cpu_started,
             'derived_bytes':measured_bytes,'geometry_candidate_rooms':house_entry['candidate_room_count'],
             'selected_acoustic_rooms':len(selected),'adapter_materials':result['adapted_materials'],
             'portable_visual':visual,'navmesh':nav,'acoustic_package':str(package) if package else None,
             'resume_enabled':args.resume,'qualified':False,'native_ue_render_and_material_readback':'not_run',
             'native_floor_offset_and_foot_contact':'not_run'}
    write(root/'conversion_result.json',receipt)
    progress(args.output,f"{house}：CPU 转换完成，{measured_bytes:,} 字节；最多 {len(selected)} 间待声学实测。")
    return receipt

def convert(args):
    import traceback
    estimate=json.loads((args.output/'evidence/storage_estimate.json').read_text())
    if estimate['derived_estimate_bytes']>estimate['limit_bytes']:
        raise SystemExit('STOP: initial storage estimate exceeds 50 GB')
    inventories=json.loads((args.output/'inventory.json').read_text())
    selected=[r for r in inventories if not args.house or r['house'] in args.house]
    for row in selected:
        root=args.output/'houses'/row['house']/args.build_name
        if (root/'conversion_result.json').exists():
            if args.resume:continue
            raise FileExistsError('Conversion already exists: '+str(root))
        try:
            receipt=convert_one(args,row)
            print('CONVERTED',json.dumps({k:receipt[k] for k in ['house','wall_seconds','derived_bytes','geometry_candidate_rooms']}),flush=True)
        except Exception as exc:
            write(root/f'conversion_error_{time.time_ns()}.json',{'house':row['house'],'error':repr(exc),'traceback':traceback.format_exc(),
                  'native_visual_status':'not_run','training_admitted':False})
            print(traceback.format_exc(),flush=True)
            raise
        total=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if total>estimate['limit_bytes']:
            raise SystemExit('STOP: measured derived output exceeds 50 GB')

def refresh_materials(args):
    from tools.rooms.prepare_interioragent_kujiale_adapter import prepare_adapter
    from tools.rooms.kujiale_material_translation import expand_stage
    from tools.acoustics.extract_usd_acoustic_snapshot import _extract_stage
    inventories=json.loads((args.output/'inventory.json').read_text())
    for inventory in inventories:
        if args.house and inventory['house'] not in args.house:continue
        house=inventory['house'];root=completed_build(args.output,house,args.build_name)
        if not (root/'conversion_result.json').exists():raise ValueError('Geometry conversion not complete: '+house)
        receipt=root/'material_refresh.json'
        if receipt.exists():
            if args.resume:continue
            raise FileExistsError(receipt)
        total=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if total>50_000_000_000:raise SystemExit('STOP: derived output exceeds 50 GB')
        started=time.monotonic();cpu_started=time.process_time()
        adapter=root/'adapter_expanded';adapter.mkdir(exist_ok=False)
        prepare_adapter(source=Path(inventory['source']),output=adapter/'preview.usda',
            profile_path=root/'adapter/profile.json',result_path=adapter/'receipt.json',replace=False,whole_stage=True)
        records=expand_stage(adapter/'preview.usda',Path(inventory['source']))
        visual=portable_preview(adapter/'preview.usda',root/'engine_expanded',preserve_preview_graphs=True)
        # Material edits must not change a single surface vertex or triangle.
        v,t,*_=_extract_stage(source=root/'engine_expanded/render.usdc',room_id=house,
                              transform_profile='kujiale_z_up_y_back_to_habitat')
        parity=np.array_equal(v[:,[0,2,1]],np.load(root/'surface/vertices.npy')) and np.array_equal(t[:,[0,2,1]],np.load(root/'surface/triangles.npy'))
        if not parity:raise ValueError('Expanded material export changed geometry: '+house)
        write(receipt,{'house':house,'status':'research_only_pending_native_UE','engine_directory':str(root/'engine_expanded'),
              'derived_stage':str(root/'engine_expanded/render.usdc'),'exact_surface_array_parity':parity,
              'wall_seconds':time.monotonic()-started,'driver_cpu_seconds':time.process_time()-cpu_started,
              'visual':visual,'materials':records,'native_render_defect_fraction':None})
        progress(args.output,f"{house}：扩展材质参数、原生 Preview 图保留完成，表面数组逐项相同；原生画面仍待核验。")
        print('MATERIAL_REFRESH',house,time.monotonic()-started,parity,flush=True)

def fix_native_scalars(args):
    import shutil
    from pxr import Usd,UsdShade,Gf
    from tools.rooms.kujiale_material_translation import numeric_plan
    from tools.acoustics.extract_usd_acoustic_snapshot import _extract_stage
    inventories=json.loads((args.output/'inventory.json').read_text())
    for inventory in inventories:
        house=inventory['house']
        if args.house and house not in args.house:continue
        root=completed_build(args.output,house,args.build_name);receipt=root/'native_scalar_fix.json'
        if receipt.exists():
            if args.resume:continue
            raise FileExistsError(receipt)
        started=time.monotonic();cpu_started=time.process_time()
        refresh_path=root/'material_refresh.json';refresh=json.loads(refresh_path.read_text())
        source=Usd.Stage.Open(inventory['source']);path=Path(refresh['derived_stage'])
        backup=path.with_name('render_before_native_scalar_fix.usdc')
        forecast=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())+path.stat().st_size
        if forecast>50_000_000_000:raise SystemExit('STOP: scalar correction backup would exceed 50GB')
        if backup.exists():raise FileExistsError(backup)
        shutil.copy2(path,backup)
        prior=root/'material_refresh_before_native_scalar_fix.json'
        if prior.exists():raise FileExistsError(prior)
        shutil.copy2(refresh_path,prior)
        stage=Usd.Stage.Open(str(path));stage.SetEditTarget(stage.GetRootLayer());changed=[]
        for prim in source.Traverse():
            if not prim.IsA(UsdShade.Material):continue
            original=UsdShade.Material(prim).ComputeSurfaceSource()
            if not original or not original[0]:continue
            shader=original[0];values={i.GetBaseName():i.Get() for i in shader.GetInputs()}
            plan=numeric_plan(values,shader.GetIdAttr().Get(),str(shader.GetSourceAsset('mdl') or ''))
            if plan['family']!='base_v1':continue
            preview=UsdShade.Shader(stage.GetPrimAtPath(prim.GetPath().AppendChild('AVEnginePreviewSurface')))
            if not preview:continue
            edits=[]
            for target,value in plan['constants'].items():
                inp=preview.GetInput(target)
                if not inp:continue
                if inp.GetConnectedSource():continue
                converted=Gf.Vec3f(*value) if isinstance(value,tuple) else float(value)
                if inp.Get()!=converted:
                    inp.Set(converted);edits.append(target)
            node=UsdShade.Shader(stage.GetPrimAtPath(prim.GetPath().AppendChild('AVEngineBaseColorTexture')))
            if node and float(values.get('IsBaseColorTex') or 0.)>.5:
                colour=plan['constants']['diffuseColor'];vector=Gf.Vec4f(*colour,1.)
                for key in ['scale','fallback']:
                    inp=node.GetInput(key)
                    if inp and inp.Get()!=vector:inp.Set(vector);edits.append('base_texture_'+key)
            if edits:changed.append({'material':str(prim.GetPath()),'corrected_inputs':edits,'source_constants':plan['constants']})
        stage.GetRootLayer().Save()
        v,t,*_=_extract_stage(source=path,room_id=house,transform_profile='kujiale_z_up_y_back_to_habitat')
        parity=np.array_equal(v[:,[0,2,1]],np.load(root/'surface/vertices.npy')) and np.array_equal(t[:,[0,2,1]],np.load(root/'surface/triangles.npy'))
        if not parity:raise ValueError('Scalar fix changed surface arrays: '+house)
        result={'house':house,'materials_corrected':len(changed),'changes':changed,'exact_surface_array_parity':True,
                'previous_stage':str(backup),'previous_material_receipt':str(prior),'stage':str(path),
                'wall_seconds':time.monotonic()-started,'driver_cpu_seconds':time.process_time()-cpu_started,
                'texture_files_modified':0,'native_ue_visual_status':'not_run'}
        write(receipt,result)
        refresh['native_scalar_correction']={k:v for k,v in result.items() if k!='changes'}
        refresh['wall_seconds']+=result['wall_seconds'];refresh['driver_cpu_seconds']+=result['driver_cpu_seconds']
        # The prior receipt is retained; only this task's current receipt is updated.
        with refresh_path.open('w') as f:json.dump(refresh,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        progress(args.output,f"{house}：原生 USD 向量标量修正完成，{len(changed)} 个材质，未改纹理，几何数组逐项相同；旧 USD 与 receipt 保留。")
        print('NATIVE_SCALAR_FIX',house,len(changed),result['wall_seconds'],flush=True)

def acceptance(args):
    import argparse
    from tools.acoustics.accept_houses import measure
    baseline=json.loads(args.baseline_manifest.read_text())
    houses=[]
    for p in sorted((args.output/'houses').glob(f'*/{args.build_name}/acceptance_house.json')):
        h=json.loads(p.read_text())
        if h['acoustic_rooms'] and (not args.house or h['house'] in args.house):houses.append(h)
    manifest={k:v for k,v in baseline.items() if k!='houses'}
    manifest['houses']=houses
    manifest['code_base']='34146293f66b2cb2610e4c6f1a4a3613b7390423'
    manifest['selection']={'family':'kujiale','split':'train','maximum_sampled_rooms_per_house':3,
           'method':'same frozen geometry/placement thresholds; choose_rooms from baseline 3414629',
           'native_ue_visual_eligibility':'not_run',
           'material_claim':'shared semantic research assumptions, not measured absorption coefficients'}
    manifest_path=args.output/('acceptance_manifest_'+('_'.join(args.house) if args.house else 'all')+'.json')
    write(manifest_path,manifest)
    result_root=args.output/('acoustics_'+('_'.join(args.house) if args.house else 'all'))
    for h in houses:
        measure(argparse.Namespace(manifest=manifest_path,output=result_root,house=h['house'],
                                  reuse_package=Path(h['acoustic_package_manifest']),omit_degenerate_for_diagnostics=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','convert','acceptance','refresh-materials','fix-native-scalars'])
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--thresholds',type=Path,required=True)
    p.add_argument('--baseline-manifest',type=Path)
    p.add_argument('--material-rules',type=Path)
    p.add_argument('--house',action='append')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--build-name',default='build_v4')
    args=p.parse_args()
    if args.command in {'convert','acceptance'} and args.baseline_manifest is None:p.error('Supply --baseline-manifest')
    if args.command=='convert' and args.material_rules is None:p.error('Supply --material-rules')
    {'audit':audit,'convert':convert,'acceptance':acceptance,'refresh-materials':refresh_materials,'fix-native-scalars':fix_native_scalars}[args.command](args)

if __name__=='__main__':main()
