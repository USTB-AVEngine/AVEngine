"""Materialize one isolated Kujiale map and record native floor/material evidence.

Execute in UE 5.5 PythonScriptCommandlet. The existing USD importer remains
the importer; this wrapper adds measurements without altering source geometry.
"""
from __future__ import annotations
import json, math, os, runpy, time, struct
from pathlib import Path
import unreal


def vector(value):
    return unreal.Vector(*[float(x) for x in value])


def point(value):
    return [float(value.x),float(value.y),float(value.z)]


def floor_trace(world,location,ignore=None):
    top=[location[0],location[1],location[2]+35.]
    bottom=[location[0],location[1],location[2]-80.]
    response=unreal.SystemLibrary.line_trace_single(
        world,vector(top),vector(bottom),unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
        True,ignore or [],unreal.DrawDebugTrace.NONE,True)
    if response is None:
        return {'status':'no_native_collision_hit','start_cm':top,'end_cm':bottom}
    if isinstance(response,(tuple,list)):
        hit,result=response
    else:
        hit,result=True,response
    if not hit:return {'status':'no_native_collision_hit','start_cm':top,'end_cm':bottom}
    broken=unreal.GameplayStatics.break_hit_result(result)
    # Blueprint BreakHitResult names are stable; preserve all returned fields
    # as API evidence and use native HitResult impact point when exposed.
    fields=list(broken) if isinstance(broken,(tuple,list)) else None
    location_value=getattr(result,'impact_point',None)
    if location_value is None:
        # BreakHitResult: blocking_hit, initial_overlap, time, distance,
        # location, impact_point, normal, impact_normal, ...
        if fields is None or len(fields)<8:raise RuntimeError('native HitResult schema unsupported')
        location_value=fields[5]
    return {'status':'hit','impact_point_ue_cm':point(location_value),
            'impact_normal':point(getattr(result,'impact_normal',fields[7] if fields else unreal.Vector())),
            'break_hit_result_field_count':len(fields) if fields else None}


def material_record(material):
    if material is None:return {'status':'missing','path':None,'default_material':True}
    path=material.get_path_name()
    record={'path':path,'class':material.get_class().get_name(),
            'default_material':any(x in path for x in ['WorldGridMaterial','DefaultMaterial','DefaultWhiteGrid','DefaultTexture']),
            'textures':[]}
    base=material
    seen=set()
    while isinstance(base,unreal.MaterialInstance) and base.get_path_name() not in seen:
        seen.add(base.get_path_name());base=base.get_editor_property('parent')
        if base is None:break
    if isinstance(base,unreal.Material):
        record['base_material']=base.get_path_name()
        record['blend_mode']=str(base.get_editor_property('blend_mode'))
        record['two_sided']=bool(base.get_editor_property('two_sided'))
    try:
        names=unreal.MaterialEditingLibrary.get_texture_parameter_names(material)
        for name in names:
            texture=(unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(material,name)
                     if isinstance(material,unreal.MaterialInstance) else
                     unreal.MaterialEditingLibrary.get_material_default_texture_parameter_value(material,name))
            record['textures'].append({'parameter':str(name),'path':texture.get_path_name() if texture else None})
        record['texture_api_status']='pass'
    except Exception as error:
        record['texture_api_status']='unsupported';record['texture_api_error']=str(error)
    record['status']='default' if record['default_material'] else 'material_present'
    return record


start=time.monotonic()
request_path=Path(os.environ['AVENGINE_USD_LEVEL_IMPORT_REQUEST'])
request=json.loads(request_path.read_text())
plan=json.loads(Path(request['acceptance_plan_path']).read_text())
audit_path=Path(request['native_editor_audit_path'])
audit={'schema':'kujiale_native_ue_editor_audit_v1','house':plan['house'],
       'engine_version':unreal.SystemLibrary.get_engine_version(),'status':'started',
       'source_usd':request['source_usd'],'map_path':request['map_path'],
       'floor_reference':[],'materials':[],'mesh_components':[],'errors':[]}
try:
    if request.get('expected_authored_lights') and (request.get('audit_only') or not request.get('correct_native_usd_y_reflection')):
        raise ValueError('new authored-light route requires a fresh map import and explicit native Y adapter')
    if request.get('audit_only'):
        if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(request['map_path']):
            raise RuntimeError('could not load own imported map for audit resume')
    else:
        importer=Path(__file__).with_name('import_usd_stage_to_level_editor.py')
        runpy.run_path(str(importer),run_name='__main__')
    if request.get('base_native_map_path'):
        original=json.loads(Path(request['base_native_editor_audit']).read_text())
        if original['map_path']!=request['base_native_map_path'] or original['house']!=plan['house'] or original['status']!='measured':
            raise ValueError('base native geometry/material provenance differs')
        material_reference=Path(request['geometry_material_reference_usd'])
        if not material_reference.is_file() or material_reference.stem!=Path(original['source_usd']).stem:
            raise ValueError('native material reference must preserve original imported stage namespace')
        audit.update(light_import_source_usd=request['source_usd'],source_usd=str(material_reference),
                     base_native_editor_audit=request['base_native_editor_audit'],
                     base_geometry_source_usd=original['source_usd'],geometry_material_reference_qa_only=True)
    audit['import_script_seconds']=time.monotonic()-start
    world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    actors=unreal.EditorLevelLibrary.get_all_level_actors()
    expected=request.get('expected_bounds_ue_cm')
    if request.get('correct_native_usd_y_reflection'):
        mesh_actors=[a for a in actors if a.get_components_by_class(unreal.StaticMeshComponent)]
        if request.get('expected_authored_lights'):
            # New bake_v3 preserves lamps' complete world matrices. Mirror the
            # entire imported hierarchy once, then reflect emitted light poses
            # explicitly: negative actor scale alone does not mirror light yaw.
            import importlib.util
            helper_path=Path(__file__).resolve().parents[2]/'src/avengine/capture/kujiale_native_lighting.py'
            spec=importlib.util.spec_from_file_location('own_native_light_assertions',helper_path)
            native_light_helpers=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native_light_helpers)
            reflect_native_light_pose=native_light_helpers.reflect_native_light_pose
            light_actors=[a for a in actors if a.get_components_by_class(unreal.LightComponentBase)]
            saved=[]
            for actor in light_actors:
                for component in actor.get_components_by_class(unreal.LightComponentBase):
                    rotation=component.get_world_rotation()
                    saved.append((component,point(component.get_world_location()),
                                  [rotation.pitch,rotation.yaw,rotation.roll],
                                  point(component.get_forward_vector()),point(component.get_up_vector())))
            # A reflected parent has negative determinant. UE FTransform's
            # parent-relative quaternion decomposition cannot represent that
            # reflection for child lights. Keep their original world state,
            # detach only lamp actors, then reflect positions and yaw/roll.
            for actor in light_actors:
                actor.detach_from_actor(unreal.DetachmentRule.KEEP_WORLD,
                                        unreal.DetachmentRule.KEEP_WORLD,
                                        unreal.DetachmentRule.KEEP_WORLD)
            roots=[]
            for actor in ([] if request.get("base_native_map_path") else mesh_actors):
                parent=actor.get_attach_parent_actor()
                while parent is not None:
                    actor=parent;parent=actor.get_attach_parent_actor()
                if actor not in roots:roots.append(actor)
            for actor in roots:
                position=point(actor.get_actor_location());scale=point(actor.get_actor_scale3d())
                actor.set_actor_scale3d(vector([scale[0],-scale[1],scale[2]]))
                actor.set_actor_location(vector([position[0],-position[1],position[2]]),False,True)
            direction_assertions=[]
            for component,position,rotation,forward,up in saved:
                position,rotation=reflect_native_light_pose(position,rotation)
                component.set_world_location_and_rotation(vector(position),unreal.Rotator(pitch=rotation[0],yaw=rotation[1],roll=rotation[2]),False,True)
                observed_position=point(component.get_world_location())
                position_error=max(abs(a-b) for a,b in zip(position,observed_position))/100
                errors=[]
                for before,observed in [(forward,point(component.get_forward_vector())),(up,point(component.get_up_vector()))]:
                    reflected=[before[0],-before[1],before[2]]
                    errors.append(max(abs(a-b) for a,b in zip(reflected,observed)))
                if position_error>.0005 or max(errors)>1e-4:
                    audit['failed_light_pose']={'component':component.get_path_name(),'position_expected':position,
                                                'position_observed':observed_position,'position_error_m':position_error,
                                                'rotation_expected':rotation,'forward_up_errors':errors,
                                                'forward_before':forward,'up_before':up,
                                                'forward_after':point(component.get_forward_vector()),
                                                'up_after':point(component.get_up_vector())}
                    raise ValueError('native authored light reflection readback disagrees')
                direction_assertions.append({'component':component.get_path_name(),
                                             'position_error_m':position_error,'forward_up_max_error':max(errors)})
            audit['light_direction_assertions']=direction_assertions
            audit['light_coordinate_adapter']='original native world light position, yaw and roll mirrored once in Y; no fitted placement'
        else:
            roots=[a for a in mesh_actors if a.get_attach_parent_actor() not in mesh_actors]
            for actor in roots:
                actor.set_actor_scale3d(unreal.Vector(1.,-1.,1.))
                actor.set_actor_location(unreal.Vector(0.,0.,0.),False,True)
        audit['base_native_geometry_reused_without_transform']=bool(request.get('base_native_map_path'))
        audit['native_coordinate_adapter']={
            'source':'USD Z-up right handed; UE USD importer reverses Y',
            'canonical_source_to_world':'source USD [x,y,z] m -> canonical [x,z,y] m',
            'canonical_to_ue':'[x,y,z] m -> UE [100x,100z,100y] cm',
            'import_actor_scale_xyz':[1.,-1.,1.],
            'import_actor_translation_cm':[0.,0.,0.],
            'source_mesh_or_navmesh_modified':False,
            'native_export_vertex_or_ray_assertion_required':True}
        expected=None # No fitting: the declared axis adapter is authoritative.
    if expected:
        mesh_actors=[a for a in actors if a.get_components_by_class(unreal.StaticMeshComponent)]
        bounds=[a.get_actor_bounds(False) for a in mesh_actors]
        lo=[min(point(origin)[i]-point(extent)[i] for origin,extent in bounds) for i in range(3)]
        hi=[max(point(origin)[i]+point(extent)[i] for origin,extent in bounds) for i in range(3)]
        extent_error=max(abs((hi[i]-lo[i])-(expected[1][i]-expected[0][i])) for i in range(3))
        if extent_error>.05:
            raise RuntimeError(f'Native/source extent mismatch {extent_error} cm; cannot repair with a pivot translation')
        delta=[(expected[0][i]+expected[1][i]-lo[i]-hi[i])/2 for i in range(3)]
        roots=[a for a in mesh_actors if a.get_attach_parent_actor() not in mesh_actors]
        for actor in roots:
            old=point(actor.get_actor_location())
            actor.set_actor_location(vector([old[i]+delta[i] for i in range(3)]),False,True)
        audit['import_pivot_translation']={'method':'native bounds versus declared source bounds; translation only',
            'before_bounds_ue_cm':[lo,hi],'expected_bounds_ue_cm':expected,
            'delta_ue_cm':delta,'extent_max_error_cm':extent_error,
            'native_export_vertex_or_ray_assertion_required':True}
    if request.get('expected_authored_lights'):
        assert_authored_light_positions=native_light_helpers.assert_authored_light_positions
        expected_lights=json.loads(Path(request['expected_authored_lights']).read_text())['lights']
        native_lights=[]
        for actor in actors:
            for component in actor.get_components_by_class(unreal.LightComponentBase):
                rotation=component.get_world_rotation()
                record={'component':component.get_path_name(),'actor':actor.get_actor_label(),
                        'component_class':component.get_class().get_name(),
                        'position_ue_cm':point(component.get_world_location()),
                        'rotation_pitch_yaw_roll':[rotation.pitch,rotation.yaw,rotation.roll],
                        'intensity':float(component.get_editor_property('intensity'))}
                for property_name in ['visible','mobility','affects_world','cast_shadows','indirect_lighting_intensity','intensity_units','light_color','temperature','use_temperature',
                                      'source_radius','source_width','source_height','attenuation_radius',
                                      'inner_cone_angle','outer_cone_angle']:
                    try:
                        value=component.get_editor_property(property_name)
                        record[property_name]=value if isinstance(value,(str,bool,int,float)) else str(value)
                    except Exception:pass
                native_lights.append(record)
        audit['light_components']=native_lights
        assertions=assert_authored_light_positions(expected_lights,native_lights)
        lighting={'status':'pass','map_path':request['map_path'],'expected_light_count':len(expected_lights),
                  'light_components':native_lights,'light_position_assertions':assertions,
                  'light_direction_assertions':audit.get('light_direction_assertions',[]),
                  'light_transform_assertions_passed':True,'ad_hoc_fill_count':len(request.get('visual_lights',[])),
                  'native_direction_readbacks_recorded':True,'new_geometry_parity_required':True,
                  'engine_version':audit['engine_version'],'source_usd':request['source_usd']}
        if lighting['ad_hoc_fill_count']:raise ValueError('new authored-light route forbids invented fill lights')
        if request.get('authored_light_sidecar_receipt'):
            sidecar=json.loads(Path(request['authored_light_sidecar_receipt']).read_text())
            size_checks=[]
            for entry in sidecar['lights']:
                name=entry['sidecar_prim'].rsplit('/',1)[-1]
                matches=[c for c in native_lights if c['actor']==name or name in c['component'].split('.')[-1]]
                if len(matches)!=1:raise ValueError('native photometry component match not unique: '+name)
                native=matches[0];photometry=entry['photometry']
                if not photometry.get('world_size_bake_required'):continue
                expected={'intensity':photometry['expected_native_intensity_lumens']}
                for attr,prop in [('inputs:radius','source_radius'),('inputs:width','source_width'),('inputs:height','source_height')]:
                    if attr in photometry['baked_inputs']:expected[prop]=photometry['baked_inputs'][attr]*sidecar['meters_per_unit']*100
                errors={key:abs(float(native[key])-value)/max(abs(value),1.) for key,value in expected.items()}
                if max(errors.values(),default=0)>1e-5:raise ValueError('native world-size/photometry differs from sidecar: '+name+' '+str(errors))
                size_checks.append({'source_prim':entry['source_prim'],'native_component':native['component'],'expected':expected,'readback':{k:native[k] for k in expected},'relative_errors':errors,'status':'pass'})
            lighting['world_photometry_assertions']=size_checks
            lighting['world_photometry_assertions_passed']=True
            lighting['authored_light_sidecar_receipt']=request['authored_light_sidecar_receipt']
        audit['authored_light_import']=lighting
    material_by_path={}
    for actor in actors:
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=component.get_editor_property('static_mesh')
            if mesh is None:continue
            component.set_collision_profile_name('BlockAll')
            component.set_collision_enabled(unreal.CollisionEnabled.QUERY_ONLY)
            record={'actor':actor.get_actor_label(),'component':component.get_path_name(),
                    'mesh':mesh.get_path_name(),'location_ue_cm':point(component.get_world_location()),
                    'material_paths':[]}
            origin,extent=actor.get_actor_bounds(False)
            record.update(bounds_origin_ue_cm=point(origin),bounds_extent_ue_cm=point(extent))
            for slot in range(component.get_num_materials()):
                material=component.get_material(slot)
                m=material_record(material)
                key=m.get('path') or '<missing>'
                material_by_path.setdefault(key,m)
                record['material_paths'].append(key)
            audit['mesh_components'].append(record)
    audit['materials']=list(material_by_path.values())
    # LevelExporterSTL exports BSP only and can succeed with no StaticMesh
    # triangles. Read actual render LOD 0 sections instead; never trust that
    # exporter return value as geometry evidence.
    vertices=[];triangles=[];sections=[]
    for actor in actors:
        for component in actor.get_components_by_class(unreal.StaticMeshComponent):
            mesh=component.get_editor_property('static_mesh')
            if mesh is None:continue
            mesh.set_editor_property('allow_cpu_access',True)
            transform=component.get_world_transform()
            for section in range(mesh.get_num_sections(0)):
                v,t,*_=unreal.ProceduralMeshLibrary.get_section_from_static_mesh(mesh,0,section)
                offset=len(vertices)
                vertices.extend(point(unreal.MathLibrary.transform_location(transform,p)) for p in v)
                triangles.extend(int(i)+offset for i in t)
                material=component.get_material(section)
                sections.append({'component':component.get_path_name(),'section':section,
                    'vertex_start':offset,'vertex_count':len(v),'triangle_count':len(t)//3,
                    'material_path':material.get_path_name() if material else None})
    if not vertices or not triangles:raise RuntimeError('native render LOD 0 geometry is empty')
    prefix=Path(request['native_geometry_prefix'])
    vertex_path=prefix.with_suffix('.vertices.f32');triangle_path=prefix.with_suffix('.triangles.u32')
    with vertex_path.open('wb') as f:
        for vertex in vertices:f.write(struct.pack('<3f',*vertex))
    with triangle_path.open('wb') as f:
        for triangle in triangles:f.write(struct.pack('<I',triangle))
    audit['native_render_geometry']={'method':'ProceduralMeshLibrary.get_section_from_static_mesh render LOD 0',
        'world_vertices_ue_cm_f32':str(vertex_path),'triangles_u32':str(triangle_path),
        'vertex_count':len(vertices),'face_count':len(triangles)//3,'sections':sections}
    for room in plan['rooms']:
        probes=[]
        for i,witness in enumerate(room['witnesses']):
            for role,height in [('camera',1.5),('source_1',1.2),('source_2',1.2)]:
                canonical=list(witness[role+'_m']);canonical[1]-=height
                ue=[canonical[0]*100,canonical[2]*100,canonical[1]*100]
                measured=floor_trace(world,ue)
                measured.update(witness_index=i,role=role,nav_point_m=canonical)
                if measured['status']=='hit':
                    y=measured['impact_point_ue_cm'][2]/100
                    measured.update(native_visual_floor_y_m=y,nav_to_visual_offset_m=y-canonical[1])
                probes.append(measured)
        values=[p['nav_to_visual_offset_m'] for p in probes if p['status']=='hit']
        floor={'room_id':room['room_id'],'method':'native UE complex collision line_trace_single',
               'probes':probes,'probe_count':len(probes),'successful_probe_count':len(values),
               'native_ue_floor_offset_m':sorted(values)[len(values)//2] if values else None,
               'status':'pass' if len(values)==len(probes) and probes else 'fail'}
        # A native Character owns a real UCapsuleComponent. These transient
        # probes are removed before saving; exported source geometry is intact.
        capsule_records=[]
        for probe in probes:
            if probe['status']!='hit':continue
            foot=probe['impact_point_ue_cm']
            actor=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.Character,vector([foot[0],foot[1],foot[2]+75.]),unreal.Rotator())
            if actor is None:
                capsule_records.append({'status':'fail','reason':'native_character_probe_spawn_failed'});continue
            capsule=actor.get_component_by_class(unreal.CapsuleComponent)
            capsule.set_capsule_size(20.,75.,True)
            observed=point(capsule.get_world_location())
            half=float(capsule.get_scaled_capsule_half_height())
            radius=float(capsule.get_scaled_capsule_radius())
            gap=(observed[2]-half-foot[2])/100
            capsule_records.append({'witness_index':probe['witness_index'],'role':probe['role'],
                'radius_cm':radius,'half_height_cm':half,'center_ue_cm':observed,
                'bottom_floor_gap_m':gap,'status':'pass' if abs(gap)<=.005 else 'fail'})
            unreal.EditorLevelLibrary.destroy_actor(actor)
        floor['native_capsule_probes']=capsule_records
        floor['capsule_status']='pass' if len(capsule_records)==len(probes) and all(p['status']=='pass' for p in capsule_records) else 'fail'
        audit['floor_reference'].append(floor)
    audit['map_saved']=bool(unreal.EditorLoadingAndSavingUtils.save_map(world,request['map_path']))
    if not audit['map_saved']:raise RuntimeError('own native map could not be saved')
    audit['status']='measured'
    if request.get('native_lighting_audit_path'):
        if not audit.get('authored_light_import'):raise ValueError('cannot write an unmeasured lighting audit')
        with Path(request['native_lighting_audit_path']).open('x') as stream:
            json.dump(audit['authored_light_import'],stream,indent=2);stream.write('\n')
except Exception as error:
    audit['status']='failed';audit['errors'].append(f'{type(error).__name__}: {error}')
    raise
finally:
    audit['elapsed_python_seconds']=time.monotonic()-start
    audit_path.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    unreal.log_warning('AVENGINE_NATIVE_EDITOR_AUDIT '+audit['status'])
