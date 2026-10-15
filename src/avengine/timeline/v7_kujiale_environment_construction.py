"""Construct required native animals/objects without a scene-specific plan."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,os
import numpy as np


def native_class_candidates(inventory,classes,project,seed):
    """Admitted class, native ID, original packages; no species substitution."""
    content=Path(project).parent/'Content';result=[]
    for cls in classes:
        candidates=[a for a in inventory['assets']if a.get('ue_native_physical_status')=='pass'and a.get('ue_import_status')=='pass'and not a.get('silent_decor')and cls in(a.get('allowed_v7_classes_candidate')or[])and a.get('ue_asset_paths')and all((content/(p.partition('.')[0][6:]+'.uasset')).is_file()for p in a['ue_asset_paths'])]
        candidates.sort(key=lambda a:(max(a['dimensions_m']),a['asset_id']))
        if not candidates:raise ValueError('no admitted native physical asset for required class '+cls)
        offset=seed%len(candidates);result.append((cls,candidates[offset:]+candidates[:offset]))
    return result

def bind_inventory_project(inputs,house,output):
    """A runtime view keeps the source registration and original package bytes."""
    from avengine.registry.kujiale_ue_measurements import sha256
    source=Path(inputs['kujiale_ue_asset_admission']);inventory=json.loads(source.read_text());project=Path(house['input']['ue_config']['uproject']);bound=deepcopy(inventory)
    for asset in bound['assets']:
        asset['original_registered_ue_project']=asset['ue_project'];asset['ue_project']=str(project)
    bound['runtime_project_binding']={'source_registration':str(source),'source_sha256':sha256(source),'native_package_identity':'original registered object paths and bytes; actual joint UE readback still mandatory','original_registration_modified':False}
    target=Path(output)/'inputs/native_runtime_bindings';target.mkdir(parents=True,exist_ok=True);path=target/('inventory_'+sha256(source)[:16]+'.json')
    if path.exists():
        if json.loads(path.read_text())!=bound:raise ValueError('native runtime binding differs; use a new declared runtime root')
    else:
        temporary=path.with_name(path.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(bound,indent=2)+'\n');temporary.replace(path)
    updated=deepcopy(inputs);updated['kujiale_ue_asset_admission']=str(path);return updated

def _separated(low,high,envelopes,margin=.025):return all(np.any(np.minimum(high,bhi)-np.maximum(low,blo)<-margin)for blo,bhi in envelopes)

def _clear_segment(origin,target,envelopes):
    direction=target-origin
    for low,high in envelopes:
        enter,leave=0.,1.
        for axis in range(3):
            if abs(direction[axis])<1e-12:
                if not low[axis]<=origin[axis]<=high[axis]:enter,leave=1.,0.;break
            else:
                a=(low[axis]-origin[axis])/direction[axis];b=(high[axis]-origin[axis])/direction[axis];enter=max(enter,min(a,b));leave=min(leave,max(a,b))
        if enter<=leave and 0<leave and enter<1:return False
    return True

def construct_environment(row,house,inputs,classes,*,max_assets=32,max_supports=1600):
    from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry,import_vertices,rotation,camera_candidates
    from avengine.qa.kujiale_asset_paws import landmark_contacts
    inventory=json.loads(Path(inputs['kujiale_ue_asset_admission']).read_text());geometry=NativeSupportGeometry(house['input']['ue_config']['native_floor_surface']);room=next(r for r in house['checked_rooms']if r['room_label']==row['room_id']);catalog,polygon=geometry.support_catalog(house,room);listener=np.asarray(row['measured_static_layout']['listener_m'],float);envelopes=[];plans=[]
    for point in row['measured_static_layout']['valid'][:3]:
        root=np.asarray(point['root_m'],float);envelopes.append((root+[-.4,0,-.4],root+[.4,1.9,.4]))
    for cls,candidates in native_class_candidates(inventory,classes,house['input']['ue_config']['uproject'],row['scene_seed']):
        accepted=None;rejections=[]
        for asset in candidates[:max_assets]:
            primary=json.loads(Path(asset['primary_native_evidence']).read_text());manifest_path=Path(primary['input_manifest']);manifest=json.loads(manifest_path.read_text());metadata=next(a for a in manifest['assets']if a['asset_id']==asset['asset_id']);import_path=manifest_path.parent.parent/'evidence/import'/manifest['revision']/(asset['asset_id']+'.json');vertices=import_vertices(json.loads(import_path.read_text()));error=np.abs(np.ptp(vertices,axis=0)-np.asarray(asset['dimensions_m']))/np.maximum(asset['dimensions_m'],1e-5)
            if error.max()>.15:raise ValueError('admitted native size changed before construction')
            tolerance=.012 if asset['category']!='object'else .015;supports=sorted(catalog[asset['attachment_surface']],key=lambda q:abs(np.linalg.norm((np.asarray(q['point_m'])-listener)[[0,2]])-3.1))
            for support in supports[:max_supports]:
                n=np.asarray(support['normal']);angles=[math.atan2(-n[2],n[0])]if asset['attachment_surface']=='wall'else[0.,math.pi/2,math.pi,-math.pi/2]
                for yaw in angles:
                    matrix=rotation(yaw);rotated=vertices@matrix.T;rough=np.asarray(support['point_m'])-(rotated.min(axis=0)+rotated.max(axis=0))/2
                    if not _separated(rotated.min(axis=0)+rough,rotated.max(axis=0)+rough,envelopes):continue
                    place=geometry.placement(vertices,support,yaw,tolerance)
                    if place is None:continue
                    world=rotated+np.asarray(place['root_m']);low,high=world.min(axis=0),world.max(axis=0);center=(low+high)/2;ray=center-listener;length=np.linalg.norm(ray);horizontal=np.linalg.norm(ray[[0,2]])
                    if not _separated(low,high,envelopes)or horizontal<.2 or abs(math.degrees(math.atan2(ray[1],horizontal)))>23 or not _clear_segment(listener,center,envelopes):continue
                    from avengine.capture.kujiale_pilot_environment import binding_camera_yaw
                    yaw_center=binding_camera_yaw(listener,[p['position_m']for p in row['measured_static_layout']['valid'][:2]])
                    bearing=math.degrees(math.atan2(ray[0],-ray[2]));offset=(bearing-yaw_center+180)%360-180
                    if abs(offset)>110:continue
                    if geometry.ray(listener,ray/length,length-.04)is not None:continue
                    paws=landmark_contacts(geometry,metadata,vertices,place)
                    if asset['category']in ['dog','cat']and paws['status']!='pass':continue
                    views=[]
                    for view in camera_candidates(vertices,place,polygon):
                        eye=np.asarray(view['position_m']);delta=np.asarray(view['target_m'])-eye;distance=np.linalg.norm(delta)
                        if geometry.ray(eye,delta/distance,distance-.02)is None:views.append(view)
                    if len(views)<3:continue
                    place.update(paw_contacts=paws,camera_candidates=views,native_world_bounds_m=[low.tolist(),high.tolist()]);point=np.asarray(place['root_m'])+matrix@np.asarray(asset['emitter_local_m']);accepted={'asset_id':asset['asset_id'],'sound_class':cls,'silent_decor':False,'placement':place,'ue_asset_paths':asset['ue_asset_paths'],'size':{'status':'pass','native_dimensions_m':np.ptp(vertices,axis=0).tolist(),'relative_errors':error.tolist(),'tolerance_fraction':.15},'expected_emitter_m':point.tolist(),'import_receipt':str(import_path),'status':'pass_geometry_requires_current_native_scene'};envelopes.append((low,high));break
                if accepted:break
            if accepted:break
        if not accepted:raise ValueError('bounded native physical construction found no support for required class '+cls)
        plans.append(accepted)
    return plans

def separate_environment_schedule(row,semantic,*,effect_duration_s=2.):
    """Fit complete authored speech before effects, then observe the real body."""
    from avengine.timeline.v7_kujiale_construction import authored_pair
    plan=row['production_scenario'];gender='female'if plan['gender']=='F'else'male';scenario,sounds,offsets=authored_pair(semantic,gender,row['scene_seed'],latest_end_s=4.8);byid={s['sound_asset_id']:s for s in semantic['sounds']};end=max(offset+byid[sid]['synthesis_readback']['duration_s']for sid,offset in zip(sounds,offsets));effect_start=math.ceil((end+.5)*10)/10;plan.update(semantic_scenario_id=scenario,semantic_sound_ids=sounds,semantic_sound_subset_ids=sounds,semantic_role_start_offsets_s=offsets,environment_start_offsets_s=[effect_start]);return end,effect_start
