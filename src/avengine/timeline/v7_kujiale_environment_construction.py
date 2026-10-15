"""Construct required native animals/objects without a scene-specific plan."""
from __future__ import annotations
from copy import deepcopy
from collections import Counter
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

class NativeSupportProposalCache:
    """Original placement anchor math, cached before expensive native queries."""
    def __init__(self,vertices):self.vertices=np.asarray(vertices,float);self.rotations={}
    def pose(self,support,yaw):
        from avengine.qa.kujiale_asset_geometry import rotation
        normal=np.asarray(support['normal'],float);key=(float(yaw),tuple(normal))
        if key not in self.rotations:
            matrix=rotation(yaw);local_n=matrix.T@normal;values=self.vertices@local_n;extreme=float(values.min());band=min(.002,.05*float(np.ptp(values)));contact=self.vertices[values<=extreme+band+1e-9];anchor=(contact.min(axis=0)+contact.max(axis=0))/2;anchor+=local_n*(extreme-float(anchor@local_n));rotated=self.vertices@matrix.T
            self.rotations[key]=(matrix,rotated,anchor,rotated.min(axis=0),rotated.max(axis=0))
        matrix,rotated,anchor,low,high=self.rotations[key];root=np.asarray(support['point_m'],float)-matrix@anchor
        return matrix,rotated,root,low+root,high+root


def construct_environment(row,house,inputs,classes,*,max_assets=32,max_supports=1600):
    from avengine.timeline.v7_kujiale_environment_acoustics import EnvironmentPositionAdmission
    admission=EnvironmentPositionAdmission(row,house,inputs)
    try:return _construct_environment(row,house,inputs,classes,max_assets=max_assets,max_supports=max_supports,acoustic_admission=admission)
    finally:admission.close()


def _construct_environment(row,house,inputs,classes,*,max_assets,max_supports,acoustic_admission):
    from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry,import_vertices,rotation,camera_candidates
    from avengine.capture.kujiale_source_floor_paws import PreparedPawContacts
    inventory=json.loads(Path(inputs['kujiale_ue_asset_admission']).read_text());geometry=NativeSupportGeometry(house['input']['ue_config']['native_floor_surface']);room=next(r for r in house['checked_rooms']if r['room_label']==row['room_id']);catalog,polygon=geometry.support_catalog(house,room);listener=np.asarray(row['measured_static_layout']['listener_m'],float);envelopes=[];plans=[]
    for point in row['measured_static_layout']['valid'][:3]:
        root=np.asarray(point['root_m'],float);envelopes.append((root+[-.4,0,-.4],root+[.4,1.9,.4]))
    for cls,candidates in native_class_candidates(inventory,classes,house['input']['ue_config']['uproject'],row['scene_seed']+11*int(row.get('construction_request',{}).get('construction_attempt',0))):
        accepted=None;rejections=[]
        # First keep the planned view, then try the smallest actual registered
        # bodies on every legal support outside the fixed view (Claude ddce1120).
        from avengine.capture.v7_offscreen_layout import placement_construction_fallback
        requested_view=row.get('construction_request',{}).get('source_view_conditions',{}).get(cls)
        if requested_view not in {None,'inside','outside'}:raise ValueError('unknown requested physical source view')
        stages=[] if requested_view=='outside' else [(False,a) for a in candidates[:max_assets]]
        if requested_view!='inside':stages += [(True,a) for a in sorted(candidates,key=lambda a:(max(a['dimensions_m']),a['asset_id']))]
        for offscreen,asset in stages:
            fallback_scene=deepcopy(row)
            declaration={'asset_id':asset['asset_id'],'endpoint_id':'environment'+str(len(plans)+1),
                         'kind':'animal' if asset['category'] in {'dog','cat'} else 'object'}
            if offscreen and placement_construction_fallback(fallback_scene,declaration,asset['attachment_surface'],
                    row.get('native_sector_targets',{}).get(asset['asset_id']),False)!='offscreen':continue
            print('AUTO_NATIVE_ASSET_CANDIDATE',row['house'],row['room_id'],cls,asset['asset_id'],flush=True)
            primary=json.loads(Path(asset['primary_native_evidence']).read_text());manifest_path=Path(primary['input_manifest']);manifest=json.loads(manifest_path.read_text());metadata=next(a for a in manifest['assets']if a['asset_id']==asset['asset_id']);import_path=manifest_path.parent.parent/'evidence/import'/manifest['revision']/(asset['asset_id']+'.json');vertices=import_vertices(json.loads(import_path.read_text()));error=np.abs(np.ptp(vertices,axis=0)-np.asarray(asset['dimensions_m']))/np.maximum(asset['dimensions_m'],1e-5)
            if error.max()>.15:raise ValueError('admitted native size changed before construction')
            prepared_paws=PreparedPawContacts(metadata,vertices);support_poses=NativeSupportProposalCache(vertices);rejected=Counter()
            tolerance=.012 if asset['category']!='object'else .015
            # Nearer offscreen animals separate the floor reflection from the
            # unchanged 0.5ms direct-path check. This ranks, never admits, a pose.
            preferred_distance=1.5 if offscreen and asset['category'] in {'dog','cat'} else 3.1
            supports=sorted(catalog[asset['attachment_surface']],key=lambda q:abs(np.linalg.norm((np.asarray(q['point_m'])-listener)[[0,2]])-preferred_distance))
            for support in (supports if offscreen else supports[:max_supports]):
                n=np.asarray(support['normal']);angles=[math.atan2(-n[2],n[0])]if asset['attachment_surface']=='wall'else[0.,math.pi/2,math.pi,-math.pi/2]
                for yaw in angles:
                    matrix,rotated,proposed_root,low,high=support_poses.pose(support,yaw)
                    if not _separated(low,high,envelopes):rejected['actual_human_or_required_entity_envelope_prefilter']+=1;continue
                    center=(low+high)/2;ray=center-listener;length=np.linalg.norm(ray);horizontal=np.linalg.norm(ray[[0,2]])
                    if any(np.linalg.norm((center-np.asarray(p['root_m']))[[0,2]])<1.0 for p in row['measured_static_layout']['valid']):rejected['owner_human_animal_center_spacing']+=1;continue
                    if horizontal<.2:rejected['source_coincides_with_listener']+=1;continue
                    if not offscreen and (abs(math.degrees(math.atan2(ray[1],horizontal)))>23 or not _clear_segment(listener,center,envelopes)):rejected['listener_view_or_human_separation']+=1;continue
                    from avengine.capture.kujiale_pilot_environment import binding_camera_yaw
                    yaw_center=row['measured_static_layout'].get('fixed_camera_yaw_deg',binding_camera_yaw(listener,[p['position_m']for p in row['measured_static_layout']['valid'][:2]]))
                    bearing=math.degrees(math.atan2(ray[0],-ray[2]));offset=(bearing-yaw_center+180)%360-180
                    if offscreen:
                        if abs(offset)<60:rejected['offscreen_bearing_below_60_degrees']+=1;continue
                    else:
                        if abs(offset)>35:rejected['required_fixed_camera_span']+=1;continue
                        if geometry.ray(listener,ray/length,length-.04)is not None:rejected['native_camera_to_entity_ray']+=1;continue
                    place=geometry.placement(vertices,support,yaw,tolerance)
                    if place is None:rejected['original_native_support_contact_collision_or_negative_control']+=1;continue
                    place['construction_support_pose_cache']={'root_max_abs_error_m':float(np.max(np.abs(proposed_root-np.asarray(place['root_m'])))),'original_native_contact_collision_and_negative_control_rerun':True}
                    paws=prepared_paws.measure(geometry,place)
                    if asset['category']in ['dog','cat']and paws['status']!='pass':rejected['original_four_native_paw_12mm_contact']+=1;continue
                    views=[]
                    for view in camera_candidates(vertices,place,polygon):
                        eye=np.asarray(view['position_m']);delta=np.asarray(view['target_m'])-eye;distance=np.linalg.norm(delta)
                        if geometry.ray(eye,delta/distance,distance-.02)is None:views.append(view)
                    if len(views)<3:rejected['three_independent_clear_native_views']+=1;continue
                    point=np.asarray(place['root_m'])+matrix@np.asarray(asset['emitter_local_m'])
                    acoustic=acoustic_admission.check('environment'+str(len(plans)+1),point)
                    if acoustic['status']=='blocked':rejected['unchanged_5_degree_native_acoustic_direction']+=1;continue
                    place['construction_acoustic_admission']=acoustic
                    place.update(paw_contacts=paws,camera_candidates=views,native_world_bounds_m=[low.tolist(),high.tolist()]);point=np.asarray(place['root_m'])+matrix@np.asarray(asset['emitter_local_m']);accepted={'asset_id':asset['asset_id'],'sound_class':cls,'silent_decor':False,'placement':place,'ue_asset_paths':asset['ue_asset_paths'],'size':{'status':'pass','native_dimensions_m':np.ptp(vertices,axis=0).tolist(),'relative_errors':error.tolist(),'tolerance_fraction':.15},'expected_emitter_m':point.tolist(),'import_receipt':str(import_path),'status':'pass_geometry_requires_current_native_scene'};envelopes.append((low,high));break
                if accepted:break
            if accepted:break
            rejections.append({'asset_id':asset['asset_id'],'support_candidates':min(len(supports),max_supports),'rejections':dict(rejected)});print('AUTO_NATIVE_ASSET_REJECTED',json.dumps(rejections[-1],sort_keys=True),flush=True)
        if not accepted:raise ValueError('bounded native physical construction found no support for required class '+cls+': '+json.dumps(rejections,sort_keys=True))
        if offscreen:
            accepted['intentional_offscreen_role']=fallback_scene['native_view_roles'][asset['asset_id']]
            accepted['construction_fallbacks']=fallback_scene['construction_fallbacks']
            row.setdefault('construction_fallbacks',[]).extend(accepted['construction_fallbacks'])
        print('AUTO_NATIVE_PHYSICAL_PLAN',row['house'],row['room_id'],accepted['asset_id'],accepted['placement']['root_m'],flush=True)
        plans.append(accepted)
    return plans

def separate_environment_schedule(row,semantic,*,effect_duration_s=2.):
    """Fit complete authored speech before effects, then observe the real body."""
    from avengine.timeline.v7_kujiale_construction import authored_pair
    plan=row['production_scenario'];gender='female'if plan['gender']=='F'else'male';scenario,sounds,offsets=authored_pair(semantic,gender,row['scene_seed'],latest_end_s=4.8);byid={s['sound_asset_id']:s for s in semantic['sounds']};end=max(offset+byid[sid]['synthesis_readback']['duration_s']for sid,offset in zip(sounds,offsets));effect_start=math.ceil((end+.5)*10)/10;plan.update(semantic_scenario_id=scenario,semantic_sound_ids=sounds,semantic_sound_subset_ids=sounds,semantic_role_start_offsets_s=offsets,environment_start_offsets_s=[effect_start]);return end,effect_start
