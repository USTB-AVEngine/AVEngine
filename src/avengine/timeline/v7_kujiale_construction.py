"""Construct registered binding quartets in any admitted native Kujiale room.

No room-ID switch, hand picked position, voice, actor or event offset is used.
Geometry proposals remain requests until all original native and QA gates pass.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib,itertools,json,math
import numpy as np

REVISION='kujiale-fixed-natural-construction-v3-complete-fixed-view'
FACTORS=('appearance','audio','camera','event_schedule','pose','questions','trajectory','window')

def factor_seeds(seed):return {key:int.from_bytes(hashlib.sha256(f'{seed}/{key}'.encode()).digest()[:8],'big')for key in FACTORS}

def authored_pair(manifest,gender,seed,*,latest_end_s=9.):
    """Choose distinct whole meanings/voices, then stagger by actual duration."""
    groups=defaultdict(list)
    for sound in manifest['sounds']:
        if sound['voice_reference']['gender']==gender:groups[sound['scenario_id']].append(sound)
    options=[]
    for scenario,sounds in sorted(groups.items()):
        for a,b in itertools.combinations(sounds,2):
            if a['role_id']==b['role_id']or a['voice_reference']['speaker_id']==b['voice_reference']['speaker_id']:continue
            first=.5;second=first+float(a['synthesis_readback']['duration_s'])+.3
            if second>7 or second+float(b['synthesis_readback']['duration_s'])>latest_end_s:continue
            options.append((scenario,[a['sound_asset_id'],b['sound_asset_id']],[first,round(second,6)]))
    if not options:raise ValueError('no complete staggered authored pair fits the requested native scene')
    return options[seed%len(options)]

def registered_people(catalog,registration,seed):
    allowed={a['asset_id']for a in registration['assets']};pool=[a for a in catalog if a['asset_id']in allowed and a.get('dialogue_voice_binding_eligible') and a.get('registered_native_ue')]
    pairs=[]
    for a,b in itertools.combinations(pool,2):
        if a['visual_gender']==b['visual_gender']and a['body_identity_id']==b['body_identity_id']and a['color']!=b['color']:
            other=[c for c in pool if c['color']not in {a['color'],b['color']}]
            if other:pairs.append((a,b,other))
    if not pairs:raise ValueError('formal native registration needs two same-type distinguishable speakers and a silent different appearance')
    a,b,other=pairs[seed%len(pairs)]
    if seed%2:a,b=b,a
    c=sorted(other,key=lambda x:x['asset_id'])[(seed//2)%len(other)]
    return [a['asset_id'],b['asset_id'],c['asset_id']],a['visual_gender']

def _preferred_group(candidates,listener,attempt=0):
    from avengine.capture.kujiale_pilot_environment import binding_pair_fits_view,comparison_person_candidates,construct_dialogue_camera
    listener=np.asarray(listener,float);candidates=[p for p in candidates if np.linalg.norm((np.asarray(p['root_m'])-listener)[[0,2]])>=1.5-1e-9];ordered=sorted(candidates,key=lambda p:(abs(np.linalg.norm((np.asarray(p['position_m'])-listener)[[0,2]])-2.4),p['position_m']))
    if len(ordered)<3:raise ValueError('no clear separated three-person group')
    roots=np.asarray([p['root_m']for p in ordered])[:,[0,2]];rays=np.asarray([p['position_m']for p in ordered])-listener;radii=np.linalg.norm(rays[:,[0,2]],axis=1);bearings=np.degrees(np.arctan2(rays[:,0],-rays[:,2]));distance=np.linalg.norm(roots[:,None,:]-roots[None,:,:],axis=2);angles=np.abs((bearings[:,None]-bearings[None,:]+180)%360-180)
    compatible=(distance>=1.0-1e-9)&(np.abs(radii[:,None]-radii[None,:])>=.6-1e-9)&(angles>=8.-1e-7);bits=[int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little')for row in compatible];pairs=(distance>=1.0-1e-9)&(angles>=16.-1e-7)&(angles<=60.+1e-7)
    # A failed appearance may belong to either speaker. Prefer a completely
    # new speaking pair before reusing a location; never require extra room
    # clearance or remove a person when only overlapping alternatives exist.
    used_pairs=set();used_roots=set();last_legal=None
    for choice in range(attempt+1):
        best=None;best_shared=3
        for i,first in enumerate(ordered):
            for j in np.flatnonzero(pairs[i,i+1:])+i+1:
                if (i,int(j))in used_pairs:continue
                shared=int(i in used_roots)+int(j in used_roots)
                if shared>=best_shared:continue
                common=bits[i]&bits[j]
                if not common:continue
                second=ordered[j]
                if np.linalg.norm(roots[i]-roots[j])<1.0 or not binding_pair_fits_view(listener,first['position_m'],second['position_m']):continue
                indices=[]
                while common:
                    lowest=common&-common;index=lowest.bit_length()-1;common-=lowest
                    if min(np.linalg.norm(roots[index]-roots[k])for k in [i,j])>=1.0:
                        axis=roots[j]-roots[i];delta=roots[index]-roots[i];side=abs(axis[0]*delta[1]-axis[1]*delta[0])/np.linalg.norm(axis)
                        if side>=.25:indices.append(index)
                other=[ordered[k]for k in indices];thirds=comparison_person_candidates(listener,[first['position_m'],second['position_m']],[c['position_m']for c in other])
                if not thirds:continue
                for third in thirds:
                    try:construct_dialogue_camera(listener,[first['position_m'],second['position_m']],additional_positions=[third])
                    except ValueError:continue
                    c=next(c for c in other if c['position_m']==third);best=(i,int(j),[deepcopy(first),deepcopy(second),deepcopy(c)]);best_shared=shared
                    break
                else:continue
                if shared==0:break
            if best_shared==0:break
        if best is None:
            if last_legal is not None:return last_legal
            raise ValueError('no clear separated three-person group for this room/camera construction attempt')
        if choice==attempt:return best[2]
        last_legal=best[2];used_pairs.add(best[:2]);used_roots.update(best[:2])
    raise ValueError('no clear separated three-person group for this room/camera construction attempt')

def select_group(candidates,listener,attempt=0):
    """Try original sampling margins, then construct without treating them as QA gates."""
    try:return _preferred_group(candidates,listener,attempt)
    except ValueError as preferred_error:
        from avengine.capture.kujiale_pilot_environment import construct_dialogue_camera
        listener=np.asarray(listener,float)
        for minimum in (1.5,1.2):
            pool=[p for p in candidates if np.linalg.norm((np.asarray(p['root_m'])-listener)[[0,2]])>=minimum-1e-9]
            if len(pool)<3:continue
            roots=np.asarray([p['root_m']for p in pool])[:,[0,2]]
            distances=np.linalg.norm(roots[:,None,:]-roots[None,:,:],axis=2)
            radii=np.linalg.norm(np.asarray([p['position_m']for p in pool])-listener,axis=1)
            for offset in range(len(pool)):
                i=(offset+int(attempt))%len(pool)
                for j in np.flatnonzero(distances[i]>=1.-1e-9):
                    thirds=np.flatnonzero((distances[i]>=1.-1e-9)&(distances[j]>=1.-1e-9))
                    thirds=sorted(thirds,key=lambda k:(-min(abs(radii[k]-radii[i]),abs(radii[k]-radii[j])),int(k)))
                    for k in thirds:
                        axis=roots[j]-roots[i];delta=roots[k]-roots[i]
                        if abs(axis[0]*delta[1]-axis[1]*delta[0])/np.linalg.norm(axis)<.25:continue
                        group=[pool[int(n)]for n in (i,j,k)]
                        try:construct_dialogue_camera(listener,[p['position_m']for p in group[:2]],additional_positions=[group[2]['position_m']])
                        except ValueError:continue
                        selected=deepcopy(group)
                        for point in selected:point['group_construction_preference_fallback']={'sampling_margins_are_not_qa_gates':True,'camera_minimum_m':minimum,'native_admission_required':True}
                        return selected
        raise preferred_error


def visible_candidates(mesh,listener,candidates):
    if not candidates:return []
    roots=np.asarray([p['root_m']for p in candidates]);targets=(roots[:,None,:]+np.array([[0.,.9,0.],[0.,1.2,0.],[0.,1.5,0.]])[None,:,:]).reshape(-1,3)
    origins=np.tile(listener,(len(targets),1));rays=targets-origins;lengths=np.linalg.norm(rays,axis=1);hits,indices,_=mesh.ray.intersects_location(origins,rays/lengths[:,None],multiple_hits=False);blocked=np.zeros(len(targets),dtype=bool)
    if len(hits):blocked[indices]=np.linalg.norm(hits-origins[indices],axis=1)<lengths[indices]-.03
    blocked=blocked.reshape(-1,3);radii=np.linalg.norm((roots-listener)[:,[0,2]],axis=1);order=sorted(range(len(candidates)),key=lambda i:(int(blocked[i].sum()),abs(radii[i]-2.4)))
    # This is a proposal filter, not admission. A clear chest ray can still
    # yield valid native appearance with a partially occluded head/legs.
    return [candidates[i]for i in order if not blocked[i,1]and 1.2<=radii[i]<=10.]


def air_camera_proposals(polygon,native_floor_y,step,eye_height_m=1.5):
    from shapely.geometry import Point
    x0,z0,x1,z1=polygon.bounds
    return [[float(x),float(native_floor_y)+float(eye_height_m),float(z)]for x in np.arange(x0+.05,x1,step)for z in np.arange(z0+.05,z1,step)if polygon.covers(Point(x,z))]


def construction_search_steps(attempt):
    """A failed native layout advances to a denser grid, rather than repeating .5 m."""
    levels=(.5,.35,.25,.15,.1)
    return levels[min(max(int(attempt),0)//3,len(levels)-1):]


def construction_camera_batch(views,attempt):
    """Rotate through the complete legal camera pool; no permanent 128-point subset."""
    if not views:return []
    count=min(len(views),128*(1+max(int(attempt),0)//6))
    start=(max(int(attempt),0)//3*128)%len(views)
    return [views[(start+i)%len(views)] for i in range(count)]


def construct_layout(house,room_id,runtime,*,attempt=0,eye_height_m=1.5):
    from shapely.geometry import shape,Point
    from avengine.acoustics.runtime import load_habitat_runtime
    from avengine.capture.native_floor_reference import NativeFloorReference
    from avengine.capture.kujiale_spear_runtime import to_ue
    from tools.rooms.check_kujiale_native_capsule_contact import query_capsule
    room=next(r for r in house['checked_rooms']if r['room_label']==room_id);declaration=next(r for r in json.loads(Path(house['input']['ue_config']['acceptance_plan_path']).read_text())['rooms']if r['room_id']==room_id)
    polygon=shape(declaration['floor_polygon']).buffer(-.05);floor=NativeFloorReference(house['input']['ue_config']['native_floor_surface']);floor.validate_probes(room['floor_reference']);proposal_floor_y=floor.room_floor_y_m if floor.room_floor_y_m is not None else floor.proposal_floor_y_m;hs,_=load_habitat_runtime(runtime_mode='current-installed',**runtime['runtime']);pf=hs.PathFinder()
    if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('native room NavMesh unavailable')
    x0,z0,x1,z1=polygon.bounds;seen=set();all_candidates=[];statistics={'proposed':0,'nav_rejected':0,'native_floor_rejected':0,'capsule_rejected':0};native_probes=[p['nav_point_m']for p in room['floor_reference']['probes']if p['role'].startswith('source')]
    for step in construction_search_steps(attempt):
        proposals=native_probes+[[float(x),proposal_floor_y,float(z)]for x in np.arange(x0+.05,x1,step)for z in np.arange(z0+.05,z1,step)if polygon.covers(Point(x,z))]
        for target in proposals:
            key=(round(target[0],4),round(target[2],4))
            if key in seen:continue
            seen.add(key);statistics['proposed']+=1;target=np.asarray(target,float);nav=np.asarray(pf.snap_point(target),float)
            if not np.isfinite(nav).all()or np.linalg.norm(nav-target)>.15:statistics['nav_rejected']+=1;continue
            try:root=np.asarray(floor.sample(nav),float)
            except ValueError:statistics['native_floor_rejected']+=1;continue
            cap=query_capsule(floor.mesh,{'witness_index':statistics['proposed'],'role':'source','capsule':{'radius_cm':20.,'half_height_cm':75.,'center_ue_cm':to_ue(root+[0,.75,0])}})
            if cap['status']!='pass':statistics['capsule_rejected']+=1;continue
            all_candidates.append({'position_m':(root+[0,1.3,0]).tolist(),'root_m':root.tolist(),'capsule_check':cap})
        views=[room['placement']['camera_m'],*[w['placement']['camera_m']for w in room.get('native_witnesses',[])]]
        if eye_height_m!=1.5:
            lowered=[]
            for eye in views:
                eye=[eye[0],proposal_floor_y+eye_height_m,eye[2]]
                cap=query_capsule(floor.mesh,{'witness_index':0,'role':'camera','capsule':{'radius_cm':25.,'half_height_cm':25.,'center_ue_cm':to_ue(eye)}})
                if cap['status']=='pass':lowered.append(eye)
            views=lowered
        # Add the same legal room's native-navigation camera positions. The
        # old witness often offers three nearly identical collinear views.
        camera_eyes=[(np.asarray(p['root_m'])+[0,eye_height_m,0]).tolist()for p in all_candidates]+air_camera_proposals(polygon,proposal_floor_y,step,eye_height_m)
        for eye in camera_eyes:
            cap=query_capsule(floor.mesh,{'witness_index':0,'role':'camera','capsule':{'radius_cm':25.,'half_height_cm':25.,'center_ue_cm':to_ue(eye)}})
            if cap['status']=='pass':views.append(list(eye))
        views=list({tuple(v):v for v in views}.values())
        complete_camera_count=len(views);views=construction_camera_batch(views,attempt)
        print('AUTO_ROOM_GEOMETRY',house['house'],room_id,{'grid_step':step,'native_candidates':len(all_candidates),'cameras':len(views),'complete_camera_pool':complete_camera_count,'rejections':statistics},flush=True)
        failures=[]
        for camera_index in range(len(views)):
            listener=np.asarray(views[(camera_index+attempt//3)%len(views)],float);candidates=visible_candidates(floor.mesh,listener,all_candidates)
            for candidate in candidates:
                p=np.asarray(candidate['position_m']);candidate['body_yaw_deg']=math.degrees(math.atan2(listener[0]-p[0],-(listener[2]-p[2])))
            try:selected=select_group(candidates,listener,attempt%3)
            except ValueError:failures.append({'camera_m':listener.tolist(),'clear_candidates':len(candidates)});continue
            return {'house':house['house'],'room_id':room_id,'status':'CPU native geometry constructed; complete UE readback required','construction_revision':REVISION,'construction_attempt':attempt,'listener_m':listener.tolist(),'center_yaw_deg':0.,'valid':selected,'native_candidates':[p['position_m']for p in candidates],'candidate_count':statistics['proposed'],'clear_candidate_count':len(candidates),'search_rejections':statistics,'camera_candidates_tested':camera_index+1,'complete_camera_pool':complete_camera_count,'grid_step_m':step,'native_room_floor_y_m':floor.room_floor_y_m,'native_camera_eye_height_m':eye_height_m,'camera_construction':'actual 25cm native air clearance; camera has no standing-foot/NavMesh requirement','room_floor_polygon':declaration['floor_polygon'],'frozen_source_clearance_m':.15,'capsule_radius_m':.2,'capsule_half_height_m':.75,'admission_thresholds_changed':False}
    raise ValueError('no native geometry construction after expanded same-room search: '+json.dumps({'rejections':statistics,'native_body_candidates':len(all_candidates),'camera_failures':failures}))

def requested_camera_eye_height(request,inputs):
    required=set(request.get('required_sound_classes',[]))
    if not required:return 1.5
    inventory=json.loads(Path(inputs['kujiale_ue_asset_admission']).read_text());assets=inventory['assets'].values()if isinstance(inventory['assets'],dict)else inventory['assets']
    low=any(a.get('ue_native_physical_status')=='pass'and a.get('attachment_surface')=='floor'and a['dimensions_m'][1]<.9 and required.intersection(a.get('allowed_v7_classes_candidate',[]))for a in assets)
    # A fixed lower listener/camera keeps the complete static RIR/FOA model,
    # while putting a low physical body inside the unchanged vertical view.
    return 1.0 if low else 1.5


def construct_request(request,house,runtime,catalog,semantic,registration):
    seed=int(request['scene_seed']);attempt=int(request.get('construction_attempt',0));ids,gender=registered_people(catalog,registration,seed);scenario,sounds,offsets=authored_pair(semantic,gender,seed,latest_end_s=4.8)
    row={'schema':'v7-recorded-registered-standing-v1','house':house['house'],'room_id':request['room_id'],'scene_id':request['scene_id'],'independent_parent_scene_id':request['scene_id'],'scene_seed':seed,'scene_index':seed,'seeds':factor_seeds(seed),'split':{'development':'dev'}.get(house['split'],house['split']),'duration_s':30,'members':['v0_a0','v0_a1','v1_a0','v1_a1'],'window_count':5,'window_start_ticks':[0,50,100,150,200],'questions_per_window':14,'question_quota_base':int(request.get('question_quota_base',0)),'counterfactual_plan':'appearance_x_emitter_association','allowed_member_changes':['actor_appearance_assignment','dry_bus_to_emitter_assignment'],'assignment_status':'admitted','render_path':'ue_spear','shared_factors':['house','room','materials','camera','background','trajectory','event_times'],'construction_request':deepcopy(request)}
    row['generation_policy']='owner-fixed-camera-natural-layout-20261015-v2'
    row['measured_static_layout']=construct_layout(house,request['room_id'],runtime,attempt=attempt,eye_height_m=requested_camera_eye_height(request,runtime))
    from avengine.timeline.v7_fixed_camera_layout import natural_person_yaws
    from avengine.capture.kujiale_pilot_environment import construct_dialogue_camera
    layout=row['measured_static_layout']
    from shapely.geometry import shape
    centroid=shape(layout['room_floor_polygon']).centroid;room_center=[centroid.x,layout['listener_m'][1],centroid.y]
    yaws,modes=natural_person_yaws([p['position_m']for p in layout['valid']],layout['listener_m'],seed=row['seeds']['pose'],room_center=room_center,return_modes=True)
    for point,yaw,mode in zip(layout['valid'],yaws,modes):point.update(body_yaw_deg=yaw,initial_orientation_mode=mode)
    layout['initial_orientation_sampling']={'policy':'seeded per-person mixture of conversation, room center and free direction; no index template','seed':row['seeds']['pose'],'modes':modes,'room_center_m':room_center,'turns':0,'idle_status':'registered static native geometry; skeletal Idle integration not yet verified'}
    camera=construct_dialogue_camera(layout['listener_m'],[p['position_m']for p in layout['valid'][:2]],additional_positions=[p['position_m']for p in layout['valid'][2:]])
    layout['fixed_camera_construction']=camera
    layout['fixed_camera_yaw_deg']=camera['yaw_deg'];layout['orientation_construction']='seeded independent natural mixture; no fixed per-index pose; no turning';layout['human_minimum_center_distance_m']=1.0
    row['production_scenario']={'speaker_count':2,'gender':'F'if gender=='female'else'M','asset_ids':ids,'semantic_scenario_id':scenario,'semantic_sound_ids':sounds,'semantic_sound_subset_ids':sounds,'semantic_role_start_offsets_s':offsets,'environment_sound_classes':[],'camera_policy':'binding_static','camera_yaw_deg':0.,'binding_camera_yaw_offset_deg':0.,'recipe_revision':REVISION,'requested_posture_policy':'registered stand; full original native episode admission mandatory','category_focus_class':None,'non_speech_class':None}
    sounds_by_id={s['sound_asset_id']:s for s in semantic['sounds']}
    speech_end=max(offset+sounds_by_id[sid]['synthesis_readback']['duration_s']for sid,offset in zip(sounds,offsets))
    row['production_scenario'].update(camera_policy='binding_static',camera_yaw_deg=layout['fixed_camera_yaw_deg'])
    required=request.get('required_sound_classes',[])
    if required:
        from avengine.timeline.v7_kujiale_environment_construction import construct_environment,separate_environment_schedule
        plans=construct_environment(row,house,runtime,required)
        row['production_scenario'].update(native_environment=plans,environment_sound_classes=list(required),category_focus_class=required[0])
        end,effect_start=separate_environment_schedule(row,semantic)
    return row
