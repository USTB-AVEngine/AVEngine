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

REVISION='kujiale-room-binding-construction-v1'
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

def select_group(candidates,listener,attempt=0):
    from avengine.capture.kujiale_pilot_environment import binding_pair_fits_view,comparison_person_candidates
    listener=np.asarray(listener,float);ordered=sorted(candidates,key=lambda p:(abs(np.linalg.norm((np.asarray(p['position_m'])-listener)[[0,2]])-2.4),p['position_m']))
    found=0
    for i,first in enumerate(ordered):
        for second in ordered[i+1:]:
            if np.linalg.norm((np.asarray(first['root_m'])-second['root_m'])[[0,2]])<.85:continue
            if not binding_pair_fits_view(listener,first['position_m'],second['position_m']):continue
            other=[c for c in ordered if min(np.linalg.norm((np.asarray(c['root_m'])-a['root_m'])[[0,2]])for a in [first,second])>=.85]
            thirds=comparison_person_candidates(listener,[first['position_m'],second['position_m']],[c['position_m']for c in other])
            for third in thirds[:3]:
                if found==attempt:
                    c=next(c for c in other if c['position_m']==third);return [deepcopy(first),deepcopy(second),deepcopy(c)]
                found+=1
    raise ValueError('no clear separated three-person group for this room/camera construction attempt')

def construct_layout(house,room_id,runtime,*,attempt=0):
    from shapely.geometry import shape,Point
    from avengine.acoustics.runtime import load_habitat_runtime
    from avengine.capture.native_floor_reference import NativeFloorReference
    from avengine.capture.kujiale_spear_runtime import to_ue
    from tools.rooms.check_kujiale_native_capsule_contact import query_capsule
    room=next(r for r in house['checked_rooms']if r['room_label']==room_id)
    declaration=next(r for r in json.loads(Path(house['input']['ue_config']['acceptance_plan_path']).read_text())['rooms']if r['room_id']==room_id)
    polygon=shape(declaration['floor_polygon']).buffer(-.3);floor=NativeFloorReference(house['input']['ue_config']['native_floor_surface']);hs,_=load_habitat_runtime(runtime_mode='current-installed',**runtime['runtime']);pf=hs.PathFinder()
    if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('native room NavMesh unavailable')
    views=[room['placement'],*[w['placement']for w in room.get('native_witnesses',[])]];views=list({tuple(v['camera_m']):v for v in views}.values());failures=[]
    for camera_index in range(min(len(views),12)):
        listener=np.asarray(views[(camera_index+attempt//3)%len(views)]['camera_m'],float);x0,z0,x1,z1=polygon.bounds;candidates=[];seen=set();probes=0
        for step in [.5,.35,.25]:
            for x in np.arange(x0+.1,x1,step):
                for z in np.arange(z0+.1,z1,step):
                    key=(round(float(x),4),round(float(z),4))
                    if key in seen or not polygon.covers(Point(x,z)):continue
                    seen.add(key)
                    if not 1.1<=math.hypot(x-listener[0],z-listener[2])<=5.5:continue
                    probes+=1;target=np.array([x,declaration['source_floor_y_m'],z]);nav=np.asarray(pf.snap_point(target),float)
                    if not np.isfinite(nav).all()or np.linalg.norm(nav-target)>.15:continue
                    try:root=np.asarray(floor.sample(nav),float)
                    except ValueError:continue
                    cap=query_capsule(floor.mesh,{'witness_index':probes,'role':'source','capsule':{'radius_cm':20.,'half_height_cm':75.,'center_ue_cm':to_ue(root+[0,.75,0])}})
                    if cap['status']!='pass':continue
                    clear=True
                    for height in [.9,1.2,1.5]:
                        ray=root+[0,height,0]-listener;hits,_,_=floor.mesh.ray.intersects_location(listener[None],(ray/np.linalg.norm(ray))[None],multiple_hits=False)
                        if len(hits)and np.linalg.norm(hits[0]-listener)<np.linalg.norm(ray)-.03:clear=False;break
                    if clear:
                        p=root+[0,1.3,0];yaw=math.degrees(math.atan2(listener[0]-p[0],-(listener[2]-p[2])));candidates.append({'position_m':p.tolist(),'root_m':root.tolist(),'body_yaw_deg':yaw,'capsule_check':cap})
            try:selected=select_group(candidates,listener,attempt%3)
            except ValueError:continue
            return {'house':house['house'],'room_id':room_id,'status':'CPU native geometry constructed; complete UE readback required','construction_revision':REVISION,'construction_attempt':attempt,'listener_m':listener.tolist(),'center_yaw_deg':0.,'valid':selected,'native_candidates':[p['position_m']for p in candidates],'candidate_count':probes,'clear_candidate_count':len(candidates),'room_floor_polygon':declaration['floor_polygon'],'frozen_source_clearance_m':.15,'capsule_radius_m':.2,'capsule_half_height_m':.75,'admission_thresholds_changed':False}
        failures.append({'camera_m':listener.tolist(),'probes':probes,'clear_candidates':len(candidates)})
    raise ValueError('no native geometry construction after bounded camera/grid search: '+json.dumps(failures))

def construct_request(request,house,runtime,catalog,semantic,registration):
    seed=int(request['scene_seed']);attempt=int(request.get('construction_attempt',0));ids,gender=registered_people(catalog,registration,seed);scenario,sounds,offsets=authored_pair(semantic,gender,seed)
    row={'schema':'v7-recorded-registered-standing-v1','house':house['house'],'room_id':request['room_id'],'scene_id':request['scene_id'],'independent_parent_scene_id':request['scene_id'],'scene_seed':seed,'scene_index':seed,'seeds':factor_seeds(seed),'split':{'development':'dev'}.get(house['split'],house['split']),'duration_s':30,'members':['v0_a0','v0_a1','v1_a0','v1_a1'],'window_count':5,'window_start_ticks':[0,50,100,150,200],'questions_per_window':14,'question_quota_base':int(request.get('question_quota_base',0)),'counterfactual_plan':'appearance_x_emitter_association','allowed_member_changes':['actor_appearance_assignment','dry_bus_to_emitter_assignment'],'assignment_status':'admitted','render_path':'ue_spear','shared_factors':['house','room','materials','camera','background','trajectory','event_times'],'construction_request':deepcopy(request)}
    row['measured_static_layout']=construct_layout(house,request['room_id'],runtime,attempt=attempt)
    row['production_scenario']={'speaker_count':2,'gender':'F'if gender=='female'else'M','asset_ids':ids,'semantic_scenario_id':scenario,'semantic_sound_ids':sounds,'semantic_sound_subset_ids':sounds,'semantic_role_start_offsets_s':offsets,'environment_sound_classes':[],'camera_policy':'binding_static','camera_yaw_deg':0.,'binding_camera_yaw_offset_deg':0.,'recipe_revision':REVISION,'requested_posture_policy':'registered stand; full original native episode admission mandatory','category_focus_class':None,'non_speech_class':None}
    return row
