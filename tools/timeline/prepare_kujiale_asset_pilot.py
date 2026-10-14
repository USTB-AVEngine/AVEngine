#!/usr/bin/env python3
"""Prepare joint native-LOD placements for a train-only Kujiale asset pilot."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
from copy import deepcopy
from collections import defaultdict,Counter
import numpy as np
from avengine.timeline.continuous_pilot import write_json as _publish_json
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        # Completed plan products are retained, never overwritten during recovery.
        saved=json.loads(path.read_text())
        expected=json.loads(json.dumps(value,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
        if saved!=expected:raise ValueError("completed plan differs; use a fresh planned output: "+str(path))
        return
    _publish_json(path,value)
from avengine.qa.kujiale_asset_geometry import NativeSupportGeometry,import_vertices,rotation,camera_candidates
from avengine.qa.kujiale_asset_paws import landmark_contacts

def body_key(row):
    return (row['category'],row['body_id']) if row['category']!='object' else ('object',row['object_type'])

def separated(low,high,envelopes,margin=.025):
    return all(np.any(np.minimum(high,bhi)-np.maximum(low,blo)<-margin) for blo,bhi in envelopes)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','prior-root','houses','seeds','object-roster','source-admission']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();started=time.time()
    (a.root/'inputs/parents').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.prior_root/'inputs/assets_final.json').read_text())
    book=json.loads((a.prior_root/'registrations/v7_kujiale_ue_asset_admission_20261012.json').read_text())
    native={x['asset_id']:x for x in book['assets']}
    objects={x['asset_id']:x for x in json.loads(a.object_roster.read_text())['assets']}
    classes=Counter(x['sound_class'] for x in json.loads(a.source_admission.read_text())['records'])
    grouped=defaultdict(list)
    for row in manifest['assets']:grouped[body_key(row)].append(row)
    source_animals={}
    for path in manifest['registration_paths']:
        d=json.loads(Path(path).read_text())
        for key in ['animals','append_new_animals','v6_idle_animals']:
            for item in d.get(key,[]):
                if isinstance(item,dict) and item.get('asset_id'):source_animals[item['asset_id']]=item
    selected=[]
    for key,values in sorted(grouped.items()):
        # Exact registered bodies; colour does not exclude an admitted asset.
        values.sort(key=lambda x:(x.get('walk_allowed',False),max(x['dimensions_m']),x['asset_id']))
        selected.append(deepcopy(values[0]))
    # Both admitted air-conditioner mounting forms are silent, visible decorations.
    ac=[x for x in selected if x.get('object_type')=='air_conditioner'][0]
    selected.extend(deepcopy(x) for x in manifest['assets'] if x.get('object_type')=='air_conditioner' and x['asset_id']!=ac['asset_id'])
    for row in selected:
        if row['category']=='object':allowed=objects[row['asset_id']]['allowed_v7_classes_candidate']
        else:
            allowed={'dog':['dog_bark'],'cat':['cat_meow'],'bird':['pet_bird_'+(source_animals.get(row['asset_id'],{}).get('sound_appearance_family') or 'canary')]}[row['category']]
        row['allowed_v7_classes_candidate']=allowed
        if not row['silent_decor'] and not any(classes[c] for c in allowed):
            raise ValueError('registered asset has no admitted matching recording: '+row['asset_id']+' '+str(allowed))
    oldrows=[json.loads(x) for x in a.seeds.read_text().splitlines() if x.strip()]
    houses=json.loads(a.houses.read_text())['houses']
    ids=['kujiale_0003','kujiale_0009','kujiale_0020','kujiale_0022','kujiale_0030','kujiale_0036']
    hs=[deepcopy(next(h for h in houses if h['house']==hid)) for hid in ids]
    rows=[]
    for hi,h in enumerate(hs):
        for j in range(2):
            seed=deepcopy(next(x for x in oldrows if x['house']==h['house'] and x['room_id']=='R0'))
            sid=f'v7_kujiale_ue_assets_pilot_{hi*2+j:06d}'
            seed.update(scene_id=sid,independent_parent_scene_id=sid,members=['v0_a0','v0_a1','v1_a0','v1_a1'],window_count=2,questions_per_window=14,question_quota_base=(hi*2+j)*2)
            seed['production_scenario'].update(speaker_count=2,non_speech_class='laughter',environment_sound_classes=[],camera_policy='review_sweep_360',camera_yaw_deg=float(seed['measured_static_layout']['center_yaw_deg']),semantic_content_kind=None)
            for k in ['semantic_scenario_id','semantic_sound_ids','semantic_sound_subset_ids','selection_coverage_seals']:seed['production_scenario'].pop(k,None);seed.pop(k,None)
            rows.append(seed)
    # Round-robin species, then balance mounting surfaces and room occupancy.
    allocation=[[] for _ in rows]
    animals=[x for x in selected if x['category']!='object']
    for i,row in enumerate(animals):allocation[i%12].append(row)
    items=[x for x in selected if x['category']=='object']
    # Spread table and wall items; each scene has at most two small objects.
    for row in sorted(items,key=lambda x:(-max(x['dimensions_m']),x['attachment_surface'],x['asset_id'])):
        idx=min(range(12),key=lambda i:(sum(x['attachment_surface']==row['attachment_surface'] and x['category']=='object' for x in allocation[i]),len(allocation[i]),i))
        allocation[idx].append(row)
    for hi,h in enumerate(hs):
        room=next(x for x in h['checked_rooms'] if x['room_label']=='R0')
        g=NativeSupportGeometry(h['input']['ue_config']['native_floor_surface']);catalog,poly=g.support_catalog(h,room)
        write_json(a.root/'evidence'/f'support_catalog_{h["house"]}.json',{'house':h['house'],'room_id':'R0','catalog':catalog,'floor_y_m':g.floor.room_floor_y_m})
        for idx in [hi*2,hi*2+1]:
            scene=rows[idx];envelopes=[];placements=[]
            for p0 in scene['measured_static_layout']['valid'][:3]:
                p=np.array(p0.get('root_m') or [p0['position_m'][0],p0['position_m'][1]-1.3,p0['position_m'][2]])
                # Conservative standing human envelope; actual native skin is rechecked in the episode.
                envelopes.append((p+[-.4,0,-.4],p+[.4,1.9,.4]))
            for row in sorted(allocation[idx],key=lambda x:(-max(x['dimensions_m']),x['asset_id'])):
                imported=json.loads((a.prior_root/'evidence/import/final'/f'{row["asset_id"]}.json').read_text())
                vertices=import_vertices(imported);expected=np.array(row['dimensions_m'])
                error=np.abs(np.ptp(vertices,axis=0)-expected)/np.maximum(expected,1e-5)
                if error.max()>.15:raise ValueError('native dimensions exceed unchanged 15% gate')
                surface=row['attachment_surface'];tolerance=.012 if row['category']!='object' else .015
                supports=catalog[surface]
                listener=np.array(scene['measured_static_layout']['listener_m'])
                supports=sorted(supports,key=lambda q:abs(np.linalg.norm((np.array(q['point_m'])-listener)[[0,2]])-3.1))
                accepted=None
                for support in supports:
                    if row.get('object_type')=='air_conditioner' and surface=='wall' and support['point_m'][1]-support['floor_y_m']<2.2:continue
                    n=np.array(support['normal']);angles=[math.atan2(-n[2],n[0])] if surface=='wall' else [0.,math.pi/2,math.pi,-math.pi/2]
                    for yaw in angles:
                        # Cheap envelope rejection precedes native triangle queries.
                        R=rotation(yaw);v=vertices@R.T
                        root=np.array(support['point_m'])-(v.min(axis=0)+v.max(axis=0))/2
                        if not separated(v.min(axis=0)+root,v.max(axis=0)+root,envelopes):continue
                        placement=g.placement(vertices,support,yaw,tolerance)
                        if placement is None:continue
                        v=vertices@R.T+np.array(placement['root_m']);lo,high=v.min(axis=0),v.max(axis=0)
                        if not separated(lo,high,envelopes):continue
                        placement['paw_contacts']=landmark_contacts(g,row,vertices,placement)
                        if placement['paw_contacts']['status']=='fail':continue
                        views=camera_candidates(vertices,placement,poly);clear=[]
                        for view in views:
                            eye=np.array(view['position_m']);d=np.array(view['target_m'])-eye;length=np.linalg.norm(d)
                            if g.ray(eye,d/length,float(length)-.02) is None:clear.append(view)
                        if len(clear)<3:continue
                        placement['camera_candidates']=clear;placement['native_world_bounds_m']=[lo.tolist(),high.tolist()]
                        accepted=placement;envelopes.append((lo,high));break
                    if accepted:break
                if accepted is None:raise ValueError('no joint native support/clearance plan: '+scene['scene_id']+' '+row['asset_id'])
                allowed=row['allowed_v7_classes_candidate'];cls=next((c for c in allowed if classes[c]),None)
                if 'music_playback' in allowed:cls='music_playback'
                point=np.array(accepted['root_m'])+rotation(accepted['yaw_rad'])@np.array(row.get('emitter_local_m') or [0,0,0])
                placements.append({'asset_id':row['asset_id'],'sound_class':cls,'silent_decor':row['silent_decor'],
                    'placement':accepted,'ue_asset_paths':native[row['asset_id']]['ue_asset_paths'],
                    'size':{'status':'pass','native_dimensions_m':np.ptp(vertices,axis=0).tolist(),'relative_errors':error.tolist(),'tolerance_fraction':.15},
                    'expected_emitter_m':point.tolist(),'import_receipt':str(a.prior_root/'evidence/import/final'/f'{row["asset_id"]}.json'),
                    'status':'pass_geometry_requires_current_native_scene'})
            scene['production_scenario']['native_environment']=placements
            scene['production_scenario']['environment_sound_classes']=[x['sound_class'] for x in placements if not x['silent_decor']]
            scene['production_scenario']['category_focus_class']=scene['production_scenario']['environment_sound_classes'][0]
            write_json(a.root/'inputs/parents'/f'{scene["scene_id"]}.json',scene)
            print('JOINT_PLAN',scene['scene_id'],h['house'],[(x['asset_id'],x['sound_class']) for x in placements],flush=True)
        del g
    for h in hs:
        ue=h['input']['ue_config'];ue.update(uproject=str(a.root/'ue_v1/runtime/SpearSim/SpearSim.uproject'),output_root=str(a.root),graphics_adapter=0)
    write_json(a.root/'inputs/houses.json',{'houses':hs})
    for row in book['assets']:
        extra=next((x for x in selected if x['asset_id']==row['asset_id']),None)
        if extra:
            for key in ['perch','functional_front_axis','allowed_v7_classes_candidate']:row[key]=extra.get(key)
        row['prior_ue_project']=row['ue_project'];row['ue_project']=str(a.root/'ue_v1/runtime/SpearSim/SpearSim.uproject')
        row['admission_decision']='native_asset_admitted_color_questions_require_current_scene_match'
    book.update(scope='owner-authorized round21 pilot; prior physical receipts retained; current joint scene checks required',labels_modified=False,color_policy='owner 2026-10-13: colour mismatches never reject an asset; reject only affected current-scene colour questions')
    write_json(a.root/'registrations/assets.json',book)
    write_json(a.root/'inputs/assets_selected.json',{'revision':'pilot','assets':selected,'prior_root':str(a.prior_root)})
    (a.root/'inputs/pilot_seeds.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    write_json(a.root/'evidence/planning_summary.json',{'status':'pass','parents':len(rows),'houses':ids,'animal_bodies':sorted(set(body_key(x) for x in selected if x['category']!='object')),'object_types':sorted(set(x['object_type'] for x in selected if x['category']=='object')),'selected_assets':len(selected),'source_admission_inspected':str(a.source_admission),'wall_seconds':time.time()-started})
if __name__=='__main__':main()
