"""Reuse complete native room/category witnesses in ordinary production.

The original contract and normal intake validator remain unchanged. A required
category can only be added to a new derived contract using actual SHA-bound
native captures and a completely passed formal room.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,os
from avengine.registry.kujiale_ue_measurements import read_registration,sha256


def required_category_witnesses(row,inputs):
    required={'human.standing':None};bindings=row.get('production_scenario',{}).get('native_environment',[])
    if not bindings:return required
    from avengine.timeline.v7_production_inputs import asset_category
    records=json.loads(Path(inputs['kujiale_ue_asset_admission']).read_text())['assets'];records=records.values()if isinstance(records,dict)else records;by={a['asset_id']:a for a in records}
    for binding in bindings:
        asset=deepcopy(by[binding['asset_id']])
        if asset.get('category')in {'dog','cat','bird'}:asset.setdefault('species_id',asset['category'])
        asset.setdefault('kind','animal'if asset.get('species_id')in {'dog','cat','bird'}else'object')
        required[asset_category(asset,{},'kujiale')]=asset['asset_id']
    return required


def witness_categories(category):return category.get('categories')or[category['category']]


def verify_category_witness(category,row):
    if category.get('status')!='pass_current_full_native'or category.get('house')!=row['house']or category.get('room_id')!=row['room_id']or category.get('thresholds_changed')is not False:raise ValueError('formal native room category witness differs')
    categories=witness_categories(category);proofs=category['native_witnesses']
    if len(proofs)!=2 or {p['visual']for p in proofs}!={'v0','v1'}:raise ValueError('formal category requires both complete native visuals')
    for proof in proofs:
        source=Path(proof['native_evidence'])
        if sha256(source)!=proof['sha256']:raise ValueError('formal room category native evidence changed')
        native=json.loads(source.read_text());standing=native['standing_admission']
        if proof['frames']!=450 or native['frame_count']!=450 or len(native['frames'])!=450 or native['native_body_foot_status']!='pass'or native['native_source_material_status']!='pass'or len(standing)<3 or any(p['status']!='pass'or not p['known_bad_detected']for p in standing):raise ValueError('formal category lacks original full contact/material/negative-control gates')
        if proof['asset_ids']!=native['renderer']['registered_asset_ids']:raise ValueError('formal standing category asset readback differs')
        for kind in categories:
            if kind=='human.standing':continue
            asset_id=category['witness_asset_ids'][kind];physical=native.get('native_physical_environment',{});entities=physical.get('current_joint_admission',{})
            if physical.get('status')!='pass'or physical.get('frame_count')!=450 or len(physical.get('frames',[]))!=450 or not physical.get('all_sources_have_drawable_geometry'):raise ValueError('new physical category lacks complete native body frames')
            admitted=[a for a in entities.values()if a['asset_id']==asset_id and a['status']=='pass'and a['attachment']['status']=='pass'and sum(v['status']=='pass'for v in a['views'])>=3]
            if not admitted:raise ValueError('new physical category lacks original native contact/collision/three-view admission')
            if any(not any(e['asset_id']==asset_id and e.get('source_geometry_present')for e in frame['entities'])for frame in physical['frames']):raise ValueError('required native category body missing in a measured frame')
    return categories


def bind_formal_room_category(row,inputs,output,*,registration=None):
    path=inputs.get('production_input_contract')
    if not path or inputs.get('render_backend')!='ue_spear':return row,inputs,None
    path=Path(path).resolve();document=json.loads(path.read_text());key=row['house']+'::'+row['room_id'];cell=next((r for r in document['rooms']if r['key']==key),None)
    if cell is None:return row,inputs,None
    missing=set(required_category_witnesses(row,inputs))-set(cell['categories'])
    if not missing:return row,inputs,None
    formal=registration if registration is not None else read_registration(inputs.get('kujiale_native_measurement_registration'))[0];room=next((r for r in formal.get('rooms',[])if r['house']==row['house']and r['room_id']==row['room_id']and r.get('smoke_status')=='pass'),{})
    available=room.get('current_native_category_qualifications')or([room['current_native_category_qualification']]if room.get('current_native_category_qualification')else[]);selected=[];additions=set();witnesses={}
    for category in available:
        if not missing.intersection(witness_categories(category)):continue
        measured=verify_category_witness(category,row);selected.append(category);additions.update(measured);witnesses.update(category.get('witness_asset_ids')or{'human.standing':category['native_witnesses'][0]['asset_ids'][0]})
    if not selected:return row,inputs,None
    old_binding={'manifest_path':str(path),'manifest_sha256':sha256(path)}
    if row.get('production_sampling')is not None and row['production_sampling']!=old_binding:raise ValueError('recipe and runtime production input contracts differ')
    derived=deepcopy(document);target_cell=next(r for r in derived['rooms']if r['key']==key);target_cell['categories']=sorted(set(target_cell['categories'])|additions);target_cell.setdefault('witness_asset_ids',{}).update(witnesses);target_cell['current_native_category_qualifications']=deepcopy(selected)
    binding={'house':row['house'],'room_id':row['room_id'],'categories_added':sorted(additions),'original_contract':old_binding,'source':'formal production UE native measurements; complete passed room','thresholds_changed':False};derived['formal_native_room_category_binding']=binding
    content=json.dumps(derived,ensure_ascii=False,indent=2)+'\n';digest=hashlib.sha256(content.encode()).hexdigest();target=Path(output)/'parents'/row['scene_id']/'formal_runtime_inputs'/('production_contract_'+digest[:20]+'.json')
    from avengine.timeline.v7_storage_budget import check_write
    check_write(target,expected_bytes=len(content.encode()));target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_text()!=content:raise ValueError('derived formal native category contract changed')
    else:
        temporary=target.with_name(target.name+f'.{os.getpid()}.tmp');temporary.write_text(content);temporary.replace(target)
    new_row=deepcopy(row);new_row.pop('production_sampling',None);new_inputs=deepcopy(inputs);new_inputs['production_input_contract']=str(target);new_inputs['formal_native_room_category_binding']=deepcopy(binding)
    return new_row,new_inputs,binding
