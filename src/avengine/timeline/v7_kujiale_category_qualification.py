"""Measure a missing room/category before adding its new production witness.

The old legality contract is unchanged. Diagnostic native captures do not
count as parent smoke, and the normal validator must pass the derived contract.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
from avengine.registry.kujiale_ue_measurements import sha256

def qualify_missing_humans(row,house,inputs,catalog,source,human,contract,output):
    original=Path(inputs['production_input_contract']);document=json.loads(original.read_text());key=row['house']+'::'+row['room_id'];cell=next(r for r in document['rooms']if r['key']==key)
    if 'human.standing'in cell['categories']:return inputs,None
    from avengine.timeline.v7_registered_generation import generate_registered_parent
    probe=deepcopy(row);probe.pop('production_sampling',None);probe['scene_id']+='_standing_category_probe';probe['independent_parent_scene_id']=probe['scene_id'];native_inputs=deepcopy(inputs);native_inputs.pop('production_input_contract',None);native=Path(output)/'room_category_qualification';proofs=[]
    for visual in ['v0','v1']:
        result=generate_registered_parent(probe,house,native_inputs,catalog,source,human,str(native),native_only_visual=visual)
        marker=native/'parents'/probe['scene_id']/('visual_'+visual+'_completed.json');saved=json.loads(marker.read_text());evidence=marker.parent/saved['attempt']/'native_registered_visual_evidence.json';readback=json.loads(evidence.read_text())
        if readback['frame_count']!=450 or readback['native_body_foot_status']!='pass'or readback['native_source_material_status']!='pass':raise ValueError('new standing category lacks complete original native gates')
        if len(readback['standing_admission'])<3 or any(c['status']!='pass'or not c['known_bad_detected']for c in readback['standing_admission']):raise ValueError('new standing category lacks actual support/collision negative-control checks')
        proofs.append({'visual':visual,'native_evidence':str(evidence),'sha256':sha256(evidence),'frames':450,'asset_ids':readback['renderer']['registered_asset_ids']})
    category={'status':'pass_current_full_native','house':row['house'],'room_id':row['room_id'],'category':'human.standing','native_witnesses':proofs,'thresholds_changed':False,'original_contract':{'path':str(original),'sha256':sha256(original)}}
    target=Path(output)/'inputs/current_room_categories';target.mkdir(exist_ok=True);receipt=target/(row['house']+'_'+row['room_id']+'_standing.json');receipt.write_text(json.dumps(category,indent=2)+'\n')
    derived=deepcopy(document);room=next(r for r in derived['rooms']if r['key']==key);room['categories']=sorted(set(room['categories'])|{'human.standing'});room.setdefault('witness_asset_ids',{})['human.standing']=proofs[0]['asset_ids'][0];room['current_native_category_qualification']={'path':str(receipt),'sha256':sha256(receipt)};derived['additional_current_native_room_witnesses']=[category];path=target/(row['house']+'_'+row['room_id']+'_production_contract.json');path.write_text(json.dumps(derived,indent=2)+'\n');qualified=deepcopy(inputs);qualified['production_input_contract']=str(path);return qualified,category
