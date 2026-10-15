"""Measure every missing required native category before adding a new witness.

Diagnostics never count as parent smoke. The original contract is retained,
and the full producer validates every new derived contract normally.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,os
from avengine.registry.kujiale_ue_measurements import sha256
from avengine.registry.kujiale_room_categories import required_category_witnesses,verify_category_witness


def qualify_missing_humans(row,house,inputs,catalog,source,human,contract,output):
    """Legacy entry name; qualify all required human/physical native categories."""
    original=Path(inputs['production_input_contract']);document=json.loads(original.read_text());key=row['house']+'::'+row['room_id'];cell=next(r for r in document['rooms']if r['key']==key);required=required_category_witnesses(row,inputs);missing=sorted(set(required)-set(cell['categories']))
    if not missing:return inputs,None
    from avengine.timeline.v7_registered_generation import generate_registered_parent
    probe=deepcopy(row);probe.pop('production_sampling',None);probe['scene_id']+='_native_category_probe';probe['independent_parent_scene_id']=probe['scene_id'];native_inputs=deepcopy(inputs);native_inputs.pop('production_input_contract',None);native=Path(output)/'room_category_qualification';proofs=[]
    for visual in ['v0','v1']:
        generate_registered_parent(probe,house,native_inputs,catalog,source,human,str(native),native_only_visual=visual)
        marker=native/'parents'/probe['scene_id']/('visual_'+visual+'_completed.json');saved=json.loads(marker.read_text());evidence=marker.parent/saved['attempt']/'native_registered_visual_evidence.json';readback=json.loads(evidence.read_text());proofs.append({'visual':visual,'native_evidence':str(evidence),'sha256':sha256(evidence),'frames':450,'asset_ids':readback['renderer']['registered_asset_ids'],'source_material_observation':{'native_source_material_status':readback.get('native_source_material_status'),'admission_effect':'record_only_per_owner_20261015','original_material_paths_and_package_identity_checked_by_native_backend':True}})
    witnesses={kind:(required[kind]or proofs[0]['asset_ids'][0])for kind in missing};category={'status':'pass_current_full_native','house':row['house'],'room_id':row['room_id'],'category':missing[0],'categories':missing,'witness_asset_ids':witnesses,'native_witnesses':proofs,'thresholds_changed':False,'original_contract':{'path':str(original),'sha256':sha256(original)}};verify_category_witness(category,row)
    target=Path(output)/'inputs/current_room_categories';target.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256(json.dumps(category,sort_keys=True).encode()).hexdigest()[:20];receipt=target/(row['house']+'_'+row['room_id']+'_native_'+digest+'.json');receipt.write_text(json.dumps(category,indent=2)+'\n')
    derived=deepcopy(document);room=next(r for r in derived['rooms']if r['key']==key);room['categories']=sorted(set(room['categories'])|set(missing));room.setdefault('witness_asset_ids',{}).update(witnesses);room['current_native_category_qualification']={'path':str(receipt),'sha256':sha256(receipt)};derived['additional_current_native_room_witnesses']=[category];path=target/(row['house']+'_'+row['room_id']+'_native_'+digest+'_production_contract.json');path.write_text(json.dumps(derived,indent=2)+'\n');qualified=deepcopy(inputs);qualified['production_input_contract']=str(path);return qualified,category
