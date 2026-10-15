import copy,json
from pathlib import Path
import pytest
from avengine.registry.kujiale_room_categories import bind_formal_room_category
from avengine.registry.kujiale_ue_measurements import sha256
pytestmark=pytest.mark.fast_unit


def data(tmp_path):
    source=tmp_path/'original_contract.json';source.write_text(json.dumps({'rooms':[{'key':'h::R','categories':['animal.dog']}]}));proofs=[]
    for visual in ['v0','v1']:
        native={'frames':[{} for _ in range(450)],'frame_count':450,'native_body_foot_status':'pass','native_source_material_status':'pass','standing_admission':[{'status':'pass','known_bad_detected':True}for _ in range(3)],'renderer':{'registered_asset_ids':['a','b','c']}};path=tmp_path/(visual+'.json');path.write_text(json.dumps(native));proofs.append({'visual':visual,'native_evidence':str(path),'sha256':sha256(path),'frames':450,'asset_ids':['a','b','c']})
    category={'status':'pass_current_full_native','house':'h','room_id':'R','category':'human.standing','thresholds_changed':False,'native_witnesses':proofs};formal={'rooms':[{'house':'h','room_id':'R','smoke_status':'pass','current_native_category_qualification':category}]};row={'house':'h','room_id':'R','scene_id':'brand_new_parent','production_sampling':{'manifest_path':str(source.resolve()),'manifest_sha256':sha256(source)}};inputs={'render_backend':'ue_spear','production_input_contract':str(source)};return source,formal,row,inputs


def test_fresh_production_uses_measured_formal_category_without_mutating_original(tmp_path):
    from avengine.timeline.v7_production_inputs import attach_runtime_contract
    source,formal,row,inputs=data(tmp_path);original=source.read_bytes();old_row=copy.deepcopy(row)
    result,new_inputs,binding=bind_formal_room_category(row,inputs,tmp_path/'new_output',registration=formal)
    assert binding['thresholds_changed']is False and source.read_bytes()==original and row==old_row
    derived=json.loads(Path(new_inputs['production_input_contract']).read_text());assert derived['rooms'][0]['categories']==['animal.dog','human.standing']
    attached=attach_runtime_contract(result,new_inputs);assert attached['production_sampling']['manifest_sha256']==sha256(new_inputs['production_input_contract'])
    assert str(tmp_path/'new_output/parents/brand_new_parent')in new_inputs['production_input_contract']


def test_unpassed_room_cannot_promote_a_diagnostic_only_category(tmp_path):
    _,formal,row,inputs=data(tmp_path);formal['rooms'][0]['smoke_status']='running'
    assert bind_formal_room_category(row,inputs,tmp_path,registration=formal)==(row,inputs,None)


@pytest.mark.parametrize('bad',['sha','frames','negative_control','other_room'])
def test_formal_category_cannot_bypass_unchanged_native_witness_checks(tmp_path,bad):
    _,formal,row,inputs=data(tmp_path);category=formal['rooms'][0]['current_native_category_qualification'];proof=category['native_witnesses'][0]
    if bad=='sha':proof['sha256']='0'*64
    elif bad=='other_room':category['room_id']='elsewhere'
    else:
        path=Path(proof['native_evidence']);native=json.loads(path.read_text())
        if bad=='frames':native['frame_count']=449
        else:native['standing_admission'][0]['known_bad_detected']=False
        path.write_text(json.dumps(native));proof['sha256']=sha256(path)
    with pytest.raises(ValueError):bind_formal_room_category(row,inputs,tmp_path,registration=formal)


def test_formal_category_does_not_accept_a_mismatched_existing_runtime_contract(tmp_path):
    _,formal,row,inputs=data(tmp_path);row['production_sampling']['manifest_sha256']='not_the_original'
    with pytest.raises(ValueError,match='contracts differ'):bind_formal_room_category(row,inputs,tmp_path,registration=formal)


def test_ordinary_registered_producer_enters_formal_binding_before_native_work(monkeypatch):
    from avengine.registry import kujiale_room_categories
    from avengine.timeline.v7_registered_generation import generate_registered_parent
    def reached(row,inputs,output):raise RuntimeError('ordinary production reached formal category binding')
    monkeypatch.setattr(kujiale_room_categories,'bind_formal_room_category',reached)
    with pytest.raises(RuntimeError,match='ordinary production reached'):generate_registered_parent({}, {}, {}, [], {}, {}, '/no_files_written')
