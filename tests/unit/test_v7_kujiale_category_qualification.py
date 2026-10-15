import json
from pathlib import Path
import pytest
from avengine.timeline.v7_kujiale_category_qualification import qualify_missing_humans
pytestmark=pytest.mark.fast_unit

def test_existing_legality_does_not_need_or_fabricate_new_witness(tmp_path):
    original=tmp_path/'contract.json';original.write_text(json.dumps({'rooms':[{'key':'h::R','categories':['human.standing']}]}));inputs={'production_input_contract':str(original)}
    result,proof=qualify_missing_humans({'house':'h','room_id':'R'},{},inputs,[],{},{},{},tmp_path)
    assert result==inputs and proof is None

def test_missing_category_requires_complete_native_capture_and_preserves_original(tmp_path,monkeypatch):
    from avengine.timeline import v7_registered_generation
    original=tmp_path/'contract.json';document={'rooms':[{'key':'h::R','categories':['animal.dog']}],'schema':'v7-production-inputs-v1'};original.write_text(json.dumps(document));before=original.read_bytes();inputs={'production_input_contract':str(original)};row={'house':'h','room_id':'R','scene_id':'s'};captured=[]
    def native(probe,house,probe_inputs,catalog,source,human,output,*,native_only_visual):
        assert 'production_input_contract'not in probe_inputs;captured.append(native_only_visual);root=Path(output)/'parents'/probe['scene_id'];folder=root/('attempt_'+native_only_visual);folder.mkdir(parents=True);visual={'frame_count':450,'native_body_foot_status':'pass','native_source_material_status':'pass','standing_admission':[{'status':'pass','known_bad_detected':True}for _ in range(3)],'renderer':{'registered_asset_ids':['a','b','c']}};(folder/'native_registered_visual_evidence.json').write_text(json.dumps(visual));(root/('visual_'+native_only_visual+'_completed.json')).write_text(json.dumps({'attempt':folder.name}));return {'not_a_parent_admission':True}
    monkeypatch.setattr(v7_registered_generation,'generate_registered_parent',native)
    result,proof=qualify_missing_humans(row,{},inputs,[],{},{},{},tmp_path)
    assert captured==['v0','v1'] and proof['thresholds_changed']is False and original.read_bytes()==before
    derived=json.loads(Path(result['production_input_contract']).read_text());assert derived['rooms'][0]['categories']==['animal.dog','human.standing']
