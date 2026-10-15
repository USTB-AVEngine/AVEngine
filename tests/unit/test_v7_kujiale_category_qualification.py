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
        assert 'production_input_contract'not in probe_inputs;captured.append(native_only_visual);root=Path(output)/'parents'/probe['scene_id'];folder=root/('attempt_'+native_only_visual);folder.mkdir(parents=True);visual={'frames':[{} for _ in range(450)],'frame_count':450,'native_body_foot_status':'pass','native_source_material_status':'pass','standing_admission':[{'status':'pass','known_bad_detected':True}for _ in range(3)],'renderer':{'registered_asset_ids':['a','b','c']}};(folder/'native_registered_visual_evidence.json').write_text(json.dumps(visual));(root/('visual_'+native_only_visual+'_completed.json')).write_text(json.dumps({'attempt':folder.name}));return {'not_a_parent_admission':True}
    monkeypatch.setattr(v7_registered_generation,'generate_registered_parent',native)
    result,proof=qualify_missing_humans(row,{},inputs,[],{},{},{},tmp_path)
    assert captured==['v0','v1'] and proof['thresholds_changed']is False and original.read_bytes()==before
    derived=json.loads(Path(result['production_input_contract']).read_text());assert derived['rooms'][0]['categories']==['animal.dog','human.standing']


def test_missing_native_dog_category_is_measured_without_removing_the_dog(tmp_path,monkeypatch):
    from avengine.timeline import v7_registered_generation
    from avengine.registry.kujiale_room_categories import bind_formal_room_category
    original=tmp_path/'contract.json';original.write_text(json.dumps({'rooms':[{'key':'h::R','categories':['human.standing']}]}));before=original.read_bytes();inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'assets':[{'asset_id':'required_dog','category':'dog'}]}));inputs={'render_backend':'ue_spear','production_input_contract':str(original),'kujiale_ue_asset_admission':str(inventory)};row={'house':'h','room_id':'R','scene_id':'fresh_required_body','production_scenario':{'native_environment':[{'asset_id':'required_dog'}]}};calls=[]
    def native(probe,house,probe_inputs,catalog,source,human,output,*,native_only_visual):
        assert probe['production_scenario']['native_environment']==row['production_scenario']['native_environment'];calls.append(native_only_visual);root=Path(output)/'parents'/probe['scene_id'];folder=root/('attempt_'+native_only_visual);folder.mkdir(parents=True)
        physical={'status':'pass','frame_count':450,'all_sources_have_drawable_geometry':True,'current_joint_admission':{'environment1':{'asset_id':'required_dog','status':'pass','attachment':{'status':'pass'},'views':[{'status':'pass'}for _ in range(3)]}},'frames':[{'entities':[{'asset_id':'required_dog','source_geometry_present':True}]}for _ in range(450)]}
        visual={'frames':[{}for _ in range(450)],'frame_count':450,'native_body_foot_status':'pass','native_source_material_status':'pass','standing_admission':[{'status':'pass','known_bad_detected':True}for _ in range(3)],'renderer':{'registered_asset_ids':['a','b','c']},'native_physical_environment':physical};(folder/'native_registered_visual_evidence.json').write_text(json.dumps(visual));(root/('visual_'+native_only_visual+'_completed.json')).write_text(json.dumps({'attempt':folder.name}));return {}
    monkeypatch.setattr(v7_registered_generation,'generate_registered_parent',native)
    result,proof=qualify_missing_humans(row,{},inputs,[],{},{},{},tmp_path)
    assert calls==['v0','v1'] and proof['categories']==['animal.dog'] and source_unchanged(original,before)
    assert json.loads(Path(result['production_input_contract']).read_text())['rooms'][0]['categories']==['animal.dog','human.standing']
    formal={'rooms':[{'house':'h','room_id':'R','smoke_status':'pass','current_native_category_qualifications':[proof]}]};new_row={**row,'scene_id':'ordinary_fresh_parent'};_,bound,binding=bind_formal_room_category(new_row,inputs,tmp_path/'normal_production',registration=formal)
    assert binding['categories_added']==['animal.dog']and source_unchanged(original,before)
    witness=proof['native_witnesses'][0];path=Path(witness['native_evidence']);visual=json.loads(path.read_text());visual['native_physical_environment']['frames'][12]['entities']=[];path.write_text(json.dumps(visual))
    from avengine.registry.kujiale_ue_measurements import sha256
    witness['sha256']=sha256(path)
    with pytest.raises(ValueError,match='body missing'):bind_formal_room_category(new_row,inputs,tmp_path/'must_fail',registration=formal)


def source_unchanged(path,raw):return path.read_bytes()==raw
