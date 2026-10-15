import pytest
from avengine.timeline.v7_production_inputs import room_eligible_catalog,planning_catalog
pytestmark=pytest.mark.fast_unit


def test_unavailable_species_and_mount_are_filtered_before_construction():
    assets=[{'asset_id':'h','kind':'human'},{'asset_id':'d','kind':'animal','species_id':'dog'},
      {'asset_id':'c','kind':'animal','species_id':'cat'},
      {'asset_id':'t','kind':'object','attachment_surface':'table'},
      {'asset_id':'w','kind':'object','attachment_surface':'wall'}]
    room={'house':'new','room_id':'R','family':'hm3d','categories':['standing_human','floor_dog','table_object']}
    assert [a['asset_id']for a in room_eligible_catalog(assets,room)]==['h','d','t']
    assert len(assets)==5
    assert planning_catalog({'house':'new','room_id':'R'},assets,{'rooms':[room]})==room_eligible_catalog(assets,room)


def test_room_absent_from_contract_fails_before_any_native_work():
    with pytest.raises(ValueError,match='planning room absent'):
        planning_catalog({'house':'new','room_id':'R'},[],{'rooms':[]})


def test_seated_role_uses_its_own_legal_category():
    human={'asset_id':'h','kind':'human'};room={'family':'mp3d','categories':['standing_human']}
    assert room_eligible_catalog([human],room)==[human]
    assert room_eligible_catalog([human],room,{'h':{'action':'sit'}})==[]
