import pytest
from avengine.timeline.v7_fresh_acceptance_planning import spread_legal_rooms
pytestmark=pytest.mark.fast_unit


def test_house_spread_excludes_test_and_uses_room_inventory_without_saved_scene():
    houses=[];rooms=[]
    for house in ['a','b','test']:
        houses.append({'house':house,'split':'train','checked_room_ids':[house+'1',house+'2'],
            'checked_rooms':[{'room_label':house+str(i),'floor_polygon':{'type':'Polygon','coordinates':[[[0,0],[i,0],[i,i],[0,i],[0,0]]]}}for i in [1,2]]})
        rooms.extend({'house':house,'room_id':house+str(i),'family':'hm3d','categories':['standing_human']}for i in [1,2])
    result=spread_legal_rooms({'rooms':rooms,'test_houses':['test']},houses,{'hm3d':2},27)
    assert {r['house']for r in result}=={'a','b'}
    assert all(r['room']['room_id'].endswith('2')for r in result)
    assert all(r['handwritten_scene_configuration_inputs_read']==[]for r in result)
    assert result==spread_legal_rooms({'rooms':rooms,'test_houses':['test']},houses,{'hm3d':2},27)
    with pytest.raises(ValueError,match='independent house inventory'):spread_legal_rooms({'rooms':rooms,'test_houses':['test']},houses,{'hm3d':3},27)
