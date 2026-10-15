import pytest
from tools.timeline.run_v7_motion_quartet_batch import candidate_rooms
pytestmark=pytest.mark.fast_unit

def test_room_constructor_uses_only_legal_training_cells_and_excludes_touched_or_test_houses():
    houses=[{'house':name,'family':family,'split':'train','acoustic_status':'pass','checked_rooms':[{'room_label':'large','floor_area_m2':20},{'room_label':'legal','floor_area_m2':10}]}for name,family in [('good-h','hm3d'),('good-m','mp3d'),('touched','hm3d'),('test','mp3d')]]
    for house in houses:
        for room in house['checked_rooms']:
            side=room['floor_area_m2']**.5
            room['floor_polygon']={'type':'Polygon','coordinates':[[[0,0],[side,0],[side,side],[0,side],[0,0]]]}
    contract={'test_houses':['test'],'rooms':[{'house':h['house'],'room_id':'legal','categories':['standing_human']}for h in houses]}
    choices=candidate_rooms(houses,contract,{'touched'})
    assert [h['house']for h,rooms in choices]==['good-h','good-m']
    assert all([r['room_label']for r in rooms]==['legal']for h,rooms in choices)


def test_explicit_legal_room_is_not_discarded_by_new_room_size_ranking():
    room={'room_label':'narrow','floor_area_m2':6.,'floor_polygon':{'type':'Polygon','coordinates':[[[0,0],[4,0],[4,1.5],[0,1.5],[0,0]]]}}
    houses=[{'house':'new','family':'hm3d','split':'train','acoustic_status':'pass','checked_rooms':[room]}]
    contract={'test_houses':[],'rooms':[{'house':'new','room_id':'narrow','categories':['standing_human']}]}
    assert candidate_rooms(houses,contract,room_id='narrow')==[(houses[0],[room])]
