import pytest
from tools.timeline.run_v7_motion_quartet_batch import candidate_rooms
pytestmark=pytest.mark.fast_unit

def test_room_constructor_uses_only_legal_training_cells_and_excludes_touched_or_test_houses():
    houses=[{'house':name,'family':family,'split':'train','acoustic_status':'pass','checked_rooms':[{'room_label':'large','floor_area_m2':20},{'room_label':'legal','floor_area_m2':10}]}for name,family in [('good-h','hm3d'),('good-m','mp3d'),('touched','hm3d'),('test','mp3d')]]
    contract={'test_houses':['test'],'rooms':[{'house':h['house'],'room_id':'legal','categories':['standing_human']}for h in houses]}
    choices=candidate_rooms(houses,contract,{'touched'})
    assert [h['house']for h,rooms in choices]==['good-h','good-m']
    assert all([r['room_label']for r in rooms]==['legal']for h,rooms in choices)
