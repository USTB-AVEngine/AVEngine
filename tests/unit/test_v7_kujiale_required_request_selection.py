import pytest
pytestmark=pytest.mark.fast_unit

from tools.timeline.probe_kujiale_automatic_request import select_ordinary_task


def task(i,house):return {'slot':i,'automatic_registered_room':True,'row':{'house':house,'room_id':'R0'}}


def test_required_categories_prefer_current_legal_witness_without_scene_overrides():
    tasks=[task(0,'a'),task(1,'b'),task(2,'c')]
    contract={'rooms':[{'key':'a::R0','categories':['human.standing']},{'key':'b::R0','categories':['human.standing','animal.dog']},{'key':'c::R0','categories':['human.standing','animal.dog']}]}
    chosen=select_ordinary_task(tasks,contract,{'human.standing','animal.dog'},passed_houses={'b'})
    assert chosen is tasks[2]
    assert chosen['row']=={'house':'c','room_id':'R0'}


def test_missing_native_witness_still_gets_constructed_instead_of_skipped():
    tasks=[task(0,'a')]
    assert select_ordinary_task(tasks,{'rooms':[{'key':'a::R0','categories':[]}]},{'human.standing','animal.dog'})is tasks[0]


def test_required_human_animal_group_uses_room_area_and_avoids_exhausted_room():
    tasks=[task(0,'a'),task(1,'b'),task(2,'c')]
    contract={'rooms':[{'key':h+'::R0','categories':['human.standing','animal.dog']}for h in ('a','b','c')]}
    required={'human.standing','animal.dog'}
    areas={'a::R0':5,'b::R0':30,'c::R0':20}
    assert select_ordinary_task(tasks,contract,required,room_areas=areas)is tasks[1]
    assert select_ordinary_task(tasks,contract,required,room_areas=areas,attempted_rooms={('b','R0')})is tasks[2]


def test_required_animal_attempts_continue_beyond_the_old_cap_and_stop_at_the_deadline():
    import datetime
    from tools.timeline.probe_kujiale_automatic_request import construction_attempts
    before=datetime.datetime(2026,10,15,14,59,tzinfo=datetime.timezone.utc)
    deadline=before+datetime.timedelta(minutes=1)
    moments=iter([before,before,deadline])
    assert list(construction_attempts(12,deadline,clock=lambda:next(moments)))==[12,13]
