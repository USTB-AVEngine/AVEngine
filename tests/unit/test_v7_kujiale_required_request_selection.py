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
