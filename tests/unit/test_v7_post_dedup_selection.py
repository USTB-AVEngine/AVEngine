from collections import Counter
import pytest
from avengine.timeline.v7_post_dedup_selection import select_independent_rows
pytestmark=pytest.mark.fast_unit


def candidates():
    result=[]
    for m,n in [('vision',30),('audio',40),('audiovisual',50)]:
        for i in range(n):
            r={'question_id':m+str(i),'window_id':'p__w00','window_start_s':0.,
               'independent_parent_scene_id':'p'+str(i%4),'qa_id':m,'modality':m,'canonical_answer':'yes'}
            result.append({'row':r,'key':[r['independent_parent_scene_id'],m,i,'yes']})
    return result


def test_balances_after_dedup_and_preserves_scarce_family_and_parents():
    entries=candidates()
    for entry in entries[-2:]:entry['row']['qa_id']='rare'
    rows,audit=select_independent_rows(entries)
    assert len(rows)==100
    assert Counter(r['modality']for r in rows)=={'vision':30,'audio':25,'audiovisual':45}
    assert Counter(r['qa_id']for r in rows)['rare']==2
    assert len({r['independent_parent_scene_id']for r in rows})==4
    assert audit['independent_candidates_not_selected']==20


def test_duplicate_later_window_does_not_increase_capacity_or_take_earliest_place():
    entries=candidates();early=entries[0]
    later={'row':{**early['row'],'question_id':'duplicate','window_id':'p__w01','window_start_s':5.},'key':early['key']}
    rows,audit=select_independent_rows([later,*entries])
    assert audit['physical_duplicates_dropped']==1
    assert 'duplicate'not in {r['question_id']for r in rows}


def test_same_wording_with_different_physical_answer_remains_independent():
    entries=candidates();q=entries[0]
    entries.append({'row':{**q['row'],'question_id':'different_answer','canonical_answer':'no'},'key':[*q['key'][:-1],'no']})
    _,audit=select_independent_rows(entries)
    assert audit['capacity_by_modality']['vision']==31


def test_motion_batch_floor_overrides_the_ordinary_three_question_floor():
    entries=candidates()
    for i,e in enumerate(entries):
        if e['row']['modality']=='audiovisual':e['row']['qa_id']='QA-15'if i%2 else'other_AV'
    rows,audit=select_independent_rows(entries,family_minimums={'QA-15':25})
    assert Counter(r['qa_id']for r in rows)['QA-15']==25
    assert audit['available_family_floors']['QA-15']==25
