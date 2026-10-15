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
    assert len(rows)==101
    assert Counter(r['modality']for r in rows)=={'vision':30,'audio':25,'audiovisual':46}
    assert Counter(r['qa_id']for r in rows)['rare']==2
    assert len({r['independent_parent_scene_id']for r in rows})==4
    assert audit['independent_candidates_not_selected']==19


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


def test_twenty_row_quantization_does_not_block_real_family_floors():
    entries=[]
    groups=[('vision','QA-14',23),('vision','QA-11',4),('vision','QA-10',2)]
    groups += [('audio','a'+str(i),39)for i in range(8)]
    groups += [('audiovisual','av'+str(i),25)for i in range(13)]
    for modality,family,count in groups:
        for i in range(count):
            row={'question_id':family+'_'+str(i),'window_id':'p__w00','window_start_s':0.,
                 'independent_parent_scene_id':'p'+str(i%13),'qa_id':family,'modality':modality,'canonical_answer':'yes'}
            entries.append({'row':row,'key':[row['independent_parent_scene_id'],family,i,'yes']})
    rows,audit=select_independent_rows(entries)
    assert len(rows)==98
    assert Counter(r['modality']for r in rows)=={'vision':29,'audio':25,'audiovisual':44}
    counts=Counter(r['qa_id']for r in rows)
    assert counts['QA-10']==2
    assert all(n>=3 for f,n in counts.items()if f!='QA-10')
    assert len({r['independent_parent_scene_id']for r in rows})==13
    assert audit['physical_duplicates_dropped']==0


def test_small_valid_batch_uses_per_row_rounding():
    from avengine.timeline.v7_post_dedup_selection import rounded_modality_quotas
    assert rounded_modality_quotas({'vision':2,'audio':2,'audiovisual':3}, {'vision':2,'audio':2,'audiovisual':3})=={'vision':2,'audio':2,'audiovisual':3}


def test_true_visual_shortfall_still_requires_more_scene_conditions():
    from avengine.timeline.v7_post_dedup_selection import rounded_modality_quotas
    with pytest.raises(ValueError,match='available family floor'):
        rounded_modality_quotas({'vision':1,'audio':100,'audiovisual':100},{'vision':1,'audio':3,'audiovisual':3})
