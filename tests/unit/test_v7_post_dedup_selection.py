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


def test_visual_capacity_shortage_is_reported_without_rejecting_valid_answers():
    entries=candidates();entries=[e for e in entries if e['row']['modality']!='vision']
    rows,report=select_independent_rows(entries)
    assert len(rows)==90
    assert report['status']=='capacity_limited_partial'
    assert report['actual_proportions']['vision']==0
    assert report['quality_observations']
    assert report['source_candidates_and_answers_changed']is False


def test_contradictory_answers_are_still_rejected_in_partial_batches():
    from copy import deepcopy
    entries=candidates();bad=deepcopy(entries[0]);bad['row']['canonical_answer']='wrong'
    with pytest.raises(ValueError,match='inconsistent canonical'):
        select_independent_rows(entries+[bad])


def test_family_house_floor_keeps_independent_support_from_two_houses():
    entries=candidates()
    for e in entries:e['row']['house']='h0'
    for e in entries[-5:]:e['row']['qa_id']='rare'
    entries[-1]['row'].update(house='h1',independent_parent_scene_id='p4')
    entries[-1]['key'][0]='p4'
    entries[31]['row'].update(house='h1',independent_parent_scene_id='p4')
    entries[31]['key'][0]='p4'
    rows,report=select_independent_rows(entries,minimum_houses_per_family=2)
    assert {r['house']for r in rows if r['qa_id']=='rare'}=={'h0','h1'}
    assert report['selected_family_house_counts']['rare']==2
    assert Counter(r['modality']for r in rows)=={'vision':30,'audio':25,'audiovisual':46}
    assert all(r['canonical_answer']=='yes'for r in rows)


def test_house_floor_reports_the_available_single_house_without_inventing_one():
    entries=candidates()
    for e in entries:e['row']['house']='only_real_house'
    rows,report=select_independent_rows(entries,minimum_houses_per_family=2)
    assert report['available_family_house_floors']=={'vision':1,'audio':1,'audiovisual':1}
    assert set(r['house']for r in rows)=={'only_real_house'}


def test_configured_special_family_floor_survives_final_modality_sampling():
    from avengine.timeline.v7_post_dedup_selection import manifest_family_minimums
    entries=candidates()
    for e in entries:
        if e['row']['modality']=='vision':e['row']['qa_id']='QA-10'
    manifest={'family_question_minimums':{'QA-10':30},'motion_quartet_final_targets':{'qa_ids':['QA-15'],'minimum_independent_questions_per_family':3}}
    rows,report=select_independent_rows(entries,family_minimums=manifest_family_minimums(manifest))
    assert Counter(r['qa_id']for r in rows)['QA-10']==30
    assert Counter(r['modality']for r in rows)=={'vision':30,'audio':25,'audiovisual':46}
    assert report['source_candidates_and_answers_changed']is False


def test_per_family_house_targets_do_not_add_global_sampling_requirements():
    from avengine.timeline.v7_post_dedup_selection import manifest_house_minimums
    entries=candidates();av=0
    for e in entries:
        r=e['row'];r['house']='common_house'
        if r['modality']=='audiovisual':
            if av<40:
                r['qa_id']='QA-27'
                house=0 if av<31 else av-30
                r['house']='target_house_'+str(house)
                r['independent_parent_scene_id']='target_parent_'+str(house)
                e['key'][0]=r['independent_parent_scene_id']
            else:r['qa_id']='other_AV'
            av+=1
    m={'minimum_family_distinct_houses':1,'family_distinct_house_minimums':{'QA-27':10}}
    rows,report=select_independent_rows(entries,family_minimums={'QA-27':30},minimum_houses_per_family=manifest_house_minimums(m))
    target=[r for r in rows if r['qa_id']=='QA-27']
    assert len(target)>=30 and len({r['house']for r in target})==10
    assert report['available_family_house_floors']['QA-27']==10
    assert report['available_family_house_floors']['vision']==1
    assert report['available_family_house_floors']['audio']==1
    assert Counter(r['modality']for r in rows)=={'vision':30,'audio':25,'audiovisual':46}
    assert report['source_candidates_and_answers_changed']is False


def test_house_target_manifest_preserves_original_scalar_default():
    from avengine.timeline.v7_post_dedup_selection import manifest_house_minimums
    assert manifest_house_minimums({'minimum_family_distinct_houses':2})==2
    assert manifest_house_minimums({'minimum_family_distinct_houses':2,'family_distinct_house_minimums':{'QA-27':10}})=={'*':2,'QA-27':10}


def test_full_view_keeps_earliest_window_and_never_applies_modality_sampling():
    from avengine.timeline.v7_post_dedup_selection import all_independent_rows
    def entry(key,start,answer='yes'):
        return {'key':key,'row':{'window_start_s':start,'window_id':'w'+str(start),'question_id':str(key)+str(start),
            'independent_parent_scene_id':'parent','canonical_answer':answer,'modality':'audio'}}
    rows=all_independent_rows([entry('a',5),entry('b',10),entry('a',0)])
    assert len(rows)==2 and rows[0]['window_start_s']==0
    with pytest.raises(ValueError,match='inconsistent canonical'):
        all_independent_rows([entry('a',0),entry('a',5,'no')])
