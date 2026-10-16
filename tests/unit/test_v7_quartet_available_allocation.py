import json
import pytest
import avengine.timeline.v7_full_question_delivery as delivery
pytestmark=pytest.mark.fast_unit


def test_quad_keeps_four_members_and_fourteen_real_candidates_when_visual_pool_is_empty(tmp_path,monkeypatch):
    names=['v0_a0','v0_a1','v1_a0','v1_a1'];members=[]
    for member in names:
        folder=tmp_path/member/'window';folder.mkdir(parents=True);pub=[];gold=[]
        for i in range(30):
            q={'question_id':str(i),'logical_question_id':str(i),'qa_id':'QA-01'if i<15 else'QA-04','protocol_panel':'v6','question_zh':str(i),'question_en':str(i),'modality':'audio'if i<15 else'audiovisual','form':'open','answer_type':'boolean'}
            pub.append(q);gold.append({'question_id':str(i),'logical_question_id':str(i),'canonical_answer':'yes'})
        (folder/'questions.json').write_text(json.dumps(pub));(folder/'private_answers.json').write_text(json.dumps(gold));members.append({'member':member,'windows_directory':str(folder.parent),'windows':[{'window_id':'window'}]})
    monkeypatch.setattr(delivery,'filter_observed_nuisance_candidates',lambda p,g,w:(p,{}))
    class StopAtOriginalGroupSelector(Exception):pass
    def select(p,g,allocations,**kwargs):
        assert set(allocations)==set(names)
        for a in allocations.values():assert a.get('vision',0)==0 and sum(a.values())==14 and a['audiovisual']>=5
        assert kwargs['binding_items_per_window']==5
        raise StopAtOriginalGroupSelector()
    monkeypatch.setattr('avengine.qa.v7_group_selection.select_counterfactual_group',select)
    with pytest.raises(StopAtOriginalGroupSelector):delivery.finalize_parent_question_pools({'scene_id':'p','members':names,'window_count':1,'questions_per_window':14,'question_quota_base':0},members,tmp_path)
