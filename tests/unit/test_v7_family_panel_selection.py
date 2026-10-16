from collections import Counter
import pytest
import avengine.timeline.v7_admitted_pool_reselection as selection
pytestmark=pytest.mark.fast_unit


def test_legacy_audio_classifier_does_not_fill_appearance_bound_qa21_target(tmp_path,monkeypatch):
    entries=[]
    for i,panel in enumerate(['STARSS23_v0.2','v6','v6']):
        r={'question_id':str(i),'window_id':'p__w00','window_start_s':0.,'independent_parent_scene_id':'p','house':'h','qa_id':'QA-21','protocol_panel':panel,'question_en':'Which sound category best describes the sound?','modality':'audiovisual','canonical_answer':'cat'}
        entries.append({'row':r,'key':['p','QA-21',i,'cat']})
    monkeypatch.setattr(selection,'parent_entries',lambda *args,**kwargs:{'entries':entries,'new_visual_candidate_rows':0,'source_seal':'original','proofs':[]})
    receipt={'scene_id':'p','slot':0,'independent_questions':3}
    rows,report=selection.balanced_batch([receipt],tmp_path,family_protocol_panels={'QA-21':'v6'})
    assert len(rows)==2 and all(r['protocol_panel']=='v6'for r in rows)
    assert report['candidate_rows_outside_requested_family_panel']=={'QA-21:STARSS23_v0.2':1}
    assert all(r['canonical_answer']=='cat'for r in rows)
