from collections import Counter
import numpy as np
import pytest
from avengine.timeline.v7_demand import demand_snapshot,default_targets,record_selected_parent,read_demand_state
from avengine.timeline.v7_wet_support import event_wet_support


def test_default_target_keeps_thirty_percent_vision():
    targets=default_targets()
    assert sum(targets.values())==pytest.approx(1)
    assert sum(targets[f] for f in ('QA-05','QA-07','QA-10','QA-11','QA-14','QA-20'))==pytest.approx(.3)


def test_feedback_prioritizes_underproduced_family_from_actual_counts():
    targets={'QA-06':.5,'QA-19':.5}
    s=demand_snapshot({'logical_questions':100,'family_counts':{'QA-19':100},'target_shares':targets},next_questions=14)
    assert s['family_deficits']['QA-06']==57
    assert s['family_deficits']['QA-19']==0
    assert s['selection_counts']['QA-06']<s['selection_counts']['QA-19']


def test_feedback_rejects_forms_or_candidates_counted_as_actual_items():
    with pytest.raises(ValueError):
        demand_snapshot({'logical_questions':14,'family_counts':{'QA-06':28}})


def test_completed_parent_resume_is_idempotent_and_failed_parent_is_ignored(tmp_path):
    p=tmp_path/'state.json'
    r={'status':'integration_pilot_admitted','scene_id':'parent',
       'windows':[{'logical_questions':14,'family_counts':{'QA-06':3,'QA-19':11}}]}
    record_selected_parent(p,r);record_selected_parent(p,r)
    assert read_demand_state(p)['logical_questions']==14
    record_selected_parent(p,{'status':'failed_not_formal','scene_id':'failed'})
    r['windows'][0]['family_counts']={'QA-06':7,'QA-19':7}
    record_selected_parent(p,r)
    assert read_demand_state(p)['family_counts']=={'QA-06':7,'QA-19':7}


def test_wet_silence_bound_contains_every_exact_convolution_sample():
    dry=np.zeros(100);dry[20:25]=[1,-2,3,1,2]
    kernels=np.zeros((4,20));kernels[:,4:9]=np.arange(20).reshape(4,5)+1
    result=event_wet_support(dry,[{'event_id':'e','output_start_sample':10,'output_end_sample_exclusive':40}],[kernels])['e']
    exact=np.stack([np.convolve(dry,h) for h in kernels])
    nonzero=np.flatnonzero(np.any(exact!=0,axis=0));lo,hi=result['support_samples_global']
    assert (lo,hi)==(int(nonzero[0]),int(nonzero[-1])+1)
    assert hi>25  # measured native tail is retained after the dry source stops
    assert result['silence_threshold_added'] is False


def test_moving_native_delay_spread_expands_silence_bound():
    dry=np.zeros(100);dry[20]=1
    h=np.zeros((4,50));h[:,4]=1;h[:,40]=.01
    r=event_wet_support(dry,[{'event_id':'e','output_start_sample':10,'output_end_sample_exclusive':30}],[h],delay_spread_samples=8)['e']
    assert r['support_samples_global']==[16,69]


def test_overlapping_endpoint_events_do_not_forge_per_event_silence():
    dry=np.ones(100);h=np.ones((4,20))
    events=[{'event_id':'a','output_start_sample':10,'output_end_sample_exclusive':40},
            {'event_id':'b','output_start_sample':30,'output_end_sample_exclusive':50}]
    assert event_wet_support(dry,events,[h])=={}


def test_all_33_families_have_scene_requirements_and_minimum_share():
    from avengine.timeline.v7_demand import FAMILY_SCENE_REQUIREMENTS
    from avengine.qa.v7_families import FAMILIES,PANELS
    expected=set(FAMILIES+PANELS+("V7-CLASS34","V7-CONTENT-BIND"))
    assert len(expected)==33
    assert set(default_targets())==set(FAMILY_SCENE_REQUIREMENTS)==expected
    assert min(default_targets().values())>=.01
    assert all(r["conditions"] for r in FAMILY_SCENE_REQUIREMENTS.values())
