from types import SimpleNamespace
import pytest
from avengine.construction.joint import CheckResult
from avengine.assets.animal_motion.partial_witnesses import retain_family_witnesses
pytestmark=pytest.mark.fast_unit


def plan_fixture():
    items=[{'config':{'query':{'qa_id':qa}},'offsets_s':[offset]} for offset in [0.,10.] for qa in ['QA-06','QA-15','QA-16','QA-17']]
    row={'motion_during_condition_plan':{'plans':items},'motion_requests':{'animal':{}},
        'motion_during_condition_requests':[{'qa_id':item['config']['query']['qa_id'],'offsets_s':item['offsets_s']} for item in items],
        'scene_demand':{'condition_families':{'motion_quartet':[]}}}
    return SimpleNamespace(metadata={'row':row,'animal_route':{'emission_slots':[{'offset_s':0.},{'offset_s':10.}]}})


def test_one_bad_muzzle_window_does_not_drop_other_windows_or_families():
    plan=plan_fixture()
    checks=[CheckResult('animal:event:%d:%s'%(i,item['config']['query']['qa_id']),'pass') for i,item in enumerate(plan.metadata['row']['motion_during_condition_plan']['plans'])]
    checks += [CheckResult('animal:muzzle_distance:0','fail','original_catalog_counter_trend'),CheckResult('animal:muzzle_distance:1','pass'),CheckResult('animal:all_delivery_speeds','pass')]
    kept=list(retain_family_witnesses(plan,checks,mover='animal',families=['QA-06','QA-15','QA-16','QA-17']))
    assert all(c.status=='pass' for c in kept)
    row=plan.metadata['row'];assert len(row['motion_during_condition_plan']['plans'])==7
    assert any(item['config']['query']['qa_id']=='QA-15' and item['offsets_s']==[10.] for item in row['motion_during_condition_plan']['plans'])
    assert len(row['motion_requests']['animal']['motion_during_condition_plan_for_fit']['plans'])==7
    assert row['family_witness_retention']['observations'][0]['original_check']['status']=='fail'
    assert len(row['scene_demand']['required_actual_candidate_families'])==4


def test_physical_speed_failure_is_preserved_and_optional_optimisation_is_only_recorded():
    plan=plan_fixture()
    kept=list(retain_family_witnesses(plan,[CheckResult('animal:all_delivery_speeds','fail','actual_illegal_speed'),CheckResult('animal:nearer_call_range:0','fail','optional_optimisation')],mover='animal',families=['QA-06','QA-15','QA-16','QA-17']))
    assert len(kept)==1 and kept[0].status=='fail'
    assert plan.metadata['family_witness_retention']['observations'][0]['original_check']['status']=='fail'


def test_unavailable_family_does_not_remove_entities_sources_or_other_families():
    plan=plan_fixture()
    checks=[CheckResult('animal:muzzle_distance:0','fail'),CheckResult('animal:muzzle_distance:1','fail')]
    assert list(retain_family_witnesses(plan,checks,mover='animal',families=['QA-06','QA-15','QA-16','QA-17']))==[]
    row=plan.metadata['row'];assert row['scene_demand']['required_actual_candidate_families']==['QA-06','QA-16','QA-17']
    assert row['motion_quartet_planned_verification']['QA-15'] is False
    assert len(row['motion_during_condition_plan']['plans'])==6
