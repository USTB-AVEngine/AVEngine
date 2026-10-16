import pytest
from tools.timeline import monitor_v7_acceptance as monitor
pytestmark=pytest.mark.fast_unit


def test_readonly_baseline_parent_root_does_not_double_count_owned_work(monkeypatch):
    manifest={'output_root':'/baseline/phase2','peer_output_roots':['/baseline','/other']}
    own=[{'pid':1,'pss_bytes':10},{'pid':2,'pss_bytes':20}];all_records=[*own,{'pid':3,'pss_bytes':30}]
    monkeypatch.setattr(monitor,'output_processes',lambda roots:own if roots==[manifest['output_root']]else all_records)
    peers=monitor.peer_processes(manifest)
    assert [p['pid']for p in peers]==[3]
    assert sum(r['pss_bytes']for r in own+peers)==60
    assert monitor.peer_processes(manifest,own_pids={1,2})==peers


def test_no_peer_roots_needs_no_process_tree_scan(monkeypatch):
    monkeypatch.setattr(monitor,'output_processes',lambda roots:pytest.fail('unnecessary process scan'))
    assert monitor.peer_processes({'output_root':'/own'})==[]


def test_new_house_feedback_counts_only_current_successful_family_houses(tmp_path):
    import json
    jobs=tmp_path/'evidence/jobs';jobs.mkdir(parents=True)
    tasks=[]
    for slot,phase,house,status,counts in [(0,'v1','old','passed',{'QA-14':20}),(1,'phase2','a','passed',{'QA-14':1}),(2,'phase2','a','passed',{'QA-14':2}),(3,'phase2','b','failed',{}),(4,'phase2','c','passed',{'QA-03':1})]:
        tasks.append({'slot':slot,'generation_phase':phase,'target_family':'QA-14','row':{'house':house}})
        (jobs/f'{slot:03d}.json').write_text(json.dumps({'status':status,'house':house,'reader':{'status':'pass'},'export':{'question_family_counts':counts}}))
    manifest={'parents':tasks,'new_family_house_targets':{'QA-14':10},'new_family_house_epoch':'phase2'}
    assert monitor.new_family_house_gaps(manifest,tmp_path)=={'QA-14':9}
    assert monitor.new_family_house_gaps({'parents':tasks},tmp_path)=={}


def test_frozen_table_surplus_does_not_hide_requested_new_house_deficit():
    task={'slot':100,'target_family':'QA-14'};other={'slot':0,'target_family':'V7-CLASS34'}
    args=({'QA-14':111,'V7-CLASS34':30},{},0.,{}, {})
    assert monitor.construction_priority(task,*args,new_house_gaps={'QA-14':5})<monitor.construction_priority(other,*args,new_house_gaps={'QA-14':5})
