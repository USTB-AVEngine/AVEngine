from pathlib import Path
import json,os
import pytest
pytestmark=pytest.mark.fast_unit


def test_registered_constructor_result_reaches_registered_generator_with_its_admission(tmp_path,monkeypatch):
    import avengine
    import tools.timeline.run_v7_acceptance as worker
    import avengine.timeline.v7_demand as demand
    import avengine.timeline.v7_family_dispatch as dispatch
    import avengine.timeline.v7_unattended_resources as resources
    import avengine.timeline.v7_cpu_visual_runtime as visual
    import avengine.timeline.v7_native_resource_slots as slots
    import avengine.timeline.v7_registered_generation as registered
    class SelectedRegistered(BaseException):pass
    row={'house':'h','room_id':'r','scene_id':'p','scene_seed':7,'scene_demand':{'requested_conditions':[]},'members':['v0_a0','v1_a0','v0_a1','v1_a1'],'automatic_construction':{'family':'V7-CONTENT-BIND'}}
    data={'runtime':{'semantic_manifest':'semantic'},'assets':[],'houses':{'houses':[{'house':'h'}]},
        'contract':{'rooms':[{'house':'h','room_id':'r'}],'inputs':{'source_admission':{'path':'source'}}},
        'source':{'records':[]},'semantic':{},'human_phase2':{'tag':'phase2'},'human_registered':{'tag':'registered'}}
    monkeypatch.setattr(worker,'load',lambda path:data[str(path)])
    monkeypatch.setattr(os,'getpriority',lambda *args:10)
    monkeypatch.setattr(os,'sched_getaffinity',lambda pid:{16,17})
    monkeypatch.setattr(os,'sched_setaffinity',lambda *args:None)
    monkeypatch.setattr(resources,'configure_parent_storage',lambda *args:None)
    monkeypatch.setattr(visual,'configure_cpu_visual',lambda *args,**kwargs:{'LP_NUM_THREADS':1})
    monkeypatch.setattr(demand,'read_demand_state',lambda *args:{})
    monkeypatch.setenv(demand.SELECTION_FEEDBACK_ENV,'prior')
    monkeypatch.setattr(dispatch,'construct_requested_scene',lambda *args,**kwargs:{'profile':'registered','row':dict(row)})
    monkeypatch.setattr(slots,'acquire_native_resource',lambda *args:None)
    def generate(used,house,inputs,catalog,source,human,output):
        assert human['tag']=='registered'and used['members']==row['members']
        assert inputs['phase2']is False and used['window_start_ticks']==[0,50,100,150,200]
        raise SelectedRegistered()
    monkeypatch.setattr(registered,'generate_registered_parent',generate)
    manifest={'parents':[{'slot':0,'device':'cpu','profile':'phase2','constructor':'automatic_family','target_family':'V7-CONTENT-BIND','row':row}],
        'output_root':str(tmp_path),'cpu_affinity_ids':[16,17],'worktree':str(Path(avengine.__file__).resolve().parents[2]),'code_commit':'unit',
        'runtime_cpu':'runtime','assets_cpu':'assets','houses_cpu':'houses','contract':'contract','human_cpu':'human_phase2','human_registered_cpu':'human_registered'}
    with pytest.raises(SelectedRegistered):worker.run_worker(manifest,0)
    saved=json.loads((tmp_path/'inputs/dispatch_profiles/000.json').read_text())
    assert saved['profile']=='registered'
    assert manifest['parents'][0]['profile']=='phase2'
