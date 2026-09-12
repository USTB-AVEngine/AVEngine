import importlib.util
from copy import deepcopy
import json
from pathlib import Path
from avengine.qa.unified_catalog import publishable_query_window
from avengine.dataset.production_spec import production_request_from_legacy

spec=importlib.util.spec_from_file_location("sparse_backfill", Path(__file__).parents[2]/"tools/dataset/backfill_sparse_qa.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_question_forms_and_multiple_queries_cannot_inflate_scene_count():
    config={"baseline_counts":{"QA-07":1},"minimum_by_qa":{"QA-07":20}}
    state={"sources":{"new":{"generated_qa_ids":["QA-07"],"new_physical_scene":True},
                      "old":{"generated_qa_ids":["QA-07"],"new_physical_scene":False}}}
    assert module.counts(state, config)=={"QA-07":2}
    assert module.deficits(state, config)=={"QA-07":18}


def test_resume_accounting_reads_each_wave_once_and_reserves_unknown_audio(tmp_path):
    p=tmp_path/"waves/0001/run/state.json";p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"native_accounting_totals":{"native_visual_worlds":2,
         "native_acoustic_launch_attempts":2,"native_acoustic_contexts_known":4}}))
    assert module.totals(tmp_path)=={"visual":2,"audio":2,"rlr_reserved":8}
    assert module.totals(tmp_path)=={"visual":2,"audio":2,"rlr_reserved":8}


def test_scene_signature_ignores_run_and_seed_but_detects_trajectory():
    plan={"episode_id":"one","seed":1,"scene":{"room_id":"room"},"clock":{"frame_count":150},
          "visual_plan":{"camera":{"candidate_id":"A","position_m":[0,1,0]},
           "actors":[{"actor_id":"source1","asset_id":"person"}],
           "frames":[{"actor_states":[{"actor_id":"source1","root_transform":{"translation_m":[0,0,0]}}]}]}}
    other=deepcopy(plan);other["episode_id"]="two";other["seed"]=2;other["visual_plan"]["camera"]["candidate_id"]="B"
    assert module.scene_key(plan)==module.scene_key(other)
    other["visual_plan"]["frames"][0]["actor_states"][0]["root_transform"]["translation_m"][0]=1
    assert module.scene_key(plan)!=module.scene_key(other)


def test_entry_public_window_contains_observed_entry():
    facts={"time":{"frame_rate_hz":15,"frame_count":150},"sampling":{"time_display_precision":0}}
    result=publishable_query_window(facts,[47,120],start_rounding="floor")
    assert result["query_window_s"]==[3,8]
    assert result["query_window_s"][0]<=47/15<result["query_window_s"][1]
    assert publishable_query_window(facts,[47,120])["query_window_s"]==[4,8]

def test_parallel_execution_is_bounded_without_changing_native_caps(tmp_path):
    module.write(tmp_path/"execution_options.json",
                 {"planning_workers":4,"wave_size":6,"max_parallel":6})
    assert module.execution_options(tmp_path)=={
        "planning_workers":4,"wave_size":6,"max_parallel":6}
    module.write(tmp_path/"execution_options.json", {"planning_workers":5})
    import pytest
    with pytest.raises(ValueError):
        module.execution_options(tmp_path)


def test_parallel_cpu_job_writes_a_reusable_result_without_native(tmp_path, monkeypatch):
    source={"episode_id":"independent", "request":{"seed":4}}
    candidate=tmp_path/"candidate"
    def plan(request, output, **kwargs):
        (output/"plan").mkdir(parents=True)
        data={"scene":{"room_id":"room"},"clock":{"frame_count":150},
              "visual_plan":{"camera":{},"actors":[],"frames":[]}}
        module.write(output/"plan/episode_plan.json", data)
        return {"plan":str(output/"plan/episode_plan.json")}
    monkeypatch.setattr(module,"plan_visual_variant",plan)
    result=module.plan_candidate((1,source,candidate,"template"))
    assert result["outcome"]=="cpu_plan_accepted"
    assert module.read(candidate/"cpu_result.json")==result
    assert not (candidate/"episode/capture").exists()
