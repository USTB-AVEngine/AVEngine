import pytest
from avengine.qa.audio_variants import schedule_variant
from avengine.qa import unified_catalog as u

def test_audio_targets_preserve_complete_clips_tail_and_silent_candidates():
    clock={"sample_rate_hz":100,"sample_count":1000}
    sounds=[{"sample_rate_hz":100,"sample_count":200,"sound_asset_id":"a"},
            {"sample_rate_hz":100,"sample_count":240,"sound_asset_id":"b"}]
    for profile,n in [("single_first",1),("single_second",1),("sequential",2),("overlap",2),("three_events",3)]:
        rows=schedule_variant(sounds,["person1","person2"],clock,profile=profile,seed=2)
        assert len(rows)==n
        assert len({r["event_id"] for r in rows})==n
        assert max(r["end_sample"] for r in rows)<=700
        assert all(r["end_sample"]-r["start_sample"]==r["sample_count"] for r in rows)
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                if a["actor_id"]==b["actor_id"]:
                    assert min(a["end_sample"],b["end_sample"])<=max(a["start_sample"],b["start_sample"])
    with pytest.raises(ValueError,match="not a crop"):
        schedule_variant([{**s,"sample_count":500} for s in sounds],["a","b"],clock)

def test_overlap_answer_matches_the_requested_compiler_branch():
    from avengine.qa.batch_delivery import observed_branch
    for token,branch in [("yes","overlap"),("no","disjoint")]:
        item={"truth":{"value":token},"forms":{"open":{"truth":token}}}
        assert u._emitted_branch("QA-05",item)==branch
        assert u._qa_target_item_matches({"branch":branch},"QA-05",item,{})
        assert observed_branch(item,("overlap","disjoint"),qa_id="QA-05")==branch

def test_scene_sound_candidates_replace_a_large_unobserved_taxonomy():
    from avengine.qa.choice_support import apply_choice_support
    item={"qa_id":"QA-21","forms":{"open":{"truth":"cough"},
          "mcq":{"options":[{"value":v,"label_en":v} for v in ["speech_playback","cough","dog_bark","printer"]],
                 "gold":{"correct_index":1}}},
          "model_input":{"mcq":{"options":[]},"open":{}},"evidence":{"observed_sound_classes":["speech_playback","cough"]}}
    fresh=apply_choice_support(item)
    assert "mcq" not in fresh["forms"]
    assert fresh["forms"]["open"]["truth"]=="cough"
    assert fresh["evidence"]["mcq_support"]["real_candidate_count"]==2


def test_scheduler_uses_declared_timeline_and_emitter_endpoints():
    sounds = [{"sample_rate_hz": 100, "sample_count": 120} for _ in range(2)]
    rows = schedule_variant(sounds, ["dog", "cat"],
                            {"sample_rate_hz": 100, "sample_count": 1000, "time_base_hz": 1000},
                            endpoints={"dog": "dog_snout", "cat": "cat_snout"}, seed=4)
    assert {r["source_endpoint_id"] for r in rows} == {"dog_snout", "cat_snout"}
    assert all(r["start_tick"] == 10 * r["start_sample"] for r in rows)
    with pytest.raises(ValueError, match="integer"):
        schedule_variant(sounds, ["a", "b"], {"sample_rate_hz": 100, "sample_count": 1000, "time_base_hz": 333})


def planner_fixture(tmp_path, monkeypatch):
    import json
    import numpy as np
    import soundfile as sf
    from avengine.rooms import qa_delivery
    from avengine.dataset import source_capabilities
    monkeypatch.setattr(qa_delivery, "_asset_registry", lambda *args: {"blue": {}, "green": {}})
    monkeypatch.setattr(source_capabilities, "sound_compatibility",
                        lambda asset, sound, config: {"compatible": sound["sound_class"] != "engine"})
    sources = []
    for i in range(2):
        root = tmp_path/f"source{i}"
        (root/"plan").mkdir(parents=True)
        (root/"request.json").write_text("{}")
        plan = {"episode_id": f"world{i}", "clock": {"sample_rate_hz": 100, "sample_count": 1000},
                "visual_plan": {"actors": [{"actor_id": "a", "asset_id": "blue"}, {"actor_id": "b", "asset_id": "green"}]}}
        (root/"plan/episode_plan.json").write_text(json.dumps(plan))
        sources.append(str(root))
    sounds = []
    for cls in ["speech_playback", "cough", "laughter", "engine"]:
        path = tmp_path/f"{cls}.wav"
        sf.write(path, np.ones(100, dtype=np.float32)*.1, 100)
        sounds.append({"path": str(path), "sound_asset_id": cls, "sound_class": cls,
                       "sample_rate_hz": 100, "sample_count": 100})
    pool = tmp_path/"pool.json"
    pool.write_text(json.dumps({"sounds": sounds}))
    return {"sources": sources, "pool": str(pool), "variants": 16, "seed": 42}, pool


def test_planner_balances_answers_and_appearances_without_manual_jobs(tmp_path, monkeypatch):
    from collections import Counter, defaultdict
    from avengine.qa.audio_variants import plan_audio_jobs
    config, _ = planner_fixture(tmp_path, monkeypatch)
    plan = plan_audio_jobs(config, repository=tmp_path)
    assert plan == plan_audio_jobs(config, repository=tmp_path)
    assert not plan["deficits"] and len(plan["jobs"]) == 16
    assert set(plan["planned_class_counts"].values()) == {8}
    assert set(plan["planned_source_counts"].values()) == {8}
    assert Counter(j["qa01_answer"] for j in plan["jobs"]) == {"yes": 8, "no": 8}
    assert Counter(j["event_count"] for j in plan["jobs"]) == {1: 8, 2: 6, 3: 2}
    assert Counter(j["qa05_branch"] for j in plan["jobs"] if j["qa05_branch"]) == {"overlap": 4, "disjoint": 4}
    assert all("engine" not in j["sounds"] for j in plan["jobs"])
    assert plan["new_visual_renders"] == 0


def test_planner_reports_full_clip_deficits_and_rejects_wrong_metadata(tmp_path, monkeypatch):
    import json
    import numpy as np
    import soundfile as sf
    from avengine.qa.audio_variants import plan_audio_jobs
    config, pool = planner_fixture(tmp_path, monkeypatch)
    data = json.loads(pool.read_text())
    for sound in data["sounds"]:
        sf.write(sound["path"], np.ones(800)*.1, 100)
        sound["sample_count"] = 800
    pool.write_text(json.dumps(data))
    result = plan_audio_jobs(config, repository=tmp_path)
    assert not result["jobs"] and len(result["deficits"]) == 16
    data["sounds"][0]["sample_count"] = 100
    pool.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata differs"):
        plan_audio_jobs(config, repository=tmp_path)
