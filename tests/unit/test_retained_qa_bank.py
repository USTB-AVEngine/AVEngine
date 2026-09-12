import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("bank_cli",REPO/"tools/dataset/generate_retained_qa_bank.py")
bank=importlib.util.module_from_spec(spec);spec.loader.exec_module(bank)

def candidate(qa="QA-01",answer="yes",room="hm3d",asset="asset",sound="sound"):
    item={"qa_id":qa,"question_id":qa,"status":"pass","question":{"en":"Question "+qa,"zh":"问题"+qa},
          "model_input":{"open":{"question_en":"Question "+qa,"question_zh":"问题"+qa}},
          "forms":{"open":{"truth":answer}},"truth":{"value":answer},"evidence":{}}
    return {"item":item,"source":{"facts_path":"facts","room_family":room,"asset_ids":[asset],
            "sound_asset_ids":[sound]},"media":{"video_sha256":"v","audio_sha256":"a"}}

def checkpoint(c):
    return {"status":"pass","source":c["source"],"media":c["media"],"questions":{"items":[c["item"]]}}

def test_option_permutations_not_new_and_conflicting_answers_excluded():
    a=candidate();b=candidate()
    a["item"]["model_input"]["mcq"]={"options":["yes","no"]}
    b["item"]["model_input"]["mcq"]={"options":["no","yes"]}
    unique,stats=bank.unique_candidates([checkpoint(a),checkpoint(b)])
    assert len(unique)==1 and stats["duplicate_candidates"]==1
    unique,stats=bank.unique_candidates([checkpoint(a),checkpoint(candidate(answer="no"))])
    assert not unique and stats["conflicting_observation_groups"]==1

def test_selection_covers_available_types_rooms_assets_and_sounds():
    rows=[candidate(qa=f"QA-{i+1:02d}",room=r,asset=f"a{i}",sound=f"s{i}") for i,r in enumerate(
        ["hm3d","apartment","mp3d","kujiale"])]
    selected=bank.select_balanced(rows*2,4)
    assert {r["source"]["room_family"] for r in selected}=={"hm3d","mp3d","apartment","kujiale"}
    assert len({r["item"]["qa_id"] for r in selected})==4

def test_selection_minimum_per_type_prioritizes_distinct_facts_and_worlds():
    rows=[]
    for qa in ("QA-01","QA-02"):
        for i in range(2):
            row=candidate(qa=qa,asset=f"{qa}-a{i}",sound=f"{qa}-s{i}")
            row["source"].update(facts_path=f"{qa}-facts-{i}",world_id=f"{qa}-world-{i}")
            rows.append(row)
    selected=bank.select_balanced(rows,4,minimum_per_type=2)
    assert {r["item"]["qa_id"] for r in selected}=={"QA-01","QA-02"}
    assert all(sum(r["item"]["qa_id"]==qa for r in selected)==2
               for qa in ("QA-01","QA-02"))
    assert len({r["source"]["facts_path"] for r in selected})==4
    assert len({r["source"]["world_id"] for r in selected})==4

def test_export_reports_minimum_per_type_deficits(tmp_path):
    selected=[candidate("QA-01"),candidate("QA-02"),candidate("QA-02")]
    video=tmp_path/"source.mp4";audio=tmp_path/"source.wav"
    video.write_bytes(b"video");audio.write_bytes(b"audio")
    for i,row in enumerate(selected):
        row["media"].update(video=str(video),audio=str(audio),
                            video_sha256="v",audio_sha256="a")
        row["source"]["facts_path"]=f"source_{i}/facts.json"
    out=tmp_path/"export";out.mkdir()
    report=bank.export(out,selected,[checkpoint(row) for row in selected],{},3,{},
                       minimum_per_type=2)
    assert report["deficits_by_qa"]["QA-01"]==1
    assert report["minimum_per_type_met"] is False
    assert report["status"]=="completed_with_coverage_gaps"

def test_resume_reuses_completed_source_and_exports_without_gold(tmp_path,monkeypatch):
    manifest=tmp_path/"manifest.json";policy=tmp_path/"policy.json";cfg=tmp_path/"config.json"
    sources=[]
    video=tmp_path/"video.mp4";audio=tmp_path/"audio.wav";video.write_bytes(b"v");audio.write_bytes(b"a")
    for i in range(2):
        f=tmp_path/f"facts{i}.json";f.write_text(json.dumps({"index":i}))
        sources.append({"facts_path":str(f),"room_family":"hm3d","asset_ids":[f"a{i}"],"sound_asset_ids":[f"s{i}"]})
    manifest.write_text(json.dumps({"sources":sources}));policy.write_text("{}")
    cfg.write_text(json.dumps({"sources_manifest":str(manifest),"acceptance_policy":str(policy),
                              "target_questions":2,"items_per_type":1,"seed":"test"}))
    calls=[]
    def generate(facts,**kwargs):
        i=facts["index"];calls.append(i)
        return {"items":[candidate(qa=f"QA-{i+1:02d}")["item"]]}
    monkeypatch.setattr(bank,"generate_unified_questions",generate)
    monkeypatch.setattr(bank,"whole_degree_display",lambda x:x)
    monkeypatch.setattr(bank,"validate_source",lambda s,f:{"video":str(video),"audio":str(audio),
        "video_sha256":"v","audio_sha256":"a"})
    out=tmp_path/"run"
    bank.run(SimpleNamespace(config=cfg,output=out,resume=False,max_sources=1))
    first=out/"sources/0000/checkpoint.json";before=first.read_bytes();stamp=first.stat().st_mtime_ns
    bank.run(SimpleNamespace(config=cfg,output=out,resume=True,max_sources=1))
    assert calls==[0,1] and first.read_bytes()==before and first.stat().st_mtime_ns==stamp
    public=[json.loads(s) for s in (out/"public/questions.jsonl").read_text().splitlines()]
    assert len(public)==2
    for row in public:bank.assert_public_safe(row["forms"])
    assert (out/"report.json").is_file()
    bank.run(SimpleNamespace(config=cfg,output=out,resume=True,max_sources=None))
    assert calls==[0,1]

def test_changed_resume_input_is_refused(tmp_path):
    out=tmp_path/"run";out.mkdir()
    manifest=tmp_path/"manifest.json";manifest.write_text(json.dumps({"sources":[{"facts_path":"x"}]}))
    policy=tmp_path/"policy.json";policy.write_text("{}")
    cfg=tmp_path/"config.json";cfg.write_text(json.dumps({"sources_manifest":str(manifest),
        "acceptance_policy":str(policy),"target_questions":1,"items_per_type":1,"seed":"new"}))
    (out/"run_config.json").write_text("{}")
    with pytest.raises(ValueError,match="resume inputs differ"):
        bank.run(SimpleNamespace(config=cfg,output=out,resume=True,max_sources=None))


def test_export_preserves_episode_local_id_collisions_without_pairing_shift(tmp_path):
    selected=[candidate(qa="QA-01",answer="yes"),candidate(qa="QA-01",answer="no"),
              candidate(qa="QA-02",answer="yes")]
    selected[1]["item"]["question_id"]=selected[0]["item"]["question_id"]
    audio=tmp_path/"source.wav";audio.write_bytes(b"audio fixture")
    for i,c in enumerate(selected):
        video=tmp_path/f"source{i}.mp4";video.write_bytes(f"video{i}".encode())
        c["media"].update(video=str(video),audio=str(audio),video_sha256=f"v{i}",audio_sha256="audio")
        c["source"]["facts_path"]=f"source_{i}/facts.json"
    out=tmp_path/"export";out.mkdir()
    bank.export(out,selected,[checkpoint(c) for c in selected],{},3,{})
    public=[json.loads(s) for s in (out/"public/questions.jsonl").read_text().splitlines()]
    private=[json.loads(s) for s in (out/"private/answers.jsonl").read_text().splitlines()]
    sources=[json.loads(s) for s in (out/"private/sources.jsonl").read_text().splitlines()]
    assert len(public)==len(private)==len(sources)==3
    for i,(p,a,s,c) in enumerate(zip(public,private,sources,selected,strict=True)):
        assert p["qa_id"]==a["qa_id"]==c["item"]["qa_id"]
        assert p["forms"]==c["item"]["model_input"]
        assert a["truth"]==c["item"]["truth"]
        assert a["source_question_id"]==c["item"]["question_id"]
        assert s["facts_path"]==c["source"]["facts_path"]
        assert p["media"]["video"]==f"media/video_v{i}.mp4"
    assert json.loads((out/"report.json").read_text())["exported_question_count"]==3
