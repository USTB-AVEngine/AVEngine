#!/usr/bin/env python3
"""CPU-only question-bank generation from retained native media; resumable per source."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
import numpy as np
import soundfile as sf
from avengine.qa.unified_catalog import generate_unified_questions, model_input_questions
from avengine.qa.binding_catalog import whole_degree_display

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp_{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)

def resolve(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (REPOSITORY / p).resolve()

def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def validate_source(source, facts):
    clock = facts["time"]
    if int(clock["frame_count"]) != 150 or abs(float(clock["frame_rate_hz"])-15) > 1e-6:
        raise ValueError("source is not a 150-frame, 15-fps Episode")
    if int(clock["sample_rate_hz"]) != 16000 or int(clock["sample_count"]) != 160000:
        raise ValueError("source audio clock is not 10 seconds at 16 kHz")
    paths = facts.get("source_paths") or {}
    video = resolve(source.get("video_path") or paths["video"])
    audio = resolve(source.get("audio_path") or paths.get("mixture_audio") or paths["audio_readback"])
    if not video.is_file() or not audio.is_file():
        raise ValueError("native video/audio missing")
    info = sf.info(audio)
    if info.samplerate != 16000 or info.frames != 160000 or info.channels != 2:
        raise ValueError("audio file does not match the declared stereo clock")
    pcm, _ = sf.read(audio, dtype="float32", always_2d=True)
    if not np.isfinite(pcm).all() or not np.any(pcm):
        raise ValueError("audio is non-finite or silent")
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,nb_frames,avg_frame_rate,duration", "-of", "json", str(video)]
    probe = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)["streams"][0]
    fps = probe["avg_frame_rate"].split("/")
    rate = float(fps[0]) / float(fps[1])
    if abs(rate-15) > 1e-6 or abs(float(probe["duration"])-10) > .07 or int(probe["nb_frames"]) != 150:
        raise ValueError("video file does not match the declared Episode clock")
    return {"video": str(video), "audio": str(audio), "video_sha256": file_hash(video),
            "audio_sha256": file_hash(audio), "video_probe": probe,
            "audio_probe": {"sample_rate": info.samplerate, "frames": info.frames, "channels": info.channels}}

def observation_key(item, media):
    # An option permutation does not create a new question.
    return (media["video_sha256"], media["audio_sha256"], item["qa_id"],
            json.dumps(item["question"], ensure_ascii=False, sort_keys=True))

def unique_candidates(checkpoints):
    groups = {}; conflicted = set(); duplicates = 0
    for checkpoint in checkpoints:
        if checkpoint["status"] != "pass":
            continue
        for item in checkpoint["questions"]["items"]:
            if item.get("status") != "pass" or not item.get("forms"):
                continue
            key = observation_key(item, checkpoint["media"])
            answer = json.dumps(item["truth"].get("value"), ensure_ascii=False, sort_keys=True)
            candidate = {"item": item, "source": checkpoint["source"], "media": checkpoint["media"]}
            if key in groups:
                if groups[key][0] != answer:
                    conflicted.add(key)
                else:
                    duplicates += 1
            else:
                groups[key] = (answer, candidate)
    return [value[1] for key, value in groups.items() if key not in conflicted], {
        "duplicate_candidates": duplicates, "conflicting_observation_groups": len(conflicted)}

def tokens(candidate):
    s = candidate["source"]
    result = {("qa",candidate["item"]["qa_id"]), ("room",s["room_family"])}
    result.update(("asset",v) for v in s.get("asset_ids",[]))
    result.update(("sound",v) for v in s.get("sound_asset_ids",[]))
    if s.get("world_id"):
        result.add(("world",s["world_id"]))
    return result

def select_balanced(candidates, limit, minimum_per_type=0):
    minimum_per_type = int(minimum_per_type)
    if minimum_per_type < 0:
        raise ValueError("minimum_per_type must be non-negative")
    selected = []; remaining = list(candidates); covered = set()
    selected_by_qa = Counter(); facts_by_qa = defaultdict(set); worlds_by_qa = defaultdict(set)
    qa_types = sorted({candidate["item"]["qa_id"] for candidate in remaining})
    while minimum_per_type and len(selected) < limit:
        progressed = False
        for qa_id in qa_types:
            if len(selected) >= limit or selected_by_qa[qa_id] >= minimum_per_type:
                continue
            eligible = [i for i,candidate in enumerate(remaining)
                        if candidate["item"]["qa_id"] == qa_id]
            if not eligible:
                continue
            facts = facts_by_qa[qa_id]; worlds = worlds_by_qa[qa_id]
            def priority(index):
                candidate = remaining[index]; source = candidate["source"]
                fact = source.get("facts_path"); world = source.get("world_id")
                return (int(bool(fact) and fact not in facts),
                        int(bool(world) and world not in worlds),
                        len(tokens(candidate) - covered), -index)
            best = max(eligible, key=priority)
            item = remaining.pop(best); selected.append(item)
            selected_by_qa[qa_id] += 1; covered.update(tokens(item))
            fact = item["source"].get("facts_path")
            world = item["source"].get("world_id")
            if fact: facts.add(fact)
            if world: worlds.add(world)
            progressed = True
        if not progressed:
            break
    # First cover available source assets, sounds, rooms, worlds and types.
    while remaining and len(selected) < limit:
        best = max(range(len(remaining)), key=lambda i: len(tokens(remaining[i])-covered))
        if not (tokens(remaining[best])-covered):
            break
        item = remaining.pop(best); selected.append(item); covered.update(tokens(item))
    buckets = defaultdict(deque)
    for item in remaining:
        buckets[(item["item"]["qa_id"],item["source"]["room_family"])].append(item)
    while buckets and len(selected) < limit:
        for key in sorted(list(buckets)):
            selected.append(buckets[key].popleft())
            if not buckets[key]:
                del buckets[key]
            if len(selected) >= limit:
                break
    return selected

def assert_public_safe(value):
    if isinstance(value, dict):
        if {"gold","truth","correct_answer","correct_index"} & set(value):
            raise ValueError("gold leaked into public model input")
        for child in value.values():
            assert_public_safe(child)
    elif isinstance(value, list):
        for child in value:
            assert_public_safe(child)

def export(out, selected, checkpoints, policy, target, dedup, minimum_per_type=0):
    # Question IDs are local to their original episode/catalog. Distinct
    # media observations may legitimately reuse one, so give the projection
    # unique IDs before calling the single-catalog public projector.
    items = [{**c["item"], "question_id": f"bank_projection_{index:08d}"}
             for index,c in enumerate(selected)]
    public = model_input_questions({"items":items, "angle_followups":[]})
    if len(public["items"]) != len(selected):
        raise ValueError("public projection dropped selected observations")
    for row,candidate in zip(public["items"],selected,strict=True):
        if row["qa_id"] != candidate["item"]["qa_id"] or row["forms"] != candidate["item"].get("model_input",{}):
            raise ValueError("public projection differs from its source question")
    answers = []; links = []; copied = {}
    for index,(row,candidate) in enumerate(zip(public["items"],selected,strict=True)):
        media = candidate["media"]; row["media"] = {}
        for kind,suffix in (("video",".mp4"),("audio",".wav")):
            key=(kind,media[kind+"_sha256"])
            destination=out/"media"/(kind+"_"+key[1]+suffix)
            if key not in copied:
                destination.parent.mkdir(exist_ok=True)
                temp=destination.with_suffix(destination.suffix+f".tmp_{os.getpid()}")
                shutil.copyfile(media[kind],temp);os.replace(temp,destination)
                copied[key]=str(destination.relative_to(out))
            row["media"][kind]=copied[key]
        row["media_clock"]={"frame_count":150,"frame_rate_hz":15,"sample_rate_hz":16000,"sample_count":160000}
        assert_public_safe(row["forms"])
        item=candidate["item"]
        answers.append({"question_id":row["question_id"],"source_question_id":item["question_id"],
                        "qa_id":item["qa_id"],"truth":item["truth"],"forms":item["forms"],
                        "evidence":item["evidence"],"research_only":True})
        links.append({"question_id":row["question_id"],**candidate["source"]})
    for path,rows in ((out/"public/questions.jsonl",public["items"]),
                      (out/"private/answers.jsonl",answers),(out/"private/sources.jsonl",links)):
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+f".tmp_{os.getpid()}")
        with temp.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        os.replace(temp,path)
    successful=[c for c in checkpoints if c["status"]=="pass"]
    minimum_per_type = int(minimum_per_type)
    qa_counts = Counter(c["item"]["qa_id"] for c in selected)
    qa_types = {f"QA-{i:02d}" for i in range(1,26)}
    qa_types.update(item["qa_id"] for checkpoint in checkpoints
                    for item in checkpoint.get("questions",{}).get("items",[]))
    deficits_by_qa = {qa_id: minimum_per_type - qa_counts.get(qa_id, 0)
                      for qa_id in sorted(qa_types)
                      if qa_counts.get(qa_id, 0) < minimum_per_type}
    report={"status":"completed" if len(selected)>=target else "completed_below_target",
      "finished_at":now(),"target_question_count":target,"exported_question_count":len(selected),
      "qa_counts":dict(qa_counts),"minimum_per_type":minimum_per_type,
      "deficits_by_qa":deficits_by_qa,"minimum_per_type_met":not deficits_by_qa,
      "room_counts":dict(Counter(c["source"]["room_family"] for c in selected)),
      "asset_ids":sorted({v for c in selected for v in c["source"].get("asset_ids",[])}),
      "sound_asset_ids":sorted({v for c in selected for v in c["source"].get("sound_asset_ids",[])}),
      "asset_types":sorted({v for c in selected for v in c["source"].get("asset_types",[])}),
      "world_ids":sorted({c["source"]["world_id"] for c in selected if c["source"].get("world_id")}),
      "source_count":len(checkpoints),"successful_source_count":len(successful),
      "failed_sources":[{"facts_path":c["source"]["facts_path"],"reason":c.get("reason")} for c in checkpoints if c["status"]!="pass"],
      "missing_qa_types":sorted({f"QA-{i:02d}" for i in range(1,26)}-{c["item"]["qa_id"] for c in selected}),
      "missing_room_families":sorted({"apartment","hm3d","mp3d","kujiale"}-{c["source"]["room_family"] for c in selected}),
      "acceptance_policy":policy,"media_file_count":len(copied),"new_native_visual":0,
      "new_audio_renders":0,"new_rlr_contexts":0,**dedup,
      "claim_boundary":"Research QA bank derived from retained native media with configured question tolerance. Counts are main questions, not MCQ/Open forms; no new capture, original V1 quota completion, full asset qualification or human calibration is claimed."}
    report["source_counts_by_qa"] = {
        qa:len({c["source"]["facts_path"] for c in selected if c["item"]["qa_id"]==qa})
        for qa in report["qa_counts"]}
    report["known_world_counts_by_qa"] = {
        qa:len({c["source"]["world_id"] for c in selected
                if c["item"]["qa_id"]==qa and c["source"].get("world_id")})
        for qa in report["qa_counts"]}
    report["sources_without_world_id_by_qa"] = {
        qa:len({c["source"]["facts_path"] for c in selected
                if c["item"]["qa_id"]==qa and not c["source"].get("world_id")})
        for qa in report["qa_counts"]}
    pool_assets={v for c in checkpoints for v in c["source"].get("asset_ids",[])}
    pool_sounds={v for c in checkpoints for v in c["source"].get("sound_asset_ids",[])}
    report["unused_pool_asset_ids"]=sorted(pool_assets-set(report["asset_ids"]))
    report["unused_pool_sound_asset_ids"]=sorted(pool_sounds-set(report["sound_asset_ids"]))
    if report["missing_qa_types"] or report["missing_room_families"] or deficits_by_qa:
        report["status"]="completed_with_coverage_gaps"
    write(out/"report.json",report)
    (out/"README.md").write_text(
       "# AVEngine question bank\n\n"+f"Questions: {len(selected)}. Target ceiling: {target}.\n\n"+
       "Public inputs: public/questions.jsonl. Private answers: private/answers.jsonl. Media paths are relative to this dataset root.\n\n"+
       "Coverage, omissions and source failures are in report.json; source provenance is private/sources.jsonl. "+
       "The acceptance policy is explicit. Original native media and old gold are unchanged. No native rendering was performed.\n")
    return report

def run(args):
    cfg=read(resolve(args.config));manifest=read(resolve(cfg["sources_manifest"]))
    sources=manifest["sources"] if isinstance(manifest,dict) else manifest
    if int(cfg["target_questions"]) <= 0 or int(cfg["items_per_type"]) <= 0:
        raise ValueError("target_questions and items_per_type must be positive")
    if args.max_sources is not None and args.max_sources <= 0:
        raise ValueError("max-sources must be positive")
    if not sources:
        raise ValueError("empty source manifest")
    policy=read(resolve(cfg["acceptance_policy"]));out=Path(args.output).resolve()
    snapshot={"config":cfg,"sources":sources,"acceptance_policy":policy}
    if not args.resume:
        out.mkdir(parents=True)
        write(out/"run_config.json",snapshot)
    elif read(out/"run_config.json") != snapshot:
        raise ValueError("resume inputs differ; use a fresh output")
    controller={"pid":os.getpid(),"ppid":os.getppid(),"argv":sys.argv,"cwd":str(Path.cwd()),
                "started_at":now(),"proc_start_ticks":Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()[19]}
    write(out/"controllers"/f"{os.getpid()}.json",controller);write(out/"controller.json",controller)
    if (out/"report.json").is_file():
        if (out/"EXPORT_SUPERSEDED.json").is_file():
            corrected=(out/read(out/"EXPORT_SUPERSEDED.json")["replacement"]).resolve()
            if read(corrected/"VALIDATION.json").get("status") != "pass":
                raise ValueError("corrected export is not verified")
            print(json.dumps({"status":read(corrected/"report.json")["status"],
                              "output":str(corrected),"original_export_superseded":True}),flush=True)
            return
        print(json.dumps({"status":"already_completed","output":str(out)}),flush=True);return
    if args.resume and (out/"progress.json").exists():
        prior=read(out/"progress.json");prior.update(status="resuming",pid=os.getpid(),updated_at=now())
        write(out/"progress.json",prior)
    checkpoints=[];new_sources=0
    for index,source in enumerate(sources):
        checkpoint_path=out/"sources"/f"{index:04d}"/"checkpoint.json"
        adopted=checkpoint_path.exists()
        if adopted:
            checkpoint=read(checkpoint_path)
            if checkpoint["source"]["facts_path"] != source["facts_path"]:
                raise ValueError("checkpoint source identity differs")
        else:
            if args.max_sources is not None and new_sources>=args.max_sources:
                break
            try:
                facts=read(source["facts_path"]);media=validate_source(source,facts)
                facts=deepcopy(facts);facts.setdefault("sampling",{})["acceptance_policy"]=policy
                facts["sampling"]["time_display_precision"]=0
                facts["sampling"].setdefault("qa_sampling",{})["time_display_precision"]=0
                questions=whole_degree_display(generate_unified_questions(
                    facts,items_per_type=int(cfg["items_per_type"]),seed=str(cfg["seed"]),
                    include_angle_followups=False))
                checkpoint={"status":"pass","source":source,"media":media,"questions":questions}
            except Exception as error:
                checkpoint={"status":"failed","source":source,"reason":f"{type(error).__name__}: {error}"}
            write(checkpoint_path,checkpoint);new_sources+=1
        checkpoints.append(checkpoint)
        if adopted:
            continue
        progress={"status":"generating","updated_at":now(),"pid":os.getpid(),
            "processed_sources":len(checkpoints),"total_sources":len(sources),
            "source_failures":sum(c["status"]!="pass" for c in checkpoints),
            "raw_candidate_count":sum(len(c.get("questions",{}).get("items",[])) for c in checkpoints),
            "native_visual":0,"audio_renders":0,"rlr_contexts":0}
        write(out/"progress.json",progress)
        print(json.dumps(progress),flush=True)
    if len(checkpoints)<len(sources):
        progress["status"]="preflight_complete";write(out/"progress.json",progress);return
    candidates,dedup=unique_candidates(checkpoints)
    minimum_per_type = int(cfg.get("min_questions_per_type", 0))
    if minimum_per_type < 0:
        raise ValueError("min_questions_per_type must be non-negative")
    selected=select_balanced(candidates,int(cfg["target_questions"]),minimum_per_type)
    report=export(out,selected,checkpoints,policy,int(cfg["target_questions"]),dedup,
                  minimum_per_type)
    write(out/"progress.json",{"status":report["status"],"updated_at":now(),
         "processed_sources":len(checkpoints),"total_sources":len(sources),
         "exported_question_count":len(selected),"report":"report.json","pid":os.getpid()})
    print(json.dumps({"status":report["status"],"questions":len(selected),"output":str(out)}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-sources",type=int)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as error:
        out=Path(args.output)
        if out.is_dir():
            write(out/"failure.json",{"status":"failed","at":now(),"reason":f"{type(error).__name__}: {error}"})
        raise

if __name__=="__main__":
    main()
