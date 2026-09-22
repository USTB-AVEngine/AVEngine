"""Plan answer-changing audio over a retained visual capture without cropping clips."""
from copy import deepcopy
from pathlib import Path
import json
import random
import shutil
import subprocess
import sys


def schedule_variant(sounds, actors, clock, *, profile="sequential", seed=0, reserve_tail_s=3.0, endpoints=None):
    """Use complete waveforms; a silent actor still remains a visual candidate."""
    if len(sounds)!=2 or len(actors)!=2 or len(set(actors))!=2:
        raise ValueError("audio replay requires two different real actors and two sounds")
    rate=int(clock["sample_rate_hz"]); ticks=int(clock.get("time_base_hz",rate*3))
    if ticks % rate:raise ValueError("audio samples must map to integer timeline ticks")
    tick_ratio=ticks//rate
    deadline=int(clock["sample_count"])-round(reserve_tail_s*rate)
    sizes=[int(s["sample_count"]) for s in sounds]
    if any(int(s["sample_rate_hz"])!=rate or n<=0 for s,n in zip(sounds,sizes)):
        raise ValueError("sound clocks must match the retained scene")
    gap=round(.3*rate)
    if profile=="single_first":
        indices=[0];starts=[0]
    elif profile=="single_second":
        indices=[1];starts=[0]
    elif profile=="sequential":
        indices=[0,1];starts=[0,sizes[0]+gap]
    elif profile=="overlap":
        indices=[0,1];starts=[0,min(round(.6*rate),max(1,sizes[0]//3))]
    elif profile=="three_events":
        # Repeat the shorter complete utterance after it ends. A person never
        # talks over their own previous event, and the longer second speaker
        # can overlap the repeated utterance naturally.
        first=min(range(2),key=lambda i:sizes[i]);second=1-first
        indices=[first,second,first];starts=[0,min(round(.6*rate),max(1,sizes[first]//3)),sizes[first]+round(1.05*rate)]
    else:raise ValueError(f"unsupported audio profile: {profile}")
    end=max(start+sizes[i] for start,i in zip(starts,indices))
    if end>deadline:
        raise ValueError("complete utterances and the retained tail reserve do not fit; use shorter speech, not a crop")
    offset=random.Random(seed).randint(0,deadline-end)
    rows=[]
    for number,(i,start) in enumerate(zip(indices,starts),1):
        sound=deepcopy(sounds[i]);actor=actors[i];start+=offset;stop=start+sizes[i]
        rows.append({**sound,"actor_id":actor,"source_endpoint_id":(endpoints or {}).get(actor,f"{actor}_mouth"),
            "event_id":f"event_{number:03d}","start_sample":start,"end_sample":stop,
            "end_sample_exclusive":stop,"start_tick":start*tick_ratio,"end_tick":stop*tick_ratio,
            "end_tick_exclusive":stop*tick_ratio,"source_start_sample":0,"source_end_sample_exclusive":sizes[i],
            "linear_gain":1.0,"event_unit":"independent_source_playback_onset"})
    return rows


def prepare_audio_variant(source_root, output, speech_manifest, sound_ids, *, repository,
                          profile="sequential", swap=False, seed=0, human_nonverbal_classes=()):
    """Copy plans and reuse captured pixels; only audio-facing fields change."""
    from avengine.rooms.qa_delivery import _asset_registry
    from avengine.rooms.conditioned_sampler import request_sound_class_config
    from avengine.dataset.source_capabilities import sound_compatibility
    from avengine.timeline.current_mp3d_dynamic_audio import _active_intervals_from_pcm
    source,output,repository=map(lambda p:Path(p).resolve(),(source_root,output,repository))
    manifest=Path(speech_manifest).resolve();payload=json.loads(manifest.read_text())
    by_id={r["sound_asset_id"]:r for r in payload["sounds"]}
    sounds=[deepcopy(by_id[i]) for i in sound_ids]
    plan=json.loads((source/"plan/episode_plan.json").read_text())
    original=deepcopy(plan);request=json.loads((source/"request.json").read_text())
    declarations=plan["visual_plan"]["actors"]
    actors=[r["actor_id"] for r in declarations]
    if len(actors)!=2:raise ValueError("retained audio replay currently requires exactly two visual actors")
    if swap:actors.reverse()
    registry=_asset_registry(repository,request.get("source_registry"))
    actor_records={a["actor_id"]:registry[a["asset_id"]] for a in declarations}
    config=deepcopy(request_sound_class_config(request) or {})
    if human_nonverbal_classes:
        config["human_nonverbal_sound_classes"]=list(dict.fromkeys([*config.get("human_nonverbal_sound_classes",[]),*human_nonverbal_classes]))
        request.setdefault("sound_selection",{})["sound_class_config"]=deepcopy(config)
    policy=request.setdefault("qa_sampling",{}).setdefault("acceptance_policy",{})
    if human_nonverbal_classes:
        policy["policy_id"]=str(policy.get("policy_id","ordinary"))+"_nonverbal_v1"
    policy["sound_class_options"]=list(dict.fromkeys([*policy.get("sound_class_options",[]),*[s["sound_class"] for s in sounds]]))
    compatibility=[]
    for actor,sound in zip(actors,sounds):
        voice=sound.get("voice_preset","")
        if "gender" not in sound and voice.endswith(("男","女")):
            sound["gender"]="M" if voice.endswith("男") else "F"
        verdict=sound_compatibility(actor_records[actor],sound,config)
        if not verdict["compatible"]:raise ValueError(f"sound/actor pairing is incompatible: {actor}: {verdict}")
        compatibility.append({"actor_id":actor,"sound_asset_id":sound["sound_asset_id"],**verdict})
        intervals,detector=_active_intervals_from_pcm(Path(sound["path"]),start_sample=0,end_sample=int(sound["sample_count"]))
        sound["source_activity_intervals_samples"]=intervals
        sound["activity_coordinate"]="source_asset_samples"
        sound["activity_measurement"]=detector
        if voice:sound["speaker_id"]="tts:"+voice
    events=schedule_variant(sounds,actors,plan["clock"],profile=profile,seed=seed,
        endpoints={a["actor_id"]:a["source_endpoint_id"] for a in declarations})
    output.mkdir(parents=True,exist_ok=False)
    (output/"capture").symlink_to((source/"capture").resolve(),target_is_directory=True)
    shutil.copytree(source/"plan",output/"plan")
    # A renderer may update a cache receipt. Clone the small retained cache so
    # those writes cannot modify historical outputs, even on a cache hit.
    cache=source/"delivery/audio_rir_cache"
    if cache.is_dir():
        shutil.copytree(cache,output/"rir_cache")
        request["rir_cache"]=str(output/"rir_cache")
    else:request.pop("rir_cache",None)
    for owner in (request,request.get("runtime",{}),plan):owner.pop("prepared_manifest",None)
    request["sound_pool"]=str(output/"sound_pool.json")
    request["audio_assignment_targets"]={e["event_id"]:e["actor_id"] for e in events}
    # episode_id names the retained visual world and its geometric RIR jobs;
    # changing dry speech creates an audio variant, not another visual world.
    plan.update(audio_variant_id=output.name,audio_events=events,voice_bindings=events,
                audio_assignment_targets=request["audio_assignment_targets"],audio_target_compatibility=compatibility)
    request["episode_id"]=plan["episode_id"]
    plan["request"]=deepcopy(request)
    for key in ("clock","scene","visual_plan","activity_plan","planned_conditions","condition_profile","camera_condition_sampling","room_capabilities"):
        if plan.get(key)!=original.get(key):raise ValueError(f"audio replay changed captured visual state: {key}")
    def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    write(output/"sound_pool.json",{"sounds":sounds,"source_manifest":str(manifest)})
    write(output/"request.json",request)
    write(output/"plan/episode_plan.json",plan)
    write(output/"plan/audio_events.json",events)
    write(output/"plan/voice_bindings.json",events)
    producer={"repository":str(repository),"git_commit":subprocess.check_output(["git","-C",str(repository),"rev-parse","HEAD"],text=True).strip(),
              "python":sys.executable,"code_state":"working_tree" if subprocess.check_output(["git","-C",str(repository),"status","--porcelain"],text=True).strip() else "committed",
              "capture_reused_from":str(source),"source_speech_manifest":str(manifest)}
    write(output/"producer_version.json",producer)
    targets={"profile":profile,"events":len(events),"sounding_actors":len({e["actor_id"] for e in events}),
             "silent_actors":sorted(set(actors)-{e["actor_id"] for e in events}),"capture_reused":True,
             "reserve_tail_s":3.0,"source_root":str(source),"swap":swap,"seed":seed,
             "note":"These are requested audio targets; verify realized source activity and questions after native rendering."}
    write(output/"audio_variant_targets.json",targets)
    return targets


def plan_audio_jobs(config, *, repository):
    """Choose compatible full clips and balanced profiles without hand-authored jobs.

    The mix targets QA-01/05/23. Least-used sound classes, appearances and
    recordings guide assignment for QA-21. Observed answers remain authoritative.
    """
    from collections import Counter
    import itertools
    import math
    import soundfile as sf
    from avengine.rooms.qa_delivery import _asset_registry
    from avengine.rooms.conditioned_sampler import request_sound_class_config
    from avengine.dataset.source_capabilities import sound_compatibility

    budget = int(config["variants"])
    if budget <= 0:
        raise ValueError("variants must be positive")
    rng = random.Random(int(config.get("seed", 0)))
    pool_path = Path(config["pool"]).resolve()
    sounds = json.loads(pool_path.read_text())["sounds"]
    ids = [s["sound_asset_id"] for s in sounds]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sound_asset_id in audio pool")
    for sound in sounds:
        # Bad/missing PCM is an input error, not a reason to invent a fallback.
        info = sf.info(sound["path"])
        if info.frames != int(sound["sample_count"]) or info.samplerate != int(sound["sample_rate_hz"]) or info.channels != 1:
            raise ValueError(f"sound metadata differs from mono PCM: {sound['sound_asset_id']}")
        voice = sound.get("voice_preset", "")
        if "gender" not in sound and voice.endswith(("男", "女")):
            sound["gender"] = "M" if voice.endswith("男") else "F"
    weights = config.get("profile_weights", {"single": .5, "sequential": .25, "overlap": .125, "three_events": .125})
    if not weights or set(weights) - {"single", "sequential", "overlap", "three_events"} or any(float(v) < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("invalid audio profile weights")
    ideal = {k: budget * v / sum(weights.values()) for k, v in weights.items()}
    counts = {k: math.floor(v) for k, v in ideal.items()}
    for k in sorted(ideal, key=lambda k: ideal[k] - counts[k], reverse=True)[:budget-sum(counts.values())]:
        counts[k] += 1
    profiles = [k for k, n in counts.items() for _ in range(n)]
    rng.shuffle(profiles)
    sources, rejected = [], []
    for source_row in config["sources"]:
        row = {"root": source_row} if isinstance(source_row, str) else dict(source_row)
        source = Path(row["root"]).resolve()
        plan = json.loads((source/"plan/episode_plan.json").read_text())
        request = json.loads((source/"request.json").read_text())
        actors = plan["visual_plan"]["actors"]
        if len(actors) != 2:
            rejected.append({"source": str(source), "reason": "requires two retained visual actors"})
            continue
        registry = _asset_registry(Path(repository), request.get("source_registry"))
        compatibility = deepcopy(request_sound_class_config(request) or {})
        compatibility["human_nonverbal_sound_classes"] = list(dict.fromkeys([
            *compatibility.get("human_nonverbal_sound_classes", []), *config.get("human_nonverbal_classes", [])]))
        per_actor = [[s for s in sounds if sound_compatibility(registry[a["asset_id"]], s, compatibility)["compatible"]
                      and int(s["sample_rate_hz"]) == int(plan["clock"]["sample_rate_hz"])] for a in actors]
        if not all(per_actor):
            rejected.append({"source": str(source), "reason": "no compatible sound for one or both actors"})
            continue
        sources.append({**row, "root": str(source), "plan": plan, "actors": actors, "sounds": per_actor})
    class_counts, actor_counts, clip_counts, source_counts = Counter(), Counter(), Counter(), Counter()
    jobs, deficits, singles = [], [], 0
    for i, profile in enumerate(profiles):
        if profile == "single":
            profile = ["single_first", "single_second"][singles % 2]
            singles += 1
        candidates = []
        seed = rng.randrange(2**31)
        for source in sources:
            actors = source["actors"]
            for pair in itertools.product(*source["sounds"]):
                try:
                    events = schedule_variant(pair, [a["actor_id"] for a in actors], source["plan"]["clock"], profile=profile, seed=seed)
                except ValueError:
                    continue
                # QA-21 asks per actor/class, not once per repeated playback.
                actual = {(e["actor_id"], e["sound_class"], e["sound_asset_id"]) for e in events}
                inc = Counter(c for _, c, _ in actual)
                appearance = {a["actor_id"]: a["asset_id"] for a in actors}
                score = (source_counts[source["root"]],
                         sum(2*class_counts[c]*n+n*n for c,n in inc.items()),
                         sum(2*actor_counts[(appearance[a],c)]+1 for a,c,_ in actual),
                         sum(clip_counts[s] for _,_,s in actual), rng.random())
                candidates.append((score, source, pair, actual, events))
        if not candidates:
            deficits.append({"profile": profile, "reason": "no compatible complete clip pair fits the retained clock and tail"})
            continue
        _, source, pair, actual, events = min(candidates, key=lambda c: c[0])
        appearance = {a["actor_id"]: a["asset_id"] for a in source["actors"]}
        for actor, cls, sound in actual:
            class_counts[cls] += 1
            actor_counts[(appearance[actor], cls)] += 1
            clip_counts[sound] += 1
        source_counts[source["root"]] += 1
        jobs.append({"id": f"variant_{i:04d}", "source": source["root"], "split": source.get("split"),
                     "world_id": source.get("world_id", source["plan"]["episode_id"]),
                     "sounds": [s["sound_asset_id"] for s in pair], "profile": profile, "seed": seed,
                     "event_count": len(events), "qa01_answer": "no" if profile.startswith("single") else "yes",
                     "qa05_branch": None if profile.startswith("single") else "disjoint" if profile == "sequential" else "overlap"})
    return {"jobs": jobs, "requested_variants": budget, "deficits": deficits, "rejected_sources": rejected,
            "planned_class_counts": dict(class_counts), "planned_source_counts": dict(source_counts),
            "new_visual_renders": 0, "reserve_tail_s": 3.0}
