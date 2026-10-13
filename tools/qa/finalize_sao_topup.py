"""Finish owner-authorized SAO readmission, bounded top-up and file-based listening report."""
from __future__ import annotations
import argparse, collections, copy, html, json, pathlib, pickle, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
from avengine.qa.v7_sao_fingerprint import waveform_fingerprint, robust_alignment, possible_references
from avengine.timeline.v7_recorded_catalog import recorded_catalog_entry
from avengine.timeline.v7_audio_sources import filter_admission, source_allowed_in_split
from tools.qa.admit_sao_topup import load_scored, lines, sha, norm_path, dump, progress, maximum_group_activity

def once(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"different completed artifact: {path}"
    else:
        with path.open("x") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def origin(r):
    return str(r.get("original_group") or r.get("source_original_id") or r.get("fsd50k_id") or r.get("source_origin") or r["sound_asset_id"])

VARIANTS = {
 "fire_alarm": ["motor-driven mechanical building fire alarm bell", "red steel building evacuation alarm bell with repeated hammer impacts", "old brass mechanical fire alarm bell ringing urgently", "wall mounted electromechanical evacuation bell", "industrial building fire alarm with coarse metallic bell strokes", "large mechanical fire bell with an uneven raspy ring", "school building evacuation fire bell with a heavy clapper", "factory fire alarm bell with fast dry metallic impacts", "compact mechanical fire alarm bell with a sharp metallic clang", "motor-driven fire alarm gong outside a stairwell", "large red mechanical fire alarm bell with dense repeated clangs", "old building evacuation bell with rattling metal"],
 "microwave_hum": ["small countertop microwave oven magnetron and cooling fan hum", "older microwave oven low mechanical hum", "compact microwave oven steady fan and transformer buzz", "large kitchen microwave oven running its magnetron", "microwave oven warm midrange motor drone", "small microwave oven with a slightly rattling cooling fan", "microwave oven operating with a soft electric buzz", "microwave oven coarse low pitched running hum", "microwave oven thin high pitched cooling fan whirr", "commercial microwave oven deep mechanical hum", "compact microwave oven humming during heating", "older microwave oven fan running steadily"],
 "cat_meow": ["adult domestic cat giving short natural meows", "kitten making small high pitched meows", "older cat giving a low rough meow", "tabby cat giving repeated brief meows", "small domestic cat making a nasal meow", "large domestic cat giving a deep meow", "cat making a soft questioning meow", "cat giving a hoarse short meow", "cat producing a loud clear meow", "domestic cat making two separated meows", "young cat giving a thin squeaky meow", "adult cat giving a long natural meow"],
 "printer": ["desktop laser printer feeding a sheet with motor whirrs", "inkjet printer carriage moving back and forth", "office laser printer printing one page", "small inkjet printer with sharp carriage clicks", "older dot matrix printer with rapid mechanical chatter", "compact laser printer rollers turning and paper feeding", "office printer with a low motor and paper rustle", "inkjet printer making short repeated carriage sweeps", "desktop printer with rattling gears and feed rollers", "laser printer ejecting a page", "small printer starting its paper feed mechanism", "older office printer producing a rough motor whirr"],
 "alarm_beep": ["single electronic warning buzzer giving flat short beeps", "small appliance alarm giving repeated high pitched beeps", "electronic device warning alarm with low square wave beeps", "compact warning beeper with sharp intermittent pulses", "battery device alarm beeping at a steady pace", "simple electronic alarm buzzer with coarse raspy beeps", "warning device giving two repeated flat beeps", "small digital warning alarm with thin sharp pulses", "electronic equipment giving spaced warning beeps", "plain piezo warning buzzer producing short flat tones", "portable alarm beeper with a dry nasal pulse", "single electronic alarm device giving repeated uniform beeps"],
 "clock_tick": ["small analog wall clock ticking steadily", "older mechanical wall clock with dry tick tock clicks", "quartz clock with soft regular ticks", "large wall clock with low wooden sounding tick tock", "small mantel clock with sharp metal ticks", "analog clock with a faint rattling tick", "quartz wall clock with crisp second hand clicks", "old mechanical clock with heavy alternating ticks", "compact desk clock making regular light ticks", "analog wall clock with hollow ticks", "small mechanical clock with fine metallic ticks", "old mantel clock with coarse dry ticking"],
 "blender": ["kitchen blender motor running with a rough mechanical roar", "small countertop blender producing a steady motor buzz", "large blender with a deep whirring motor", "older kitchen blender with rattling motor noise", "compact smoothie blender running at high speed", "powerful kitchen blender with a harsh blade motor whirr", "small blender giving a thin high speed motor drone", "countertop blender with a dense coarse buzzing motor", "kitchen blender motor briefly starting then running", "older blender with a slightly uneven motor whine", "large blender with a loud low mechanical roar", "compact kitchen blender with a sharp continuous whirr"],
 "microwave_beep": ["countertop microwave oven giving three end of cooking beeps", "small microwave oven giving two short flat beeps", "older microwave oven completion buzzer beeping", "microwave oven button confirmation beep", "compact microwave oven giving spaced high pitched completion beeps", "kitchen microwave oven giving four short beeps", "microwave oven timer finishing with plain beeps", "commercial microwave oven completion buzzer", "older microwave oven giving a low raspy completion beep", "small microwave oven keypad beeps", "microwave oven giving repeated thin completion beeps", "countertop microwave oven giving a sharp final beep"],
 "telephone": ["old desk telephone ringing with two mechanical bells", "landline telephone producing repeated bell rings", "rotary telephone with a metallic twin bell ringer", "corded desk telephone ringing with a coarse bell", "older office telephone ringing sharply", "small landline telephone producing a plain electronic ring", "vintage telephone with a rattling bell ringer", "desk telephone giving spaced double rings", "large old telephone with a deep mechanical bell", "corded office telephone ringing with a high pitched bell", "rotary desk phone ringing with a dry metallic tremble", "single telephone ringer producing repeated flat electronic pulses"],
 "pet_bird_parrot": ["pet parrot making natural short squawks and chirps", "small pet parrot giving a rough natural squawk", "pet parrot producing short sharp nonverbal bird chirps", "adult parrot making a low raspy bird call", "small parrot giving brief squeaky bird calls", "pet parrot producing a harsh natural screech", "parrot making separated dry squawks", "parrot making a soft chattering nonverbal bird call", "pet parrot giving high pitched natural chirps", "parrot giving two short rough squawks", "small pet parrot making raspy bird chirps", "adult parrot making short clear bird calls"],
 "doorbell": ["single house doorbell with a mechanical ding dong chime", "old mechanical doorbell ringing with a small bell", "wired household doorbell giving two dry chime strikes", "compact doorbell with a simple sharp ding dong", "old house doorbell with a rattling bell buzzer", "doorbell making one clear metallic ding", "wired doorbell with a low resonant ding dong", "small mechanical doorbell ringing briefly", "plain household doorbell with two short metallic chimes", "vintage doorbell with a coarse electric bell", "single doorbell giving a bright dry ding dong", "small doorbell with a sharp double chime"],
 "alarm_clock": ["old mechanical alarm clock ringing with two small bells", "bedside alarm clock producing a coarse buzzing alarm", "analog alarm clock with a sharp metal bell", "digital alarm clock giving uniform spaced beeps", "small bedside alarm clock ringing loudly", "old wind up alarm clock with rattling twin bells", "compact alarm clock producing fast flat beeps", "mechanical bedside clock with a dry bell rattle", "digital bedside alarm clock giving short repeated beeps", "large old alarm clock ringing with a low metal bell", "small alarm clock with a thin buzzing alarm", "wind up alarm clock with rapid sharp bell impacts"],
 "pet_bird_canary": ["pet canary making brief natural chirps", "small canary giving a clear short bird call", "yellow canary producing separated natural chirps", "canary making a soft high pitched bird call", "pet canary giving a brief natural warble", "canary producing two short sharp chirps", "small canary making thin natural tweets", "canary giving irregular brief bird calls", "pet canary making a raspy short chirp", "canary producing a few natural high bird calls", "small canary making light separated chirps", "pet canary giving a clear dry bird call"]
}

def plan(a):
    o = a.output
    config = json.loads(a.config.read_text())
    raw = json.loads((o/"admission_v12_r12/dry_recorded_admission.json").read_text())["records"]
    old = json.loads((o/"admission_v11_r12_polyphase/dry_recorded_admission.json").read_text())["records"]
    independent = {r["sound_asset_id"] for r in old if not [x for x in r["current_policy_blockers"] if x != "generated_source_human_speech_calibration_failed"]}
    assert len(independent) == 177 and len(raw) == 856
    passed = [r for r in raw if r.get("strict_v7_eligible")]
    assert {r["sound_asset_id"] for r in passed} <= independent
    by_class = []
    for c, spec in config["classes"].items():
        count = sum(r["sound_class"] == c for r in passed)
        gap = max(0, 30-spec["existing_originals"]-count)
        spec["round3_gap"] = gap
        spec["round3_prompts"] = [
            dict(id=f"r3_{c}_{i:02d}", duration_s=4 + i%5,
                 text=f"A real field recording of one {description}. Single source device or animal, natural irregular sound. Dry sound, close microphone at {['20 cm','50 cm','one metre'][i%3]}, no reverberation, no music, no melody, not musical, no human speech or voice, no other sound sources.")
            for i, description in enumerate(VARIANTS[c])]
        by_class.append(dict(sound_class=c,existing_originals=spec["existing_originals"],accepted_r12=count,final_originals_before_round3=spec["existing_originals"]+count,gap=gap,round3_budget=gap*3))
    once(o/"round3_config_v12.json", config)
    receipt = dict(status="pass",prechecked_candidate_count=177,accepted_r12=len(passed),by_class=by_class,
                   round3_budget=sum(r["round3_budget"] for r in by_class),classes=[r["sound_class"] for r in by_class if r["gap"]],
                   first_admission=str(o/"admission_v12_r12/dry_recorded_admission.json"),
                   calibration=str(a.calibration),calibration_sha256=sha(a.calibration),beats_threshold=.1917129977,beats_minimum_margin=.05)
    once(o/"ROUND3_PLAN_v12.json",receipt)
    progress(o,f"v12 第一二轮正式准入 {len(passed)} 条；第三轮按剩余缺口 3 倍计划 {receipt['round3_budget']} 条，不重做已完成生成。")
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

def dedup(a):
    o=a.output;start=time.time()
    rows=[r for r in load_scored(o,a.calibration) if r["candidate"]["round"]==3]
    expected=lines(o/"candidates_r3.jsonl")
    assert len(rows)==len(expected)>0
    old=[r for r in load_scored(o,a.calibration) if r["candidate"]["round"] in (1,2)]
    previous=json.loads((o/"FINGERPRINT_REPAIR_VALIDATION_v11_r12_polyphase.json").read_text())
    assert previous["status"]=="pass" and previous["after_detection_rate"]>=.99
    module=pathlib.Path(__file__).resolve().parents[2]/"src/avengine/qa/v7_sao_fingerprint.py"
    identity=dict(v9_sha256=sha(a.v9),model_scores_r3_sha256=sha(o/"model_scores_r3.jsonl"),
                  candidate_ids_sha256=sha(o/"candidates_r3.jsonl"),fingerprint_module_sha256=sha(module),
                  human_calibration_sha256=sha(a.calibration))
    once(o/"evidence/deduplication_v12_r3_inputs.json",identity)
    if (o/"deduplication_v12_r3.json").exists():
        print("DEDUP_ALREADY_COMPLETED",flush=True);return
    def fingerprint(row):
        r=row["candidate"];x,sr=sf.read(r["path"],dtype="float32")
        assert sr==16000 and x.ndim==1 and sha(r["path"])==r["sha256"]
        return r["sound_asset_id"],waveform_fingerprint(x,sr)
    with ThreadPoolExecutor(max_workers=8) as pool:
        queries=dict(pool.map(fingerprint,rows))
    control_path=o/"FINGERPRINT_REPAIR_VALIDATION_v12_r3.json"
    if control_path.exists():
        validation=json.loads(control_path.read_text())
    else:
        from scipy import signal
        def inspect(row):
            r=row["candidate"];x,sr=sf.read(r["path"],dtype="float32");base=queries[r["sound_asset_id"]]
            hashes={h for h,_ in base["landmarks"]};result=[]
            for name,wave in [("gain",x*.137),("resample_roundtrip",signal.resample_poly(signal.resample_poly(x,3,2),2,3)[:len(x)]),("crop",x[len(x)//5:4*len(x)//5])]:
                q=waveform_fingerprint(wave,sr);qh={h for h,_ in q["landmarks"]};common=len(hashes&qh)
                retrieved="original" in possible_references({"original":common},len(qh),{"original":len(hashes)})
                aligned=robust_alignment(q,base)
                result.append(dict(sound_asset_id=r["sound_asset_id"],transform=name,detected=bool(retrieved and aligned["matched"]),reference_retrieved=retrieved,common_hashes=common,alignment=aligned))
            return result
        with ThreadPoolExecutor(max_workers=8) as pool:
            controls=[r for batch in pool.map(inspect,rows) for r in batch]
        detected=sum(r["detected"] for r in controls)
        total=previous["after_count"]+len(controls);combined=previous["after_detected"]+detected
        validation=dict(status="pass" if combined/total>=.99 else "fail",count=len(controls),detected=detected,detection_rate=detected/len(controls),
                        combined_count=total,combined_detected=combined,combined_detection_rate=combined/total,
                        prior_validation=str(o/"FINGERPRINT_REPAIR_VALIDATION_v11_r12_polyphase.json"),controls=controls,failures=[r for r in controls if not r["detected"]],
                        controls_are_quality_validation_only=True,positive_control_failure_is_not_a_candidate_rejection=True)
        once(control_path,validation)
        progress(o,f"第三轮指纹对照完成：{detected}/{len(controls)}，累计 {combined}/{total}；对照失败不拒候选。")
    assert validation["status"]=="pass" and validation["combined_detection_rate"]>=.99
    references=pickle.loads((o/"cache/v9_waveform_fingerprints_v11_polyphase.pkl").read_bytes())
    assert references["v9_sha256"]==sha(a.v9)
    fingerprints=references["fingerprints"]
    cache=o/"cache/existing_generated_waveform_fingerprints_v12.pkl"
    if cache.exists():
        retained=pickle.loads(cache.read_bytes())
        assert retained["source_sha256"]==sha(module)
        assert retained["audio_sha256"]=={r["candidate"]["sound_asset_id"]:r["candidate"]["sha256"] for r in old}
        generated=retained["fingerprints"]
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:
            generated=dict(pool.map(fingerprint,old))
        cache.write_bytes(pickle.dumps(dict(source_sha256=sha(module),audio_sha256={r["candidate"]["sound_asset_id"]:r["candidate"]["sha256"] for r in old},fingerprints=generated),protocol=4))
    fingerprints.update({"generated:"+k:v for k,v in generated.items()})
    inverted=collections.defaultdict(list);pcm=collections.defaultdict(list);hash_counts={}
    def register(key,fp):
        fingerprints[key]=fp;hashes={h for h,_ in fp["landmarks"]};hash_counts[key]=len(hashes)
        for h in hashes:inverted[h].append(key)
        if fp["pcm_sha256"]:pcm[fp["pcm_sha256"]].append(key)
    for key,value in list(fingerprints.items()):register(key,value)
    ordered=sorted(rows,key=lambda r:(-int(r["beats_pass"] and r["activity_pass"] and r["human_speech_pass"]),-r["target_confidence"],-r["margin"],r["candidate"]["sound_asset_id"]))
    partial=o/"deduplication_v12_r3.partial.jsonl"
    completed=lines(partial) if partial.exists() else []
    assert [r["sound_asset_id"] for r in completed]==[r["candidate"]["sound_asset_id"] for r in ordered[:len(completed)]]
    results={};compared=0
    with partial.open("a") as stream,ThreadPoolExecutor(max_workers=8) as pool:
        for i,row in enumerate(ordered):
            r=row["candidate"];key=r["sound_asset_id"];query=queries[key]
            if i<len(completed):
                receipt=completed[i];assert receipt["audio_sha256"]==r["sha256"]
            else:
                if time.time()+30>=a.deadline:raise SystemExit(124)
                counts=collections.Counter()
                hashes={h for h,_ in query["landmarks"]}
                for h in hashes:counts.update(inverted.get(h,()))
                possible=possible_references(counts,len(hashes),hash_counts)
                possible.update(pcm.get(query["pcm_sha256"],()))
                def compare(reference):
                    return dict(robust_alignment(query,fingerprints[reference]),reference_key=reference)
                hits=[h for h in pool.map(compare,sorted(possible)) if h["matched"]]
                reason="audio_fingerprint_matches_existing_v9" if any(h["reference_key"].startswith("v9:") for h in hits) else "audio_fingerprint_matches_generated_candidate" if hits else None
                receipt=dict(sound_asset_id=key,status="blocked" if hits else "pass",reason=reason,matches=hits,audio_sha256=r["sha256"],compared_references=len(possible),positive_controls_are_not_an_admission_gate=True,coverage_complete_for_listed_roots=True)
                stream.write(json.dumps(receipt,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
            results[key]=receipt;compared+=receipt["compared_references"]
            if receipt["status"]=="pass":register("generated:"+key,query)
            if i%12==0:print("DEDUP_R3",i,len(ordered),"MATCHED",sum(r["status"]=="blocked" for r in results.values()),flush=True)
    report=dict(records=results,matches=[r for r in results.values() if r["status"]=="blocked"],errors=[],complete_for_listed_roots=True,
                reference_roots=[dict(manifest=str(a.v9),kind="all v9 dry inputs"),dict(root=str(o/"mono_16000"),kind="all three rounds of generated candidates")],
                reference_file_count=1914+len(old),reference_waveform_count=1914+len(old),pair_comparisons=compared,seconds=time.time()-start,
                positive_control_validation=str(control_path),positive_control_count=validation["combined_count"],positive_control_detected=validation["combined_detected"],
                reused_r12_dedup=str(o/"deduplication_v11_r12_polyphase.json"),
                method="unchanged repaired weak spectral hash retrieval followed by full shorter waveform normalized correlation >=0.995 for >=1 second, both 16k sample phases",
                scope_limit="approximate fingerprint retrieval over v9 and all generated candidates; measured control recall, no perceptual similarity claim",human_listening_used=False)
    once(o/"deduplication_v12_r3.json",report)
    progress(o,f"第三轮 {len(rows)} 条去重完成：实际匹配 {len(report['matches'])} 条，新增比较 {compared} 对。")
    print("DEDUP_R3_DONE",len(rows),len(report["matches"]),compared,report["seconds"],flush=True)

def publish(a):
    o=a.output;v9=json.loads(a.v9.read_text());cci=json.loads(a.cci_v9.read_text())
    plan=json.loads((o/"ROUND3_PLAN_v12.json").read_text())
    stages=["v12_r12"]+(["v12_r3"] if plan["round3_budget"] else [])
    raw=[];selected=[];manifests=[]
    for stage in stages:
        manifest=o/f"admission_{stage}/dry_recorded_admission.json"
        records=json.loads(manifest.read_text())["records"];raw.extend(records);manifests.append(str(manifest))
        selected.extend(recorded_catalog_entry(r,str(manifest)) for r in records if r.get("strict_v7_eligible"))
    scored=load_scored(o,a.calibration)
    assert len(raw)==len(scored)==len({r["sound_asset_id"] for r in raw})
    assert len(v9["records"])==1914
    admitted,rejected=filter_admission(selected,split="train")
    assert len(admitted)==len(selected) and not rejected
    assert not filter_admission(selected,split="dev")[0] and not filter_admission(selected,split="test")[0]
    mirror=None
    for real,remote in zip(v9["records"],cci["records"]):
        assert real["sound_asset_id"]==remote["sound_asset_id"]
        p=real.get("normalized_path");q=remote.get("normalized_path")
        if p and q and p!=q and q.endswith(p):mirror=q[:-len(p)];break
    assert mirror
    def mirror_row(row):
        r=copy.deepcopy(row)
        for holder,key in [(r,"normalized_path"),(r["normalization"],"path"),(r["screens"]["normalization"],"path")]:
            assert pathlib.Path(holder[key]).resolve().is_relative_to(o.resolve());holder[key]=mirror+holder[key]
        return r
    readbacks=[]
    for r in v9["records"]+selected:
        info=sf.info(norm_path(r));assert info.samplerate==16000 and info.channels==1 and info.frames>0
        readbacks.append(dict(sound_asset_id=r["sound_asset_id"],path=norm_path(r),sample_rate_hz=info.samplerate,channels=info.channels,sample_count=info.frames))
    for r in selected:
        assert r["allowed_splits"]==["train"] and r["split"]=="train"
        assert r["label_provenance"]=="generated_stable_audio_open_1.0" and r["license"]=="Stability AI Community License"
        assert r["owner_acceptance"]=="owner 2026-10-12 接受 IV(b) 风险"
        assert r["human_speech_screen"]["threshold"]==.38 and r["human_speech_screen"]["maximum_score"]<=.38
        assert r["whisper_transcription_recording_only"]["participates_in_admission"] is False
        assert sha(r["normalization"]["path"])==r["normalization"]["sha256"]
        activity=r["normalization"]["measured_activity"]
        assert activity["active_duration_s"]>=1 and maximum_group_activity(activity["source_activity_intervals_samples"],16000)>=1
        assert not source_allowed_in_split(r,"dev") and not source_allowed_in_split(r,"test")
        model=next(m for m in r["automatic_label"] if m["status"]=="pass")
        assert model["gate"]["threshold"]==.1917129977 and model["gate"]["minimum_margin"]==.05
    by_class=[];config=json.loads(a.config.read_text())
    for c,spec in config["classes"].items():
        rows=[r for r in raw if r["sound_class"]==c];new=[r for r in selected if r["sound_class"]==c]
        count=len({origin(r) for r in v9["records"]+new if r["sound_class"]==c})
        assert count==spec["existing_originals"]+len(new)
        by_class.append(dict(sound_class=c,candidates=len(rows),round3_candidates=sum(r["generation"]["prompt_id"].startswith("r3_") for r in rows),accepted=len(new),
                             accepted_r12=sum(r["generation"]["prompt_id"].startswith("r3_") is False for r in new),
                             accepted_r3=sum(r["generation"]["prompt_id"].startswith("r3_") for r in new),
                             existing_originals=spec["existing_originals"],final_originals=count,deficit=max(0,30-count),
                             shortfall_reason="authorized third-round budget exhausted; other candidates failed the unchanged screens" if count<30 else None,
                             rejection_reasons=dict(collections.Counter(x for r in rows if not r.get("strict_v7_eligible") for x in r["current_policy_blockers"]))))
    protected=json.loads((o/"evidence/resume_v12_audit.json").read_text())["protected_sha256"]
    for path,digest in protected.items():assert sha(path)==digest,path
    outputs=[]
    for base,original,name,added in [(v9,a.v9,"source_admission_v12_classes27_english_sao.json",selected),
                                     (cci,a.cci_v9,"cci_source_admission_v12_classes27_english_sao.json",[mirror_row(r) for r in selected])]:
        doc=copy.deepcopy(base);doc["profile"]+="-sao-train-only-v12-20261013"
        doc["derived_from_v9"]=dict(path=str(original),sha256=sha(original))
        doc["sao_topup"]=dict(status="pass_with_shortfalls" if any(r["deficit"] for r in by_class) else "pass",added_records=len(selected),
                              human_speech_threshold=.38,calibration=str(a.calibration),calibration_sha256=sha(a.calibration),
                              license="Stability AI Community License",owner_acceptance="owner 2026-10-12 接受 IV(b) 风险",
                              allowed_splits_for_generated=["train"],machine_only=True,shortfalls=[r for r in by_class if r["deficit"]])
        doc["source_manifests"]=list(base["source_manifests"])+manifests;doc["records"]+=added
        ids=[r["sound_asset_id"] for r in doc["records"]];assert len(ids)==len(set(ids))
        assert doc["records"][:1914]==base["records"]
        once(o/name,doc);outputs.append(dict(path=str(o/name),records=len(ids),sha256=sha(o/name)))
    pull=[r["normalized_path"].lstrip("/") for r in selected]+[(o/"cci_source_admission_v12_classes27_english_sao.json").as_posix().lstrip("/")]
    path=o/"cci_pull_list_v12.txt";content="\n".join(pull)+"\n"
    if path.exists():assert path.read_text()==content
    else:
        with path.open("x") as f:f.write(content)
    once(o/"cci_expected_new_files_v12.json",{r["sound_asset_id"]:dict(path=mirror+r["normalized_path"],sha256=r["normalization"]["sha256"]) for r in selected})
    counts={c:len({origin(r) for r in v9["records"]+selected if r["sound_class"]==c}) for c in sorted({r["sound_class"] for r in v9["records"]+selected})}
    assert len(counts)==27
    once(o/"V12_READBACK.json",dict(status="pass",records=len(readbacks),generated_additions=len(selected),unique_ids=True,mono_16000=True,
                                   generated_train_only=True,existing_records_unchanged=True,cci_prefix=mirror,readbacks=readbacks,all_class_original_counts=counts))
    summary=dict(status="pass_with_shortfalls" if any(r["deficit"] for r in by_class) else "pass",candidate_count=len(raw),accepted=len(selected),by_class=by_class,
                 third_round=plan,outputs=outputs,all_class_original_counts=counts,calibration=str(a.calibration),human_speech_threshold=.38,
                 raw_admissions=manifests,cci_pull_list=str(path),cci_mirror_prefix=mirror,cci_copy_execution="not_run; prepared on 48g only",
                 rejection_reasons=dict(collections.Counter(x for r in raw if not r.get("strict_v7_eligible") for x in r["current_policy_blockers"])),
                 report=str(o/"listen_v12/index.html"),human_listening_used=False,protected_sha256=protected)
    once(o/"DELIVERY_v12.json",summary)
    progress(o,f"v12 输入及 CCI 拉取清单完成：新增 {len(selected)}，共 {1914+len(selected)} 条；唯一 ID、单声道 16k、训练集限定及旧输入不变均通过。")
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def listen(a):
    from scipy.signal import spectrogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o=a.output;folder=o/"listen_v12";folder.mkdir(exist_ok=True)
    delivery=json.loads((o/"DELIVERY_v12.json").read_text())
    cal=json.loads(a.calibration.read_text())
    doc=json.loads(pathlib.Path(delivery["outputs"][0]["path"]).read_text())
    scored=load_scored(o,a.calibration)
    reference=lines(pathlib.Path(cal["scores_path"]))
    real_scores={r["sound_asset_id"]:r["score"] for r in reference if r["polarity"]=="positive_real_target"}
    esc=lambda s:html.escape(str(s))
    positive=[r["score"] for r in reference if r["polarity"]=="positive_real_target"]
    negative=[r["score"] for r in reference if r["polarity"]=="negative_human_speech"]
    generated=[r["candidate"]["human_speech_screen"]["maximum_score"] for r in scored]
    figures=[]
    def save_figure(fig,name):
        path=folder/name
        if not path.exists():fig.savefig(path,dpi=140,bbox_inches="tight")
        plt.close(fig);figures.append(name);return name
    fig,ax=plt.subplots(figsize=(11,4))
    bins=np.linspace(0,1,51)
    for values,label in [(positive,"real target recordings (183)"),(negative,"VCTK speech (739)"),(generated,f"generated candidates ({len(generated)})")]:
        ax.hist(values,bins=bins,histtype="step",linewidth=1.6,density=True,label=label)
    ax.axvline(.38,color="black",linestyle="--",label="owner threshold = 0.38")
    ax.set(xlabel="maximum posterior of 20 human-voice classes",ylabel="density",xlim=(0,1))
    ax.legend(fontsize=9);cal_plot=save_figure(fig,"human_speech_calibration.png")
    fig,axes=plt.subplots(4,4,figsize=(16,11));axes=axes.ravel()
    for ax,row in zip(axes,delivery["by_class"]):
        values=[r["target_confidence"] for r in scored if r["candidate"]["sound_class"]==row["sound_class"]]
        ax.hist(values,bins=np.linspace(0,1,26),color="#4475a6")
        ax.axvline(.1917129977,color="#b03636",linestyle="--")
        ax.set(title=row["sound_class"],xlim=(0,1),xlabel="target posterior",ylabel="candidates")
    for ax in axes[len(delivery["by_class"]):]:ax.set_visible(False)
    fig.tight_layout();beats_plot=save_figure(fig,"beats_target_distribution.png")
    h=['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
       '<meta name="viewport" content="width=device-width,initial-scale=1"><title>v7 Stable Audio Open v12 准入与试听报告</title>',
       '<style>body{max-width:1120px;margin:24px auto;padding:0 18px;font:16px/1.65 system-ui;color:#202631}h1{font-size:27px}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d7dee7;padding:7px;text-align:left}th{background:#edf2f8}img{max-width:100%}audio{width:min(100%,550px)}code{overflow-wrap:anywhere}.note{padding:14px;background:#edf4fa}figure{margin:18px 0}figcaption{font-size:14px;overflow-wrap:anywhere}</style><body>',
       f'<h1>结论：v12 新增 {delivery["accepted"]} 条训练集生成音频，共 {1914+delivery["accepted"]} 条；{sum(r["deficit"]==0 for r in delivery["by_class"])}/13 个薄类达到 30 个不同来源</h1>',
       '<p class="note">按 owner 新决策使用人说话阈值 0.38。第一、二轮复用已完成的模型和去重结果，第三轮仅按剩余缺口生成。BEATs 目标分数 ≥0.1917129977、与不相容类的差值 ≥0.05、有效活动段 ≥1 秒及原响度归一化均保持不变。Whisper 自由转写仅记录，不参与准入；所有判定使用机器指标，未人工试听。v9、v10、v11 与旧报告均未修改。</p>',
       '<h2>人说话检查的校准</h2>',
       '<p>同一 BEATs AS2M cpt2，float32；5 秒块后验取平均，末块至少补至 1 秒，再对原来的 20 个人声类别取最大值。生成音频分数高于 0.38 拒绝，等于 0.38 保留。</p>',
       '<table><tr><th>指标</th><th>结果</th></tr><tr><td>正式阈值</td><td>0.38，owner 2026-10-13 决策</td></tr><tr><td>真实目标录音放行</td><td>178/183 = 97.268%</td></tr><tr><td>VCTK 真人朗读拦截</td><td>739/739 = 100%；最低分 0.3813137710</td></tr></table>',
       '<p>下列 5 条真实录音在新阈值下不放行。owner 将其解释为会学舌的鹦鹉和金丝雀；本次未试听验证。该阈值仅用于新增生成音频，v9 的这些记录保留原样。</p>',
       '<table><tr><th>ID</th><th>类别</th><th>分数</th><th>最高人声类</th></tr>']
    for r in cal["positive_blocked_records"]:
        h.append(f'<tr><td><code>{esc(r["sound_asset_id"])}</code></td><td>{esc(r["sound_class"])}</td><td>{r["score"]:.10f}</td><td>{esc(r["maximum_head"]["name"])}</td></tr>')
    h.append('</table><p>20 类：'+esc("; ".join(r["name"] for r in cal["heads"]))+'</p>')
    h.append(f'<figure><img src="{cal_plot}" alt="真实录音、VCTK 和生成候选的人声分数分布"><figcaption>同一套人声分数分布，虚线为 0.38。</figcaption></figure>')
    h.append('<h2>每类候选、准入与来源数</h2><table><tr><th>类别</th><th>全部候选</th><th>第三轮候选</th><th>通过</th><th>第三轮通过</th><th>最终不同来源</th><th>缺口</th></tr>')
    for r in delivery["by_class"]:
        h.append('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in ["sound_class","candidates","round3_candidates","accepted","accepted_r3","final_originals","deficit"])+'</tr>')
    h.append('</table><p>第三轮按剩余缺口的 3 倍生成一次；仍有缺口的类如实保留短缺，未增加第四轮或放宽门槛。通过的生成音频全部加入 v12，没有按配额截掉富余。每个种子候选对应一个原始来源；猫叫的 13 条真实录音按 6 个原始来源计算。</p>')
    h.append('<h2>拒绝原因与指纹验证</h2><p>原因可重叠；未放宽任何 BEATs 门槛。</p><table><tr><th>原因</th><th>数量</th></tr>')
    reason_labels={
        "no_source_calibrated_automatic_label_model_pass":"BEATs 目标分数或不相容类间隔未达标",
        "generated_source_effective_activity_below_one_second_or_missing":"有效活动段不足 1 秒",
        "generated_source_human_speech_or_missing_calibrated_receipt":"最大人声分数高于 0.38",
        "original_bird_species_metadata_missing":"鸟类原始子类型元数据不足"}
    for reason,count in sorted(delivery["rejection_reasons"].items()):
        label=reason_labels.get(reason,reason)
        h.append(f'<tr><td>{esc(label)} <code>{esc(reason)}</code></td><td>{count}</td></tr>')
    fp=json.loads((o/"FINGERPRINT_REPAIR_VALIDATION_v12_r3.json").read_text()) if (o/"FINGERPRINT_REPAIR_VALIDATION_v12_r3.json").exists() else None
    old_fp=json.loads((o/"FINGERPRINT_REPAIR_VALIDATION_v11_r12_polyphase.json").read_text())
    h.append('</table><p>指纹对照：原方法 2200/2568 = 85.670%；修复后第一二轮 2566/2568 = 99.922%。')
    if fp:
        h.append(f'第三轮 {fp["detected"]}/{fp["count"]}，累计 {fp["combined_detected"]}/{fp["combined_count"]} = {100*fp["combined_detection_rate"]:.3f}%。')
    h.append('只因实际匹配拒绝，不因阳性对照失败拒绝。弱频谱指纹检索后，以完整较短波形 ≥1 秒且归一化相关 ≥0.995 确認，检查两种 16k 采样相位；不声称覆盖人工听感相似度。</p>')
    h.append(f'<h2>BEATs 分数分布</h2><figure><img src="{beats_plot}" alt="各类目标分数直方图"><figcaption>红线为原目标分数门槛；还必须满足不相容类间隔和其余检查。</figcaption></figure>')
    ffmpeg=shutil.which("ffmpeg");ffprobe=shutil.which("ffprobe")
    assert ffmpeg and ffprobe,"ffmpeg/ffprobe unavailable"
    audio=[]
    h.append('<h2>通过音频的试听、波形与频谱</h2><p>音频为独立 MP3 文件，页面按文件名引用。优先展示 3 条通过的生成音频；生成样例不足 2 条时，以已准入的 v9 真实录音补足，并明确标注来源。</p>')
    for summary in delivery["by_class"]:
        c=summary["sound_class"];members=[r for r in doc["records"] if r["sound_class"]==c]
        gen=[r for r in members if r.get("label_provenance")=="generated_stable_audio_open_1.0"]
        gen.sort(key=lambda r:(-r["automatic_label"][0].get("confidence",0),r["sound_asset_id"]))
        samples=[];prompts=set()
        for r in gen:
            prompt=r["generation"]["prompt_id"]
            if prompt not in prompts:samples.append(r);prompts.add(prompt)
            if len(samples)==3:break
        if len(samples)<min(3,len(gen)):
            for r in gen:
                if r not in samples:samples.append(r)
                if len(samples)==min(3,len(gen)):break
        if len(samples)<2:
            real=[r for r in members if r.get("label_provenance")!="generated_stable_audio_open_1.0" and real_scores.get(r["sound_asset_id"],0)<=.38]
            samples.extend(real[:2-len(samples)])
        assert 2<=len(samples)<=3,c
        h.append(f'<h3>{esc(c)}：生成通过 {summary["accepted"]} 条，最终来源 {summary["final_originals"]}</h3>')
        for r in samples:
            sid=r["sound_asset_id"];name=sid+".mp3";mp3=folder/name
            if not mp3.exists():
                subprocess.run([ffmpeg,"-nostdin","-v","error","-n","-i",norm_path(r),"-map_metadata","-1","-ac","1","-ar","16000","-c:a","libmp3lame","-q:a","3","-threads","1",str(mp3)],check=True)
            probe=json.loads(subprocess.check_output([ffprobe,"-v","error","-show_streams","-show_format","-of","json",str(mp3)],text=True))
            stream=probe["streams"][0];assert stream["codec_name"]=="mp3" and stream["channels"]==1 and int(stream["sample_rate"])==16000
            assert float(probe["format"]["duration"])>=1
            subprocess.run([ffmpeg,"-nostdin","-v","error","-threads","1","-i",str(mp3),"-f","null","-"],check=True,stdout=subprocess.DEVNULL)
            generated_audio=r.get("label_provenance")=="generated_stable_audio_open_1.0"
            provenance="生成音频，全部检查通过；仅训练集" if generated_audio else "v9 真实录音，原准入记录"
            receipt=dict(sound_asset_id=sid,sound_class=c,file=name,path=str(mp3),sha256=sha(mp3),source_path=norm_path(r),
                         source_kind="generated_passed" if generated_audio else "v9_real_admitted",source_license=r.get("license") or r.get("source_license"),
                         allowed_splits=["train"] if generated_audio else r.get("allowed_splits"),probe=probe)
            audio.append(receipt)
            h.append(f'<figure><figcaption><code>{esc(sid)}</code> — {esc(provenance)}</figcaption><audio controls preload="none"><source src="{esc(name)}" type="audio/mpeg"></audio>')
            if generated_audio:
                h.append(f'<p>人声分数 {r["human_speech_screen"]["maximum_score"]:.6f}；提示词：{esc(r["generation"]["prompt"])}；种子 {r["generation"]["seed"]}；Whisper（仅记录）：{esc(r["whisper_transcription_recording_only"].get("blind_decoded_text",""))}</p>')
            wave,sr=sf.read(norm_path(r),dtype="float32");fig,axes=plt.subplots(2,1,figsize=(11,4.5),sharex=True)
            stride=max(1,len(wave)//20000);axes[0].plot(np.arange(0,len(wave),stride)/sr,wave[::stride],linewidth=.6);axes[0].set(ylabel="amplitude",title=sid)
            freq,times,power=spectrogram(wave,sr,nperseg=512,noverlap=384)
            axes[1].pcolormesh(times,freq/1000,10*np.log10(np.maximum(power,1e-12)),shading="auto",cmap="magma",vmin=-100,vmax=-20)
            axes[1].set(xlabel="time (s)",ylabel="frequency (kHz)",ylim=(0,8))
            fig.tight_layout();figure=save_figure(fig,sid+".png")
            h.append(f'<img src="{esc(figure)}" alt="{esc(c)} 波形与频谱"></figure>')
    h.append('<h2>许可与输入文件</h2><p>生成模型为 Stable Audio Open 1.0；许可为 Stability AI Community License。元数据包含 <code>label_provenance=generated_stable_audio_open_1.0</code>、<code>allowed_splits=["train"]</code> 和「owner 2026-10-12 接受 IV(b) 风险」。owner 已接受第 IV(b) 条关于不得用输出创建或改进基础生成式 AI 模型的限制风险。开发集和测试集仍仅使用真实录音。</p>')
    for r in delivery["outputs"]:
        h.append(f'<p><code>{esc(r["path"])}</code>：{r["records"]} 条；SHA256 <code>{r["sha256"]}</code></p>')
    h.append('<p>证据：<code>HUMAN_SPEECH_CALIBRATION_v12.json</code>、<code>ROUND3_PLAN_v12.json</code>、<code>DELIVERY_v12.json</code>、各轮原准入清单和指纹验证收据。CCI 镜像输入仅在 48g 准备，未在 CCI 执行作业或复制。</p></body></html>')
    page="\n".join(h);assert "data:" not in page
    path=folder/"index.html"
    if path.exists():assert path.read_text()==page
    else:
        with path.open("x") as f:f.write(page)
    for name in figures:
        from PIL import Image
        with Image.open(folder/name) as im:im.verify()
    once(folder/"audio_manifest.json",audio)
    once(o/"LISTEN_VALIDATION_v12.json",dict(status="pass",report=str(path),sha256=sha(path),audio_count=len(audio),figure_count=len(figures),
                                          all_audio_file_references_exist=True,all_mp3_mono_16000=True,all_mp3_decoded_successfully=True,no_data_uri=True,per_class_counts=dict(collections.Counter(r["sound_class"] for r in audio)),
                                          generated_audio_count=sum(r["source_kind"]=="generated_passed" for r in audio),real_audio_count=sum(r["source_kind"]=="v9_real_admitted" for r in audio),
                                          png_structure_verified=True,human_listening_used=False,full_browser_playback="not_run"))
    progress(o,f"v12 试听报告完成：{len(audio)} 个独立 MP3、{len(figures)} 张波形/频谱/分布图；全部文件引用和编码读回通过，无 data URI。")
    print("LISTEN_V12_DONE",str(path),len(audio),len(figures),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("plan","dedup","publish","listen"))
    for name in ("output","config","v9","cci-v9","calibration"):
        p.add_argument("--"+name,type=pathlib.Path,required=True)
    p.add_argument("--deadline",type=float,default=float("inf"))
    a=p.parse_args()
    globals()[a.action](a)

if __name__=="__main__":
    main()
