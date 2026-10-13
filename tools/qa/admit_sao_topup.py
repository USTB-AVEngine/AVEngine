"""Screen SAO top-up candidates with unchanged BEATs and recorded admission."""
from __future__ import annotations
import argparse, collections, copy, hashlib, importlib.util, json, pathlib, pickle, sys, time, types
import numpy as np
import soundfile as sf
from scipy import signal
from avengine.dataset.sound_segments import measure_activity
from avengine.qa.v7_sao_admission import OWNER_ACCEPTANCE
from avengine.qa.v7_sound_extensions import ontology_relatives, score_and_rival
from avengine.qa.v7_auto_admission import audio_fingerprint, fingerprint_index, fingerprint_alignment
from avengine.qa.v7_extended_admission import build_recorded_admission
from avengine.timeline.v7_recorded_catalog import recorded_catalog_entry
from avengine.timeline.v7_audio_sources import filter_admission, source_allowed_in_split

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def dump(path,value):
 pathlib.Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))

def lines(path):
 return [json.loads(s) for s in pathlib.Path(path).read_text().splitlines() if s]

def progress(o,text):
 with (o/"PROGRESS.md").open("a") as f:f.write("- "+time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())+" "+text+"\n")

def target_mapping(config,v9,base_mapping):
 result={c:{"target_mids":list(base_mapping["class_mids"].get(c,[]))} for c in config["classes"]}
 for c,v in result.items():
  if not v["target_mids"]:
   v["target_mids"]=next(r["target_mids"] for r in v9["records"] if r["sound_class"]==c and r.get("target_mids"))
 return result

def norm_path(r):
 return r.get("normalized_path") or (r.get("normalization") or {}).get("path") or (r.get("normalized_research_derivative") or {}).get("path") or r["path"]

def maximum_group_activity(intervals,rate,max_gap_s=1.0):
 best=total=0.;end=None
 for lo,hi in intervals:
  if end is None or (lo-end)/rate>max_gap_s:total=0.
  total+=(hi-lo)/rate;best=max(best,total);end=hi
 return best

def load_scored(o):
 rows=[]
 for path in sorted(o.glob("model_scores_r*.jsonl")):rows+=lines(path)
 assert len({r["candidate"]["sound_asset_id"] for r in rows})==len(rows)
 standard={}
 for path in sorted(o.glob("whisper_standard_r*.jsonl")):
  standard.update({r["sound_asset_id"]:r for r in lines(path)})
 calibration_path=o/"HUMAN_SPEECH_CALIBRATION_v11.json"
 calibration=json.loads(calibration_path.read_text()) if calibration_path.exists() else None
 for r in rows:
  candidate=r["candidate"]
  proof=standard.get(candidate["sound_asset_id"],candidate.get("automatic_no_speech",{}))
  assert proof.get("audio_sha256")==candidate["sha256"]
  recorded=copy.deepcopy(proof)
  recorded["historical_status"]=recorded.get("status")
  recorded.update(status="recorded",role="recording_only",participates_in_admission=False,
   rule="transcription retained only; any historical Whisper decision is superseded")
  candidate.pop("automatic_no_speech",None)
  candidate["whisper_transcription_recording_only"]=recorded
  r.pop("whisper_pass",None)
  if calibration:
   maximum_head=max(calibration["heads"],key=lambda h:r["beats"]["scores_by_mid"][h["mid"]])
   value=r["beats"]["scores_by_mid"][maximum_head["mid"]]
   valid=calibration["status"]=="pass"
   passed=valid and value<=calibration["threshold"]
   candidate["human_speech_screen"]=dict(status="pass" if passed else "blocked" if valid else "blocked_calibration_failed",
    model_id="BEATs_AS2M_cpt2",audio_sha256=candidate["sha256"],maximum_score=value,maximum_head=maximum_head,
    threshold=calibration["threshold"],heads=calibration["heads"],calibration_path=str(calibration_path),
    calibration_sha256=sha(calibration_path),human_listening_used=False,
    calibration={k:calibration[k] for k in ["status","positive_count","positive_pass_rate","negative_count","negative_block_rate","checkpoint_sha256","v9_sha256"]})
   r["human_speech_pass"]=passed
  else:r["human_speech_pass"]=False
 return rows

def score(a):
 import torch
 from transformers import WhisperProcessor, WhisperForConditionalGeneration
 start=time.time();o=a.output
 config=json.loads(a.config.read_text());v9=json.loads(a.v9.read_text())
 mapping=json.loads(a.mapping.read_text());ontology=json.loads(a.ontology.read_text())
 targets=target_mapping(config,v9,mapping);relatives=ontology_relatives(ontology)
 source=a.beats_source
 pkg=types.ModuleType("_v7_sao_beats");pkg.__path__=[str(source)];sys.modules[pkg.__name__]=pkg
 spec=importlib.util.spec_from_file_location(pkg.__name__+".BEATs",source/"BEATs.py")
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 checkpoint=torch.load(a.beats_weight,map_location="cpu",weights_only=False)
 beats=mod.BEATs(mod.BEATsConfig(checkpoint["cfg"]));beats.load_state_dict(checkpoint["model"],strict=True)
 beats=beats.eval().requires_grad_(False).to("cuda")
 labels={int(k):v for k,v in checkpoint["label_dict"].items()}
 names={r["id"]:r["name"] for r in ontology}
 processor=WhisperProcessor.from_pretrained(str(a.whisper),local_files_only=True)
 whisper=WhisperForConditionalGeneration.from_pretrained(str(a.whisper),local_files_only=True).eval().requires_grad_(False).to("cuda")
 whisper.generation_config.forced_decoder_ids=None
 whisper.generation_config.language=None;whisper.generation_config.task="transcribe"
 records=lines(o/f"candidates_r{a.round}.jsonl")
 out=o/f"model_scores_r{a.round}.jsonl"
 completed=lines(out) if out.exists() else []
 done={r["candidate"]["sound_asset_id"] for r in completed}
 print("SCREEN_MODELS_READY",len(records),len(done),flush=True)
 with out.open("a") as stream:
  for r in records:
   if r["sound_asset_id"] in done:continue
   if time.time()+30>=a.deadline:break
   x,sr=sf.read(r["path"],dtype="float32",always_2d=True)
   assert sr==16000 and x.shape[1]==1 and np.isfinite(x).all()
   x=x[:,0];assert sha(r["path"])==r["sha256"]
   chunks=[]
   for lo in range(0,len(x),80000):
    wave=x[lo:lo+80000]
    if len(wave)<16000:wave=np.pad(wave,(0,16000-len(wave)))
    with torch.inference_mode():prob=beats.extract_features(torch.from_numpy(wave)[None].to("cuda"))[0][0].cpu().numpy()
    chunks.append(prob)
   probs=np.mean(chunks,axis=0);top=np.argsort(probs)[-8:][::-1]
   b=dict(sound_asset_id=r["sound_asset_id"],sha256=r["sha256"],
    scores_by_mid={labels[k]:float(v) for k,v in enumerate(probs)},
    top8=[dict(mid=labels[int(k)],label=names.get(labels[int(k)]),score=float(probs[k])) for k in top],
    chunk_count=len(chunks),input_sample_count=len(x),input_rate_hz=16000,model_device="cuda",precision="float32")
   confidence,rival,rival_mid=score_and_rival(b,targets[r["sound_class"]]["target_mids"],relatives)
   features=processor(x,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
   with torch.inference_mode():
    ids=whisper.generate(features.input_features.to("cuda"),attention_mask=features.attention_mask.to("cuda"),
     task="transcribe",language=None,do_sample=False,max_new_tokens=128)
   text=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
   lexical=any(ch.isalpha() or ch.isdigit() for ch in text)
   asr=dict(status="blocked" if lexical else "pass",model_id="openai/whisper-small",model_path=str(a.whisper),
    audio_sha256=r["sha256"],blind_decoded_text=text,token_ids=ids[0].tolist(),
    language_forced=False,reference_text_prompted=False,human_listening_used=False,
    device="cuda",precision="float32",rule="reject any Unicode letter or digit in blind automatic multilingual transcription",
    machine_only_no_audibility_claim=True)
   activity=measure_activity(x,16000,sound_class="unknown")
   group=maximum_group_activity(activity["source_activity_intervals_samples"],16000)
   active=activity["active_duration_s"]
   act=dict(status="pass" if active>=1 and group>=1 else "blocked",audio_sha256=r["sha256"],
    active_duration_s=active,longest_group_s=group,group_max_gap_s=1.0,
    measurement=activity,minimum_effective_activity_s=1.0)
   candidate=dict(r,candidate_classes=[r["sound_class"]],target_mids=targets[r["sound_class"]]["target_mids"],
    source_metadata=dict(title=r["prompt"].split(". Dry sound")[0],description=r["prompt"],
     tags=[r["sound_class"]],license=r["source_license"],uploader="Stable Audio Open 1.0"),
    official_collection_labels=[],official_zip_crc_verified=False,
    source_original_waveform=r["stereo_path"],source_origin=r["stereo_path"],
    provenance_status="generated_candidate_pending_machine_screens",automatic_no_speech=asr,
    automatic_activity_readback=act,license=r["source_license"])
   row=dict(candidate=candidate,beats=b,target_confidence=confidence,margin=confidence-rival,rival_mid=rival_mid,
    beats_pass=confidence>=.1917129977 and confidence-rival>=.05,whisper_pass=not lexical,
    activity_pass=act["status"]=="pass")
   stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");stream.flush();completed.append(row)
   if len(completed)%12==0:
    print("SCORED",len(completed),"BEATS",sum(t["beats_pass"] for t in completed),
     "WHISPER",sum(t["whisper_pass"] for t in completed),"ACTIVITY",sum(t["activity_pass"] for t in completed),flush=True)
 receipt=dict(count=len(completed),expected=len(records),complete=len(completed)==len(records),
  seconds=time.time()-start,checkpoint=str(a.beats_weight),checkpoint_sha256=sha(a.beats_weight),
  source_files={n:sha(source/n) for n in ["BEATs.py","backbone.py","modules.py"]},
  whisper_model_sha256=sha(a.whisper/"model.safetensors"),whisper_forced_language=None,
  beats_threshold=.1917129977,beats_minimum_margin=.05,targets=targets,human_listening_used=False,
  gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(),device="cuda",torch_version=torch.__version__)
 dump(o/f"screen_receipt_r{a.round}.json",receipt)
 progress(o,f"第 {a.round} 轮 Whisper-small / BEATs / 活动检查完成：{len(completed)}/{len(records)}，未降低门槛。")
 print("SCREEN_DONE",json.dumps(receipt),flush=True)

def duplicate(alignment):
 return alignment["exact_normalized_pcm"] or alignment["matched_hashes"]>=12 and alignment["aligned_fraction"]>=.25

def candidate_reference_keys(counts, query_hash_count, reference_indices):
 # Either direction can pass alignment: a short reference inside a long query
 # must not be pruned by the long query's denominator.
 return {key for key,n in counts.items() if n>=12 and
  (n/max(1,query_hash_count)>=.20 or n/max(1,len(reference_indices[key]))>=.20)}

def dedup(a):
 from tools.qa.readmit_sao_topup import dedup as repaired_dedup
 repaired_dedup(a)

def build(a):
 o=a.output;start=time.time()
 config=json.loads(a.config.read_text());v9=json.loads(a.v9.read_text());mapping=json.loads(a.mapping.read_text())
 targets=target_mapping(config,v9,mapping);scored=load_scored(o)
 dedup=json.loads((o/f"deduplication_{a.stage}.json").read_text())
 assert dedup["complete_for_listed_roots"] and not dedup["errors"]
 classes=list(config["classes"]);groups={c:mapping["specific_metadata_groups"][c] for c in classes if mapping["specific_metadata_groups"].get(c)}
 candidates=[r["candidate"] for r in scored];beats=[r["beats"] for r in scored]
 assert {r["sound_asset_id"] for r in candidates}==set(dedup["records"])
 result=build_recorded_admission(pool=[r for r in v9["records"] if r["sound_class"]!="speech_playback"],
  old_inventory=v9["records"],old_admission={"records":[]},vctk_lineages=[],fsd_rows=candidates,beats_rows=beats,clap_rows=[],
  ontology=json.loads(a.ontology.read_text()),collection_mids={},class_targets=targets,deduplications=[dedup],
  output=o/f"admission_{a.stage}",existing_gate_classes=tuple(classes),existing_subtype_metadata_groups=groups,
  generated_source_authorization=dict(owner_acceptance=OWNER_ACCEPTANCE,allowed_splits=["train"],classes=classes))
 raw=result["records"];summary=[]
 for c in classes:
  members=[r for r in raw if r["sound_class"]==c];passed=[r for r in members if r.get("strict_v7_eligible")]
  reasons=collections.Counter(reason for r in members if not r.get("strict_v7_eligible") for reason in r.get("current_policy_blockers",[]))
  # Show the independently measured screens; builder blockers can overlap.
  srows=[r for r in scored if r["candidate"]["sound_class"]==c]
  summary.append(dict(sound_class=c,existing_originals=config["classes"][c]["existing_originals"],
   needed=config["classes"][c]["needed"],candidates=len(members),machine_pass=len(passed),rejection_reasons=dict(reasons),
   beats_below_threshold=sum(r["target_confidence"]<.1917129977 for r in srows),
   beats_margin_below_threshold=sum(r["margin"]<.05 for r in srows),
   human_speech_rejections=sum(not r["human_speech_pass"] for r in srows),
   whisper_recording_only=True,
   activity_rejections=sum(not r["activity_pass"] for r in srows)))
 summary_doc=dict(stage=a.stage,by_class=summary,all_candidates=len(raw),machine_pass=sum(r["machine_pass"] for r in summary),
  retry_classes=[r["sound_class"] for r in summary if r["machine_pass"]<r["needed"]],
  raw_admission=str(o/f"admission_{a.stage}"/"dry_recorded_admission.json"),
  human_listening_used=False,seconds=time.time()-start)
 dump(o/f"SUMMARY_{a.stage}.json",summary_doc)
 progress(o,f"{a.stage} 原 thin top-up 构建器准入完成：{summary_doc['machine_pass']}/{len(raw)} 机器通过。")
 print("ADMISSION_DONE",json.dumps(summary_doc,ensure_ascii=False),flush=True)

def publish(a):
 o=a.output;summary=json.loads((o/"SUMMARY_final.json").read_text())
 raw=json.loads((o/"admission_final"/"dry_recorded_admission.json").read_text())["records"]
 v9=json.loads(a.v9.read_text());cci=json.loads(a.cci_v9.read_text())
 selected=[];class_summary=[]
 for s in summary["by_class"]:
  valid=[r for r in raw if r["sound_class"]==s["sound_class"] and r.get("strict_v7_eligible")]
  by_prompt=collections.defaultdict(list)
  for r in valid:by_prompt[r["generation"]["prompt_id"]].append(r)
  for group in by_prompt.values():group.sort(key=lambda r:(-r["automatic_label"][0]["confidence"],r["sound_asset_id"]))
  chosen=[]
  while len(chosen)<s["needed"] and any(by_prompt.values()):
   for name in sorted(by_prompt):
    if by_prompt[name] and len(chosen)<s["needed"]:chosen.append(by_prompt[name].pop(0))
  selected+=chosen
  class_summary.append(dict(s,added_to_v10=len(chosen),qualified_spares=len(valid)-len(chosen),
   final_originals=s["existing_originals"]+len(chosen),deficit=max(0,s["needed"]-len(chosen))))
 manifest=str(o/"admission_final"/"dry_recorded_admission.json")
 new=[recorded_catalog_entry(r,manifest) for r in selected]
 admitted,rejected=filter_admission(new,split="train");assert not rejected and len(admitted)==len(new)
 assert not filter_admission(new,split="dev")[0] and not filter_admission(new,split="test")[0]
 mirror=None
 for r,c in zip(v9["records"],cci["records"]):
  assert r["sound_asset_id"]==c["sound_asset_id"]
  p=r.get("normalized_path");q=c.get("normalized_path")
  if p and q and q.endswith(p) and p!=q:mirror=q[:-len(p)];break
 assert mirror
 def cci_record(row):
  r=copy.deepcopy(row)
  for holder,key in [(r,"normalized_path"),(r["normalization"],"path"),(r["screens"]["normalization"],"path")]:
   assert holder[key].startswith("/data/jzy/");holder[key]=mirror+holder[key]
  return r
 readbacks=[]
 for r in v9["records"]+new:
  p=norm_path(r);info=sf.info(p)
  assert info.samplerate==16000 and info.channels==1 and info.frames>0,str(p)
  readbacks.append(dict(sound_asset_id=r["sound_asset_id"],path=p,sample_rate_hz=info.samplerate,channels=info.channels,sample_count=info.frames))
 for r in new:
  assert r["allowed_splits"]==["train"] and r["label_provenance"]=="generated_stable_audio_open_1.0"
  assert r["license"]=="Stability AI Community License" and r["owner_acceptance"]==OWNER_ACCEPTANCE
  assert not source_allowed_in_split(r,"dev") and not source_allowed_in_split(r,"test")
 origins=lambda r:str(r.get("original_group") or r.get("source_original_id") or r.get("fsd50k_id") or r.get("source_origin") or r["sound_asset_id"])
 for s in class_summary:
  actual={origins(r) for r in v9["records"]+new if r["sound_class"]==s["sound_class"]}
  assert len(actual)==s["final_originals"]
  assert s["final_originals"]>=30 or s["deficit"]>0
 all_class_original_counts={}
 for c in sorted({r["sound_class"] for r in v9["records"]+new}):
  all_class_original_counts[c]=len({origins(r) for r in v9["records"]+new if r["sound_class"]==c})
 assert len(all_class_original_counts)==27
 before=json.loads((o/"evidence"/"v9_sha256_before.json").read_text())
 for path in [a.v9,a.cci_v9]:assert before[path.name]==sha(path)
 outputs=[]
 for base,old_path,name,added in [(v9,a.v9,"source_admission_v10_classes27_english_sao.json",new),
   (cci,a.cci_v9,"cci_source_admission_v10_classes27_english_sao.json",[cci_record(r) for r in new])]:
  doc=copy.deepcopy(base);doc["profile"]+="-sao-train-only-topup-20261012"
  doc["derived_from_v9"]=dict(path=str(old_path),sha256=sha(old_path))
  doc["sao_topup"]=dict(owner_acceptance=OWNER_ACCEPTANCE,license="Stability AI Community License",
   allowed_splits=["train"],machine_only=True,added_records=len(new),shortfalls=[r for r in class_summary if r["deficit"]])
  doc["source_manifests"]=list(base["source_manifests"])+[manifest];doc["records"]+=added
  ids=[r["sound_asset_id"] for r in doc["records"]];assert len(ids)==len(set(ids))
  with (o/name).open("x") as f:json.dump(doc,f,ensure_ascii=False,indent=2,allow_nan=False)
  outputs.append(dict(path=str(o/name),records=len(ids),sha256=sha(o/name)))
 pull=[r["normalized_path"].lstrip("/") for r in new]+[(o/"cci_source_admission_v10_classes27_english_sao.json").as_posix().lstrip("/")]
 (o/"cci_pull_list_v10.txt").write_text("\n".join(pull)+"\n")
 dump(o/"cci_expected_new_files_v10.json",{r["sound_asset_id"]:dict(path=mirror+r["normalized_path"],sha256=r["normalization"]["sha256"]) for r in new})
 dump(o/"V10_READBACK.json",dict(status="pass",records=len(readbacks),readbacks=readbacks,
  duplicate_ids=False,generated_train_only=True,cci_prefix=mirror,
  all_class_original_counts=all_class_original_counts,human_listening_used=False))
 before=json.loads((o/"evidence"/"v9_sha256_before.json").read_text())
 for path in [a.v9,a.cci_v9]:assert before[path.name]==sha(path)
 dump(o/"DELIVERY.json",dict(outputs=outputs,by_class=class_summary,added_records=len(new),machine_pass=summary["machine_pass"],
  candidate_count=summary["all_candidates"],v9_unchanged=True,cci_mirror_prefix=mirror,
  cci_mirror_execution="not_run; task forbids CCI jobs and does not authorize writing CCI",
  rejection_reasons=dict(collections.Counter(reason for r in raw if not r.get("strict_v7_eligible") for reason in r.get("current_policy_blockers",[]))),
  report=str(o/"REPORT.html"),human_listening_used=False,validation_status="pass",
  missing_classes=[r["sound_class"] for r in class_summary if r["deficit"]],
  all_class_original_counts=all_class_original_counts,
  other_class_shortfalls={c:n for c,n in all_class_original_counts.items() if n<30 and c not in {r["sound_class"] for r in class_summary}}))
 progress(o,f"v10 新文件与拉取清单完成：新增 {len(new)} 条，共 {len(v9['records'])+len(new)} 条；v9 SHA256 未变，单声道 16k/train-only/唯一 ID 全部通过。")
 print("V10_DONE",json.dumps(outputs),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("action",choices=["score","dedup","build","publish"]);p.add_argument("--output",type=pathlib.Path,required=True)
 p.add_argument("--config",type=pathlib.Path,required=True);p.add_argument("--v9",type=pathlib.Path,required=True)
 p.add_argument("--cci-v9",type=pathlib.Path);p.add_argument("--mapping",type=pathlib.Path,required=True);p.add_argument("--ontology",type=pathlib.Path,required=True)
 p.add_argument("--beats-source",type=pathlib.Path);p.add_argument("--beats-weight",type=pathlib.Path);p.add_argument("--whisper",type=pathlib.Path)
 p.add_argument("--round",type=int,default=1);p.add_argument("--stage",default="r1");p.add_argument("--deadline",type=float,default=float("inf"))
 a=p.parse_args();{"score":score,"dedup":dedup,"build":build,"publish":publish}[a.action](a)
if __name__=="__main__":main()
