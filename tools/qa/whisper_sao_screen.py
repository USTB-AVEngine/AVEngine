"""Apply Whisper-small's standard multilingual no-speech decoding to SAO."""
import argparse, hashlib, json, pathlib, time, types
import torch, soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.generation_whisper import _get_attr_from_logit_processors, WhisperNoSpeechDetection

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--output",type=pathlib.Path,required=True);p.add_argument("--round",type=int,required=True)
 p.add_argument("--model-dir",type=pathlib.Path,required=True);p.add_argument("--deadline",type=float,required=True)
 a=p.parse_args();o=a.output;start=time.time();torch.set_num_threads(8);torch.set_num_interop_threads(1)
 processor=WhisperProcessor.from_pretrained(str(a.model_dir),local_files_only=True)
 model=WhisperForConditionalGeneration.from_pretrained(str(a.model_dir),local_files_only=True).eval().requires_grad_(False).to("cuda")
 model.generation_config.forced_decoder_ids=None;model.generation_config.language=None
 model.generation_config.max_length=None
 capture=[];original=model._need_fallback
 def audit(self,sequence,outputs,index,processors,config,vocab_size,temperature):
  result=original(sequence,outputs,index,processors,config,vocab_size,temperature)
  lp=float(self._retrieve_avg_logprobs(outputs[index]["scores"],sequence,temperature))
  ns=float(_get_attr_from_logit_processors(processors,WhisperNoSpeechDetection,"no_speech_prob")[index])
  capture.append(dict(raw_text=processor.decode(sequence,skip_special_tokens=True),avg_logprob=lp,
   no_speech_prob=ns,temperature=temperature,needs_fallback=bool(result[0]),should_skip=bool(result[1])))
  return result
 # Record upstream decisions in memory; do not alter the model or library files.
 model._need_fallback=types.MethodType(audit,model)
 source=[json.loads(s) for s in (o/f"model_scores_r{a.round}.jsonl").read_text().splitlines()]
 path=o/f"whisper_standard_r{a.round}.jsonl"
 completed=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
 done={r["sound_asset_id"]:r for r in completed}
 with path.open("a") as stream:
  for entry in source:
   r=entry["candidate"];sid=r["sound_asset_id"]
   if sid in done:
    assert done[sid]["audio_sha256"]==r["sha256"];continue
   if time.time()+60>=a.deadline:break
   capture.clear();x,sr=sf.read(r["path"],dtype="float32")
   assert sr==16000 and x.ndim==1 and sha(r["path"])==r["sha256"]
   f=processor(x,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
   torch.manual_seed(r["seed"])
   with torch.inference_mode():
    out=model.generate(f.input_features.to("cuda"),attention_mask=f.attention_mask.to("cuda"),
     task="transcribe",language=None,max_new_tokens=128,return_timestamps=True,
     return_dict_in_generate=True,output_scores=True,no_speech_threshold=.6,
     logprob_threshold=-1.,compression_ratio_threshold=2.4,temperature=(0.,.2,.4,.6,.8,1.),
     condition_on_prev_tokens=False)
   text=processor.batch_decode(out["sequences"],skip_special_tokens=True)[0].strip()
   assert capture,"Whisper decode audit missing"
   last=capture[-1]
   # Do not accept a lexical decoding merely because timestamp extraction omitted it.
   if not text and not last["should_skip"] and any(c.isalpha() or c.isdigit() for c in last["raw_text"]):
    text=last["raw_text"].strip()
   lexical=any(c.isalpha() or c.isdigit() for c in text)
   proof=dict(sound_asset_id=sid,status="blocked" if lexical else "pass",model_id="openai/whisper-small",
    model_path=str(a.model_dir),audio_sha256=r["sha256"],blind_decoded_text=text,
    original_greedy_decoded_text=r["automatic_no_speech"]["blind_decoded_text"],
    raw_decode_attempts=list(capture),avg_logprob=last["avg_logprob"],no_speech_prob=last["no_speech_prob"],
    standard_no_speech_skipped=last["should_skip"],no_speech_threshold=.6,logprob_threshold=-1.,
    compression_ratio_threshold=2.4,temperatures=[0.,.2,.4,.6,.8,1.],seed=r["seed"],
    language_forced=False,reference_text_prompted=False,human_listening_used=False,device="cuda",precision="float32",
    rule="upstream standard no-speech rule (p_no_speech>0.6 AND avg_logprob<-1.0); reject any Unicode letter/digit in remaining blind transcription",
    implementation_source="https://raw.githubusercontent.com/openai/whisper/main/whisper/transcribe.py",
    timestamp_omission_cannot_hide_lexical_decoding=True)
   stream.write(json.dumps(proof,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
   completed.append(proof)
   if len(completed)%24==0:print("WHISPER_STANDARD",len(completed),"PASS",sum(r["status"]=="pass" for r in completed),flush=True)
 receipt=dict(round=a.round,count=len(completed),expected=len(source),complete=len(completed)==len(source),
  passed=sum(r["status"]=="pass" for r in completed),seconds=time.time()-start,
  no_speech_threshold=.6,logprob_threshold=-1.,compression_ratio_threshold=2.4,
  model_sha256=sha(a.model_dir/"model.safetensors"),gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(),
  human_listening_used=False,controls=str(o/"evidence"/"whisper_standard_pilot.json"))
 (o/f"whisper_standard_receipt_r{a.round}.json").write_text(json.dumps(receipt,indent=2))
 with (o/"PROGRESS.md").open("a") as f:
  f.write("- "+time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())+f" 第 {a.round} 轮补全 Whisper 标准无语音解码：{len(completed)}/{len(source)}，通过 {receipt['passed']}；原始转写/解码概率全部保留。\n")
 print("WHISPER_STANDARD_DONE",json.dumps(receipt),flush=True)
if __name__=="__main__":main()
