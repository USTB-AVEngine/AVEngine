"""Private resident Whisper judge and task receipts for CPU dialogue expansion."""
from __future__ import annotations
from pathlib import Path
import os,json,hashlib,datetime,time,fcntl,shutil,resource
ROOT=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_dialogue_voices_20261012')
STAGE=ROOT/'stage2'
R=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_families_20261004/round5_20261004')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.writing');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def progress(message):
 with (ROOT/'PROGRESS.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('- '+stamp()+'：'+message+'\n')
def job(name,**value):
 with (ROOT/'JOBS.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p=ROOT/'JOBS.json';j=json.loads(p.read_text());row=next((r for r in j['jobs'] if r['name']==name),None)
  if row is None:row={'name':name,'machine':'48g'};j['jobs'].append(row)
  row.update(value);save(p,j)
def register(name,log,**extra):
 job(name,pid=os.getpid(),parent_pid=os.getppid(),process_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],log=str(log),command=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),started_utc=stamp(),status='running',device='cpu',nice=os.getpriority(os.PRIO_PROCESS,0),**extra)
def finish(name,tick,**extra):
 u=resource.getrusage(resource.RUSAGE_SELF);status=extra.pop('status','finished');job(name,status=status,finished_utc=stamp(),wall_seconds=time.monotonic()-tick,cpu_user_seconds=u.ru_utime,cpu_system_seconds=u.ru_stime,**extra)
_last_disk_check=0.0
def guard():
 c=json.loads((STAGE/'config.json').read_text())
 if time.time()>c['deadline_epoch']-1200:raise RuntimeError('stop for final checks twenty minutes before deadline')
 if shutil.disk_usage('/data').free<c['min_free_bytes']:raise RuntimeError('/data free space below 150 GB')
 global _last_disk_check
 if time.monotonic()-_last_disk_check>60:
  import subprocess
  lines=subprocess.check_output(['du','-sb',str(ROOT),c['worktree']],text=True).splitlines()
  own_bytes=sum(int(line.split()[0]) for line in lines)
  if own_bytes>c['max_new_bytes']:raise RuntimeError('authorized outputs/worktree exceed 20 GB')
  _last_disk_check=time.monotonic()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=''  or os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('CPU/nice contract not satisfied')
from avengine.qa.v7_dialogue_voice_audit import contains_han as has_han, english_language_gate as gate, language_distribution
class WhisperJudge:
 def __init__(self,threads=2):
  import torch
  from transformers import WhisperProcessor,WhisperForConditionalGeneration
  torch.set_num_threads(threads);torch.set_num_interop_threads(1);self.torch=torch
  self.model_path='/data/models/whisper-small';self.processor=WhisperProcessor.from_pretrained(self.model_path,local_files_only=True);self.model=WhisperForConditionalGeneration.from_pretrained(self.model_path,local_files_only=True).eval().to('cpu');self.model.generation_config.forced_decoder_ids=None
 def evaluate(self,path,reference=None,slots=None,reference_check=False):
  import soundfile as sf,numpy as np
  from scipy.signal import resample_poly
  from math import gcd
  from avengine.qa.v7_recorded_audio import semantic_asr_comparison
  tick=time.monotonic();wave,rate=sf.read(path,dtype='float32');original_rate=rate;original_count=len(wave)
  if wave.ndim==2:wave=wave.mean(axis=1)
  if rate!=16000:
   g=gcd(rate,16000);wave=resample_poly(wave,16000//g,rate//g).astype('float32');rate=16000
  if not len(wave) or not np.isfinite(wave).all() or len(wave)>30*rate:raise ValueError('invalid audio or would require truncation')
  features=self.processor.feature_extractor(wave,sampling_rate=rate,return_tensors='pt',return_attention_mask=True)
  with self.torch.inference_mode():
   sot=self.torch.full((1,1),self.model.generation_config.decoder_start_token_id,dtype=self.torch.long)
   encoded=self.model.get_encoder()(features.input_features,return_dict=True)
   logits=self.model(encoder_outputs=encoded,decoder_input_ids=sot,use_cache=False).logits[0,-1].float().tolist();probs=language_distribution(logits,self.model.generation_config.lang_to_id);top=max(probs,key=probs.get)
   en=self.model.generate(encoder_outputs=encoded,attention_mask=features.attention_mask,language='en',task='transcribe',do_sample=False,max_new_tokens=256)
   forced=self.processor.batch_decode(en,skip_special_tokens=True)[0].strip()
   free_tokens=self.model.generate(encoder_outputs=encoded,attention_mask=features.attention_mask,language=None,task='transcribe',do_sample=False,max_new_tokens=256)
   free=self.processor.batch_decode(free_tokens,skip_special_tokens=True)[0].strip()
  proof=semantic_asr_comparison(reference,forced,slots or []) if reference is not None else {'status':'not_applicable','wer':None,'all_critical_slots_correct':None}
  if reference_check:
   # References have no QA answer slots. WER and negation still use the identical normalizer.
   proof['all_critical_slots_correct']=proof['negation_counts_match'];proof['status']='pass' if proof['wer']<=.1 and proof['negation_counts_match'] else 'blocked';proof['reference_sentence_check']=True
  proof.update(model_id='openai/whisper-small',model_path=self.model_path,reference_text=reference,blind_decoded_text=forced,reference_text_or_answer_prompted_into_ASR=False,human_listening_used=False,device='cpu',language='en',audio_sha256=digest(path))
  lid={'method':'SOT-only forward; softmax over every language token; no language/text prompt','english_probability':probs['en'],'top_language':top,'top_languages':sorted(probs.items(),key=lambda x:-x[1])[:5],'transcript_free_language':free,'free_language_forced':False,'language_probabilities':probs}
  lid['shared_encoder_output_reused']=True
  lid['encoder_reuse_equivalence_receipt']=str(STAGE/'evidence/encoder_reuse_equivalence.json')
  # A Chinese negative is rejected by the observed language/free-text criteria independently of WER.
  result=gate(proof,top,free) if reference is not None else {'status':'blocked' if top!='en' or has_han(free) else 'unclassified','reasons':(['language_top1_not_en'] if top!='en' else [])+(['free_transcription_contains_Han'] if has_han(free) else []),'forced_English_ASR_comparison':'not_applicable_without_authored_English_reference'}
  return {'path':str(path),'audio_sha256':proof['audio_sha256'],'ASR':proof,'language_identification':lid,'language_gate':result,'sample_rate_hz':original_rate,'sample_count':original_count,'duration_s':original_count/original_rate,'wall_seconds':time.monotonic()-tick,'checked_utc':stamp(),'audio_cropped_or_repeated':False}
