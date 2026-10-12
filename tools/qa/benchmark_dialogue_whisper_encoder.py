"""Compare legacy and shared-encoder Whisper outputs on full English/Chinese clips."""
from _dialogue_voice_runtime import *
import numpy as np,soundfile as sf,math
from scipy.signal import resample_poly

class LegacyWhisperJudge(WhisperJudge):
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
   logits=self.model(input_features=features.input_features,decoder_input_ids=sot,use_cache=False).logits[0,-1].float().tolist();probs=language_distribution(logits,self.model.generation_config.lang_to_id);top=max(probs,key=probs.get)
   en=self.model.generate(input_features=features.input_features,attention_mask=features.attention_mask,language='en',task='transcribe',do_sample=False,max_new_tokens=256)
   forced=self.processor.batch_decode(en,skip_special_tokens=True)[0].strip()
   free_tokens=self.model.generate(input_features=features.input_features,attention_mask=features.attention_mask,language=None,task='transcribe',do_sample=False,max_new_tokens=256)
   free=self.processor.batch_decode(free_tokens,skip_special_tokens=True)[0].strip()
  proof=semantic_asr_comparison(reference,forced,slots or []) if reference is not None else {'status':'not_applicable','wer':None,'all_critical_slots_correct':None}
  if reference_check:
   # References have no QA answer slots. WER and negation still use the identical normalizer.
   proof['all_critical_slots_correct']=proof['negation_counts_match'];proof['status']='pass' if proof['wer']<=.1 and proof['negation_counts_match'] else 'blocked';proof['reference_sentence_check']=True
  proof.update(model_id='openai/whisper-small',model_path=self.model_path,reference_text=reference,blind_decoded_text=forced,reference_text_or_answer_prompted_into_ASR=False,human_listening_used=False,device='cpu',language='en',audio_sha256=digest(path))
  lid={'method':'SOT-only forward; softmax over every language token; no language/text prompt','english_probability':probs['en'],'top_language':top,'top_languages':sorted(probs.items(),key=lambda x:-x[1])[:5],'transcript_free_language':free,'free_language_forced':False,'language_probabilities':probs}
  # A Chinese negative is rejected by the observed language/free-text criteria independently of WER.
  result=gate(proof,top,free) if reference is not None else {'status':'blocked' if top!='en' or has_han(free) else 'unclassified','reasons':(['language_top1_not_en'] if top!='en' else [])+(['free_transcription_contains_Han'] if has_han(free) else []),'forced_English_ASR_comparison':'not_applicable_without_authored_English_reference'}
  return {'path':str(path),'audio_sha256':proof['audio_sha256'],'ASR':proof,'language_identification':lid,'language_gate':result,'sample_rate_hz':original_rate,'sample_count':original_count,'duration_s':original_count/original_rate,'wall_seconds':time.monotonic()-tick,'checked_utc':stamp(),'audio_cropped_or_repeated':False}


def main():
    register('stage2_encoder_equivalence',STAGE/'logs/encoder_equivalence.log',threads=1)
    start=time.monotonic();judge=WhisperJudge(1);results=[]
    old=json.loads((R/'natural_batch_ready/semantic_manifest.json').read_text())['sounds'];bad=next(s for s in old if s['sound_asset_id']=='daily_reminder_home_living_room__role_2');inputs=[old[0],old[30],old[100],old[200],bad]
    inputs += [{'path':r['sound']['path'],'transcript':r['task']['transcript'],'semantic_answer':r['task']['semantic_answer']} for r in [json.loads(p.read_text()) for p in sorted((STAGE/'task_records').glob('*.json'))] if r.get('terminal') and r['status']=='pass'][:3]
    inputs += [{'path':n['path']} for n in json.loads((STAGE/'negative_selection.json').read_text())[:2]]
    for row in inputs:
     baseline=LegacyWhisperJudge.evaluate(judge,row['path'],row.get('transcript'),row.get('semantic_answer',{}).get('critical_semantic_slots'))
     tick=time.monotonic();wave,rate=sf.read(row['path'],dtype='float32')
     if wave.ndim==2:wave=wave.mean(axis=1)
     if rate!=16000:
      g=math.gcd(rate,16000);wave=resample_poly(wave,16000//g,rate//g).astype('float32')
     f=judge.processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True);model=judge.model;torch=judge.torch
     with torch.inference_mode():
      encoded=model.get_encoder()(f.input_features,return_dict=True)
      sot=torch.full((1,1),model.generation_config.decoder_start_token_id,dtype=torch.long)
      logits=model(encoder_outputs=encoded,decoder_input_ids=sot,use_cache=False).logits[0,-1].float().tolist();probs=language_distribution(logits,model.generation_config.lang_to_id)
      forced=model.generate(encoder_outputs=encoded,attention_mask=f.attention_mask,language='en',task='transcribe',do_sample=False,max_new_tokens=256)
      free=model.generate(encoder_outputs=encoded,attention_mask=f.attention_mask,language=None,task='transcribe',do_sample=False,max_new_tokens=256)
     forced_text=judge.processor.batch_decode(forced,skip_special_tokens=True)[0].strip();free_text=judge.processor.batch_decode(free,skip_special_tokens=True)[0].strip();delta=max(abs(probs[k]-baseline['language_identification']['language_probabilities'][k]) for k in probs);equal=forced_text==baseline['ASR']['blind_decoded_text'] and free_text==baseline['language_identification']['transcript_free_language'] and delta==0
     result={'path':row['path'],'audio_sha256':digest(row['path']),'forced_text_identical':forced_text==baseline['ASR']['blind_decoded_text'],'free_text_identical':free_text==baseline['language_identification']['transcript_free_language'],'maximum_language_probability_difference':delta,'status':'pass' if equal else 'blocked','baseline_seconds':baseline['wall_seconds'],'cached_encoder_seconds':time.monotonic()-tick,'baseline':baseline,'cached_forced_text':forced_text,'cached_free_text':free_text};results.append(result);print('EQUIVALENCE',row['path'],equal,result['baseline_seconds'],result['cached_encoder_seconds'],flush=True)
    summary={'status':'pass' if all(r['status']=='pass' for r in results) and len(results)==10 else 'blocked','checked':len(results),'all_forced_and_free_texts_identical':all(r['forced_text_identical'] and r['free_text_identical'] for r in results),'maximum_probability_difference':max(r['maximum_language_probability_difference'] for r in results),'baseline_mean_seconds':sum(r['baseline_seconds'] for r in results)/len(results),'cached_mean_seconds':sum(r['cached_encoder_seconds'] for r in results)/len(results),'rows':results,'finished_utc':stamp()};save(STAGE/'evidence/encoder_reuse_equivalence.json',summary);finish('stage2_encoder_equivalence',start,equivalence_status=summary['status']);print('BENCHMARK_DONE',json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    main()
