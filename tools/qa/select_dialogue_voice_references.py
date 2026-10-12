"""Qualify diverse gender-matched VCTK reference speakers by machine transcription."""
from _dialogue_voice_runtime import *
import sys
VCTK=Path('/data/datasets/VCTK-Corpus-0.92')
PREFERRED={'female':['p225','p234','p238','p248','p266','p303','p314','p335'],'male':['p226','p245','p251','p292','p302','p311','p326','p347']}
BACKUP={'female':['p228','p249','p261','p283','p307','p323','p294','p343'],'male':['p243','p298','p376','p304','p316','p334','p374','p363']}
def speaker_info():
 rows={}
 for line in (VCTK/'speaker-info.txt').read_text().splitlines()[1:]:
  p=line.split()
  if len(p)>=4:rows[p[0]]={'speaker_id':p[0],'age':int(p[1]),'gender':{'F':'female','M':'male'}[p[2]],'accent':p[3],'region':' '.join(p[4:]),'speaker_info_path':str(VCTK/'speaker-info.txt'),'speaker_info_sha256':digest(VCTK/'speaker-info.txt')}
 return rows

def qualify(speaker,judge,info):
 target=STAGE/'reference_records'/(speaker+'.json')
 if target.exists():
  row=json.loads(target.read_text())
  if row['status']=='pass':assert row['reference']['source_waveform_sha256']==digest(row['reference']['source_waveform'])
  return row
 import soundfile as sf
 trials=[]
 for number in ['012','010','008','009','001','011','004']:
  wave=VCTK/'wav48_silence_trimmed'/speaker/(speaker+'_'+number+'_mic1.flac');text=VCTK/'txt'/speaker/(speaker+'_'+number+'.txt')
  if not wave.is_file() or not text.is_file():continue
  metadata=sf.info(wave)
  if not 2.5<=metadata.duration<=12 or metadata.channels!=1:continue
  guard();transcript=text.read_text().strip();proof=judge.evaluate(wave,transcript,reference_check=True);trials.append(proof)
  if proof['language_gate']['status']=='pass':
   ref={'dataset':'VCTK-0.92','speaker_id':speaker,'original_id':speaker+'_'+number,'utterance_id':number,**info[speaker],'source_waveform':str(wave),'source_waveform_sha256':digest(wave),'source_transcript':transcript,'source_transcript_path':str(text),'source_transcript_sha256':digest(text),'microphone':'mic1','split':'train','language':'en','license_spdx':'CC-BY-4.0','license_chain_status':'pass','license_text':str(VCTK/'license_text.txt'),'license_text_sha256':digest(VCTK/'license_text.txt'),'corpus_readme':str(VCTK/'README.txt'),'source_uri':'https://datashare.ed.ac.uk/handle/10283/3443','attribution':'Yamagishi, Veaux, MacDonald (2019), CSTR VCTK Corpus 0.92, University of Edinburgh','reference_ASR_receipt':str(target),'reference_ASR':proof,'source_waveform_already_corpus_silence_trimmed':True,'additional_trimming_or_repetition':False}
   result={'status':'pass','reference':ref,'trials':trials};save(target,result);print('REFERENCE_PASS',speaker,number,proof['ASR']['wer'],info[speaker]['accent'],flush=True);return result
  if len(trials)>=3:break
 result={'status':'blocked','speaker_id':speaker,'trials':trials,'reason':'no clean reference among at most three candidates'};save(target,result);print('REFERENCE_FAIL',speaker,flush=True);return result

def main():
 tick=time.monotonic();register('stage2_references',STAGE/'logs/references.log',threads=2);info=speaker_info();judge=WhisperJudge(2);selected={g:[] for g in PREFERRED};swaps=[]
 for gender,voices in PREFERRED.items():
  for speaker in voices:
   assert info[speaker]['gender']==gender
   row=qualify(speaker,judge,info)
   if row['status']!='pass':
    for alternative in BACKUP[gender]:
     if alternative in [r['speaker_id'] for r in selected[gender]]:continue
     replacement=qualify(alternative,judge,info)
     if replacement['status']=='pass':row=replacement;swaps.append({'retired':speaker,'replacement':alternative,'gender':gender,'reason':'reference failed'});break
   if row['status']=='pass':selected[gender].append(row['reference'])
 result={'status':'pass' if all(len(v)==8 for v in selected.values()) else 'blocked','selected':selected,'replacements':swaps,'backup_candidates':BACKUP,'finished_utc':stamp(),'wall_seconds':time.monotonic()-tick};save(STAGE/'references.json',result);progress('新增参考音色挑选完成：'+str({g:len(v) for g,v in selected.items()})+'，参考转写与语种检查 '+result['status']+'。');finish('stage2_references',tick,reference_status=result['status']);print('REFERENCE_SUMMARY',json.dumps({k:v for k,v in result.items() if k!='selected'}),flush=True)
if __name__=='__main__':main()
