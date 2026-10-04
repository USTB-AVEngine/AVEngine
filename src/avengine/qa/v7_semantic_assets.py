"""Small CPU natural-TTS canaries for authored semantic scenarios.

CosyVoice2 code, weights and a local licence receipt are explicit inputs.
No model is downloaded or formant engine substituted by this module.
Native roles, visual gender and automatic ASR remain separate admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy import signal
from avengine.qa.v7_recorded_audio import NATURAL_SPEECH_POLICY

def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def synthesize(script_path, output, *, model_dir, code_receipt, maximum_utterances=8,
               scenario_ids=(), cpu_threads=2):
    if not 1 <= maximum_utterances <= 8:
        raise ValueError('canary synthesis allows one to eight utterances; large production belongs to the owner')
    model_dir = Path(model_dir)
    receipt = json.loads(Path(code_receipt).read_text())
    license_path = Path(receipt['model_license_path'])
    if (not (model_dir/'cosyvoice2.yaml').is_file()
            or receipt.get('engine') != 'CosyVoice2' or not receipt.get('code_revision')
            or receipt.get('model_id') != 'FunAudioLLM/CosyVoice2-0.5B'
            or receipt.get('model_license') != 'Apache-2.0'
            or 'Apache License' not in license_path.read_text()
            or not receipt.get('model_revision')
            or 'license: apache-2.0' not in Path(receipt.get('model_license_declaration_path','')).read_text()):
        raise ValueError('actual CosyVoice2 model/code/licence receipt is incomplete')
    script_bytes = Path(script_path).read_bytes()
    script_sha256 = hashlib.sha256(script_bytes).hexdigest()
    script = json.loads(script_bytes)
    selected = [s for s in script['scenarios'] if not scenario_ids or s['id'] in scenario_ids]
    if not selected:
        raise ValueError('no requested authored scenarios')
    tasks = [(s,r) for s in selected for r in s['roles']][:maximum_utterances]
    # Validate all reference plans before importing a heavyweight model.
    for scenario, role in tasks:
        reference = role.get('voice_reference_plan') or {}
        if (reference.get('dataset') != 'VCTK-0.92' or reference.get('license_chain_status') != 'pass'
                or reference.get('license_spdx') != 'CC-BY-4.0' or reference.get('split') != 'train'
                or reference.get('gender') != role['required_visual_gender']
                or not Path(reference.get('source_waveform','')).is_file()
                or not Path(reference.get('source_transcript_path','')).is_file()
                or Path(reference['source_transcript_path']).read_text().strip()!=reference['source_transcript']):
            raise ValueError('actual licensed VCTK training reference is missing')
        info=sf.info(reference['source_waveform'])
        if info.channels!=1 or not 1 <= info.duration <= 30:
            raise ValueError('reference must be a complete mono VCTK sentence within the model limit')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    if torch.cuda.is_available():
        raise ValueError('this entry point is CPU only')
    torch.set_num_threads(max(1, min(int(cpu_threads),2)))
    from cosyvoice.cli.cosyvoice import CosyVoice2
    engine = CosyVoice2(str(model_dir), load_jit=False, load_trt=False, fp16=False)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    sounds = []
    for scenario, role in tasks:
        reference = role['voice_reference_plan']; config=role['synthesis_config']
        torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
        chunks=[]
        for result in engine.inference_zero_shot(role['transcript'],reference['source_transcript'],
                        reference['source_waveform'],stream=False,speed=1.0,text_frontend=False):
            chunk=result['tts_speech'].detach().cpu().numpy().reshape(-1)
            if not len(chunk) or not np.isfinite(chunk).all():
                raise ValueError('natural TTS returned empty or nonfinite samples')
            chunks.append(chunk)
        if not chunks:
            raise ValueError('natural TTS returned no complete speech')
        complete = np.concatenate(chunks)
        rate=int(engine.sample_rate); divisor=math.gcd(rate,16000)
        samples=signal.resample_poly(complete,16000//divisor,rate//divisor)
        sid=scenario['id']+'__'+role['role_id']
        path=output/(sid+'.wav'); sf.write(path,samples,16000,subtype='FLOAT')
        readback,sr=sf.read(path,always_2d=True)
        if sr!=16000 or len(readback)!=len(samples):
            raise ValueError('complete authored speech changed on disk')
        semantic = next(x for x in scenario['utterances'] if x['answer']==role['meaning_answer'])
        sounds.append({'sound_asset_id':sid,'scenario_id':scenario['id'],'role_id':role['role_id'],
           'script_author':'project_author','transcript':role['transcript'],'semantic_answer':semantic,
           'speech_policy':NATURAL_SPEECH_POLICY,'voice_reference':reference,
           'voice_gender':role['voice_gender_plan'],'emission_mechanism':'human_live_natural_tts',
           'source_real_waveform':False,'content_class':'speech','is_program_synthesized':True,
           'synthesis_config':dict(config,model_license=receipt['model_license'],actual_waveform=str(path)),'path':str(path),'sha256':_file_digest(path),
           'qualification':'natural_TTS_candidate_pending_ASR_and_native_role',
           'formal_blockers':['automatic_speech_content_readback_missing','native_actor_visual_gender_and_complete_utterance_not_rendered'],
           'synthesis_readback':{'path':str(path),'sample_rate_hz':16000,'sample_count':len(samples),
                'source_sample_rate_hz':rate,'source_sample_count':len(complete),'duration_s':len(samples)/16000,
                'transcript':role['transcript'],'content_complete':True,'trimmed_or_repeated':False,
                'resampling_method':'scipy.signal.resample_poly','device':'cpu',
                'engine':{'name':'CosyVoice2','natural_neural_tts':True,'model_id':receipt['model_id'],
                          'model_revision':receipt['model_revision'],'model_license':receipt['model_license'],
                          'model_license_path':str(license_path),'code_revision':receipt['code_revision']}},
           'automatic_content_readback':{'status':'pending','human_listening_used':False}})
        print('SYNTHESIZED',sid,len(samples)/16000,flush=True)
        (output/'semantic_manifest.partial.json').write_text(json.dumps({'sounds':sounds},ensure_ascii=False,indent=2))
    manifest={'sounds':sounds,'dialogues':{'scenarios':selected},'script_source':str(script_path),'script_source_sha256':script_sha256,
       'actual_TTS_reference_speaker_ids':sorted({s['voice_reference']['speaker_id'] for s in sounds}),
       'complete_natural_TTS_candidates':len(sounds),'formal_usable':0,
       'qualification':'actual complete natural speech; automatic ASR and native role evidence still required'}
    (output/'semantic_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--code-receipt',type=Path,required=True)
    p.add_argument('--maximum-utterances',type=int,default=8);p.add_argument('--scenario',action='append',default=[])
    a=p.parse_args();r=synthesize(a.script,a.output,model_dir=a.model_dir,code_receipt=a.code_receipt,
                                 maximum_utterances=a.maximum_utterances,scenario_ids=a.scenario)
    print(json.dumps({'complete_natural_speech_clips':len(r['sounds']),'formal_usable':0}))
if __name__=='__main__':
    main()
