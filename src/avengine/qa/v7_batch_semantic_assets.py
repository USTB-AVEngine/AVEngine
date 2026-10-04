"""CPU batch natural speech with bounded retries and blind ASR.

The authored catalog and the producer's formal standing roster are immutable
inputs. Passing source speech is distinct from a native dialogue episode.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import soundfile as sf
from scipy import signal

from avengine.qa.v7_recorded_audio import NATURAL_SPEECH_POLICY, semantic_asr_comparison

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)

def make_batch_plan(script_path, roster_path, code_receipt, model_dir):
    script_path, roster_path = Path(script_path), Path(roster_path)
    script = json.loads(script_path.read_bytes())
    registry = json.loads(roster_path.read_bytes())
    receipt = json.loads(Path(code_receipt).read_bytes())
    model_dir = Path(model_dir)
    if (registry.get('schema') != 'avengine_v7_human_roster_v1'
            or registry.get('summary', {}).get('dual_backend_usable_minor') !=
            sum(len(m['minor_variants']) for m in registry['majors'])):
        raise ValueError('formal dual-backend roster count is not complete')
    if (receipt.get('engine') != 'CosyVoice2'
            or receipt.get('model_id') != 'FunAudioLLM/CosyVoice2-0.5B'
            or receipt.get('model_license') != 'Apache-2.0'
            or not receipt.get('code_revision') or not receipt.get('model_revision')
            or not (model_dir/'cosyvoice2.yaml').is_file()
            or 'Apache License' not in Path(receipt['model_license_path']).read_text()
            or 'license: apache-2.0' not in Path(receipt['model_license_declaration_path']).read_text()):
        raise ValueError('local natural model/code licence receipt incomplete')
    root = Path(registry['artifact_root']).resolve()
    if roster_path.resolve() != root/'registrations/human_roster.json':
        raise ValueError('formal roster path/root mismatch')
    standing = []
    for major in registry['majors']:
        if major.get('gender') not in {'male','female'}:
            continue  # do not infer lx gender from its name or waveform
        for variant in major['minor_variants']:
            if (variant.get('dual_backend_static_usable') is not True
                    or variant.get('habitat', {}).get('stand', {}).get('status') != 'pass'
                    or variant.get('ue', {}).get('status') != 'pass'
                    or variant.get('habitat', {}).get('status') != 'pass'):
                continue
            point = np.asarray(variant['mouth_local_glb_m'], dtype=float)
            glb = (root/variant['habitat']['glb']).resolve()
            if not glb.is_relative_to(root) or not glb.is_file() or point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('registered stand/mouth resource missing')
            standing.append({'asset_id':variant['variant_id'], 'body_identity_id':major['major_id'],
                'visual_gender':major['gender'], 'pose':'stand', 'color':variant['color'],
                'habitat_glb':str(glb), 'habitat_glb_sha256':digest(glb),
                'mouth_local_glb_m':point.tolist(), 'mouth_local_ue_cm':variant['mouth_local_ue_cm'],
                'mouth_source':variant['mouth']['source'],
                'source_registry':str(roster_path),'source_registry_sha256':digest(roster_path)})
    tasks = []
    for scenario in script['scenarios']:
        voices, bodies, colors = set(), set(), set()
        for role in scenario['roles']:
            ref = role.get('voice_reference_plan') or {}
            if (ref.get('dataset') != 'VCTK-0.92' or ref.get('license_chain_status') != 'pass'
                    or ref.get('license_spdx') != 'CC-BY-4.0' or ref.get('split') != 'train'
                    or ref.get('gender') != role.get('required_visual_gender')
                    or ref.get('gender') != role.get('voice_gender_plan')
                    or not Path(ref.get('source_waveform','')).is_file()
                    or not Path(ref.get('source_transcript_path','')).is_file()
                    or Path(ref['source_transcript_path']).read_text().strip() != ref['source_transcript']):
                raise ValueError('licensed and gender-matched VCTK reference missing')
            if ref['speaker_id'] in voices:
                raise ValueError('different scenario roles must use different speakers')
            voices.add(ref['speaker_id'])
            candidates = [s for s in standing if s['visual_gender'] == ref['gender']
                          and s['body_identity_id'] not in bodies and s['color'] not in colors
                          and s['color'] in {'red','green','blue','yellow','white','black'}]
            candidates.sort(key=lambda x: hashlib.sha256(
                (scenario['id']+'|'+role['role_id']+'|'+x['asset_id']).encode()).digest())
            if not candidates:
                raise ValueError('not enough distinct registered standing appearances')
            actor = deepcopy(candidates[0]);bodies.add(actor['body_identity_id']);colors.add(actor['color'])
            semantic = next(x for x in scenario['utterances'] if x['answer'] == role['meaning_answer'])
            if not semantic.get('critical_semantic_slots') or semantic['text'] != role['transcript']:
                raise ValueError('authored text/meaning annotation mismatch')
            tasks.append({'sound_asset_id':scenario['id']+'__'+role['role_id'],
                'scenario_id':scenario['id'],'role_id':role['role_id'],
                'transcript':role['transcript'],'semantic_answer':deepcopy(semantic),
                'voice_reference':deepcopy(ref),'voice_gender':role['voice_gender_plan'],
                'synthesis_config':deepcopy(role['synthesis_config']),
                'standing_actor_plan':actor})
    return {'schema':'v7_natural_batch_plan_v1','tasks':tasks,'dialogues':{'scenarios':script['scenarios']},
            'script_source':str(script_path),'script_source_sha256':digest(script_path),
            'source_registry':str(roster_path),'source_registry_sha256':digest(roster_path),
            'code_receipt':str(code_receipt),'model_dir':str(model_dir),'model_receipt':receipt,
            'static_registry_count':len(standing),'total_tasks':len(tasks),
            'visual_gender_plan_matched':True,'native_placement_verified':False,
            'attempt_cap_includes_initial':True,'max_attempts':3,
            'actual_reference_speaker_ids':sorted({r['voice_reference']['speaker_id'] for r in tasks})}

def _worker(plan_path, output, worker_index, workers, asr_model, cpu_threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    plan = json.loads(Path(plan_path).read_text());output = Path(output)
    worker_log = output/'logs'/f'worker_{worker_index:02d}.log'
    with worker_log.open('a', buffering=1) as log:
        import sys
        sys.stdout = log;sys.stderr = log
        save_json(output/'jobs'/f'worker_{worker_index:02d}.json', {
            'pid':os.getpid(),'parent_pid':os.getppid(),'worker_index':worker_index,
            'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            'plan':str(plan_path),'log':str(worker_log),'device':'cpu'})
        started = time.time()
        engine = CosyVoice2(plan['model_dir'],load_jit=False,load_trt=False,fp16=False)
        processor = WhisperProcessor.from_pretrained(asr_model,local_files_only=True)
        asr = WhisperForConditionalGeneration.from_pretrained(asr_model,local_files_only=True).eval()
        asr.generation_config.forced_decoder_ids = None
        refs = {}
        for task in plan['tasks'][worker_index::workers]:
            ref = task['voice_reference']
            if ref['speaker_id'] not in refs:
                engine.add_zero_shot_spk(ref['source_transcript'],ref['source_waveform'],ref['speaker_id'])
                refs[ref['speaker_id']] = (ref['source_waveform'],ref['source_transcript'])
            elif refs[ref['speaker_id']] != (ref['source_waveform'],ref['source_transcript']):
                raise ValueError('same voice identity has conflicting reference cache')
        for task in plan['tasks'][worker_index::workers]:
            target = output/'records'/(task['sound_asset_id']+'.json')
            if target.is_file():
                previous = json.loads(target.read_text())
                if previous.get('plan_sha256') != digest(plan_path):
                    raise ValueError('cannot resume a different authored plan')
                if previous.get('terminal') is True:
                    if previous.get('status')=='pass' and digest(previous['sound']['path']) != previous['sound']['sha256']:
                        raise ValueError('saved passing source waveform changed')
                    continue
                attempts = previous['attempts']
            else:
                attempts = []
            sound = None
            for attempt_index in range(len(attempts),plan['max_attempts']):
                one_started = time.time()
                seed = int(task['synthesis_config']['seed']) + 1000003*attempt_index
                path = output/'audio'/f"{task['sound_asset_id']}__attempt_{attempt_index+1}.wav"
                attempt = {'attempt':attempt_index+1,'seed':seed,'path':str(path)}
                try:
                    if path.exists():
                        raise ValueError('orphaned_attempt_waveform_retained_not_overwritten')
                    torch.manual_seed(seed);np.random.seed(seed%(2**32));random.seed(seed)
                    ref = task['voice_reference']
                    chunks = []
                    with torch.inference_mode():
                        for result in engine.inference_zero_shot(task['transcript'],
                                ref['source_transcript'],ref['source_waveform'],
                                zero_shot_spk_id=ref['speaker_id'],
                                stream=False,speed=1.0,text_frontend=False):
                            chunk = result['tts_speech'].detach().cpu().numpy().reshape(-1)
                            if not len(chunk) or not np.isfinite(chunk).all():
                                raise ValueError('natural_TTS_empty_or_nonfinite')
                            chunks.append(chunk)
                    if not chunks:raise ValueError('natural_TTS_no_complete_speech')
                    complete = np.concatenate(chunks);rate = int(engine.sample_rate)
                    divisor = math.gcd(rate,16000)
                    samples = signal.resample_poly(complete,16000//divisor,rate//divisor)
                    sf.write(path,samples,16000,subtype='FLOAT')
                    waveform,sr = sf.read(path,dtype='float32')
                    if waveform.ndim != 1 or sr != 16000 or len(waveform)!=len(samples):
                        raise ValueError('natural_waveform_readback_changed')
                    features = processor(waveform,sampling_rate=sr,return_tensors='pt',return_attention_mask=True)
                    with torch.inference_mode():
                        tokens = asr.generate(features.input_features,attention_mask=features.attention_mask,
                            language='en',task='transcribe',max_new_tokens=256,do_sample=False)
                    hypothesis = processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
                    comparison = semantic_asr_comparison(task['transcript'],hypothesis,
                                            task['semantic_answer']['critical_semantic_slots'])
                    proof = {**comparison,'model_id':'openai/whisper-small','model_path':asr_model,
                         'source':str(target),'audio_sha256':digest(path),
                         'reference_text':task['transcript'],'blind_decoded_text':hypothesis,
                         'human_listening_used':False,'reference_text_or_answer_prompted_into_ASR':False,
                         'device':'cpu','language':'en'}
                    # Never feed the answer/reference as an ASR prompt.
                    passed = proof['status']=='pass'
                    attempt.update(status='pass' if passed else 'blocked',duration_s=len(waveform)/sr,
                          ASR=proof,reason=None if passed else 'automatic_ASR_WER_or_critical_meaning_failed',
                          seconds=time.time()-one_started)
                    if passed:
                        receipt = plan['model_receipt']
                        sound = {**deepcopy(task),'script_author':'project_author',
                            'speech_policy':NATURAL_SPEECH_POLICY,
                            'voice_identity':'vctk:'+ref['speaker_id'],'voice_preset':'vctk:'+ref['speaker_id'],
                            'emission_mechanism':'human_live_natural_tts','content_class':'speech',
                            'is_program_synthesized':True,'source_real_waveform':False,
                            'source_dataset':'project_authored_CosyVoice2','source_uri':receipt['code_source'],
                            'path':str(path),'sha256':digest(path),'automatic_content_readback':proof,
                            'synthesis_config':dict(task['synthesis_config'],seed=seed,
                                model_license=receipt['model_license'],actual_waveform=str(path),
                                zero_shot_reference_cache_in_memory=True),
                            'synthesis_readback':{'path':str(path),'sample_rate_hz':sr,
                                'sample_count':len(waveform),'source_sample_rate_hz':rate,
                                'source_sample_count':len(complete),'duration_s':len(waveform)/sr,
                                'transcript':task['transcript'],'content_complete':True,
                                'trimmed_or_repeated':False,'device':'cpu',
                                'resampling_method':'scipy.signal.resample_poly',
                                'engine':{'name':'CosyVoice2','natural_neural_tts':True,
                                    'model_id':receipt['model_id'],'model_revision':receipt['model_revision'],
                                    'model_license':receipt['model_license'],'code_revision':receipt['code_revision'],
                                    'model_license_path':receipt['model_license_path']}},
                            'qualification':'natural_TTS_content_pass_registered_stand_plan_pending_native',
                            'formal_blockers':['native_registered_standing_actor_and_complete_placement_missing']}
                except Exception as error:
                    attempt.update(status='blocked',reason=str(error),error_type=type(error).__name__,
                                   seconds=time.time()-one_started)
                    print(traceback.format_exc(),flush=True)
                attempts.append(attempt)
                terminal = sound is not None or len(attempts)>=plan['max_attempts']
                save_json(target,{'sound_asset_id':task['sound_asset_id'],'scenario_id':task['scenario_id'],
                      'role_id':task['role_id'],'plan_sha256':digest(plan_path),
                      'status':'pass' if sound is not None else 'dropped' if terminal else 'retrying',
                      'terminal':terminal,'attempts':attempts,'sound':sound,
                      'standing_actor_plan':task['standing_actor_plan']})
                print('RESULT',task['sound_asset_id'],attempt['attempt'],attempt['status'],
                      attempt.get('duration_s'),attempt.get('ASR',{}).get('blind_decoded_text'),flush=True)
                if sound is not None:break
        return {'worker':worker_index,'seconds':time.time()-started}

def assemble_batch(plan_path, output):
    output = Path(output);plan = json.loads(Path(plan_path).read_text())
    rows = [json.loads((output/'records'/(t['sound_asset_id']+'.json')).read_text()) for t in plan['tasks']
            if (output/'records'/(t['sound_asset_id']+'.json')).is_file()]
    terminal = [r for r in rows if r['terminal']]
    sounds = [r['sound'] for r in terminal if r['status']=='pass']
    counts = []
    for s in plan['dialogues']['scenarios']:
        rr = [r for r in terminal if r['scenario_id']==s['id']]
        good = [r for r in rr if r['status']=='pass']
        counts.append({'scenario_id':s['id'],'planned_lines':len(s['roles']),
            'processed_lines':len(rr),'usable_lines':len(good),'dropped_lines':len(rr)-len(good),
            'all_planned_lines_passed':len(good)==len(s['roles']),
            'at_least_two_usable_roles':len(good)>=2,
            'actual_reference_speakers':[r['sound']['voice_reference']['speaker_id'] for r in good],
            'durations_s':[r['sound']['synthesis_readback']['duration_s'] for r in good],
            'native_usable_questions':0})
    summary = {'planned_scenarios':len(counts),'planned_lines':len(plan['tasks']),
        'processed_terminal_lines':len(terminal),'passed_lines':len(sounds),
        'dropped_lines':sum(r['status']=='dropped' for r in terminal),
        'first_attempt_passed':sum(r['status']=='pass' and len(r['attempts'])==1 for r in terminal),
        'total_attempts':sum(len(r['attempts']) for r in rows),
        'passing_attempts':sum(a['status']=='pass' for r in rows for a in r['attempts']),
        'complete_scenarios':sum(r['all_planned_lines_passed'] for r in counts),
        'scenarios_with_at_least_two_usable_roles':sum(r['at_least_two_usable_roles'] for r in counts),
        'line_pass_rate':len(sounds)/max(1,len(plan['tasks'])),
        'attempt_cap_includes_initial':True,'maximum_attempts':plan['max_attempts'],
        'failure_reason_counts':dict(Counter(a.get('reason') for r in rows for a in r['attempts'] if a['status']!='pass')),
        'native_dialogue_questions':0,'human_listening':False,
        'actual_reference_speaker_ids':sorted({r['voice_reference']['speaker_id'] for r in sounds}),
        'plan_sha256':digest(plan_path),'source_registry_sha256':plan['source_registry_sha256'],
        'complete':len(terminal)==len(plan['tasks'])}
    save_json(output/'semantic_manifest.json',{'sounds':sounds,'dialogues':plan['dialogues'],
        'script_source':plan['script_source'],'script_source_sha256':plan['script_source_sha256'],
        'source_registry':plan['source_registry'],'source_registry_sha256':plan['source_registry_sha256'],
        'batch_summary':summary,'formal_usable':0})
    save_json(output/'synthesis_statistics.json',{'summary':summary,'scenarios':counts})
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script',type=Path,required=True);p.add_argument('--roster',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--code-receipt',type=Path,required=True)
    p.add_argument('--asr-model',default='/data/models/whisper-small')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    if not 1<=a.workers<=8 or not 1<=a.cpu_threads<=2:raise ValueError('CPU batch is limited to 16 threads')
    if a.output.exists() and not a.resume:raise ValueError('output exists; explicit resume preserves prior attempts')
    for sub in ['audio','records','logs','jobs']:(a.output/sub).mkdir(parents=True,exist_ok=True)
    plan = make_batch_plan(a.script,a.roster,a.code_receipt,a.model_dir);plan_path = a.output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('resume inputs changed')
    else:save_json(plan_path,plan)
    started=time.time()
    errors = []
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_worker,str(plan_path),str(a.output),i,a.workers,a.asr_model,a.cpu_threads) for i in range(a.workers)]
        for f in as_completed(futures):
            try:print('WORKER_DONE',json.dumps(f.result()),flush=True)
            except Exception as error:
                errors.append(str(error));print('WORKER_ERROR',traceback.format_exc(),flush=True)
            print('PROGRESS',json.dumps(assemble_batch(plan_path,a.output)),flush=True)
    summary=assemble_batch(plan_path,a.output);summary['seconds']=time.time()-started;summary['worker_errors']=errors
    save_json(a.output/'run_summary.json',summary);print('BATCH_DONE',json.dumps(summary),flush=True)
    if errors or not summary['complete']:raise SystemExit(1)
if __name__=='__main__':main()
