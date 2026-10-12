#!/usr/bin/env python3
"""Audit retained dialogue clips with unprompted Whisper language probabilities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

from avengine.qa.v7_dialogue_voice_audit import baseline_language_status, language_distribution


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else stream_digest(file)


def stream_digest(file):
    value = hashlib.sha256()
    for chunk in iter(lambda: file.read(1024 * 1024), b''):
        value.update(chunk)
    return value.hexdigest()


def record_job(out, name, update):
    with (out/'JOBS.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out/'JOBS.json'
        jobs = json.loads(path.read_text())
        entry = next((row for row in jobs['jobs'] if row['name'] == name), None)
        if entry is None:
            entry = {'name': name, 'machine': '48g'}
            jobs['jobs'].append(entry)
        entry.update(update)
        save(path, jobs)


def progress(out, message):
    with (out/'PROGRESS.md').open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write('- ' + stamp() + '：' + message + '\n')


def worker(manifest_path, outpath, index, workers, model_path, threads):
    out = Path(outpath)
    logfile = out/'logs'/f'language_worker_{index:02d}.log'
    with logfile.open('a', buffering=1) as log:
        sys.stdout = log
        sys.stderr = log
        name = f'language_worker_{index:02d}'
        start = time.monotonic()
        record_job(out, name, {'pid': os.getpid(), 'parent_pid': os.getppid(),
                             'process_start_ticks': Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                             'log': str(logfile), 'status': 'running', 'started_utc': stamp(),
                             'device': 'cpu', 'threads': threads,
                             'command': 'audit_dialogue_voice_language.py multiprocessing worker'})
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from avengine.qa.v7_recorded_audio import semantic_asr_comparison
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True).eval().to('cpu')
        records = []
        sounds = json.loads(Path(manifest_path).read_text())['sounds']
        for sound in sounds[index::workers]:
            tick = time.monotonic()
            target = out/'language_records'/(sound['sound_asset_id']+'.json')
            if target.exists():
                previous = json.loads(target.read_text())
                if previous.get('terminal') and previous.get('audio_sha256') == digest(sound['path']):
                    records.append(previous)
                    continue
                raise ValueError('retained language receipt conflicts with source')
            row = {'sound_asset_id': sound['sound_asset_id'], 'scenario_id': sound['scenario_id'],
                   'role_id': sound['role_id'], 'path': sound['path'],
                   'model_path': model_path, 'model_id': 'openai/whisper-small',
                   'method': 'SOT-only forward; softmax over all generation_config.lang_to_id tokens',
                   'method_source': 'https://github.com/openai/whisper/blob/main/whisper/decoding.py',
                   'language_or_text_prompt_used': False, 'audio_modified': False,
                   'threshold': 0.9, 'terminal': True, 'device': 'cpu'}
            try:
                row['audio_sha256'] = digest(sound['path'])
                row['audio_sha256_matches'] = row['audio_sha256'] == sound['sha256']
                if not row['audio_sha256_matches']:
                    raise ValueError('retained waveform SHA mismatch')
                wave, rate = sf.read(sound['path'], dtype='float32')
                if rate != 16000 or wave.ndim != 1 or len(wave) == 0 or not np.isfinite(wave).all():
                    raise ValueError('retained audio must be finite mono 16 kHz')
                if len(wave) > 30 * rate:
                    raise ValueError('refuse to truncate speech for language detection')
                features = processor.feature_extractor(wave, sampling_rate=rate, return_tensors='pt', return_attention_mask=True)
                sot = torch.full((1,1), model.generation_config.decoder_start_token_id, dtype=torch.long)
                with torch.inference_mode():
                    logits = model(input_features=features.input_features, decoder_input_ids=sot, use_cache=False).logits[0,-1].float().cpu().tolist()
                probs = language_distribution(logits, model.generation_config.lang_to_id)
                row.update(english_probability=probs['en'], detected_language=max(probs,key=probs.get),
                           language_probabilities=probs, sample_rate_hz=rate, sample_count=len(wave),
                           duration_s=len(wave)/rate, checked_at_utc=stamp())
                proof = sound['automatic_content_readback']
                recheck = semantic_asr_comparison(sound['transcript'], proof['blind_decoded_text'], sound['semantic_answer']['critical_semantic_slots'])
                row['retained_asr_recomputed_from_existing_transcription'] = recheck
                row['new_blind_transcription_run'] = False
                row['status'] = 'pass' if probs['en'] >= 0.9 else 'blocked'
                row['reason'] = None if row['status'] == 'pass' else 'english_probability_below_0.9'
            except Exception as error:
                row.update(status='blocked', reason=str(error), error_type=type(error).__name__)
            row['wall_seconds'] = time.monotonic() - tick
            save(target, row)
            records.append(row)
            print('RESULT', row['sound_asset_id'], row['status'], row.get('english_probability'), row.get('detected_language'), flush=True)
            if len(records) % 25 == 0:
                progress(out, f'旧音频语种 worker {index} 完成 {len(records)} 条。')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result = {'status': 'finished', 'finished_utc': stamp(), 'checked': len(records),
                  'wall_seconds': time.monotonic()-start, 'cpu_user_seconds': usage.ru_utime,
                  'cpu_system_seconds': usage.ru_stime, 'maxrss_kib': usage.ru_maxrss}
        record_job(out, name, result)
        progress(out, f'旧音频语种 worker {index} 完成全部 {len(records)} 条。')
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or args.workers * args.threads > 32:
        parser.error('CPU model thread budget must be 1..32')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.getpriority(os.PRIO_PROCESS,0) < 10:
        parser.error('CUDA must be disabled and niceness at least 10')
    out = Path(args.output)
    (out/'language_records').mkdir(exist_ok=True)
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, args.manifest, args.output, i, args.workers, args.model, args.threads) for i in range(args.workers)]
        for future in as_completed(futures):
            print('WORKER_DONE', json.dumps(future.result()), flush=True)
    manifest = json.loads(Path(args.manifest).read_text())
    records = [json.loads((out/'language_records'/(s['sound_asset_id']+'.json')).read_text()) for s in manifest['sounds']]
    summary = baseline_language_status(records, len(manifest['sounds']))
    probabilities = sorted(r['english_probability'] for r in records if 'english_probability' in r)
    summary.update(source_manifest=args.manifest, source_manifest_sha256=digest(args.manifest),
                   model=args.model, workers=args.workers, cpu_threads_per_worker=args.threads,
                   wall_seconds=time.monotonic()-started, finished_utc=stamp(),
                   english_probability_min=min(probabilities) if probabilities else None,
                   english_probability_max=max(probabilities) if probabilities else None,
                   english_probability_mean=sum(probabilities)/len(probabilities) if probabilities else None,
                   failure_records=[r for r in records if r['status'] != 'pass'])
    save(out/'language_baseline_audit.json', summary)
    progress(out, f'旧 {len(records)} 条语种检查收尾：{summary["passing_count"]} 条达到 0.9，{summary["nonpassing_count"]} 条未达到；前置检查 {summary["status"]}。')
    record_job(out, 'language_parent', {'status': 'finished', 'finished_utc': stamp(), 'wall_seconds': time.monotonic()-started})
    print('DONE', json.dumps({k:v for k,v in summary.items() if k != 'failure_records'}), flush=True)


if __name__ == '__main__':
    main()
