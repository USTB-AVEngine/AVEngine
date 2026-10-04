#!/usr/bin/env python3
"""Automatically qualify recorded VCTK inputs, preserving original clip clocks.

No original file is rewritten. Speaker labels and transcripts are checked
against corpus metadata, Speech uses the already calibrated BEATs gate, and
only a completed prohibited/evaluation reference scan can close admission.
The legacy sound-library split is retained as a voice roster split; it is not
a STARSS/SAVVY evaluation-origin label. Each derivative is a constant gain.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, sys, time, types
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from avengine.timeline.continuous_pilot import write_json
from avengine.qa.v7_asset_admission import _loudness
from avengine.dataset.sound_segments import measure_activity
from avengine.qa.v7_auto_admission import label_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_beats(source, weights):
    import torch
    package = types.ModuleType('_v7_alpha_beats'); package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + '.BEATs', source / 'BEATs.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model = module.BEATs(module.BEATsConfig(checkpoint['cfg']))
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    return model, {int(k): v for k, v in checkpoint['label_dict'].items()}


def lineage(record, corpus, wave):
    """Use the previously delivered VCTK crop-chain correlation/residual bounds."""
    original = record['original_id']; speaker = record['speaker_id']; best = None
    for mic in ['mic1', 'mic2']:
        source = corpus / 'wav48_silence_trimmed' / speaker / (original + '_' + mic + '.flac')
        if not source.is_file():
            continue
        read = subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-i',str(source),
            '-ac','1','-ar','16000','-f','f32le','pipe:1'],capture_output=True,check=True)
        candidate = np.frombuffer(read.stdout,dtype='<f4').astype(float)
        if len(candidate) < len(wave):
            continue
        correlation = signal.correlate(candidate, wave, mode='valid', method='fft')
        squares = np.r_[0., np.cumsum(candidate * candidate)]
        energy = squares[len(wave):] - squares[:-len(wave)]
        ratios = correlation / np.sqrt(np.maximum(energy * np.dot(wave,wave), 1e-30))
        start = int(np.argmax(ratios)); part = candidate[start:start+len(wave)]
        gain = float(np.dot(part,wave) / max(np.dot(part,part),1e-30))
        residual = float(np.linalg.norm(part*gain-wave) / max(np.linalg.norm(wave),1e-30))
        proof = {'status':'pass' if ratios[start]>=.995 and residual<=.1 else 'blocked',
            'original_file':str(source),'original_file_sha256':sha(source),
            'original_id':original,'contiguous_original_crop_start_sample_16khz':start,
            'readback_correlation':float(ratios[start]),'relative_residual':residual,
            'correlation_minimum':.995,'relative_residual_maximum':.1,
            'bounds_source':'v7-families round4 VCTK original-crop lineage',
            'sample_count':len(wave),'constant_gain_from_original':gain,'microphone':mic}
        if best is None or proof['readback_correlation'] > best['readback_correlation']:
            best = proof
        if proof['status']=='pass':
            break
    return best or {'status':'blocked','reason':'original VCTK mono recording missing'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--model-source',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--class-map',type=Path,required=True)
    p.add_argument('--label-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deduplication',type=Path)
    p.add_argument('--model-scores',type=Path,help='CPU BEATs JSONL from an existing model environment')
    args = p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); cpu=time.process_time()
    inventory=json.loads(args.inventory.read_text())['records']
    speakers={r.split()[0]:r.split()[2] for r in (args.corpus/'speaker-info.txt').read_text().splitlines()[1:] if len(r.split())>=3}
    license_path=args.corpus/'license_text.txt'; text=license_path.read_text()
    if '4.0' not in text or not any(x in text.lower() for x in ['attribution','cc by','cc-by']):
        raise ValueError('VCTK corpus license does not contain the required CC BY 4.0 grant')
    retained=out/'VCTK_license_text.txt'
    if retained.exists() and retained.read_text()!=text:raise ValueError('retained license changed')
    if not retained.exists():retained.write_text(text)
    calibration=json.loads(args.label_calibration.read_text())
    threshold=calibration['threshold']; margin=calibration['minimum_margin']
    mapping=json.loads(args.class_map.read_text())['class_mids']
    records=[]; model=None; labels=None
    external_scores={r['sound_asset_id']:r for r in (json.loads(line) for line in args.model_scores.read_text().splitlines())} if args.model_scores else {}
    for i,original in enumerate(inventory):
        sid=original['sound_asset_id']; receipt_path=out/'technical'/ (sid+'.json')
        if receipt_path.exists():
            row=json.loads(receipt_path.read_text())
            if row['source_pcm_sha256']!=sha(original['path']):raise ValueError('source audio changed after receipt')
            records.append(row);continue
        row=dict(original);row.update(source_type='recorded',source_dataset='VCTK-Corpus 0.92',
            source_origin=original['source_uri'],source_pcm_sha256=sha(original['path']),
            sound_class='speech_playback',emission_mechanism='unassigned_recorded_dry',
            compatible_emission_mechanisms=['human_live_recorded_dry','loudspeaker_playback'])
        wave,rate=sf.read(row['path'],dtype='float64',always_2d=True)
        if rate!=16000 or wave.shape[1]!=1 or not np.isfinite(wave).all():raise ValueError('invalid VCTK mono clock')
        x=wave[:,0];screens={}
        screens['license']={'status':'pass','license':'CC-BY-4.0','original_id':row['original_id'],
            'source_uri':row['source_uri'],'attribution':row['attribution'],
            'retained_text':str(retained),'license_sha256':sha(license_path)}
        chain=lineage(row,args.corpus,x)
        screens['provenance']={**chain,'dataset':'VCTK-Corpus 0.92','original_clip_is_mono':True,
            'not_real_spatial_or_evaluation_origin':True,
            'legacy_library_split':row['legacy_library_split'],
            'legacy_split_meaning':'v6 dry-library speaker roster train/eval; not benchmark origin'}
        meta=json.loads(Path(row['metadata_path']).read_text())
        transcript=args.corpus/'txt'/row['speaker_id']/(row['original_id']+'.txt')
        normalize_text=lambda s:' '.join(str(s).split())
        speaker_ok=speakers.get(row['speaker_id'])==row['gender']==meta.get('gender') and row['gender'] in ['F','M']
        transcript_ok=transcript.is_file() and normalize_text(transcript.read_text())==normalize_text(row['transcript'])==normalize_text(meta.get('transcript'))
        screens['speaker_labels']={'status':'pass' if speaker_ok and transcript_ok else 'blocked',
            'original_speaker_info':str(args.corpus/'speaker-info.txt'),'speaker_id':row['speaker_id'],
            'gender':row['gender'],'gender_label_matches_original':speaker_ok,
            'transcript_matches_original':transcript_ok,'perceptual_gender_classifier_used':False}
        activity=measure_activity(x,rate,sound_class='unknown')
        screens['automatic_activity']={'status':'pass' if activity['source_activity_intervals_samples'] else 'blocked',
            'activity_intervals_samples':activity['source_activity_intervals_samples'],
            'coordinate':'original_clip_samples','sample_count':len(x),'sample_rate_hz':rate,
            'duration_s':len(x)/rate,'measurement':activity,'activity_is_qa_event_count':False}
        loud=_loudness(row['path'])
        if loud['input_i'] is None or loud['input_tp'] is None:raise ValueError('VCTK loudness not measurable')
        gain=min(-23-loud['input_i'],-3.02-loud['input_tp'])
        dest=out/'normalized'/ (sid+'.wav');dest.parent.mkdir(exist_ok=True)
        for attempt in range(3):
            sf.write(dest,x*10**(gain/20),rate,subtype='FLOAT');post=_loudness(dest)
            if post['input_tp'] is not None and post['input_tp']<=-3:break
            gain+=-3.02-post['input_tp']
        derivative,_=sf.read(dest,dtype='float64');error=float(np.max(np.abs(derivative-x*10**(gain/20))))
        screens['normalization']={'status':'pass' if post['input_tp'] is not None and post['input_tp']<=-3 and error<=1e-6 else 'blocked',
            'path':str(dest),'readback_loudness':post,'original_loudness':loud,'applied_gain_db':gain,
            'method':'constant_gain_with_true_peak_cap','target_integrated_lufs':-23,'true_peak_ceiling_dbfs':-3,
            'common_lufs_target_achieved':post['input_i'] is not None and abs(post['input_i']+23)<=.1,
            'constant_gain_max_error':error,'sample_count_preserved':len(x),'sha256':sha(dest)}
        if args.model_scores:
            scored=external_scores[row['sound_asset_id']]
            if scored['sha256']!=row['source_pcm_sha256']:raise ValueError('CPU model receipt is not bound to this source audio')
            scores=scored['scores_by_mid']; top8=scored['top8']
        else:
            if model is None:model,labels=load_beats(args.model_source,args.weights)
            import torch
            chunks=[]
            for offset in range(0,len(x),80000):
                chunk=x[offset:offset+80000].astype(np.float32)
                if len(chunk)<16000:chunk=np.pad(chunk,(0,16000-len(chunk)))
                with torch.inference_mode():chunks.append(model.extract_features(torch.from_numpy(chunk)[None])[0][0].numpy())
            probabilities=np.mean(chunks,axis=0)
            scores={labels[k]:float(v) for k,v in enumerate(probabilities)}
            top8=[{'mid':labels[int(k)],'score':float(probabilities[k])} for k in np.argsort(probabilities)[-8:][::-1]]
        confidence=max(scores.get(m,0.) for m in mapping['speech_playback'])
        competitors=[max([scores.get(m,0.) for m in mids],default=0.) for cls,mids in mapping.items()
            if cls!='speech_playback' and mids and not set(mids).intersection(mapping['speech_playback'])]
        observed_margin=confidence-max(competitors,default=0.)
        screens['automatic_label']={'status':'pass' if confidence>=threshold and observed_margin>=margin else 'blocked',
            'confidence':confidence,'threshold':threshold,'label_margin':observed_margin,'minimum_margin':margin,
            'calibration_source':str(args.label_calibration),'checkpoint':str(args.weights),
            'audio_sha256_matches_model_receipt':True,'sha256':row['source_pcm_sha256'],
            'top8':top8,
            'speaker_gender_source':'original VCTK speaker-info.txt; Speech classifier does not infer gender'}
        row.update(screens=screens,normalized_path=str(dest),duration_s=len(x)/rate,
            formal_status='blocked_pending_reference_scan',strict_v7_eligible=False)
        receipt_path.parent.mkdir(exist_ok=True);write_json(receipt_path,row);records.append(row)
        if i%20==0:print('VCTK_TECHNICAL',i+1,len(inventory),round(time.perf_counter()-start,2),flush=True)
    dedup=json.loads(args.deduplication.read_text()) if args.deduplication else {}
    admitted=[];rejected=[]
    for row in records:
        proof=dedup.get('records',{}).get(row['sound_asset_id'],{})
        row['screens']['deduplication']={'status':proof.get('status','unverified') if dedup.get('complete_for_listed_roots') else 'unverified',
            'coverage_complete_for_listed_roots':dedup.get('complete_for_listed_roots',False),
            'reference_file_count':dedup.get('reference_file_count'),'reference_roots':dedup.get('reference_roots'),
            'nearest':proof.get('nearest'),'source':str(args.deduplication) if args.deduplication else None}
        blockers=[name+':'+s['status'] for name,s in row['screens'].items() if s.get('status')!='pass']
        row.update(formal_status='admitted' if not blockers else 'blocked',strict_v7_eligible=not blockers,formal_blockers=blockers)
        (rejected if blockers else admitted).append(row)
    summary={'schema':'v7-vctk-automatic-admission-v1','records':records,'record_count':len(records),
        'speaker_count':len({r['speaker_id'] for r in records}),'admitted_count':len(admitted),'blocked_count':len(rejected),
        'source_kind':'public recorded dry mono','license':'CC-BY-4.0','formal_qa_windows_count':0,
        'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start,
        'gpu_jobs':0,'old_thresholds_preserved':True,
        'model_checkpoint_sha256':sha(args.weights),'reference_scan_scope_limit':dedup.get('scope_limit')}
    name='VCTK_ADMISSION.json' if args.deduplication else 'VCTK_TECHNICAL_PENDING_DEDUP.json'
    write_json(out/name,summary)
    with (out/('VCTK_ADMISSION.csv' if args.deduplication else 'VCTK_TECHNICAL.csv')).open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['sound_asset_id','original_id','speaker_id','gender','duration_s','integrated_lufs','true_peak_dbfs','speech_confidence','formal_status','blockers'])
        for r in records:writer.writerow([r['sound_asset_id'],r['original_id'],r['speaker_id'],r['gender'],r['duration_s'],r['screens']['normalization']['readback_loudness']['input_i'],r['screens']['normalization']['readback_loudness']['input_tp'],r['screens']['automatic_label']['confidence'],r['formal_status'],';'.join(r['formal_blockers'])])
    print('DONE',name,len(records),len(admitted),summary['wall_seconds'],flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
