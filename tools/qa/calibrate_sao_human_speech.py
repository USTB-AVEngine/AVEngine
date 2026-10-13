"""Calibrate a BEATs human speech/singing screen against retained real v9 audio."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, pathlib, sys, time, types
import numpy as np
import soundfile as sf

HUMAN_ROOTS = ('/m/09x0r', '/m/015lz1', '/m/02rtxlg', '/m/07qfr4h')

def human_heads(ontology):
    by_id = {r['id']: r for r in ontology}
    selected = set()
    def visit(mid):
        if mid in selected:
            return
        selected.add(mid)
        for child in by_id[mid].get('child_ids', []):
            visit(child)
    for root in HUMAN_ROOTS:
        visit(root)
    return [{'mid': mid, 'name': by_id[mid]['name']} for mid in sorted(selected)]

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_beats(source, weight):
    import torch
    pkg = types.ModuleType('_sao_human_beats')
    pkg.__path__ = [str(source)]
    sys.modules[pkg.__name__] = pkg
    spec = importlib.util.spec_from_file_location(pkg.__name__ + '.BEATs', source / 'BEATs.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
    model = mod.BEATs(mod.BEATsConfig(checkpoint['cfg']))
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().requires_grad_(False).to('cuda'), {int(k): v for k, v in checkpoint['label_dict'].items()}

def beats_scores(model, labels, wave):
    import torch
    chunks = []
    for lo in range(0, len(wave), 80000):
        x = wave[lo:lo + 80000]
        if len(x) < 16000:
            x = np.pad(x, (0, 16000 - len(x)))
        with torch.inference_mode():
            chunks.append(model.extract_features(torch.from_numpy(x)[None].to('cuda'))[0][0].cpu().numpy())
    scores = np.mean(chunks, axis=0)
    return {labels[k]: float(value) for k, value in enumerate(scores)}

def distribution(values):
    x = np.asarray(values)
    return {'count': len(x), 'min': float(x.min()), 'median': float(np.median(x)),
            'p90': float(np.quantile(x, .9)), 'p95': float(np.quantile(x, .95)),
            'p99': float(np.quantile(x, .99)), 'max': float(x.max())}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'v9', 'config', 'ontology', 'beats-source', 'beats-weight'):
        p.add_argument('--' + name, type=pathlib.Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    start = time.time()
    classes = set(json.loads(a.config.read_text())['classes'])
    v9 = json.loads(a.v9.read_text())['records']
    selected = [r for r in v9 if r['sound_class'] in classes or r['sound_class'] == 'speech_playback']
    heads = human_heads(json.loads(a.ontology.read_text()))
    model, labels = load_beats(a.beats_source, a.beats_weight)
    assert set(h['mid'] for h in heads) <= set(labels.values())
    target = a.output / 'human_speech_calibration_scores_v11.jsonl'
    rows = [json.loads(s) for s in target.read_text().splitlines()] if target.exists() else []
    done = {r['sound_asset_id'] for r in rows}
    assert len(done) == len(rows)
    with target.open('a') as stream:
        for r in selected:
            if r['sound_asset_id'] in done:
                continue
            if time.time() + 15 >= a.deadline:
                raise RuntimeError('calibration deadline reached before full sample coverage')
            path = r.get('normalized_path') or r.get('normalization', {}).get('path') or r['path']
            wave, rate = sf.read(path, dtype='float32', always_2d=True)
            assert rate == 16000 and wave.shape[1] == 1 and np.isfinite(wave).all()
            scores = beats_scores(model, labels, wave[:, 0])
            chosen = max(heads, key=lambda h: scores[h['mid']])
            row = dict(sound_asset_id=r['sound_asset_id'], sound_class=r['sound_class'], path=path,
                       audio_sha256=sha(path), polarity='negative_human_speech' if r['sound_class'] == 'speech_playback' else 'positive_real_target',
                       source_dataset=r.get('source_dataset'), source_original_id=r.get('source_original_id'),
                       score=scores[chosen['mid']], maximum_head=chosen,
                       human_scores_by_mid={h['mid']: scores[h['mid']] for h in heads},
                       input_rate_hz=rate, sample_count=len(wave))
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            rows.append(row)
            if len(rows) % 24 == 0:
                print('CALIBRATION', len(rows), len(selected), flush=True)
    positive = [r['score'] for r in rows if r['polarity'] == 'positive_real_target']
    negative = [r['score'] for r in rows if r['polarity'] == 'negative_human_speech']
    assert len(positive) == sum(r['sound_class'] in classes for r in v9)
    assert len(negative) >= 100
    margin = .01
    candidate_threshold = max(positive) + margin
    positive_pass = sum(s <= candidate_threshold for s in positive)
    negative_block = sum(s > candidate_threshold for s in negative)
    valid = positive_pass == len(positive) and negative_block / len(negative) >= .95 and candidate_threshold < 1
    receipt = dict(status='pass' if valid else 'calibration_failed_no_threshold_authorized',
                   threshold=candidate_threshold if valid else None, proposed_threshold=candidate_threshold,
                   rule='reject if maximum selected BEATs posterior > threshold; threshold = maximum real positive + 0.01',
                   positive_margin=margin, positive_count=len(positive), positive_pass=positive_pass,
                   positive_pass_rate=positive_pass / len(positive), negative_count=len(negative),
                   negative_block=negative_block, negative_block_rate=negative_block / len(negative),
                   positive_distribution=distribution(positive), negative_distribution=distribution(negative),
                   maximum_possible_block_rate_at_positive_max=sum(s > max(positive) for s in negative) / len(negative),
                   heads=heads, checkpoint=str(a.beats_weight), checkpoint_sha256=sha(a.beats_weight),
                   v9_path=str(a.v9), v9_sha256=sha(a.v9), scores_path=str(target), scores_sha256=sha(target),
                   expected_selected_records=len(selected), complete=len(rows) == len(selected),
                   source_files={n: sha(a.beats_source / n) for n in ('BEATs.py', 'backbone.py', 'modules.py')},
                   sampling='all v9 real recordings in the 13 target classes; all v9 speech_playback records',
                   precision='float32', pooling='same thin top-up: mean 5-second chunk posterior, last chunk padded to >=1 second',
                   whisper_role='recording_only_not_an_admission_gate', seconds=time.time() - start,
                   gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(), human_listening_used=False)
    (a.output / 'HUMAN_SPEECH_CALIBRATION_v11.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    with (a.output / 'PROGRESS.md').open('a') as f:
        f.write('- ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' v11 人说话校准完成：' + receipt['status'] + f'；正例 {positive_pass}/{len(positive)}，反例拦截 {negative_block}/{len(negative)}。\n')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
