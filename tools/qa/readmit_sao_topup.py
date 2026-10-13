"""Repair SAO fingerprint checks and audit every retained candidate under v11 policy."""
from __future__ import annotations
import argparse, collections, json, pathlib, pickle, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
from scipy import signal
from avengine.qa.v7_sao_fingerprint import waveform_fingerprint, robust_alignment, possible_references
from tools.qa.admit_sao_topup import load_scored, lines, sha, dump, progress, norm_path, build


def controls(a):
    start = time.time()
    rows = load_scored(a.output)
    def inspect(row):
        r = row['candidate']
        x, sr = sf.read(r['path'], dtype='float32')
        assert sr == 16000 and x.ndim == 1 and sha(r['path']) == r['sha256']
        original = waveform_fingerprint(x, sr)
        originals = {h for h, _ in original['landmarks']}
        variants = {'gain': x * .137,
                    'resample_roundtrip': signal.resample_poly(signal.resample_poly(x, 3, 2), 2, 3)[:len(x)],
                    'crop': x[len(x) // 5:4 * len(x) // 5]}
        result = []
        for name, wave in variants.items():
            query = waveform_fingerprint(wave, sr)
            hashes = {h for h, _ in query['landmarks']}
            common = len(hashes & originals)
            retrieved = 'original' in possible_references({'original': common}, len(hashes), {'original': len(originals)})
            aligned = robust_alignment(query, original)
            result.append(dict(sound_asset_id=r['sound_asset_id'], transform=name, detected=bool(retrieved and aligned['matched']),
                               reference_retrieved=retrieved, common_hashes=common,
                               query_unique_hashes=len(hashes), original_unique_hashes=len(originals),
                               crop_grid_phase_8k_samples=(len(x) // 5 // 2) % 128 if name == 'crop' else None,
                               alignment=aligned))
        return result
    measured = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, result in enumerate(pool.map(inspect, rows)):
            measured.extend(result)
            if (index + 1) % 48 == 0:
                print('CONTROLS', index + 1, len(rows), flush=True)
    detected = sum(r['detected'] for r in measured)
    old = json.loads((a.output / 'deduplication_final.json').read_text())
    report = dict(status='pass' if detected / len(measured) >= .99 else 'fail',
                  before_count=old['positive_control_count'], before_detected=old['positive_control_detected'],
                  after_count=len(measured), after_detected=detected, after_detection_rate=detected / len(measured),
                  controls=measured, failures=[r for r in measured if not r['detected']],
                  by_transform={name: dict(count=sum(r['transform'] == name for r in measured),
                                          detected=sum(r['transform'] == name and r['detected'] for r in measured))
                                for name in ('gain', 'resample_roundtrip', 'crop')},
                  positive_controls_are_quality_validation_only=True, no_per_candidate_control_failure_rejection=True,
                  seconds=time.time() - start, waveform_threshold=.995, minimum_aligned_seconds=1.,
                  diagnosis='crop changes the STFT frame grid; sparse ticks have fewer than 12 unique landmark hashes; repair confirms full shorter waveform using gain-invariant FFT normalized correlation and weak hash retrieval',
                  human_listening_used=False)
    dump(a.output / f'FINGERPRINT_REPAIR_VALIDATION_{a.stage}.json', report)
    progress(a.output, f'{a.stage} 指纹修复对照：{detected}/{len(measured)} 检出；对照失败不作为音频拒绝理由。')
    print('CONTROLS_DONE', report['status'], detected, len(measured), report['seconds'], flush=True)


def dedup(a):
    start = time.time()
    o = a.output
    validation = json.loads((o / f'FINGERPRINT_REPAIR_VALIDATION_{a.stage}.json').read_text())
    assert validation['status'] == 'pass' and validation['after_detection_rate'] >= .99
    rows = load_scored(o)
    v9 = json.loads(a.v9.read_text())['records']
    references = {f"v9:{r['sound_asset_id']}": dict(sound_asset_id=r['sound_asset_id'], path=norm_path(r)) for r in v9}
    cache = o / 'cache' / 'v9_waveform_fingerprints_v11_polyphase.pkl'
    if cache.exists():
        retained = pickle.loads(cache.read_bytes())
        assert retained['v9_sha256'] == sha(a.v9)
        fingerprints = retained['fingerprints']
    else:
        def read_reference(item):
            key, reference = item
            x, rate = sf.read(reference['path'], dtype='float32', always_2d=True)
            assert rate == 16000 and x.shape[1] == 1
            return key, waveform_fingerprint(x[:, 0], rate)
        fingerprints = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for index, (key, value) in enumerate(pool.map(read_reference, references.items())):
                fingerprints[key] = value
                if index % 240 == 0:
                    print('REFERENCE_FINGERPRINT', index, len(references), flush=True)
        cache.write_bytes(pickle.dumps(dict(v9_sha256=sha(a.v9), fingerprints=fingerprints), protocol=4))
    inverted = collections.defaultdict(list)
    pcm = collections.defaultdict(list)
    hash_counts = {}
    def register(key, fingerprint):
        fingerprints[key] = fingerprint
        hashes = {h for h, _ in fingerprint['landmarks']}
        hash_counts[key] = len(hashes)
        for h in hashes:
            inverted[h].append(key)
        if fingerprint['pcm_sha256']:
            pcm[fingerprint['pcm_sha256']].append(key)
    for key, value in list(fingerprints.items()):
        register(key, value)
    ordered = sorted(rows, key=lambda r: (-int(r['beats_pass'] and r['activity_pass']), -r['target_confidence'], -r['margin'], r['candidate']['sound_asset_id']))
    results, matches = {}, []
    compared = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, row in enumerate(ordered):
            r = row['candidate']
            wave, rate = sf.read(r['path'], dtype='float32')
            assert rate == 16000 and wave.ndim == 1 and sha(r['path']) == r['sha256']
            query = waveform_fingerprint(wave, rate)
            hashes = {h for h, _ in query['landmarks']}
            counts = collections.Counter()
            for h in hashes:
                counts.update(inverted.get(h, ()))
            possible = possible_references(counts, len(hashes), hash_counts)
            possible.update(pcm.get(query['pcm_sha256'], ()))
            def compare(key):
                aligned = robust_alignment(query, fingerprints[key])
                return dict(aligned, reference_key=key)
            hits = [r for r in pool.map(compare, sorted(possible)) if r['matched']]
            compared += len(possible)
            reason = ('audio_fingerprint_matches_existing_v9' if any(h['reference_key'].startswith('v9:') for h in hits)
                      else 'audio_fingerprint_matches_generated_candidate' if hits else None)
            results[r['sound_asset_id']] = dict(status='blocked' if hits else 'pass', reason=reason, matches=hits,
                                               audio_sha256=r['sha256'], compared_references=len(possible),
                                               reference_v9_records=len(v9), generated_previous_comparisons=index,
                                               coverage_complete_for_listed_roots=True,
                                               positive_controls_are_not_an_admission_gate=True)
            if hits:
                matches.append(dict(sound_asset_id=r['sound_asset_id'], matches=hits))
            else:
                register('generated:' + r['sound_asset_id'], query)
            if index % 24 == 0:
                print('DEDUP_V11', index, len(ordered), 'MATCHED', len(matches), 'PAIR_COMPARISONS', compared, flush=True)
    report = dict(reference_roots=[dict(manifest=str(a.v9), kind='all existing v9 dry inputs'),
                                  dict(root=str(o / 'mono_16000'), kind='all retained generated candidates')],
                  reference_file_count=len(v9), reference_waveform_count=len(v9), records=results, matches=matches,
                  errors=[], complete_for_listed_roots=True, pair_comparisons=compared, seconds=time.time() - start,
                  positive_control_validation=str(o / f'FINGERPRINT_REPAIR_VALIDATION_{a.stage}.json'),
                  positive_control_count=validation['after_count'], positive_control_detected=validation['after_detected'],
                  method='weak spectral-hash retrieval (>=4 shared hashes; sparse <12 hash references or queries exhaustively compared), then exact normalized PCM or full shorter 8k waveform normalized correlation >=0.995 over >=1 second, checking both 16k sample phases',
                  human_listening_used=False,
                  scope_limit='v9 and retained generated inputs only; approximate hash retrieval has the measured control recall; no claim of exhaustive waveform comparison for every pair or perceptual/human similarity')
    dump(o / f'deduplication_{a.stage}.json', report)
    progress(o, f'{a.stage} 全部 {len(rows)} 条候选修复去重完成：实际匹配 {len(matches)} 条，阳性对照 {validation["after_detected"]}/{validation["after_count"]}。')
    print('DEDUP_V11_DONE', len(rows), len(matches), compared, report['seconds'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('controls', 'dedup', 'build', 'pipeline'))
    for name in ('output', 'v9', 'config', 'mapping', 'ontology'):
        p.add_argument('--' + name, type=pathlib.Path, required=True)
    p.add_argument('--stage', default='v11_r12')
    a = p.parse_args()
    worktree = pathlib.Path(__file__).resolve().parents[2]
    source_paths = ['tools/qa/readmit_sao_topup.py', 'tools/qa/admit_sao_topup.py',
                    'src/avengine/qa/v7_sao_fingerprint.py', 'src/avengine/qa/v7_sao_admission.py',
                    'src/avengine/qa/v7_extended_admission.py', 'src/avengine/qa/v7_auto_admission.py']
    identity = dict(source_sha256={path: sha(worktree / path) for path in source_paths},
                    v9_sha256=sha(a.v9), config_sha256=sha(a.config), mapping_sha256=sha(a.mapping),
                    ontology_sha256=sha(a.ontology), control_threads=8, pair_comparison_threads=8)
    destination = a.output / 'evidence' / f'readmit_execution_source_{a.stage}.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == identity
    else:
        dump(destination, identity)
    if a.action == 'pipeline':
        controls(a)
        dedup(a)
        build(a)
    else:
        {'controls': controls, 'dedup': dedup, 'build': build}[a.action](a)

if __name__ == '__main__':
    main()
