#!/usr/bin/env python3
"""Re-run the unchanged full native verifier from retained files only, one group at a time.

No answer bytes, models, renderer, native RLR simulation or original intermediates
are read. Lossless original-byte reconstruction is checked before the original
verification functions run; every original threshold and full-media check stays.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from avengine.qa.frozen_panel_retention import hydrated_group, read, sha, write, RetainedSoundFile, require_configuration_binding


def verify(root, *, certificate=None):
    retention = root / 'retention'
    index = read(retention / 'manifest.json')
    commit_files = [retention / 'COMMITMENTS.sha256'] if (retention / 'COMMITMENTS.sha256').exists() else [
        retention / 'groups' / (Path(p).stem + '.sha256') for p in index['groups']
        if certificate is None or Path(p).stem == certificate.stem]
    for commit_file in commit_files:
        for line in commit_file.read_text().splitlines():
            digest, relative = line.split('  ', 1)
            target = (retention / relative).resolve()
            if not target.is_relative_to(retention) or sha(target) != digest:
                raise ValueError('lossless witness commitment changed')
    spec = importlib.util.spec_from_file_location('retained_native_verifier', REPO / 'tools/qa/verify_frozen_panel.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    config = read(root / 'inputs/config.json') if (root / 'inputs/config.json').exists() else read(root / 'inputs/shard_config.json')
    results = []
    for relative in index['groups']:
        manifest = read(retention / relative)
        if certificate is not None and manifest['group'] != certificate.stem:
            continue
        if sha(REPO / 'tools/qa/verify_frozen_panel.py') != manifest['native_verifier_sha256']:
            raise ValueError('native verifier changed; no threshold substitution allowed')
        if sha(REPO / 'tools/qa/score_frozen_panel.py') != manifest['scorer_sha256']:
            raise ValueError('original scorer changed')
        cert_path = root / 'scoring_private/certificates' / (manifest['group'] + '.json')
        if sha(cert_path) != manifest['certificate_sha256']:
            raise ValueError('original sealed certificate changed')
        require_configuration_binding(root, config, manifest)
        begin = time.monotonic()
        try:
            with hydrated_group(root, manifest) as resolver:
                original_path = native.Path
                original_soundfile = native.sf
                retained_soundfile = RetainedSoundFile(original_soundfile, resolver)
                native.sf = retained_soundfile
                native.Path = resolver  # Explicit lossless input resolution; no verifier logic edits.
                try:
                    row = native.verify_one(root, cert_path)
                    row['direct_audio_reads_resolved_to_retained_files'] = True
                    row['retained_audio_read_count'] = len(retained_soundfile.reads)
                finally:
                    native.Path = original_path
                    native.sf = original_soundfile
            row['including_lossless_restore_wall_seconds'] = time.monotonic() - begin
        except Exception as exc:
            row = {'group': manifest['group'], 'status': 'fail', 'reason': type(exc).__name__ + ': ' + str(exc)}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if not results:
        raise ValueError('no retained certificates verified')
    passed = [row for row in results if row['status'] == 'pass']
    distinct = all(len({row[key] for row in passed}) == len(passed)
                   for key in ['physical_parent_sha256', 'video_pair_sha256'])
    config = read(root / 'inputs/config.json') if (root / 'inputs/config.json').exists() else read(root / 'inputs/shard_config.json')
    if certificate is None:
        expected = len(config['houses']) * config['groups_per_house']
        if config['split'] == 'test':
            if index.get('requested_indices') is not None:
                expected = len(index['requested_indices'])
            elif (root / 'run_state.json').exists():
                expected = len(read(root / 'run_state.json')['indices'])
        if len(results) != expected:
            raise ValueError('retained verification does not cover the requested panel')
    if any(row['physical_parent_sha256'] in config.get('excluded_physical_parent_sha256', []) or
           row['video_pair_sha256'] in config.get('excluded_video_pair_sha256', []) for row in passed):
        raise ValueError('formal panel reused pilot geometry/media')
    seal_audit = {'status': 'not_run_not_sealed'}
    if (root / 'public/COMMITMENTS.sha256').exists():
        seal_audit = json.loads(subprocess.check_output([sys.executable, str(REPO / 'tools/qa/score_frozen_panel.py'),
                                                        'audit', '--root', str(root)], text=True))
    return {'status': 'pass' if len(passed) == len(results) and distinct and seal_audit['status'] in {'pass', 'not_run_not_sealed'} else 'fail',
            'groups': results, 'distinct_physical_parents': len({r['physical_parent_sha256'] for r in passed}),
            'distinct_video_pairs': len({r['video_pair_sha256'] for r in passed}),
            'original_native_inputs_read': False, 'external_dependency_reads_resolved_to_retained_files': True, 'retained_configuration_commitment_verified': True, 'three_layers_using_original_verifier': True,
            'scoring_script_audit': seal_audit, 'answers_opened': False,
            'threshold_changes': False, 'truth_changes': False, 'models_run': False}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--certificate', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('retained verifier is CPU-only')
    result = verify(a.root.resolve(), certificate=a.certificate)
    write(a.output, result)
    if result['status'] != 'pass':
        sys.exit(1)


if __name__ == '__main__':
    main()
