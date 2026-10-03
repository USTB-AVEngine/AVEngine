#!/usr/bin/env python3
"""Stage lossless witnesses, measure three artifact classes, or prune passed parents.

Pruning only targets explicitly enumerated regular files inside certified
production parents. Original score/answers/certificates/commitments, public
media, rejected parents, datasets, other checkouts and symlinks are preserved.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from avengine.qa.frozen_panel_retention import encode_file, read, sha, write, configuration_identity


def disk(paths, excluded=None):
    seen = set(excluded or [])
    size = allocated = 0
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            continue
        info = path.stat()
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        size += info.st_size
        allocated += info.st_blocks * 512
    return {'logical_bytes': size, 'allocated_bytes': allocated}, seen


def freeze(root):
    retention = root / 'retention'
    if (retention / 'COMMITMENTS.sha256').exists():
        raise ValueError('retention is already frozen')
    files = sorted(p for p in retention.rglob('*') if p.is_file() and 'temporary' not in p.parts)
    with (retention / 'COMMITMENTS.sha256').open('x') as f:
        for path in files:
            f.write(f'{sha(path)}  {path.relative_to(retention)}\n')
    for path in files + [retention / 'COMMITMENTS.sha256']:
        path.chmod(0o400)
    return {'status': 'pass', 'action': 'freeze', 'groups': len(read(retention / 'manifest.json')['groups'])}


def stage(root, certificate=None):
    retention = root / 'retention'
    if retention.exists() and (certificate is None or (retention / 'COMMITMENTS.sha256').exists()):
        raise ValueError('retention already exists; inspect instead of overwriting')
    retention.mkdir(mode=0o700, exist_ok=True)
    config = read(root / 'inputs/config.json') if (root / 'inputs/config.json').exists() else read(root / 'inputs/shard_config.json')
    certificates = [certificate] if certificate is not None else sorted((root / 'scoring_private/certificates').glob('*.json'))
    if not certificates:
        raise ValueError('no certified parents')
    previous_index = retention / 'manifest.json'
    groups = read(previous_index)['groups'] if previous_index.exists() else []
    shared = {}
    identity_path = root / 'scoring_private/build_code_identity.json'
    sealed_inputs = read(identity_path).get('explicit_inputs_sha256', {}) if identity_path.exists() else {}
    for cert_path in certificates:
        cert = read(cert_path)
        group = cert['group']
        production = (root / 'production' / group).resolve()
        if Path(cert['provenance']['recipe']).parent.resolve() != production:
            raise ValueError('native parent scope does not match certificate')
        public_map = {}
        videos = {}
        for member in cert['media']['members']:
            vi, ai = member['video_index'], member['audio_index']
            request = member['request']
            videos[vi] = request['video']
            for name in ('foa', 'binaural', 'mono'):
                public_map[production / 'audio' / f'a{ai}_{name}.wav'] = request[name]
            public_map[production / 'visual' / f'v{vi}' / 'video.mp4'] = request['video']
        records = []
        ledger = []
        for path in sorted(production.rglob('*')):
            if path.is_symlink() or not path.is_file():
                continue
            if not path.resolve().is_relative_to(production):
                raise ValueError('production file escaped certified parent')
            info = path.stat()
            base = {'source': str(path), 'sha256': sha(path), 'bytes': info.st_size,
                    'allocated_bytes': info.st_blocks * 512, 'device': info.st_dev, 'inode': info.st_ino}
            ledger.append(base)
            if path in public_map:
                public = root / 'public' / public_map[path]
                if sha(public) != base['sha256']:
                    raise ValueError('private/public media bytes differ')
                records.append({**base, 'kind': 'public_media', 'public_media': public_map[path]})
            elif path.suffix in {'.json', '.npy', '.npz'}:
                vi = next((int(part[1:]) for part in path.parts if part in {'v0', 'v1'}), None)
                rgb = videos[vi] if path.name == 'rgb.npy' and vi is not None else None
                records.append(encode_file(retention, path, public_video=rgb))
        dependencies = [config['registry'], cert['construction']['audio_receipt']['hrtf']['path']]
        dependencies += [d['path'] for d in cert['construction']['audio_receipt']['dry_audio']]
        house = next(h for h in config['houses'] if h['house'] == cert['house'])
        dependencies.append(house['acceptance_result'])
        dep_records = []
        for reference in dependencies:
            if reference not in shared:
                record = encode_file(retention, Path(reference))
                if reference in sealed_inputs and record['sha256'] != sealed_inputs[reference]:
                    raise ValueError('external dependency differs from the original sealed input identity')
                record['source_resolved'] = str(Path(reference).resolve())
                shared[reference] = record
            dep_records.append(shared[reference])
        manifest = {'schema': 'frozen_panel_lossless_retention_v1', 'group': group,
                    'configuration_normalized_sha256': configuration_identity(config),
                    'certificate_sha256': sha(cert_path), 'production_root': str(production),
                    'native_files': records, 'dependencies': dep_records, 'original_file_ledger': ledger,
                    'native_verifier_sha256': sha(REPO / 'tools/qa/verify_frozen_panel.py'),
                    'scorer_sha256': sha(REPO / 'tools/qa/score_frozen_panel.py'),
                    'original_commitments_sha256': sha(root / 'public/COMMITMENTS.sha256') if (root / 'public/COMMITMENTS.sha256').exists() else None,
                    'threshold_changes': False, 'truth_changes': False}
        manifest_path = retention / 'groups' / (group + '.json')
        write(manifest_path, manifest)
        proof_files = {manifest_path, *(retention / x['payload'] for x in records + dep_records if 'payload' in x)}
        with (retention / 'groups' / (group + '.sha256')).open('x') as f:
            for p in sorted(proof_files):
                f.write(f'{sha(p)}  {p.relative_to(retention)}\n')
        groups.append(str(manifest_path.relative_to(retention)))
        print(json.dumps({'group': group, 'witness': 'staged', 'original_files': len(ledger)}, ensure_ascii=False), flush=True)
    result_index = {'schema': 'frozen_panel_lossless_retention_v1', 'groups': groups,
          'source_root': str(root), 'original_score_and_truth_unchanged': True,
          'requested_indices': read(root / 'run_state.json')['indices'] if (root / 'run_state.json').exists() else None,
          'classification': {'release': 'public media/question', 'sealed': 'scoring_private plus lossless minimal witnesses/commitments',
                             'temporary': 'native arrays/stems/execution duplicates, erased only after unchanged native verification'}}
    if previous_index.exists():
        previous_index.write_text(json.dumps(result_index, ensure_ascii=False, indent=2)+'\n')
    else:
        write(previous_index, result_index)
    if certificate is None:
        freeze(root)
    return {'status': 'pass', 'groups': len(groups), 'action': 'stage', 'deleted_files': 0}


def prune(root, verification, certificate=None):
    checked = read(verification)
    index = read(root / 'retention/manifest.json')
    if certificate is not None:
        index['groups'] = [p for p in index['groups'] if Path(p).stem == certificate.stem]
    if (checked['status'] != 'pass' or checked.get('original_native_inputs_read') is not False or
            checked.get('external_dependency_reads_resolved_to_retained_files') is not True):
        raise ValueError('prune requires a passing retained-only unchanged native verification')
    if len(checked['groups']) != len(index['groups']) or not all(x['three_layers_verified'] for x in checked['groups']):
        raise ValueError('prune verification does not cover every retained parent')
    passed = {x['group']: x['certificate_sha256'] for x in checked['groups']}
    removed = []
    for relative in index['groups']:
        manifest = read(root / 'retention' / relative)
        group_root = Path(manifest['production_root'])
        if passed.get(manifest['group']) != manifest['certificate_sha256']:
            raise ValueError('prune certificate verification identity differs')
        if group_root != (root / 'production' / manifest['group']).resolve():
            raise ValueError('prune scope differs from certified production parent')
        for row in manifest['original_file_ledger']:
            path = Path(row['source'])
            if not path.is_relative_to(group_root) or path.is_symlink():
                raise ValueError('unsafe prune file')
            if not path.exists():
                continue
            info = path.stat()
            if (info.st_dev, info.st_ino, info.st_size) != (row['device'], row['inode'], row['bytes']) or sha(path) != row['sha256']:
                raise ValueError('native intermediate changed after audit; stop pruning')
            path.unlink()
            removed.append({'path': str(path), 'sha256': row['sha256'], 'bytes': row['bytes']})
        print(json.dumps({'group': manifest['group'], 'intermediates_pruned': True}), flush=True)
    # Answers are only opened by the unchanged scoring script.
    audit = {'status': 'not_run_not_sealed'}
    if (root / 'public/COMMITMENTS.sha256').exists():
        audit = json.loads(subprocess.check_output([sys.executable, str(REPO / 'tools/qa/score_frozen_panel.py'), 'audit', '--root', str(root)], text=True))
    if audit['status'] not in {'pass', 'not_run_not_sealed'}:
        raise ValueError('original seal changed after prune')
    result = {'status': 'pass', 'action': 'prune', 'removed_files': removed,
              'original_seal_audit': audit, 'truth_changes': False, 'threshold_changes': False,
              'utc': datetime.now(timezone.utc).isoformat()}
    receipt = root / 'retention/prune_receipt.json' if certificate is None else root / 'retention/prune_receipts' / (certificate.stem + '.json')
    write(receipt, result)
    return {'status': 'pass', 'action': 'prune', 'groups': len(index['groups']), 'removed_files': len(removed)}


def measure(root):
    index = read(root / 'retention/manifest.json')
    release, release_inodes = disk(p for p in (root / 'public').rglob('*') if p.name != 'COMMITMENTS.sha256')
    config_path = root / 'inputs/config.json' if (root / 'inputs/config.json').exists() else root / 'inputs/shard_config.json'
    sealed_paths = [config_path] + list((root / 'scoring_private').rglob('*')) + list((root / 'retention').rglob('*')) + [root / 'public/COMMITMENTS.sha256']
    sealed, sealed_inodes = disk(sealed_paths, release_inodes)
    original = []
    third = set()
    removed_size = removed_alloc = 0
    for relative in index['groups']:
        manifest = read(root / 'retention' / relative)
        for row in manifest['original_file_ledger']:
            key = (row['device'], row['inode'])
            if key not in release_inodes | third:
                third.add(key)
                removed_size += row['bytes']
                removed_alloc += row['allocated_bytes']
    existing_paths = [p for relative in index['groups'] for p in Path(read(root / 'retention' / relative)['production_root']).rglob('*')]
    remaining, _ = disk(existing_paths, sealed_inodes)
    groups = len(index['groups'])
    return {'status': 'pass', 'groups': groups, 'release': release, 'sealed_with_minimal_lossless_witnesses': sealed,
            'temporary_before': {'logical_bytes': removed_size, 'allocated_bytes': removed_alloc},
            'temporary_remaining': remaining,
            'per_group_allocated_bytes': {'release': release['allocated_bytes']/groups, 'sealed': sealed['allocated_bytes']/groups,
                                          'temporary_before': removed_alloc/groups, 'temporary_remaining': remaining['allocated_bytes']/groups},
            'projection_720_allocated_bytes': {'release': release['allocated_bytes']/groups*720,
                'sealed': sealed['allocated_bytes']/groups*720,
                'total_retained': (release['allocated_bytes']+sealed['allocated_bytes'])/groups*720},
            'source': str(root), 'hardlinks_deduplicated': True, 'external_symlinks_followed': False,
            'pending_new_materials_size': 'not_measured'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['stage', 'prune', 'measure', 'freeze'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--certificate', type=Path)
    parser.add_argument('--commit-prune', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'stage':
        result = stage(root, args.certificate)
    elif args.action == 'freeze':
        result = freeze(root)
    elif args.action == 'measure':
        result = measure(root)
    else:
        if not args.commit_prune or args.verification is None:
            raise ValueError('prune needs the retained-only audit and explicit --commit-prune')
        result = prune(root, args.verification, args.certificate)
    write(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
