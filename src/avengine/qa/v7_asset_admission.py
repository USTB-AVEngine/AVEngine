"""Read-only inventory and technical qualification of v7 dry/human candidates.

Recorded mono inputs are allowed; real spatial/evaluation recordings are excluded. An
inventory or an automatic loudness measurement never promotes a historical
research asset to formal publication. Outputs live only in a fresh task root.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
import soundfile as sf
from scipy import signal

from avengine.timeline.continuous_scene import QA21_CLASSES

PERMISSIVE = {
    'http://creativecommons.org/publicdomain/zero/1.0/',
    'http://creativecommons.org/licenses/by/3.0/',
    'https://creativecommons.org/publicdomain/zero/1.0/',
    'https://creativecommons.org/licenses/by/3.0/',
}


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _loudness(path):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-threads', '1',
                             '-i', str(path), '-af', 'loudnorm=I=-23:TP=-3:LRA=7:print_format=json',
                             '-f', 'null', '-'], capture_output=True, text=True, check=True)
    match = re.search(r'\{\s*"input_i".*?\}', result.stderr, re.S)
    if not match:
        raise ValueError('FFmpeg did not produce a loudness readback')
    return {key: _finite(value) for key, value in json.loads(match.group()).items()}


def _official_metadata(path):
    records = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if member.endswith(('dev_clips_info_FSD50K.json', 'eval_clips_info_FSD50K.json')):
                values = json.loads(archive.read(member))
                for key, value in values.items():
                    records[str(key)] = dict(value, metadata_member=member)
    return records


def audit_sounds(pool_path, official_zip, mapping_path, output, *, workers=2, normalize_research=False):
    pool = _json(pool_path)
    mapping = {r['sound_class']: r for r in _json(mapping_path)['v6_classes']}
    official = _official_metadata(official_zip)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def inspect(record):
        cls = record['sound_class']
        source = record.get('source_origin') or record.get('source_pcm_path')
        row = {'sound_asset_id': record['sound_asset_id'], 'sound_class': cls,
               'path': record['path'], 'source_origin': source,
               'qa21_domain_member': cls in QA21_CLASSES,
               'starss_mapping': mapping[cls], 'label_provenance': 'real_recording_metadata',
               'strict_v7_eligible': False, 'formal_status': 'blocked',
               'formal_blockers': ['automatic_admission_receipts_missing'],
               'human_review_required': False,
               'segmentation': record.get('event_segmentation'),
               'crop': {k: v for k, v in record.items() if k.startswith('source_crop') or k in ('source_offset_s',)},
               'activity_intervals_samples': record.get('source_activity_intervals_samples'),
               'existing_normalization': record.get('source_normalization')}
        match = re.search(r'fsd50k_(\d+)', str(source))
        if match:
            fsd_id = match.group(1)
            meta = official.get(fsd_id)
            row['fsd50k_id'] = fsd_id
            row['official_metadata'] = meta
            if meta:
                row['license'] = meta.get('license')
                row['source_uri'] = f'https://freesound.org/s/{fsd_id}/'
                row['license_candidate_eligible'] = meta.get('license') in PERMISSIVE
                row['attribution'] = {k: meta.get(k) for k in ('title', 'username', 'uploader', 'license')}
            else:
                row['license_candidate_eligible'] = False
                row['formal_blockers'].append('official_clip_metadata_missing')
        elif 'vctk_' in str(source):
            row.update(license='CC-BY-4.0', license_candidate_eligible=False,
                       license_status='official_release_only_crop_chain_pending',
                       source_uri='https://datashare.ed.ac.uk/handle/10283/3443')
        else:
            row.update(license=None, license_candidate_eligible=False)
        if cls == 'any_audioset_class_playback':
            row['formal_blockers'].append('composite_requires_underlying_content_and_loudspeaker_mechanism')
        path = Path(record['path'])
        try:
            audio, rate = sf.read(path, dtype='float64', always_2d=True)
            row['readback'] = {'sample_rate_hz': rate, 'sample_count': len(audio), 'channels': audio.shape[1],
                               'duration_s': len(audio) / rate, 'finite': bool(np.isfinite(audio).all()),
                               'peak_dbfs': 20 * math.log10(max(float(np.abs(audio).max()), 1e-30)),
                               'rms_dbfs': 20 * math.log10(max(float(np.sqrt(np.mean(audio ** 2))), 1e-30)),
                               'silence_fraction_exact_zero': float(np.mean(audio == 0))}
            row['loudness'] = _loudness(path)
            row['technical_status'] = ('pass' if rate == 16000 and audio.shape[1] == 1 and len(audio) > 0 and
                                        np.isfinite(audio).all() and np.abs(audio).max() < 1 else 'fail')
            row['loudness_normalized_to_common_target'] = False
            if normalize_research and row['technical_status'] == 'pass' and row.get('license_candidate_eligible') and row['loudness']['input_i'] is not None:
                # A constant gain cannot change content, duration or segmentation.
                peak = float(np.abs(signal.resample_poly(audio[:, 0], 4, 1)).max())
                lufs_gain = -23 - row['loudness']['input_i']
                peak_dbfs = 20 * math.log10(max(peak, 1e-30))
                if row['loudness']['input_tp'] is not None:
                    peak_dbfs = max(peak_dbfs, row['loudness']['input_tp'])
                peak_gain = -3.02 - peak_dbfs
                gain_db = min(lufs_gain, peak_gain)
                dest = output / 'normalized_research' / cls / (record['sound_asset_id'] + '.wav')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise ValueError('normalized derivative already exists')
                sf.write(dest, audio * 10 ** (gain_db / 20), rate, subtype='FLOAT')
                row['normalized_research_derivative'] = {
                    'path': str(dest), 'applied_gain_db': gain_db,
                    'target_integrated_lufs': -23, 'true_peak_ceiling_dbfs': -3,
                    'peak_ceiling_limited': peak_gain < lufs_gain,
                    'source_sample_count_preserved': len(audio), 'source_sample_rate_preserved': rate,
                    'readback_loudness': _loudness(dest), 'strict_v7_eligible': False}
                derivative = row['normalized_research_derivative']
                for attempt in range(3):
                    measured_tp = derivative['readback_loudness']['input_tp']
                    if measured_tp is None or measured_tp <= -3:
                        break
                    gain_db += -3.02 - measured_tp
                    sf.write(dest, audio * 10 ** (gain_db / 20), rate, subtype='FLOAT')
                    derivative.update(applied_gain_db=gain_db, peak_ceiling_limited=True, readback_loudness=_loudness(dest))
                derivative['true_peak_ceiling_verified'] = derivative['readback_loudness']['input_tp'] is not None and derivative['readback_loudness']['input_tp'] <= -3
                measured_i = derivative['readback_loudness']['input_i']
                row['loudness_normalized_to_common_target'] = measured_i is not None and abs(measured_i + 23) <= 0.1
            row['recommended_normalization'] = {'target_integrated_lufs': -23, 'true_peak_ceiling_dbfs': -3,
                                                'method': 'constant gain, preserve duration and source segmentation',
                                                'applied': 'normalized_research_derivative' in row}
        except Exception as error:
            row['technical_status'] = 'fail'
            row['error'] = str(error)
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(inspect, pool['sounds']))
    by_class = []
    for cls in mapping:
        rows = [r for r in records if r['sound_class'] == cls]
        durations = [r['readback']['duration_s'] for r in rows if 'readback' in r]
        loudness = [r['loudness']['input_i'] for r in rows if r.get('loudness', {}).get('input_i') is not None]
        by_class.append({'sound_class': cls, 'existing_records': len(rows),
                         'technical_pass': sum(r['technical_status'] == 'pass' for r in rows),
                         'permissive_license_candidates': sum(bool(r.get('license_candidate_eligible')) for r in rows),
                         'research_candidates_with_pcm_and_permissive_license': sum(r['technical_status'] == 'pass' and bool(r.get('license_candidate_eligible')) for r in rows),
                         'strict_v7_usable': 0, 'qa21_domain_member': cls in QA21_CLASSES,
                         'duration_range_s': [min(durations), max(durations)] if durations else None,
                         'integrated_lufs_range': [min(loudness), max(loudness)] if loudness else None,
                         'starss_mapping': mapping[cls]})
    result = {'source_pool': str(pool_path), 'official_metadata_zip': str(official_zip),
              'measurement_command': 'ffmpeg -i PATH -af loudnorm=I=-23:TP=-3:LRA=7:print_format=json -f null -',
              'classes': by_class, 'records': records, 'record_count': len(records),
              'strict_v7_usable': 0, 'license_counts': dict(Counter(r.get('license') for r in records)),
              'normalization_applied_in_this_run': normalize_research,
              'qualification': 'inventory only; use automatic classifier, deduplication and activity receipts for admission'}
    _write(output / 'dry_sound_admission.json', result)
    return result


def audit_humans(assets_path, output):
    assets = _json(assets_path)['assets']
    rows = []
    for asset in assets:
        path = Path(asset['asset_manifest_path'])
        manifest = _json(path)
        root = path.parent
        provenance = _json(root / 'provenance_manifest.json')
        action = _json(root / 'actions/action_manifest.json')
        row = {'asset_id': manifest['asset_id'], 'template_id': manifest.get('template_id'),
               'appearance': asset.get('appearance'), 'asset_manifest_path': str(path),
               'upstream_license': 'MIT', 'license_url': 'https://github.com/microsoft/Microsoft-Rocketbox/blob/master/LICENSE.md',
               'retained_license_decision': provenance.get('license_decision'),
               'provenance': provenance.get('source_manifest'), 'admission_state': manifest.get('admission_state'),
               'formal_usable': False, 'formal_blockers': ['retained_asset_rights_decision_review_required',
                                                         'human_visual_and_motion_contact_admission_not_run'],
               'mouth_anchors': [a for a in manifest['anchors'] if a['anchor_id'] == asset.get('emitter_anchor_id', 'muzzle')],
               'actions': [], 'sitting_action_available': any(a.get('action_id') == 'sit' for a in manifest['actions']),
               'multi_person_native_verified': [], 'multi_person_native_status': 'not_run_by_inventory_tool'}
        for declaration in manifest['actions']:
            file = next(f for f in manifest['files'] if f['role'] == declaration['poses_file_role'])
            target = root / file['path']
            with np.load(target, allow_pickle=False) as data:
                keys = list(data.files)
                shapes = {key: list(data[key].shape) if hasattr(data[key], 'shape') else {'encoded_bytes': len(data[key])} for key in keys}
            row['actions'].append({**declaration, 'path': str(target),
                                   'archive_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                                   'array_shapes': shapes, 'bake_report': action.get('action_bake_report', {}).get('snapshot', {}).get('actions')})
        row['archive_contains_both_actions'] = len(row['actions']) >= 2 and len({a['archive_sha256'] for a in row['actions']}) == 1
        row['same_archive_is_not_same_action'] = True
        rows.append(row)
    result = {'records': rows, 'runtime_assets': len(rows),
              'distinct_template_ids': len({r['template_id'] for r in rows}),
              'distinct_body_identities': None, 'body_identity_status': 'not_verified_by_template_ids', 'formal_usable': 0,
              'raw_upstream_available_is_not_runtime_admission': True}
    _write(Path(output) / 'human_asset_admission.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--official-metadata-zip', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--humans', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--normalize-research', action='store_true')
    args = parser.parse_args()
    dry = audit_sounds(args.pool, args.official_metadata_zip, args.mapping, args.output, workers=args.workers, normalize_research=args.normalize_research)
    human = audit_humans(args.humans, args.output)
    print(json.dumps({'classes': len(dry['classes']), 'dry_records': dry['record_count'],
                      'strict_v7_dry_usable': 0, 'runtime_humans': human['runtime_assets'],
                      'distinct_human_identities': human['distinct_body_identities'], 'formal_humans': 0}))


if __name__ == '__main__':
    main()
