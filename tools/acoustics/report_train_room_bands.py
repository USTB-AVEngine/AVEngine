#!/usr/bin/env python3
"""Apply training-only leakage-band policy to retained per-room native evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
LEAK_REVIEW_REASON = 'GEOMETRIC_ESCAPE_5_TO_15_PERCENT'
TYPE_NAMES = {
    'living': '客厅', 'bedroom': '卧室', 'kitchen': '厨房',
    'bathroom': '卫生间', 'office': '书房/办公室', 'dining': '餐厅',
    'entryway': '入口/门厅', 'meeting': '会议室', 'gym': '健身房',
    'recreation': '娱乐室', 'laundry': '洗衣房', 'unknown': '类型未知',
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def key(row):
    return row['house'], row['room_label']


def index_rooms(rows):
    indexed = {}
    for row in rows:
        identity = key(row)
        if identity in indexed:
            raise ValueError(f'Duplicate house/room identity: {identity}')
        indexed[identity] = row
    return indexed


def training_decision(row):
    """Waive only the leakage review reason; preserve every other frozen check."""
    failures = set(row.get('reasons', [])) - {LEAK_REVIEW_REASON}
    if row.get('stage1_status') != 'pass':
        failures.add('ROOM_RULES_NOT_PASS')
    if row.get('visual_status') != 'pass':
        failures.add('CPU_VISUAL_NOT_PASS')
    if row.get('acoustic_status') not in ('pass', 'review'):
        failures.add('NATIVE_ACOUSTIC_NOT_PASS_OR_REVIEW')
    leak = row.get('max_escape_fraction')
    if not finite(leak) or not 0 <= leak <= .15:
        failures.add('LEAKAGE_MISSING_OR_ABOVE_15_PERCENT')
    for field, limit, reason in (
        ('max_foa_angle_deg', 15, 'FOA_MISSING_OR_ABOVE_15_DEG'),
        ('max_direct_delay_ms', 3, 'DIRECT_DELAY_MISSING_OR_ABOVE_3_MS'),
    ):
        value = row.get(field)
        if not finite(value) or not 0 <= value <= limit:
            failures.add(reason)
    exponents = row.get('distance_exponents')
    if not exponents or any(not finite(x) or not .7 <= x <= 1.3 for x in exponents):
        failures.add('DISTANCE_EXPONENT_MISSING_OR_OUTSIDE_0_7_1_3')
    for field, minimum, reason in (
        ('valid_rirs', 6, 'FEWER_THAN_SIX_VALID_RIRS'),
        ('distinct_listeners', 2, 'FEWER_THAN_TWO_LISTENERS'),
    ):
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            failures.add(reason)
    if row.get('geometry_derivation_label') and row.get('geometry_comparison_status') != 'pass':
        failures.add('DERIVED_GEOMETRY_COMPARISON_NOT_PASS')
    geometry = row.get('acoustic_geometry_derivation') or {}
    if geometry.get('removed_faces') and not row.get('acoustic_before_waveform_available'):
        failures.add('ACOUSTIC_GEOMETRY_BEFORE_WAVEFORM_UNAVAILABLE')
    band = finite(leak) and .05 < leak <= .15
    if not band and row.get('acoustic_status') != 'pass':
        failures.add('STRICT_ACOUSTIC_NOT_PASS')
    if band and row.get('acoustic_status') == 'review' and LEAK_REVIEW_REASON not in row.get('reasons', []):
        failures.add('LEAKAGE_REVIEW_REASON_MISSING')
    tags = []
    if not failures:
        if band:
            tags.append('acoustic_review_band_5_15')
        if row.get('acoustic_patched'):
            tags.append('acoustic_patched')
    return dict(training_eligible=not failures,
                acoustic_review_band_5_15=band and not failures,
                acoustic_patched=bool(row.get('acoustic_patched')) and not failures,
                training_tags=tags, training_exclusion_reasons=sorted(failures),
                allowed_splits=['train'], development_test_eligible=False)


def policy_row(row):
    result = copy.deepcopy(row)
    result['previous_training_eligible'] = row['training_eligible']
    result['previous_status'] = row['status']
    result['previous_tier'] = row['tier']
    result.update(training_decision(row))
    if result['training_eligible']:
        result['status'] = 'qualified' if result['training_tags'] else 'pass'
        result['tier'] = 'training_review_band' if result['acoustic_review_band_5_15'] else 'training'
    elif row['tier'] == 'leak_5_to_15_review':
        result['tier'] = 'leak_5_to_15_other_checks_review'
    result['split'] = 'train'
    result['training_policy'] = 'owner_20261004_leakage_band_train_only'
    return result


def counts(rows):
    types = Counter(row['room_type'] for row in rows)
    return dict(houses=len({row['house'] for row in rows}), rooms=len(rows),
                room_types=dict(sorted(types.items())))


def family_counts(rows):
    return {family: counts([r for r in rows if r['family'] == family])
            for family in ('hm3d', 'mp3d')}


def csvfile(path, columns, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False, sort_keys=True)
                             if isinstance(row.get(k), (dict, list)) else row.get(k)
                             for k in columns})


def checked_path(path, roots):
    p = Path(path).resolve()
    if not any(p.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(f'Evidence escaped explicit input roots: {p}')
    if not p.is_file():
        raise ValueError(f'Missing retained evidence: {p}')
    return p


def verify_native(row, native, settings, source_roots):
    # Re-run the unchanged classifier, keeping the actual leakage measurements.
    from tools.acoustics.accept_houses import classify
    actual = copy.deepcopy(native)
    classify(actual, settings['criteria'])
    if actual['status'] != row['acoustic_status'] or set(actual['reasons']) != set(row['acoustic_reasons']):
        raise ValueError(f'Retained acoustic decision mismatch: {key(row)}')
    pairs = {
        'max_escape_fraction': actual['max_escape_fraction'],
        'max_foa_angle_deg': actual['max_foa_angle_deg'],
        'rir_count': len(actual['rirs']),
        'valid_rirs': sum(x['status'] == 'valid' for x in actual['rirs']),
        'distinct_listeners': actual['distinct_listener_positions'],
        'distance_exponents': [x.get('pressure_exponent') for x in actual['distance_attenuation']],
        'max_direct_delay_ms': max((abs(x['direct_delay_error_ms']) for x in actual['rirs']
                                    if x.get('direct_delay_error_ms') is not None), default=None),
    }
    for field, value in pairs.items():
        if row[field] != value:
            raise ValueError(f'Native metric mismatch {key(row)} {field}: {row[field]} != {value}')
    if len(native['foa_checks']) != 3 or any(x.get('fixed_check_window_ms') != .5 for x in native['foa_checks']):
        raise ValueError(f'Original fixed FOA checks incomplete: {key(row)}')
    if any(not finite(x.get('direct_delay_error_ms')) or abs(x['direct_delay_error_ms']) > 3
           for x in native['rirs']):
        raise ValueError(f'Direct arrivals incomplete: {key(row)}')
    visual = read(checked_path(row['visual_path'], source_roots))
    if visual['status'] != 'pass' or len(visual['checks']) != 3:
        raise ValueError(f'CPU view evidence incomplete: {key(row)}')
    for check in visual['checks']:
        if check['status'] != 'pass':
            raise ValueError(f'CPU view did not pass: {key(row)}')
        checked_path(check['image_path'], source_roots)
    if visual.get('geometry_derivation'):
        comparison = visual['derivation_comparison']
        if (visual['geometry_comparison_status'] != 'pass'
                or not comparison['acoustic']['exactly_equal']
                or any(x['different_pixels'] != 0 for x in comparison['cpu_views'])
                or comparison['overhead']['different_pixels'] != 0):
            raise ValueError(f'Derived geometry comparison incomplete: {key(row)}')


def verify_new_waveforms(native_rows, roots):
    import numpy as np
    archives = defaultdict(dict)
    for native in native_rows:
        for rir in native['rirs']:
            path, name = rir['rir_path'].split('#/', 1)
            checked_path(path, roots)
            archives[path][name] = rir['sample_count']
    waves = samples = 0
    for path, requested in sorted(archives.items()):
        with np.load(path, allow_pickle=False) as data:
            for name, length in requested.items():
                wave = data[name]
                if (wave.ndim != 2 or wave.shape != (4, length) or not 0 < length <= 64000
                        or not np.isfinite(wave).all() or not np.any(wave != 0)):
                    raise ValueError(f'Invalid retained native waveform: {path}#/{name}')
                waves += 1
                samples += int(wave.size)
    return dict(archives=len(archives), waveforms=waves, samples=samples)


def holdouts(previous, sources):
    reserved = set(previous)
    def scan(value):
        if isinstance(value, str) and re.fullmatch(r'hm3d_val_\d{5}_[A-Za-z0-9]+', value):
            reserved.add(value)
        elif isinstance(value, list):
            for item in value:
                scan(item)
        elif isinstance(value, dict):
            for name, item in value.items():
                scan(name)
                scan(item)
    for source in sources:
        scan(read(source))
    return reserved


def patch_candidates(root, room_index, settings, reserved):
    """Read completed patch-room evidence only; never launch or change that task."""
    audit = dict(root=str(root) if root else None, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                 status='not_supplied', admitted_rooms=[], rejected_rooms=[], pending_houses=[])
    if root is None:
        return {}, audit
    required = ['summary.json', 'all_house_results.json', 'evidence/room_verification_execution.json']
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        audit.update(status='pending_final_room_delivery', missing_final_files=missing)
        for path in sorted(list((root / 'pilot/houses').glob('*/complete.json'))
                           + list((root / 'batch/houses').glob('*/complete.json'))):
            record = read(path.parent / 'patch_record.json')
            if record['patch_count']:
                audit['pending_houses'].append(dict(house=record['house'], patch_count=record['patch_count'],
                                                    patch_record_path=str(path.parent / 'patch_record.json')))
        return {}, audit
    summary = read(root / 'summary.json')
    execution = read(root / 'evidence/room_verification_execution.json')
    if summary.get('missing_houses') or execution.get('missing'):
        audit.update(status='incomplete_final_room_delivery', missing_houses=summary.get('missing_houses'),
                     missing_verifications=execution.get('missing'))
        return {}, audit
    additions = {}
    audit['status'] = 'completed_delivery_checked'
    for house in read(root / 'all_house_results.json'):
        if house.get('group') != 'blocked' or not house.get('patch_count'):
            continue
        if house['house'] in reserved:
            audit['rejected_rooms'].append(dict(house=house['house'], reasons=['HOLDOUT_RESERVED']))
            continue
        verification_path = house.get('room_verification_path')
        if not verification_path:
            audit['pending_houses'].append(dict(house=house['house'], reasons=['PER_ROOM_VERIFICATION_MISSING']))
            continue
        verification = read(checked_path(verification_path, [root]))
        record_path = checked_path(house['patch_record_path'], [root])
        record = read(record_path)
        if record['patch_count'] <= 0 or record.get('consistency_error'):
            raise ValueError(f'Patch summary/record mismatch: {house["house"]}')
        for room in verification['rooms']:
            identity = house['house'], room['room_label']
            if identity not in room_index:
                audit['rejected_rooms'].append(dict(house=identity[0], room_label=identity[1], reasons=['NOT_ROUND3_TRAIN_CANDIDATE']))
                continue
            original = room_index[identity]
            if training_decision(original)['training_eligible']:
                # Keep an already eligible original room, without gratuitously replacing its geometry/RIR.
                continue
            baseline = read(checked_path(room['runs']['baseline']['result_path'], [root]))
            local = read(checked_path(room['runs']['local']['result_path'], [root]))
            native_rooms = local['input']['acoustic_rooms']
            if len(native_rooms) != 1 or native_rooms[0]['room_label'] != identity[1]:
                raise ValueError(f'Patch room scope mismatch: {identity}')
            if native_rooms[0]['placement'] != original['placement']:
                audit['rejected_rooms'].append(dict(house=identity[0], room_label=identity[1], reasons=['PATCH_PLACEMENT_DIFFERS_FROM_ROUND3']))
                continue
            replacement = copy.deepcopy(original)
            replacement.update(acoustic_status=local['status'], acoustic_reasons=local['reasons'],
                reasons=sorted((set(original['reasons']) - set(original['acoustic_reasons'])) | set(local['reasons'])),
                max_escape_fraction=local.get('max_escape_fraction'), max_foa_angle_deg=local.get('max_foa_angle_deg'),
                distance_exponents=[x.get('pressure_exponent') for x in local.get('distance_attenuation', [])],
                rir_count=len(local.get('rirs', [])), valid_rirs=sum(x['status']=='valid' for x in local.get('rirs', [])),
                distinct_listeners=local.get('distinct_listener_positions', 0),
                max_direct_delay_ms=max((abs(x['direct_delay_error_ms']) for x in local.get('rirs', [])
                                        if x.get('direct_delay_error_ms') is not None), default=None),
                result_path=room['runs']['local']['result_path'], unpatched_result_path=original['result_path'],
                acoustic_patched=True, patch_record_path=str(record_path), patch_verification_path=str(verification_path),
                patch_baseline_result_path=room['runs']['baseline']['result_path'],
                patch_package_manifest=record['local_package_manifest'])
            decision = training_decision(replacement)
            if not decision['training_eligible']:
                audit['rejected_rooms'].append(dict(house=identity[0], room_label=identity[1], reasons=decision['training_exclusion_reasons']))
                continue
            if not local['max_escape_fraction'] < baseline['max_escape_fraction']:
                audit['rejected_rooms'].append(dict(house=identity[0], room_label=identity[1], reasons=['NO_MEASURED_PATCH_LEAKAGE_IMPROVEMENT']))
                continue
            verify_native(replacement, local, settings[original['family']], [root, Path(original['visual_path']).parents[4]])
            additions[identity] = replacement
            audit['admitted_rooms'].append(dict(house=identity[0], room_label=identity[1], result_path=replacement['result_path'],
                                                training_tags=decision['training_tags']))
    return additions, audit


def aggregate_house(original, candidate, rows, native):
    result = copy.deepcopy(original)
    accepted = [r for r in rows if r['training_eligible']]
    labels = {r['room_label'] for r in accepted}
    tags = sorted({tag for r in accepted for tag in r['training_tags']})
    result.update(accepted_room_count=len(accepted), accepted_room_labels=[r['room_label'] for r in accepted],
                  training_eligible=bool(accepted), batch_eligible=bool(accepted), split='train', allowed_splits=['train'],
                  development_test_eligible=False, training_tags=tags,
                  acoustic_review_band_5_15=any(r['acoustic_review_band_5_15'] for r in accepted),
                  acoustic_patched=any(r['acoustic_patched'] for r in accepted),
                  training_review_band_room_count=sum(r['acoustic_review_band_5_15'] for r in accepted),
                  acoustic_patched_room_count=sum(r['acoustic_patched'] for r in accepted),
                  strict_original_room_count=sum(r['previous_training_eligible'] for r in accepted),
                  review_leak_room_count=sum(not r['training_eligible'] and r['previous_tier']=='leak_5_to_15_review' for r in rows),
                  other_review_room_count=sum(not r['training_eligible'] and r['previous_tier']=='other_review' for r in rows),
                  failed_room_count=sum(not r['training_eligible'] and r['previous_tier']=='fail' for r in rows),
                  accepted_room_types=dict(sorted(Counter(r['room_type'] for r in accepted).items())),
                  status=('qualified' if tags else 'pass') if accepted else original['status'],
                  native_status=('review' if any(r['acoustic_status']=='review' for r in accepted) else 'pass') if accepted else original['native_status'],
                  visual_status='pass' if accepted else original['visual_status'],
                  accepted_area_sum_m2=sum(r['floor_area_m2'] for r in accepted),
                  max_escape_fraction=max((r['max_escape_fraction'] for r in accepted), default=None),
                  max_foa_angle_deg=max((r['max_foa_angle_deg'] for r in accepted), default=None),
                  reasons=sorted({reason for r in rows if not r['training_eligible'] for reason in r['training_exclusion_reasons']}),
                  accepted_room_training_tags={r['room_label']:r['training_tags'] for r in accepted},
                  per_room_evidence=[dict(room_label=r['room_label'], result_path=r['result_path'], visual_path=r['visual_path'],
                                          training_tags=r['training_tags']) for r in accepted],
                  scope='training only; accepted_room_labels are the exact allowed rooms; native review remains recorded')
    result['input'].update(split='train', allowed_splits=['train'], accepted_room_count=len(accepted),
                           accepted_room_labels=result['accepted_room_labels'],
                           acoustic_rooms=[r for r in candidate['acoustic_rooms'] if r['room_label'] in labels])
    for field in ('rirs', 'distance_attenuation', 'leakage', 'foa_checks'):
        result[field] = [item for row in accepted for item in native[key(row)].get(field, [])]
    rt = [r['rt60_mid_s'] for r in result['rirs'] if r['status']=='valid' and r.get('rt60_mid_s') is not None]
    result['rt60_mid_median_s'] = statistics.median(rt) if rt else None
    result['rt60_valid_fraction'] = len(rt)/len(result['rirs']) if result['rirs'] else None
    result['distinct_listener_positions'] = len({tuple(r['listener_m']) for r in result['rirs']})
    result['stage4'].update(status=result['status'], scope='training-only owner policy; raw per-room acoustic statuses preserved')
    return result


def report(args):
    sys.path[:0] = [str(REPO), str(REPO/'src')]
    source, base, output = args.source_root.resolve(), args.base.resolve(), args.output.resolve()
    if not base.is_relative_to(source) or not output.is_relative_to(source.parent) or output.is_relative_to(source):
        raise ValueError('Report must be a fresh sibling of the retained round-three run')
    destination = output/'report_final'
    destination.mkdir(exist_ok=False)
    sources = [base/name for name in ('all_room_results.json', 'all_house_results.json', 'training_houses_v1.json',
                                      'training_rooms_v1.json', 'old_house_veto_training_houses_v1.json', 'summary.json')]
    sources += [source/'inputs/candidates.json', source/'evidence/final_validation_v2.json']
    before = {str(p): digest(p) for p in sources}
    validation = read(source/'evidence/final_validation_v2.json')
    if validation['status'] != 'pass' or not validation['holdout_disjoint']:
        raise ValueError('Round-three native/visual delivery was not validated')
    rows = read(base/'all_room_results.json')
    room_index = index_rooms(rows)
    old_houses = read(base/'old_house_veto_training_houses_v1.json')
    original_houses = {h['house']:h for h in read(base/'all_house_results.json')}
    candidates = {h['house']:h for h in read(source/'inputs/candidates.json')}
    previous_summary = read(base/'summary.json')
    settings = {f:read(source/'inputs'/f'{f}_settings.json') for f in ('hm3d','mp3d')}
    reserved = holdouts(previous_summary['reserved_houses'], args.holdout_source)
    if set(candidates) & reserved:
        raise ValueError(f'Training candidates intersect current holdouts: {set(candidates)&reserved}')
    strict = [r for r in rows if r['training_eligible']]
    for row in strict:
        if not training_decision(row)['training_eligible'] or row['max_escape_fraction'] > .05:
            raise ValueError(f'Existing strict training room no longer passes: {key(row)}')
    old_ids = {h['house'] for h in old_houses}
    old_rows = [r for r in rows if r['house'] in old_ids]
    if len(old_rows) != sum(h['candidate_room_count'] for h in old_houses):
        raise ValueError('Original house-veto room distribution does not match its candidate labels')
    base_rows = [policy_row(r) for r in rows]
    band = [r for r in base_rows if r['acoustic_review_band_5_15']]
    blocked_band = [r for r in base_rows if r['previous_tier']=='leak_5_to_15_review' and not r['training_eligible']]
    additions, patch_audit = patch_candidates(args.patch_root, room_index, settings, reserved)
    final_rows = [policy_row(additions.get(key(r), r)) for r in rows]
    accepted = [r for r in final_rows if r['training_eligible']]
    native = {}
    new_native = []
    roots = [source] + ([args.patch_root] if args.patch_root else [])
    for row in accepted:
        result = read(checked_path(row['result_path'], roots))
        native[key(row)] = result
        if not row['previous_training_eligible']:
            verify_native(row, result, settings[row['family']], roots)
            new_native.append(result)
    waveform_validation = verify_new_waveforms(new_native, roots)
    groups = defaultdict(list)
    for row in final_rows:
        groups[row['house']].append(row)
    houses = [aggregate_house(original_houses[h], candidates[h], one, native) for h,one in groups.items()]
    comparison = {
        'old_house_veto':family_counts(old_rows),
        'per_room_strict_le_5':family_counts(strict),
        'per_room_plus_review_band':family_counts([r for r in base_rows if r['training_eligible']]),
    }
    types = list(TYPE_NAMES)
    types += sorted({r['room_type'] for r in rows} - set(types))
    distributions = []
    for policy, families in comparison.items():
        for family, value in families.items():
            for room_type in types:
                distributions.append(dict(policy=policy, family=family, room_type=room_type,
                                          room_type_zh=TYPE_NAMES.get(room_type,room_type),
                                          rooms=value['room_types'].get(room_type,0)))
    patch_rooms = [r for r in accepted if r['acoustic_patched']]
    code = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    summary = dict(code_commit=code, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_report=str(base), source_code_commit=previous_summary['code_commit'], cpu_only=True,
        measurement_revision='e3dc246 first arrival + 0.5 ms; retained waveforms only',
        frozen_thresholds_unchanged=True, shared_classifier_unchanged=True, human_labels_filter=False,
        all_candidate_rooms=len(rows), comparison=comparison, final_training=family_counts(accepted),
        review_band_admitted=family_counts(band), review_band_blocked=family_counts(blocked_band),
        review_band_blocked_reasons=dict(Counter(x for r in blocked_band for x in r['training_exclusion_reasons'])),
        acoustic_patched_additions=family_counts(patch_rooms), patch_delivery_status=patch_audit['status'],
        room_type_names=TYPE_NAMES, test_development_modified=False, reserved_houses=sorted(reserved),
        waveform_validation_newly_admitted=waveform_validation, original_strict_waveform_validation=str(source/'evidence/final_validation_v2.json'),
        qualification='training pool admission; CPU scan-ray views, formal Habitat RGB and production episodes not verified',
        policy='only GEOMETRIC_ESCAPE_5_TO_15_PERCENT is waived for train; all other frozen review/fail reasons remain blocking')
    save(destination/'summary.json',summary)
    save(destination/'all_room_results.json',final_rows)
    save(destination/'all_house_results.json',houses)
    save(destination/'training_houses_v1.json',[h for h in houses if h['training_eligible']])
    for name, records in [('training_rooms_v1',accepted), ('review_band_admitted_rooms',band),
                          ('review_band_blocked_rooms',blocked_band), ('acoustic_patched_rooms',patch_rooms),
                          ('strict_training_rooms_v1',strict), ('old_house_veto_training_rooms_v1',old_rows)]:
        save(destination/f'{name}.json',records)
    for source_name,target_name in [('training_houses_v1.json','strict_training_houses_v1.json'),
                                    ('old_house_veto_training_houses_v1.json','old_house_veto_training_houses_v1.json')]:
        shutil.copyfile(base/source_name,destination/target_name)
    columns = ['house','family','room_label','room_type','original_split','split','stage1_status','historical_stage4_status',
        'acoustic_status','visual_status','status','tier','training_eligible','acoustic_review_band_5_15','acoustic_patched',
        'training_tags','allowed_splits','development_test_eligible','previous_training_eligible','previous_status','previous_tier',
        'smy_label','so_seen','m1_seen','v6_seen','max_escape_fraction','max_foa_angle_deg','distance_exponents',
        'rir_count','valid_rirs','distinct_listeners','max_direct_delay_ms','rt60_mid_median_s','rt60_valid_fraction',
        'floor_area_m2','short_side_m','nav_main_area_m2','black_fraction','floor_y_m','geometry_derivation_label',
        'geometry_comparison_status','acoustic_before_waveform_available','reasons','training_exclusion_reasons',
        'result_path','visual_path','unpatched_result_path','patch_record_path','patch_verification_path','patch_package_manifest']
    csvfile(destination/'room_results.csv',columns,final_rows)
    for name in ('training_rooms_v1','review_band_admitted_rooms','review_band_blocked_rooms','acoustic_patched_rooms',
                 'strict_training_rooms_v1','old_house_veto_training_rooms_v1'):
        csvfile(destination/f'{name}.csv',columns,read(destination/f'{name}.json'))
    house_columns = ['house','family','original_split','candidate_area_sum_m2','candidate_room_count','accepted_room_count',
        'accepted_room_labels','strict_original_room_count','training_review_band_room_count','acoustic_patched_room_count',
        'review_leak_room_count','other_review_room_count','failed_room_count','native_status','visual_status','status',
        'batch_eligible','training_eligible','acoustic_review_band_5_15','acoustic_patched','training_tags','allowed_splits',
        'development_test_eligible','rt60_mid_median_s','max_escape_fraction','max_foa_angle_deg','accepted_room_types',
        'accepted_room_training_tags','reasons','result_directory','smy_labels']
    csvfile(destination/'house_results.csv',house_columns,houses)
    csvfile(destination/'room_type_distribution.csv',['policy','family','room_type','room_type_zh','rooms'],distributions)
    csvfile(destination/'comparison_counts.csv',['policy','family','houses','rooms','room_types'],
            [dict(policy=p,family=f,**v) for p,fs in comparison.items() for f,v in fs.items()])
    after = {str(p):digest(p) for p in sources}
    if before != after:
        raise ValueError('Retained round-three inputs changed during report generation')
    save(output/'evidence/input_provenance.json',dict(source_report=str(base), source_sha256=before,
                                                    retained_inputs_unchanged=True, code_commit=code))
    save(output/'evidence/patch_integration.json',patch_audit)
    save(output/'evidence/holdout_audit.json',dict(sources=list(map(str,args.holdout_source)), reserved_houses=sorted(reserved),
        training_house_intersection=[], test_development_modified=False))
    save(output/'evidence/validation.json',dict(status='pass', code_commit=code, all_room_rows=len(final_rows),
        unique_house_room_keys=len(room_index), strict_original_rooms_preserved=len(strict),
        newly_admitted_original_band_rooms=len(band), acoustic_patched_added_rooms=len(patch_rooms),
        newly_admitted_raw_metrics_and_cpu_views_verified=True, newly_admitted_waveforms=waveform_validation,
        all_lists_train_only=True, holdout_disjoint=True, original_inputs_unchanged=True,
        type_distribution_sums_correct=all(sum(v['room_types'].values())==v['rooms'] for fs in comparison.values() for v in fs.values())))
    hm, mp = summary['final_training']['hm3d'], summary['final_training']['mp3d']
    lines = [f"结论：纳入其余检查全通过的漏声 5–15% 房间后，HM3D 可训练 {hm['houses']} 套、{hm['rooms']} 间，MP3D {mp['houses']} 套、{mp['rooms']} 间；新增漏声档 HM3D {summary['review_band_admitted']['hm3d']['rooms']} 间、MP3D {summary['review_band_admitted']['mp3d']['rooms']} 间，均标 acoustic_review_band_5_15，只用于训练。",'',
        f'机器 48g；独立检出 `{REPO}`；分支 claude/v7-train-houses-20261003；提交 `{code}`，未 push。新目录 `{output}`。数字出处：summary.json、comparison_counts.csv、room_type_distribution.csv；逐间原测量和画面路径在 room_results.csv。','',
        '|口径|HM3D 套 / 间|MP3D 套 / 间|','|---|---:|---:|']
    titles = {'old_house_veto':'原口径：整套原定 ≤3 间，一票否决', 'per_room_strict_le_5':'逐间，只收漏声 ≤5%',
              'per_room_plus_review_band':'逐间，增加漏声 >5% 且 ≤15% 合格档'}
    for policy,families in comparison.items():
        lines.append(f"|{titles[policy]}|{families['hm3d']['houses']} / {families['hm3d']['rooms']}|{families['mp3d']['houses']} / {families['mp3d']['rooms']}|")
    lines += ['', '三种口径都以第 3 回合完成检查后的相同候选房间为基准，未换抽样。旧口径间数包含未抽测房间，是整套推断；新口径按精确 accepted_room_labels 收录。套数去重，同一套不同档可重叠。第 2 回合冻结旧口径另为 HM3D 95 套/602 间、MP3D 10 套/70 间，本次对拍使用第 3 回合补完后的 601/70 间。','',
        '房间类型沿用原 room_type；office 记“书房/办公室”，unknown 单列，不推断语义。以下各行类型之和等于该口径间数。','',
        '|家族 / 口径|'+'|'.join(TYPE_NAMES.get(t,t) for t in types)+'|总间数|',
        '|---|'+'|'.join('---:' for t in types)+'|---:|']
    for policy,families in comparison.items():
        for family,value in families.items():
            lines.append('|'+family.upper()+' / '+titles[policy]+'|'+'|'.join(str(value['room_types'].get(t,0)) for t in types)+f"|{value['rooms']}|")
    lines += ['', '本轮只豁免 GEOMETRIC_ESCAPE_5_TO_15_PERCENT，不把原 acoustic_status=review 改写为 pass；训练收录状态记 qualified，标签和 allowed_splits=[train] 同时写入房间及房屋。FOA、距离衰减、有效 RIR、两个听者、直达偏差和 CPU 画面逐项复核，原 RT60、探针完整性、直达路径和几何对拍待审原因继续阻止放行；未改冻结阈值或共享判分器。恰好 5% 属于原严格档，恰好 15% 属于训练漏声档，超过 15% 不通过。', '',
        f"原漏声待审共 HM3D 62 间、MP3D 90 间；本轮纳入 {len(band)} 间，余下 HM3D {summary['review_band_blocked']['hm3d']['rooms']} 间、MP3D {summary['review_band_blocked']['mp3d']['rooms']} 间继续待审，详见 review_band_blocked_rooms.csv。原因计数可重叠：`{json.dumps(summary['review_band_blocked_reasons'],ensure_ascii=False,sort_keys=True)}`。",'',
        f"声学补洞增量：HM3D {summary['acoustic_patched_additions']['hm3d']['houses']} 套/{summary['acoustic_patched_additions']['hm3d']['rooms']} 间，MP3D {summary['acoustic_patched_additions']['mp3d']['houses']} 套/{summary['acoustic_patched_additions']['mp3d']['rooms']} 间；核对状态 `{patch_audit['status']}`，证据 ../evidence/patch_integration.json。只合并已完成的真实补片逐间验收，标 acoustic_patched；no-op、候选补片及尚未逐间验证的整套通过不计救回。补洞增量另列，不混入三种原几何口径的对拍。",'',
        '交付名单：training_houses_v1.json / training_rooms_v1.csv 为更新后训练名单；strict_training_* 是 ≤5% 逐间原名单；old_house_veto_training_* 是原整套口径。house_results.csv / all_house_results.json 保持原字段并追加标签、逐间范围及类型分布。训练必须使用 accepted_room_labels，不能把同屋剩余待审或失败房间带入。smy 和 SO/M1/v6 只记录，不参与筛选。', '',
        f"新增收录保存波形实际读取校验：{waveform_validation['waveforms']} 条、{waveform_validation['samples']} 个 float 样本、{waveform_validation['archives']} 个 NPZ；原 ≤5% 波形沿用 `{source/'evidence/final_validation_v2.json'}` 的完整校验。未重新运行声学或渲染作业。实际命令：`{Path(sys.executable)} tools/acoustics/report_train_room_bands.py --source-root {source} --base {base} --output {output}`（holdout/patch 输入见 ../evidence/command.json）；使用一个空闲 CPU 核、单线程 BLAS/OMP，日志 ../logs/report.log。",'',
        '测试/开发保持漏声 ≤5%、其余全过和整套严格口径；本工具只写新训练产物，不写测试/开发名单。原始数据、旧结果、其他检出、共享环境保持只读。正式 Habitat RGB、生产视频/音频窗口和模型训练未验证。本轮无继续运行的训练、声学或视图作业。']
    if patch_audit['status'] != 'completed_delivery_checked':
        lines += ['', '声学补洞任务尚无完整最终逐间验收交付，本版未把它的候选房屋算作已救回；其共享作业由另一任务管理，本任务只读结果。完成后可用同一工具和新的 output 目录合并，保持本版及第 3 回合产物原样。']
    (destination/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print('TRAINING_BANDS_COMPLETE',json.dumps(dict(final_training=summary['final_training'], review_band_admitted=summary['review_band_admitted'],
            review_band_blocked=summary['review_band_blocked'], acoustic_patched_additions=summary['acoustic_patched_additions'],
            patch_delivery_status=summary['patch_delivery_status'], comparison=comparison),ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--holdout-source',type=Path,action='append',default=[])
    parser.add_argument('--patch-root',type=Path)
    report(parser.parse_args())


if __name__ == '__main__':
    main()
