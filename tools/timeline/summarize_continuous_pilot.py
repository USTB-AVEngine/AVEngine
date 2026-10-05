#!/usr/bin/env python3
"""Read back continuous-pilot media, summarize coverage, and estimate CPU throughput."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from avengine.spatial_audio.audio import read_float32_wav
from avengine.timeline.continuous_pilot import write_json


def read(path):
    return json.loads(Path(path).read_text())


def scenes(root):
    records = []
    for pointer in sorted(Path(root).glob('houses/*/scenes/*/completed.json')):
        receipt = read(pointer)
        records.append((pointer.parent, receipt))
    if not records:
        raise ValueError('no completed continuous scenes')
    return records


def probe_video(path, expected_frames):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
               '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames',
               '-of', 'json', str(path)]
    result = read_json_process(command)['streams'][0]
    if (int(result['width']), int(result['height']), result['r_frame_rate'], int(result['nb_read_frames'])) != (1280, 720, '15/1', expected_frames):
        raise ValueError(f'video contract mismatch: {path}: {result}')
    return result


def read_json_process(command):
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def ensure_public(value):
    forbidden = {'gold', 'answer', 'canonical_answer', 'correct_answer', 'correct_option', 'teacher_answer',
                 'source_id', 'actor_id', 'physical_source_id'}
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ValueError('private answer/identity leaked into public request')
        for child in value.values():
            ensure_public(child)
    elif isinstance(value, list):
        for child in value:
            ensure_public(child)


def readback(scene_records):
    """Verify encoded bytes and an independently written FOA coordinate transform."""
    output, question_ids = [], set()
    for root, receipt in scene_records:
        scene = receipt['scene']; attempt = root / receipt['attempt']
        rate = 16000
        stems = [read_float32_wav(attempt / f'private_{key}_raw_full_tail.wav').samples.astype(float)
                 for key in scene['sources']]
        # All world stems have the same full-tail length for these plans.
        n = max(a.shape[1] for a in stems)
        raw = np.zeros((4, n))
        for a in stems:
            raw[:, :a.shape[1]] += a
        forward = np.asarray(scene['camera_forward'], float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 1., 0.]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        xyz = raw[[3, 1, 2]] / math.sqrt(3)
        independently_rotated = np.stack((raw[0], -(right @ xyz), up @ xyz, forward @ xyz))
        full = read_float32_wav(attempt / 'scene_foa_full_tail.wav')
        if full.sample_rate_hz != rate or full.samples.shape != (4, n):
            raise ValueError('full-tail FOA shape/rate mismatch')
        transform_error = float(np.max(np.abs(full.samples - independently_rotated)))
        if transform_error > 2e-6:
            raise ValueError('world N3D to camera SN3D media mismatch')
        full_video = probe_video(attempt / 'scene_video.mp4', round(scene['duration_s'] * 15))
        for window in receipt['windows']:
            folder = root / window['path']
            foa = read_float32_wav(folder / 'audio_foa.wav')
            mono = read_float32_wav(folder / 'audio_mono.wav')
            if foa.sample_rate_hz != rate or foa.samples.shape != (4, 160000) or mono.sample_rate_hz != rate or mono.samples.shape != (1, 160000):
                raise ValueError('window audio contract mismatch')
            start = round(window['start_s'] * rate)
            if not np.array_equal(foa.samples, full.samples[:, start:start+160000]):
                raise ValueError('window reset or crop mismatch')
            if not np.array_equal(mono.samples, foa.samples[:1]):
                raise ValueError('mono is not the same W channel')
            video = probe_video(folder / 'video.mp4', 150)
            public, private, audit = [read(folder / x) for x in ('public_requests.json', 'private_answers.json', 'private_protocol_audit.json')]
            ensure_public(public)
            if {q['question_id'] for q in public} != {q['question_id'] for q in private}:
                raise ValueError('public/private question join mismatch')
            for q in public:
                if q['question_id'] in question_ids:
                    raise ValueError('duplicate global question primary key')
                question_ids.add(q['question_id'])
            private_by_id = {a['question_id']: a for a in private}
            for q in public:
                if q['qa_id'] == 'QA-23' and int(private_by_id[q['question_id']]['canonical_answer']) != len(audit['segments']):
                    raise ValueError('QA-23 does not count all half-open activity segments')
                if q['qa_id'] == 'STARSS-SPEAKERS':
                    expected = len({e['source_id'] for e in audit['segments'] if e['class_id'] in {0, 1}})
                    if int(private_by_id[q['question_id']]['canonical_answer']) != expected:
                        raise ValueError('speaker deduplication failed')
            if not np.all(np.isfinite(foa.samples)) or not np.any(foa.samples):
                raise ValueError('audio is nonfinite or silent')
            output.append({'scene_id': scene['scene_id'], 'duration_s': scene['duration_s'],
                           'window_id': window['window_id'], 'window_start_s': window['start_s'],
                           'independent_parent_scene_id': window['parent_scene_id'],
                           'audio_peak_abs': float(np.max(abs(foa.samples))),
                           'audio_w_rms': float(np.sqrt(np.mean(np.square(foa.samples[0].astype(float))))),
                           'independent_foa_transform_max_abs_error': transform_error,
                           'full_scene_video': full_video, 'window_video': video,
                           'qa_forms': dict(Counter(q['qa_id'] for q in public)),
                           'question_count': len(public), 'exclusion_counts': dict(Counter(e['reason'] for e in audit['exclusions'])),
                           'media_directory': str(folder), **window['coverage']})
    return output


def csv_save(path, rows):
    with Path(path).open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def throughput(rate, extra_cpu_seconds):
    # Engineering workload, explicitly not an observed STARSS distribution.
    cases = []
    per_parent = 1 + 2 * math.ceil(30*.60/.25) + 2*8  # static + moving + conservative corners
    routes = 96  # 16 houses x 6 checked reusable routes; a proposal, not built.
    bank_rirs = routes * (math.ceil(30*.60/.05) + math.ceil(30*.60/.25) + 8)
    for name, workers, machine_rate, availability in (
        ('48g', 64, rate, 1.), ('CCI_plan_only', 128, rate*2, .875)):
        effective_rate = machine_rate*availability
        for independent_parents, cuts, label in ((1000, 3, '3000 correlated windows / 1000 parents'),
                                                 (1500, 2, '3000 correlated windows / 1500 parents'),
                                                 (2000, 3, '2000 independent parents / 6000 related windows'),
                                                 (3000, 3, '3000 independent parents / 9000 related windows'),
                                                 (4000, 3, '4000 independent parents / 12000 related windows')):
            members = independent_parents*1.35  # conservative recompute all counterfactual members
            rirs = members*per_parent
            native_h = rirs/effective_rate/3600
            # The non-RIR term is a CPU proxy-video pilot extrapolation, NOT formal Habitat RGB.
            extra_h = members*extra_cpu_seconds/(workers*.50*availability)/3600
            bank_h = bank_rirs/effective_rate/3600
            total = native_h + extra_h + bank_h
            # Checking each entirely new path at 50 ms defeats route-bank amortization.
            per_parent_validation = members*(2*math.ceil(30*.60/.05))/effective_rate/3600
            cases.append({'machine': name, 'workers': workers, 'case': label,
                          'independent_parents': independent_parents, 'cuts_per_parent': cuts,
                          'delivered_windows_before_counterfactuals': independent_parents*cuts,
                          'counterfactual_factor': 1.35, 'rirs_per_member': per_parent,
                          'production_rirs': rirs, 'effective_rirs_per_second': effective_rate,
                          'native_rir_hours': native_h, 'non_rir_proxy_cpu_hours': extra_h,
                          'one_time_96_route_bank_hours': bank_h,
                          'planned_hours_margin_125': total*1.25, 'planned_hours_margin_150': total*1.5,
                          'additional_hours_if_every_member_has_new_dense_validated_paths': per_parent_validation,
                          'CCI_single_core_relative_speed_assumption': 1. if name == 'CCI_plan_only' else None,
                          'reboot_availability': availability, 'production_native_RGB_budget_verified': False})
    return cases


def make_plots(out, benches, scene_rows, warm_rows, validations, reviews):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([x['workers'] for x in benches], [x['rirs_per_simulation_span_second'] for x in benches], 'o-')
    axes[0].set(xlabel='CPU processes, 1 RLR thread each', ylabel='Native RIR/s (simulation span)', title='48g measured throughput')
    values = [np.mean([r['wall_seconds'] for r in warm_rows]),
              np.mean([r['wall_seconds'] for r in scene_rows if r['duration_s'] == 60]),
              np.mean([r['wall_seconds'] for r in scene_rows if r['duration_s'] == 30])]
    axes[1].bar(['30 s warm', '60 s warm', '30 s initial'], values, color=['#4682b4', '#ef9854', '#898989'])
    axes[1].set(ylabel='Seconds / scene, including 3 windows', title='Keep cold and warm cache costs separate')
    fig.tight_layout(); fig.savefig(out/'throughput_and_length.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, policy in enumerate(('quarter_second', 'half_meter', 'selected')):
        axes[0].plot(range(1, len(validations)+1), [v['policies'][policy]['max_foa_error_deg'] for v in validations], 'o-', label=policy)
    axes[0].axhline(5, color='red', linestyle='--'); axes[0].legend()
    axes[0].set(xlabel='House / route case', ylabel='Maximum native vs blended FOA error (deg)', title='0.5 m alone fails near-field routes')
    axes[1].bar(range(1, len(reviews)+1), [r['encoded_marker_vs_foa_error_deg'] for r in reviews])
    axes[1].axhline(5, color='red', linestyle='--')
    axes[1].set(xlabel='Fixed encoded window check', ylabel='Marker vs FOA error (deg)', title='Actual CPU proxy video readback')
    fig.tight_layout(); fig.savefig(out/'foa_validation.png', dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); a = args.artifact_root; out = args.output
    out.mkdir(parents=True, exist_ok=False)
    completed = scenes(a/'pilot_final')
    warm_completed = scenes(a/'pilot_warm30_control')
    windows = readback(completed)
    scene_rows, warm_rows = [], []
    for values, target in ((completed, scene_rows), (warm_completed, warm_rows)):
        for root, r in values:
            d = r['scene']; cache = r['cache_delta']; plan = read(root/r['attempt']/'rir_plan.json')
            target.append({'scene_id': d['scene_id'], 'house': d['house'], 'split': d['split'],
                           'duration_s': d['duration_s'], 'programmed_events': len(d['events']),
                           'wall_seconds': r['wall_seconds'], 'process_cpu_seconds': r['process_cpu_seconds'],
                           'child_cpu_seconds': r['child_cpu_seconds'], 'total_cpu_seconds': r['total_cpu_seconds'],
                           'native_rir_cpu_seconds': cache['native_cpu_seconds'],
                           'non_rir_total_cpu_seconds': r['total_cpu_seconds']-cache['native_cpu_seconds'],
                           'new_rirs': cache['new_rirs'], 'cache_hits': cache['cache_hits'],
                           'moving_keyframes': len(plan['voice2']['keyframes']),
                           'moving_unique_positions': plan['voice2']['unique_positions'],
                           'window_count': len(r['windows']), 'question_count': r['question_count'],
                           'peak_rss_gib': r['peak_rss_bytes']/2**30,
                           'receipt_path': str(root/'completed.json')})
    benches = sorted(read(a/'benchmark_small/summary.json')+read(a/'benchmark_64/summary.json'), key=lambda x:x['workers'])
    benchmark_rows = [{k:b[k] for k in ('workers', 'rlr_threads_per_worker', 'measured_rirs', 'controller_wall_seconds',
                                        'simulation_span_seconds', 'rirs_per_simulation_span_second',
                                        'native_simulation_sum_cpu_seconds', 'simultaneous_memory_estimate_bytes')} for b in benches]
    validations = []
    for p in sorted((a/'pilot_final').glob('houses/*/validation_s*/interpolation_validation.json')):
        r = read(p)
        validations.append({'house': p.parent.parent.name, 'route': p.parent.name, 'source_path': str(p),
                            'reference_samples': r['reference_samples'], 'policies': {}})
        for k,v in r['policies'].items():
            checks = v['checks']; drr = np.abs([c['drr_difference_db'] for c in checks])
            moving_drr = [abs(c['drr_difference_db']) for c in checks if c['time_s']%10 >= 2 and c['time_s']%10 <= 6 or c['time_s']%10 >= 8]
            rt = [abs(c['rt60_interpolated_s']-c['rt60_reference_s']) for c in checks
                  if c['rt60_interpolated_s'] is not None and c['rt60_reference_s'] is not None]
            validations[-1]['policies'][k] = {key:v[key] for key in ('keyframes','unique_positions','max_foa_error_deg','max_geometry_error_deg','pass_5deg')}
            validations[-1]['policies'][k].update({'median_abs_drr_difference_db':float(np.median(drr)),
                   'max_abs_drr_difference_db':float(max(drr)), 'moving_drr_abs_p95_db':float(np.percentile(moving_drr,95)),
                   'max_abs_rt60_difference_s':float(max(rt)) if rt else None,
                   'interpolated_tail_truncated_count':sum(c['interpolated_tail_truncated'] for c in checks)})
    reviews = read(a/'review_5_windows/checks.json')
    extra_cpu = max(float(np.mean([r['total_cpu_seconds'] for r in warm_rows])),
                    float(np.mean([r['non_rir_total_cpu_seconds'] for r in scene_rows if r['duration_s']==30])))
    estimates = throughput(benches[-1]['rirs_per_simulation_span_second'], extra_cpu)
    flags = ('carry_in','continuous_ge4s','same_actor_repeated','moving_while_active','stop_then_voice_resume','two_source_overlap_ge05s','more_than6')
    coverage = []
    for length in (30,60):
        ws = [w for w in windows if w['duration_s']==length]
        coverage.append({'duration_s':length, 'windows':len(ws),
                         'activity_segments_min':min(w['activity_segments'] for w in ws),
                         'activity_segments_max':max(w['activity_segments'] for w in ws),
                         **{k:sum(bool(w[k]) for w in ws)/len(ws) for k in flags}})
    summary = {'status':'CPU_native_audio_and_proxy_visual_pilot_completed',
               'generated_utc':datetime.now(timezone.utc).isoformat(), 'artifact_root':str(a),
               'parent_count':len({r['scene']['independent_parent_scene_id'] for _,r in completed}),
               'main_30s_scene_count':sum(r['duration_s']==30 for r in scene_rows),
               'paired_60s_scene_count':sum(r['duration_s']==60 for r in scene_rows),
               'window_count':len(windows), 'question_form_count':sum(r['question_count'] for r in scene_rows),
               'cold_or_initial_30s_mean_wall_s':float(np.mean([r['wall_seconds'] for r in scene_rows if r['duration_s']==30])),
               'cold_or_initial_30s_mean_cpu_s':float(np.mean([r['total_cpu_seconds'] for r in scene_rows if r['duration_s']==30])),
               'warm30_mean_wall_s':float(np.mean([r['wall_seconds'] for r in warm_rows])),
               'warm30_mean_cpu_s':float(np.mean([r['total_cpu_seconds'] for r in warm_rows])),
               'warm60_mean_wall_s':float(np.mean([r['wall_seconds'] for r in scene_rows if r['duration_s']==60])),
               'warm60_mean_cpu_s':float(np.mean([r['total_cpu_seconds'] for r in scene_rows if r['duration_s']==60])),
               'conservative_non_rir_30s_cpu_seconds_for_model':extra_cpu,
               'production_rirs_first_pilot':sum(r['new_rirs'] for r in scene_rows),
               'controller':read(min((a/'pilot_final').glob('controller_receipt_*.json'))),
               'benchmarks':benchmark_rows, 'coverage':coverage, 'validation':validations,
               'fixed_window_checks':reviews,'throughput_estimates':estimates,
               'media_readback':{'windows':len(windows),'native_sn3d_transform_max_abs_error':max(w['independent_foa_transform_max_abs_error'] for w in windows),
                    'video_frames_per_window':150,'audio_samples_per_channel_per_window':160000,'finite_nonzero_audio':True,
                    'public_requests_without_private_answer_or_identity':True,'whole_history_crop_exact':True},
               'formal_native_Habitat_RGB_verified':False,'full_v6_28_question_types_wired':False,
               'moving_RIR_waveform_equivalence_verified':False,'long_tail_profiles_measured':False,'CCI_measured':False}
    write_json(out/'summary.json',summary); write_json(out/'media_readback.json',windows)
    csv_save(out/'scene_timings.csv',scene_rows); csv_save(out/'warm30_control_timings.csv',warm_rows)
    csv_save(out/'benchmarks.csv',benchmark_rows); csv_save(out/'coverage.csv',coverage); csv_save(out/'throughput_estimates.csv',estimates)
    csv_save(out/'window_results.csv',[{k:w[k] for k in ('window_id','independent_parent_scene_id','duration_s','window_start_s','activity_segments',*flags,'overlap_seconds','question_count','media_directory')} for w in windows])
    make_plots(out,benches,scene_rows,warm_rows,validations,reviews)
    print(json.dumps({k:summary[k] for k in ('status','parent_count','window_count','question_form_count','production_rirs_first_pilot')},ensure_ascii=False))


if __name__ == '__main__':
    main()
