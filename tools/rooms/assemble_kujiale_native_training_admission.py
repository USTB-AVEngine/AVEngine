#!/usr/bin/env python3
"""Assemble native Kujiale room admission and offline evidence without new gates."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import datetime
import hashlib
import html
import json
import os
import shutil
import statistics
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inline_image(path):
    content = Path(path).read_bytes()
    if content.startswith(b'\x89PNG\r\n\x1a\n'):
        mime = 'image/png'
    elif content.startswith(b'\xff\xd8'):
        mime = 'image/jpeg'
    else:
        raise ValueError('unsupported review image: '+str(path))
    return '<img loading="lazy" src="data:'+mime+';base64,'+base64.b64encode(content).decode()+'">'


def scalar_max(items, key, default=None):
    values = [row[key] for row in items if row.get(key) is not None]
    return max(values, default=default)


def native_house_directory(root, house):
    recovered = root/'ue_round6/recovered_houses'/(house+'_fresh_v2')
    if (recovered/'result.json').is_file() and read(recovered/'result.json')['status'] == 'completed':
        return recovered
    return root/'ue_round6/houses'/house


def collect_rooms(root):
    acoustic_houses = [row['house'] for row in read(root/'acoustic_pass_train_candidates.json')['houses']]
    capsule = {row['house']: row for row in read(root/'ue_round6/native_capsule_contact_v1/result.json')['records']}
    floors = {row['house']: row for row in read(root/'ue_round6/native_floor_spatial_v1/result.json')['records']}
    alignment = {row['house']: row for row in read(root/'ue_round7/native_acoustic_alignment_v2/result.json')['records']}
    overrides = {(row['house'], row['room_id']): row for row in read(root/'ue_round6/native_room_overrides.json')['records']}
    geometry_rooms = {(row['house'], row['room_label']): row for row in read(root/'rooms_results.json')
                      if row['house'] in acoustic_houses and row['geometry_eligible']}
    measured, rows, house_details = {}, [], {}
    for house in acoustic_houses:
        directory = native_house_directory(root, house)
        job = read(directory/'result.json')
        if job['status'] != 'completed':
            raise ValueError('native house not completed: '+house)
        qa_path = directory/'qa/result.json'
        qa = read(qa_path)
        plan = read(directory/'acceptance_plan.json')
        cap_rooms = {row['room_id']: row for row in capsule[house]['rooms']}
        floor_rooms = {row['room_id']: row for row in floors[house]['rooms']}
        aligned_rooms = {row['room_id']: row for row in alignment[house]['rooms']}
        house_details[house] = {'directory': str(directory), 'qa_result': str(qa_path),
                                'job': job, 'plan': plan, 'native_floor_surface': floors[house]['native_floor_surface']}
        for original in qa['rooms']:
            room = deepcopy(original)
            key = (house, room['room_id'])
            actual_qa_path = qa_path
            cap = cap_rooms[room['room_id']]
            floor_record = floor_rooms[room['room_id']]
            validation = floor_record.get('validation')
            if validation is None:
                if room['automated_room_status'] == 'pass':
                    raise ValueError('a visually qualified room lacks measured floor validation')
                validation = {'status': 'not_run_room_rejected',
                              'reason': floor_record.get('status', 'rejected room has no floor probes')}
            selected_indices = [row['index'] for row in room['selected_witnesses']]
            if key in overrides:
                override = overrides[key]
                actual_qa_path = Path(override['qa_result'])
                room = next(r for r in read(actual_qa_path)['rooms'] if r['room_id'] == room['room_id'])
                renewed = read(override['contact_result'])
                if renewed['status'] != 'pass':
                    raise ValueError('capsule witness override was not revalidated')
                cap = {'status': 'pass' if all(c['status'] == 'pass' for c in renewed['capsule_checks']) else 'fail',
                       'capsule_checks': renewed['capsule_checks'], 'failures': []}
                validation = renewed['native_floor_surface_validation']
                selected_indices = override['original_frozen_witness_indices']
            aligned = aligned_rooms[room['room_id']]
            failures = list(room['failures'])
            if cap['status'] != 'pass':
                failures.extend(cap.get('failures') or ['native_capsule_check_failed_or_incomplete'])
            if validation['status'] != 'pass':
                failures.append('native_spatial_floor_reference_failed_or_incomplete')
            if aligned['status'] != 'pass':
                failures.append('native_acoustic_support_geometry_alignment_failed_or_incomplete')
            passed = room['automated_room_status'] == 'pass' and cap['status'] == 'pass' and validation['status'] == 'pass' and aligned['status'] == 'pass'
            base = geometry_rooms[key]
            views = room['views']
            material = [v['native_geometry_material_checks'] for v in views if 'native_geometry_material_checks' in v]
            floor_ref = deepcopy(room['floor_reference'])
            floor_ref.update(production_support_method='per-point actual native LOD0 downward ray; median is diagnostic',
                             native_floor_surface=floors[house]['native_floor_surface'], spatial_validation=validation)
            row = {key: base[key] for key in ['house', 'room_label', 'room_type', 'floor_area_m2', 'short_side_m',
                                              'nav_main_area_m2', 'floor_height_m', 'dominant_floor_area_fraction']}
            row.update(render_path='ue_spear', room_training_admitted=passed,
                       native_room_status=room['automated_room_status'], capsule_status=cap['status'],
                       spatial_floor_status=validation['status'], acoustic_geometry_alignment_status=aligned['status'],
                       accepted_perspective_count=len(room['selected_witnesses']),
                       frozen_witness_indices=json.dumps(selected_indices),
                       max_accepted_black_fraction=scalar_max([v['metrics'] for v in views], 'black_fraction'),
                       max_tested_black_fraction=scalar_max([v['metrics'] for v in room['tested_views']], 'black_fraction'),
                       max_accepted_overexposed_fraction=scalar_max([v['metrics'] for v in views], 'overexposed_fraction'),
                       max_missing_solid_face_fraction=scalar_max(material, 'missing_solid_face_fraction'),
                       max_penetration_solid_fraction=scalar_max(material, 'penetration_solid_fraction'),
                       max_default_or_missing_texture_fraction=scalar_max(material, 'default_or_missing_texture_fraction'),
                       glass_affected=any(v.get('opaque_glass_fraction', 0) > 0 for v in material),
                       max_opaque_glass_fraction=scalar_max(material, 'opaque_glass_fraction'),
                       native_ue_floor_offset_m=floor_ref.get('native_ue_floor_offset_m'),
                       max_spatial_floor_depth_error_m=validation.get('max_native_depth_error_m'),
                       alignment_placement_count=aligned['placement_count'],
                       max_native_acoustic_distance_error_m=aligned['max_surface_distance_error_m'],
                       source_appearance_status='pass_original_native_highlights_only',
                       human_idle_walk_contact_status='pass_seven_registered_humans',
                       formal_generation_ready=False, formal_dataset_count=0,
                       reasons=';'.join(dict.fromkeys(failures)), qa_result=str(actual_qa_path))
            rows.append(row)
            measured[key] = {'room': room, 'floor_reference': floor_ref, 'capsule': cap,
                             'alignment': aligned, 'row': row, 'qa_result': str(actual_qa_path)}
    if len(rows) != 44 or len(house_details) != 21:
        raise ValueError('all 21 houses / 44 rooms must be represented')
    return measured, rows, house_details


def collect_sources(root):
    before = read(root/'ue_v1/native_asset_gallery_all_v1/result.json')
    current = read(root/'ue_round7/asset_gallery_original_v1/result.json')
    motion = read(root/'ue_round7/human_motion_contact_original_v1/result.json')
    restore = read(root/'ue_round7/evidence/original_ue_material_restore.json')
    if current.get('status') != 'completed' or current['asset_count'] != 59 or current['appearance_pass_count'] != 59:
        raise ValueError('original native appearance recheck incomplete')
    if motion['status'] != 'completed' or motion['passed_humans'] != 7:
        raise ValueError('registered human motion recheck incomplete')
    if restore['status'] != 'pass' or restore['restored_mic_count'] != 75:
        raise ValueError('native original MIC restoration incomplete')
    for row in restore['records']:
        if sha(row['path']) != row['restored_expected_sha256']:
            raise ValueError('restored original material changed: '+row['path'])
    before_by_id = {row['asset_id']: row for row in before['records']}
    rows = []
    for asset in current['records']:
        old = before_by_id[asset['asset_id']]
        metrics = [view['appearance_metric'] for view in asset['views']]
        if any(metric.get('color_checks_used') is not False for metric in metrics):
            raise ValueError('an inapplicable UE color gate was retained')
        old_failures = sorted({failure for view in old['views'] for failure in view['appearance_metric'].get('failures', [])})
        rows.append({'asset_id': asset['asset_id'], 'entity_class': asset['entity_class'],
                     'appearance_status': asset['status'], 'max_highlight_ratio': max(m['highlight_ratio'] for m in metrics),
                     'highlight_limit': metrics[0]['thresholds']['max_highlight_ratio'], 'native_view_count': len(asset['views']),
                     'UE_color_gate_used': False, 'old_inapplicable_color_gate_status': old['status'],
                     'old_inapplicable_failure_reasons': ';'.join(old_failures),
                     'original_material_slot_readback_status': asset['original_material_readback']['status'],
                     'final_failure_reasons': ';'.join(sorted({f for m in metrics for f in m.get('failures', [])}))})
    samples = [sample for human in motion['records'] for action in human['actions'] for sample in action['samples']]
    m2_summary = {'schema': motion['schema'], 'humans': 7, 'passed_humans': 7,
                  'registered_actions': ['idle', 'walk'], 'sample_count': len(samples),
                  'native_depth_view_count': sum(len(sample['views']) for sample in samples),
                  'min_foot_gap_m': min(s['gap_m'] for s in samples),
                  'max_foot_gap_m': max(s['gap_m'] for s in samples),
                  'max_native_phase_error_seconds': max(s['native_animation']['absolute_error_seconds'] for s in samples),
                  'machine_seconds': motion['machine_seconds'], 'contact_tolerance_m': motion['contact_tolerance_m'],
                  'method': motion['method'], 'coverage': motion['coverage'],
                  'original_materials': True, 'pose_offsets_or_foot_IK_used': False,
                  'source_result': str(root/'ue_round7/human_motion_contact_original_v1/result.json')}
    return current, rows, motion, m2_summary, restore


def stage_visual_evidence(generation):
    parent = next((generation/'parents').iterdir())
    sealed = parent/'delivery_completed.json'
    if sealed.is_file():
        entry = read(sealed)
        provenance = read(parent/entry['attempt']/'provenance_v0_a0.json')
        return provenance['native_visual_readback'], entry['result'], parent
    visual = next(parent.glob('visual_v0_attempt_*/native_visual_evidence.json'))
    return read(visual), read(generation/'result.json')['result'], parent


def collect_benchmarks(paths):
    rounds = [read(path) | {'result_path': str(path)} for path in paths]
    instances = [row['parallel_instances'] for row in rounds]
    if len(set(instances)) != len(instances) or any(n not in (1, 2, 4, 8) for n in instances):
        raise ValueError('valid measurements must be unique supported concurrency rounds')
    rows, trials = [], []
    for record in sorted(rounds, key=lambda row: row['parallel_instances']):
        if record['status'] != 'completed' or record['completed_scenes'] != record['parallel_instances']:
            raise ValueError('failed/stopped/withdrawn concurrency measurements cannot be admitted')
        jobs = record['jobs']
        if any(job['guard_status'] != 'pass' for job in jobs):
            raise ValueError('unprotected renderer measurement')
        samples_path = Path(record['result_path']).parent/'system_samples.jsonl'
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
        topology = record['topology_gpu_indices']
        gpu_measurements = [[float(value.strip()) for value in line.split(',')]
                            for sample in samples for line in sample['gpu'].splitlines()]
        used_gpu_measurements = [g for g in gpu_measurements if int(g[0]) in set(topology)]
        intervals = [right['elapsed_seconds']-left['elapsed_seconds'] for left, right in zip(samples, samples[1:])]
        row = {'parallel_instances': record['parallel_instances'],
               'gpu_topology': json.dumps(topology),
               'group_wall_seconds': record['group_machine_seconds'],
               'mean_scene_wall_seconds': statistics.mean(j['pilot_machine_seconds'] for j in jobs),
               'min_scene_wall_seconds': min(j['pilot_machine_seconds'] for j in jobs),
               'max_scene_wall_seconds': max(j['pilot_machine_seconds'] for j in jobs),
               'scenes_per_hour': record['scenes_per_hour'],
               'owned_CPU_occupied_cores_lower_bound': record['mean_cpu_occupied_cores_lower_bound'],
               'assigned_CPU_threads': record['parallel_instances']*8,
               'max_owned_process_tree_peak_mib': max(j['peak_mib'] for j in jobs),
               'mean_system_CPU_percent': statistics.mean(s['system_cpu_percent'] for s in samples),
               'peak_system_memory_used_bytes': max(s['memory']['used'] for s in samples),
               'swap_used_change_bytes': samples[-1]['swap']['used']-samples[0]['swap']['used'],
               'owned_IO_read_bytes_lower_bound': record['task_owned_io_lower_bound']['read_bytes'],
               'owned_IO_write_bytes_lower_bound': record['task_owned_io_lower_bound']['write_bytes'],
               'system_disk_read_bytes': samples[-1]['disk_io_system']['read_bytes']-samples[0]['disk_io_system']['read_bytes'],
               'system_disk_write_bytes': samples[-1]['disk_io_system']['write_bytes']-samples[0]['disk_io_system']['write_bytes'],
               'selected_GPU_mean_utilization_percent': statistics.mean(g[2] for g in used_gpu_measurements),
               'selected_GPU_max_utilization_percent': max(g[2] for g in used_gpu_measurements),
               'selected_GPU_mean_memory_utilization_percent': statistics.mean(g[3] for g in used_gpu_measurements),
               'telemetry_sample_count': len(samples),
               'telemetry_median_interval_seconds': statistics.median(intervals),
               'telemetry_max_interval_seconds': max(intervals),
               'measurement_scope': record['scope'], 'source': record['result_path']}
        rows.append(row)
        for job in jobs:
            generation = Path(job['pilot_result']).parent
            result = read(job['pilot_result'])
            visual, parent_receipt, parent = stage_visual_evidence(generation)
            if visual['native_source_material_status'] != 'pass' or visual['native_body_foot_status'] != 'pass':
                raise ValueError('original native source/foot check failed in measured full scene')
            windows = []
            for member in parent_receipt['members']:
                for window in member['windows']:
                    directory = Path(member['windows_directory'])/window['window_id']
                    core = [directory/name for name in ['video.mp4', 'audio_foa.wav', 'audio_mono.wav',
                                                        'questions.json', 'private_answers.json', 'evidence.json']]
                    if not all(path.is_file() for path in core):
                        raise ValueError('complete scene lacks a self-contained window')
                    questions = read(directory/'questions.json')
                    logical = len({q['logical_question_id'] for q in questions})
                    windows.append({'window_id': window['window_id'], 'directory': str(directory),
                                    'core_bytes': sum(path.stat().st_size for path in core),
                                    'logical_questions': logical, 'serialized_questions': len(questions),
                                    'files': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in core]})
            trials.append({'parallel_instances': record['parallel_instances'], 'job_index': job['job_index'],
                           'house': result['house'], 'room_id': result['room_id'],
                           'pilot_result': job['pilot_result'], 'guard_result': job['guard_result'],
                           'pilot_wall_seconds': result['elapsed_seconds'],
                           'visual_wall_seconds': visual['wall_seconds'],
                           'frame_count': visual['frame_count'],
                           'source_material_status': visual['native_source_material_status'],
                           'human_foot_status': visual['native_body_foot_status'],
                           'actor_frame_count': sum(len(f['actors']) for f in visual['frames']),
                           'failed_actor_frames': sum(a['native_foot_contact']['status'] != 'pass'
                                                     for f in visual['frames'] for a in f['actors']),
                           'maximum_root_error_m': max(a['root_max_error'] for f in visual['frames'] for a in f['actors']),
                           'peak_mib': job['peak_mib'],
                           'owned_CPU_seconds_lower_bound': job['owned_cpu_seconds_lower_bound'],
                           'windows': windows, 'blocking_reasons': result['blocking_reasons'],
                           'audio': [m['audio'] for m in parent_receipt['members']],
                           'parent': str(parent), 'measured_30s_research_replica_only': True})
    return rounds, rows, trials


def collect_renderer_measurements(root, rounds, incomplete_paths):
    attempts = [read(path) | {'result_path': str(path)} for path in incomplete_paths]
    measured = []
    for record in sorted(rounds+attempts, key=lambda row: row['parallel_instances']):
        visual_rows = []
        for job in record['jobs']:
            directory = Path(job['pilot_result']).parent
            path = next(directory.glob('parents/*/visual_*_attempt_*/native_visual_evidence.json'), None)
            if path is None and record['status'] == 'completed':
                visual, _, _ = stage_visual_evidence(directory)
            elif path is not None:
                visual = read(path)
            else:
                continue
            config = read(directory.parent/'house.json')['input']['ue_config']
            if not config.get('unpause_animation_sampling') or not config.get('original_material_restore_receipt'):
                raise ValueError('withdrawn native renderer profile entered the final measurements')
            if visual.get('frame_count') != 450 or visual.get('native_source_material_status') != 'pass' or visual.get('native_body_foot_status') != 'pass':
                continue
            visual_rows.append({'job_index': job['job_index'], 'visual_wall_seconds': visual['wall_seconds'],
                                'frame_count': visual['frame_count'], 'peak_mib': job['peak_mib'],
                                'source_material_status': visual['native_source_material_status'],
                                'human_foot_status': visual['native_body_foot_status'],
                                'source': str(path) if path else str(directory/'parents')})
        measured.append({'parallel_instances': record['parallel_instances'],
                         'completed_native_visual_segments': len(visual_rows),
                         'requested_instances': record['parallel_instances'],
                         'full_generation_status': record['status'],
                         'full_generation_completed_scenes': record.get('completed_scenes', 0),
                         'gpu_topology': record['topology_gpu_indices'],
                         'mean_visual_wall_seconds': statistics.mean(v['visual_wall_seconds'] for v in visual_rows) if visual_rows else None,
                         'max_visual_wall_seconds': max((v['visual_wall_seconds'] for v in visual_rows), default=None),
                         'owned_CPU_occupied_cores_lower_bound_for_whole_attempt': record.get('mean_cpu_occupied_cores_lower_bound'),
                         'peak_mib': max((j['peak_mib'] for j in record['jobs']), default=0),
                         'scope': 'native visual segment only; interrupted full pipelines are not successful scene-throughput measurements',
                         'source': record['result_path'], 'visual_jobs': visual_rows,
                         'interruption_reason': 'external_audio_disk_reserve' if record['status'] != 'completed' else None})
    write(root/'ue_round7/evidence/final_renderer_measurements_v1.json', {'records': measured,
          'full_generation_measurement_complete': sorted(r['parallel_instances'] for r in rounds) == [1, 2, 4, 8]})
    return measured


def write_review(output, room_rows, measured, gallery, summary, comparisons):
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>酷家乐原生UE训练房源准入v1</title>',
            '<style>body{font:14px system-ui;margin:20px;background:#fafafa;color:#222}'
            '.room{display:grid;grid-template-columns:230px repeat(4,300px);gap:10px;padding:12px 0;border-top:1px solid #bbb;overflow:auto}'
            '.asset{display:grid;grid-template-columns:260px repeat(3,300px);gap:10px;padding:10px 0;border-top:1px solid #ccc;overflow:auto}'
            '.compare{display:grid;grid-template-columns:200px repeat(2,450px);gap:10px;padding:12px 0;overflow:auto}'
            'figure{margin:0}img{width:100%;height:auto}figcaption{font-size:12px}.pass{color:#146533}.fail{color:#a22}code{overflow-wrap:anywhere}</style>',
            f'<h1>UE房源准入v1：{summary["admitted_house_count"]}套、{summary["admitted_room_count"]}间</h1>',
            '<p>21套／44间全部实渲检查；2560×1440→1280×720，85°，固定EV3、4000K、Lumen。'
            '每间一行：俯视图及3个原冻结摆位视角。失败房保留实际失败图，空白格表示未得到该合格视角。'
            '正式v7生成协议仍有独立阻断，正式数据量为0。图片全部内嵌，可离线打开；未发送owner。</p>']
    for row in room_rows:
        key = (row['house'], row['room_label'])
        room = measured[key]['room']
        status = 'pass' if row['room_training_admitted'] else 'fail'
        page.append('<div class="room" data-room="'+html.escape('/'.join(key))+'">')
        page.append('<div><b>'+html.escape('/'.join(key)+' '+row['room_type'])+'</b><p class="'+status+'">'+
                    ('房源准入' if status == 'pass' else '拒绝')+'</p><p>'+html.escape(row['reasons'] or '全部房源检查通过')+
                    '</p><p>面积 '+f'{row["floor_area_m2"]:.2f}'+' m²；可走 '+f'{row["nav_main_area_m2"]:.2f}'+
                    ' m²</p><p>原候选索引 '+html.escape(row['frozen_witness_indices'])+'</p></div>')
        views = room['views'] if status == 'pass' else room['tested_views'][:4]
        for view in views[:4]:
            metrics = view['metrics']
            page.append('<figure>'+inline_image(view['path_720p'])+'<figcaption>'+html.escape(view['view']['view_id'])+
                        f' · 黑{metrics["black_fraction"]:.2%} · 过曝{metrics["overexposed_fraction"]:.2%} · '+
                        html.escape(metrics['status'])+'</figcaption></figure>')
        page.extend(['<div>没有该合格实渲视角</div>']*(4-len(views[:4])))
        page.append('</div>')
    page.append('<h1>59个原UE资产：仅按原高光门槛重判，59通过</h1><p>不使用贴图sRGB色差／色度门槛；'
                'BaseColor只用于原高光检测的亮度基准，未重新校准颜色门槛。原材质槽逐个读回。</p>')
    for asset in gallery['records']:
        page.append('<div class="asset" data-asset="'+html.escape(asset['asset_id'])+'"><div><b>'+html.escape(asset['asset_id'])+
                    '</b><p class="pass">'+html.escape(asset['status'])+'</p></div>')
        for view in asset['views']:
            page.append('<figure>'+inline_image(view['lit_path'])+'<figcaption>原材质 · 高光 '+
                        f'{view["appearance_metric"]["highlight_ratio"]:.2%}'+'</figcaption></figure>')
        page.append('</div>')
    page.append('<h1>撤回前后：两套完整试跑的原生画面对照</h1><p>左为本任务已撤回的派生材质试验，'
                '右为恢复原MIC、无覆盖／无额外补光的同房试跑。世界动作采样修正也有变化，'
                '因此这组是状态对照，不伪称逐像素相等。恢复依据为75个原uasset的SHA和59个原材质槽读回。</p>')
    for row in comparisons:
        page.append('<div class="compare"><div>'+html.escape(row['house']+' '+row['frame'])+'</div>')
        for name in ('before_image', 'after_image'):
            page.append('<figure>'+inline_image(row[name])+'<figcaption>'+html.escape('已撤回' if name == 'before_image' else '原UE材质恢复后')+'</figcaption></figure>')
        page.append('</div>')
    page.append('</html>')
    content = '\n'.join(page)
    if content.count('data-room=') != 44 or content.count('data-asset=') != 59:
        raise ValueError('offline review row count differs')
    (output/'ACCEPTANCE_ue_v1.html').write_text(content)


def extract_formal_blockers(repository):
    path = repository/'src/avengine/timeline/v7_candidate_generation.py'
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FORMAL_BLOCKERS' for t in node.targets):
            return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
                    'source_path': str(path), 'blockers': list(ast.literal_eval(node.value))}
    raise ValueError('formal generator blocker declaration not found')


def material_comparisons(root, output, trials):
    directory = output/'material_restore_comparison'
    directory.mkdir()
    rows = []
    for house in ['kujiale_0022', 'kujiale_0030']:
        old = root/'ue_v1/generation'/(house+'_full_pilot_v3')
        before = next(old.glob('parents/*/visual_v0_attempt_*/scene_video.mp4'), None)
        trial = next(t for t in trials if t['house'] == house)
        parent = Path(trial['parent'])
        after = next(parent.glob('visual_v0_attempt_*/scene_video.mp4'), None)
        if after is None:
            compact_records = [row for path in (root/'ue_round7/evidence').glob('completed_pilot_compaction*/result.json')
                               for row in read(path)['records'] if row.get('witness')]
            retained = next(row for row in compact_records if row['parent'] == str(parent))
            after = next(Path(w['retained']) for w in retained['witness'] if w['source'].endswith('/scene_video.mp4'))
        if before is None or not before.is_file() or not after.is_file():
            raise ValueError('before/after material movie witness is unavailable')
        for frame, seconds in [('first', 0), ('middle', 15), ('last', 29.9)]:
            images = {}
            for key, video in [('before_image', before), ('after_image', after)]:
                image = directory/(house+'_'+frame+'_'+key+'.png')
                subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-ss', str(seconds), '-i', str(video),
                                '-frames:v', '1', '-threads', '2', str(image)], check=True)
                images[key] = str(image)
            rows.append({'house': house, 'frame': frame, **images, 'before_video': str(before), 'after_video': str(after),
                         'same_world_geometry': True, 'same_camera_and_lighting_settings': True,
                         'identical_animation_sampling_between_historical_versions': False,
                         'byte_identical_to_owner_v6_movie_verified': False,
                         'restoration_authority': '75 original MIC hashes and live original material slots, not a new color threshold'})
    return rows


def write_generation_inputs(root, output, measured, house_details, admitted_houses, gallery):
    destination = output/'generation_inputs'
    destination.mkdir()
    template = read(root/'ue_round7/generation_inputs/kujiale_0030_original_v1/house.json')
    asset_ids = [row['asset_id'] for row in gallery['records']]
    for entry in admitted_houses:
        house = entry['house']
        detail = house_details[house]
        directory = destination/house
        directory.mkdir()
        plan = deepcopy(detail['plan'])
        plan['rooms'] = [row for row in plan['rooms'] if row['room_id'] in entry['room_ids']]
        checked = []
        for room_plan in plan['rooms']:
            item = measured[(house, room_plan['room_id'])]
            selected = item['room']['selected_witnesses']
            room_plan['witnesses'] = [w['placement'] for w in selected]
            room_plan['views'] = [v['view'] for v in item['room']['views']]
            floor = item['floor_reference']
            surface = floor['native_floor_surface']
            audit = read(surface['native_editor_audit'])
            geometry = audit['native_render_geometry']
            for key, path in [('native_editor_audit_sha256', surface['native_editor_audit']),
                              ('native_vertices_sha256', geometry['world_vertices_ue_cm_f32']),
                              ('native_triangles_sha256', geometry['triangles_u32'])]:
                if sha(path) != surface[key]:
                    raise ValueError('declared measured floor geometry changed before delivery')
            checked.append({'room_label': room_plan['room_id'], 'room_type': room_plan['room_type'],
                            'placement': selected[0]['placement'], 'native_witnesses': selected,
                            'floor_reference': floor, 'native_qa_status': 'pass'})
        plan_path = directory/'acceptance_plan.json'
        write(plan_path, plan)
        ue = deepcopy(template['input']['ue_config'])
        ue.update(acceptance_plan_path=str(plan_path), native_floor_surface=detail['native_floor_surface'],
                  source_asset_allowlist=asset_ids, native_source_material_bindings=str(root/'ue_round7/evidence/original_source_material_bindings.json'),
                  original_material_restore_receipt=str(root/'ue_round7/evidence/original_ue_material_restore.json'))
        build = Path(plan['build_directory'])
        config = {'schema': 'v7-admitted-physical-native-house-inputs-v1',
                  'house': house, 'family': 'kujiale', 'split': 'train', 'known_exposure': True,
                  'checked_room_ids': entry['room_ids'], 'checked_rooms': checked,
                  'acoustic_status': 'pass', 'native_rgb_status': 'automated_native_ue_pass',
                  'native_qa_result': detail['qa_result'],
                  'acoustic_package_manifest': str(build/read(build/'engine_asset_index.json')['acoustic_package']),
                  'input': {'house': house, 'family': 'kujiale', 'visual_backend': 'ue_spear',
                            'ue_map_path': plan['map_path'], 'ue_config': ue,
                            'navmesh_source': str(build/'navigation/navigation.navmesh'),
                            'semantic_source': str(build/'rooms_screened.json')},
                  'formal_dataset_count': 0, 'formal_generation_ready': False,
                  'qualification_scope': 'physical training room; seed-specific native layout/motion and full v7 content admission remain mandatory'}
        if house in ('kujiale_0022', 'kujiale_0030'):
            measured_input = read(root/'ue_round7/generation_inputs'/(house+'_original_v1')/'house.json')
            for current in checked:
                prior = next((r for r in measured_input['checked_rooms'] if r['room_label'] == current['room_label']), None)
                if prior:
                    current['placement'] = prior['placement']
                    current['placement_scope'] = 'retained tested original-material 30s research seed layout; native floor is sampled per point'
        if not Path(config['input']['semantic_source']).is_file():
            raise ValueError('operational room semantics path missing')
        write(directory/'house.json', config)
    write(output/'source_asset_allowlist.json', {'schema': 'kujiale_native_source_appearance_allowlist_v1',
                                               'asset_ids': asset_ids, 'color_gate_used': False,
                                               'appearance_result': str(root/'ue_round7/asset_gallery_original_v1/result.json'),
                                               'motion_scope': 'appearance does not grant arbitrary new clips or asset-size layouts'})


def write_planning(root, output, summary, trials, house_rows):
    fastest = max(summary['throughput'], key=lambda row: row['scenes_per_hour'])
    rate = fastest['scenes_per_hour']
    windows = [w for trial in trials for w in trial['windows']]
    window_bytes = statistics.mean(w['core_bytes'] for w in windows)
    cpu_seconds = statistics.mean(t['owned_CPU_seconds_lower_bound'] for t in trials)
    per_member_bytes = statistics.mean(sum(w['core_bytes'] for w in t['windows']) for t in trials)
    proposals = []
    for parents in [2300, 4800]:
        members = parents*1.6
        media = round(members*per_member_bytes)
        proposals.append({'independent_parents': parents, 'assumed_member_factor': 1.6,
                          'member_scenes': members, 'legacy_two_10s_windows': round(members*2),
                          'legacy_14_questions_per_window': round(members*2*14),
                          'legacy_12_questions_per_primary_window': round(members*12),
                          '30s_two_human_machine_hours_proxy': members/rate,
                          '30s_two_human_machine_hours_proxy_with_25_percent_margin': members/rate*1.25,
                          'owned_CPU_hours_lower_bound_proxy': members*cpu_seconds/3600,
                          'core_media_bytes_proxy': media,
                          'transfer_hours_at_user_measured_1_7_MBps': media/1700000/3600,
                          'required_member_scenes_per_hour_for_36h': members/36,
                          'scope': 'V7_SPEC design-only Kujiale 2300..4800 extra-parent range; not a frozen allocated Kujiale quota, not measured full multi-class v4/v5 throughput'})
    planning = {'location': '48g native UE/SPEAR generation; CCI combines final self-contained media and question files',
                'batch_generation_executed': False, 'formal_dataset_count': 0,
                'admitted_room_count': summary['admitted_room_count'], 'rejected_rooms_excluded': True,
                'native_concurrency_best_measured_scenes_per_hour': rate,
                'best_measured_instances': fastest['parallel_instances'], 'best_gpu_topology': fastest['gpu_topology'],
                'mean_core_window_bytes': window_bytes, 'mean_two_window_core_bytes_per_research_member': per_member_bytes,
                'mean_owned_CPU_seconds_lower_bound_per_research_member': cpu_seconds,
                'conditional_design_range': proposals,
                'V7_Kujiale_frozen_seed_allocation_present': False,
                'full_v7_native_layout_dry_program_and_qa_families_admitted': False,
                '36_hour_formal_generation_verified': False,
                'transfer_rate_source': 'user round5: existing 48g -> CCI measurement about 1.7 MB/s, not remeasured in this task',
                'transfer_tool': '/home/neos/jzy/tools/cci_pull_from_48g.sh',
                'transfer_performed': False,
                'storage_policy': 'task <=80GB; shared /data reserve >=150GB. Bulk media exceeds remaining task capacity; needs assigned storage or explicit authorization to release verified local delivered copies. No deletion of delivered media is inferred.',
                'capacity_scope': 'only frozen three-position two-source room feasibility is measured; 0..8-actor/full class capacity and independent seed diversity are not established'}
    write(output/'generation_and_transfer_plan.json', planning)
    per_house = []
    for house in house_rows:
        count = house['admitted_room_count']
        per_house.append({'house': house['house'], 'admitted_rooms': count,
                          'conditional_room_proportional_parent_plan_min': count*2300/summary['admitted_room_count'],
                          'conditional_room_proportional_parent_plan_max': count*4800/summary['admitted_room_count'],
                          'CPU_hours_lower_bound_proxy_min': count*2300/summary['admitted_room_count']*1.6*cpu_seconds/3600,
                          'CPU_hours_lower_bound_proxy_max': count*4800/summary['admitted_room_count']*1.6*cpu_seconds/3600,
                          'scope': 'unallocated design-only arithmetic; not measured room capacity or authorized batch generation'})
    write_csv(output/'per_house_generation_plan.csv', per_house)
    return planning


def write_engine_manifests(root, output, house_details, admitted_houses):
    files, seen = [], set()
    for row in admitted_houses:
        house = row['house']
        plan = house_details[house]['plan']
        build = Path(plan['build_directory'])
        index = read(build/'engine_asset_index.json')
        import_result = read(Path(house_details[house]['directory'])/'import_result.json')
        native_import = root/'ue_v1/runtime/SpearSim/Content'/import_result['destination_path'].removeprefix('/Game/')
        if not native_import.is_dir():
            raise ValueError('measured native imported Content missing')
        roots = [build/'engine_expanded', build/'surface', build/'navigation', (build/index['acoustic_package']).parent, native_import]
        map_file = root/'ue_v1/runtime/SpearSim/Content'/(plan['map_path'].removeprefix('/Game/')+'.umap')
        paths = [f for directory in roots if directory.is_dir() for f in directory.rglob('*') if f.is_file()]
        paths += [map_file, build/'engine_asset_index.json', build/'rooms_screened.json', build/'semantic_rooms.json',
                  Path(house_details[house]['native_floor_surface']['native_editor_audit'])]
        audit = read(paths[-1])
        paths += [Path(audit['native_render_geometry'][key]) for key in ['world_vertices_ue_cm_f32', 'triangles_u32']]
        for path in sorted(set(paths)):
            if path in seen:
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError('final engine asset missing or symbolic: '+str(path))
            if root not in path.resolve().parents:
                raise ValueError('raw/shared file accidentally entered derived engine manifest')
            seen.add(path)
            files.append({'house': house, 'path': str(path), 'bytes': path.stat().st_size,
                          'sha256': sha(path), 'raw_interioragent_package': False,
                          'location': '48g only; native generation assets are not sent to CCI in this route'})
    write(output/'NATIVE_DERIVED_ENGINE_ASSETS.json', {'files': files, 'bytes': sum(f['bytes'] for f in files),
                                                     'transfer_performed': False, 'raw_package_included': False,
                                                     'license': 'noncommercial internal research, attribution, no redistribution'})
    (output/'NATIVE_DERIVED_ENGINE_ASSETS.sha256').write_text(''.join(f['sha256']+'  '+f['path']+'\n' for f in files))
    source_content = root/'ue_v1/runtime/SpearSim/Content/MyAssets'
    source_files = []
    for path in sorted(source_content.rglob('*')):
        if not path.is_file() or path.suffix not in ('.uasset', '.umap', '.ubulk', '.uexp'):
            continue
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError('source Content closure escaped the isolated project')
        source_files.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    write(output/'NATIVE_SOURCE_CONTENT_DEPENDENCIES.json', {
        'files': source_files, 'bytes': sum(f['bytes'] for f in source_files),
        'scope': 'conservative isolated source Content library; only 59 allowlisted IDs have appearance admission',
        'original_materials_restored': True, 'transfer_performed': False,
        'location': '48g native runtime only; not part of the CCI media transfer'})
    # Retain the original converted 25-house transfer proposal as history.
    # The final route transfers final media; no original USD/MDL/HF package,
    # UE installation, source asset Content library or DDC enters that bundle.
    write(output/'CCI_DERIVED_ASSET_ROUTE.json', {'render_path': 'ue_spear', 'generation_host': '48g',
                                                'native_engine_assets': str(output/'NATIVE_DERIVED_ENGINE_ASSETS.json'),
                                                'CCI_payload': 'only self-contained final windows, questions, private answers and compact provenance',
                                                'original_package_included': False, 'UE_installation_or_DDC_included': False,
                                                'transfer_performed': False})


def write_media_transfer_manifest(root, output, trials):
    files, examples = [], []
    for house in ('kujiale_0022', 'kujiale_0030'):
        trial = next(t for t in trials if t['house'] == house)
        parent = Path(trial['parent'])
        sealed = read(parent/'delivery_completed.json')
        delivery = parent/sealed['attempt']
        for relative in sealed['required_files']:
            path = delivery/relative
            if not path.is_file() or path.is_symlink() or root not in path.resolve().parents:
                raise ValueError('media transfer contains a missing file or root escape')
            files.append({'house': house, 'scene_id': parent.name, 'path': str(path),
                          'relative_to_task_root': str(path.relative_to(root)),
                          'bytes': path.stat().st_size, 'sha256': sha(path),
                          'research_only': True, 'formal_training_admitted': False})
        examples.append({'house': house, 'scene_id': parent.name, 'parent_receipt': str(parent/'delivery_completed.json'),
                         'window_count': len(trial['windows']), 'formal_training_admitted': False})
    write(output/'CCI_GENERATED_WINDOWS_AND_PROVENANCE.json', {
        'files': files, 'bytes': sum(f['bytes'] for f in files), 'examples': examples,
        'source_rate_MBps': 1.7, 'source_rate_origin': 'user round5 measurement; not remeasured here',
        'estimated_transfer_seconds': sum(f['bytes'] for f in files)/1700000,
        'raw_interioragent_package_included': False, 'native_engine_content_included': False,
        'transfer_performed': False, 'formal_dataset_count': 0})
    (output/'CCI_GENERATED_WINDOWS_AND_PROVENANCE.sha256').write_text(
        ''.join(f['sha256']+'  '+f['path']+'\n' for f in files))
    (output/'CCI_GENERATED_WINDOWS_AND_PROVENANCE.list').write_text(''.join(f['path']+'\n' for f in files))
    write(output/'CCI_PATH_MAP.json', {
        'source_prefix': str(root),
        'planned_CCI_prefix': '/mnt/afs/xinyuanqian/jzy/hardbank/mirror'+str(root),
        'purpose': 'preserve absolute historical provenance via mirrored pull; window public-media paths remain relative',
        'window_questions_and_answers_modified': False, 'transfer_performed': False,
        'source_plan': str(root/'ue_v1/delivery_v2/CCI_PATH_MAP.json')})


def storage_snapshot(root, output):
    locations = [root, Path(__file__).resolve().parents[2]]
    inode_seen, rows = set(), []
    for directory in locations:
        logical, unique_logical, allocated, count = 0, 0, 0, 0
        for current, subdirs, names in os.walk(directory, followlinks=False):
            subdirs[:] = [n for n in subdirs if not (Path(current)/n).is_symlink()]
            for name in names:
                path = Path(current)/name
                if path.is_symlink():
                    continue
                stat = path.stat()
                count += 1
                logical += stat.st_size
                identity = (stat.st_dev, stat.st_ino)
                if identity not in inode_seen:
                    inode_seen.add(identity)
                    unique_logical += stat.st_size
                    allocated += stat.st_blocks*512
        rows.append({'path': str(directory), 'file_count': count,
                     'logical_path_bytes': logical, 'unique_inode_logical_bytes': unique_logical,
                     'unique_inode_allocated_bytes': allocated})
    snapshot = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'locations': rows, 'unique_inode_allocated_bytes': sum(r['unique_inode_allocated_bytes'] for r in rows),
                'logical_path_bytes': sum(r['logical_path_bytes'] for r in rows),
                'limit_bytes': 80000000000, 'shared_data_free_bytes': shutil.disk_usage(root).free,
                'shared_data_reserved_bytes': 150000000000,
                'original_HF_package_or_other_checkouts_included': False,
                'method': 'task output and own worktree only; do not follow links; count allocation once per device/inode'}
    if snapshot['unique_inode_allocated_bytes'] > snapshot['limit_bytes']:
        raise ValueError('own derived task storage exceeds 80GB')
    write(output/'STORAGE_SNAPSHOT.json', snapshot)
    return snapshot


def write_report(root, output, summary, house_rows, asset_rows, motion, planning, storage):
    admitted = summary['admitted_room_count']
    count = summary['admitted_house_count']
    native_seconds = [row['native_import_to_acceptance_machine_seconds'] for row in house_rows
                      if row['native_import_to_acceptance_machine_seconds'] is not None]
    lines = [f'25套中有{count}套、{admitted}间完成UE/SPEAR训练房源准入；21套44间均已检查，'
             '59个插入资产恢复原材质后全部通过，正式v7数据尚未生成。', '',
             f'原生导入到房间验收实测每套{min(native_seconds)/60:.2f}–{max(native_seconds)/60:.2f}分钟'
             f'（中位{statistics.median(native_seconds)/60:.2f}分钟，已热DDC）；当前检查没有人工看图步骤。'
             '不合格房间保持拒绝。完整v7多类别动作/录音程序/题型族的生成准入仍由对应生成器验证，'
             '这里没有修改判分器、真值、声学或画面门槛，也没有启动配额批量生成。', '',
             '数据来源：本目录的 house_results.csv、room_results.csv、native_source_assets.csv、'
             'native_M2_contact.json、concurrency_throughput.csv；底层见证路径在相应JSON/CSV中。', '',
             '## 逐套盘点与原生房源验收', '',
             '|房子|原语义房间数|语义面积合计m²|声学|UE候选/准入|导入至验收分钟|结论/原因|',
             '|---|---:|---:|---|---:|---:|---|']
    for row in house_rows:
        seconds = row['native_import_to_acceptance_machine_seconds']
        note = '房源准入；仅列出的房间可用' if row['training_admitted'] else row['reasons']
        lines.append('|'+row['house'].removeprefix('kujiale_')+'|'+str(row['metadata_room_count'])+'|'+
                     f'{row["metadata_area_sum_m2"]:.2f}'+'|'+row.get('native_status', '')+'|'+
                     str(row['native_candidate_rooms'])+'/'+str(row['admitted_room_count'])+'|'+
                     (f'{seconds/60:.2f}' if seconds is not None else '未跑')+'|'+note.replace('|', '/')+'|')
    lines += ['', '房型统计、测得的地面高度簇与面积明细在 house_results.csv；高度簇不能直接当建筑楼层数。'
              '原始USD的虚拟UE/MDL引用未闭合，25套的 usd_complete 均为 false；没有宣称原包完整。'
              '可运行的派生USD、原生地图、语义、导航及声学包分别校验并列入派生资产清单。', '',
              'CPU转换沿用第3回合185537b结果；逐套原转换时长来自 houses/<house>/<build>/conversion_result.json。'
              '本次原生机时来自 ue_round6/houses/<house>/result.json，0009使用'
              ' ue_round6/recovered_houses/kujiale_0009_fresh_v2/result.json。'
              '0003/0065 R0另有冻结候选重新摆位复验，来源 ue_round6/native_room_overrides.json；'
              '未修改房间几何或冻结摆位规则。', '',
              '## 原材质恢复与59资产重判', '',
              '隔离项目中75个MIC已恢复到原备份的相同SHA；32张线性MR及派生MIC移到'
              ' ue_round7/withdrawn_material_assets，不再参与运行。原始资产、共享Content未改。'
              '依据 ue_round7/evidence/original_ue_material_restore.json，59个资产实际材质槽'
              '读回均匹配原路径，见 ue_round7/asset_gallery_original_v1/result.json。', '',
              'UE只保留原高光比例门槛：毛皮2%、衣服4%、普通道具8%、金属12%；使用同一UE BaseColor'
              '亮度基准和原高光判定式，没有放宽阈值。贴图sRGB色差/色度项不用于UE，也没有启用新的颜色门槛或校准。'
              'Lumen、EV3、4000K和阴影会使Lit图与贴图颜色不同；旧30个失败不能据此判定UE外观异常。', '',
              f'59/59通过，三个视角各实际渲染，最高高光比例{summary["max_source_highlight_ratio"]:.4%}；'
              '逐个结果见下表和 native_source_assets.csv。', '',
              '|资产ID|UE高光最大值|保留门槛|结果|旧不适用失败原因|',
              '|---|---:|---:|---|---|']
    for row in asset_rows:
        lines.append('|'+row['asset_id']+'|'+f'{row["max_highlight_ratio"]:.4%}'+'|'+
                     f'{row["highlight_limit"]:.0%}'+'|'+row['appearance_status']+'|'+
                     (row['old_inapplicable_failure_reasons'] or '原检查已过')+'|')
    lines += ['', '0022、0030均重新跑完整30秒原生流程。撤回前后各抽0/15/29.9秒画面，'
              '见 material_restore_comparison.json 和离线页；恢复后无派生覆盖、无额外源资产补光。'
              '历史版本同时修正了动画采样，因此对照用于查看撤回前后的状态；没有宣称与owner旧v6成片逐像素相等。'
              '恢复原样的可核验证据是75个原uasset SHA和59个实际材质槽。', '',
              '## 44间房间、地板、对齐及M2', '',
              '全部21套44间完成原生1440p→720p俯视图和冻结候选视角及材质/缺面/穿模/玻璃检查；'
              '视觉候选再做低相机深度地板探测、胶囊和实际LOD0断言，提前拒绝的房间保留未运行状态。'
              '至少3个合格摆位视角才准入；'
              f'{admitted}间通过、{44-admitted}间拒绝，逐间原因在 room_results.csv 和离线页。'
              '不够3个视角的房间保持拒绝；未补洞、拟合坐标或改门槛。', '',
              '每个合格房间9个低相机地板样本，与实际UE LOD0逐点向下射线对比；'
              '生产 floor_reference 使用逐点地板，常量中位偏移只作诊断。0034斜面常量基准会产生'
              '约3.89cm偏差，逐点实测处理后按原2cm规则验收。'
              '来源 ue_round6/native_floor_spatial_v1/result.json 和各房间 floor_reference。'
              '0.20m半径、0.75m半高胶囊沿可走面检查；0003/0065只重新挑原冻结候选并实渲复验。', '',
              '门洞/相机/两声源对齐：读取engine_asset_index声明的最终声学包，与实际原生LOD0在'
              '在有3个摆位的房间按3个候选摆位×3个角色×26方向比较，原0.5mm容差不变；'
              '来源 ue_round7/native_acoustic_alignment_v2/result.json。视觉失败房的几何探测不会使它自动准入。', '',
              f'M2：7/7人物，注册idle/walk全片15FPS及生产30秒相位覆盖，{summary["M2"]["sample_count"]}个实际姿态、'
              f'{summary["M2"]["native_depth_view_count"]}个前后深度视角，脚底相对实测地板范围'
              f'{summary["M2"]["min_foot_gap_m"]*100:.3f}至{summary["M2"]["max_foot_gap_m"]*100:.3f}cm。'
              '原2cm门槛全部通过；无根节点补偏、无逐帧脚IK。实际解除世界暂停并给姿态2个原生帧求值，'
              '动画GetPosition最大相位误差见native_M2_contact.json。该结论覆盖已注册idle/walk；其他新动作尚未验证。', '',
              '|人物|idle姿态数|walk姿态数|最大绝对脚底偏差cm|结果|',
              '|---|---:|---:|---:|---|']
    for human in motion['records']:
        actions = {action['action_id']: action for action in human['actions']}
        gap = max(abs(s['gap_m']) for a in human['actions'] for s in a['samples'])
        lines.append('|'+human['asset_id']+'|'+str(actions['idle']['sample_count'])+'|'+
                     str(actions['walk']['sample_count'])+'|'+f'{gap*100:.3f}'+'|'+human['status']+'|')
    lines += ['', '## 完整场景及1/2/4/8并行实测', '',
              '每个实例实际执行原v3研究管线：30秒、450帧、两人、两段10秒窗口、RLR声音、出题、'
              '窗口核心文件齐全；按原材质/动作检查后才计数。RIR为每任务冷缓存，DDC已热。'
              f'重复内容只有0022与0030两个固定房间/种子世界，不能把{summary["full_scene_research_replicate_count"]}次'
              '已完成吞吐复跑算成同样数量的独立训练场景。', '',
              '|并行实例|实际GPU索引|整批墙钟分钟|每场景墙钟均值分钟|场景/小时|占用CPU核数下界|每进程树峰值MiB|',
              '|---:|---|---:|---:|---:|---:|---:|']
    for row in summary['throughput']:
        lines.append('|'+str(row['parallel_instances'])+'|'+row['gpu_topology']+'|'+
                     f'{row["group_wall_seconds"]/60:.2f}'+'|'+f'{row["mean_scene_wall_seconds"]/60:.2f}'+'|'+
                     f'{row["scenes_per_hour"]:.3f}'+'|'+f'{row["owned_CPU_occupied_cores_lower_bound"]:.2f}'+'|'+
                     f'{row["max_owned_process_tree_peak_mib"]:.0f}'+'|')
    lines += ['', '完整场景只完成1与8实例测量；2与4实例由于共享/data空闲跌破150GB被自己的守护器停止，'
              '未将中断时间当作完整场景吞吐。尚缺2/4有效全链路结果，需要恢复相应空间余量后在新输出版本重测。', '',
              '|并行实例|完成450帧UE渲染段数|渲染段每场景均值分钟|完整生成状态|峰值MiB|',
              '|---:|---:|---:|---|---:|']
    for row in summary['renderer_measurements']:
        seconds = row['mean_visual_wall_seconds']
        lines.append('|'+str(row['parallel_instances'])+'|'+str(row['completed_native_visual_segments'])+'|'+
                     (f'{seconds/60:.2f}' if seconds is not None else '未完成')+'|'+row['full_generation_status']+'|'+f'{row["peak_mib"]:.0f}'+'|')
    lines += ['', '渲染段来自每个实际native_visual_evidence.json，均通过原材质和脚底检查；'
              '若渲染段未完成，表中照实列0。独立记录见native_renderer_measurements.json；'
              '中断尝试的CPU占用只代表中断前全尝试，不冒充完整场景CPU用量。', '',
              '表中完整场景时间从资源预检完成后的实际整批启动算起，包括UE启动、渲染、声音和出题；'
              '整批协调器从05:42启动到8实例06:08正式开工另有约26分钟预检/连接延迟，原因未验证，未从运行记录删除。'
              '首次全新着色器冷DDC时间不包含在这组重复吞吐内。显存以nvidia-smi -q -x的图形/计算进程树统计，'
              '目标1秒采一次；真实采样间隔和最长空档列在concurrency_throughput.csv，不把查询延迟伪称严格1Hz。'
              '每卡启动空闲须足够“同卡实例数×已测峰值+4GiB”；只使用48g，未抢占其他作业。', '',
              '原材质有效完整结果来源：ue_round7/concurrency_original_v2/n1/result.json、'
              ' ue_round7/concurrency_original_v3/n8/result.json。中断记录为v2/n2和v3/n4。'
              '旧暂停动画/PBR覆盖的吞吐结果已作废；未得到有效的原材质2/4完整结果。', '',
              '瓶颈判断依据与限制：逐场景visual_wall_seconds、实际CPU核数下界、GPU利用率、系统磁盘与本进程IO'
              '分别在native_full_scene_measurements.json和concurrency_throughput.csv。'
              '这条流程每帧有UE求值、RGB与深度读回及RPC等待；现有统计能定位到原生渲染流程，'
              '不能把所有等待精确归为某个GPU内核、CPU线程或磁盘延迟。共享GPU/磁盘总利用率包含他人负载，'
              '没有把它算作本任务独占占用。', '',
              '## 生成、存储与传输方案', '',
              f'{count}个房源已有自己的UE地图和generation_inputs；只有准入房间进入'
              'train_houses.json，render_path=ue_spear。生成在48g，CCI只接收自包含媒体、题目、私有答案和压缩见证；'
              '不传原InteriorAgent包、不传UE项目/DDC。现有30秒研究流程支持UE，但最新版完整v7多类别原生布局、'
              '录音dry程序及题型族尚未闭合；FORMAL_BLOCKERS和读取的生成器提交见summary.json。'
              '没有把对方已经通过的80/80声学校准误报为失败；本任务旧研究渲染路径也不能替代其正式协议。', '',
              '当前V7_SPEC的Kujiale额外2300–4800独立父场景是设计区间，不是已冻结并分到这批房间的种子配额。'
              '下面仅按1.6成员/父场景和本次两人30秒流程估算；实际0–8演员、多类别v4/v5吞吐未验证。', '',
              '|设计父场景|假定成员场景|实测吞吐推算小时|加25%余量小时|自产核心媒体GB|按1.7MB/s传输小时|36小时所需场景/小时|',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for proposal in planning['conditional_design_range']:
        lines.append('|'+str(proposal['independent_parents'])+'|'+str(round(proposal['member_scenes']))+'|'+
                     f'{proposal["30s_two_human_machine_hours_proxy"]:.1f}'+'|'+
                     f'{proposal["30s_two_human_machine_hours_proxy_with_25_percent_margin"]:.1f}'+'|'+
                     f'{proposal["core_media_bytes_proxy"]/1e9:.2f}'+'|'+
                     f'{proposal["transfer_hours_at_user_measured_1_7_MBps"]:.2f}'+'|'+
                     f'{proposal["required_member_scenes_per_hour_for_36h"]:.1f}'+'|')
    lines += ['', f'实测最好吞吐{planning["native_concurrency_best_measured_scenes_per_hour"]:.3f}场景/小时；'
              '按上述设计区间达不到≤36小时。只有先解决正式程序/布局阻断，再优化原生帧求值与读回或增加可用资源，'
              f'才可重新承诺配额耗时；未使用不合格{44-admitted}间来凑44间。每套按准入房数比例分配的条件算式和CPU小时下界'
              '在per_house_generation_plan.csv，不是独立种子容量的测量。', '',
              f'每个10秒窗口核心文件平均{planning["mean_core_window_bytes"]/1e6:.3f}MB；'
              f'两个窗口每研究成员平均{planning["mean_two_window_core_bytes_per_research_member"]/1e6:.3f}MB。'
              '媒体大小来自每个实际6核心文件的stat与sha256；渲染/RIR中间件另列且已验后压缩。'
              '题目数应区分旧研究窗口14个逻辑题与新版配额主窗口12题，估算中已分别标注。'
              '1.7MB/s来源是用户第5回合实测，此次没有重新测传输或向CCI写文件。', '',
              f'本任务产物及自有worktree唯一inode分配占用{storage["unique_inode_allocated_bytes"]/1e9:.3f}GB，'
              f'逐路径逻辑占用{storage["logical_path_bytes"]/1e9:.3f}GB；快照时/data空闲'
              f'{storage["shared_data_free_bytes"]/1e9:.3f}GB。80GB本任务上限及150GB共享盘预留保持；'
              'STORAGE_SNAPSHOT.json含命令等价统计方法和时间，最终封存后再次刷新。'
              '大批量媒体会超出剩余本任务存储额度，需要另行指定容量；未擅自删除交付物或他人的数据。', '',
              '派生资产体积/sha256：NATIVE_DERIVED_ENGINE_ASSETS.json与NATIVE_SOURCE_CONTENT_DEPENDENCIES.json；'
              '两套研究窗口及见证的拟传清单：CCI_GENERATED_WINDOWS_AND_PROVENANCE.json/.sha256/.list。'
              '仅内部非商业研究、引用、不得再分发；此次没有实际传输。', '',
              '清理只针对本任务已完成中间件：先封存完整窗口和必要见证，再调用原v7_compaction；'
              '原材质备份、已作废Habitat产物、原始包和共享缓存保留。0042是通用声学包例外，'
              '先前错误清理的3个数组已按原manifest SHA逐字节恢复，见'
              ' ue_round7/evidence/current_0042_package_restoration_v1/result.json；没有用重算门槛放行。', '',
              '## 改动、验证与交付位置', '',
              '只在48g自有 AVEngine-codex-v7-kujiale-20261003 worktree/分支中改动；独立提交、不push。'
              '改动包含原MIC恢复工具、UE专用高光检查、真实动画相位与脚底测量、逐点floor_reference、'
              '全部房间批验/安全摆位复验、原生与声学断言、CPU/GPU/存储守护、并行测量及交付汇总工具。'
              '正式生成器需要的cherry-pick提交和文件清单见CHERRY_PICK.json；默认共享生产行为没有更改。', '',
              '48g主要命令：restore_kujiale_native_source_assets.py；check_v7_kujiale_native_assets.py；'
              'run_kujiale_native_room_batch.py；check_kujiale_native_capsule_contact.py；'
              'verify_kujiale_native_floor_references.py；check_kujiale_native_motion_contact.py；'
              'check_kujiale_native_acoustic_alignment.py；benchmark_kujiale_native_concurrency.py --rounds 8 4 2；'
              'assemble_kujiale_native_training_admission.py。实际参数/PID/create_time/日志在'
              'ue_round6/scripts和ue_round7/scripts及FINAL_VERIFICATION.json；未在CCI启动作业。', '',
              '本轮第一次完整fast_unit结果：361 passed、36 skipped、7013 deselected，43.33秒；'
              '来源 ue_round7/logs/tests_fast_v1.log。跳过的是当前独立worktree未带的既有v6绑定facts，'
              '没有宣称这些跳过项已验证。最终提交前检查结果另见FINAL_VERIFICATION.json。', '',
              '离线验收页：'+str(output/'ACCEPTANCE_ue_v1.html')+'。44间每间一行平铺；另含59资产与两套恢复前后对照，'
              '图片全部内嵌。没有发送owner。提交号见COMMIT.json；进程最终状态见PROCESS_AUDIT.json。', '',
              '尚未完成：2/4实例有效全链路吞吐（共享盘预留不足）、正式多类别v7生成程序/题型族准入、'
              '冻结Kujiale种子分配、配额批量生成和实际CCI传输。'
              '当前正式训练数据量为0。UE原生生成的CCI环境兼容性未实测；按owner决定，当前路线在48g运行。', '']
    (output/'REPORT_zh.md').write_text('\n'.join(lines))
    brief = [lines[0], '', f'原材质75 MIC恢复；59/59资产只按原高光门槛通过，UE不使用贴图色差检查。'
             f'M2为7/7人物、{summary["M2"]["sample_count"]}姿态通过。44间已完成原生检查；'
             f'{44-admitted}间拒绝，详见逐间结果。', '',
             f'最好实际吞吐{planning["native_concurrency_best_measured_scenes_per_hour"]:.3f}场景/小时，'
             f'单进程树峰值{summary["peak_owned_process_tree_mib"]:.0f}MiB；'
             '实测适用于30秒两人研究流程；2/4实例完整流程因共享盘预留不足中断，尚未完成。'
             '完整v7吞吐未验证，设计区间推算无法在36小时内完成。', '',
             f'任务唯一inode分配{storage["unique_inode_allocated_bytes"]/1e9:.3f}GB。名单train_houses.json；'
             '房间room_results.csv；资产native_source_assets.csv；全部预算和限制见REPORT_zh.md。', '',
             '离线页（未发送owner）：'+str(output/'ACCEPTANCE_ue_v1.html'), '',
             '提交号见COMMIT.json；正式v7数据仍为0；没有启动批量生成或CCI传输。', '']
    (output/'BRIEF_zh.md').write_text('\n'.join(brief))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-results', type=Path, nargs='+', required=True)
    parser.add_argument('--incomplete-benchmark-results', type=Path, nargs='*', default=[])
    parser.add_argument('--generator-repository', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if root not in output.parents:
        raise ValueError('delivery must stay in this task root')
    output.mkdir(parents=True, exist_ok=False)
    measured, room_rows, house_details = collect_rooms(root)
    gallery, asset_rows, motion, m2_summary, restore = collect_sources(root)
    rounds, throughput, trials = collect_benchmarks(args.benchmark_results)
    renderer_measurements = collect_renderer_measurements(root, rounds, args.incomplete_benchmark_results)
    blockers = extract_formal_blockers(args.generator_repository.resolve())
    inventory = {row['house']: row for row in read(root/'inventory.json')}
    originals = {row['house']: row for row in read(root/'all_house_results.json')}
    build_selection = read(root/'evidence/build_selection.json')
    with (root/'house_results.csv').open() as stream:
        base_rows = list(csv.DictReader(stream))
    house_rows, all_houses, admitted_houses, admitted_rooms = [], [], [], []
    for base in base_rows:
        house = base['house']
        inv = inventory[house]
        rooms = [r for r in room_rows if r['house'] == house]
        admitted = [r['room_label'] for r in rooms if r['room_training_admitted']]
        native = house_details.get(house)
        build = Path(native['plan']['build_directory']) if native else root/'houses'/house/build_selection.get(house, 'build_v4')
        conversion = read(build/'conversion_result.json')
        failures = list(dict.fromkeys(reason for r in rooms for reason in r['reasons'].split(';') if reason))
        reasons = [] if admitted else failures or originals[house]['reasons']
        row = dict(base)
        row.update(render_path='ue_spear', visual_status='completed_native_ue' if native else 'not_run_acoustic_review',
                   status='qualified_room_environment' if admitted else 'review',
                   batch_eligible=False, training_admitted=bool(admitted), room_training_admitted=bool(admitted),
                   native_room_pass_count=sum(r['native_room_status'] == 'pass' for r in rooms),
                   admitted_room_count=len(admitted), admitted_room_ids=json.dumps(admitted),
                   native_candidate_rooms=len(rooms), native_qa_directory=native['directory'] if native else '',
                   formal_generation_ready=False, formal_dataset_count=0,
                   metadata_room_count=inv['metadata_room_count'], metadata_area_sum_m2=inv['metadata_area_sum_m2'],
                   room_type_counts=json.dumps(inv['room_type_counts'], ensure_ascii=False),
                   measured_floor_levels_m=json.dumps(inv['measured_floor_levels_m']), usd_complete=inv['usd_complete'],
                   native_import_to_acceptance_machine_seconds=native['job']['machine_seconds'] if native else None,
                   CPU_conversion_wall_seconds=conversion['wall_seconds'],
                   CPU_conversion_driver_seconds=conversion['driver_cpu_seconds'],
                   CPU_conversion_source=str(build/'conversion_result.json'),
                   reasons=';'.join(reasons), formal_blockers=';'.join(blockers['blockers']))
        house_rows.append(row)
        full = deepcopy(originals[house])
        full.update({k: row[k] for k in ['status', 'visual_status', 'render_path', 'batch_eligible', 'training_admitted',
                                       'room_training_admitted', 'native_room_pass_count', 'admitted_room_count',
                                       'formal_generation_ready', 'formal_dataset_count']})
        full.update(admitted_room_ids=admitted, native_room_results=rooms, reasons=reasons,
                    formal_blockers=blockers['blockers'], native_qa_result=native['qa_result'] if native else None)
        all_houses.append(full)
        if admitted:
            admitted_houses.append({'house': house, 'family': 'kujiale', 'split': 'train', 'render_path': 'ue_spear',
                                    'room_ids': admitted, 'native_map_path': native['plan']['map_path'],
                                    'native_qa_result': native['qa_result'], 'acoustic_status': 'pass'})
            for room_id in admitted:
                item = measured[(house, room_id)]
                admitted_rooms.append({'house': house, 'room_id': room_id, 'room_type': item['room']['room_type'],
                                       'render_path': 'ue_spear', 'frozen_witnesses': item['room']['selected_witnesses'],
                                       'floor_reference': item['floor_reference'], 'qa_result': item['qa_result']})
    summary = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'raw_house_count': 25, 'acoustic_pass_house_count': 21, 'native_checked_room_count': len(room_rows),
               'admitted_house_count': len(admitted_houses), 'admitted_room_count': len(admitted_rooms),
               'rejected_room_count': sum(not r['room_training_admitted'] for r in room_rows),
               'native_source_appearance_pass_count': len(asset_rows), 'native_source_appearance_count': 59,
               'max_source_highlight_ratio': max(r['max_highlight_ratio'] for r in asset_rows),
               'original_MIC_restored_count': 75, 'withdrawn_MR_texture_count': 32,
               'UE_color_gate_used': False, 'M2': m2_summary,
               'peak_owned_process_tree_mib': max(r['max_owned_process_tree_peak_mib'] for r in throughput),
               'throughput': throughput, 'full_scene_research_replicate_count': len(trials),
               'renderer_measurements': renderer_measurements,
               'full_generation_concurrency_1_2_4_8_complete': len(throughput) == 4,
               'uncompleted_requested_concurrency': sorted(set((1,2,4,8))-{r['parallel_instances'] for r in throughput}),
               'unique_house_seed_worlds_in_throughput_test': 2,
               'formal_dataset_count': 0, 'formal_generator': blockers,
               'room_admission_scope': 'physical environment / original UE source appearance / registered human idle-walk contact; formal dataset admission is separate',
               'html_sent_to_owner': False}
    write_csv(output/'house_results.csv', house_rows)
    write_csv(output/'room_results.csv', room_rows)
    write_csv(output/'native_source_assets.csv', asset_rows)
    write_csv(output/'concurrency_throughput.csv', throughput)
    write(output/'all_house_results.json', all_houses)
    write(output/'room_results.json', room_rows)
    write(output/'summary.json', summary)
    write(output/'native_M2_contact.json', m2_summary)
    write(output/'native_source_material_restore.json', restore)
    write(output/'native_full_scene_measurements.json', trials)
    write(output/'native_renderer_measurements.json', {'records': renderer_measurements,
          'full_generation_complete_for_all_requested_rounds': len(throughput) == 4})
    write(output/'train_houses.json', {'schema': 'kujiale_native_training_room_admission_v1',
                                    'split': 'train', 'render_path': 'ue_spear',
                                    'house_count': len(admitted_houses), 'room_count': len(admitted_rooms),
                                    'houses': admitted_houses, 'rooms': admitted_rooms,
                                    'source_asset_appearance_ids': [r['asset_id'] for r in asset_rows],
                                    'registered_idle_walk_human_ids': [r['asset_id'] for r in motion['records']],
                                    'formal_generation_ready': False, 'formal_dataset_count': 0,
                                    'formal_blockers': blockers['blockers']})
    floor_directory = output/'floor_reference'
    floor_directory.mkdir()
    for house in house_details:
        rooms = [r for r in admitted_rooms if r['house'] == house]
        write(floor_directory/(house+'.json'), {'house': house, 'render_path': 'ue_spear',
                                              'native_floor_surface': house_details[house]['native_floor_surface'],
                                              'rooms': {r['room_id']: r['floor_reference'] for r in rooms}})
    comparisons = material_comparisons(root, output, trials)
    write(output/'material_restore_comparison.json', comparisons)
    write_review(output, room_rows, measured, gallery, summary, comparisons)
    # The report, storage/transfer manifests and generation inputs are completed
    # in the same invocation by the reporting helpers below.
    write_generation_inputs(root, output, measured, house_details, admitted_houses, gallery)
    planning = write_planning(root, output, summary, trials, house_rows)
    write_engine_manifests(root, output, house_details, admitted_houses)
    write_media_transfer_manifest(root, output, trials)
    storage = storage_snapshot(root, output)
    write_report(root, output, summary, house_rows, asset_rows, motion, planning, storage)
    with (root/'PROGRESS_zh.md').open('a') as stream:
        stream.write('\n\n## '+summary['time_utc']+' 原生训练房源准入v1交付完成\n\n- '+
                     str(summary['admitted_house_count'])+'套 / '+str(summary['admitted_room_count'])+
                     '间，44间全部检查；59个原材质资产重判通过；7个人物注册idle/walk通过。'+
                     '逐套/逐间/逐资产表、已完成1/8完整吞吐及2/4中断记录、生成和传输预算、离线内嵌HTML见 '+str(output)+
                     '。仅房源准入，完整正式v7数据量仍为0；没有发送owner或传CCI。\n')


if __name__ == '__main__':
    main()
