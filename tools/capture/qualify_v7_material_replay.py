#!/usr/bin/env python3
"""Replay v3 source-material changes in fresh CPU native processes.

This one-off before/after audit keeps the four-member reference read-only.
It checks all-frame depth/semantic and mouth readbacks, actor-free scan pixels,
and unchanged reference FOA/questions. It does not admit motion or occlusion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from avengine.assets.source_materials import measure_source_appearance, prepare_source_glb
from avengine.capture.continuous_native_video import NativeContinuousVideo
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_candidate_generation import seeded_scene, member_scene
from avengine.timeline.v7_native_questions import make_questions


def read(path):
    return json.loads(Path(path).read_text())


def legacy_renderer(revision):
    source = subprocess.check_output(
        ['git', 'show', revision + ':src/avengine/capture/continuous_native_video.py'], text=True,
    )
    namespace = {'__name__': '_v7_material_replay_baseline'}
    exec(compile(source, 'git:' + revision + ':continuous_native_video.py', 'exec'), namespace)
    return namespace['NativeContinuousVideo']


class ObservationAudit:
    """One-off comparisons only; no new production identity or gate."""
    def __init__(self, sim):
        self.sim = sim
        self.rows = []

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def get_sensor_observations(self):
        obs = self.sim.get_sensor_observations()
        self.rows.append({
            name: hashlib.sha256(np.ascontiguousarray(obs[name]).tobytes()).hexdigest()
            for name in ('depth', 'semantic')
        })
        return obs


def render_job(task):
    case, phase, root, revision = task
    root = Path(root) / case['case_id'] / phase
    root.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    working, scene = seeded_scene(case['row'], case['house'])
    cls = legacy_renderer(revision) if phase == 'before' else NativeContinuousVideo
    kwargs = {} if phase == 'before' else {'source_material_cache_directory': root / 'source_material_cache'}
    video = cls(working, scene, case['inputs'], case['assets'], **kwargs)
    try:
        # Identical actor-free controls prove that loading source shaders did
        # not alter scan drawables. The camera is fixed for this whole scene.
        for i, entry in enumerate(video.actors):
            entry['object'].translation = video.mn.Vector3(2000 + i * 10, 2000, 2000)
        scan = video.sim.get_sensor_observations()
        np.savez_compressed(root / 'actor_free_scan.npz', **{
            k: np.asarray(scan[k]) for k in ('rgb', 'depth', 'semantic')
        })
        Image.fromarray(np.asarray(scan['rgb'])[:, :, :3]).save(root / 'actor_free_scan.png')
        recorder = ObservationAudit(video.sim)
        video.sim = recorder
        result = video.encode(scene, root / 'scene_video.mp4', review_frames=[0, 225, 449])
        assert len(recorder.rows) == result['frame_count'] == 450
        write_json(root / 'depth_semantic_comparison.json', recorder.rows)
        if phase == 'before':
            # Measure the old pixels under the unchanged new acceptance policy;
            # failed old samples are evidence, never silently accepted output.
            checks = []
            for entry in video.actors:
                _, material = prepare_source_glb(
                    entry['bundle'].paths_by_role['visual'], asset_id=entry['declaration']['asset_id'],
                    entity_class=entry['inputs'].asset.get('entity_class', 'articulated_human'),
                    output_directory=root / 'reference_material_metadata',
                )
                for frame in (0, 225, 449):
                    with np.load(root / f'native_masks_{frame:04d}.npz', allow_pickle=False) as data:
                        mask = data['semantic'] == entry['semantic_id']
                    rgb = np.asarray(Image.open(root / f'native_frame_{frame:04d}.png'))
                    checks.append({
                        'endpoint_id': entry['endpoint_id'], 'frame_index': frame,
                        **measure_source_appearance(rgb, mask,
                            palette=np.asarray(material['source_palette_srgb']),
                            max_highlight_ratio=max(p['max_highlight_ratio'] for p in material['profiles']), strict=False),
                    })
            write_json(root / 'old_material_checks.json', {'checks': checks})
        measured = subprocess.check_output([
            'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames:format=duration',
            '-of', 'json', str(root / 'scene_video.mp4'),
        ], text=True)
        encoded = json.loads(measured)
        assert encoded['streams'][0]['nb_read_frames'] == '450'
        assert float(encoded['format']['duration']) == 30.
        receipt = {'case_id': case['case_id'], 'phase': phase, 'house': working['house'],
            'frame_count': 450, 'encoded_media': encoded, 'compute_device': 'CPU',
            'wall_seconds': time.perf_counter() - start, 'process_CPU_seconds': time.process_time() - cpu,
            'formal_dataset_count': 0, 'baseline_git_revision': revision if phase == 'before' else None}
        write_json(root / 'receipt.json', receipt)
        return receipt
    finally:
        video.close()


def variant_assets(row, assets, visual):
    result = deepcopy(assets)
    if visual.startswith('v1') != (row['seeds']['appearance'] % 2 == 1):
        endpoints = [a['endpoint_id'] for a in result]
        result.reverse()
        for asset, endpoint in zip(result, endpoints, strict=True):
            asset['endpoint_id'] = endpoint
    return result


def compare_case(case, output):
    root = output / case['case_id']
    before, after = root / 'before', root / 'after'
    b, a = read(before / 'native_visual_evidence.json'), read(after / 'native_visual_evidence.json')
    assert b['observed_mouth_trajectories'] == a['observed_mouth_trajectories']
    old_fingerprints = read(before / 'depth_semantic_comparison.json')
    new_fingerprints = read(after / 'depth_semantic_comparison.json')
    assert len(old_fingerprints) == len(new_fingerprints) == 450
    exact = {k: sum(x[k] == y[k] for x, y in zip(old_fingerprints, new_fingerprints, strict=True))
             for k in ('depth', 'semantic')}
    raw_identical = all(n == 450 for n in exact.values())
    sampled = []
    for frame in (0, 225, 449):
        with np.load(before / f'native_masks_{frame:04d}.npz', allow_pickle=False) as x, np.load(after / f'native_masks_{frame:04d}.npz', allow_pickle=False) as y:
            delta = np.abs(x['depth'] - y['depth'])
            sampled.append({'frame': frame, 'depth_changed_pixels': int(np.count_nonzero(delta)),
                'depth_max_abs_m': float(delta.max()), 'semantic_changed_pixels': int(np.count_nonzero(x['semantic'] != y['semantic']))})
    count_deltas = [abs(x['visible_pixels'] - y['visible_pixels'])
        for bf, af in zip(b['frames'], a['frames'], strict=True)
        for x, y in zip(bf['actors'], af['actors'], strict=True)]
    with np.load(before / 'actor_free_scan.npz', allow_pickle=False) as old, np.load(after / 'actor_free_scan.npz', allow_pickle=False) as new:
        scan_equal = {k: bool(np.array_equal(old[k], new[k])) for k in old.files}
    assert all(scan_equal.values()), scan_equal
    checks = read(after / 'source_material_checks.json')['checks']
    assert len(checks) == 6 and all(c['status'] == 'pass' for c in checks), checks
    # Small side-by-side review artifacts retain the native source pixels.
    for frame in (0, 225, 449):
        old = Image.open(before / f'native_frame_{frame:04d}.png').convert('RGB')
        new = Image.open(after / f'native_frame_{frame:04d}.png').convert('RGB')
        combined = Image.new('RGB', (old.width * 2, old.height))
        combined.paste(old, (0, 0)); combined.paste(new, (old.width, 0))
        combined.save(root / f'before_after_{frame:04d}.png')
    return {'case_id': case['case_id'], 'house': case['house']['house'], 'room': case['row']['room_id'],
        'status': 'pass_materials_raw_sensor_identity_failed' if not raw_identical else 'pass_material_render_regression_only', 'first_middle_last_checks': len(checks),
        'material_passes': len(checks), 'raw_sensors_identical': raw_identical,
        'exact_equal_frames': exact, 'sampled_sensor_differences': sampled,
        'max_visible_source_pixel_count_difference': max(count_deltas),
        'mouth_trajectories_equal': True, 'actor_free_scan_equal': scan_equal,
        'occlusion_admission': 'not_run', 'formal_dataset_count': 0,
        'old_material_checks': read(before / 'old_material_checks.json'),
        'new_material_checks': checks}


def replay_group(reference, row, house, inputs, cases, output):
    sealed = read(reference / 'delivery_completed.json')
    original = reference / sealed['attempt']
    by_visual = {c['visual']: c for c in cases if c['role'] == 'four_member_reference'}
    working, base = seeded_scene(row, house)
    members, unchanged = [], []
    for member in row['members']:
        case = by_visual[member[:2]]
        video_root = output / case['case_id'] / 'after'
        visual = read(video_root / 'native_visual_evidence.json')
        retained = read(original / ('provenance_' + member + '.json'))['native_visual_readback']
        assert visual['observed_mouth_trajectories'] == retained['observed_mouth_trajectories']
        scene = member_scene(base, visual, member)
        appearances = {base['sources'][a['endpoint_id']]['actor_id']: a.get('appearance_zh',
            {'blue': '蓝色上衣的人', 'yellow': '黄色上衣的人'}[a['appearance']['top_color']]) for a in case['assets']}
        windows = []
        for i, start in enumerate(scene['window_starts_s']):
            wid = f'{row["scene_id"]}__{member}__w{i:02d}'
            source, dest = original / wid, output / 'four_member_windows' / wid
            dest.mkdir(parents=True, exist_ok=False)
            first = round(start * 15)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-threads', '1',
                '-i', str(video_root / 'scene_video.mp4'), '-vf', f'select=between(n\\,{first}\\,{first+149}),setpts=N/(15*TB)',
                '-frames:v', '150', '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast', '-crf', '18',
                '-pix_fmt', 'yuv420p', str(dest / 'video.mp4')], check=True)
            for name in ('audio_foa.wav', 'audio_mono.wav'):
                shutil.copyfile(source / name, dest / name)
                assert (source / name).read_bytes() == (dest / name).read_bytes()
            media = {'video': 'video.mp4', 'audio_foa': 'audio_foa.wav', 'audio_mono': 'audio_mono.wav', 'foa_format': 'ACN_WYZX_SN3D_camera'}
            questions, answers, validation = make_questions(scene, start, wid, visual, appearances, media,
                quota_index=row['question_quota_base'] + row['members'].index(member) * row['window_count'] + i,
                count=row['questions_per_window'])
            assert questions == read(source / 'questions.json'), 'material change altered public question'
            assert answers == read(source / 'private_answers.json'), 'material change altered private answer'
            write_json(dest / 'questions.json', questions); write_json(dest / 'private_answers.json', answers)
            write_json(dest / 'evidence.json', {'scene': scene, 'member': member, 'native_visual_evidence': str(video_root / 'native_visual_evidence.json'),
                'reference_window': str(source), 'audio_provenance': 'unchanged reference FOA reused; no RLR rerender',
                'question_checks': validation, 'formal_dataset_count': 0})
            windows.append({'window_id': wid}); unchanged.append(wid)
        members.append({'member': member, 'windows_directory': str(output / 'four_member_windows'), 'windows': windows})
    from avengine.timeline.v7_candidate_generation import audit_parent_group
    group = audit_parent_group(row, base, members)
    write_json(output / 'four_member_comparison.json', {'status': 'pass', 'unchanged_questions_answers_and_FOA_windows': unchanged,
        'windows': len(unchanged), 'logical_questions': sum(len(read(output / 'four_member_windows' / w / 'questions.json')) for w in unchanged),
        'counterfactual_audit': group, 'audio_regenerated': False, 'formal_dataset_count': 0})
    return group


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-parent', type=Path, required=True)
    p.add_argument('--houses', type=Path, required=True); p.add_argument('--seeds-train', type=Path, required=True)
    p.add_argument('--seeds-dev', type=Path, required=True); p.add_argument('--runtime-inputs', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline-commit', required=True)
    p.add_argument('--S-train-house'); p.add_argument('--S-dev-house')
    p.add_argument('--resume-from', type=Path)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU only; CUDA_VISIBLE_DEVICES must be empty')
    a.output.mkdir(parents=True, exist_ok=False)
    recipe = read(a.reference_parent / 'input_recipe.json'); cases = []
    for v in ('v0', 'v1'):
        cases.append({'case_id': 'four_member_' + v, 'role': 'four_member_reference', 'visual': v,
            'row': recipe['seed_row'], 'house': recipe['house'], 'inputs': recipe['runtime_inputs'],
            'assets': variant_assets(recipe['seed_row'], recipe['assets'], v)})
    by_id = {h['house']: h for h in read(a.houses)['houses']}
    inputs, assets = read(a.runtime_inputs), read(a.assets)['assets']
    for split, path in [('train', a.seeds_train), ('dev', a.seeds_dev)]:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        wanted = a.S_train_house if split == 'train' else a.S_dev_house
        row = next(r for r in rows if len(r['members']) == 1 and (wanted is None or r['house'] == wanted))
        cases.append({'case_id': 'S_' + split, 'role': 'S_house', 'visual': 'v0', 'row': row,
            'house': by_id[row['house']], 'inputs': inputs, 'assets': variant_assets(row, assets, 'v0')})
    write_json(a.output / 'render_inputs.json', {'cases': cases, 'baseline_commit': a.baseline_commit})
    tasks = [(c, phase, str(a.output), a.baseline_commit) for c in cases for phase in ('before', 'after')]
    receipts, pending = [], []
    previous = read(a.resume_from / 'render_inputs.json') if a.resume_from else None
    for task in tasks:
        case, phase, _, _ = task
        src = a.resume_from / case['case_id'] / phase if a.resume_from else None
        required = ['receipt.json', 'scene_video.mp4', 'native_visual_evidence.json',
                    'depth_semantic_comparison.json', 'actor_free_scan.npz',
                    *[f'native_frame_{f:04d}.png' for f in (0, 225, 449)],
                    *[f'native_masks_{f:04d}.npz' for f in (0, 225, 449)],
                    'old_material_checks.json' if phase == 'before' else 'source_material_checks.json']
        old_case = next((c for c in previous['cases'] if c['case_id'] == case['case_id']), None) if previous else None
        if (previous and previous['baseline_commit'] == a.baseline_commit and old_case == case
                and all((src / f).is_file() for f in required)):
            dest = a.output / case['case_id'] / phase
            shutil.copytree(src, dest, copy_function=os.link)
            write_json(dest.parent / ('reuse_from_' + phase + '.json'), {'source': str(src), 'same_input_case': True,
                'reason': 'completed native render retained; failed or changed cases rerun'})
            receipts.append(read(src / 'receipt.json'))
        else:
            pending.append(task)
    with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts.extend(pool.map(render_job, pending))
    compared = [compare_case(c, a.output) for c in cases]
    group = replay_group(a.reference_parent, recipe['seed_row'], recipe['house'], recipe['runtime_inputs'], cases, a.output)
    raw_identity_passed = all(c['raw_sensors_identical'] for c in compared)
    write_json(a.output / 'summary.json', {'status': 'pass' if raw_identity_passed else 'fail_raw_sensor_identity_materials_pass', 'cases': compared,
        'other_products_all_identical': raw_identity_passed,
        'native_frames_before_and_after': sum(r['frame_count'] for r in receipts),
        'material_first_middle_last_passes': sum(c['material_passes'] for c in compared),
        'render_receipts': receipts, 'counterfactual_status': group['status'],
        'occlusion_admission': 'not_run', 'formal_dataset_count': 0})
    if not raw_identity_passed:
        raise SystemExit(2)  # Preserve the failed dimension; no tolerance waiver.


if __name__ == '__main__':
    main()
