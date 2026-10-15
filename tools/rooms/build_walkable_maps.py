#!/usr/bin/env python3
"""Precompute reusable 5cm native static room maps on CPU, with bounded workers."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil
from avengine.rooms.walkable_map import room_map_path, WalkableMap, source_identity
from avengine.rooms.walkable_map_builder import build_room_map, native_static_video, stage_path, render_map_png


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def progress(root, line):
    with (Path(root)/'PROGRESS.md').open('a') as stream:
        stream.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+line+'\n')


def worker(job_path):
    job = read(job_path)
    root = Path(job['output_root'])
    rows = []
    video = None
    try:
        for room in job['rooms']:
            began = time.monotonic()
            target = room_map_path(root, room['room_label'])
            try:
                if target.exists():
                    result = WalkableMap.load(target)
                    if result.metadata['floor_polygon'] != room['floor_polygon'] or abs(result.floor_reference-room['measured_floor_y_m']) > .0001:
                        raise ValueError('cached room polygon/floor changed; move stale map explicitly before rebuilding')
                    status = 'reused'
                else:
                    if video is None:
                        video = native_static_video(job['house'], job['runtime'])
                    result = build_room_map(video, job['house'], room, code_commit=job['code_commit'], output=target)
                    status = 'built'
                png = None
                if room['room_label'] in job['png_rooms']:
                    if video is None:
                        video = native_static_video(job['house'], job['runtime'])
                    from avengine.assets.v7_phase2_floor import original_native_stage_mesh
                    mesh, _ = original_native_stage_mesh(video, stage_path(job['house']))
                    png = str(target.with_suffix('.png'))
                    render_map_png(result, mesh, png)
                row = {'room_id': room['room_label'], 'house_id': job['house']['house'], 'family': job['house']['family'],
                       'status': status, 'map': str(target), 'png': png, 'wall_seconds': time.monotonic()-began,
                       'compute_seconds': result.metadata['compute_seconds'], 'cell_counts': result.metadata['cell_counts']}
            except Exception as error:
                import traceback
                traceback.print_exc()
                row = {'room_id': room['room_label'], 'house_id': job['house']['house'], 'status': 'failed',
                       'wall_seconds': time.monotonic()-began, 'reason': str(error)}
            rows.append(row)
            write(job['result'], {'rows': rows})
            progress(root, '房间 '+room['room_label']+' '+json.dumps(row, ensure_ascii=False))
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if video is not None:
            video.sim.close()


def _own_tree_pss():
    process = psutil.Process()
    total = 0
    for child in [process, *process.children(recursive=True)]:
        try:
            total += child.memory_full_info().pss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--houses', type=Path, action='append')
    p.add_argument('--runtime', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--priority-manifest', type=Path, action='append', default=[])
    p.add_argument('--room-id', action='append', default=[])
    p.add_argument('--families', nargs='+', default=['hm3d', 'mp3d'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--png-count', type=int, default=3)
    p.add_argument('--priority-only', action='store_true')
    p.add_argument('--worker-job', type=Path)
    a = p.parse_args()
    if a.worker_job:
        worker(a.worker_job)
        return
    if not 1 <= a.workers <= 22:
        p.error('use 1..22 single-thread workers; supervisor stays within 24 CPU processes')
    if not a.houses or not a.runtime or not a.output_root:
        p.error('--houses, --runtime and --output-root are required')
    root = a.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for directory in ('inputs/map_jobs', 'evidence/map_jobs', 'logs/map_jobs'):
        (root/directory).mkdir(parents=True, exist_ok=True)
    import avengine
    expected = Path(__file__).resolve().parents[2]/'src/avengine/__init__.py'
    if Path(avengine.__file__).resolve() != expected:
        raise ValueError('Python imported another checkout: '+str(avengine.__file__))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=expected.parents[2], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src/avengine/rooms/walkable_map.py',
                                    'src/avengine/rooms/walkable_map_builder.py', 'tools/rooms/build_walkable_maps.py'],
                                   cwd=expected.parents[2], text=True)
    if dirty:
        raise ValueError('commit map producer files before creating attributed caches')
    priority = set(a.room_id)
    for manifest in a.priority_manifest:
        for task in read(manifest).get('parents', []):
            row = task.get('row', task)
            if row.get('room_id'):
                priority.add(row['room_id'])
    houses = {}
    for inventory in a.houses:
        for house in read(inventory)['houses']:
            if house['family'] not in a.families:
                continue
            if house['house'] not in houses:
                houses[house['house']] = {**house, 'checked_rooms': []}
            existing = {r['room_label'] for r in houses[house['house']]['checked_rooms']}
            houses[house['house']]['checked_rooms'].extend(r for r in house['checked_rooms'] if r['room_label'] not in existing)
    jobs = []
    for house in houses.values():
        rooms = [r for r in house['checked_rooms'] if (not a.room_id or r['room_label'] in a.room_id)
                 and (not a.priority_only or r['room_label'] in priority)]
        rooms.sort(key=lambda r: (r['room_label'] not in priority, r['room_label']))
        if rooms:
            jobs.append((house, rooms))
    jobs.sort(key=lambda j: (not any(r['room_label'] in priority for r in j[1]), j[0]['house']))
    png_rooms = [r['room_label'] for _, rooms in jobs for r in rooms if r['room_label'] in priority][:a.png_count]
    if len(png_rooms) < a.png_count:
        png_rooms.extend(r['room_label'] for _, rooms in jobs for r in rooms if r['room_label'] not in png_rooms)
        png_rooms = png_rooms[:a.png_count]
    all_rooms = {r['room_label'] for _, rooms in jobs for r in rooms}
    write(root/'evidence/MAP_INVENTORY.json', {'source_inventories': [str(p) for p in a.houses], 'room_count': len(all_rooms),
          'priority_rooms': sorted(priority), 'missing_priority_rooms': sorted(priority-all_rooms), 'png_rooms': png_rooms,
          'source_code_commit': commit, 'import_file': str(expected), 'workers': a.workers})
    pending = []
    for index, (house, rooms) in enumerate(jobs):
        path = root/'inputs/map_jobs'/f'{index:04d}.json'
        result = root/'evidence/map_jobs'/f'{index:04d}.json'
        write(path, {'house': house, 'rooms': rooms, 'runtime': read(a.runtime), 'output_root': str(root),
                     'code_commit': commit, 'png_rooms': png_rooms, 'result': str(result)})
        pending.append((index, path, result))
    active = []
    paused = False
    peak_pss = 0
    start = time.monotonic()
    progress(root, f'CPU地图批次开始，房间={len(all_rooms)}，优先={len(priority & all_rooms)}，worker={a.workers}，commit={commit}')
    while pending or active:
        remaining = []
        for index, process, result_path, log in active:
            if process.poll() is None:
                remaining.append((index, process, result_path, log))
                continue
            existing_rows = read(result_path)['rows'] if result_path.exists() else []
            completed = {r['room_id'] for r in existing_rows}
            job = read(root/'inputs/map_jobs'/f'{index:04d}.json')
            for room in job['rooms']:
                if room['room_label'] not in completed:
                    existing_rows.append({'room_id': room['room_label'], 'house_id': job['house']['house'],
                        'status': 'failed', 'reason': f'worker exited {process.returncode}; see {log}'})
            write(result_path, {'rows': existing_rows})
        active = remaining
        pss = _own_tree_pss()
        peak_pss = max(peak_pss, pss)
        available = psutil.virtual_memory().available
        constrained = available < 30_000_000_000 or pss > 39_000_000_000
        if constrained and not paused:
            # Only subprocesses launched here are suspended; no shared process scan.
            for _, process, _, _ in active:
                if process.poll() is None:
                    os.kill(process.pid, signal.SIGSTOP)
            paused = True
            progress(root, f'资源暂停 own_pss={pss} available={available}')
        elif paused and available >= 35_000_000_000 and pss < 39_000_000_000:
            for _, process, _, _ in active:
                if process.poll() is None:
                    os.kill(process.pid, signal.SIGCONT)
            paused = False
            progress(root, '资源恢复')
        while pending and not paused and len(active) < a.workers and pss+(len(active)+1)*2_500_000_000 < 38_000_000_000:
            index, job, result = pending.pop(0)
            log = root/'logs/map_jobs'/f'{index:04d}.log'
            with log.open('a') as stream:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker-job', str(job)],
                                           stdout=stream, stderr=subprocess.STDOUT)
            active.append((index, process, result, str(log)))
        rows = []
        for result in sorted((root/'evidence/map_jobs').glob('*.json')):
            rows.extend(read(result)['rows'])
        report = {'status': 'running' if pending or active else 'complete', 'rooms_reported': len(rows),
                  'rooms_built_or_reused': sum(r['status'] in {'built', 'reused'} for r in rows),
                  'failed': sum(r['status'] == 'failed' for r in rows), 'rows': rows, 'source_code_commit': commit,
                  'wall_seconds': time.monotonic()-start, 'own_peak_pss_bytes': peak_pss, 'workers': a.workers,
                  'active': [{'job': i, 'pid': process.pid, 'log': log} for i, process, _, log in active], 'paused': paused}
        write(root/'MAP_BUILD_REPORT.json', report)
        if pending or active:
            time.sleep(5)
    progress(root, f'CPU地图批次完成：{report["rooms_built_or_reused"]}间成功，{report["failed"]}间失败；报告 MAP_BUILD_REPORT.json')


if __name__ == '__main__':
    main()
