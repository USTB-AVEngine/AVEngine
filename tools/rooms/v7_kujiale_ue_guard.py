#!/usr/bin/env python3
"""Run one isolated native UE task with graphics-aware GPU and disk accounting.

Ownership is established by Popen, PID creation time and command identity;
this program never searches for or terminates unrelated editors.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
import psutil
import resource


def parse_gpu_xml(raw: str) -> list[dict]:
    rows = []
    for index, gpu in enumerate(ET.fromstring(raw).findall('gpu')):
        def memory(value):
            if value is None or value in {'N/A', '[Not Supported]'}:
                return None
            return float(value.split()[0])
        processes = []
        for proc in gpu.findall('./processes/process_info'):
            processes.append({'pid': int(proc.findtext('pid')),
                              'type': proc.findtext('type'),
                              'name': proc.findtext('process_name'),
                              'memory_mib': memory(proc.findtext('used_memory'))})
        rows.append({'index': index, 'uuid': gpu.findtext('uuid'),
                     'free_mib': memory(gpu.findtext('./fb_memory_usage/free')),
                     'used_mib': memory(gpu.findtext('./fb_memory_usage/used')),
                     'processes': processes})
    return rows


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], check=True,
                            text=True, capture_output=True, timeout=15)
    return parse_gpu_xml(result.stdout)


def logical_bytes(root: Path) -> int:
    total = 0
    for directory, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = Path(request['budget_root']).resolve()
    run = Path(request['run_directory']).resolve()
    if root not in run.parents or run.exists():
        raise ValueError('guard run must be new and below the budget root')
    run.mkdir(parents=True)
    command = request['command']
    if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
        raise ValueError('command must be a string argv array')
    gpu_index = int(request.get('gpu_index', 0))
    limit = int(request.get('storage_limit_bytes', 50_000_000_000))
    previous_peak = request.get('previous_peak_mib')
    required = 16 * 1024 if previous_peak is None else float(previous_peak) + 4 * 1024
    required = max(required, float(request.get('minimum_free_mib', 0)))
    initial = gpu_snapshot()
    selected = initial[gpu_index]
    size = logical_bytes(root)
    reserve = int(request.get('reserve_bytes', 0))
    receipt = {'schema': 'kujiale_native_ue_guard_v1', 'request': str(args.request),
               'start_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'initial_gpu': initial, 'required_free_mib': required,
               'initial_logical_bytes': size, 'reserve_bytes': reserve,
               'storage_limit_bytes': limit, 'command': command,
               'status': 'preflight'}
    if selected['free_mib'] is None or selected['free_mib'] < required or size + reserve > limit:
        receipt['status'] = 'blocked_preflight'
        write_json(run / 'result.json', receipt)
        raise RuntimeError('GPU free memory or storage reserve is insufficient')
    environment = dict(os.environ)
    environment.update(request.get('environment', {}))
    for name in request.get('unset_environment', []):
        environment.pop(name, None)
    started = time.monotonic()
    available_cpus = sorted(os.sched_getaffinity(0))
    cpus = set(request.get('cpu_affinity', available_cpus[:8]))
    if not cpus or not cpus.issubset(available_cpus):
        raise ValueError('requested CPU affinity is not available')
    receipt['cpu_affinity'] = sorted(cpus)
    owned = {}
    cpu_by_identity = {}
    stop_reason = None
    child = None
    def on_signal(number, _frame):
        nonlocal stop_reason
        stop_reason = f'guard_signal_{number}'
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    peak = 0.0
    try:
        def configure_child():
            os.sched_setaffinity(0, cpus)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        with (run / 'stdout.log').open('w') as output, (run / 'samples.jsonl').open('w') as samples:
            child = subprocess.Popen(command, cwd=request['cwd'], env=environment,
                                     stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     preexec_fn=configure_child)
            parent = psutil.Process(child.pid)
            owned[parent.pid] = (parent.create_time(), parent.cmdline())
            receipt['pid'] = parent.pid
            receipt['pid_create_time'] = parent.create_time()
            write_json(run / 'launch.json', receipt)
            (run / 'task.pid').write_text(str(child.pid) + '\n')
            tick = 0
            while True:
                try:
                    for proc in [parent] + parent.children(recursive=True):
                        try:
                            owned[proc.pid] = (proc.create_time(), proc.cmdline())
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except psutil.NoSuchProcess:
                    pass
                active = set()
                active_cpu = []
                for pid, (created, cmdline) in list(owned.items()):
                    try:
                        proc = psutil.Process(pid)
                        if proc.create_time() == created and proc.cmdline() == cmdline:
                            active.add(pid)
                            measured = proc.cpu_times()
                            row = {'pid': pid, 'create_time': created,
                                   'user_seconds': measured.user,
                                   'system_seconds': measured.system}
                            key = (pid, created)
                            previous = cpu_by_identity.get(key)
                            if previous is None or measured.user + measured.system >= previous['user_seconds'] + previous['system_seconds']:
                                cpu_by_identity[key] = row
                            active_cpu.append(row)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                snapshot = gpu_snapshot()
                mine = []
                total_mib = 0.0
                for gpu in snapshot:
                    for proc in gpu['processes']:
                        if proc['pid'] in active:
                            mine.append(dict(proc, gpu_index=gpu['index']))
                            if proc['memory_mib'] is None:
                                stop_reason = 'owned_graphics_memory_unavailable'
                            else:
                                total_mib += proc['memory_mib']
                            if gpu['index'] != gpu_index:
                                stop_reason = 'owned_process_used_unselected_gpu'
                peak = max(peak, total_mib)
                if peak + 4096 > selected['free_mib']:
                    stop_reason = 'measured_peak_plus_4g_exceeds_initial_free'
                remaining_free = snapshot[gpu_index]['free_mib']
                if remaining_free is not None and remaining_free < 4096 and total_mib:
                    stop_reason = 'concurrent_gpu_load_left_less_than_4g'
                size = logical_bytes(root)
                if size > limit - int(request.get('stop_margin_bytes', 750_000_000)):
                    stop_reason = 'storage_limit_margin'
                samples.write(json.dumps({'elapsed_seconds': time.monotonic()-started,
                                          'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                          'owned_gpu_processes': mine, 'owned_total_mib': total_mib,
                                          'owned_process_cpu_times': active_cpu,
                                          'owned_tree_cpu_seconds_lower_bound': sum(v['user_seconds'] + v['system_seconds'] for v in cpu_by_identity.values()),
                                          'peak_mib': peak, 'selected_free_mib': remaining_free,
                                          'logical_bytes': size}) + '\n')
                samples.flush()
                if stop_reason or child.poll() is not None:
                    break
                if time.monotonic()-started > request.get('maximum_seconds', 7200):
                    stop_reason = 'time_budget_exceeded'
                    break
                tick += 1
                time.sleep(max(0.0, started + tick - time.monotonic()))
    finally:
        # Exact recorded identities only, including shader workers that outlive
        # a failed RPC constructor. Never pkill or touch a foreign process.
        alive = []
        for pid, (created, commandline) in reversed(list(owned.items())):
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == created and proc.cmdline() == commandline:
                    if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                        proc.terminate()
                        alive.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        gone, still_alive = psutil.wait_procs(alive, timeout=15)
        for proc in still_alive:
            try:
                created, commandline = owned[proc.pid]
                if proc.create_time() == created and proc.cmdline() == commandline:
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if child is not None:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
        receipt.update({'status': 'stopped' if stop_reason else ('pass' if child is not None and child.returncode == 0 else 'failed'),
                        'stop_reason': stop_reason,
                        'returncode': None if child is None else child.returncode,
                        'elapsed_seconds': time.monotonic()-started, 'peak_mib': peak,
                        'owned_tree_cpu_seconds_lower_bound': sum(v['user_seconds'] + v['system_seconds'] for v in cpu_by_identity.values()),
                        'owned_process_cpu_times_last': list(cpu_by_identity.values()),
                        'cpu_measurement': 'psutil user+system times for exact owned PID/create-time identities, sampled every second; excludes missed short-lived children and final unsampled CPU, hence a lower bound',
                        'final_logical_bytes': logical_bytes(root),
                        'owned_processes': [{'pid':pid,'create_time':info[0],'command':info[1]} for pid,info in owned.items()]})
        write_json(run / 'result.json', receipt)
    if receipt['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
