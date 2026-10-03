#!/usr/bin/env python3
"""CPU-only allocation probe. GPU queries never create a CUDA/EGL context."""
from __future__ import annotations
import argparse
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid

GIB = 1024**3

def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

def command(argv, timeout=3):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"argv": argv, "error": str(e)}

def cgroup_limits():
    # Read both the mounted namespace root and the process's visible ancestors.
    paths = {Path('/sys/fs/cgroup')}
    raw = read('/proc/self/cgroup') or ''
    for line in raw.splitlines():
        bits = line.split(':', 2)
        if len(bits) == 3 and bits[0] == '0':
            p = Path('/sys/fs/cgroup') / bits[2].lstrip('/')
            while p.is_relative_to('/sys/fs/cgroup'):
                paths.add(p)
                if p == Path('/sys/fs/cgroup'):
                    break
                p = p.parent
    mount_root='/'
    for line in (read('/proc/self/mountinfo') or '').splitlines():
        fields=line.split()
        if '-' in fields and fields[fields.index('-')+1]=='cgroup2' and fields[4]=='/sys/fs/cgroup':
            mount_root=fields[3];break
    rows = []
    for p in sorted(paths):
        cpu, memory = read(p/'cpu.max'), read(p/'memory.max')
        if cpu is None and memory is None:
            continue
        row = {"canonical_path":str(Path(mount_root)/p.relative_to('/sys/fs/cgroup')),"path": str(p), "cpu.max": cpu, "memory.max": memory, "memory.current": read(p/'memory.current'), "cpuset.cpus.effective": read(p/'cpuset.cpus.effective')}
        if cpu and cpu.split()[0] != 'max':
            q, period = map(int, cpu.split()); row['cpu_quota_cores'] = q / period
        if memory and memory != 'max':
            row['memory_limit_bytes'] = int(memory)
        rows.append(row)
    return raw, rows

def storage(path):
    try:
        s = os.statvfs(path)
        return {"path": str(path), "total_bytes": s.f_blocks*s.f_frsize, "available_bytes": s.f_bavail*s.f_frsize, "note": "filesystem-wide capacity; not the owner's 2 TB allocation"}
    except OSError as e:
        return {"path": str(path), "error": str(e)}

def gpu_inventory():
    q = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'])
    all_gpus = []
    if q.get('returncode') == 0:
        for line in q['stdout'].splitlines():
            b = [x.strip() for x in line.split(',')]
            if len(b) == 4:
                all_gpus.append(dict(zip(('index','uuid','name','memory_mib'), b)))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None and os.environ.get('NVIDIA_VISIBLE_DEVICES') not in (None,'all'):
        visible=os.environ.get('NVIDIA_VISIBLE_DEVICES')
    if visible is None:
        nodes={p.name.removeprefix('nvidia') for p in Path('/dev').glob('nvidia[0-9]*')}
        gpus = [r for r in all_gpus if r['index'] in nodes]
    elif not visible.strip() or visible.strip() == '-1':
        gpus = []
    else:
        tokens = visible.split(',')
        gpus = [r for r in all_gpus if r['index'] in tokens or any(r['uuid'].startswith(x) for x in tokens)]
    return {"query": q, "all": all_gpus, "visible": gpus, "visible_count": len(gpus), "CUDA_VISIBLE_DEVICES": visible}

def io_probe(directory, mib=2):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    p = directory / f'.probe-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}'
    block = os.urandom(mib*1024**2)
    try:
        t = time.perf_counter()
        with p.open('xb') as f:
            f.write(block); f.flush(); os.fsync(f.fileno())
        write_seconds = time.perf_counter()-t
        t = time.perf_counter(); got = p.read_bytes(); read_seconds = time.perf_counter()-t
        if got != block:
            raise RuntimeError('AFS readback differed from write')
        return {"bytes": len(block), "write_fsync_seconds": write_seconds, "write_MB_s": len(block)/1e6/write_seconds, "read_seconds": read_seconds, "read_MB_s": len(block)/1e6/read_seconds, "read_kind": "warm read; page cache not flushed", "sha256": hashlib.sha256(got).hexdigest()}
    finally:
        p.unlink(missing_ok=True)  # Only the temporary file created above.

def probe(afs_root=None, io=False):
    mem = {}
    for line in (read('/proc/meminfo') or '').splitlines():
        k, v = line.split(':', 1); mem[k] = int(v.split()[0])*1024
    affinity = sorted(os.sched_getaffinity(0))
    raw, limits = cgroup_limits()
    cpus = min([len(affinity)] + [r['cpu_quota_cores'] for r in limits if 'cpu_quota_cores' in r])
    memory = min([mem.get('MemTotal', 0)] + [r['memory_limit_bytes'] for r in limits if 'memory_limit_bytes' in r])
    spares=[max(0,r['memory_limit_bytes']-int(r['memory.current'])) for r in limits if r.get('memory.current') and 'memory_limit_bytes' in r]
    available = min([mem.get('MemAvailable',memory),memory]+spares)
    model = next((x.split(':',1)[1].strip() for x in (read('/proc/cpuinfo') or '').splitlines() if x.startswith('model name')), None)
    rank = {k:os.environ.get(k) for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','GROUP_WORLD_SIZE','SENSECORE_PYTORCH_NNODES','SENSECORE_PYTORCH_NODE_RANK','LADDER_JOB_ID')}
    if afs_root and io:Path(afs_root).mkdir(parents=True,exist_ok=True)
    return {"schema": "cci_datagen_probe_v1", "host": socket.gethostname(), "pid": os.getpid(), "utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), "cpu_model": model, "nproc": command(['nproc']), "nproc_without_omp": command(['env','-u','OMP_NUM_THREADS','-u','OMP_THREAD_LIMIT','nproc']), "nproc_all": command(['nproc','--all']), "logical_cpu_count": os.cpu_count(), "affinity_cpus": affinity, "effective_cpu_cores": cpus, "cgroup_raw": raw, "cgroup_limits": limits, "memory_limit_bytes": memory, "memory_available_bytes": available, "memory_info": mem, "shm": storage('/dev/shm'), "local_disk": storage('/tmp'), "afs": storage(afs_root) if afs_root else None, "afs_io": io_probe(afs_root) if io and afs_root else None, "gpus": gpu_inventory(), "ranks": rank, "egl": {"libraries": {k:ctypes.util.find_library(k) for k in ('EGL','OpenGL','GLdispatch','cuda')}, "habitat_render": "unverified_cpu_only_probe", "note": "Actual EGL rendering is checked by the GPU pilot, never by this CPU-only command"}}

def allocate(reports, *, mode='dedicated', training_ranks=0, loader_workers=0, training_gib=33, worker_gib=4, cpu_reserve_fraction=.10, memory_reserve_fraction=.10):
    if mode == 'mixed' and (training_ranks <= 0 or loader_workers < 0):
        raise ValueError('mixed mode requires declared training ranks and loader workers')
    if worker_gib <= 0 or training_gib <= 0 or not 0 <= cpu_reserve_fraction < 1 or not 0 <= memory_reserve_fraction < 1:
        raise ValueError('invalid resource policy')
    by_host = {}
    for r in reports:
        by_host.setdefault(r['host'], []).append(r)
    result = {}
    for host, rows in by_host.items():
        rows.sort(key=lambda r:int(r['ranks'].get('RANK') or 0))
        union = set().union(*(set(r['affinity_cpus']) for r in rows))
        # Equal namespace limits are one shared pool; disjoint process limits can
        # be summed only when their affinity masks are also disjoint.
        disjoint = sum(len(r['affinity_cpus']) for r in rows) == len(union)
        cpu_groups={}
        for r in rows:
            limited=[x for x in r.get('cgroup_limits',[]) if x.get('cpu_quota_cores')==r['effective_cpu_cores']]
            identity=limited[-1].get('canonical_path',limited[-1]['path']) if limited else (tuple(r['affinity_cpus']) if disjoint else 'shared_unknown_scope')
            cpu_groups[identity]=max(cpu_groups.get(identity,0),r['effective_cpu_cores'])
        common_cpu=None
        for r in rows:
            scope={x.get('canonical_path',x['path']):x['cpu_quota_cores'] for x in r.get('cgroup_limits',[]) if 'cpu_quota_cores' in x}
            common_cpu=scope if common_cpu is None else {k:min(v,scope[k]) for k,v in common_cpu.items() if k in scope}
        cores = min([len(union),sum(cpu_groups.values())]+list((common_cpu or {}).values()))
        cpu_reserve = max(1, math.ceil(cores*cpu_reserve_fraction))
        usable = max(0, math.floor(cores)-cpu_reserve-training_ranks*loader_workers)
        memory_groups={};available_groups={}
        for r in rows:
            limited=[x for x in r.get('cgroup_limits',[]) if x.get('memory_limit_bytes')==r['memory_limit_bytes']]
            identity=limited[-1].get('canonical_path',limited[-1]['path']) if limited else 'shared_unknown_scope'
            memory_groups[identity]=max(memory_groups.get(identity,0),r['memory_limit_bytes'])
            available_groups[identity]=max(available_groups.get(identity,0),r['memory_available_bytes'])
        common_memory=None;common_available=None
        for r in rows:
            scope={x.get('canonical_path',x['path']):x['memory_limit_bytes'] for x in r.get('cgroup_limits',[]) if 'memory_limit_bytes' in x}
            free={x.get('canonical_path',x['path']):max(0,x['memory_limit_bytes']-int(x['memory.current'])) for x in r.get('cgroup_limits',[]) if 'memory_limit_bytes' in x and x.get('memory.current')}
            common_memory=scope if common_memory is None else {k:min(v,scope[k]) for k,v in common_memory.items() if k in scope}
            common_available=free if common_available is None else {k:min(v,free[k]) for k,v in common_available.items() if k in free}
        memory=min([sum(memory_groups.values()),min(r.get('memory_info',{}).get('MemTotal',sum(memory_groups.values())) for r in rows)]+list((common_memory or {}).values()))
        available=min([sum(available_groups.values()),min(r.get('memory_info',{}).get('MemAvailable',sum(available_groups.values())) for r in rows)]+list((common_available or {}).values()))
        spare = max(0, min(available, int(memory*(1-memory_reserve_fraction))-int(training_ranks*training_gib*GIB)))
        count = min(usable, int(spare//(worker_gib*GIB)))
        assignments = {int(r['ranks'].get('RANK') or 0):[] for r in rows}
        eligible = {int(r['ranks'].get('RANK') or 0):set(r['affinity_cpus']) for r in rows}
        quotas = {int(r['ranks'].get('RANK') or 0):min(math.floor(r['effective_cpu_cores']),int(r['memory_available_bytes']//(worker_gib*GIB))) for r in rows}
        for cpu in sorted(union):
            if sum(map(len, assignments.values())) >= count:
                break
            candidates = [rank for rank, cpuset in eligible.items() if cpu in cpuset and len(assignments[rank]) < quotas[rank]]
            if candidates:
                rank = min(candidates, key=lambda rank:(len(assignments[rank]),rank)); assignments[rank].append(cpu)
        result[host] = {"node_cpu_cores": cores, "reserved_cpu_cores": cpu_reserve, "training_loader_cpu_cores": training_ranks*loader_workers, "memory_spare_bytes": spare, "worker_memory_budget_bytes": int(worker_gib*GIB), "cpu_workers": {str(k):len(v) for k,v in assignments.items()}, "worker_affinity": {str(k):v for k,v in assignments.items()}, "total_cpu_workers": sum(map(len, assignments.values())), "memory_estimate": "runtime limit and available memory; training_gib includes loaders"}
    return result

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--afs-root'); p.add_argument('--io',action='store_true')
    a = p.parse_args(); r = probe(a.afs_root,a.io); write_json(a.output,r)
    print(json.dumps({k:r[k] for k in ('host','effective_cpu_cores','memory_limit_bytes','ranks','egl')},ensure_ascii=False))
if __name__ == '__main__':
    main()
