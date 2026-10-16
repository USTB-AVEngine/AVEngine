"""Opt-in, host-local single-core leases for unchanged fresh RIR processes.

A policy's cpu_ids are an explicit grant, never inferred from low utilization.
Producers hold reserve_cpus() for their main/visual cores. Kernel flock leases
survive a coordinator crash in the native child and release on process exit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ENV_KEY = 'AVENGINE_V7_RIR_POOL'
MIN_AVAILABLE_BYTES = 40_000_000_000


def _available_bytes():
    return next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines()
                if s.startswith('MemAvailable:'))


def _pss(pid, start_ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[19] != str(start_ticks):
            return 0
        return next(int(s.split()[1]) * 1024 for s in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()
                    if s.startswith('Pss:'))
    except (OSError, StopIteration, IndexError):
        return 0


def _try_lock(path):
    stream = path.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def _write(stream, record):
    stream.seek(0)
    stream.truncate()
    json.dump(record, stream)
    stream.flush()


@dataclass
class CPULease:
    stream: object
    cpu: int
    record: dict

    def close(self):
        if self.stream is not None:
            # Closing, rather than LOCK_UN, lets an inherited native child keep
            # this same open-file-description lease after coordinator failure.
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class RIRWorkPool:
    def __init__(self, root):
        self.root = Path(root)
        self.policy = json.loads((self.root / 'policy.json').read_text())
        self._validate(self.policy)
        (self.root / 'cores').mkdir(exist_ok=True)

    @staticmethod
    def _validate(policy):
        cpus = policy['cpu_ids']
        if not cpus or len(cpus) != len(set(cpus)) or any(type(c) is not int or c < 0 or c >= os.cpu_count() for c in cpus):
            raise ValueError('pool requires explicitly granted CPU IDs')
        if not 1 <= int(policy['max_concurrent']) <= len(cpus):
            raise ValueError('invalid pool concurrency')
        if int(policy.get('minimum_available_bytes', MIN_AVAILABLE_BYTES)) < MIN_AVAILABLE_BYTES:
            raise ValueError('pool must retain at least 40 GB MemAvailable')
        if int(policy.get('reserve_bytes_per_rir', 2_000_000_000)) <= 0:
            raise ValueError('invalid native memory reservation')

    @classmethod
    def configured(cls):
        raw = os.environ.get(ENV_KEY)
        if not raw:
            return None
        try:
            return cls(raw)
        except (OSError, ValueError, KeyError, TypeError):
            return None  # unavailable optional service: use the original worker

    def try_acquire(self, request=''):
        with (self.root / 'admission.lock').open('a+') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            policy = json.loads((self.root / 'policy.json').read_text())
            self._validate(policy)
            if not policy.get('enabled', True):
                raise OSError('RIR pool disabled')
            active = 0
            pending = 0
            candidate = None
            try:
                # Inspect every live lease, including removed policy CPU IDs;
                # stale unlocked JSON never reserves memory or concurrency.
                for path in sorted((self.root / 'cores').glob('*.lock')):
                    probe = _try_lock(path)
                    if probe is not None:
                        probe.close()
                        continue
                    record = json.loads(path.read_text())
                    if record.get('kind') == 'rir':
                        active += 1
                        pending += max(0, int(record['reserved_bytes']) - _pss(record.get('child_pid', 0), record.get('child_start_ticks', '')))
                reserve = int(policy.get('reserve_bytes_per_rir', 2_000_000_000))
                if active >= int(policy['max_concurrent']) or _available_bytes() - pending - reserve < int(policy.get('minimum_available_bytes', MIN_AVAILABLE_BYTES)):
                    return None
                for cpu in policy['cpu_ids']:
                    candidate = _try_lock(self.root / 'cores' / f'{cpu:04d}.lock')
                    if candidate is None:
                        continue
                    record = {'kind': 'rir', 'cpu_id': cpu, 'coordinator_pid': os.getpid(),
                              'reserved_bytes': reserve, 'request': str(request), 'acquired_unix': time.time()}
                    _write(candidate, record)
                    lease = CPULease(candidate, cpu, record)
                    candidate = None
                    return lease
                return None
            finally:
                if candidate is not None:
                    candidate.close()

    def acquire(self, request=''):
        while True:
            lease = self.try_acquire(request)
            if lease is not None:
                return lease
            time.sleep(.1)

    def launch(self, command, log, request=''):
        lease = self.acquire(request)
        try:
            env = {**os.environ, 'AVENGINE_V7_RIR_POOL_CPU': str(lease.cpu),
                   'AVENGINE_V7_RIR_POOL_LEASE_FD': str(lease.stream.fileno())}
            child = subprocess.Popen(['taskset', '--cpu-list', str(lease.cpu), *command],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                pass_fds=(lease.stream.fileno(),))
            try:start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            except FileNotFoundError:start_ticks=''  # already-exited short child
            lease.record.update(child_pid=child.pid,child_start_ticks=start_ticks)
            with (self.root / 'admission.lock').open('a+') as guard:
                fcntl.flock(guard, fcntl.LOCK_EX)
                _write(lease.stream, lease.record)
            return PooledChild(child, lease)
        except BaseException:
            lease.close()
            raise


class PooledChild:
    def __init__(self, child, lease):
        self.child = child
        self.lease = lease
        self.pid = child.pid

    def wait(self):
        try:
            return self.child.wait()
        finally:
            self.lease.close()


@contextmanager
def reserve_cpus(root, cpu_ids, owner):
    """Hold main/visual core leases before launching a producer; never preempt.

    Wait for existing native children to finish. All reservers take the same
    registration lock to avoid partial-set deadlock. RIR work can use the rest.
    """
    pool = RIRWorkPool(root)
    streams = []
    with (pool.root / 'registration.lock').open('a+') as registration:
        fcntl.flock(registration, fcntl.LOCK_EX)
        try:
            for cpu in sorted(set(cpu_ids)):
                if type(cpu) is not int or cpu < 0 or cpu >= os.cpu_count():
                    raise ValueError('invalid reserved CPU')
                # Use admission lock for atomic metadata publication.
                while True:
                    with (pool.root / 'admission.lock').open('a+') as guard:
                        fcntl.flock(guard, fcntl.LOCK_EX)
                        stream = _try_lock(pool.root / 'cores' / f'{cpu:04d}.lock')
                        if stream is not None:
                            _write(stream, {'kind': 'producer', 'cpu_id': cpu, 'owner': str(owner), 'pid': os.getpid()})
                            streams.append(stream)
                            break
                    time.sleep(.1)
            # Release registration serialization while holding the core leases.
            fcntl.flock(registration, fcntl.LOCK_UN)
            yield tuple(stream.fileno() for stream in streams)
        finally:
            for stream in reversed(streams):
                stream.close()


def inherited_lease():
    """Only bypass the legacy task-local limiter when a live pool FD exists."""
    try:
        fd = int(os.environ['AVENGINE_V7_RIR_POOL_LEASE_FD'])
        cpu = int(os.environ['AVENGINE_V7_RIR_POOL_CPU'])
        os.fstat(fd)
        if os.sched_getaffinity(0) != {cpu}:
            raise ValueError('native process differs from leased CPU')
        return {'cpu_id': cpu, 'inherited_lease_fd': fd, 'pool': os.environ[ENV_KEY]}
    except KeyError:
        return None


def scheduler_workers(inputs):
    pool = RIRWorkPool.configured()
    if pool is not None and pool.policy.get('enabled', True):
        return int(pool.policy['max_concurrent'])
    return min(16, max(1, int(inputs.get('fresh_rir_workers', 1))))


def launch_or_local(command, log, local, request=''):
    shared = RIRWorkPool.configured()
    if shared is not None:
        try:
            return shared.launch(command, log, request)
        except OSError:
            log.write(b'RIR pool unavailable; falling back to bounded local worker\n')
            log.flush()
    return local()


def _cpu_list(raw):
    cpus = set()
    for part in raw.split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-', 1))
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    actions = parser.add_subparsers(dest='action', required=True)
    initialize = actions.add_parser('init', help='create a new explicit CPU grant; never overwrite a policy')
    initialize.add_argument('--cpus', required=True)
    initialize.add_argument('--max-concurrent', type=int, required=True)
    initialize.add_argument('--reserve-bytes-per-rir', type=int, default=2_000_000_000)
    reserve = actions.add_parser('run', help='register producer cores before launching its process')
    reserve.add_argument('--cpus', required=True)
    reserve.add_argument('--owner', required=True)
    reserve.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cpus = _cpu_list(args.cpus)
    if args.action == 'init':
        policy = {'enabled': True, 'cpu_ids': cpus, 'max_concurrent': args.max_concurrent,
                  'minimum_available_bytes': MIN_AVAILABLE_BYTES,
                  'reserve_bytes_per_rir': args.reserve_bytes_per_rir}
        RIRWorkPool._validate(policy)
        root = Path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'policy.json').open('x') as stream:
            json.dump(policy, stream, indent=2)
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('run requires a producer command')
    with reserve_cpus(args.root, cpus, args.owner) as descriptors:
        env = {**os.environ, ENV_KEY: str(Path(args.root).resolve())}
        child = subprocess.Popen(['taskset', '--cpu-list', ','.join(map(str, cpus)), *command],
                                 stdin=subprocess.DEVNULL, env=env, pass_fds=descriptors)
        return child.wait()


if __name__ == '__main__':
    raise SystemExit(main())
