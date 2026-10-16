import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from avengine.timeline import v7_rir_work_pool as pool
from avengine.timeline.v7_rir_exact_reuse import exact_entry, identity_key
from avengine.timeline.v7_rir_submission import RIRSubmissionQueue

pytestmark = pytest.mark.fast_unit


@pytest.fixture
def policy(tmp_path):
    root = tmp_path / 'pool'
    root.mkdir()
    cpus = sorted(os.sched_getaffinity(0))[:2]
    value = {'cpu_ids': cpus, 'max_concurrent': len(cpus), 'reserve_bytes_per_rir': 1000}
    (root / 'policy.json').write_text(json.dumps(value))
    return root, value


def test_core_registration_release_and_stale_metadata(policy, monkeypatch):
    root, value = policy
    monkeypatch.setattr(pool, '_available_bytes', lambda: 100_000_000_000)
    shared = pool.RIRWorkPool(root)
    with pool.reserve_cpus(root, value['cpu_ids'], 'visual'):
        assert shared.try_acquire() is None
    lease = shared.try_acquire()
    assert lease.cpu in value['cpu_ids']
    lease.close()
    again = shared.try_acquire()
    assert again is not None
    again.close()


def test_concurrency_and_pending_memory_are_atomic(policy, monkeypatch):
    root, value = policy
    value['max_concurrent'] = 1
    (root / 'policy.json').write_text(json.dumps(value))
    monkeypatch.setattr(pool, '_available_bytes', lambda: 100_000_000_000)
    shared = pool.RIRWorkPool(root)
    first = shared.try_acquire()
    assert shared.try_acquire() is None
    first.close()
    monkeypatch.setattr(pool, '_available_bytes', lambda: pool.MIN_AVAILABLE_BYTES + 999)
    assert shared.try_acquire() is None
    monkeypatch.setattr(pool, '_available_bytes', lambda: pool.MIN_AVAILABLE_BYTES + 1000)
    lease = shared.try_acquire()
    assert lease is not None
    lease.close()


def test_low_memory_floor_rejected(policy):
    root, value = policy
    value['minimum_available_bytes'] = 39_999_999_999
    (root / 'policy.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='40 GB'):
        pool.RIRWorkPool(root)


def test_unavailable_pool_uses_original_launcher(tmp_path, monkeypatch):
    from avengine.timeline import v7_phase2_fresh_rir as fresh
    monkeypatch.setenv(pool.ENV_KEY, str(tmp_path / 'absent'))
    sentinel = object()
    calls = []
    monkeypatch.setattr(fresh.subprocess, 'Popen', lambda *a, **kw: calls.append((a, kw)) or sentinel)
    assert fresh._launch_native_child(['example'], None, {}) is sentinel
    assert calls[0][0] == (['example'],)


def test_real_child_cpu_binding_and_release(policy, monkeypatch, tmp_path):
    root, value = policy
    monkeypatch.setenv(pool.ENV_KEY, str(root))
    with (tmp_path / 'native.log').open('wb') as log:
        child = pool.RIRWorkPool(root).launch([sys.executable, '-c',
            'import os;print(sorted(os.sched_getaffinity(0)),flush=True)'], log)
        cpu = child.lease.cpu
        assert child.wait() == 0
    assert json.loads((tmp_path / 'native.log').read_text()) == [cpu]
    lease = pool.RIRWorkPool(root).try_acquire()
    assert lease is not None
    lease.close()


def test_inherited_cpu_lease_survives_coordinator_exit(policy, tmp_path):
    root, value = policy
    value['max_concurrent'] = 1
    (root / 'policy.json').write_text(json.dumps(value))
    marker = tmp_path / 'child.done'
    # Child outlives its immediate launcher; no kill or external process touch.
    script = '''import os,sys
from pathlib import Path
from avengine.timeline.v7_rir_work_pool import RIRWorkPool
with Path(sys.argv[2]).open('wb') as log:
 child=RIRWorkPool(sys.argv[1]).launch([sys.executable,'-c',"import time,pathlib;time.sleep(.4);pathlib.Path("+repr(sys.argv[3])+").write_text('done')"],log)
os._exit(0)
'''
    subprocess.run([sys.executable, '-c', script, str(root), str(tmp_path / 'log'), str(marker)], check=True)
    shared = pool.RIRWorkPool(root)
    assert shared.try_acquire() is None
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert marker.exists()
    deadline = time.monotonic() + 5
    lease = None
    while lease is None and time.monotonic() < deadline:
        lease = shared.try_acquire()
        if lease is None:
            time.sleep(.02)
    assert lease is not None
    lease.close()


def request(tmp_path):
    target = tmp_path / 'render'
    target.mkdir()
    r = {'inputs': {'runtime': {}}, 'expected_native_identity': {'simulation': {'thread_count': 1}, 'listener_position_m': [0, 0, 0], 'package_id': 'house-a'},
         'source_position_m': [1., 2., 3.], 'evidence_role': 'primary', 'reference_trial': None,
         'output': str(target / 'rir.npy'), 'report': str(target / 'result.json'), 'request_id': 'origin'}
    (target / 'request.json').write_text(json.dumps(r))
    return r


def test_exact_identity_keeps_runtime_pose_house_parameters_and_roles(tmp_path):
    import copy
    r = request(tmp_path)
    original = identity_key(r)[0]
    for changes in [('source_position_m', [1., 2., 3.00000000000001]), ('evidence_role', 'independent_midpoint_reference'), ('reference_trial', 0)]:
        other = copy.deepcopy(r)
        other[changes[0]] = changes[1]
        assert identity_key(other)[0] != original
    for key, val in [('package_id', 'house-b'), ('listener_position_m', [0, 0, 1]), ('simulation', {'thread_count': 1, 'rays': 10})]:
        other = copy.deepcopy(r)
        other['expected_native_identity'][key] = val
        assert identity_key(other)[0] != original
    other = copy.deepcopy(r)
    other['inputs']['runtime'] = {'revision': 2}
    assert identity_key(other)[0] != original
    other = copy.deepcopy(r)
    other['request_id'] = 'different-attempt'
    other['output'] = '/different/output.npy'
    other['inputs']['fresh_rir_workers'] = 88
    assert identity_key(other)[0] == original


def test_exact_reuse_copies_wave_and_retains_original_report(tmp_path, monkeypatch):
    import hashlib
    r = request(tmp_path)
    monkeypatch.setenv('AVENGINE_V7_RIR_EXACT_CACHE', str(tmp_path / 'cache'))
    wave = np.arange(32, dtype=np.float32).reshape(4, 8)
    np.save(r['output'], wave)
    sha = hashlib.sha256(Path(r['output']).read_bytes()).hexdigest()
    Path(r['report']).write_text(json.dumps({'request_id': r['request_id'], 'output_sha256': sha}))
    with exact_entry(r) as entry:
        assert entry.hit(r) is None
        entry.publish(r, r['report'])
    other = {**r, 'request_id': 'consumer', 'output': str(tmp_path / 'new' / 'rir.npy')}
    Path(other['output']).parent.mkdir()
    with exact_entry(other) as entry:
        result = entry.hit(other)
        assert np.array_equal(result[0], wave)
        assert result[1] == r['report'] and result[2] == 0
    assert Path(other['output']).read_bytes() == Path(r['output']).read_bytes()
    other['reference_trial'] = 1
    other['evidence_role'] = 'independent_midpoint_reference'
    with exact_entry(other) as entry:
        assert entry.hit(other) is None


def test_reference_trials_never_coalesce(tmp_path):
    r = request(tmp_path)
    keys = {identity_key({**r, 'evidence_role': 'independent_midpoint_reference', 'reference_trial': trial})[0] for trial in range(3)}
    assert len(keys) == 3


def test_submission_trials_and_primary_overlap_without_changing_consumption_order():
    started = threading.Barrier(4)
    calls = []
    class Renderer:
        def _render_one(self, position, role, trial):
            calls.append((role, trial))
            started.wait(timeout=5)
            return np.full((4, 8), trial if trial is not None else 10), role, trial
    queue = RIRSubmissionQueue(Renderer(), 4)
    try:
        queue.primary([[1, 2, 3]])
        queue.references([[1, 2, 3]])
        assert queue.get([1, 2, 3], 'primary', None)[0][0, 0] == 10
        for trial in range(3):
            assert queue.get([1, 2, 3], 'independent_midpoint_reference', trial)[0][0, 0] == trial
        assert len(calls) == 4
    finally:
        queue.executor.shutdown()
