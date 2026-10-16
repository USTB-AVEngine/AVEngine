"""Host-wide exact-request reuse, with original native process evidence.

A primary kernel can never satisfy a reference request. Each of the three
reference trials has its own identity and independently produced native file.
The digest is an index for exact identities, not a new acoustic admission gate:
without it a shared cache could return a different pose/runtime/trial's kernel.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import numpy as np

ENV_KEY = 'AVENGINE_V7_RIR_EXACT_CACHE'


def _runtime_files(value):
    records = {}
    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and item.startswith('/'):
            path = Path(item)
            if path.is_file():
                stat = path.stat()
                records[str(path.resolve())] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    visit(value)
    return records


def exact_identity(request):
    native = request['expected_native_identity']
    if native['simulation']['thread_count'] != 1:
        raise ValueError('exact reuse requires the unchanged one-thread profile')
    # Include source adapter code and installed native library/file revisions;
    # runtime path alone is insufficient when an installation is replaced.
    src = Path(__file__).resolve().parents[1]
    code = {}
    for relative in ('timeline/v7_phase2_fresh_rir.py', 'acoustics/rir_cache.py', 'acoustics/runtime.py'):
        code[relative] = hashlib.sha256((src / relative).read_bytes()).hexdigest()
    pilot = src.parent.parent / 'tools/timeline/pilot_continuous_scenes.py'
    code['native_renderer'] = hashlib.sha256(pilot.read_bytes()).hexdigest()
    return {'native_identity': native, 'source_position_m': request['source_position_m'],
            'evidence_role': request['evidence_role'], 'reference_trial': request['reference_trial'],
            'runtime_inputs': request['inputs']['runtime'],
            'runtime_files': _runtime_files(native), 'adapter_code': code}


def identity_key(request):
    identity = exact_identity(request)
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest(), identity


class ExactReuseEntry:
    def __init__(self, directory, identity):
        self.directory = directory
        self.identity = identity

    def hit(self, request):
        manifest = self.directory / 'entry.json'
        if not manifest.is_file():
            return None
        entry = json.loads(manifest.read_text())
        if entry['identity'] != self.identity:
            return None
        origin = json.loads(Path(entry['native_report']).read_text())
        source_request = json.loads(Path(entry['native_request']).read_text())
        _, source_identity = identity_key(source_request)
        if source_identity != self.identity or origin['request_id'] != source_request['request_id']:
            raise ValueError('exact cache native origin changed')
        source = self.directory / 'rir.npy'
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if sha != entry['output_sha256'] or sha != origin['output_sha256']:
            raise ValueError('exact cache waveform differs from native origin')
        wave = np.load(source, allow_pickle=False)
        if wave.ndim != 2 or wave.shape[0] != 4 or not np.isfinite(wave).all() or not np.any(wave):
            raise ValueError('invalid exact cache waveform')
        output = Path(request['output'])
        with output.open('xb') as target, source.open('rb') as stream:
            shutil.copyfileobj(stream, target)
        receipt = {'status': 'exact_reuse', 'native_report': entry['native_report'],
                   'native_request': entry['native_request'], 'output_sha256': sha,
                   'new_native_processes': 0, 'original_process_evidence_retained': True}
        (output.parent / 'exact_reuse.json').write_text(json.dumps(receipt))
        return wave, entry['native_report'], 0.

    def publish(self, request, report):
        output = Path(request['output'])
        origin = json.loads(Path(report).read_text())
        sha = hashlib.sha256(output.read_bytes()).hexdigest()
        if origin['output_sha256'] != sha:
            raise ValueError('native output changed before exact reuse publication')
        # Identity flock serializes publishers. Files are private until the
        # manifest is atomically committed; killed copies leave only own partials.
        partial = self.directory / (uuid.uuid4().hex + '.partial')
        with partial.open('xb') as target, output.open('rb') as stream:
            shutil.copyfileobj(stream, target)
        os.replace(partial, self.directory / 'rir.npy')
        entry = {'identity': self.identity, 'native_request': str(output.parent / 'request.json'),
                 'native_report': str(report), 'output_sha256': sha}
        partial = self.directory / (uuid.uuid4().hex + '.partial')
        partial.write_text(json.dumps(entry, allow_nan=False))
        os.replace(partial, self.directory / 'entry.json')


@contextmanager
def exact_entry(request):
    raw = os.environ.get(ENV_KEY)
    if not raw:
        yield None
        return
    # Cache unavailability may fall back; malformed retained evidence must not.
    try:
        key, identity = identity_key(request)
        directory = Path(raw) / key[:2] / key
        directory.mkdir(parents=True, exist_ok=True)
        stream = (directory / 'identity.lock').open('a+')
    except OSError:
        yield None
        return
    with stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield ExactReuseEntry(directory, identity)
