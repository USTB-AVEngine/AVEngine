"""Lossless minimal witnesses for the unchanged frozen-panel native verifier.

RGB is recovered from the already lossless public video; static native arrays
are summarized by one exact observed slice and multiplicity. Small native
geometry, depth-authority masks, RIRs, plans and explicit dependencies retain a
compressed lossless witness. Original inputs are never rewritten.
"""
import base64
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def blob(root, data):
    key = hashlib.sha256(data).hexdigest()
    target = root / 'blobs' / (key + '.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as f:
                f.write(data)
    return str(target.relative_to(root))


def encode_file(retention, path, *, public_video=None):
    path = Path(path)
    record = {'source': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    if path.suffix != '.npy':
        record.update(kind='file', payload=blob(retention, path.read_bytes()))
        return record
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.flags.f_contiguous and not array.flags.c_contiguous:
        record.update(kind='file', payload=blob(retention, path.read_bytes()))
        return record
    with path.open('rb') as f:
        header = f.read(array.offset)
    record.update(shape=list(array.shape), dtype=array.dtype.str,
                  npy_header=base64.b64encode(header).decode())
    if public_video is not None:
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] not in (3, 4):
            raise ValueError('native RGB cannot be losslessly reconstructed from the public RGB24 video')
        alpha = None
        if array.shape[-1] == 4:
            alpha = int(array[0, 0, 0, 3])
            if any(np.any(frame[..., 3] != alpha) for frame in array):
                raise ValueError('varying alpha requires an explicit witness; no silent dropping')
        record.update(kind='public_rgb', public_video=public_video, alpha_constant=alpha)
    elif array.ndim and len(array) and all(np.array_equal(x, array[0]) for x in array):
        record.update(kind='constant_axis0', payload=blob(retention, np.asarray(array[0]).tobytes()))
    else:
        record.update(kind='file', payload=blob(retention, path.read_bytes()))
    return record


def restore_file(root, record, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    retention = root / 'retention'
    kind = record['kind']
    if kind == 'public_media':
        source = (root / 'public' / record['public_media']).resolve()
        if not source.is_relative_to((root / 'public').resolve()):
            raise ValueError('retained public media escaped root')
        os.link(source, target)
    elif kind == 'file':
        with gzip.open(retention / record['payload'], 'rb') as source, target.open('xb') as out:
            for chunk in iter(lambda: source.read(1 << 20), b''):
                out.write(chunk)
    elif kind == 'constant_axis0':
        with gzip.open(retention / record['payload'], 'rb') as source:
            row = source.read()
        expected = int(np.prod(record['shape'][1:])) * np.dtype(record['dtype']).itemsize
        if len(row) != expected:
            raise ValueError('constant-slice witness shape differs')
        with target.open('xb') as out:
            out.write(base64.b64decode(record['npy_header']))
            for _ in range(record['shape'][0]):
                out.write(row)
    elif kind == 'public_rgb':
        frames, height, width, channels = record['shape']
        video = (root / 'public' / record['public_video']).resolve()
        if not video.is_relative_to((root / 'public').resolve()):
            raise ValueError('retained video escaped root')
        process = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(video),
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1'], stdout=subprocess.PIPE)
        count = 0
        with target.open('xb') as out:
            out.write(base64.b64decode(record['npy_header']))
            while data := process.stdout.read(height * width * 3):
                if len(data) != height * width * 3:
                    raise ValueError('partial retained RGB frame')
                rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                if channels == 4:
                    rgba = np.empty((height, width, 4), np.uint8)
                    rgba[..., :3] = rgb
                    rgba[..., 3] = record['alpha_constant']
                    data = rgba.tobytes()
                out.write(data)
                count += 1
        if process.wait() or count != frames:
            raise ValueError('lossless RGB witness clock/decode failed')
    else:
        raise ValueError('unknown retained witness kind')
    if target.stat().st_size != record['bytes'] or sha(target) != record['sha256']:
        raise ValueError('restored native file differs from pre-prune bytes: ' + record['source'])


@contextlib.contextmanager
def hydrated_group(root, manifest):
    """Materialize one disposable group, refusing undeclared external reads."""
    root = Path(root).resolve()
    temporary = root / 'retention' / 'temporary'
    temporary.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='verify_', dir=temporary) as dirname:
        work = Path(dirname)
        original = Path(manifest['production_root'])
        mapped = work / 'production'
        aliases = {}
        for record in manifest['native_files']:
            source = Path(record['source'])
            if not source.is_relative_to(original):
                raise ValueError('native witness is outside its declared parent')
            restore_file(root, record, mapped / source.relative_to(original))
        for record in manifest['dependencies']:
            destination = work / 'dependencies' / record['sha256']
            if not destination.exists():
                restore_file(root, record, destination)
            aliases[record['source']] = destination
            aliases[record.get('source_resolved', record['source'])] = destination

        def retained_path(value, *parts):
            p = Path(value, *parts)
            if p.is_relative_to(original):
                return mapped / p.relative_to(original)
            if str(p) in aliases:
                return aliases[str(p)]
            if not p.is_absolute() or p.is_relative_to(work):
                return p
            if p.is_relative_to(root / 'production'):
                raise ValueError('undeclared native parent dependency')
            if p.is_relative_to(root):
                return p
            raise ValueError('unretained external input: ' + str(p))

        yield retained_path


class RetainedSoundFile:
    """Resolve direct libsndfile reads as well as the native verifier's Path reads."""
    def __init__(self, original, resolver):
        self.original = original
        self.resolver = resolver
        self.reads = []

    def read(self, file, *args, **kwargs):
        target = self.resolver(file)
        self.reads.append({'declared': str(file), 'retained': str(target)})
        return self.original.read(target, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


def configuration_identity(config):
    """Bind every configuration field except the output directory changed by assembly."""
    value = dict(config)
    value.pop('output', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def require_configuration_binding(root, config, manifest):
    expected = manifest.get('configuration_normalized_sha256')
    if expected:
        if configuration_identity(config) != expected:
            raise ValueError('retained configuration differs from its frozen construction input')
        return
    # Legacy pilot already binds the exact configuration in its original private seal.
    path = root / 'inputs/config.json' if (root / 'inputs/config.json').exists() else root / 'inputs/shard_config.json'
    identity = read(root / 'scoring_private/build_code_identity.json')
    values = [value for key, value in identity['explicit_inputs_sha256'].items()
              if key.endswith('/inputs/' + path.name)]
    if len(values) != 1 or sha(path) != values[0]:
        raise ValueError('legacy retained configuration differs from its original scoring commitment')
