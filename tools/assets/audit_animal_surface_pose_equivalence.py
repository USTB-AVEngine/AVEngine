"""Bridge existing full CPU surface measurements to saved native posed meshes.

All topology and vertices are read back for every measured frame. A uniform
translation per frame is irrelevant to stretch and self intersection; any
other discrepancy above the existing 50 micrometre export tolerance prevents
reuse. This tool neither renders a room nor recomputes an acoustic response.
"""
import argparse, datetime, hashlib, json, time
from pathlib import Path
import numpy as np
from avengine.assets.roster_motion.io import read_glb
from avengine.assets.roster_motion.pose import pose
from avengine.assets.roster_motion.episode import triangle_mesh
from tools.assets.measure_native_motion_surface import actual_pose

TOLERANCE_M = 5e-5  # unchanged export_walk posed geometry tolerance

def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def audit(native_path, cpu_path):
    native, nh = read(native_path)
    cpu, ch = read(cpu_path)
    result = {'native_motion': str(native_path), 'native_sha256': nh, 'cpu_checks': str(cpu_path),
              'cpu_sha256': ch, 'asset_id': native['asset_id'], 'frames': [],
              'tolerance_m': TOLERANCE_M, 'same_topology_required': True,
              'allowed_transform': 'per-frame uniform translation only; self-surface invariant',
              'status': 'unverified', 'audio_status': 'deferred_to_production'}
    if native.get('source_master') != cpu.get('source_master'):
        result['reason'] = 'different_target_skin_source'
        return result
    sf = cpu.get('surface_checks', {})
    rows = native.get('frames', [])
    if len(rows) != len(cpu.get('frames', [])) or len(rows) != len(sf.get('frames', [])):
        result['reason'] = 'incomplete_or_different_frame_coverage'
        return result
    if any(f.get('pose_buffer_is_transient') for f in rows):
        result['reason'] = 'bounded_native_buffers_are_not_archived_per_frame_no_vertex_equivalence_claim'
        return result
    if sf.get('stream_metadata'):
        result['reason'] = 'streamed_surface_requires_its_explicit_evaluator_manifest_not_a_gltf_action'
        return result
    glb = Path(sf['input'])
    document, binary = read_glb(glb)
    action = next(a for a in document['animations'] if a.get('name') == sf.get('action', 'Walking_Repaired'))
    result['surface_pose_source'] = str(glb)
    result['surface_pose_sha256'] = hashlib.sha256(glb.read_bytes()).hexdigest()
    start = time.monotonic()
    for i, (n, c) in enumerate(zip(rows, cpu['frames'])):
        if n['time_s'] != c['time_s'] or sf['frames'][i]['frame'] != i:
            result['reason'] = 'surface_native_clock_differs'
            return result
        if n.get('source_phase_s') != c.get('source_phase_s'):
            result['reason'] = 'source_animation_phase_differs'
            return result
        parts, _ = pose(document, binary, action, n['time_s'], ground=False, canonical_flip=False)
        expected, faces = triangle_mesh(document, binary, parts)
        observed, observed_faces = actual_pose(n['pose_glb'])
        if not np.array_equal(faces, observed_faces) or expected.shape != observed.shape:
            result['reason'] = 'actual_native_topology_differs'
            return result
        delta = observed.astype(float) - expected
        shift = np.median(delta, axis=0)
        error = float(np.linalg.norm(delta - shift, axis=1).max())
        result['frames'].append({'frame': i, 'time_s': n['time_s'], 'vertex_count': len(expected),
                                  'uniform_translation_m': shift.tolist(), 'shape_error_m': error})
        if not np.isfinite(error) or error > TOLERANCE_M:
            result.update(status='fail', reason='actual_native_pose_not_equivalent_to_measured_surface')
            return result
    result.update(status='pass', maximum_shape_error_m=max(f['shape_error_m'] for f in result['frames']),
                  frame_count=len(rows), elapsed_s=time.monotonic()-start,
                  reason='every actual native posed mesh matches the existing surface-measured pose up to uniform translation')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    index, ih = read(a.index)
    a.output.mkdir(parents=True, exist_ok=True)
    results = []
    seen = set()
    for asset in index['assets']:
        for candidate in asset['walking_candidates']:
            for cpu in candidate['cpu_checks']:
                key = (candidate['native_motion'], cpu)
                if key in seen: continue
                seen.add(key)
                token = hashlib.sha256(('|'.join(key)).encode()).hexdigest()[:16]
                path = a.output / (asset['asset_id'] + '_' + token + '.json')
                native, nh = read(key[0]); cp, ch = read(cpu)
                if path.exists():
                    old = json.loads(path.read_text())
                    if old.get('native_sha256') == nh and old.get('cpu_sha256') == ch:
                        results.append({'asset_id': asset['asset_id'], 'status': old['status'], 'evidence': str(path)})
                        continue
                    raise RuntimeError('evidence changed; choose a new output version: ' + str(path))
                try: result = audit(key[0], cpu)
                except Exception as e:
                    result = {'asset_id': asset['asset_id'], 'status': 'unverified', 'reason': repr(e),
                              'native_motion': key[0], 'native_sha256': nh, 'cpu_checks': cpu, 'cpu_sha256': ch}
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                results.append({'asset_id': asset['asset_id'], 'status': result['status'], 'evidence': str(path)})
                print(asset['asset_id'], result['status'], result.get('maximum_shape_error_m', result.get('reason')), flush=True)
    summary = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source_index': str(a.index), 'source_index_sha256': ih, 'assets': results,
               'read_only_external_sources': True, 'native_rendering_run': False, 'audio_run': False}
    (a.output / 'index.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
