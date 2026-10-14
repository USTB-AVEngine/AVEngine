"""Audit actual UE AnimPose skin against the target's unchanged binding geometry."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import psutil

from avengine.assets.animal_motion.profiles import PAWS, Rig, paw_points
from avengine.assets.roster_motion.episode import triangle_mesh
from avengine.assets.roster_motion.io import read_glb
from avengine.assets.roster_motion.pose import accessor, globals_at, pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=900.)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 900:
        raise ValueError('owner_requires_at_most_15_minutes_per_case')
    started = time.monotonic()
    evidence = next(x for x in json.loads(args.manifest.read_text())['assets'] if x['case'] == args.case)
    motion = json.loads(Path(evidence['source_motion_metadata']).read_text())
    native = json.loads(Path(evidence['native']).read_text())
    record = next(x for x in json.loads(args.inventory.read_text())['assets'] if x['asset_id'] == evidence['asset_id'])
    rig = Rig.load(record)
    if rig.path.resolve() != Path(evidence['source_rig']).resolve():
        raise ValueError('registered_source_rig_differs')
    args.output.mkdir(parents=True, exist_ok=False)
    document, binary = read_glb(evidence['imported_geometry_glb'])
    source = rig.document
    basis = np.asarray(motion['basis_baked_in_mesh_and_inverse_binds'], float)
    source_default = globals_at(source, rig.binary)
    source_meshes = [i for i, n in enumerate(source['nodes']) if 'mesh' in n]
    metric_meshes = [i for i, n in enumerate(document['nodes']) if 'mesh' in n]
    if len(source_meshes) != len(metric_meshes):
        raise ValueError('source_mesh_instance_count_changed')
    name_to_metric = {n.get('name'): i for i, n in enumerate(document['nodes'])}
    binding_source = dict(source_default)
    recovered = {}
    rest_parts = []
    for old_index, metric_index in zip(source_meshes, metric_meshes):
        old_node, node = source['nodes'][old_index], document['nodes'][metric_index]
        old_primitives = source['meshes'][old_node['mesh']]['primitives']
        primitives = document['meshes'][node['mesh']]['primitives']
        if len(old_primitives) != len(primitives):
            raise ValueError('source_primitive_count_changed')
        mesh_matrix = source_default[old_index]
        source_skin = source['skins'][old_node['skin']]
        inverse_binds = accessor(source, rig.binary, source_skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
        for joint, inverse in zip(source_skin['joints'], inverse_binds):
            candidate = mesh_matrix @ np.linalg.inv(inverse)
            if joint in recovered and not np.allclose(recovered[joint], candidate, atol=1e-6, rtol=0):
                raise ValueError('inconsistent_original_inverse_binding')
            recovered[joint] = candidate
        for primitive_index, (old, primitive) in enumerate(zip(old_primitives, primitives)):
            old_faces = accessor(source, rig.binary, old['indices'])
            current_faces = accessor(document, binary, primitive['indices'])
            if not np.array_equal(old_faces, current_faces):
                raise ValueError('original_target_triangle_topology_changed')
            for attribute in ('JOINTS_0', 'WEIGHTS_0'):
                if not np.array_equal(accessor(source, rig.binary, old['attributes'][attribute]),
                                      accessor(document, binary, primitive['attributes'][attribute])):
                    raise ValueError('original_target_skin_influences_changed')
            positions = accessor(source, rig.binary, old['attributes']['POSITION'])
            matrix = basis @ mesh_matrix
            rest_parts.append({'node': metric_index, 'primitive': primitive_index,
                               'positions': positions @ matrix[:3, :3].T + matrix[:3, 3]})
    binding_source.update(recovered)
    binding = {i: np.eye(4) for i in range(len(document['nodes']))}
    for joint in recovered:
        name = source['nodes'][joint]['name']
        binding[name_to_metric[name]] = basis @ binding_source[joint]
    rest_vertices, rest_faces = triangle_mesh(document, binary, rest_parts)
    reproduced, _ = pose(document, binary, None, globals_override=binding, ground=False, canonical_flip=False)
    reproduced_vertices, reproduced_faces = triangle_mesh(document, binary, reproduced)
    if not np.array_equal(rest_faces, reproduced_faces):
        raise ValueError('binding_reconstruction_topology_changed')
    rest_error = float(np.max(np.linalg.norm(rest_vertices - reproduced_vertices, axis=1)))
    if rest_error > 1e-6:
        raise ValueError('inverse_bind_does_not_reproduce_original_binding_geometry:' + str(rest_error))
    region_source = [rig.body, rig.head, rig.tail, *[rig.chains[side][2] for side in PAWS]]
    region_names = [source['nodes'][i]['name'] for i in region_source]
    region_nodes = [name_to_metric[name] for name in region_names]
    binding_regions = np.asarray([binding[i] for i in region_nodes])
    paws = paw_points(rest_parts, rig.masks)
    head = np.concatenate([part['positions'][mask] for part, mask in zip(rest_parts, rig.head_masks) if mask.any()])
    landmarks = {f'foot_{i}': paws[side]['point'].tolist() for i, side in enumerate(PAWS)}
    landmarks.update(head_bbox_min=head.min(0).tolist(), head_bbox_max=head.max(0).tolist())
    rest_file = args.output / 'unchanged_binding_geometry.npz'
    np.savez_compressed(rest_file, positions=rest_vertices.astype('<f4'), faces=rest_faces.astype(np.int64), region_matrices=binding_regions)
    metadata = {'rest_npz': str(rest_file), 'registration': {'landmarks_local_m': landmarks},
                'source_master': str(rig.path), 'native_motion': evidence['native'],
                'actual_pose_evidence': str(Path(evidence['actual_local_globals_npz']).parent / 'evidence.json'),
                'frame_count': len(native['frames']), 'source_coordinate_system': 'metric +Y up, exporter basis; rigid room root excluded',
                'position_encoding': 'float32 skin reconstructed from actual UE AnimPose WORLD joint readbacks; source target topology and weights unchanged',
                'reference_pose': 'unmodified original source primitive POSITION transformed only by original mesh global and exported coordinate basis; not Idle animation',
                'inverse_bind_source_reconstruction_error_m': rest_error,
                'region_matrix_scope': 'all seven matrices from actual UE AnimPose; only Body rotation is used by this cross-region depth policy',
                'evidence_boundary': evidence['evidence_boundary'],
                'runtime_correspondence': {key: evidence[key] for key in ('maximum_runtime_vs_source_joint_position_error_m', 'maximum_runtime_clock_error_s', 'first_runtime_full_matrix_comparison')},
                'no_new_UE_render_or_RIR': True}
    meta_file = args.output / 'actual_UE_skin_stream.json'
    meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    surface_file = args.output / 'surface.json'
    checker = Path(__file__).resolve().parents[1] / 'assets/check_motion_only_surface_blender.py'
    command = [str(args.blender), '-b', '--factory-startup', '--disable-autoexec', '-t', '1',
               '--python-exit-code', '2', '--python', str(checker), '--', '--glb', str(rig.path),
               '--output', str(surface_file), '--frames', str(len(native['frames'])),
               '--skin-stream-meta', str(meta_file), '--validation-policy', 'owner_relaxed_20261014']
    remaining = args.max_seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError('per_case_15_minute_box_before_surface_worker')
    data = np.load(evidence['actual_local_globals_npz'], allow_pickle=False)
    node_indices = data['joint_node_indices']
    actual = data['actual_local_globals']
    times = data['time_s']
    if len(actual) != len(native['frames']) or len(actual) != 121:
        raise ValueError('full_121_actual_UE_poses_required')
    if actual.shape != (121, len(node_indices), 4, 4) or not np.isfinite(actual).all():
        raise ValueError('finite_complete_actual_UE_joint_matrices_required')
    if not np.allclose(times, [f['time_s'] for f in native['frames']], atol=1e-10, rtol=0):
        raise ValueError('actual_import_and_room_replay_clocks_differ')
    child = subprocess.Popen(command, stdin=subprocess.PIPE)
    process = psutil.Process(child.pid)
    created = process.create_time()
    stops = []
    timed_out = threading.Event()

    def stop_at_deadline():
        timed_out.set()
        try:
            if process.create_time() != created or not process.is_running():
                return
            for item in [process, *process.children(recursive=True)]:
                try:
                    if item.status() != psutil.STATUS_ZOMBIE:
                        stops.append({'pid': item.pid, 'created': item.create_time(), 'command': item.cmdline()})
                except psutil.Error:
                    pass
            for entry in reversed(stops):
                try:
                    item = psutil.Process(entry['pid'])
                    if item.create_time() == entry['created'] and item.cmdline() == entry['command']:
                        item.terminate()
                except psutil.Error:
                    pass
            time.sleep(3)
            for entry in reversed(stops):
                try:
                    item = psutil.Process(entry['pid'])
                    if item.create_time() == entry['created'] and item.is_running() and item.status() != psutil.STATUS_ZOMBIE:
                        item.kill()
                except psutil.Error:
                    pass
        except psutil.Error:
            pass

    timer = threading.Timer(remaining, stop_at_deadline)
    timer.daemon = True
    timer.start()
    sent = 0
    stream_error = None
    try:
        for index, timestamp in enumerate(times):
            globals_actual = {i: np.eye(4) for i in range(len(document['nodes']))}
            globals_actual.update({int(node): matrix for node, matrix in zip(node_indices, actual[index])})
            parts, _ = pose(document, binary, None, globals_override=globals_actual, ground=False, canonical_flip=False)
            vertices, faces = triangle_mesh(document, binary, parts)
            if not np.array_equal(faces, rest_faces):
                raise ValueError('actual_pose_topology_changed')
            matrices = np.asarray([globals_actual[i] for i in region_nodes])
            child.stdin.write(np.asarray([timestamp], dtype='<f8').tobytes())
            child.stdin.write(matrices.astype('<f8').tobytes())
            child.stdin.write(vertices.astype('<f4').tobytes())
            sent += 1
    except BrokenPipeError:
        pass
    except BaseException as error:
        stream_error = repr(error)
    finally:
        try:
            child.stdin.close()
        except BrokenPipeError:
            pass
        code = child.wait()
        timer.cancel()
        timer.join()
    receipt = {'asset_id': evidence['asset_id'], 'case': args.case, 'native_motion': evidence['native'],
               'actual_UE_pose_evidence': str(meta_file), 'surface': str(surface_file), 'source_master': str(rig.path),
               'exit_code': code, 'frames_sent': sent, 'elapsed_s': time.monotonic() - started,
               'per_case_limit_s': args.max_seconds, 'timebox': timed_out.is_set(),
               'verified_timebox_stops': stops, 'stream_error': stream_error, 'original_files_modified': False,
               'audio_deferred_to_production': True, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (args.output / 'execution.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt), flush=True)
    raise SystemExit(124 if timed_out.is_set() else 2 if stream_error else code)


if __name__ == '__main__':
    main()
