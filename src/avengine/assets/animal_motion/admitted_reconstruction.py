"""Rebuild transient qualified pose buffers from their recorded source recipe.

The source receipt is immutable. Rebuilt meshes must match its actual native
pose bounds and skeletal landmarks before the ordinary new-house gates run.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def source_rays(path, expected_sha, rotation_json):
    from avengine.assets.animal_motion.stage import OriginalVerticalStageRay
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    if digest(path) != expected_sha:
        raise ValueError('qualified original stage changed')
    rotation = np.asarray(json.loads(rotation_json), float)
    if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7) or abs(np.linalg.det(rotation) - 1) > 1e-7:
        raise ValueError('recorded stage transform is not a rigid rotation')
    mesh = indexed_static_mesh(path).copy()
    mesh.vertices = np.asarray(mesh.vertices) @ rotation.T
    return OriginalVerticalStageRay(path, mesh=mesh)


def reconstruct_frames(native, species, output, indices=None):
    from avengine.assets.animal_motion.profiles import Rig
    from avengine.assets.animal_motion.walk import ConstrainedWalk
    from avengine.assets.roster_motion.contact import yaw_matrix
    from avengine.assets.roster_motion.episode import triangle_mesh, externalize_unchanged_images
    from avengine.assets.roster_motion.pose import rigid_glb
    frames = deepcopy(native['frames'])
    requested = list(range(len(frames))) if indices is None else sorted(set(indices))
    transient = [i for i in requested if frames[i].get('pose_buffer_is_transient')]
    if not transient:
        if any(not Path(frames[i]['pose_glb']).is_file() for i in requested):
            raise ValueError('retained qualified pose missing')
        return frames, None
    if native['motion_mode'] == 'slide_no_animation':
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        paths = {frames[i]['pose_glb'] for i in requested}
        if native['checks']['status'] != 'pass' or any(not Path(p).is_file() for p in paths):
            raise ValueError('qualified static slide buffer missing')
        if len({digest(p) for p in paths}) != 1:
            raise ValueError('slide buffers contain changing mesh bytes')
        canonical = frames[requested[0]]['pose_glb']
        mesh = indexed_static_mesh(canonical)
        actual = np.stack((mesh.vertices.min(0), mesh.vertices.max(0)))
        for i in requested:
            if np.max(np.abs(actual - np.asarray(frames[i]['expected_pose_bounds_m']))) > 2e-5:
                raise ValueError('qualified slide buffer geometry differs from native readback')
            frames[i]['original_transient_pose_glb'] = frames[i]['pose_glb']
            frames[i]['pose_glb'] = canonical
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        receipt = {'schema': 'qualified-static-slide-buffer-view-v1', 'asset_id': native['asset_id'],
                   'canonical_pose': canonical, 'pose_sha256': digest(canonical),
                   'verified_byte_identical_source_buffers': sorted(paths), 'frames_verified': requested,
                   'joint_animation_present': False, 'original_files_modified': False,
                   'new_house_physical_admission_required': True}
        destination = output / 'RECONSTRUCTION_RECEIPT.json'
        destination.write_text(json.dumps(receipt, indent=2) + '\n')
        return frames, str(destination)
    if native['motion_mode'] != 'walk_animation' or native['checks']['status'] != 'pass':
        raise ValueError('transient poses need a passed animated native receipt')
    first = frames[transient[0]]['pose_reconstruction']
    master = first['source_master']
    candidate = first['source_inventory']
    if not candidate:
        raise ValueError('transient pose lacks source rig registration')
    record = {'asset_id': native['asset_id'], 'species': species,
              'files': {'motion_master_glb': {'path': master}}, 'walking_candidate': candidate}
    explicit = native.get('source_provenance', {}).get('explicit_motion_profile')
    if explicit:
        record['motion_profile'] = explicit
    rig = Rig.load(record).enable_skin_cache()
    route = first['route']
    p0 = np.asarray(route['root_m'], float)
    rotation = yaw_matrix(float(route['yaw_rad']))
    generator = native.get('gait_generation_floor_model')
    if generator:
        if generator['type'] != 'fixed_plane':
            raise ValueError('unsupported recorded gait floor model')
        height = float(generator['local_height_m'])
        if not math.isfinite(height):
            raise ValueError('nonfinite recorded gait floor')
        floor = lambda point: height
    else:
        transform = native['stage_transform']
        rays = source_rays(transform['source'], transform['source_sha256'],
                           json.dumps(transform['source_coordinate_to_native_world_rotation']))
        class RecordedRouteFloor:
            def many(self, points):
                return rays.many(np.asarray(points) @ rotation.T + p0) - p0[1]
            def __call__(self, point):
                return float(self.many([point])[0])
        floor = RecordedRouteFloor()
    walk = ConstrainedWalk(rig, speed=float(first['speed_m_s']), period=float(first['period_s']), floor=floor)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    document = externalize_unchanged_images(rig.document, rig.binary, output / 'textures')
    checks = []
    for i in transient:
        old = frames[i]
        recipe = old['pose_reconstruction']
        if any(recipe[k] != first[k] for k in ('source_master', 'source_inventory', 'speed_m_s', 'period_s', 'route')):
            raise ValueError('qualified reconstruction recipe changed within stream')
        if recipe.get('rigid_vertical_shift_m', 0) != 0:
            raise ValueError('animated pose has an unsupported rigid placement shift')
        parts, measured = walk.frame(float(recipe['time_s']))
        vertices, _ = triangle_mesh(rig.document, rig.binary, parts)
        bounds = np.stack((vertices.min(0), vertices.max(0)))
        bounds_error = float(np.max(np.abs(bounds - np.asarray(old['expected_pose_bounds_m']))))
        errors = {key: float(np.max(np.abs(np.asarray(measured[key]) - np.asarray(old[key]))))
                  for key in ('mouth_m', 'head_m', 'tail_m', 'world_root_m')}
        # Use the source native bound-readback tolerance; this is a source
        # reconstruction check, never a replacement for production admission.
        if bounds_error > 2e-5 or max(errors.values()) > 2e-5:
            raise ValueError('reconstructed qualified pose mismatch: ' + str((i, bounds_error, errors)))
        target = output / 'frames' / ('frame_%04d.glb' % i)
        rigid_glb(document, rig.binary, parts, target)
        old['original_transient_pose_glb'] = old['pose_glb']
        old['pose_glb'] = str(target)
        checks.append({'frame': i, 'bounds_error_m': bounds_error, 'landmark_errors_m': errors,
                       'rebuilt_pose_sha256': digest(target)})
    receipt = {'schema': 'qualified-animal-transient-pose-reconstruction-v1', 'asset_id': native['asset_id'],
               'source_master': master, 'source_master_sha256': digest(master),
               'source_candidate': candidate, 'source_candidate_sha256': digest(candidate),
               'reconstructed_indices': transient, 'checks': checks, 'original_files_modified': False,
               'new_house_physical_admission_required': True, 'production_gates_changed': False}
    (output / 'RECONSTRUCTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return frames, str(output / 'RECONSTRUCTION_RECEIPT.json')
