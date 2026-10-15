"""Construct animal support on measured terrain; retain imperfect paws as evidence.

Owner 2026-10-15: cosmetic paw contact must not invalidate a sound question.
The original skin, scene collision, navigation and native emitter checks remain.
"""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
from avengine.assets.roster_motion.contact import yaw_matrix

POLICY = {'revision': 'owner-animal-appearance-construction-20261015',
          'appearance_only': True, 'unresolved_paw_quality': 'record_only',
          'truth_and_answer_gates_changed': False}


def framewise_route_height(poses, meshes, roots, source, indices, yaws, floor, species, local_paw):
    """Translate each whole pose vertically from its measured skin/paw support."""
    placed = np.asarray(roots, float).copy()
    shifts = []; warnings = []
    for i, (path, root, index, yaw) in enumerate(zip(poses, roots, indices, yaws)):
        rotation = yaw_matrix(float(yaw))
        vertices = np.asarray(meshes[path].vertices)
        low = vertices[vertices[:, 1] <= vertices[:, 1].min() + .08]
        world = low @ rotation.T + root
        skin_delta = float(np.min(world[:, 1] - floor.many(world)))
        lower = -.001 - skin_delta
        preferred = .0004 - skin_delta
        frame = source[int(index)]
        feet = list(frame['feet'].values())
        soles = np.asarray([rotation @ local_paw(frame, f) + root for f in feet])
        deltas = soles[:, 1] - floor.many(soles)
        upper = math.inf
        for f, delta in zip(feet, deltas):
            lower = max(lower, (-.001 if species == 'cat' else -.008) - float(delta))
            if f['in_contact']:
                upper = min(upper, .012 - float(delta))
        shift = max(lower, min(preferred, upper))
        placed[i, 1] += shift; shifts.append(float(shift))
        if lower > upper:
            warnings.append({'frame_index': i, 'reason': 'paw contact cannot share skin-safe root height',
                             'minimum_shift_m': float(lower), 'maximum_shift_m': float(upper)})
    return placed, {'method': 'per-frame measured whole-body ground following',
                    'per_frame_correction': True, 'joint_poses_changed': False,
                    'vertical_translations_m': shifts, 'appearance_warnings': warnings,
                    'skin_floor_gate_unchanged': True, **POLICY}


class StationaryPawGrounder:
    """Reuse the registered target rig and two-bone IK for stopped walk poses.

    Only derived pose buffers are written. Failed/missing IK is reported, with
    the actual unmodified pose and measured contacts retained by the caller.
    """
    def __init__(self, native, species, directory):
        self.native = native; self.species = species; self.directory = Path(directory)
        self.walk = None; self.cache = {}

    def _generator(self, frame):
        if self.walk is not None:
            return self.walk
        from avengine.assets.animal_motion.profiles import Rig
        from avengine.assets.animal_motion.walk import ConstrainedWalk
        recipe = frame.get('pose_reconstruction')
        if not recipe or self.native.get('motion_mode') != 'walk_animation':
            raise ValueError('registered rigid pose has no reconstructible articulated IK recipe')
        record = {'asset_id': self.native['asset_id'], 'species': self.species,
                  'files': {'motion_master_glb': {'path': recipe['source_master']}},
                  'walking_candidate': recipe['source_inventory']}
        profile = self.native.get('source_provenance', {}).get('explicit_motion_profile')
        if profile: record['motion_profile'] = profile
        rig = Rig.load(record).enable_skin_cache()
        model = self.native.get('gait_generation_floor_model', {})
        if model.get('type') != 'fixed_plane':
            raise ValueError('stationary IK requires the registered fixed-plane gait recipe')
        self.walk = ConstrainedWalk(rig, speed=float(recipe['speed_m_s']),
                                    period=float(recipe['period_s']),
                                    floor=lambda p: float(model['local_height_m']))
        return self.walk

    def ground(self, frame, path, root, yaw, floor):
        key = (str(path), tuple(np.round(root, 7)), round(float(yaw), 7))
        if key in self.cache: return self.cache[key]
        from avengine.assets.animal_motion.walk import solve_leg
        from avengine.assets.animal_motion.profiles import paw_points
        from avengine.assets.roster_motion.episode import externalize_unchanged_images
        from avengine.assets.roster_motion.pose import rigid_glb
        walk = self._generator(frame); rig = walk.rig
        walk.frame(float(frame['pose_reconstruction']['time_s']))
        globals_ = deepcopy(walk.last_globals)
        rotation = yaw_matrix(float(yaw)); maximum_reach = 0.
        for _ in range(8):
            parts = rig.skin(globals_); paws = paw_points(parts, rig.masks)
            for side, chain in rig.chains.items():
                local = np.asarray(paws[side]['point']); world = rotation @ local + root
                dy = float(floor(world)) + .0004 - world[1]
                target = globals_[chain[2]][:3, 3].copy(); target[1] += dy
                linear = globals_[chain[2]][:3, :3].copy()
                maximum_reach = max(maximum_reach, solve_leg(rig.document, globals_, chain, target, linear))
        parts = rig.skin(globals_); paws = paw_points(parts, rig.masks); marks = rig.landmarks(globals_)
        measured = deepcopy(frame); measured.update(root_world_m=[0., 0., 0.],
            mouth_m=marks['mouth'].tolist(), head_m=marks['head'].tolist(), tail_m=marks['tail'].tolist(),
            head_world_m=marks['head'].tolist(), tail_world_m=marks['tail'].tolist())
        measured['feet'] = {}
        for side, paw in paws.items():
            local = np.asarray(paw['point']); world = rotation @ local + root
            delta = float(world[1] - floor(world))
            measured['feet'][side] = {'sole_world_m': local.tolist(), 'support_world_m': local.tolist(),
                'support_target_world_m': local.tolist(), 'floor_delta_m': delta, 'in_contact': abs(delta) <= .012}
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / ('stationary_%03d.glb' % len(self.cache))
        document = externalize_unchanged_images(rig.document, rig.binary, self.directory / 'textures')
        rigid_glb(document, rig.binary, parts, target)
        receipt = {'method': 'registered target two-bone IK on actual measured ground',
                   'status': 'corrected', 'maximum_reach_error_m': float(maximum_reach),
                   'source_pose': str(path), 'derived_pose': str(target),
                   'actual_paw_deltas_m': {s: f['floor_delta_m'] for s, f in measured['feet'].items()},
                   'original_asset_modified': False}
        self.cache[key] = measured, str(target), receipt
        return self.cache[key]


def construct_animal_floor_support(sim, runtime, meta, support):
    """Retry an uneven floor by grounding the lowest real support point.

    Every ray must still hit real stage geometry, and later native semantic,
    skin, clearance, emitter and visibility checks still run. Residual heights
    of individual paws are cosmetic, and are never fabricated as contacts.
    """
    from avengine.capture.v7_visible_attachment import ray, rotation, support_points
    root, yaw, normal, local = support_points(meta, support)
    deltas = []; hits = []
    for q in local:
        point = root + rotation(yaw) @ q
        hit = ray(sim, runtime, point + normal * .10, -normal, .22)
        if hit is None: return None
        deltas.append(float((point - hit) @ normal)); hits.append(hit.tolist())
    if not deltas: return None
    shift = .0004 - min(deltas)
    revised = deepcopy(support)
    revised['point'] = (np.asarray(support['point']) + normal * shift).tolist()
    return {'root_m': (root + normal * shift).tolist(), 'yaw_rad': yaw, 'normal': normal.tolist(),
            'contact_deltas_m': [d + shift for d in deltas], 'support_raycast_points_m': hits,
            'support': revised, 'threshold_m': .015, 'actual_mesh_attachment': meta['_attachment_template'][2],
            'support_identification': 'all original native support rays; lowest point rigidly grounded',
            'animal_visual_fallback_policy': dict(POLICY), 'animal_floor_support': True,
            'appearance_warnings': [{'reason': 'uneven per-paw support recorded after root grounding',
                                     'original_contact_deltas_m': deltas, 'root_translation_m': shift}]}


def attachment_has_hard_error(attachment):
    """Keep real skin collisions/types strict, classify only animal paw gaps."""
    visual_only = (attachment.get('animal_floor_support') is True and
                   attachment.get('animal_visual_fallback_policy', {}).get('revision') == POLICY['revision'])
    return (attachment['attachment_surface'] != attachment['support_type'] or not attachment['skin_pass'] or
            (attachment['attachment_worst_error_m'] > .015 and not visual_only))
