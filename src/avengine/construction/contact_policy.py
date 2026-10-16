"""Owner contact construction policy; measurements never stand in for QA truth."""
from copy import deepcopy
import math
import numpy as np

POLICY = {
    'policy': 'construct_then_record',
    'date': '2026-10-16',
    'owner_quote': '人的身体和地面结合那个问题，不要再卡那么死了，这对我们题目的没有帮助，现在还有什么题目是没达标的？原因是什么，该怎么做？不要私自再加限制哈',
}
# Existing native placement step/floor-level allowance, not a contact threshold.
FLOOR_STEP_RANGE_M = .08


def is_body(asset):
    return (asset.get('kind') in {'human', 'performer', 'animal'} or
            asset.get('entity_class') in {'articulated_human', 'articulated_animal'})


def contact_quality_observation(frames=(), *, deltas=(), semantics=(), **details):
    """Retain measured residuals, including unsupported stance labels."""
    frames = list(frames)
    feet = [f for frame in frames for f in frame.get('feet', {}).values()]
    all_deltas = [float(v) for v in deltas] + [float(f['floor_delta_m']) for f in feet]
    stance = [float(v) for v in deltas] + [float(f['floor_delta_m']) for f in feet if f.get('in_contact')]
    def maximum(values):
        values = list(values)
        return max(values) if values else None
    return {
        **POLICY,
        'max_gap_m': maximum(max(0., v) for v in stance),
        'max_penetration_m': maximum(max(0., -v) for v in all_deltas),
        'max_sliding_m': maximum(float(f['support_lock_error_m']) for f in feet if 'support_lock_error_m' in f),
        'max_ik_reach_deficit_m': maximum(float(f['ik_reach_deficit_so_far_m']) for f in frames if 'ik_reach_deficit_so_far_m' in f),
        'max_pelvis_lowering_m': maximum(float(f['pelvis_lowering_m']) for f in frames if 'pelvis_lowering_m' in f),
        'foot_semantic_labels': list(semantics),
        'no_planted_support_frames': sum(bool(f.get('feet')) and not any(p.get('in_contact') for p in f['feet'].values()) for f in frames),
        **details,
    }


def native_contact_row(contact, probe_id):
    """Preserve real stage contact locations to distinguish floor from furniture."""
    on_a = int(contact.object_id_a) == int(probe_id)
    point = contact.position_on_b_in_ws if on_a else contact.position_on_a_in_ws
    return {'object_id_a': int(contact.object_id_a), 'object_id_b': int(contact.object_id_b),
            'contact_distance_m': float(contact.contact_distance),
            'stage_contact_point_m': np.asarray(point, float).tolist(),
            'normal_m': np.asarray(contact.contact_normal_on_b_in_ws, float).tolist()}


def nonfloor_contacts_clear(contacts, floor_y, tolerance_m):
    """Apply the existing penetration tolerance only to non-floor contacts."""
    for contact in contacts:
        point, normal = contact.get('stage_contact_point_m'), contact.get('normal_m')
        # Missing classification evidence retains the original strict decision.
        floor_contact = (point is not None and normal is not None
            and abs(float(point[1])-float(floor_y)) <= FLOOR_STEP_RANGE_M
            and abs(float(normal[1])) > max(abs(float(normal[0])), abs(float(normal[2]))))
        if not floor_contact and contact['contact_distance_m'] < -tolerance_m:
            return False
    return True


def construct_body_floor_support(sim, runtime, meta, support, *, floor_y=None):
    """Ground a body on a real root support; partial foot samples are observations.

    A root over a void or a furniture top is still an illegal placement. Missing
    peripheral rays are not replaced by fabricated hits.
    """
    from avengine.capture.v7_visible_attachment import ray, rotation, support_points
    root, yaw, normal, local = support_points(meta, support)
    reference = float(support.get('floor_y', support['point'][1]) if floor_y is None else floor_y)
    root_probe = np.array([root[0], reference, root[2]])
    root_hit = ray(sim, runtime, root_probe + [0., .2, 0.], np.array([0., -1., 0.]), .85)
    if root_hit is None or abs(float(root_hit[1])-reference) > FLOOR_STEP_RANGE_M:
        return None
    points = [root + rotation(yaw) @ q for q in local]
    hits, deltas, missing = [], [], []
    for index, point in enumerate(points):
        origin = np.array([point[0], reference + .2, point[2]])
        hit = ray(sim, runtime, origin, np.array([0., -1., 0.]), .85)
        if hit is None:
            missing.append(index)
            continue
        if abs(float(hit[1])-reference) > FLOOR_STEP_RANGE_M:
            return None
        deltas.append(float(point[1]-hit[1]))
        hits.append(np.asarray(hit, float).tolist())
    if not hits:
        # Partial sampling is cosmetic; no ground beneath any actual support
        # point is still an illegal standing position.
        return None
    # Fit the original lowest support surface before retaining per-foot residuals.
    shift = float(root_hit[1]-min(p[1] for p in points))
    grounded = root + [0., shift, 0.]
    revised = deepcopy(support)
    revised['point'] = (np.asarray(support['point']) + [0., shift, 0.]).tolist()
    residuals = [d + shift for d in deltas]
    observation = contact_quality_observation(deltas=residuals, root_support_verified=True,
        root_support_point_m=np.asarray(root_hit, float).tolist(),
        root_translation_y_m=shift, missing_attachment_probe_indices=missing,
        measured_attachment_probe_count=len(hits), requested_attachment_probe_count=len(points))
    return {'root_m': grounded.tolist(), 'yaw_rad': yaw, 'normal': normal.tolist(),
        'contact_deltas_m': residuals, 'support_raycast_points_m': hits,
        'support': revised, 'threshold_m': .015,
        'actual_mesh_attachment': meta['_attachment_template'][2],
        'support_identification': 'actual stage root support; measured peripheral rays retained separately',
        'contact_quality_observation': observation}


class MotionRouteUnavailable(ValueError):
    """Only a failed route search, never an arbitrary runtime/asset exception."""
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.evidence = dict(evidence or {})


def element_degradation(scene, entry, original, replacement, reason, families=()):
    row = {'asset_id': entry['declaration']['asset_id'], 'endpoint_id': entry['endpoint_id'],
           'original_plan': deepcopy(original), 'replacement': replacement, 'reason': str(reason),
           'dropped_families': sorted(set(families)),
           'family_effect':'only this element plan; unchanged native QA still decides all remaining candidates', **POLICY}
    scene.setdefault('element_degradation', []).append(row)
    return row


def degrade_motion_to_standing(scene, entry, request, reason):
    """Keep the already qualified static body and its existing non-footstep sound."""
    ep, aid = entry['endpoint_id'], entry['declaration']['asset_id']
    # Footstep sources cannot play from a stationary shoe. Their source program
    # needs a producer-owned rebuild, never an invented native readback.
    if any(p.get('endpoint_id') == ep for p in scene.get('planned_footsteps', [])):
        raise MotionRouteUnavailable('stationary fallback needs footstep program reconstruction',
                                     {'asset_id': aid, 'route_failure': str(reason)})
    families = set()
    demand=scene.get('scene_demand',{})
    family=demand.get('target_family') or scene.get('automatic_construction',{}).get('family')
    original_movers=scene.get('production_scenario',{}).get('motion_requests',{})
    target=demand.get('conditioned_post_motion',{}).get('target_asset_id')
    if family and (target==aid or (len(original_movers)==1 and ep in original_movers)):
        families.add(family)
    for container in (scene, scene.get('production_scenario', {})):
        for item in container.get('motion_during_condition_requests', []):
            if item.get('target_asset_id') == aid:
                families.add(item['qa_id'])
        container.get('motion_requests', {}).pop(ep, None)
        container.get('motion_requests', {}).pop(aid, None)
    if request.get('joint_entry_route'):
        families.add('QA-07')
    if request.get('joint_qa09_route'):
        families.add('QA-09')
    if request.get('distance_trend'):
        families.add('QA-15')
    scene['pending_registered_walk_tts']=[p for p in scene.get('pending_registered_walk_tts',[]) if p!=ep]
    entry.pop('motion_episode', None)
    entry.pop('phase2_required_foot_emitters', None)
    return element_degradation(scene, entry, request, 'stand', reason, families)


def optional_bystander(scene, entry):
    """Only explicitly silent, unrequested entities can be removed in-place."""
    a = entry['declaration']; aid, ep = a['asset_id'], entry['endpoint_id']
    if not a.get('silent_decor'):
        return False
    if ep in scene.get('production_scenario', {}).get('motion_requests', {}):
        return False
    from avengine.capture.v7_offscreen_layout import visibility_required
    if visibility_required(scene,a):
        return False
    required = scene.get('scene_demand', {}).get('required_asset_ids', [])
    if aid in required:
        return False
    for key in ('native_occlusion_relations', 'native_dynamic_occlusion_relations'):
        if any(aid in (r.get('target_asset_id'), r.get('occluder_asset_id')) for r in scene.get(key, [])):
            return False
    automatic = scene.get('automatic_construction', {})
    if aid == automatic.get('qa21_target_asset_id') or aid in automatic.get('qa21_target_asset_ids', []):
        return False
    if any(e.get('endpoint_id') == ep for e in scene.get('events', [])):
        return False
    return True



class OptionalPlacementUnavailable(ValueError):
    """Named failure consumed by the existing recipe-level element fallback."""
    def __init__(self, entry, degradation):
        a=entry['declaration']
        super().__init__('no strict native '+a.get('attachment_surface','floor')+
            ' placement for '+a['asset_id']+' [optional element reconstruction]')
        self.element_degradation=degradation


def optional_placement_failure(scene, entry, reason):
    """Request a coherent recipe rebuild; never leave ghost assets or audio."""
    if not optional_bystander(scene,entry):
        return None
    row=element_degradation(scene,entry,{'action':'place_silent_bystander'},
        'remove_on_recipe_reconstruction',reason)
    return OptionalPlacementUnavailable(entry,row)
