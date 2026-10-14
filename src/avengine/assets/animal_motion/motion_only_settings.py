"""Versioned owner motion-only settings, importable by isolated Blender Python."""
FLOOR_PENETRATION_M = .00101  # unchanged original asset-admission floor tolerance

POLICY_NAME = 'owner_relaxed_20261014'
VALIDATION_POLICY_NAMES = (POLICY_NAME,)
AUDIO_STATUS = 'deferred_to_production'


def validation_policy(name=POLICY_NAME):
    if name != POLICY_NAME:
        raise ValueError('unknown_motion_only_policy:' + str(name))
    return {
        'name': POLICY_NAME, 'schema_version': 1,
        'authority': 'owner instruction 2026-10-14 09:3x SGT',
        'direction_deg': 20., 'contact_m': .012,
        'floor_penetration_m': FLOOR_PENETRATION_M,
        'floor_rule': 'unchanged asset-admission floor tolerance',
        'plant_drift_m': .015, 'plant_speed_m_s': None,
        'cross_region_area_m2': 400e-6, 'penetration_depth_m': .010,
        'check_intraregion': False, 'maximum_edge_growth': 2.,
        'uv_seam_is_admission_criterion': False,
        'per_asset_audio': AUDIO_STATUS,
    }
