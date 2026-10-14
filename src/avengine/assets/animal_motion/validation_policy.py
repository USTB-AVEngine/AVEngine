"Explicit owner-selected animal motion acceptance policies; no runtime imports."
DIRECTION_DEG=15.
CONTACT_M=.012
PLANT_DRIFT_M=.003
PLANT_SPEED_M_S=.03

VALIDATION_POLICY_NAMES=('strict','owner_relaxed_20261014')
def validation_policy(name='strict'):
    if name not in VALIDATION_POLICY_NAMES:raise ValueError('unknown_motion_validation_policy:'+str(name))
    relaxed=name=='owner_relaxed_20261014'
    return {'name':name,'direction_deg':20. if relaxed else DIRECTION_DEG,
            'contact_m':CONTACT_M,'plant_drift_m':.015 if relaxed else PLANT_DRIFT_M,
            'plant_speed_m_s':None if relaxed else PLANT_SPEED_M_S,
            'cross_region_area_m2':400e-6 if relaxed else 25e-6,
            'penetration_depth_m':.010 if relaxed else .002,
            'check_intraregion':not relaxed,'maximum_edge_growth':2. if relaxed else 1.5,
            'per_asset_audio':'deferred_to_production' if relaxed else 'required',
            'authority':'owner instruction 2026-10-14 09:3x SGT' if relaxed else 'original animal-motion task'}
