"""Bind a capped room seed to its own admitted native house/layout, never another room."""
from copy import deepcopy
from avengine.timeline.v7_alpha_sampling import alpha_row


def bind_formal_seed(row,house,layout):
    if row['split']!='train' or row['render_path']!='ue_spear' or house['family']!='kujiale' or house['split']!='train' or house['input']['visual_backend']!='ue_spear':raise ValueError('native training-only input required')
    if row['house']!=house['house'] or layout['house']!=house['house'] or layout['room_id']!=row['room_id'] or row['room_id'] not in house['checked_room_ids']:raise ValueError('seed, own native house and measured room layout do not match')
    if layout['status']!='pass' or len(layout.get('valid',[]))!=4:raise ValueError('current four-body formal alpha layout is not qualified')
    if row.get('room_parent_slot') not in range(16):raise ValueError('room parent cap exceeded')
    if row['members'] not in [['v0_a0'],['v0_a0','v0_a1','v1_a0','v1_a1']]:raise ValueError('incomplete counterfactual group')
    bound=alpha_row(row,layout,row['question_quota_base']//8,force_quartet=False)
    bound.update(generation_admission='CPU_layout_bound_pending_actual_native_formal_checks',physical_room_admission=True)
    return bound
