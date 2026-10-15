"""Owner-approved human motion quality; answer/readback checks stay separate."""

HUMAN_QUALITY = {
    'revision': 'owner-human-quality-20261015-relaxed',
    # owner 10-15: "什么离地不能超过8毫米，因为他动作原因我觉得不用太过严格吧" -- a planted foot within a few cm of
    # the floor, sliding <= 2 cm or sinking <= 3 cm, is invisible in a 640x360 render and changes no answer.
    'foot_contact_limit_m': .03,
    'support_lock_limit_m': .02,
    'foot_penetration_limit_m': .03,
    # Same-level support for a walking person: scanned floors, rugs and thresholds vary by a few cm; furniture
    # tops (>= 35 cm) and steps (>= 15 cm) stay far outside this band (gate audit G027: was 15 mm).
    'floor_level_tolerance_m': .05,
    'minimum_nonfloor_clearance_m': 0.,
    'actual_intersections_allowed': False,
}
