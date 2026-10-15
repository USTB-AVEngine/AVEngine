"""Owner-approved human motion quality; answer/readback checks stay separate."""

HUMAN_QUALITY = {
    'revision': 'owner-human-quality-20261015-relaxed',
    # owner 10-15: "什么离地不能超过8毫米，因为他动作原因我觉得不用太过严格吧" -- a planted foot within a few cm of
    # the floor, sliding <= 2 cm or sinking <= 3 cm, is invisible in a 640x360 render and changes no answer.
    'foot_contact_limit_m': .03,
    'support_lock_limit_m': .02,
    'foot_penetration_limit_m': .03,
    'minimum_nonfloor_clearance_m': 0.,
    'actual_intersections_allowed': False,
}
