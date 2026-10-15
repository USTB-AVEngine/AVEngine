"""Resolve coarse and named construction conditions to their requested families.

Detailed conditions are already published in family_scene_requirements. Keep
descriptive construction notes, and retain explicit family targets as targets.
"""
def requested_families(demand):
    mapping={name:set(families) for name,families in demand.get('condition_families',{}).items()}
    for family,requirements in demand.get('family_scene_requirements',{}).items():
        for name in requirements.get('conditions',[]):
            mapping.setdefault(name,set()).add(family)
    result=set(demand.get('required_independent_family_counts',{}))
    result.update(demand.get('batch_minimum_independent_family_counts',{}))
    for name in demand.get('requested_conditions',[]):
        result.update(mapping.get(name,()))
    return result
