"""Stable acceptance task identity, independent of scheduling/list order."""


def task_by_slot(manifest,slot):
    matches=[task for task in manifest['parents']if task.get('slot')==slot]
    if len(matches)!=1:
        raise ValueError('acceptance slot must identify exactly one task: '+str(slot))
    return matches[0]
