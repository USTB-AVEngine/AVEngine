"""Owner-authorized single independent native reference for new parents.

2026-10-16 owner: 不用做三次对比了吧，我觉得一次就行了。
No spot triple audits. Thread count/randomness changes require revalidation.
"""
import os

ENV_KEY='AVENGINE_V7_RIR_SINGLE_REFERENCE'
OWNER_STATEMENT='不用做三次对比了吧，我觉得一次就行了'
SINGLE_REVISION='owner-single-native-reference-20261016-v1'
LEGACY_REVISION='legacy-three-native-references-v1'


def reference_count():
    value=os.environ.get(ENV_KEY,'1')
    if value not in {'0','1'}:raise ValueError(ENV_KEY+' must be 0 or 1')
    return 1 if value=='1' else 3


def policy_revision(count):
    return SINGLE_REVISION if count==1 else LEGACY_REVISION


def checkpoint_references(items,count):
    if len(items)not in ({1,3}if count==1 else{3}):
        raise ValueError('independent checkpoint reference count incompatible with selected policy')
    return items[:count]
