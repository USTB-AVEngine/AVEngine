"""Explicit production plugin dispatch; missing configuration keeps legacy routing.

The configured plugin comes from the parent's committed source snapshot. A
plugin construction error is reported directly, never replaced with an
unrequested legacy construction or with changed QA answers.
"""
from importlib import import_module


def construct_requested_scene(*args,family,plugin_specs=None,**kwargs):
    specification=(plugin_specs or {}).get(family)
    if specification is None:
        from avengine.timeline.v7_family_constructors import construct_family_scene
        constructor=construct_family_scene
    else:
        constructor=getattr(import_module(specification['module']),specification['callable'])
    return constructor(*args,family=family,**kwargs)
