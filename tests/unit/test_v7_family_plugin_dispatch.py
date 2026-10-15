from types import SimpleNamespace
import pytest
from avengine.timeline.v7_family_dispatch import construct_requested_scene

pytestmark=pytest.mark.fast_unit


def test_unconfigured_family_preserves_original_constructor_and_inputs(monkeypatch):
    import avengine.timeline.v7_family_constructors as legacy
    calls=[]
    monkeypatch.setattr(legacy,'construct_family_scene',lambda *args,**kwargs:calls.append((args,kwargs))or{'profile':'phase2'})
    assert construct_requested_scene('room',family='QA-14',seed=17)=={'profile':'phase2'}
    assert calls==[(('room',),{'family':'QA-14','seed':17})]


def test_explicit_plugin_gets_original_inputs_without_changing_other_families(monkeypatch):
    import avengine.timeline.v7_family_dispatch as dispatch
    calls=[]
    module=SimpleNamespace(build=lambda *args,**kwargs:calls.append((args,kwargs))or{'profile':'phase2','row':{'native_required':True}})
    monkeypatch.setattr(dispatch,'import_module',lambda name:module if name=='test_plugin'else None)
    result=construct_requested_scene('room','house',family='QA-10',plugin_specs={'QA-10':{'module':'test_plugin','callable':'build'}},seed=23)
    assert result['row']['native_required']
    assert calls==[(('room','house'),{'family':'QA-10','seed':23})]


def test_plugin_failure_is_not_replaced_with_a_legacy_scene(monkeypatch):
    import avengine.timeline.v7_family_dispatch as dispatch
    import avengine.timeline.v7_family_constructors as legacy
    def fail(*args,**kwargs):raise ValueError('requested occluder has no solution')
    monkeypatch.setattr(dispatch,'import_module',lambda name:SimpleNamespace(build=fail))
    monkeypatch.setattr(legacy,'construct_family_scene',lambda *args,**kwargs:pytest.fail('silent fallback'))
    with pytest.raises(ValueError,match='requested occluder'):
        construct_requested_scene('room',family='QA-10',plugin_specs={'QA-10':{'module':'test_plugin','callable':'build'}})
