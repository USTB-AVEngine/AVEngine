import pytest
from avengine.capture import v7_offscreen_layout as layout

pytestmark = pytest.mark.fast_unit


def _entry(aid):
    return {'declaration': {'asset_id': aid}}


def test_dynamic_outside_role_requires_the_later_entry_not_zero_pixels_forever():
    dynamic = {'native_view_roles': {'a': {'view': 'outside', 'static': False, 'required_visible_sometime': True}}}
    assert layout.native_pixels_ok(dynamic, [_entry('a')], {'a': 200})          # enters later: ok
    assert not layout.native_pixels_ok(dynamic, [_entry('a')], {'a': 0})        # never enters: not ok
    static = {'native_view_roles': {'a': {'view': 'outside', 'static': True}}}
    assert layout.native_pixels_ok(static, [_entry('a')], {'a': 0})
    assert not layout.native_pixels_ok(static, [_entry('a')], {'a': 200})       # static outside stays strict


def test_refresh_proves_a_dynamic_role_on_frame_zero_pixels(monkeypatch):
    seen = {}

    class Video:
        actors = [_entry('a')]

        def _apply_entry(self, entry, scene, frame_index=0):
            return {}

        def _native_observed_pixels(self, scene, entries):
            return {'a': 200}, [{'time_s': 0., 'pixels': {'a': 0}}, {'time_s': 5., 'pixels': {'a': 200}}]

    def proof(video, scene, entry, visible_pixels):
        seen['pixels'] = visible_pixels
        return {'status': 'pass'}

    monkeypatch.setattr(layout, 'native_projection_proof', proof)
    scene = {'native_view_roles': {'a': {'view': 'outside', 'static': False, 'required_visible_sometime': True}}}
    checks = [{'asset_id': 'a'}]
    layout.refresh_actual_offscreen_proofs(Video(), scene, checks)
    assert seen['pixels'] == 0 and checks[0]['intentional_offscreen_proof'] == {'status': 'pass'}


def test_floor_cover_tolerates_a_floating_point_seam():
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union
    seam = unary_union([box(0, 0, 1, 1), box(1 + 3e-9, 0, 2, 1)])        # two parts with a 3 nm seam
    line = LineString([(0.5, 0.5), (1.5, 0.5)])
    assert not seam.covers(line) and seam.buffer(1e-6).covers(line)
    assert not seam.buffer(1e-6).covers(LineString([(0.5, 0.5), (2.5, 0.5)]))   # a real exit still fails


def test_initial_uncompiled_outside_drawable_can_reach_motion_preparation():
    scene={'native_view_roles':{'a':{'view':'outside','static':False,'required_visible_sometime':True}}}
    assert layout.native_pixels_ok(scene,[_entry('a')],{'a':0},new_entry_id='a')
    assert not layout.native_pixels_ok(scene,[_entry('a')],{'a':200},new_entry_id='a')
    assert not layout.native_pixels_ok(scene,[_entry('a')],{'a':0})
    compiled={**_entry('a'),'motion_episode':{'real_native_episode':True}}
    assert not layout.native_pixels_ok(scene,[compiled],{'a':0},new_entry_id='a')
    assert layout.native_pixels_ok(scene,[compiled],{'a':200},new_entry_id='a')
