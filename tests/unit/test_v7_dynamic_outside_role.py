import pytest
from avengine.capture import v7_offscreen_layout as layout

pytestmark = pytest.mark.fast_unit


def _entry(aid):
    return {'declaration': {'asset_id': aid}}


def _compiled(aid):
    return {'declaration': {'asset_id': aid}, 'motion_episode': {'pose_glbs': ['x']}}


def test_dynamic_outside_role_requires_the_later_entry_not_zero_pixels_forever():
    dynamic = {'native_view_roles': {'a': {'view': 'outside', 'static': False, 'required_visible_sometime': True}}}
    assert layout.native_pixels_ok(dynamic, [_compiled('a')], {'a': 200})       # enters later: ok
    assert not layout.native_pixels_ok(dynamic, [_compiled('a')], {'a': 0})     # never enters: not ok
    static = {'native_view_roles': {'a': {'view': 'outside', 'static': True}}}
    assert layout.native_pixels_ok(static, [_entry('a')], {'a': 0})
    assert not layout.native_pixels_ok(static, [_entry('a')], {'a': 200})       # static outside stays strict


def test_refresh_proves_a_dynamic_role_on_frame_zero_pixels(monkeypatch):
    seen = {}

    class Video:
        actors = [_compiled('a')]                                             # the final refresh runs after compile

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


def test_uncompiled_dynamic_outside_role_is_checked_at_its_frame_zero_pose():
    # qualify_standing runs before the walk is compiled: the standing frame-zero pose must be outside (0 px)
    dynamic = {'native_view_roles': {'a': {'view': 'outside', 'static': False, 'required_visible_sometime': True}}}
    assert layout.native_pixels_ok(dynamic, [_entry('a')], {'a': 0})
    assert not layout.native_pixels_ok(dynamic, [_entry('a')], {'a': 150})      # starts in view: not an entry


def test_seam_is_closed_before_the_erosion():
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union
    seam = unary_union([box(0, 0, 1, 1), box(1 + 3e-9, 0, 2, 1)])
    line = LineString([(0.5, 0.5), (1.5, 0.5)])
    assert not seam.buffer(-.02).buffer(1e-6).covers(line)                     # erode first: the seam is 4 cm wide
    assert seam.buffer(1e-6).buffer(-.02).covers(line)                         # close first: covered
    assert not seam.buffer(1e-6).buffer(-.02).covers(LineString([(0.5, 0.5), (1.995, 0.5)]))  # 2 cm margin kept
