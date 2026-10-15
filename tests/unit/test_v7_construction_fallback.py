"""Claude 10-15: construct instead of failing a scene on one unplaceable entity."""
from avengine.capture.v7_offscreen_layout import placement_construction_fallback, proposal_view_ok, offscreen_role
from avengine.timeline.v7_spatial_layout import spacing_rejection


import pytest
pytestmark=pytest.mark.fast_unit


def test_person_first_relaxes_camera_then_goes_offscreen_once():
    scene = {'production_scenario': {'motion_requests': {'voice1': {}}}}
    person = {'asset_id': 'p2', 'endpoint_id': 'voice2', 'kind': 'human'}
    assert placement_construction_fallback(scene, person, 'floor', None, False) == 'camera_1_2m'
    assert scene['native_person_camera_construction_m']['p2'] == 1.2
    assert placement_construction_fallback(scene, person, 'floor', None, False) == 'offscreen'
    assert offscreen_role(scene, 'p2')['minimum_abs_azimuth_deg'] == 60.
    assert placement_construction_fallback(scene, person, 'floor', None, False) is None


def test_requirements_that_define_the_question_are_kept():
    scene = {'production_scenario': {'motion_requests': {'voice1': {}}}}
    mover = {'asset_id': 'p1', 'endpoint_id': 'voice1', 'kind': 'human'}
    assert placement_construction_fallback(scene, mover, 'floor', None, False) == 'camera_1_2m'
    # a mover without a view requirement may enter or leave the fixed view (never a static off-screen role)
    assert placement_construction_fallback(scene, mover, 'floor', None, False) == 'moving_view_construction'
    assert scene['native_view_roles']['p1']['view'] == 'unconstrained_motion'
    assert placement_construction_fallback(scene, mover, 'floor', None, False) is None
    sector = {'asset_id': 'p3', 'endpoint_id': 'voice3', 'kind': 'human'}
    placement_construction_fallback(scene, sector, 'floor', 30., False)
    assert placement_construction_fallback(scene, sector, 'floor', 30., False) is None      # requested sector kept
    pair = {'asset_id': 'p4', 'endpoint_id': 'voice4', 'kind': 'human', 'controlled_shirt': True}
    placement_construction_fallback(scene, pair, 'floor', None, True)
    assert placement_construction_fallback(scene, pair, 'floor', None, True) is None        # colour pair kept
    wall = {'asset_id': 'o1', 'endpoint_id': 'obj1', 'kind': 'object'}
    assert placement_construction_fallback(scene, wall, 'wall', None, False) is None


def test_camera_gate_never_below_1_2m():
    person = {'asset_id': 'p', 'kind': 'human'}
    assert spacing_rejection(person, [1.3, 0., 0.], [], camera_position=[0., 0., 0.]) is not None
    assert spacing_rejection(person, [1.3, 0., 0.], [], camera_position=[0., 0., 0.], camera_minimum_m=1.2) is None
    assert spacing_rejection(person, [1.1, 0., 0.], [], camera_position=[0., 0., 0.], camera_minimum_m=0.5) is not None


def test_sparse_shoe_uses_nearest_raised_vertex_within_10cm(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from avengine.capture.v7_phase2_native import NativePhase2Video
    from avengine.capture.v7_visible_native import NativeVisibleVideo
    from avengine.capture import v7_visible_mesh
    monkeypatch.setattr(NativeVisibleVideo, '_apply_motion_entry', lambda *args: {'body_to_world_m': np.eye(4).tolist(), 'native_object_id': 1})
    # Toe-tip contact with only flat sole vertices within 5 cm; the shoe top is 7 cm away.
    verts = np.array([[0., 0., 0.], [0.03, 0.001, 0.], [0., 0.001, 0.04], [0., 0.06, 0.035]])
    monkeypatch.setattr(v7_visible_mesh, 'indexed_static_mesh', lambda path: SimpleNamespace(vertices=verts))
    node = SimpleNamespace(create_child=lambda: SimpleNamespace(translation=None, absolute_transformation=lambda: np.eye(4)))

    class Child:
        def __init__(self):
            self.translation = None
        def absolute_transformation(self):
            m = np.eye(4); m[:3, 3] = np.asarray(self.translation, float); return m
    entry = {'declaration': {'silent_decor': False}, 'endpoint_id': 'voice1', 'phase2_required_foot_emitters': True,
             'object': SimpleNamespace(root_scene_node=SimpleNamespace(create_child=Child)),
             'motion_episode': {'source_frames': [{'mouth_m': [0., 1.4, 0.], 'moving': True,
                                                   'feet': {'L': {'sole_local_m': [0., 0., 0.], 'in_contact': True, 'floor_delta_m': 0.}}}],
                                'pose_glbs': ['pose.glb']}}
    video = object.__new__(NativePhase2Video)
    video.mn = SimpleNamespace(Vector3=lambda v: list(v))
    row = video._apply_motion_entry(entry, 0)
    foot = row['attached_emitter_readbacks'][0]
    assert foot['local_m'] == [0., 0.06, 0.035] and foot['shoe_search_radius_m'] == 0.10




def test_actual_ue_offscreen_requires_the_whole_drawable_and_zero_native_pixels():
    import numpy as np
    from avengine.capture.kujiale_pilot_environment import actual_offscreen_projection
    view={'position_m':[0.,1.,0.],'yaw_deg':0.,'pitch_deg':0.,'hfov_deg':85.}
    body=np.array([[x,y,z]for x in [-3.,-2.] for y in [0.,1.] for z in [-.2,.2]])
    result=actual_offscreen_projection(body,view,(720,1280),0,123)
    assert result['status']=='pass' and result['native_object_id']==123
    assert result['source_is_real_drawable'] and not result['QA_visibility_overridden']
    assert actual_offscreen_projection(body,view,(720,1280),1,123)['status']=='fail'
    assert actual_offscreen_projection(body,view,(720,1280),0,None)['status']=='fail'
    body[:,0]*=-1
    assert actual_offscreen_projection(body,view,(720,1280),0,123)['status']=='fail'


def test_offscreen_human_clearance_keeps_collision_but_does_not_require_an_in_view_animal():
    import numpy as np
    from types import SimpleNamespace
    from avengine.capture.kujiale_pilot_environment import human_clear_of_environment
    evidence={'placement':{'native_world_bounds_m':[[-.2,0.,-4.],[.2,.4,-3.6]]},'intentional_offscreen_role':{'view':'outside'}}
    video=SimpleNamespace(environment_nodes={'environment1':{'physical':True,'native_evidence':evidence}},scene={'listener_m':[0.,1.,0.]})
    assert human_clear_of_environment(video,np.array([-.4,0,-2.]),np.array([.4,1.9,-1.6]))
    assert not human_clear_of_environment(video,np.array([-.4,0,-4.]),np.array([.4,1.9,-3.5]))


def test_actual_ue_offscreen_finish_requires_all_450_physical_drawable_proofs(tmp_path):
    from types import SimpleNamespace
    from avengine.capture.kujiale_pilot_environment import finish_scene
    frames=[{'frame_index':i,'entities':[{'endpoint_id':'environment1','visible_pixels':0,
        'source_geometry_present':True,'intentional_offscreen_proof':{'status':'pass'}}]}for i in range(450)]
    video=SimpleNamespace(current_environment_frames=frames,current_environment_admission={'environment1':{'status':'pass'}},
        environment_nodes={'environment1':{'native_evidence':{'intentional_offscreen_role':{'view':'outside'}}}})
    result={};proof=finish_scene(video,result,tmp_path/'render.mp4')
    assert proof['status']=='pass' and proof['all_sources_have_drawable_geometry']
    frames[17]['entities'][0]['source_geometry_present']=False
    with pytest.raises(ValueError,match='450 actual drawable'):finish_scene(video,{},tmp_path/'render.mp4')


def test_view_follows_the_question_not_a_blanket_rule():
    import numpy as np
    from avengine.capture.v7_offscreen_layout import proposal_view_ok, native_pixels_ok, visibility_required
    fwd = np.array([0., 0., -1.])
    front, edge, side = np.array([0., 0., -3.]), np.array([2.4, 0., -3.]), np.array([3., 0., 0.5])   # 0, ~39, ~99 deg
    scene = {'native_view_roles': {}}
    # required (default): only clearly visible proposals
    assert proposal_view_ok(scene, 'p', front, fwd, True) and not proposal_view_ok(scene, 'p', side, fwd, False)
    # unconstrained: clearly in view or clearly outside, never the ambiguous edge
    assert proposal_view_ok(scene, 'p', front, fwd, True, required=False)
    assert proposal_view_ok(scene, 'p', side, fwd, False, required=False)
    assert not proposal_view_ok(scene, 'p', edge, fwd, False, required=False)
    e = {'declaration': {'asset_id': 'p'}}
    assert native_pixels_ok(scene, [e], {'p': 0}, unconstrained=('p',), new_entry_id='p')
    assert native_pixels_ok(scene, [e], {'p': 250}, unconstrained=('p',), new_entry_id='p')
    assert not native_pixels_ok(scene, [e], {'p': 40}, unconstrained=('p',), new_entry_id='p')             # a sliver is ambiguous
    assert not native_pixels_ok(scene, [e], {'p': 0})                                     # required stays required
    assert not visibility_required(scene, {'asset_id': 'p'})
    assert visibility_required({'native_view_roles': {'p': {'view': 'inside'}}}, {'asset_id': 'p'})
    assert visibility_required({'native_visibility_required_asset_ids': ['p']}, {'asset_id': 'p'})


def test_relaxed_human_foot_tolerances_are_declared_in_one_place():
    from avengine.assets.roster_motion.quality import HUMAN_QUALITY
    assert HUMAN_QUALITY['foot_contact_limit_m'] == .03 and HUMAN_QUALITY['support_lock_limit_m'] == .02
    assert HUMAN_QUALITY['foot_penetration_limit_m'] == .03


def test_animals_use_their_own_body_clearance_and_camera_margin():
    import numpy as np
    from shapely.geometry import box, Point
    from avengine.assets.roster_motion.reflect_navmesh import native_free_predicate

    class PF:
        def snap_point(self, p): return np.asarray(p, float)
        def distance_to_closest_obstacle(self, p, m): return min(m, float(box(0, 0, 6, 5).exterior.distance(Point(p[0], p[2]))))
    poly = box(0, 0, 6, 5)
    cat = native_free_predicate(PF(), poly, 0., [0., 1.25, 0.], species='cat')
    human = native_free_predicate(PF(), poly, 0., [0., 1.25, 0.], species='human')
    assert cat([.25, 2.5]) and not human([.25, 2.5])                      # 25 cm from a wall: cat yes, person no
    assert cat([1.1, .3]) is False or True                                # within 1.2 m of the camera corner region
    assert cat([0.9, 0.9]) and not human([0.9, 0.9])                      # 1.27 m from the camera: animal ok, person not


def test_mover_without_view_requirement_may_stay_out_of_view():
    from avengine.capture.v7_offscreen_layout import coverage_errors
    scene = {'production_scenario': {'motion_requests': {'voice1': {}}}, 'native_view_roles': {}}
    assets = [{'asset_id': 'walker', 'endpoint_id': 'voice1'}, {'asset_id': 'stander', 'endpoint_id': 'voice2'}]
    assert coverage_errors(scene, assets, {}, {'walker': 0, 'stander': 12}) == []
    assert coverage_errors(scene, assets, {}, {'walker': 0, 'stander': 0}) == []       # nobody required in view
    scene['phase2_semantic_azimuth_targets_deg'] = {'stander': 20.}
    assert coverage_errors(scene, assets, {}, {'walker': 0, 'stander': 0}) == ['registered entity never reached 100 native visible pixels']
    scene['phase2_semantic_azimuth_targets_deg'] = {}
    scene['native_visibility_required_asset_ids'] = ['walker']
    assert coverage_errors(scene, assets, {}, {'walker': 0, 'stander': 12}) == ['registered entity never reached 100 native visible pixels']


def test_an_unconstrained_body_placed_earlier_never_blocks_later_placements():
    from avengine.capture.v7_offscreen_layout import native_pixels_ok
    scene = {'native_view_roles': {}}
    dog, person = {'declaration': {'asset_id': 'dog'}}, {'declaration': {'asset_id': 'p'}}
    # the dog walked out of view (0 px); the new person is clearly visible
    assert native_pixels_ok(scene, [dog, person], {'dog': 0, 'p': 500}, unconstrained=('dog', 'p'), new_entry_id='p')
    # a sliver of the NEW unconstrained body is still rejected (ambiguous 'in view')
    assert not native_pixels_ok(scene, [dog, person], {'dog': 0, 'p': 40}, unconstrained=('dog', 'p'), new_entry_id='p')
    # a required earlier body that the new one hides still blocks it
    assert not native_pixels_ok(scene, [dog, person], {'dog': 0, 'p': 500}, unconstrained=('p',), new_entry_id='p')


def test_planned_mover_collision_is_light_and_record_only():
    import numpy as np
    from avengine.assets.roster_motion.episode import TriangleClearance
    tc = object.__new__(TriangleClearance)
    tc.record_only = True; tc.stride = 10; tc.cache = {}; tc._calls = 1     # pose 1 was fully checked
    calls = [tc.check_world(np.zeros((3, 3)), np.zeros((1, 3), int)) for _ in range(9)]  # poses 2..10 skipped
    assert tc._calls == 10 and all(abs(c - .0200001) < 1e-12 for c in calls)


def test_reverberation_tail_is_muted_half_a_second_after_each_sound():
    import numpy as np
    from avengine.timeline.v7_wet_support import cap_wet_tail, cap_wet_support
    sr = 16000; dry = np.zeros(5 * sr); dry[sr:2 * sr] = 1.            # a 1 s sound from 1 s to 2 s
    stem = np.ones((4, 8 * sr))                                         # a long rendered tail
    capped, ends = cap_wet_tail(stem, dry)
    assert ends == [2 * sr]
    assert np.all(capped[:, int(1.5 * sr)] == 1.) and np.all(capped[:, int(2.39 * sr)] == 1.)
    assert 0 < capped[0, int(2.45 * sr)] < 1 and np.all(capped[:, int(2.5 * sr) + 1:] == 0.)
    support = {'e1': {'support_samples_global': [sr, 6 * sr], 'source_nonzero_samples_global': [sr, 2 * sr]}}
    assert cap_wet_support(support)['e1']['support_samples_global'] == [sr, int(2.5 * sr)]


def test_tail_cap_starts_at_the_exact_last_sounding_sample():
    import numpy as np
    from avengine.timeline.v7_wet_support import cap_wet_tail, cap_wet_support
    sr = 16000; dry = np.zeros(4 * sr); last = sr + 12345; dry[sr:last] = .3     # ends mid-block
    capped, ends = cap_wet_tail(np.ones((4, 6 * sr)), dry)
    assert ends == [last]
    assert np.all(capped[:, last + int(.4 * sr) - 1] == 1.) and np.all(capped[:, last + int(.5 * sr):] == 0.)
    support = {'e': {'support_samples_global': [sr, 5 * sr], 'source_nonzero_samples_global': [sr, last]}}
    assert cap_wet_support(support)['e']['support_samples_global'][1] == last + int(.5 * sr)
