"""Claude 10-15: construct instead of failing a scene on one unplaceable entity."""
from avengine.capture.v7_offscreen_layout import placement_construction_fallback, proposal_view_ok, offscreen_role
from avengine.timeline.v7_spatial_layout import spacing_rejection


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
    assert placement_construction_fallback(scene, mover, 'floor', None, False) is None      # movers stay in view
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
