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
