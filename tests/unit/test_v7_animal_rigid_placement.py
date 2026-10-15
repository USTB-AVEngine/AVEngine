import numpy as np
import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.fast_unit
from avengine.assets.animal_motion.production_episode import rigid_route_placement


class SlopedFloor:
    reference = 0.
    def many(self, points):
        return np.asarray(points)[:, 0] * .004


def stream(paw_height=0., skin_height=0.):
    foot = {'sole_world_m': [0., paw_height, 0.], 'in_contact': True}
    frame = {'head_m': [1., 0., 0.], 'tail_m': [0., 0., 0.],
             'head_world_m': [1., 0., 0.], 'tail_world_m': [0., 0., 0.],
             'root_world_m': [0., 0., 0.], 'feet': {'FL': foot}}
    mesh = SimpleNamespace(vertices=np.array([[0., skin_height, 0.], [0., .3, 0.]]))
    return ['pose', 'pose'], {'pose': mesh}, np.array([[0., -.003, 0.], [1., -.003, 0.]]), [frame], [0, 0]


def test_static_root_offset_is_replaced_by_one_rigid_stream_placement():
    args = stream()
    roots, proof = rigid_route_placement(*args, 0., SlopedFloor(), 'cat')
    shift = roots - args[2]
    assert np.allclose(shift[0], shift[1])
    assert proof['vertical_translation_m'] == pytest.approx(.0074)
    assert np.min(roots[:, 1] - SlopedFloor().many(roots)) >= .0004 - 1e-12
    assert np.max(np.abs(roots[:, 1] - SlopedFloor().many(roots))) < .012
    assert proof['per_frame_correction'] is False


def test_incompatible_paw_height_constructs_measured_skin_safe_frames_and_records_residual():
    args=stream(paw_height=.03)
    roots,proof=rigid_route_placement(*args, 0., SlopedFloor(), 'dog')
    assert proof['per_frame_correction'] is True
    assert proof['appearance_warnings']
    assert proof['truth_and_answer_gates_changed'] is False
    assert np.min(roots[:,1]-SlopedFloor().many(roots))>=-.00101
    assert np.ptp(roots[:,1])>0.


def test_original_input_roots_remain_immutable():
    args = stream()
    before = args[2].copy()
    rigid_route_placement(*args, 0., SlopedFloor(), 'dog')
    assert np.array_equal(args[2], before)


def test_preferred_positive_skin_gap_does_not_reject_a_valid_owner_contact():
    args = stream(paw_height=.0125)
    class FlatFloor:
        reference = 0.
        def many(self, points):
            return np.zeros(len(points))
    roots, proof = rigid_route_placement(*args, 0., FlatFloor(), 'dog')
    assert np.min(roots[:, 1]) >= -.00101
    assert np.max(roots[:, 1] + .0125) < .012
    assert proof['per_frame_correction'] is False


@pytest.mark.parametrize('requested', ['walk_animation', 'slide_no_animation'])
def test_route_search_preserves_requested_motion_at_each_speed(requested):
    from avengine.assets.animal_motion.production_episode import motion_route_attempts
    tiers = [[(.5, 'root', 'yaw', 'direction', speed)] for speed in [.03, .015, .0075, .003]]
    attempts = motion_route_attempts(tiers, requested, [(0., 'root', 'yaw', 'direction', 0.)])
    assert [mode for route, mode in attempts] == [requested] * 4
    assert [route[-1] for route, mode in attempts] == [.03, .015, .0075, .003]


def test_no_moving_route_cannot_silently_become_a_stationary_animal():
    from avengine.assets.animal_motion.production_episode import motion_route_attempts
    assert motion_route_attempts([[], [], [], []], 'walk_animation',
        [(0., 'root', 'yaw', 'direction', 0.)]) == []


def test_actual_static_labels_preserve_the_original_admission_and_positions():
    from avengine.assets.animal_motion.production_episode import realized_animal_motion
    scene = {'sources': {'animal': {'motion_marker': 'animated_walk',
        'motion_metadata': {'admission_path': 'immutable.json'},
        'trajectory': [{'time_s': 0., 'position_m': [1., 2., 3.]}]}}}
    record = {'species_id': 'cat', 'motion_marker': 'static_after_failed_route',
        'motion_mode': 'static_after_failed_route', 'requested_motion_mode': 'walk_animation',
        'actual_speed_m_s': 0., 'motion_degradation': 'static_after_failed_route'}
    realized_animal_motion(scene, {'native_motion_admissions': {'animal': record}})
    actual = scene['sources']['animal']
    assert actual['motion_marker'] == 'static_after_failed_route'
    assert actual['motion_metadata']['joint_animation_present'] is False
    assert actual['motion_metadata']['requested_mode'] == 'walk_animation'
    assert actual['motion_metadata']['admission_path'] == 'immutable.json'
    assert actual['trajectory'][0]['position_m'] == [1., 2., 3.]


def test_route_scoring_cannot_move_an_admitted_animal_below_camera():
    from avengine.assets.animal_motion.production_episode import route_preview_visible
    scene = {'listener_m': [0., 1.3, 0.], 'camera_yaw_knots': [
        {'yaw_deg': 0.}, {'yaw_deg': 80.}, {'yaw_deg': -80.}]}
    assert route_preview_visible(scene, [0., 0., -2.5], [0., 0., -2.4], 0., [0., .3, 0.])
    assert not route_preview_visible(scene, [0., 0., -.8], [0., 0., -.7], 0., [0., .3, 0.])
    assert not route_preview_visible(scene, [0., 0., 2.5], [0., 0., 2.4], 0., [0., .3, 0.])


def test_long_route_leaving_view_is_filtered_before_short_route_fallback():
    from avengine.assets.animal_motion.production_episode import route_preview_visible
    scene = {'listener_m': [0., 1.3, 0.], 'camera_yaw_knots': [{'yaw_deg': 0.}]}
    assert not route_preview_visible(scene, [0., 0., -2.5], [0., 0., -1.5], 0., [0., .3, 0.])
    assert route_preview_visible(scene, [0., 0., -2.5], [0., 0., -2.41], 0., [0., .3, 0.])
