"""Partial-to-clear construction uses shared geometry on free and retained paths."""
import numpy as np
from avengine.qa.answerability import MeshHandle
from avengine.qa import generation_conditions as gc
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms import conditioned_visibility as cv


def scene():
    camera = cv.CameraPose(candidate_id='wall_edge', position_m=(0., 1.55, 0.),
        forward=(0., 0., -1.), right=(1., 0., 0.), up=(0., 1., 0.),
        horizontal_fov_deg=85., resolution_hw=(720, 1280))
    vertices = np.array([[-2., -1., -2.], [0., -1., -2.], [0., 3., -2.], [-2., 3., -2.]])
    mesh = MeshHandle(vertices, np.array([[0, 1, 2], [0, 2, 3]]))
    body = cv.BodyProxy(height_m=1.8, width_m=.4)
    policy = cv.screen_policy(None)
    cloud = np.array([[0., 0., -4.], [2., 0., -4.]])
    return camera, mesh, body, policy, cloud


class StraightSpace:
    def shortest_path(self, start, end):
        return np.asarray([start, end])


def assert_windows(route, record, camera, mesh, body, policy):
    construction = record['visibility_construction']
    for key, state in [('predicted_partial_frames', 'partial'), ('predicted_clear_frames', 'visible')]:
        first, last = construction[key]
        assert cs._public_interval_is_publishable(first, last, 15.)
        assert all(cs._point_state(camera, p, body, mesh, policy, {})[0] == state
                   for p in route[first:last])
    assert len(route) == 150


def test_free_path_constructs_partial_then_clear_with_public_windows():
    camera, mesh, body, policy, cloud = scene()
    rows = cs._classify_cloud(camera, cloud, body, [0., 1.6, 0.], mesh, policy, {},
                             max_distance_m=6, cast_rays=True, target_observability=True)
    assert [r['state'] for r in rows] == ['partial', 'visible']
    requirement = cv.VisibilityRequirement(kind='visible_occluded_to_visible_clear',
                                           subject='target', observation_windows=((0, 150),))
    result = cs._occlusion_route(requirement, rows, cloud, StraightSpace(), camera,
        body, mesh, policy, {}, np.random.default_rng(3), 150, 15.,
        {'walk_speed_range_mps': [.8, .8]}, [1., 6.], target_observability=True)
    assert result is not None
    assert_windows(*result, camera, mesh, body, policy)


def test_retained_route_uses_the_same_transition_without_changing_its_shape():
    camera, mesh, body, policy, cloud = scene()
    points = np.concatenate([np.repeat(cloud[0][None], 10, axis=0),
                             np.linspace(cloud[0], cloud[1], 30),
                             np.repeat(cloud[1][None], 35, axis=0)])
    requirement = cv.VisibilityRequirement(kind='visible_occluded_to_visible_clear',
                                           subject='target', observation_windows=((0, 150),))
    result = cs._bank_visibility_route(requirement,
        [{'route_id': 'retained_shape', 'points_m': points}], camera, body,
        np.array([0., 1.6, 0.]), mesh, policy, {}, np.random.default_rng(3),
        150, [1., 6.], 30)
    assert result is not None
    route, record = result
    assert record['native_route_id'] == 'retained_shape'
    assert np.allclose(route[record['start_frame']:record['end_frame_exclusive']], points)
    assert_windows(route, record, camera, mesh, body, policy)


def test_qa11_compiles_the_implemented_transition_without_an_emitter_ray_proxy():
    conditions = gc._HANDLERS['QA-11'](subjects=(gc.ConditionSubject('target', 'target'),))
    planning = conditions[0].planning
    assert planning == {'pixel_occlusion_partial_transition': 'visible_occluded_to_visible_clear'}
    assert 'pixel_occlusion_partial_transition' in cs.describe_generator_capabilities()['knobs']
    assert 'pixel_occlusion_partial_transition' not in gc.KNOB_GAPS
    assert cs._constructive_visibility_requested(
        {'qa_targets': [{'qa_id': 'QA-11'}]},
        [cv.VisibilityRequirement(kind='visible_occluded_to_visible_clear', subject='target',
                                 observation_windows=((0, 150),))],
        {'anchor_line_of_sight': 'clear', 'speech_motion': 'all_still'})
