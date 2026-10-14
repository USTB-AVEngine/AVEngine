import numpy as np
import pytest
from avengine.assets.animal_motion.production_episode import replay_paw_support
from avengine.assets.animal_motion.checks import PLANT_DRIFT_M

pytestmark = pytest.mark.fast_unit


def test_native_vertex_support_is_not_the_different_sole_region_centre():
    frame = {'root_world_m': [5., 0., 2.], 'head_m': [1., 0., 0.], 'tail_m': [0., 0., 0.],
             'head_world_m': [6., 0., 2.], 'tail_world_m': [5., 0., 2.]}
    foot = {'sole_world_m': [5.041, 0., 2.], 'support_world_m': [5., 0., 2.],
            'support_target_world_m': [5.002, 0., 2.], 'support_vertex': [0, 7]}
    value = replay_paw_support(frame, foot, np.array([1., 0., -3.]), np.pi/2)
    assert value['support_representation'] == 'native_support_vertex'
    assert value['support_lock_error_m'] == pytest.approx(.002)
    assert value['support_lock_error_m'] < PLANT_DRIFT_M
    foot['support_target_world_m'] = [5.004, 0., 2.]
    rejected = replay_paw_support(frame, foot, np.array([1., 0., -3.]), np.pi/2)
    assert rejected['support_lock_error_m'] > PLANT_DRIFT_M


def test_legacy_sole_support_retains_its_original_distance():
    frame = {'root_world_m': [0., 0., 0.], 'head_m': [1., 0., 0.], 'tail_m': [0., 0., 0.],
             'head_world_m': [1., 0., 0.], 'tail_world_m': [0., 0., 0.]}
    foot = {'sole_world_m': [.001, 0., 0.], 'support_world_m': [0., 0., 0.]}
    value = replay_paw_support(frame, foot, np.zeros(3), 0.)
    assert value['support_representation'] == 'legacy_sole_support'
    assert value['support_lock_error_m'] == pytest.approx(.001)
