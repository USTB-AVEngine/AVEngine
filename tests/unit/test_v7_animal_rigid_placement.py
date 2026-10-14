import numpy as np
import pytest
from types import SimpleNamespace
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


def test_lift_cannot_hide_an_incompatible_planted_paw_height():
    with pytest.raises(ValueError, match='no constant rigid animal route height'):
        rigid_route_placement(*stream(paw_height=.02), 0., SlopedFloor(), 'dog')


def test_original_input_roots_remain_immutable():
    args = stream()
    before = args[2].copy()
    rigid_route_placement(*args, 0., SlopedFloor(), 'dog')
    assert np.array_equal(args[2], before)
