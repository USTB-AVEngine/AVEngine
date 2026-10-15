"""A solved entry route keeps its speed and consumes the final gait clock."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from avengine.assets.roster_motion.navmesh_execution import prepare_scene_motion, curved_route_proposals
pytestmark = pytest.mark.fast_unit


def test_solved_entry_speed_is_not_randomly_replaced():
    catalog = [{'asset_id': 'person', 'kind': 'human', 'species_id': 'human'}]
    row = {'scene_seed': 42, 'motion_requests': {'person': {'action': 'walk', 'speed_m_s': .91,
            'joint_entry_route': {'endpoint_m': [0., 0., -4.]}}}}
    result = prepare_scene_motion(row, catalog)
    assert result['motion_requests']['person']['speed_m_s'] == .91
    row['motion_requests']['person']['speed_m_s'] = .4
    with pytest.raises(ValueError, match='owner species band'):
        prepare_scene_motion(row, catalog)


def test_entry_plugin_receives_the_final_complete_step_distance(monkeypatch):
    seen = []
    def generate(video, entry, scene, distance, forward, request):
        seen.append((distance, request))
        yield 'native proposal'
    monkeypatch.setitem(sys.modules, 'avengine.construction.joint.families.qa07',
                        SimpleNamespace(native_entry_route_proposals=generate))
    distance = np.linspace(0., 4.27, 901)
    request = {'joint_entry_route': {'endpoint_m': [0., 0., -4.]}}
    assert list(curved_route_proposals(None, None, None, distance, None, request)) == ['native proposal']
    assert seen[0][0] is distance and seen[0][1] is request
