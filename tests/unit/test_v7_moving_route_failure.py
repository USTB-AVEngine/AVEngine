import pytest
from avengine.assets.animal_motion.production_episode import motion_route_attempts,require_moving_animal_mode
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize('mode',['walk_animation','slide_no_animation'])
def test_exhausted_moving_routes_cannot_emit_a_stationary_result(mode):
    assert motion_route_attempts([[],[]],mode,[('legal static body pose',)])==[]

@pytest.mark.parametrize('mode',['static_after_failed_route','static','intentional_slide_after_failed_walk_route'])
def test_static_authority_cannot_be_used_for_a_moving_request(mode):
    with pytest.raises(ValueError,match='no static fallback'):require_moving_animal_mode(mode)
