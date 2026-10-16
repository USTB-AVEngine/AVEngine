import numpy as np
import pytest
from shapely.geometry import box
from avengine.timeline.v7_spatial_layout import separated_layout
pytestmark=pytest.mark.fast_unit


def test_role_positions_survive_later_social_layout_while_headings_remain_natural():
    positions=[[-1.,1.4,-3.],[1.,1.4,-4.],[0.,1.4,2.],[.2,1.4,-2.]]
    layout={'listener_m':[0.,1.2,0.],'room_floor_polygon':box(-6,-6,6,6).__geo_interface__,'valid':[{'position_m':p,'body_yaw_deg':0.}for p in positions],'preserve_role_positions':True}
    first=separated_layout(layout,4,seed=1);other=separated_layout(layout,4,seed=2)
    assert [p['position_m']for p in first]==positions
    assert [p['position_m']for p in other]==positions
    assert [p['body_yaw_deg']for p in first]!=[p['body_yaw_deg']for p in other]
