import pytest
from avengine.capture.v7_native_contact_construction import fixed_root_offset
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize('gaps',[[.0036,.0077,.018],[-.0008,.0017,-.019]])
def test_actual_native_floor_views_allow_one_fixed_correction(gaps):
    offset=fixed_root_offset(gaps,.015)
    assert all(abs(g+offset)<=.015 for g in gaps)
    assert offset<0 if max(gaps)>.015 else offset>0

def test_incompatible_measured_support_views_require_a_new_position():
    with pytest.raises(ValueError,match='another legal position'):fixed_root_offset([-.025,.015],.015)
