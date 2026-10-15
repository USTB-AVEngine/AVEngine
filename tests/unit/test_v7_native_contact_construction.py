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


def test_later_native_corrections_keep_the_previous_measurement_bytes(tmp_path):
    from types import SimpleNamespace
    from avengine.capture.v7_native_contact_construction import publish_contact_receipt
    video=SimpleNamespace(contact_construction_path=tmp_path/'native_vertical_contact_construction.json')
    first=publish_contact_receipt(video,{'measured_gap_m':.01664});original=first.read_bytes()
    second=publish_contact_receipt(video,{'measured_gap_m':.01527})
    assert first!=second and first.read_bytes()==original
    assert video.contact_construction_path==second
