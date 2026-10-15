import pytest
from avengine.capture.v7_visible_native import attached_emitter_readback_error

@pytest.mark.parametrize("error", [0., .00011, .004999, .005])
def test_native_readback_preserves_actual_error_within_five_mm(error):
    assert attached_emitter_readback_error([error,0.,0.],[0.,0.,0.]) == pytest.approx(error)

@pytest.mark.parametrize("error", [.005001, float("nan"), float("inf")])
def test_native_readback_rejects_excess_or_nonfinite_error(error):
    with pytest.raises(ValueError, match="exceeds 5mm"):
        attached_emitter_readback_error([error,0.,0.],[0.,0.,0.])
