import json
import pytest
from avengine.timeline.v7_registered_content import retain_readback
pytestmark=pytest.mark.fast_unit


def test_changed_content_witness_is_rejected_and_original_is_preserved(tmp_path):
    p=tmp_path/'native.json';original={'clock':150,'registry':'original'}
    retain_readback(p,original)
    with pytest.raises(ValueError,match='registry'):
        retain_readback(p,{'clock':150,'registry':'different'})
    assert json.loads(p.read_text())==original
    evidence=json.loads(next(tmp_path.glob('*.rejected_*.json')).read_text())
    assert evidence['changed_fields']==['registry']
    assert evidence['original_witness_preserved']is True


def test_identical_content_witness_resume_does_not_write_a_diagnostic(tmp_path):
    p=tmp_path/'native.json';value={'clock':150}
    retain_readback(p,value);retain_readback(p,value)
    assert not list(tmp_path.glob('*.rejected_*.json'))
