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


def test_integer_actor_map_keys_resume_with_identical_json_truth(tmp_path):
    p=tmp_path/'native.json';value={'source_actor_onsets_s':{1:1.1,2:3.26}}
    retain_readback(p,value);before=p.read_bytes();mtime=p.stat().st_mtime_ns
    retain_readback(p,value)
    assert p.read_bytes()==before and p.stat().st_mtime_ns==mtime
    assert not list(tmp_path.glob('*.rejected_*.json'))


def test_changed_measured_onset_is_still_rejected_after_json_normalization(tmp_path):
    p=tmp_path/'native.json';retain_readback(p,{'source_actor_onsets_s':{1:1.1}})
    with pytest.raises(ValueError,match='source_actor_onsets_s'):
        retain_readback(p,{'source_actor_onsets_s':{1:1.2}})
    assert json.loads(p.read_text())['source_actor_onsets_s']=={'1':1.1}
