import json
import pytest
from avengine.timeline.v7_native_stage_resume import require_completed_registered_media
pytestmark=pytest.mark.fast_unit


def test_resume_requires_all_original_native_stages_and_media_before_any_renderer(tmp_path):
    row={'members':['v0_a0','v0_a1','v1_a0','v1_a1']}
    with pytest.raises(ValueError,match='missing original native stage'):
        require_completed_registered_media(tmp_path,row)
    names=['dry','visual_v0','visual_v1',*['audio_'+m for m in row['members']]]
    for name in names:
        d=tmp_path/name;d.mkdir();(d/'actual.bin').write_bytes(b'native')
        (tmp_path/(name+'_completed.json')).write_text(json.dumps({'attempt':name,'required_files':['actual.bin']}))
    assert require_completed_registered_media(tmp_path,row)==names
    (tmp_path/'audio_v1_a1/actual.bin').unlink()
    with pytest.raises(ValueError,match='original native stage incomplete'):
        require_completed_registered_media(tmp_path,row)
