from pathlib import Path
import pytest
from tools.timeline.export_v7_trial import retain
pytestmark=pytest.mark.fast_unit


def test_authorized_helper_media_is_read_only_and_destination_stays_owned(tmp_path):
    own=tmp_path/'acceptance';helper=tmp_path/'helper';own.mkdir();helper.mkdir()
    original=helper/'source.mp4';original.write_bytes(b'native-media')
    original_links=original.stat().st_nlink
    target=own/'export.mp4'
    result=retain(original,target,own,(helper,))
    assert original.read_bytes()==target.read_bytes()==b'native-media'
    assert result['bytes']==12
    assert original.stat().st_ino!=target.stat().st_ino
    assert original.stat().st_nlink==original_links


def test_helper_source_authority_never_grants_write_authority(tmp_path):
    own=tmp_path/'acceptance';helper=tmp_path/'helper';own.mkdir();helper.mkdir()
    original=helper/'source';original.write_bytes(b'evidence')
    with pytest.raises(ValueError,match='export must remain'):
        retain(original,helper/'would-write-elsewhere',own,(helper,))
    assert not (helper/'would-write-elsewhere').exists()


def test_undeclared_external_source_still_rejected(tmp_path):
    own=tmp_path/'acceptance';helper=tmp_path/'helper';own.mkdir();helper.mkdir()
    original=helper/'source';original.write_bytes(b'evidence')
    with pytest.raises(ValueError,match='must come from'):
        retain(original,own/'target',own)
    assert not (own/'target').exists()
