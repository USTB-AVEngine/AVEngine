from pathlib import Path
import json
import pytest
from avengine.timeline.v7_native_failure_retention import mark_new_attempt, retire_failed_media

pytestmark = pytest.mark.fast_unit


def test_failed_native_attempt_keeps_existing_and_new_media_and_failure(tmp_path):
    old = tmp_path / 'old.mp4'; old.write_bytes(b'original')
    marker = mark_new_attempt(tmp_path)
    new = tmp_path / 'scene_video.mp4'; new.write_bytes(b'new native frames')
    receipt = retire_failed_media(tmp_path, marker, ValueError('original rendering failure'))
    assert old.read_bytes() == b'original' and new.read_bytes() == b'new native frames'
    assert receipt['deleted_files'] == []
    assert json.loads((tmp_path/'native_attempt_failure.json').read_text())['failure'] == 'original rendering failure'


def test_retention_receipt_error_cannot_mask_original_render_failure(tmp_path, monkeypatch):
    marker = mark_new_attempt(tmp_path)
    def denied(*args, **kwargs):
        raise PermissionError('receipt not writable')
    monkeypatch.setattr(Path, 'write_text', denied)
    receipt = retire_failed_media(tmp_path, marker, RuntimeError('native failure'))
    assert receipt['failure'] == 'native failure'
    assert 'receipt_write_error' in receipt
