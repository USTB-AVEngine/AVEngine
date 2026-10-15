import datetime,json
import pytest
from avengine.timeline.v7_native_failure_retention import mark_new_attempt,retire_failed_media
from avengine.timeline.v7_storage_budget import ENV_KEY
pytestmark=pytest.mark.fast_unit

def policy(tmp_path,monkeypatch):
    (tmp_path/'inputs').mkdir();(tmp_path/'inputs/task_scope.json').write_text(json.dumps({'new_failed_attempt_retention':{'enabled':True,'created_after_utc':'2026-01-01T00:00:00+00:00','minimum_file_bytes':8,'authority':'explicit test owner'}}));monkeypatch.setenv(ENV_KEY,json.dumps({'root':str(tmp_path)}))

def test_preexisting_files_cannot_gain_a_cleanup_marker(tmp_path,monkeypatch):
    policy(tmp_path,monkeypatch);old=tmp_path/'old';old.mkdir();video=old/'scene.mp4';video.write_bytes(b'old original video');assert mark_new_attempt(old)is None
    retire_failed_media(old,None,ValueError('fail'));assert video.read_bytes()==b'old original video'

def test_new_failed_media_is_removed_only_after_durable_receipt(tmp_path,monkeypatch):
    policy(tmp_path,monkeypatch);attempt=tmp_path/'new';attempt.mkdir();marker=mark_new_attempt(attempt);video=attempt/'scene.mp4';video.write_bytes(b'new failed media');evidence=attempt/'native.json';evidence.write_text('{"failed":true}')
    receipt=retire_failed_media(attempt,marker,ValueError('actual native failure'));assert not video.exists()and evidence.exists()and marker.exists();saved=json.loads((attempt/'failed_intermediate_retention_receipt.json').read_text());assert saved['intermediate_media'][0]['bytes']==16 and saved['old_files_removed']is False
