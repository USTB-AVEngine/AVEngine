import hashlib,json,subprocess
import pytest
from avengine.qa.v7_content_binding import _native_video_probe
pytestmark=pytest.mark.fast_unit


def test_native_video_decode_is_reused_only_for_identical_verified_bytes(tmp_path,monkeypatch):
    import avengine.qa.v7_content_binding as module
    path=tmp_path/'native.mp4';path.write_bytes(b'first-encoded-native-window');calls=[]
    def probe(command,**kwargs):
        calls.append(command)
        assert '-count_frames'in command
        return subprocess.CompletedProcess(command,0,stdout=json.dumps({'streams':[{'nb_read_frames':'150'}]}))
    monkeypatch.setattr(module.subprocess,'run',probe);_native_video_probe.cache_clear()
    digest=lambda:hashlib.sha256(path.read_bytes()).hexdigest()
    first=_native_video_probe(str(path),digest());second=_native_video_probe(str(path),digest())
    assert first==second and len(calls)==1
    path.write_bytes(b'different-encoded-native-window')
    _native_video_probe(str(path),digest())
    assert len(calls)==2
    _native_video_probe.cache_clear()


def test_failed_native_decode_is_not_cached(tmp_path,monkeypatch):
    import avengine.qa.v7_content_binding as module
    calls=[]
    def fail(command,**kwargs):
        calls.append(command);raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(module.subprocess,'run',fail);_native_video_probe.cache_clear()
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):_native_video_probe(str(tmp_path/'bad.mp4'),'same-digest')
    assert len(calls)==2
    _native_video_probe.cache_clear()
