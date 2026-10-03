"""A failed MP3D input must remain review without interrupting other houses."""
import json
from types import SimpleNamespace
import pytest
from tools.acoustics.summarize_train_house_pool import visual_mp3d

pytestmark=pytest.mark.fast_unit

def test_missing_native_result_keeps_house_review_and_finishes_batch(tmp_path):
    h=dict(house='mp3d_missing',scan_id='missing',scene_directory=str(tmp_path/'source'))
    (tmp_path/'mp3d_manifest.json').write_text(json.dumps({'houses':[h]}))
    (tmp_path/'evidence').mkdir()
    visual_mp3d(SimpleNamespace(output=tmp_path,workers=1))
    d=tmp_path/'mp3d_acoustics/houses/mp3d_missing'
    assert not (d/'result.json').exists()
    check=json.loads((d/'visual_checks.json').read_text())
    assert check['status']=='review'
    assert any('NATIVE_MEASUREMENT_MISSING' in s for s in check['reasons'])
    assert json.loads((tmp_path/'evidence/mp3d_complete.json').read_text())['complete']

def test_failed_raw_loader_preserves_native_evidence_and_input(tmp_path,monkeypatch):
    import avengine.acoustics.house_acceptance_visual as visual
    raw=tmp_path/'source/broken.glb';raw.parent.mkdir();raw.write_bytes(b'original broken source')
    h=dict(house='mp3d_broken',scan_id='broken',scene_directory=str(raw.parent))
    (tmp_path/'mp3d_manifest.json').write_text(json.dumps({'houses':[h]}))
    (tmp_path/'evidence').mkdir()
    d=tmp_path/'mp3d_acoustics/houses/mp3d_broken';d.mkdir(parents=True)
    native=json.dumps({'status':'pass','foa_checks':[{'direct_line_clear':True}]})
    (d/'result.json').write_text(native)
    calls=[]
    def fail(path):
        calls.append(path)
        raise ValueError('synthetic raw loading failure')
    monkeypatch.setattr(visual,'load_raw_scan',fail)
    visual_mp3d(SimpleNamespace(output=tmp_path,workers=1))
    check=json.loads((d/'visual_checks.json').read_text())
    assert check['status']=='review' and calls==[raw]
    assert 'RAW_VISUAL_LOAD_FAILED' in check['reasons'][0]
    assert (d/'result.json').read_text()==native and raw.read_bytes()==b'original broken source'
    assert json.loads((tmp_path/'evidence/mp3d_complete.json').read_text())['complete']
