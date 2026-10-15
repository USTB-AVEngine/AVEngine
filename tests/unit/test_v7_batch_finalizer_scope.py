from pathlib import Path
import json
import pytest
from tools.timeline.finalize_v7_acceptance import finalize
pytestmark=pytest.mark.fast_unit


def test_unattended_finalizer_requires_explicit_acceptance_scope(tmp_path):
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'output_root':str(tmp_path),'dataset_role':'training','training_dataset_merge':True,'CCI_transfer':False}))
    with pytest.raises(ValueError,match='acceptance-only'):finalize(path)
    assert not(tmp_path/'export/train.jsonl').exists()


def test_fresh_empty_batch_waits_without_claiming_a_reader_pass(tmp_path,monkeypatch):
    import json
    import tools.timeline.finalize_v7_acceptance as module
    (tmp_path/'evidence').mkdir()
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'output_root':str(tmp_path),'dataset_role':'acceptance_only',
        'training_dataset_merge':False,'CCI_transfer':False,'parents':[]}))
    monkeypatch.setattr(module,'admitted_receipts',lambda manifest:([],[]))
    result=module.finalize(path)
    assert result['status']=='waiting_for_native_admission'
    assert not(tmp_path/'export/train.jsonl').exists()
    assert not(tmp_path/'evidence/FINAL_MODEL_READER.json').exists()
