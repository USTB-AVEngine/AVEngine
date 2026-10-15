from pathlib import Path
import json
import pytest
from tools.timeline.finalize_v7_acceptance import finalize
pytestmark=pytest.mark.fast_unit


def test_unattended_finalizer_requires_explicit_acceptance_scope(tmp_path):
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'output_root':str(tmp_path),'dataset_role':'training','training_dataset_merge':True,'CCI_transfer':False}))
    with pytest.raises(ValueError,match='acceptance-only'):finalize(path)
    assert not(tmp_path/'export/train.jsonl').exists()
