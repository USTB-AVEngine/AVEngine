import json
from pathlib import Path
import pytest
from tools.timeline.recover_v7_native_selection import recover
pytestmark=pytest.mark.fast_unit


def test_recovery_rejects_unapproved_source_without_writing_native_outputs(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'output_root':str(tmp_path),'peer_output_roots':[]}))
    with pytest.raises(ValueError,match='outside authorized evidence roots'):
        recover(manifest,Path('/unrelated/other-project/parent'))
    assert not(tmp_path/'native_reselections').exists()


def test_partial_question_policy_cannot_be_recovered_as_a_full_parent(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'output_root':str(tmp_path),'peer_output_roots':[]}))
    source=tmp_path/'source';source.mkdir();(source/'input_recipe.json').write_text(json.dumps({'row':{'scene_id':'p','split':'train','acceptance_allow_partial_families':True}}))
    with pytest.raises(ValueError,match='complete train-only scene required'):
        recover(manifest,source)
    assert not(tmp_path/'native_reselections').exists()
