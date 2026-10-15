import json,hashlib
import pytest
from avengine.timeline.v7_admitted_pool_reselection import compatible_prior_candidates
pytestmark=pytest.mark.fast_unit


def test_matching_original_native_pool_is_reusable_without_mutating_evidence(tmp_path):
    source=tmp_path/'seal.json';source.write_text('{"native":"passed"}')
    row={'question_en':'Which person made the sound?','choices':[],'model_answer':'the man in the blue shirt'}
    result={'source_seal':str(source),'source_seal_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'entries':[{'row':row,'key':['native','q']}]}
    cache=tmp_path/'v5.json';cache.write_text(json.dumps(result));before=source.read_bytes()
    assert compatible_prior_candidates(cache,source)==result
    assert source.read_bytes()==before
    source.write_text('{"native":"changed"}')
    assert compatible_prior_candidates(cache,source)is None


def test_prior_cache_does_not_bypass_current_english_gate(tmp_path):
    source=tmp_path/'seal.json';source.write_text('{}');cache=tmp_path/'v5.json'
    cache.write_text(json.dumps({'source_seal':str(source),'source_seal_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'entries':[{'row':{'question_en':'Which person 中文?','model_answer':'yes'}}]}))
    with pytest.raises(ValueError):compatible_prior_candidates(cache,source)
