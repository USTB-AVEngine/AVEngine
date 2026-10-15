import json,os
import pytest
from avengine.timeline.v7_native_stage_resume import compatible_relocated_pool_recipe
pytestmark=pytest.mark.fast_unit


def test_relocation_requires_identical_native_pool_bytes_and_registry(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';a=old/'raw';b=new/'raw';a.mkdir(parents=True);b.mkdir(parents=True)
    (a/'facts.json').write_text('actual facts');os.link(a/'facts.json',b/'facts.json')
    previous={'source_pool':str(a),'registry_sha':'exact'};requested={**previous,'source_pool':str(b)}
    (new/'native_stage_reuse.json').write_text(json.dumps({'source_parent':str(old)}));(new/'frozen_completed.json').write_text(json.dumps({'recipe':previous}))
    assert compatible_relocated_pool_recipe(new,'frozen',requested)==previous
    with pytest.raises(ValueError,match='more than its native source path'):
        compatible_relocated_pool_recipe(new,'frozen',{**requested,'registry_sha':'changed'})
    (b/'extra.json').write_text('different evidence')
    with pytest.raises(ValueError,match='source files differ'):
        compatible_relocated_pool_recipe(new,'frozen',requested)
