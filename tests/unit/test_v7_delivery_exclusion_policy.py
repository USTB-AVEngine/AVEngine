import json
import pytest
from tools.timeline.finalize_v7_acceptance import admitted_receipts
pytestmark=pytest.mark.fast_unit


def test_explicit_delivery_exclusion_keeps_the_original_native_receipt_unchanged(tmp_path):
    d=tmp_path/'evidence/jobs';d.mkdir(parents=True);p=d/'000.json';raw=json.dumps({'slot':0,'scene_id':'actual_scene','status':'passed','reader':{'status':'pass'}});p.write_text(raw)
    manifest={'output_root':str(tmp_path),'parents':[{'slot':0}],'delivery_excluded_parent_scene_ids':{'actual_scene':'actual native heading proof'}}
    rows,reasons=admitted_receipts(manifest)
    assert rows==[]and reasons[0]['scene_id']=='actual_scene'
    assert p.read_text()==raw
    # An empty policy does not silently ignore a broken admitted receipt.
    manifest['delivery_excluded_parent_scene_ids']={}
    with pytest.raises(KeyError):admitted_receipts(manifest)
