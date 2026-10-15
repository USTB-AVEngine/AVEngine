import json
import pytest
from avengine.timeline.v7_native_stage_resume import retain_exact_json_receipt
pytestmark=pytest.mark.fast_unit


def test_identical_recomputed_receipt_reuses_bytes_without_overwriting(tmp_path):
    p=tmp_path/'native.json';current={'status':'pass','checked':901}
    assert retain_exact_json_receipt(p,current)
    original=p.read_bytes();mtime=p.stat().st_mtime_ns
    assert not retain_exact_json_receipt(p,current)
    assert p.read_bytes()==original and p.stat().st_mtime_ns==mtime
    with pytest.raises(ValueError,match='differs from current measurements'):
        retain_exact_json_receipt(p,{'status':'blocked','checked':901})
    assert json.loads(p.read_text())==current
