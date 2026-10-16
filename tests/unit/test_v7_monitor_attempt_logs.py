import json
import pytest
from tools.timeline.monitor_v7_acceptance import next_parent_log,retained_parent_log
pytestmark=pytest.mark.fast_unit


def test_requeued_slot_gets_a_new_log_without_overwriting_prior_evidence(tmp_path):
    (tmp_path/'logs').mkdir();old=tmp_path/'logs/parent_008.log';old.write_text('original failed affinity attempt\n')
    actual=next_parent_log(tmp_path,8)
    assert actual!=old and not actual.exists()
    with actual.open('x')as stream:stream.write('new dispatch\n')
    assert old.read_text()=='original failed affinity attempt\n'
    (tmp_path/'evidence/launches').mkdir(parents=True)
    (tmp_path/'evidence/launches/008.json').write_text(json.dumps({'log':str(actual)}))
    assert retained_parent_log(tmp_path,8)==actual
