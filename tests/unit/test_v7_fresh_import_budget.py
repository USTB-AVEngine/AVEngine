import json,os
import pytest
from avengine.timeline.v7_fresh_import_budget import import_space_decision,active_import_reservation
pytestmark=pytest.mark.fast_unit
POLICY={'maximum_task_bytes':100_000_000_000,'launch_disk_policy':{'minimum_free_bytes':400_000_000_000},'per_parent_disk_reserve_bytes':3_000_000_000}


def test_copy_and_active_parent_growth_are_reserved_before_import():
    report=import_space_decision(POLICY,1_000_000_000,allocated=94_000_000_000,free=600_000_000_000,active_parents=2)
    assert report['status']=='paused_before_import'
    assert report['reason']=='task_output_reservation_no_new_import'
    assert report['active_processes_signalled']is False
    assert import_space_decision(POLICY,1_000_000_000,allocated=20_000_000_000,free=600_000_000_000,active_parents=2)['status']=='admitted'


def test_copy_cannot_consume_the_required_shared_filesystem_headroom():
    result=import_space_decision(POLICY,2_000_000_000,allocated=1,free=402_000_000_000)
    assert result['reason']=='filesystem_headroom_reservation_no_new_import'


def test_reservation_counts_only_the_still_live_exact_process(tmp_path):
    (tmp_path/'evidence').mkdir();path=tmp_path/'evidence/import_space_reservation.json'
    stat=open(f'/proc/{os.getpid()}/stat').read().rsplit(')',1)[1].split()
    value={'pid':os.getpid(),'start_ticks':stat[19],'state':'reserved','reserved_bytes':123}
    path.write_text(json.dumps(value));assert active_import_reservation(tmp_path)==123
    value['start_ticks']='reused_pid';path.write_text(json.dumps(value));assert active_import_reservation(tmp_path)==0
    value.update(start_ticks=stat[19],state='released');path.write_text(json.dumps(value));assert active_import_reservation(tmp_path)==0


def test_older_valid_receipt_gets_measured_instead_of_rejected_for_missing_size(tmp_path):
    from avengine.timeline.v7_fresh_import_budget import measured_import_payload_bytes
    (tmp_path/'seal').mkdir();media=tmp_path/'seal/audio.wav';media.write_bytes(b'x'*4096)
    pointer=tmp_path/'source_seal.json';pointer.write_text(json.dumps({'attempt':'seal','required_files':['audio.wav'],'result':{'windows':[]}}))
    assert measured_import_payload_bytes({'export':{'source_seal':str(pointer)}})==4096+pointer.stat().st_size
