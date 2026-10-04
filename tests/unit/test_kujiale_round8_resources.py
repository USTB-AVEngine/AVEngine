import os
import pytest
from tools.rooms.v7_kujiale_round8_resources import allocation,disk_violation
from tools.rooms.benchmark_kujiale_native_concurrency import choose_topology
pytestmark=pytest.mark.fast_unit


def test_allocation_deduplicates_hardlinks_and_excludes_external_symlinks(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    source=a/'data';source.write_bytes(b'x'*16384);os.link(source,b/'same')
    (b/'not_ours').symlink_to('/etc/passwd')
    got=allocation([a,b]);assert got['allocated_bytes']==source.stat().st_blocks*512
    assert got['logical_bytes']==32768 and got['unique_inodes']==1


def test_revoked_150GB_reserve_is_not_applied():
    baseline={'minimum_shared_disk_free_bytes':80*1024**3,'own_allocated_bytes_at_start':40_000_000_000,
              'maximum_new_allocated_bytes':40_000_000_000,'maximum_total_task_allocated_bytes':80_000_000_000}
    assert disk_violation(123_000_000_000,41_000_000_000,baseline) is None
    assert disk_violation(80*1024**3+1_000_000_000,41_000_000_000,baseline)=='shared_disk_80GiB_floor_buffer'
    assert disk_violation(123_000_000_000,79_000_000_000,baseline)=='round8_40GB_growth_buffer'


def test_twelve_requires_two_cards_and_4GiB_per_card():
    cards=[{'index':i,'free_mib':v} for i,v in enumerate([27000,47000,30000,20000])]
    assert choose_topology(cards,12,3908)==[1]*6+[2]*6
    cards[2]['free_mib']=27400
    assert choose_topology(cards,12,3908)==[1]*8+[2]*4
    cards=[{'index':i,'free_mib':v} for i,v in enumerate([7000,39000,15000,9000])]
    with pytest.raises(RuntimeError):choose_topology(cards,12,3908)


def test_pan_skips_only_missing_measurements_and_never_a_highlight_failure():
    from avengine.capture.continuous_ue_video import native_material_sample_status
    missing={'asset_id':'blue','metrics':{'status':'not_run','reason':'fewer_than_100_source_pixels'}}
    seen={'asset_id':'blue','metrics':{'status':'pass'}}
    assert native_material_sample_status([missing,seen],['blue'])=='fail'
    assert native_material_sample_status([missing,seen],['blue'],allow_invisible_samples=True)=='pass'
    assert native_material_sample_status([missing],['blue'],allow_invisible_samples=True)=='fail'
    assert native_material_sample_status([seen,{'asset_id':'blue','metrics':{'status':'fail'}}],['blue'],allow_invisible_samples=True)=='fail'
