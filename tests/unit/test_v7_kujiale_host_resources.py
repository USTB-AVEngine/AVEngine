import time
import pytest
from avengine.timeline.v7_kujiale_host_resources import read_host,subtree_pss
from avengine.timeline.v7_kujiale_manifest import atomic_write
pytestmark=pytest.mark.fast_unit

def test_stale_or_missing_host_pss_cannot_authorize_a_new_isolated_job(tmp_path):
    assert read_host(tmp_path)is None
    (tmp_path/'evidence').mkdir();path=tmp_path/'evidence/host_resource_sample.json';atomic_write(path,{'sample_unix':time.time()-60,'PSS_bytes':1})
    assert read_host(tmp_path)is None
    atomic_write(path,{'sample_unix':time.time(),'PSS_bytes':35_000_000_000});assert read_host(tmp_path)['PSS_bytes']==35_000_000_000

def test_native_memory_reservation_counts_renderer_descendants_once():
    sample={'processes':[{'pid':1,'ppid':0,'pss_bytes':1},{'pid':2,'ppid':1,'pss_bytes':20},{'pid':3,'ppid':2,'pss_bytes':300},{'pid':4,'ppid':0,'pss_bytes':4000}]}
    assert subtree_pss(sample,1)==321 and subtree_pss(sample,2)==320
