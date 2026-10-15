from types import SimpleNamespace
import numpy as np
import pytest
from avengine.timeline.continuous_pilot import PositionRIRCache

pytestmark=pytest.mark.fast_unit

class Renderer:
    def __init__(self):self.calls=[]
    def render(self,positions):
        self.calls.append(positions)
        return SimpleNamespace(samples=np.stack([np.full((4,8),p[0]+1.)for p in positions]),
            wall_seconds=2.,process_cpu_seconds=3.*len(positions))

def test_batch_keeps_exact_pose_order_and_deduplicates_repeated_keys(tmp_path):
    r=Renderer();c=PositionRIRCache(tmp_path,r,{'scene':'unit'})
    try:
        c.prefetch([[3,0,0],[1,0,0],[3,0,0]])
        assert r.calls==[[[3.,0.,0.],[1.,0.,0.]]]
        assert np.all(c.get([3,0,0])==4.)
        assert np.all(c.get([1,0,0])==2.)
        assert c.stats['new_rirs']==2
        assert c.stats['native_wall_seconds']==2.
        assert c.stats['native_cpu_seconds']==6.
    finally:c.db.close()

def test_resumed_exact_cache_needs_no_new_native_batch(tmp_path):
    r=Renderer();c=PositionRIRCache(tmp_path,r,{'scene':'unit'});c.prefetch([[2,0,0]]);c.db.close()
    resumed=Renderer();c=PositionRIRCache(tmp_path,resumed,{'scene':'unit'})
    try:
        c.prefetch([[2,0,0]])
        assert resumed.calls==[]
        assert np.all(c.get([2,0,0])==3.)
    finally:c.db.close()

def test_different_exact_positions_cannot_share_a_prefetched_kernel(tmp_path):
    r=Renderer();c=PositionRIRCache(tmp_path,r,{'scene':'unit'})
    try:
        c.prefetch([[1,0,0],[1.000001,0,0]])
        assert len(r.calls[0])==2
        assert not np.array_equal(c.get([1,0,0]),c.get([1.000001,0,0]))
    finally:c.db.close()

def test_invalid_position_fails_before_any_native_render(tmp_path):
    r=Renderer();c=PositionRIRCache(tmp_path,r,{'scene':'unit'})
    try:
        with pytest.raises(ValueError):c.prefetch([[float('nan'),0,0]])
        assert r.calls==[]
    finally:c.db.close()


def test_fresh_batch_preserves_each_complete_native_tail(tmp_path,monkeypatch):
    from avengine.timeline.v7_phase2_fresh_rir import FreshNativeContextRenderer
    r=FreshNativeContextRenderer({'fresh_rir_workers':2},{},{},{},tmp_path/'native')
    def render_one(position,role,trial):
        size=8 if position[0]==0 else 17
        return np.arange(4*size,dtype=float).reshape(4,size),'report_'+str(size),1.
    monkeypatch.setattr(r,'_render_one',render_one)
    batch=r.render([[0,0,0],[1,0,0]])
    assert [x.shape for x in batch.samples]==[(4,8),(4,17)]
    np.testing.assert_array_equal(batch.samples[1],np.arange(68,dtype=float).reshape(4,17))
    assert batch.native_process_evidence==['report_8','report_17']


def test_one_fresh_kernel_retains_the_scalar_batch_shape(tmp_path,monkeypatch):
    from avengine.timeline.v7_phase2_fresh_rir import FreshNativeContextRenderer
    r=FreshNativeContextRenderer({},{},{},{},tmp_path/'native')
    monkeypatch.setattr(r,'_render_one',lambda p,role,trial:(np.ones((4,11)),'report',1.))
    assert r.render([[0,0,0]]).samples.shape==(1,4,11)
