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
