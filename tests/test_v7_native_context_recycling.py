from types import SimpleNamespace
import numpy as np
import pytest
from avengine.capture.v7_native_context_recycling import recycle_pose_context
pytestmark=pytest.mark.fast_unit
class Object:
    def __init__(self,root):self.root=root.copy();self.root_scene_node=SimpleNamespace(absolute_transformation=lambda:self.root)
class Sim:
    def __init__(self):self.closed=False
    def close(self):self.closed=True
class Video:
    def __init__(self,assets):
        self.actors=[{'endpoint_id':key,'declaration':{'asset_id':value},'object':Object(np.eye(4))}for key,value in assets.items()]
        self.sim=Sim();self.hs=self.mn=self.qt=self.pathfinder=self.material_rendering=None
    def close(self):self.sim.close()

def test_recycling_keeps_required_actor_dictionaries_roots_and_motion_clock(monkeypatch):
    import avengine.assets.habitat_capture as module
    monkeypatch.setattr(module,'_apply_root_with_habitat',lambda obj,root,**kw:setattr(obj,'root',root.copy()))
    identities={'voice1':'walking','voice2':'standing','voice3':'silent'};video=Video(identities)
    video._pose_context_configuration={'arguments':(identities,),'keywords':{}};video._pose_resources_in_context=set(range(64));video.receipt={}
    video._current_visibility_stream=SimpleNamespace(last={'old':'mask'})
    entry=video.actors[0];entry['motion_episode']={'frame_count':901};entry['object'].root[0,3]=2.75;old=video.sim
    assert recycle_pose_context(video,{},128)
    assert old.closed and not video.sim.closed
    assert video.actors[0]is entry and entry['object'].root[0,3]==2.75
    assert entry['motion_episode']['frame_count']==901
    assert video._current_visibility_stream.last=={}
    assert video.receipt['native_context_restarts'][0]['actual_actor_ids_preserved']==identities


def test_completed_visual_stage_releases_its_native_capture_reservation():
    from avengine.capture.continuous_native_video import NativeContinuousVideo
    video=NativeContinuousVideo.__new__(NativeContinuousVideo);video.receipt={};video.actors=[]
    sim=Sim();lease=Sim();video.sim=sim;video._task_motion_capture_lease=lease
    video.close()
    assert sim.closed and lease.closed
    assert video.sim is None
