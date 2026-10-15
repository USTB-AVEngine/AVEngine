import json
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.assets.roster_motion.diagnostics import write_route_summary
from avengine.capture.v7_registered_state import restore_actor_state
pytestmark=pytest.mark.fast_unit

def test_each_route_rejection_retains_evidence_and_can_update_summary(tmp_path):
    path=tmp_path/'motion_route_attempt_summary.json'
    write_route_summary(path,{'full_gate_attempted':1,'failures':['floor']})
    first=json.loads(next(tmp_path.glob('motion_route_attempt_summary_*.json')).read_text())
    write_route_summary(path,{'full_gate_attempted':2,'failures':['floor','IK']})
    assert json.loads(path.read_text())['full_gate_attempted']==2
    versions=[json.loads(p.read_text())for p in tmp_path.glob('motion_route_attempt_summary_*.json')]
    assert len(versions)==2 and first in versions

def test_context_refresh_preserves_parked_body_without_fake_support_admission():
    root=np.eye(4);root[1,3]=-1000.;state={'matrix':np.eye(4)};calls=[]
    entry={'object':SimpleNamespace(root_scene_node=SimpleNamespace(absolute_transformation=lambda:state['matrix'])),
           'phase2_layout_qualified':False}
    video=SimpleNamespace(qt=None,mn=None,_apply_entry=lambda *a:calls.append('standing'),
                          _apply_motion_entry=lambda *a:calls.append('moving'))
    def apply_root(obj,matrix,**kwargs):state['matrix']=matrix.copy()
    restore_actor_state(video,entry,{},0,False,root,apply_root)
    assert not calls and entry['phase2_layout_qualified'] is False
    assert 'resolved_root_m' not in entry
    assert np.array_equal(entry['applied_root_input'],root)

def test_qualified_static_and_moving_bodies_keep_normal_readback_paths():
    calls=[];video=SimpleNamespace(_apply_entry=lambda *a:calls.append('static'),
                                   _apply_motion_entry=lambda e,i:calls.append(('moving',i)))
    restore_actor_state(video,{'phase2_layout_qualified':True},{},17,False,np.eye(4),None)
    restore_actor_state(video,{'phase2_layout_qualified':True},{},23,True,np.eye(4),None)
    assert calls==['static',('moving',23)]
