"""Body scaling and bounded static-pose grounding contracts."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/assets'))
from v7_animal_variants_size import scale_choice
from v7_animal_variants_standing import tilt_matrix,size_metrics

def test_uniform_body_scale_fits_oversized_poodle():
 before={'fore_hind_paw_span_m':.59,'robust_top_height_m':.86};target={'fore_hind_paw_span_m':.40,'robust_top_height_m':.59}
 s,method=scale_choice(before,target,'dog')
 assert np.ptp(s)==0
 assert all(abs(before[k]*s[1]/target[k]-1)<=.15 for k in target)
 assert .67<s[0]<.70

def test_inconsistent_proportions_require_declared_axis_correction():
 before={'fore_hind_paw_span_m':.17,'robust_top_height_m':.45};target={'fore_hind_paw_span_m':.27,'robust_top_height_m':.40}
 s,method=scale_choice(before,target,'cat')
 assert s[2]>1.5 and s[1]<1.
 assert 'proportion correction' in method
 assert before['fore_hind_paw_span_m']*s[2]==pytest.approx(.27)
 assert before['robust_top_height_m']*s[1]==pytest.approx(.40)

def test_bird_yaw_does_not_request_length_stretch():
 before={'principal_body_length_m':.20,'body_height_without_perch_m':.16};target=before.copy()
 s,_=scale_choice(before,target,'bird')
 assert np.allclose(s,np.ones(3))

def test_bird_disproportion_cannot_silently_warp_plumage():
 with pytest.raises(ValueError):scale_choice({'principal_body_length_m':.10,'body_height_without_perch_m':.30},{'principal_body_length_m':.20,'body_height_without_perch_m':.15},'bird')

def test_rigid_tilt_preserves_lengths_and_upper_angle_bound():
 r=tilt_matrix(3,-3);x=np.array([.4,.7,-.2])
 assert np.allclose(r.T@r,np.eye(3),atol=1e-12)
 assert np.linalg.norm(r@x)==pytest.approx(np.linalg.norm(x))
 assert np.linalg.det(r)==pytest.approx(1.)

def test_paw_span_uses_both_paws_each_end():
 v=np.array([[x,y,z] for x in [-.1,.1] for y in np.linspace(0,.7,1000) for z in np.linspace(-.2,.2,4)])
 feet={str(i):np.array(p) for i,p in enumerate([[-.1,0,-.3],[.1,.002,-.2],[-.1,.001,.2],[.1,0,.3]])}
 m=size_metrics(v,feet,'dog')
 assert m['fore_hind_paw_span_m']==pytest.approx(.5)
 assert .69<m['robust_top_height_m']<=.7


def test_vertical_tail_does_not_inflate_head_height():
 from v7_animal_variants_size import head_height
 head=np.array([[x,y,z] for x in np.linspace(-.1,.1,5) for y in np.linspace(.25,.4,10) for z in np.linspace(-.35,-.25,5)])
 tail=np.array([[0,y,.4] for y in np.linspace(0,.8,100)])
 v=np.concatenate([head,tail]);feet={str(i):np.array(p) for i,p in enumerate([[-.1,0,-.2],[.1,0,-.2],[-.1,0,.2],[.1,0,.2]])}
 assert head_height(v,feet)==pytest.approx(.4)
 assert head_height(v,feet,(head.min(0),head.max(0)))==pytest.approx(.4)


def test_whole_frame_vertical_correction_keeps_swing_foot_height():
 from v7_animal_variants_walk import whole_frame_y_correction
 feet={"front":.09,"rear":.10,"swing":.18}
 dy=whole_frame_y_correction(feet,["front","rear"])
 assert dy==pytest.approx(-.095)
 assert feet["swing"]+dy>0.08
 assert max(abs(feet[k]+dy) for k in ["front","rear"])<=.012


def test_vertical_correction_cannot_hide_incompatible_contact_span():
 from v7_animal_variants_walk import whole_frame_y_correction
 feet={"front":.05,"rear":.13}
 dy=whole_frame_y_correction(feet,["front","rear"])
 assert min(x+dy for x in feet.values())>=-.012-1e-10
 assert max(abs(x+dy) for x in feet.values())>.012


def test_vertical_correction_never_drives_noncontact_paw_below_gate():
 from v7_animal_variants_walk import whole_frame_y_correction
 feet={"front":.12,"rear":.125,"noncontact":.05}
 dy=whole_frame_y_correction(feet,["front","rear"])
 assert min(x+dy for x in feet.values())>=-.012-1e-10
 assert max(abs(feet[k]+dy) for k in ["front","rear"])>.012
