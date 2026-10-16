"""Contract regressions for color masking, support geometry and crossing requirements."""
import sys
from pathlib import Path
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/assets'))
from v7_animal_variants_color import rgb_to_lab,measure,DEFAULT
from v7_animal_variants_normalize import paw_supports,level_rotation,four_paw_layout
from v7_animal_variants_plan import build_plan

def test_lab_matches_standard_srgb_reference_values():
    actual=rgb_to_lab(np.array([[0,0,0],[255,255,255],[255,0,0]],dtype=np.uint8))
    np.testing.assert_allclose(actual,[[0,0,0],[100,0,0],[53.2408,80.0925,67.2032]],atol=.002)

def test_color_uses_actual_mask_and_excludes_small_features():
    image=np.full((40,40,3),255,np.uint8);body=np.zeros((40,40),bool);body[5:35,5:35]=True;image[5:9,5:35]=0;excluded=np.zeros_like(body);excluded[5:9,5:35]=True
    actual=measure(image,body,DEFAULT,excluded)
    assert actual['measured_label']=='white' and actual['status']=='pass'
    assert actual['excluded_eye_nose_beak_pixels']==120
    assert actual['dominant_fraction']==1
    assert measure(image,np.zeros_like(body))['status']=='fail'

def test_unconfident_multicolor_body_cannot_receive_a_label():
    image=np.zeros((30,30,3),np.uint8);image[10:20]=255;image[20:]=[0,180,0]
    actual=measure(image,np.ones((30,30),bool))
    assert actual['dominant_fraction']<DEFAULT['dominant_fraction_min']
    assert actual['status']=='fail'

def supports(foot_count=4):
    feet=[(-.14,0,-.32),(.14,0,-.32),(-.14,0,.32),(.14,0,.32)][:foot_count];parts=[]
    for center in feet:
        mesh=trimesh.creation.icosphere(subdivisions=3,radius=.05);parts.append(mesh.vertices+center)
    body=trimesh.creation.icosphere(subdivisions=2,radius=.20);parts.append(body.vertices+[0,.40,0]);return np.concatenate(parts)

def test_four_support_plane_recovers_rigid_tilt():
    rotation=Rotation.from_euler('zx',[5,-3],degrees=True).as_matrix();points,_=paw_supports(supports()@rotation.T);assert len(points)==4
    level,tilt=level_rotation(points)
    assert 4<tilt<7
    assert np.ptp((points@level.T)[:,1])<.002

def test_three_feet_are_not_invented_into_four():
    points,_=paw_supports(supports(3))
    assert len(points)!=4

def test_declared_plan_meets_all_crossing_rules():
    plan=build_plan()
    for category,minimum in plan['minimum_individuals'].items():
        animals=[x for x in plan['animals'] if x['category']==category];assert len(animals)>=minimum
        for color in plan['selected_label_colors'][category]:
            assert len({x['body_id'] for x in animals if x['intended_color_label']==color})>=plan['minimum_bodies_per_label'][category]
        for body in {x['body_id'] for x in animals if not x.get('cross_requirement_exempt_companion')}:
            assert len({x['intended_color_label'] for x in animals if x['body_id']==body and x['intended_color_label']})>=2
    birds=[x for x in plan['animals'] if x['category']=='bird'];assert sum(x['sound_appearance_family']=='canary' for x in birds)>=4;assert sum(x['sound_appearance_family']=='parrot' for x in birds)>=6


def test_four_front_toe_clusters_cannot_replace_two_pairs_of_paws():
    valid=np.array([[-.12,0,-.30],[.12,0,-.30],[-.12,0,.30],[.12,0,.30]])
    assert four_paw_layout(valid,.5)
    front_toes=np.array([[-.10,0,-.30],[-.04,0,-.30],[.04,0,-.30],[.10,0,-.30]])
    assert not four_paw_layout(front_toes,.5)
    assert not four_paw_layout(valid[:3],.5)


def test_extra_low_tail_does_not_invent_or_delete_a_paw():
    v=supports()
    tail=trimesh.creation.icosphere(subdivisions=3,radius=.025).vertices+[0,-.025,.5]
    points,details=paw_supports(np.concatenate([v,tail]))
    assert len(points)==4 and four_paw_layout(points,np.ptp(v[:,1]))
    assert all('additional_rear_centerline_support_ignored' in x for x in details)


def test_duplicate_individuals_cannot_fake_different_body_color_coverage():
    from v7_animal_variants_register import cross_check
    plan=build_plan()
    rows=[{**x,'label_color':x['intended_color_label']} for x in plan['animals']]
    assert cross_check(rows)[2]==[]
    duplicated=[{**x,'body_id':'standard_poodle'} if x['category']=='dog' and x['intended_color_label'] else x for x in rows]
    assert any('标签没有覆盖' in e for e in cross_check(duplicated)[2])


def test_color_distance_checks_worst_cross_scene_pair():
    from v7_animal_variants_register import pairwise
    def row(aid,color,mp3d,hm3d):
        return {'asset_id':aid,'category':'cat','label_color':color,'native_qa':{h:{'pooled_body_color':{'dominant_median_lab':[l,0,0]}} for h,l in [('mp3d',mp3d),('hm3d',hm3d)]}}
    animals=[row('black','black',10,10),row('gray','gray',31,23)]
    result,errors=pairwise(animals,15)
    assert errors and result['cat']['minimum_delta_e76']==13
    result,errors=pairwise([row('black','black',10,10),row('gray','gray',31,30)],15)
    assert not errors and result['cat']['status']=='pass'
