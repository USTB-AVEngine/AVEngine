"""Regression tests for mouth occlusion and the licensed pet-bird intake boundary."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/assets'))
from v7_roster_round4_manifest import mouth_visibility,pet_admitted
pytestmark=pytest.mark.fast_unit
def receipts():
 return ({'source_license':'CC-BY-4.0','movement':'static'},[{'house':'mp3d','status':'pass'},{'house':'hm3d','status':'pass'}],{'qa_status':'pass'},{'status':'pass'})
def test_uniform_occlusion_keeps_physical_voice_point_and_excludes_lip_visible_binding():
 r=mouth_visibility({'status':'qualified_uniform_occludes_lips','ue_status':'qualified_uniform_occludes_lips'})
 assert r['uniform_occludes_lips'] and not r['physical_anchor_relocated'] and r['voice_emission_allowed']
 assert not r['visible_lips_binding_allowed_habitat'] and not r['visible_lips_binding_allowed_ue']
def test_dark_ue_lips_do_not_disable_the_independently_verified_habitat_projection():
 r=mouth_visibility({'status':'pass_native_bone_projection','ue_status':'qualified_helmet_dark_lips'})
 assert r['visible_lips_binding_allowed_habitat'] and not r['visible_lips_binding_allowed_ue']
 assert not r['uniform_occludes_lips'] and r['helmet_or_lighting_limits_ue']
def test_missing_review_never_defaults_to_visible_lips():
 r=mouth_visibility({})
 assert not r['visible_lips_binding_allowed_habitat'] and not r['visible_lips_binding_allowed_ue']
def test_bird_admission_requires_distinct_native_houses():
 m,h,u,g=receipts();assert pet_admitted(m,h,u,g)
 h[1]['house']='mp3d';assert not pet_admitted(m,h,u,g)
def test_noncommercial_model_license_is_not_accepted_as_ccby():
 m,h,u,g=receipts();m['source_license']='CC-BY-NC-4.0';assert not pet_admitted(m,h,u,g)
def test_source_animations_do_not_imply_walking_intake():
 m,h,u,g=receipts();m['movement']='walk';m['source_animation_names']=['Idle Loop','Run Loop'];assert not pet_admitted(m,h,u,g)
def test_missing_native_ue_qa_blocks_asset():
 m,h,u,g=receipts();u={};assert not pet_admitted(m,h,u,g)
def test_source_scene_collision_failure_blocks_asset():
 m,h,u,g=receipts();g['status']='fail';assert not pet_admitted(m,h,u,g)
