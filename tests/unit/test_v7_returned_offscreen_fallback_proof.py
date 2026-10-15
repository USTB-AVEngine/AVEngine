import pytest
from avengine.capture.v7_offscreen_layout import coverage_errors
pytestmark=pytest.mark.fast_unit


def fixture():
    proof={'status':'pass','source_is_real_drawable':True,'all_vertices_outside_native_frustum_plane':True,
      'native_object_id':37,'native_visible_pixels':0,'QA_visibility_overridden':False}
    return [{'asset_id':'a','endpoint_id':'voice1'}],{'visible_attachment_checks':[{'asset_id':'a','intentional_offscreen_proof':proof}]},proof


def test_actual_static_native_fallback_survives_fresh_process_scene_boundary():
    assets,visual,proof=fixture();scene={}
    assert coverage_errors(scene,assets,visual,{'a':0})==[]
    assert scene=={}


def test_zero_pixels_without_actual_drawable_proof_still_fails():
    assets,visual,proof=fixture();proof['native_object_id']=None
    assert coverage_errors({},assets,visual,{'a':0})


def test_explicit_visible_target_and_unplanned_mover_are_never_silently_reclassified():
    assets,visual,proof=fixture()
    assert coverage_errors({'native_view_roles':{'a':{'view':'inside'}}},assets,visual,{'a':0})
    assert coverage_errors({'production_scenario':{'motion_requests':{'voice1':{'action':'walk'}}}},assets,visual,{'a':0})
