"""Paw cosmetics cannot hide bad native truth, or invalidate good native truth."""
from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from avengine.assets.animal_motion.production_grounding import POLICY, StationaryPawGrounder
from avengine.timeline.v7_phase2_checks import motion_checks

pytestmark = pytest.mark.fast_unit


def animal_visual():
    foot = {'in_contact': True, 'floor_delta_m': 0., 'support_lock_error_m': 0.}
    record = {'status': 'pass', 'species_id': 'dog', 'animal_visual_fallback_policy': dict(POLICY),
              'source_frames': [{'moving': i < 450, 'feet': {s: dict(foot) for s in ('FL','FR','HL','HR')}} for i in range(901)],
              'native_30Hz_readbacks': [{'body_to_world_m': np.eye(4).tolist(), 'native_emitter_readback_error_m': 0.} for _ in range(901)],
              'maximum_navmesh_horizontal_error_m': 0., 'minimum_certified_triangle_nonfloor_clearance_m': .1,
              'original_full_skin_floor_collision': {'status': 'pass', 'worst_original_skin_floor_delta_m': 0.},
              'lineage': {'roster': 'registered-original'}}
    return {'native_motion_admissions': {'animal': record}}


def test_unplanted_stopped_paw_is_reported_with_unchanged_measured_contact():
    visual = animal_visual(); frame = visual['native_motion_admissions']['animal']['source_frames'][600]
    frame['feet']['FL'].update(in_contact=False, floor_delta_m=.019, support_lock_error_m=.018)
    result = motion_checks(visual)
    assert result['status'] == 'pass'
    assert any('stationary quadruped' in w for w in result['appearance_warnings'])
    row = next(f for f in result['foot_rows'] if f['frame_index']==600 and f['foot']=='FL')
    assert row['passed'] is False and row['delta_m'] == .019 and row['in_contact'] is False


@pytest.mark.parametrize('defect', ['nav', 'emitter', 'missing_frame', 'skin', 'collision'])
def test_appearance_policy_cannot_override_answer_or_body_geometry_defects(defect):
    visual=animal_visual(); r=visual['native_motion_admissions']['animal']
    if defect=='nav': r['maximum_navmesh_horizontal_error_m']=.0021
    if defect=='emitter': r['native_30Hz_readbacks'][500]['native_emitter_readback_error_m']=.0002
    if defect=='missing_frame': r['native_30Hz_readbacks'].pop()
    if defect=='skin': r['original_full_skin_floor_collision']['worst_original_skin_floor_delta_m']=-.002
    if defect=='collision': r['minimum_certified_triangle_nonfloor_clearance_m']=0.
    assert motion_checks(visual)['status']=='fail'


def test_human_contact_policy_is_not_changed_by_an_animal_policy_field():
    visual=animal_visual();r=visual['native_motion_admissions']['animal'];r['species_id']='human'
    r['source_frames'][600]['feet']['FL']['floor_delta_m']=.03
    assert motion_checks(visual)['status']=='fail'


def test_absent_articulated_recipe_is_explicit_and_does_not_modify_the_input(tmp_path):
    frame={'feet':{'FL':{'in_contact':False}}};before=deepcopy(frame)
    grounder=StationaryPawGrounder({'motion_mode':'walk_animation'},'cat',tmp_path/'derived')
    with pytest.raises(ValueError,match='no reconstructible articulated IK recipe'):
        grounder.ground(frame,'original.glb',np.zeros(3),0.,None)
    assert frame==before and not (tmp_path/'derived').exists()


def test_native_planning_forwards_the_actual_animal_authority_contract(tmp_path,monkeypatch):
    from avengine.timeline import v7_motion_quartet_planning as planning
    from avengine.timeline import v7_motion_quartet_template as templates
    from avengine.acoustics import runtime
    contract={'animal_motion':{'registered_dog':{'mode':'walk_animation'}}}
    path=tmp_path/'contract.json';path.write_text(json.dumps(contract));seen={}
    monkeypatch.setattr(runtime,'load_habitat_runtime',lambda **kw:(SimpleNamespace(PathFinder=lambda:SimpleNamespace(load_nav_mesh=lambda p:True)),None))
    monkeypatch.setattr(templates,'motion_quartet_template',lambda *a,**kw:seen.update(kw) or {'planned':True})
    monkeypatch.setattr(planning,'plan_room_motion_quartet',lambda *a,constructor,**kw:constructor(*[{}]*6,attempt=2))
    row=planning.plan_native_room_motion_quartet({'room_id':'room'},{},{'checked_rooms':[{'room_label':'room'}],'input':{'navmesh_source':'registered.navmesh'}},[],{},[],{'runtime':{},'production_input_contract':str(path)})
    assert row=={'planned':True} and seen['contract']==contract


def test_animal_static_placement_uses_shared_offscreen_fallback_without_losing_required_identity():
    from avengine.capture.v7_offscreen_layout import placement_construction_fallback
    scene={'production_scenario':{'motion_requests':{}}}
    animal={'asset_id':'cat','endpoint_id':'voice1','kind':'animal'}
    assert placement_construction_fallback(scene,animal,'floor',None,False)=='offscreen'
    assert scene['native_view_roles']['cat']['real_registered_drawable_required']
    assert placement_construction_fallback(scene,animal,'floor',None,False) is None
    visible={'native_view_roles':{'cat':{'view':'inside'}},'production_scenario':{'motion_requests':{}}}
    assert placement_construction_fallback(visible,animal,'floor',None,False) is None


def test_animal_floor_fallback_remeasures_every_ray_and_keeps_contacts_honest(monkeypatch):
    from avengine.assets.animal_motion.production_grounding import construct_animal_floor_support, attachment_has_hard_error
    from avengine.capture import v7_visible_attachment as attachment
    meta={'attachment_surface':'floor','_attachment_template':(np.zeros(3),np.array([[0.,0.,0.],[1.,.03,0.]]),'original mesh')}
    monkeypatch.setattr(attachment,'ray',lambda sim,r,p,d,length:np.array([p[0],0.,p[2]]))
    support={'point':[0.,0.,0.],'normal':[0.,1.,0.],'surface':'floor'}
    result=construct_animal_floor_support(None,None,meta,support)
    assert result['contact_deltas_m']==pytest.approx([.0004,.0304])
    assert len(result['support_raycast_points_m'])==2 and result['appearance_warnings']
    assert support['point']==[0.,0.,0.]
    admitted={**result,'attachment_surface':'floor','support_type':'floor','skin_pass':True,'attachment_worst_error_m':.0304}
    assert not attachment_has_hard_error(admitted)
    admitted['skin_pass']=False
    assert attachment_has_hard_error(admitted)
    monkeypatch.setattr(attachment,'ray',lambda *a:None)
    assert construct_animal_floor_support(None,None,meta,support) is None


def test_reflection_m1_requires_its_advertised_original_families_without_forcing_radial_ones():
    from avengine.timeline.v7_motion_quartet_planning import validate_template_request
    catalog=[{'asset_id':'cat','kind':'animal','species_id':'cat'}]
    row={'motion_quartet_construction':{'route':{'emission_slot':{'qa06':True,'qa15':False,'qa16':False,'qa17':True}}},
         'source_classes':{'cat':['cat']},'motion_requests':{'cat':{'speed_m_s':.5}},
         'motion_during_condition_requests':[{'qa_id':qa,'target_asset_id':'cat','minimum_motion_s':1.5} for qa in ('QA-06','QA-17')]}
    validate_template_request(row,catalog)
    row['motion_during_condition_requests'].pop()
    with pytest.raises(ValueError,match='declared original conditions'):validate_template_request(row,catalog)


def test_positive_gap_below_two_cm_is_visual_only_but_intersection_is_still_hard():
    visual=animal_visual();record=visual['native_motion_admissions']['animal']
    record['minimum_certified_triangle_nonfloor_clearance_m']=.005
    result=motion_checks(visual)
    assert result['status']=='pass' and any('2cm' in w for w in result['appearance_warnings'])
    record['minimum_certified_triangle_nonfloor_clearance_m']=0.
    assert motion_checks(visual)['status']=='fail'


def test_real_moving_animal_can_start_outside_view_without_fabricating_visibility():
    from avengine.capture.v7_offscreen_layout import placement_construction_fallback,proposal_view_ok,native_pixels_ok,coverage_errors
    scene={'production_scenario':{'motion_requests':{'voice1':{}}}}
    animal={'asset_id':'cat','endpoint_id':'voice1','kind':'animal'}
    assert placement_construction_fallback(scene,animal,'floor',None,False)=='moving_view_construction'
    assert proposal_view_ok(scene,'cat',[0,0,2],[0,0,-1],False)
    entry={'declaration':animal,'object':SimpleNamespace(object_id=42)}
    assert native_pixels_ok(scene,[entry],{'cat':0})
    assert native_pixels_ok(scene,[entry],{'cat':120})
    visual={'visible_attachment_checks':[{'asset_id':'cat','moving_drawable_view_proof':{'native_object_id':42,'actual_per_frame_visibility_required':True}}]}
    assert coverage_errors(scene,[animal],visual,{'cat':0})==[]
    assert coverage_errors(scene,[animal],{}, {'cat':0})
    assert placement_construction_fallback(scene,animal,'floor',None,False) is None
    explicit={'production_scenario':{'motion_requests':{'voice1':{}}},'native_view_roles':{'cat':{'view':'inside'}}}
    assert placement_construction_fallback(explicit,animal,'floor',None,False) is None


def test_retry_receipts_are_distinct_and_preserve_the_final_real_failure(tmp_path):
    from avengine.capture.v7_offscreen_layout import record_placement_rejection
    record={'failed_asset_id':'cat','rejection_counts':{'outside_observed_camera_path':1280}}
    first=record_placement_rejection(tmp_path,'cat',record,'moving_view_construction')
    second=record_placement_rejection(tmp_path,'person',record,'camera_1_2m')
    third=record_placement_rejection(tmp_path,'person',record,'offscreen')
    assert not(tmp_path/'attachment_rejection.json').exists()
    final=record_placement_rejection(tmp_path,'person',record)
    assert len({first,second,third,final})==4
    assert json.loads(first.read_text())['status']=='retry_with_construction_fallback'
    assert json.loads(final.read_text())['status']=='fail'
    assert json.loads(first.read_text())['rejection_counts']==record['rejection_counts']


def test_derived_stationary_pose_resolves_the_actual_exported_texture_uri(tmp_path):
    from avengine.assets.animal_motion.production_grounding import stationary_pose_path
    from avengine.assets.roster_motion.episode import externalize_unchanged_images
    root=tmp_path/'stationary_ik';target=stationary_pose_path(root,0)
    content=b'hermetic unchanged embedded texture'
    source={'images':[{'bufferView':0,'mimeType':'image/png'}],
            'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':len(content)}]}
    exported=externalize_unchanged_images(source,content,root/'textures')
    texture=(target.parent/exported['images'][0]['uri']).resolve()
    assert texture.is_relative_to(root.resolve()) and texture.read_bytes()==content
    assert target.parent.name=='frames'
