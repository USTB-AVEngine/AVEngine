from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import box,mapping
from avengine.rooms.walkable_map import WalkableMap,SPECIES_RADII_M
from avengine.construction.joint import CameraPose,BodySpec,ScenePlan
from avengine.construction.joint.context import RoomContext
from avengine.construction.joint.routes import static_actor
from avengine.assets.animal_motion.negative_witness import(
    construct_negative_companion,companion_motion_request,verify_negative_comparator)
pytestmark=pytest.mark.fast_unit


def fixture(tmp_path):
    layers={k:np.ones((300,320),bool)for k in SPECIES_RADII_M}
    m=WalkableMap(np.array([-8.,-12.]),.05,0.,layers,{'version':'walkable-map-v1',
        'room_id':'fixture','species_radii_m':SPECIES_RADII_M,'source_identity':[],
        'footprint_policy':'independent-radius-coverage-v2'},np.zeros((300,320)))
    m.path=str(tmp_path/'fixture.npz')
    room={'room_label':'fixture','floor_polygon':mapping(box(-8,-12,8,3)),
        'measured_floor_y_m':0.,'placement':{'camera_m':[0.,1.1,0.]}}
    ctx=RoomContext(room,walkable_map=m,pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p,float)),
        camera_limit=1,camera_heights_m=(1.1,),width=1280,height=720)
    cam=CameraPose((0.,1.1,0.),0.,width=1280,height=720)
    animal=static_actor('animal',[0.,0.,-3.],BodySpec.for_species('dog'))
    return ctx,cam,animal


def test_later_animal_still_branch_has_real_opposite_comparator(tmp_path):
    ctx,cam,animal=fixture(tmp_path)
    human=construct_negative_companion(ctx,cam,[5.,0.,0.],BodySpec.for_species('human'),animal,
        entity_id='companion',animal_call_s=21.3,animal_duration_s=2.2,admitted_asset={'kind':'human','walk_allowed':True})
    plan=ScenePlan(cam,{'animal':animal,'companion':human})
    checks=list(verify_negative_comparator(plan,'animal','companion',21.3,2.2))
    assert all(x.status=='pass'for x in checks),checks
    request=companion_motion_request(human,cam,ctx.walkable_map,27.,1.5)
    np.testing.assert_array_equal(request['motion_quartet_route']['positions_xz_m'],human.positions_m[:,[0,2]])
    assert request['motion_quartet_route']['joint_exact_route']
    assert request['speed_m_s']==.85
    np.testing.assert_array_equal(human.positions_m[:600],np.tile(human.positions_m[0],(600,1)))
    np.testing.assert_array_equal(human.positions_m[810:],np.tile(human.positions_m[-1],(91,1)))


def test_two_real_still_entities_cannot_certify_negative_questions(tmp_path):
    _,cam,animal=fixture(tmp_path)
    human=static_actor('companion',[5.,0.,0.],BodySpec.for_species('human'))
    checks=list(verify_negative_comparator(ScenePlan(cam,{'animal':animal,'companion':human}),
        'animal','companion',21.3,2.2))
    assert {x.name for x in checks if x.status=='fail'}=={'animal:static_negative:QA-06','animal:static_negative:QA-17'}


def test_unadmitted_static_person_is_never_animated_as_a_comparator(tmp_path):
    from avengine.construction.joint.routes import ConstructionError
    ctx,cam,animal=fixture(tmp_path)
    with pytest.raises(ConstructionError,match='requires_admitted_human_walk'):
        construct_negative_companion(ctx,cam,[5.,0.,0.],BodySpec.for_species('human'),animal,
            entity_id='companion',animal_call_s=21.3,animal_duration_s=2.2,
            admitted_asset={'kind':'human','walk_allowed':False})


def test_comparator_speech_rest_positions_need_clear_original_scan_rays(tmp_path):
    from avengine.construction.joint.routes import ConstructionError
    ctx,cam,animal=fixture(tmp_path)
    ctx.raycaster=SimpleNamespace(first_hit=lambda origin,rays:(np.zeros(len(rays)),None))
    with pytest.raises(ConstructionError,match='finite_route_domain_empty'):
        construct_negative_companion(ctx,cam,[5.,0.,0.],BodySpec.for_species('human'),animal,
            entity_id='companion',animal_call_s=21.3,animal_duration_s=2.2,
            admitted_asset={'kind':'human','walk_allowed':True})


def test_no_hit_scan_ray_is_a_clear_speech_rest_position(tmp_path):
    ctx,cam,animal=fixture(tmp_path)
    ctx.raycaster=SimpleNamespace(first_hit=lambda origin,rays:(np.full(len(rays),np.inf),None))
    human=construct_negative_companion(ctx,cam,[5.,0.,0.],BodySpec.for_species('human'),animal,
        entity_id='companion',animal_call_s=21.3,animal_duration_s=2.2,
        admitted_asset={'kind':'human','walk_allowed':True})
    assert human.entity_id=='companion'
