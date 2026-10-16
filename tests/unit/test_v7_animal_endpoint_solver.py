from itertools import islice
from types import SimpleNamespace
import time
import numpy as np
import pytest
from shapely.geometry import box,mapping
from avengine.rooms.walkable_map import WalkableMap,SPECIES_RADII_M
from avengine.construction.joint import CameraPose,BodySpec
from avengine.construction.joint.context import RoomContext
from avengine.assets.animal_motion.endpoint_solver import endpoint_pairs,endpoint_routes
pytestmark=pytest.mark.fast_unit


def room_context():
    layers={key:np.ones((200,200),bool) for key in SPECIES_RADII_M}
    metadata={'version':'walkable-map-v1','room_id':'endpoints','species_radii_m':SPECIES_RADII_M,'footprint_policy':'independent-radius-coverage-v2','source_identity':[]}
    wm=WalkableMap(np.array([-5.,-8. ]),.05,0.,layers,metadata,np.zeros((200,200)))
    room={'room_label':'endpoints','floor_polygon':mapping(box(-5,-8,5,2)),'measured_floor_y_m':0.,'placement':{'camera_m':[0.,1.1,1.]}}
    ctx=RoomContext(room,walkable_map=wm,pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p,float)),camera_limit=1,grid_step_m=.25,width=1280,height=720)
    ctx.cameras=(CameraPose((0.,1.1,1.),0.,width=1280,height=720),)
    return ctx


@pytest.mark.parametrize('branch',['nearer','farther'])
def test_complete_endpoint_domain_has_hundreds_of_answer_pairs_without_a_near_ring(branch):
    ctx=room_context();body=BodySpec.for_species('cat');d={}
    pairs=list(islice(endpoint_pairs(ctx,ctx.cameras[0],body,species='cat',branch=branch,diagnostics=d),512))
    assert len(pairs)==512 and d['endpoint_pairs_in_domain']>512
    assert any(max(np.linalg.norm(a-np.array(ctx.cameras[0].position_m)),np.linalg.norm(b-np.array(ctx.cameras[0].position_m)))>4.5 for a,b,_ in pairs)
    seen=set()
    for a,b,e in pairs:
        da=np.linalg.norm(a-np.array(ctx.cameras[0].position_m));db=np.linalg.norm(b-np.array(ctx.cameras[0].position_m))
        assert (db<da) == (branch=='nearer')
        assert abs(db-da)>=.5-1e-10
        key=tuple(a)+tuple(b);assert key not in seen;seen.add(key)
        assert e['start_role']==e['end_role']=='inside_full_body'


@pytest.mark.parametrize('species,speed',[('dog',.65),('cat',.5),('slide',.4)])
def test_same_camera_position_keeps_other_yaws_and_routes_through_core(monkeypatch,species,speed):
    from avengine.construction.joint import routes
    ctx=room_context();good=ctx.cameras[0];ctx.cameras=(CameraPose(good.position_m,180.,width=1280,height=720),good)
    calls=[];original=routes.route_between
    def spy(context,a,b,**kwargs):
        calls.append((a.copy(),b.copy(),kwargs))
        return original(context,a,b,**kwargs)
    monkeypatch.setattr(routes,'route_between',spy)
    d={};camera,actor,e=next(endpoint_routes(ctx,body=BodySpec.for_species(species),species=species,branch='nearer',entity_id='animal',speed_m_s=speed,start_s=.8,event_end_s=8.,diagnostics=d))
    assert camera==good and d['cameras_inspected']==2
    assert calls and calls[0][2]['species']==species
    assert d['routes_found']>=1 and d['constructed_routes']==1
    assert len(actor.positions_m)==901 and actor.metadata['walk_intervals_s'][0][1]<=8.+1e-8
    lo,hi=routes.SPEED_BANDS[species];assert lo<=actor.metadata['speed_m_s']<=hi
    assert e['solver']=='context.grid -> route_between -> construct_route'
    assert e['native_truth_verified'] is False


def test_budget_exhaustion_is_an_observation_not_a_continuous_infeasibility_verdict():
    ctx=room_context();d={}
    assert list(endpoint_routes(ctx,body=BodySpec.for_species('cat'),species='cat',branch='nearer',entity_id='animal',speed_m_s=.5,start_s=.8,event_end_s=8.,diagnostics=d,deadline=time.monotonic()-1))==[]
    assert d=={'compute_budget_exhausted':True}


def test_joint_camera_endpoint_solve_does_not_require_an_old_camera_seed():
    from avengine.assets.animal_motion.endpoint_solver import joint_camera_endpoint_routes
    ctx=room_context();ctx.cameras=();d={}
    camera,actor,e=next(joint_camera_endpoint_routes(ctx,body=BodySpec.for_species('cat'),species='cat',branch='nearer',entity_id='animal',speed_m_s=.5,start_s=.8,event_end_s=7.,diagnostics=d,deadline=time.monotonic()+30.))
    start=np.linalg.norm(actor.positions_m[0]-np.asarray(camera.position_m));end=np.linalg.norm(actor.positions_m[-1]-np.asarray(camera.position_m))
    assert start-end>=.5-1e-8
    assert e['solver']=='physical endpoints -> route_between -> construct_route -> camera_solve'
    assert e['camera_yaw_margin_deg']>=0
    assert d['physical_pairs_in_domain']>512 and d['camera_role_solves']>=1
    assert len(actor.positions_m)==901 and e['native_truth_verified'] is False
