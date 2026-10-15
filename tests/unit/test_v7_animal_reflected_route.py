"""Hermetic execution tests; native production jobs remain separate evidence."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.assets.animal_motion import production_episode as animals

pytestmark=pytest.mark.fast_unit


def turning_route():
    times=np.arange(901)/30.
    roots=np.column_stack((2.+.6*np.clip(times-1.,0.,2.),np.zeros(901),2.+.6*np.clip(times-3.,0.,2.)))
    yaws=np.where(times<=3.,0.,-np.pi/2)
    yaws[times>5.]=np.linspace(-np.pi/2,np.pi,np.count_nonzero(times>5.))
    return times,SimpleNamespace(positions=roots,yaws=yaws,receipt={'route_kind':'reflected native NavMesh walk','native_admissible_route_pool':2})


def test_reflected_route_keeps_turns_distance_and_stationary_heading():
    times,route=turning_route()
    roots,yaws,distance,velocity=animals.animal_route_samples(times,[0,0,0],0,[1,0,0],.6,route)
    np.testing.assert_allclose(roots,route.positions)
    assert distance[-1]==pytest.approx(2.4)
    assert np.all(velocity[times<=1.]<1e-8)
    assert np.all(velocity[times>5.]<1e-8)
    assert np.all(yaws[times>5.]==pytest.approx(-np.pi/2))
    assert roots[-1,0]>roots[0,0] and roots[-1,2]>roots[0,2]


def test_legacy_straight_route_keeps_its_clock():
    ts=np.arange(901)/30.
    roots,yaws,d,v=animals.animal_route_samples(ts,[1,2,3],.4,[0,0,1],.6)
    np.testing.assert_allclose(roots[:,2],3+ts*.6)
    np.testing.assert_allclose(yaws,.4);np.testing.assert_allclose(d,ts*.6);np.testing.assert_allclose(v,.6)


def admitted_fixture(tmp_path,monkeypatch,species='dog',mode='walk_animation',empty_pool=False,nav_drift=0.):
    from avengine.assets.animal_motion import admitted_reconstruction
    from avengine.assets.roster_motion import episode,reflect_navmesh
    from avengine.capture import v7_visible_mesh
    times,route=turning_route();source=[]
    if mode=='slide_no_animation':route.positions=route.positions[0]+(route.positions-route.positions[0])*(.4/.6)
    mesh=SimpleNamespace(vertices=np.array([[-.1,0,-.1],[.1,0,-.1],[0,.2,.1]]),faces=np.array([[0,1,2]]))
    for i,t in enumerate([0.,.5,1.]):
        root=np.array([t*.03,0,0]);feet={}
        for side,x,z in [('FL',.05,.05),('FR',.05,-.05),('BL',-.05,.05),('BR',-.05,-.05)]:
            sole=root+[x,0,z];feet[side]={'sole_world_m':sole.tolist(),'floor_delta_m':0.,'in_contact':True,'support_world_m':sole.tolist()}
        pose=tmp_path/('pose_static.glb'if mode=='slide_no_animation'else f'pose_{i}.glb');pose.write_bytes(b'hermetic fixture')
        source.append({'time_s':t,'root_world_m':root.tolist(),'feet':feet,'pose_glb':str(pose),
            'mouth_m':[.1,.1,0.],'head_m':[.1,.1,0.],'tail_m':[-.1,.1,0.],
            'head_world_m':(root+[.1,.1,0]).tolist(),'tail_world_m':(root+[-.1,.1,0]).tolist()})
    native=tmp_path/'native.json';native.write_text('{}')
    master=tmp_path/'master.glb';master.write_bytes(b'hermetic fixture')
    admission=tmp_path/'admission.json';admission.write_text(json.dumps({'assets':[{'asset_id':'animal','motion_mode':mode,'selected':{'status':'pass','native_motion':str(native)}}]}))
    monkeypatch.setattr(admitted_reconstruction,'reconstruct_frames',lambda *a:(source,{'status':'test_fixture'}))
    monkeypatch.setattr(v7_visible_mesh,'indexed_static_mesh',lambda p:mesh)
    def provider(*args):
        provider.calls+=1
        if not empty_pool:yield route
    provider.calls=0
    monkeypatch.setattr(reflect_navmesh,'reflected_route_proposals',provider)
    def no_legacy(*args):raise AssertionError('reflection template must never enter legacy straight search')
    monkeypatch.setattr(animals,'native_walkable_starts',no_legacy)
    class Floor:
        def __init__(self,*a):self.reference=0.
        def __call__(self,p):return 0.
        def many(self,p):return np.zeros(len(p))
    checks=[]
    class Clearance:
        def __init__(self,*a):pass
        def check(self,v,f,root,yaw):
            checks.append((np.array(root),yaw));return .1,v@animals.yaw_matrix(yaw).T+root
        def floor_certificate(self):return {'status':'test_fixture'}
    monkeypatch.setattr(episode,'PhysicalFloor',Floor);monkeypatch.setattr(episode,'TriangleClearance',Clearance)
    matrix=np.eye(4);matrix[:3,3]=route.positions[0]
    entry={'declaration':{'asset_id':'animal','kind':'animal','species_id':species},'applied_root_input':matrix,
        'resolved_yaw_rad':0.,'object':SimpleNamespace(object_id=1)}
    video=SimpleNamespace(actors=[entry],pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)+[nav_drift,0,0]))
    scene={'production_scenario':{'motion_floor_polygon':{'type':'Polygon','coordinates':[[[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]]}},'native_placement_candidates':[]}
    request={'speed_m_s':.4 if mode=='slide_no_animation' else .6,
        'animal_motion_admission':{'admission_path':str(admission),'native_motion':str(native),'source_master':str(master),'mode':mode},
        # Deliberately omit motion_execution_revision: the template owns routing.
        'motion_quartet_route':{'reflect_walk':True,'positions_xz_m':route.positions[:,[0,2]].tolist(),'travel_yaws_rad':route.yaws.tolist()}}
    return video,entry,scene,{'room':{'measured_floor_y_m':0.}},request,route,provider,checks


@pytest.mark.parametrize('species,mode',[('dog','walk_animation'),('cat','walk_animation'),('dog','slide_no_animation')])
def test_native_animal_uses_shared_reflection_pool_and_freezes_paused_gait(tmp_path,monkeypatch,species,mode):
    video,entry,scene,house,request,route,provider,checks=admitted_fixture(tmp_path,monkeypatch,species,mode)
    evidence=animals.compile_admitted_animal(video,entry,scene,house,{},tmp_path/'output',request)
    assert provider.calls==1 and len(checks)==901
    np.testing.assert_allclose(np.array(evidence['world_roots_m'])[:,[0,2]],route.positions[:,[0,2]])
    assert evidence['scene_motion_execution']['route_kind']=='reflected native NavMesh walk'
    phase=np.array([f['source_phase_s']for f in evidence['source_frames']])
    assert np.ptp(phase[:31])==0. and np.ptp(phase[151:])==0.
    assert all(not f['moving']for f in evidence['source_frames'][:31]+evidence['source_frames'][151:])
    assert evidence['animation_distance_clock']['travelled_distance_m']==pytest.approx(np.linalg.norm(np.diff(route.positions[:,[0,2]],axis=0),axis=1).sum())
    assert evidence['actual_speed_m_s']==pytest.approx(.4 if mode=='slide_no_animation'else .6)
    assert evidence['truth_and_answer_gates_changed'] is False


def test_empty_reflection_pool_fails_without_a_straight_or_static_fallback(tmp_path,monkeypatch):
    video,entry,scene,house,request,*_=admitted_fixture(tmp_path,monkeypatch,empty_pool=True)
    with pytest.raises(ValueError,match='no route passing retained answer gates'):
        animals.compile_admitted_animal(video,entry,scene,house,{},tmp_path/'output',request)
    assert not(tmp_path/'output/motion_geometry.json').exists()


def test_full_native_two_mm_gate_remains_independent_of_pool_admission(tmp_path,monkeypatch):
    video,entry,scene,house,request,*_=admitted_fixture(tmp_path,monkeypatch,nav_drift=.0021)
    with pytest.raises(ValueError,match='no route passing retained answer gates'):
        animals.compile_admitted_animal(video,entry,scene,house,{},tmp_path/'output',request)
    rejection=json.loads((tmp_path/'output/motion_rejected.json').read_text())
    assert 'unchanged 2mm NavMesh drift'in rejection['failed_candidates'][0]['reason']


def test_height_failure_refines_rigid_seed_through_the_same_native_prefilter(monkeypatch):
    from avengine.assets.roster_motion import reflect_navmesh
    _,base=turning_route();calls=[]
    def provider(video,entry,scene,forward,request):
        calls.append(request)
        xz=np.array(request['motion_quartet_route']['positions_xz_m'])
        yield SimpleNamespace(positions=np.column_stack((xz[:,0],np.zeros(len(xz)),xz[:,1])),yaws=base.yaws.copy(),receipt={'native_admissible_route_pool':1})
    monkeypatch.setattr(reflect_navmesh,'reflected_route_proposals',provider)
    original=base.positions[:,[0,2]].copy();request={'motion_quartet_route':{'positions_xz_m':original.tolist(),'travel_yaws_rad':base.yaws.tolist()}}
    proposals=animals.animal_reflected_route_proposals(None,None,{},np.array([1.,0.,0.]),request)
    first=next(proposals);first.report_failure('no constant rigid animal route height satisfies original skin and paw gates')
    second=next(proposals)
    assert len(calls)==2 and second.receipt['animal_rigid_fitting_round']==1
    np.testing.assert_allclose(second.positions[:,[0,2]],original+[.025,0.])
    np.testing.assert_array_equal(np.array(request['motion_quartet_route']['positions_xz_m']),original)
    assert second.receipt['original_physical_gates_unchanged']is True
