from types import SimpleNamespace
import numpy as np
import pytest
from avengine.assets.animal_motion.route_solver import find_animal_route, delivery_speed_reports

pytestmark=pytest.mark.fast_unit


def context_fixture(monkeypatch, tmp_path, *, dog_free=True):
    from avengine.rooms.walkable_map import WalkableMap, SPECIES_RADII_M
    from avengine.construction.joint import context as core
    from shapely.geometry import mapping,box
    resolution=.05
    origin=np.array([-8.,-12.])
    layers={kind:np.ones((300,320),bool)for kind in SPECIES_RADII_M}
    if not dog_free:layers['dog'][:]=False;layers['human'][:]=False
    metadata={'version':'walkable-map-v1','room_id':'fixture','species_radii_m':SPECIES_RADII_M,
              'footprint_policy':'independent-radius-coverage-v2','source_identity':[]}
    walkable=WalkableMap(origin,resolution,0.,layers,metadata,np.zeros((300,320)))
    walkable.path=str(tmp_path/'fixture_map.npz')
    room={'room_label':'fixture','floor_polygon':mapping(box(-8.,-12.,8.,3.)),
          'measured_floor_y_m':0.,'placement':{'camera_m':[0.,1.3,0.]}}
    pf=SimpleNamespace(snap_point=lambda point:np.asarray(point,float))
    def build(actual_room,**kwargs):
        return core.RoomContext(actual_room,walkable_map=walkable,pathfinder=pf,
            camera_limit=kwargs['camera_limit'],grid_step_m=kwargs['grid_step_m'],
            camera_heights_m=kwargs['camera_heights_m'],width=kwargs['width'],height=kwargs['height'])
    monkeypatch.setattr(core,'build_room_context',build)
    return room,pf


@pytest.mark.parametrize('species',['dog','cat','slide'])
@pytest.mark.parametrize('seed,branch',[(0,'nearer'),(1,'farther')])
def test_joint_route_constructs_requested_branch_with_full_camera_and_crop_certificates(monkeypatch,tmp_path,species,seed,branch):
    room,pf=context_fixture(monkeypatch,tmp_path)
    route=find_animal_route(room,pf,seed,species=species,utterance_s=2.3,
                            source_asset={'dimensions_m':[.24,.61,.7]})
    assert route['emission_slot']['branch']==branch
    assert np.sign(route['emission_slot']['radial_change_m'])==(-1 if branch=='nearer'else 1)
    assert route['construction']=='joint_core_radial_animal_corridor'
    assert route['joint_core_certificate']['authority']=='planning_geometry_only'
    assert all(check['status']=='pass'for check in route['joint_core_certificate']['checks'])
    assert route['animal_native_preflight_required']
    assert route['walkable_map_bypassed'] is False
    assert route['listener_m'][1]==pytest.approx(1.1)
    assert all(check['status']=='pass'for check in delivery_speed_reports(route,species))
    p=np.array(route['positions_xz_m']);camera=np.array(route['listener_m'])[[0,2]]
    assert np.linalg.norm(p-camera,axis=1).min()>=2.6-1e-7


def test_dog_request_does_not_borrow_cat_walkable_layer(monkeypatch,tmp_path):
    room,pf=context_fixture(monkeypatch,tmp_path,dog_free=False)
    with pytest.raises(ValueError,match='candidate domain exhausted'):
        find_animal_route(room,pf,0,species='dog',utterance_s=2.3)
