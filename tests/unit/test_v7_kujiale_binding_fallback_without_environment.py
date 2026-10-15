from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.capture.v7_registered_ue import NativeRegisteredUEVideo
from avengine.capture.kujiale_spear_runtime import to_ue
pytestmark=pytest.mark.fast_unit


class SamplingReached(Exception):
    pass


def ordinary_static_video(monkeypatch):
    from tools.rooms import check_kujiale_native_capsule_contact as capsule
    from tools.rooms import check_v7_kujiale_native_assets as fields
    monkeypatch.setattr(capsule,'query_capsule',lambda *a,**k:{'status':'pass'})
    monkeypatch.setattr(fields,'vector_field',lambda bounds,key: bounds[key])
    video=NativeRegisteredUEVideo.__new__(NativeRegisteredUEVideo)
    video.assets=[{'endpoint_id':f'voice{i}','asset_id':f'person{i}'}for i in range(1,4)]
    video.declarations=[{'actor_id':f'source{i}'}for i in range(1,4)]
    positions=[[-1.,1.3,-3.],[1.,1.3,-3.],[0.,1.3,-4.5]]
    video.scene={'listener_m':[0.,1.5,0.],'camera_yaw_knots':[{'yaw_deg':0.}],
        'production_scenario':{'camera_policy':'binding_static','recipe_revision':'kujiale-fixed-natural-construction-v3-complete-fixed-view'},
        'native_placement_candidates':positions,'sources':{a['endpoint_id']:{'trajectory':[{'position_m':p}]}for a,p in zip(video.assets,positions)}}
    video.current_native_environment=False
    video.construct_registered_dialogue_camera=False
    video.shared_native_standing=None
    video.placement_overrides={};video.rejected_contact_positions={};video.vertical_root_offsets={}
    video.instance=SimpleNamespace(begin_frame=nullcontext,end_frame=nullcontext)
    video.native_floor_surface=SimpleNamespace(mesh=object())
    video._navigation_foot=lambda p:(p,0.)
    video._depth_contact_probe=lambda actor,lower=False:{'status':'fail'if lower else'pass','gap_m':-.1 if lower else 0.}
    video.runtimes={}
    for declaration,point in zip(video.declarations,positions):
        center=np.array([point[0],.75,point[2]])
        bounds={'Origin':to_ue(center),'BoxExtent':to_ue([.2,.75,.2])}
        actor=SimpleNamespace(GetActorBounds=lambda _bounds=bounds,**kw:_bounds)
        video.runtimes[declaration['actor_id']]={'visual_actor':actor}
    def apply(index):
        result=[]
        for asset,point in zip(video.assets,positions):
            p=video.placement_overrides.get(asset['endpoint_id'],point)
            result.append({**asset,'position_m':[p[0],0.,p[2]],'emitter_position_m':[p[0],1.3,p[2]]})
        return result
    video._apply=apply
    return video


@pytest.mark.parametrize('sampling',['pair','third','mouth'])
def test_original_binding_sampling_runs_without_any_animal_or_environment(monkeypatch,tmp_path,sampling):
    from avengine.capture import kujiale_pilot_environment as geometry
    video=ordinary_static_video(monkeypatch)
    name={'pair':'binding_pair_fits_view','third':'comparison_person_candidates','mouth':'binding_mouth_distance_contrast'}[sampling]
    def reached(*args,**kwargs):
        raise SamplingReached(sampling)
    monkeypatch.setattr(geometry,name,reached)
    with pytest.raises(SamplingReached,match=sampling):
        video.qualify_standing(tmp_path/'standing')
