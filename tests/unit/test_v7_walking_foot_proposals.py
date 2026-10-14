from types import SimpleNamespace
import numpy as np
import pytest
from avengine.assets.roster_motion.episode import clear_native_foot_sound_paths

pytestmark = pytest.mark.fast_unit


def test_scan_obstacle_rejects_the_low_shoe_emitter_but_inserted_bodies_do_not():
    rays=[]
    hit=SimpleNamespace(object_id=-1)
    def cast(ray,distance):
        rays.append((ray,distance))
        return SimpleNamespace(hits=[hit])
    video=SimpleNamespace(
        hs=SimpleNamespace(geo=SimpleNamespace(Ray=lambda a,b:(a,b))),
        mn=SimpleNamespace(Vector3=lambda x:np.asarray(x)),
        sim=SimpleNamespace(cast_ray=cast))
    vertices=np.array([[0.,.025,0.],[0.,.2,0.]])
    feet={'L':{'in_contact':True,'sole_local_m':[0.,0.,0.]}}
    args=(video,[0.,1.3,0.],vertices,feet,[2.,0.,0.],0.)
    assert not clear_native_foot_sound_paths(*args,set())
    hit.object_id=123
    assert clear_native_foot_sound_paths(*args,{123})
    assert np.allclose(rays[0][0][1],np.array([2.,-1.275,0.])/np.linalg.norm([2.,-1.275,0.]))
    assert rays[0][1] == pytest.approx(np.linalg.norm([2.,-1.275,0.])-.05)


def owner_quality_visual():
    from avengine.assets.roster_motion.quality import HUMAN_QUALITY
    feet={s:{'in_contact':True,'floor_delta_m':.012,'support_lock_error_m':.015}for s in ['L','R']}
    source=[{'moving':True,'feet':{s:dict(v)for s,v in feet.items()}}for _ in range(901)]
    actual=[{'root_max_error':0.,'native_emitter_readback_error_m':0.}for _ in range(901)]
    record={'status':'pass','source_frames':source,'native_30Hz_readbacks':actual,
        'maximum_navmesh_horizontal_error_m':0.,'minimum_certified_triangle_nonfloor_clearance_m':.001,
        'lineage':{'roster':'readonly'},'quality_policy':dict(HUMAN_QUALITY),
        'original_full_skin_floor_collision':{'status':'pass','worst_original_skin_floor_delta_m':0.}}
    return {'native_motion_admissions':{'voice1':record},'phase2_requested_motion':{'voice1':{'action':'walk'}}}


def test_owner_quality_boundaries_keep_complete_native_readback_checks():
    from avengine.timeline.v7_phase2_checks import motion_checks
    result=motion_checks(owner_quality_visual())
    assert result['status']=='pass' and result['thresholds_changed']


@pytest.mark.parametrize('field,value',[('floor_delta_m',.01201),('support_lock_error_m',.01501)])
def test_owner_quality_does_not_accept_errors_beyond_the_authorized_limits(field,value):
    from avengine.timeline.v7_phase2_checks import motion_checks
    visual=owner_quality_visual()
    visual['native_motion_admissions']['voice1']['source_frames'][450]['feet']['L'][field]=value
    assert motion_checks(visual)['status']=='fail'
