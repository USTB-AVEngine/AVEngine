from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.signal import fftconvolve

from avengine.timeline.continuous_scene import (
    position_at, plan_rir_keyframes, partitioned_convolve, interpolated_filter,
    raw_world_to_camera_sn3d, published_bearing, activity_segments, window_segments,
    generate_v02_questions, window_coverage,
)
from avengine.timeline.continuous_pilot import PositionRIRCache
from avengine.timeline.audio import time_varying_convolve

pytestmark = pytest.mark.fast_unit


def track(point):
    return [{"time_s": 0., "position_m": point}, {"time_s": 60., "position_m": point}]


def scene(events):
    return {"events": events,"listener_m":[0.,0.,0.],"camera_forward":[0.,0.,-1.],
            "sources":{"v1":{"trajectory":track([0.,0.,-3.])},
                       "v2":{"trajectory":track([-1.,0.,-3.])}}}


def event(start,end,source=1,cls=0,endpoint="v1"):
    return {"start_tick":start,"end_tick":end,"source_id":source,"class_id":cls,"endpoint_id":endpoint}


def test_streaming_raised_cosine_matches_existing_v6_full_tail():
    rng=np.random.default_rng(23);dry=rng.normal(size=101)
    rirs=rng.normal(size=(3,4,17));keys=[0,37,100]
    old=time_varying_convolve(dry,rirs,keys,output_sample_count=len(dry))
    new=partitioned_convolve(dry,list(rirs),keys,fade="raised_cosine")
    np.testing.assert_allclose(new,old.full_tail,atol=2e-13,rtol=1e-12)


@pytest.mark.parametrize("fade",["linear","raised_cosine"])
def test_partition_preserves_constant_filter_and_tail(fade):
    rng=np.random.default_rng(5);dry=rng.normal(size=41);ir=rng.normal(size=(4,11))
    new=partitioned_convolve(dry,[ir,ir,ir],[0,13,40],fade=fade)
    expected=np.stack([fftconvolve(dry,h) for h in ir])
    np.testing.assert_allclose(new,expected,atol=1e-13)
    assert new.shape==(4,51)


def test_source_time_tail_survives_motion_stop_and_window_boundary():
    dry=np.zeros(15);dry[3]=1
    old=np.zeros((4,9));old[:,7]=2
    new=np.zeros((4,9));new[:,7]=9
    result=partitioned_convolve(dry,[old,new],[0,8])
    assert np.allclose(result[:,10],(1-3/8)*2+(3/8)*9)
    assert np.any(result[:,8:13])  # emission before this window is retained
    assert not np.any(result[:,:7])


def test_linear_filter_interpolation_preserves_delays_without_time_stretch():
    a=np.zeros((4,6));b=np.zeros((4,9));a[:,2]=1;b[:,4]=2
    out=interpolated_filter([a,b],[0,10],5)
    assert out.shape==(4,9)
    np.testing.assert_allclose(out[:,2],.5);np.testing.assert_allclose(out[:,4],1)


def test_static_source_does_not_gain_motion_keyframes():
    keys=plan_rir_keyframes(track([1.,1.,-2.]),30.,np.zeros(3))
    assert len(keys)==2
    assert keys[0]["sample"]==0 and keys[-1]["sample"]==479999
    assert keys[0]["position_m"]==keys[-1]["position_m"]


def test_earlier_time_or_distance_bound_and_motion_corners():
    tr=[{"time_s":0.,"position_m":[0.,0.,-2.]},{"time_s":1.,"position_m":[2.,0.,-2.]},
        {"time_s":3.,"position_m":[2.,0.,-2.]}]
    keys=plan_rir_keyframes(tr,3.,np.zeros(3),max_seconds=.25,max_distance_m=.3)
    moving=[k for k in keys if k['time_s']<=1.]
    assert max(np.diff([k['time_s'] for k in moving]))<=.25
    assert max(np.linalg.norm(np.diff([k['position_m'] for k in moving],axis=0),axis=1))<=.3001
    assert any(k['time_s']==1. for k in keys)


def test_near_field_angular_subdivision():
    tr=[{"time_s":0.,"position_m":[-.5,0.,-.2]},{"time_s":1.,"position_m":[.5,0.,-.2]}]
    keys=plan_rir_keyframes(tr,1.,np.zeros(3),max_seconds=10,max_distance_m=10,max_angle_deg=4.)
    assert len(keys)>20


def test_raw_n3d_to_camera_sn3d_left_positive_once():
    # Front=-Z, right=+X: a world-left source points towards -X.
    raw=np.zeros((4,3));raw[0]=1;raw[3]=-np.sqrt(3)
    camera=raw_world_to_camera_sn3d(raw,[0.,0.,-1.])
    np.testing.assert_allclose(camera[0],1);np.testing.assert_allclose(camera[1],1)
    np.testing.assert_allclose(camera[2:],0)
    assert published_bearing([-1.,0.,0.],np.zeros(3),[0.,0.,-1.])==90.


def test_full_scene_contiguous_frames_merge_but_one_missing_frame_splits():
    events=[event(0,3),event(3,6),event(7,9)]
    segments=activity_segments(events)
    assert [(e['start_tick'],e['end_tick']) for e in segments]==[(0,6),(7,9)]


def test_half_open_crop_and_carry_in_count_once():
    segments=activity_segments([event(0,40),event(100,110),event(120,130)])
    clipped=window_segments(segments,2.)
    assert len(clipped)==2
    assert clipped[0]['carry_in'] and clipped[0]['anchor_tick']==0
    assert clipped[1]['anchor_tick']==80


def test_tied_anchor_targets_excluded_count_still_retained():
    data=scene([event(0,30),event(0,30,2,1,'v2'),event(40,60)])
    pub,answers,audit=generate_v02_questions(data,0.,'w',{})
    assert {q['qa_id'] for q in pub}>={'QA-23','STARSS-SPEAKERS'}
    assert not any('target1' in q['question_id'] or 'target2' in q['question_id'] for q in pub)
    assert sum(e.get('reason')=='tied_anchor_or_unreliable_source' for e in audit['exclusions'])==2
    assert next(a for a in answers if '__QA-23__count__open' in a['question_id'])['answer']=='3'


def test_more_than_twenty_events_kept_and_open_only():
    data=scene([event(i*3,i*3+2) for i in range(28)])
    pub,answers,audit=generate_v02_questions(data,0.,'many',{})
    counts=[q for q in pub if q['qa_id']=='QA-23']
    assert len(audit['segments'])==28 and len(counts)==1 and counts[0]['form']=='open'
    assert counts[0]['answer_domain']=='nonnegative_integer'
    assert next(a for a in answers if a['question_id']==counts[0]['question_id'])['answer']=='28'


def test_non_actor_gates_count_and_music_keeps_full_v6_domain():
    data=scene([event(0,80,1,8)])
    pub,_,audit=generate_v02_questions(data,0.,'music',{})
    assert not any(q['qa_id']=='QA-23' for q in pub)
    assert all(len(q['answer_domain'])==34 for q in pub if q['qa_id']=='QA-21')
    assert any(e.get('reason')=='non_actor_or_unreliable_source_identity' for e in audit['exclusions'])


def test_same_actor_repeated_or_cross_speech_classes_counted_once():
    data=scene([event(0,15),event(20,35,1,1),event(40,60,2,1,'v2')])
    pub,answers,_=generate_v02_questions(data,0.,'same',{})
    answer=next(a for a in answers if '__STARSS-SPEAKERS__count__open' in a['question_id'])
    assert answer['answer']=='2'
    assert not any('source_id' in q or 'actor_id' in q or 'truth' in q or 'answer' in q or 'teacher_answer' in q for q in pub)


def test_speech_gender_conflict_suppresses_speaker_question():
    data=scene([event(0,40),event(10,30,1,1)])
    pub,_,audit=generate_v02_questions(data,0.,'conflict',{})
    assert not any(q['qa_id']=='STARSS-SPEAKERS' for q in pub)
    assert any(e.get('reason')=='speech_identity_conflict' for e in audit['exclusions'])


def test_qa25_next_integer_must_still_be_active():
    data=scene([event(4,10),event(14,31)])
    pub,_,audit=generate_v02_questions(data,0.,'integer',{})
    assert not any(q['qa_id']=='QA-25' and 'target1' in q['question_id'] for q in pub)
    assert any(q['qa_id']=='QA-25' and '第2秒' in q['question_zh'] for q in pub)


def test_direction_sector_boundary_dead_zone():
    data=scene([event(0,70)])
    data['sources']['v1']['trajectory']=track([-np.sin(np.radians(22.5)),0.,-np.cos(np.radians(22.5))])
    pub,_,audit=generate_v02_questions(data,0.,'boundary',{})
    assert not any(q['qa_id']=='QA-04' for q in pub)
    assert any(e.get('reason')=='within5deg_sector_boundary' for e in audit['exclusions'])


def test_duplicate_same_class_source_rejected_before_public_qa():
    with pytest.raises(ValueError,match='duplicate'):
        activity_segments([event(0,20),event(10,30)])


class FakeRenderer:
    def __init__(self):self.calls=0
    def render(self,positions):
        self.calls+=1
        return SimpleNamespace(samples=[np.ones((4,13),dtype=np.float32)],wall_seconds=.2,process_cpu_seconds=.2)


def test_cache_restart_reuses_exact_pose_and_refuses_changed_condition(tmp_path):
    renderer=FakeRenderer();identity={'material':'one','listener':[0,0,0]}
    cache=PositionRIRCache(tmp_path,renderer,identity);cache.get([1.,2.,3.])
    resumed=PositionRIRCache(tmp_path,renderer,identity);resumed.get([1.,2.,3.])
    assert renderer.calls==1
    with pytest.raises(ValueError,match='configuration changed'):
        PositionRIRCache(tmp_path,renderer,{**identity,'material':'two'})
    resumed.get([1.000001,2.,3.]);assert renderer.calls==2


def test_cache_recovers_unfinished_record_without_overwriting_outputs(tmp_path):
    renderer=FakeRenderer();cache=PositionRIRCache(tmp_path,renderer,{'room':'one'})
    key=json.dumps([1.,2.,3.],separators=(',',':'))
    cache.db.execute('INSERT INTO rirs(pose) VALUES (?)',(key,));cache.db.commit()
    a=cache.get([1.,2.,3.]);assert a.shape==(4,13)
    assert cache.db.execute('SELECT completed FROM rirs WHERE pose=?',(key,)).fetchone()[0]==1
    resumed=PositionRIRCache(tmp_path,renderer,{'room':'one'});resumed.get([1.,2.,3.]);assert renderer.calls==1


def test_so_exposed_mp3d_only_enters_training_after_admission():
    from avengine.acoustics.house_split_policy import eligible_house_splits
    passed={'family':'mp3d','status':'pass','visual':{'status':'pass'},'known_exposure':True,'input':{'original_split':None}}
    assert eligible_house_splits(passed)==('train',)
    assert eligible_house_splits({**passed,'status':'review'})==()
    assert eligible_house_splits({**passed,'visual':{'status':'review'}})==()
    assert eligible_house_splits({**passed,'family':'hm3d','input':{'original_split':'train'}})==()


def test_clean_hm3d_val_keeps_whole_house_holdouts():
    from avengine.acoustics.house_split_policy import eligible_house_splits
    assert eligible_house_splits({'family':'hm3d','status':'pass','visual':{'status':'pass'},'known_exposure':False,'input':{'original_split':'val'}})==('development','test')


def test_native_numpy_metadata_serializes_and_invalid_metadata_leaves_no_file(tmp_path):
    from avengine.timeline.continuous_pilot import write_json
    target=tmp_path/'metadata.json'
    write_json(target,{'ok':np.bool_(True),'count':np.int64(4),'position':np.array([1.,2.,3.])})
    assert json.loads(target.read_text())=={'ok':True,'count':4,'position':[1.,2.,3.]}
    broken=tmp_path/'invalid.json'
    with pytest.raises(ValueError):write_json(broken,{'seconds':float('inf')})
    assert not broken.exists()



def test_interrupted_metadata_publication_retries_without_partial_final(tmp_path, monkeypatch):
    import avengine.timeline.continuous_pilot as pilot
    target = tmp_path/'progress.json'
    actual_link = pilot.os.link
    with monkeypatch.context() as scope:
        def interrupt(*args, **kwargs):
            raise OSError('simulated interruption before atomic publication')
        scope.setattr(pilot.os, 'link', interrupt)
        with pytest.raises(OSError):
            pilot.write_json(target, {'done': True})
    assert not target.exists()
    assert list(tmp_path.glob('.*.partial'))
    pilot.write_json(target, {'done': True})
    assert json.loads(target.read_text()) == {'done': True}


def test_two_workers_can_initialize_same_exact_cache(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import avengine.timeline.continuous_pilot as pilot
    rendezvous = threading.Barrier(2)
    actual_write = pilot.write_json
    def concurrent_write(path, value):
        if path.name == 'configuration.json':
            rendezvous.wait(timeout=5)
        return actual_write(path, value)
    monkeypatch.setattr(pilot, 'write_json', concurrent_write)
    def initialize():
        cache = pilot.PositionRIRCache(tmp_path, FakeRenderer(), {'room': 'one'})
        cache.db.close()
        return json.loads((tmp_path/'configuration.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(initialize) for _ in range(2)]
        assert [f.result() for f in futures] == [{'room': 'one'}, {'room': 'one'}]


def test_orphan_background_is_retained_and_completed_pointer_resumes(tmp_path, monkeypatch):
    import avengine.acoustics.house_acceptance_visual as visual
    from avengine.timeline.continuous_pilot import CPUProxyVideo
    orphan = tmp_path/'cpu_background.npz'
    orphan.write_bytes(b'interrupted older attempt')
    house = {'input': {'scene_directory': '/declared-read-only-scene', 'scan_id': 'scan'},
             'acoustic_package_manifest': 'package.json'}
    plan = {'listener_m': [0., 1., 0.], 'camera_forward': [0., 0., -1.],
            'sources': {'voice1': {'trajectory': track([0., 1., -2.])}}}
    mesh = SimpleNamespace(face_normals=np.zeros((1,3)))
    monkeypatch.setattr(visual, 'load_raw_scan', lambda path: (mesh, np.zeros((1,3)), {}))
    class EmptyRayGeometry:
        def __init__(self, *args):
            self.receipt = {'test_empty_geometry': True}
        def intersects_location(self, *args):
            return np.empty((0,3)), np.empty(0,dtype=int), np.empty(0,dtype=int)
    monkeypatch.setattr(visual, 'NativeRawRayIntersector', EmptyRayGeometry)
    first = CPUProxyVideo(house, plan, {}, tmp_path, width=2, height=2)
    assert orphan.read_bytes() == b'interrupted older attempt'
    config = json.loads((tmp_path/'background_configuration.json').read_text())
    assert config['data_file'] != orphan.name
    def prohibit_raw_load(path):
        raise AssertionError('complete cache should skip raw geometry')
    monkeypatch.setattr(visual, 'load_raw_scan', prohibit_raw_load)
    resumed = CPUProxyVideo(house, plan, {}, tmp_path, width=2, height=2)
    assert resumed.receipt['background_cache_reused']
    assert np.array_equal(resumed.background, first.background)
