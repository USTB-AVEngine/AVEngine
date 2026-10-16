from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import box,mapping
from avengine.rooms.walkable_map import WalkableMap,SPECIES_RADII_M
from avengine.construction.joint import FamilyRequest
from avengine.construction.joint.context import RoomContext
from avengine.construction.joint.certification import certify_plan
from avengine.construction.joint.families.motion_animal import MotionAnimalPlugin

pytestmark=pytest.mark.fast_unit


@pytest.mark.parametrize('asymmetric',[False,True])
@pytest.mark.parametrize('closer',[False,True])
@pytest.mark.parametrize('negative',[False,True])
@pytest.mark.parametrize('duration',[1.2,2.3])
@pytest.mark.parametrize('kind,species,mode',[('dog_walk','dog','walk_animation'),('cat_walk','cat','walk_animation'),('animal_slide','cat','slide_no_animation')])
def test_animal_joint_plugin_schedules_real_programs_and_verifies_later_still_branch(tmp_path,kind,species,mode,duration,negative,closer,asymmetric):
    layers={key:np.ones((300,320),bool)for key in SPECIES_RADII_M}
    metadata={'version':'walkable-map-v1','room_id':'fixture','species_radii_m':SPECIES_RADII_M,
              'footprint_policy':'independent-radius-coverage-v2','source_identity':[]}
    walkable=WalkableMap(np.array([-8.,-12.]),.05,0.,layers,metadata,np.zeros((300,320)))
    walkable.path=str(tmp_path/'map_fixture.npz')
    room={'room_label':'fixture','floor_polygon':mapping(box(-8,-12,8,3)),
          'measured_floor_y_m':0.,'placement':{'camera_m':[0.,1.1,0.]}}
    pf=SimpleNamespace(snap_point=lambda point:np.asarray(point,float))
    context=RoomContext(room,walkable_map=walkable,pathfinder=pf,camera_limit=12,camera_heights_m=(1.1,),width=1280,height=720)
    animal={'asset_id':'animal','kind':'animal','species_id':species,'walk_allowed':mode=='walk_animation',
            'dimensions_m':[.24,.6,.7],'mouth_local_glb_m':[0.,.3,0.]}
    human={'asset_id':'speaker','kind':'human','species_id':'human','walk_allowed':True,'visual_gender':'male'}
    record={'sound_asset_id':'call','sound_class':'dog_bark'if species=='dog'else'cat_meow',
            'header':{'sample_rate_hz':16000,'sample_count':round(duration*16000)},'normalized_path':'fixture_call.wav',
            'screens':{'automatic_activity':{'activity_intervals_samples':[[1600,round((duration-.07)*16000)]]}}}
    speech={'sound_asset_id':'tts','sound_class':'speech_playback','source_type':'author_tts',
            'header':{'sample_rate_hz':16000,'sample_count':24000},'normalized_path':'fixture_speech.wav',
            'screens':{'automatic_activity':{'activity_intervals_samples':[[0,24000]]}}}
    data={'animal':animal,'companion':human,'record':record,'program':None,'animal_activity_s':((.1,duration-.07),),
          'animal_duration_s':duration,'speech':speech,'speech_activity_s':((0.,1.5),),
          'speech_duration_s':1.5,'semantic_sound':{'sound_asset_id':'tts'},'scenario_id':'fixture','kind':kind}
    contract={'animal_motion':{'animal':{'mode':mode}}}
    def pure_geometry_fixture(ctx,house,runtime,asset,authority,actor,route,directory):
        import json
        directory.mkdir(parents=True)
        filename=directory/'fixture_only.json'
        native_yaw=-float(actor.headings_rad[0])
        record={'world_roots_m':actor.positions_m.tolist(),
                'source_frames':[{'mouth_m':[0.,.3,0.]}for _ in range(901)],
                'yaw_rad':native_yaw,'yaw_rad_per_frame':[native_yaw]*901}
        filename.write_text(json.dumps({'record':record}))
        actor.metadata['joint_animal_prepared_geometry']=str(filename)
        actor.metadata['native_yaw_rad']=0.
    plugin=MotionAnimalPlugin(data,{'scene_id':'fixture','scene_seed':1},[animal,human],[record,speech],
        contract,{'house':'fixture'}, {'joint_max_geometry_preparations':1},tmp_path/'geometry',geometry_preparer=pure_geometry_fixture)
    request=FamilyRequest('motion_animal','nearer',('animal',),{'movement_bouts':1,'require_static_negative_comparator':negative,'closer_nearer_call':closer,'asymmetric_companion_speech':asymmetric})
    if duration==1.2 and kind=='animal_slide':
        with pytest.raises(StopIteration):next(plugin.candidates(context,request))
        return
    iterator=plugin.candidates(context,request)
    plan=next(iterator)
    assert certify_plan(context,plan).passed
    checks=list(plugin.verify(context,request,plan))
    assert checks and all(check.status=='pass'for check in checks),[(c.name,c.reason)for c in checks if c.status!='pass']
    assert set(plan.actors)=={'animal','speaker'}
    row=plan.metadata['row']
    assert row['phase2_whole_source_recordings']['animal']==pytest.approx([round(max(1.3,3.3-duration),1)if closer else 1.3,21.3])
    assert row['joint_solver_only']
    if asymmetric:
        assert row['phase2_source_event_schedule_s']['speaker']==[6.2,17.,27.]
        assert plan.metadata['planned_silent_intervals_s'][0][1]==6.0
    from avengine.timeline.v7_full_question_delivery import current_selection_snapshot
    snapshot=current_selection_snapshot(row['scene_demand'],next_questions=70)
    assert snapshot['observed_logical_questions']==0
    assert snapshot['observed_family_counts']=={}
    assert abs(sum(snapshot['target_shares'].values())-1.)<1e-8
    assert row['automatic_construction']['solver_entry']=='avengine.construction.joint.solve'
    assert row['motion_requests']['animal']['maximum_native_route_attempts']==1
    if negative:
        human_route=row['motion_requests']['speaker']['motion_quartet_route']
        assert human_route['joint_exact_route']
        np.testing.assert_array_equal(human_route['positions_xz_m'],plan.actors['speaker'].positions_m[:,[0,2]])
        assert row['motion_requests']['speaker']['speed_m_s']==.85
    assert len(row['motion_during_condition_plan']['plans'])==4
    roots=plan.actors['animal'].positions_m
    assert np.max(np.abs(roots[20*30:]-roots[20*30]))==0.
    assert all(abs(start*10-round(start*10))<1e-8 for starts in plan.metadata['source_file_starts_s'].values()for start in starts)
    assert all(not(event.start_s<end and event.wet_end_s>begin)for event in plan.sounds for begin,end in plan.metadata['planned_silent_intervals_s'])

    from avengine.capture.v7_offscreen_layout import proposal_view_ok
    assert proposal_view_ok({'native_view_roles':row['native_view_roles']},'speaker',
        plan.actors['speaker'].positions_m[0]-np.asarray(plan.camera.position_m),plan.camera.forward,False)
    # A second companion site shares exactly the same original animal geometry.
    # One preparation remains sufficient even with the explicit per-solve cap=1.
    second=next(iterator)
    assert plugin.geometry_preparations==1
    np.testing.assert_array_equal(second.actors['animal'].positions_m,plan.actors['animal'].positions_m)
    assert second.actors['animal'].metadata['joint_animal_prepared_geometry']==plan.actors['animal'].metadata['joint_animal_prepared_geometry']
    assert not np.array_equal(second.actors['speaker'].positions_m,plan.actors['speaker'].positions_m)

    if kind=='dog_walk'and duration==1.2:
        attempts=[]
        def support_conflict_then_next_route(ctx,house,runtime,asset,authority,actor,route,directory):
            attempts.append(actor.positions_m.copy())
            if len(attempts)==1:raise ValueError('original scan support missing in unit fixture')
            pure_geometry_fixture(ctx,house,runtime,asset,authority,actor,route,directory)
        other=MotionAnimalPlugin(data,{'scene_id':'fixture_failure','scene_seed':1},[animal,human],[record,speech],
            contract,{'house':'fixture'},{'joint_max_geometry_preparations':2},tmp_path/'failed_geometry',
            geometry_preparer=support_conflict_then_next_route)
        recovered=next(other.candidates(context,request))
        assert recovered is not None and len(attempts)==2
        assert not np.array_equal(attempts[0],attempts[1])
        assert len(other.geometry_failure_cache)==1
