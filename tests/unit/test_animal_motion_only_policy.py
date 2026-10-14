import copy
import json
import math
import numpy as np
import pytest
from avengine.assets.animal_motion.motion_only_policy import (
    POLICY_NAME, qualify_motion, direction_checks, contact_checks, surface_checks,
    physical_direction_controls, validation_policy,
)
from avengine.assets.animal_motion.checks import direction_check
from avengine.assets.animal_motion.service import execute_request, completed_checks


def native_case(angle=0., mode='walk_animation'):
    frames=[]
    forward=np.array([math.sin(math.radians(angle)),0.,-math.cos(math.radians(angle))])
    for i in range(8):
        t=i/30;root=np.array([0.,0.,-.03*t]);feet={}
        for j,side in enumerate(('FL','FR','HL','HR')):
            contact=mode=='slide_no_animation' or i%4!=j
            feet[side]={'in_contact':contact,'floor_delta_m':.001 if contact else .03,
                        'support_segment':i//4,'support_world_m':[j*.02,.001,0.]}
        frames.append({'time_s':t,'world_root_m':root.tolist(),'root_world_m':root.tolist(),
            'head_world_m':(root+forward*.3+[0,.3,0]).tolist(),'tail_world_m':(root-forward*.3+[0,.3,0]).tolist(),
            'feet':feet,'body_floor_min_m':.001,'body_head_tail_pitch_deg':0.,
            'native_emitter_readback_error_m':0.,'navmesh_horizontal_error_m':0.,
            'minimum_nonfloor_clearance_m':.021,'complete_silhouette_in_frame':True,
            'mouth_world_m':(root+[0,.2,-.25]).tolist()})
    return {'asset_id':'dog','house':'room','motion_mode':mode,'status':'geometry_fail',
            'frame_count':len(frames),'frames':frames,'listener_m':[0,1,2],
            'camera_forward':[0,0,-1],'source_master':'/source.glb','source_provenance':{},
            'checks':{'status':'fail','failures':['historical_more_strict_rule']}}


def controls(native):
    rows=[]
    for angle in (180,90):
        frames=[];axis=np.array([math.sin(math.radians(angle)),0,-math.cos(math.radians(angle))])
        for f in native['frames']:
            root=np.array(f['root_world_m']);frames.append({'time_s':f['time_s'],'root':root.tolist(),'head':(root+axis*.3).tolist(),'tail':(root-axis*.3).tolist()})
        rows.append({'asset_id':native['asset_id'],'body_rotation_error_deg':angle,'physically_rotated_native_drawable':True,'frames':frames})
    return {'controls':rows}


def surface(n=8):
    return {'status':'fail','frames':[{'frame':i,'max_edge_growth':2.,'new_maximum_uv_seam_gap_m':.3,
        'regional_penetration_measurements':[
            {'region_pair':['Body','FL'],'new_area_m2':250e-6,'new_maximum_depth_m':.01},
            {'region_pair':['Body','FR'],'new_area_m2':150e-6,'new_maximum_depth_m':.001},
            {'region_pair':['Body','Body'],'new_area_m2':100.,'new_maximum_depth_m':100.},
        ]} for i in range(n)]}


def test_new_angle_policy_does_not_change_legacy_checker():
    n=native_case(19.)
    assert direction_checks(n['frames'])['status']=='pass'
    assert direction_check([f['time_s'] for f in n['frames']],[f['root_world_m'] for f in n['frames']],
                           [f['head_world_m'] for f in n['frames']],[f['tail_world_m'] for f in n['frames']])['status']=='fail'
    assert direction_checks(native_case(20.01)['frames'])['status']=='fail'


def test_only_support_paws_get_twelve_mm_limit_and_no_speed_gate():
    n=native_case();n['frames'][3]['feet']['FL']['support_world_m'][0]=.014
    assert contact_checks(n['frames'],'walk_animation')['status']=='pass'
    n['frames'][3]['feet']['FL']['support_world_m'][0]=.01501
    assert contact_checks(n['frames'],'walk_animation')['status']=='fail'
    n=native_case();n['frames'][1]['feet']['FL']['floor_delta_m']=.01201
    assert contact_checks(n['frames'],'walk_animation')['status']=='fail'


def test_cross_region_sum_depth_edge_and_same_region_ignore_are_independent():
    data=surface();assert surface_checks(data,8)['status']=='pass'
    data['frames'][2]['regional_penetration_measurements'][1]['new_area_m2']=151e-6
    assert surface_checks(data,8)['status']=='fail'
    data=surface();data['frames'][1]['regional_penetration_measurements'][0]['new_maximum_depth_m']=.01001
    assert surface_checks(data,8)['status']=='fail'
    data=surface();data['frames'][-1]['max_edge_growth']=2.01
    assert surface_checks(data,8)['status']=='fail'


def test_old_early_stop_cannot_be_promoted_without_full_coverage():
    data=surface(1)
    assert surface_checks(data,8)['status']=='unverified'
    data['frames'][0]['max_edge_growth']=2.8
    assert surface_checks(data,8)['status']=='fail'


def test_real_rotation_markers_and_recomputed_negative_angles_required():
    n=native_case();e=controls(n)
    assert physical_direction_controls(n,e)['status']=='pass'
    e['controls'][0]['physically_rotated_native_drawable']=False
    assert physical_direction_controls(n,e)['status']=='unverified'
    e=controls(n);e['controls'][0]['frames']=copy.deepcopy(n['frames'])
    assert physical_direction_controls(n,e)['status']=='fail'


def test_motion_only_does_not_hide_native_truncation_or_collision():
    n=native_case();e=controls(n)
    assert qualify_motion(n,surface(),e)['status']=='pass'
    n['frame_count']=145
    assert qualify_motion(n,surface(),e)['status']=='unverified'
    n=native_case();n['frames'][-1]['minimum_nonfloor_clearance_m']=0.
    assert qualify_motion(n,surface(),controls(n))['status']=='fail'
    n=native_case();n['frames'][-1]['native_emitter_readback_error_m']=.001
    assert qualify_motion(n,surface(),controls(n))['status']=='fail'


def test_new_request_uses_animation_despite_old_audio_failure_and_propagates_metadata(tmp_path,monkeypatch):
    import avengine.assets.animal_motion.motion_only_service as service
    def forbidden(*args,**kwargs):raise AssertionError('new native/RIR execution was not requested')
    monkeypatch.setattr(service,'run_habitat',forbidden)
    n=native_case();npth=tmp_path/'native.json';npth.write_text(json.dumps(n));apth=tmp_path/'failed_audio.json';audio={'status':'fail','motion_receipt':str(npth),'error':'historical_energy_failure'};apth.write_text(json.dumps(audio))
    assert completed_checks(n,audio,'walk_animation')['status']=='fail'
    record={'asset_id':'dog','species':'dog','has_walking':True,'files':{'motion_master_glb':{'path':'/source.glb'}}}
    result=execute_request(record,{'house':'room'},{},{'root_m':[0,0,0],'yaw_rad':0},tmp_path/'request',speed=.03,seconds=7/30,
        validation_policy=POLICY_NAME,reuse={'walk_animation':{'native':str(npth),'surface':surface(),'controls':controls(n),'audio':str(apth)}})
    assert result['status']=='pass' and result['motion_mode']=='walk_animation'
    assert result['admission_scope']=='motion_only' and result['audio_status']=='deferred_to_production'
    assert not result['strict_joint_audio_admission_claimed']
    scene=json.loads((tmp_path/'request/motion_scene.json').read_text());source=scene['sources']['animal']
    assert source['motion_metadata']['audio_deferred_to_production']
    assert 'animal_motion_audio' not in source
    assert json.loads(apth.read_text())==audio


def test_policy_settings_are_explicit_and_do_not_offer_unreviewed_variants():
    assert validation_policy()['maximum_edge_growth']==2.
    with pytest.raises(ValueError):validation_policy('anything_looser')


@pytest.mark.parametrize('has_walking', [False, True])
def test_motion_only_attempts_walking_then_native_slide_and_keeps_reason(tmp_path, monkeypatch, has_walking):
    import avengine.assets.animal_motion.motion_only_service as service
    static_pose = tmp_path / 'standing.glb'
    static_pose.write_bytes(b'owned-test-pose')
    slide = native_case(mode='slide_no_animation')
    for frame in slide['frames']:
        frame['pose_glb'] = str(static_pose)
    slide['native_direction_negative_frames'] = {}
    for row in controls(slide)['controls']:
        negative = row['frames']
        for frame in negative:
            frame['physically_rotated_native_drawable'] = True
        slide['native_direction_negative_frames'][str(row['body_rotation_error_deg'])] = negative
    calls = []
    def native(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs['mode'] == 'slide_no_animation'
        assert kwargs['physical_controls'] is True
        return slide
    monkeypatch.setattr(service, 'run_habitat', native)
    record = {'asset_id': 'dog', 'species': 'dog', 'has_walking': has_walking,
              'files': {'motion_master_glb': {'path': '/source.glb'}}}
    reuse = {}
    if has_walking:
        walking = native_case()
        path = tmp_path / 'walking.json'
        path.write_text(json.dumps(walking))
        bad_surface = surface()
        bad_surface['frames'][3]['max_edge_growth'] = 2.1
        reuse['walk_animation'] = {'native': str(path), 'surface': bad_surface, 'controls': controls(walking)}
    result = execute_request(record, {'house': 'room'}, {}, {'root_m': [0, 0, 0], 'yaw_rad': 0},
                             tmp_path / 'request', speed=.03, seconds=7/30,
                             validation_policy=POLICY_NAME, reuse=reuse)
    assert result['motion_mode'] == 'slide_no_animation'
    assert result['status'] == 'pass' and result['animation_attempted']
    assert result['motion_attempts'][0]['mode'] == 'walk_animation'
    assert result['motion_attempts'][0]['status'] == ('rejected' if has_walking else 'unavailable')
    assert result['motion_attempts'][1]['mode'] == 'slide_no_animation'
    assert result['motion_attempts'][1]['status'] == 'pass'
    assert result['audio_status'] == 'deferred_to_production'
    assert len(calls) == 1
    scene = json.loads((tmp_path / 'request/motion_scene.json').read_text())
    metadata = scene['sources']['animal']['motion_metadata']
    assert metadata['motion_mode'] == 'slide_no_animation'
    assert metadata['motion_reason'] == result['motion_attempts'][0]['reason']
    assert metadata['audio_deferred_to_production'] is True


def test_bounded_slide_buffers_require_identical_mesh_and_invariant_native_geometry(tmp_path):
    from avengine.assets.animal_motion.motion_only_policy import rigid_slide_surface_proof
    paths = [tmp_path / 'slot0.glb', tmp_path / 'slot1.glb']
    for path in paths: path.write_bytes(b'one actual registered rigid pose')
    n = native_case(mode='slide_no_animation')
    for i, f in enumerate(n['frames']):
        f.update(pose_glb=str(paths[i % 2]), pose_buffer_is_transient=True,
                 head_m=[0, .3, -.3], tail_m=[0, .3, .3], mouth_m=[0, .25, -.35],
                 mouth_bone_matrix=np.eye(4).tolist(), native_geometry_bounds_m=[[-.2, 0, -.4], [.2, .5, .4]])
    assert rigid_slide_surface_proof(n)['verified']
    n['frames'][3]['native_geometry_bounds_m'][1][1] += .01
    assert not rigid_slide_surface_proof(n)['verified']
    n['frames'][3]['native_geometry_bounds_m'][1][1] -= .01
    paths[1].write_bytes(b'changed shape')
    assert not rigid_slide_surface_proof(n)['verified']
