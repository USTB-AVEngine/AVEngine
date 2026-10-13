import unittest
from copy import deepcopy
import numpy as np
from avengine.timeline.v7_phase2_sources import source_owners,validate_phase2_bindings,audit_phase2_bindings
from avengine.timeline.v7_phase2_motion_audio import median_reference_check,emitted_intervals
from avengine.acoustics.moving_rir_acceptance import qualify_energy

def fixture():
 a={'asset_id':'registered_person','endpoint_id':'voice1','native_instance_id':1,'kind':'human','allowed_classes':['footsteps'],'silent_decor':False,'emitter_local_m':[0,1.5,0]}
 s={'phase2_profile':True,'sources':{'voice1':{'asset_id':a['asset_id'],'actor_id':1,'native_visual_kind':'registered_drawable'},
 'voice1_foot_L':{'asset_id':a['asset_id'],'actor_id':1,'native_visual_kind':'registered_drawable','owner_endpoint_id':'voice1','emitter_part':'foot_L'}},
 'events':[{'endpoint_id':'voice1_foot_L','source_id':1,'sound_class':'footsteps'}],
 'production_scenario':{'motion_requests':{'voice1':{'action':'walk'}}}}
 return a,s

def metric(delta=0):
 return {'status':'measured','foa_error_deg':0.,'geometry_error_deg':0.,'direct_energy_difference_db':delta,'total_energy_difference_db':delta,'drr_difference_db':0.,'c50_difference_db':0.,'direct_delay_difference_ms':0.,'rt60_difference_s':0.,'rt60_reference_s':.8,'tail_truncated':False,'octave_energy':[{'center_hz':f,'energy_difference_db':delta}for f in [125,250,500,1000,2000,4000]]}

class Phase2AdversarialBindingTests(unittest.TestCase):
 def test_standing_body_cannot_emit_footsteps(self):
  a,s=fixture();s['production_scenario']['motion_requests']['voice1']['action']='sit'
  with self.assertRaises(ValueError):validate_phase2_bindings(s,[a])
 def test_invisible_foot_endpoint_rejected(self):
  a,s=fixture();s['sources']['voice1_foot_L']['native_visual_kind']='audio_only'
  with self.assertRaises(ValueError):validate_phase2_bindings(s,[a])
 def test_animal_cannot_substitute_for_human_footsteps(self):
  a,s=fixture();a['kind']='animal'
  with self.assertRaises(ValueError):validate_phase2_bindings(s,[a])
 def test_event_owner_cannot_be_another_registered_body(self):
  a,s=fixture();s['events'][0]['source_id']=2
  with self.assertRaises(ValueError):validate_phase2_bindings(s,[a])
 def test_pending_contact_plan_cannot_be_sealed(self):
  a,s=fixture();s['events']=[];s['planned_footsteps']=[{'endpoint_id':'voice1'}]
  with self.assertRaises(ValueError):validate_phase2_bindings(s,[a])
  self.assertEqual(validate_phase2_bindings(s,[a],allow_pending_feet=True)['status'],'pass')
 def test_independent_native_child_readback_detects_02mm_error(self):
  a,s=fixture()
  row={'endpoint_id':'voice1','asset_id':a['asset_id'],'actor_id':1,'native_object_id':42,'visible_pixels':1000,'body_to_world_m':np.eye(4).tolist(),
   'attached_emitter_readbacks':[{'endpoint_id':'voice1_foot_L','native_object_id':42,'local_m':[0,.025,0],'position_m':[.0002,.025,0]}]}
  self.assertEqual(audit_phase2_bindings(s,[a],{'frames':[{'actors':[row]}]})['status'],'fail')
 def test_correct_native_child_has_no_invisible_emitter(self):
  a,s=fixture()
  row={'endpoint_id':'voice1','asset_id':a['asset_id'],'actor_id':1,'native_object_id':42,'visible_pixels':1000,'body_to_world_m':np.eye(4).tolist(),
   'attached_emitter_readbacks':[{'endpoint_id':'voice1_foot_L','native_object_id':42,'local_m':[0,.025,0],'position_m':[0,.025,0]}]}
  got=audit_phase2_bindings(s,[a],{'frames':[{'actors':[row]}]})
  self.assertEqual((got['status'],got['invisible_sources']),('pass',0))

class Phase2AbsoluteEnergyTests(unittest.TestCase):
 def test_signed_repeat_errors_cannot_cancel(self):
  got=median_reference_check([metric(-2),metric(0),metric(2)])
  self.assertEqual(qualify_energy([got])['status'],'fail')
 def test_unchanged_one_db_boundary(self):
  got=median_reference_check([metric(.99),metric(.99),metric(.99)])
  self.assertEqual(qualify_energy([got])['status'],'pass')
  got=median_reference_check([metric(1.01),metric(1.01),metric(1.01)])
  self.assertEqual(qualify_energy([got])['status'],'fail')
 def test_missing_independent_reference_blocks_admission(self):
  self.assertEqual(qualify_energy([median_reference_check([metric(),metric()])])['status'],'fail')
 def test_shortest_reference_decay_keeps_stricter_relative_gate(self):
  rows=[metric(),metric(),metric()]
  for r,t in zip(rows,[.2,1.,1.]):r.update(rt60_reference_s=t,rt60_difference_s=.06)
  self.assertEqual(qualify_energy([median_reference_check(rows)])['status'],'fail')
 def test_silence_placeholders_never_invent_emission(self):
  x=np.zeros(16000);x[3200:4800]=1
  self.assertEqual(emitted_intervals(x),[(3040,4960)])
  self.assertEqual(emitted_intervals(np.zeros(16000)),[])


class Phase2FullSequenceTests(unittest.TestCase):
 def valid_visual(self):
  from avengine.timeline.v7_phase2_checks import motion_checks
  feet={s:{'in_contact':True,'floor_delta_m':0.,'support_lock_error_m':0.}for s in ['L','R']}
  source=[{'moving':True,'feet':deepcopy(feet)}for _ in range(901)]
  actual=[{'root_max_error':0.,'native_emitter_readback_error_m':0.}for _ in range(901)]
  record={'status':'pass','source_frames':source,'native_30Hz_readbacks':actual,
    'maximum_navmesh_horizontal_error_m':0.,'minimum_certified_triangle_nonfloor_clearance_m':.03,'lineage':{'roster':'readonly'},'original_full_skin_floor_collision':{'status':'pass','worst_original_skin_floor_delta_m':-.0008}}
  return {'native_motion_admissions':{'voice1':record},'phase2_requested_motion':{'voice1':{'action':'walk'}}}
 def test_missing_original_skin_floor_proof_blocks_entire_sequence(self):
  from avengine.timeline.v7_phase2_checks import motion_checks
  v=self.valid_visual();v['native_motion_admissions']['voice1'].pop('original_full_skin_floor_collision')
  self.assertEqual(motion_checks(v)['status'],'fail')
 def test_original_skin_penetration_beyond_native_margin_blocks_sequence(self):
  from avengine.timeline.v7_phase2_checks import motion_checks
  v=self.valid_visual();v['native_motion_admissions']['voice1']['original_full_skin_floor_collision']['worst_original_skin_floor_delta_m']=-.00102
  self.assertEqual(motion_checks(v)['status'],'fail')
 def test_missing_requested_motion_rejected(self):
  from avengine.timeline.v7_phase2_checks import motion_checks
  v=self.valid_visual();v['native_motion_admissions']={}
  self.assertEqual(motion_checks(v)['status'],'fail')
 def test_one_bad_interior_contact_frame_blocks_whole_sequence(self):
  from avengine.timeline.v7_phase2_checks import motion_checks
  v=self.valid_visual()
  for f in v['native_motion_admissions']['voice1']['source_frames'][450]['feet'].values():f['in_contact']=False
  self.assertEqual(motion_checks(v)['status'],'fail')
 def test_one_bad_head_readback_blocks_whole_sequence(self):
  from avengine.timeline.v7_phase2_checks import motion_checks
  v=self.valid_visual();v['native_motion_admissions']['voice1']['native_30Hz_readbacks'][450]['native_emitter_readback_error_m']=.0002
  self.assertEqual(motion_checks(v)['status'],'fail')

class Phase2VerticalSurfaceTests(unittest.TestCase):
 def test_original_triangle_support_rejects_hole_and_keeps_stacked_surfaces(self):
  import tempfile
  from pathlib import Path
  from types import SimpleNamespace
  from avengine.assets.v7_phase2_floor import OriginalVerticalStageRay
  mesh=SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[0,0,1],[0,.1,0],[1,.1,0],[0,.1,1]],float),faces=np.array([[0,1,2],[3,4,5]]))
  with tempfile.TemporaryDirectory()as temp:
   p=Path(temp)/'source.glb';p.write_bytes(b'owned hermetic surface fixture')
   ray=OriginalVerticalStageRay(p,mesh=mesh)
   np.testing.assert_allclose(ray.many([[.2,0,.2]]),[.1],atol=1e-12)
   with self.assertRaises(ValueError):ray.many([[.9,0,.9]])
 def test_native_stage_z_up_becomes_world_y_up_without_reflection(self):
  from types import SimpleNamespace
  from unittest.mock import patch
  import tempfile
  from pathlib import Path
  from avengine.assets.v7_phase2_floor import original_native_stage_mesh
  attr=SimpleNamespace(orient_up=[0,0,1],orient_front=[0,1,0],scale=[1,1,1],origin=[0,0,0])
  manager=SimpleNamespace(get_template_by_handle=lambda p:attr)
  video=SimpleNamespace(sim=SimpleNamespace(metadata_mediator=SimpleNamespace(stage_template_manager=manager)),receipt={'stage':'owned'})
  mesh=SimpleNamespace(vertices=np.array([[1,2,3],[2,2,3],[1,3,3]],float),faces=np.array([[0,1,2]]))
  with tempfile.TemporaryDirectory()as temp:
   p=Path(temp)/'stage.glb';p.write_bytes(b'owned hermetic native axis fixture')
   with patch('avengine.capture.v7_visible_mesh.indexed_static_mesh',return_value=mesh):
    world,proof=original_native_stage_mesh(video,p)
   np.testing.assert_allclose(world.vertices[0],[1,3,-2],atol=1e-12)
   self.assertAlmostEqual(np.linalg.det(proof['source_coordinate_to_native_world_rotation']),1.)
   self.assertFalse(proof['triangles_decimated'])

class Phase2PosedEmitterDispatchTests(unittest.TestCase):
 def test_camera_query_without_video_index_reads_actual_seated_pose(self):
  from avengine.capture.v7_phase2_native import NativePhase2Video
  video=object.__new__(NativePhase2Video);observed=[]
  video._apply_motion_entry=lambda entry,index:observed.append(index) or {'native_emitter_readback_error_m':0.}
  got=video._apply_entry({'motion_episode':{'motion_kind':'sit'}},{})
  self.assertEqual(observed,[0])
  self.assertEqual(got['native_emitter_readback_error_m'],0.)
 def test_video_frame_query_uses_exact_30hz_pose_index(self):
  from avengine.capture.v7_phase2_native import NativePhase2Video
  video=object.__new__(NativePhase2Video);observed=[]
  video._apply_motion_entry=lambda entry,index:observed.append(index) or {}
  video._apply_entry({'motion_episode':{'motion_kind':'human_walk'}},{},frame_index=225)
  self.assertEqual(observed,[450])

if __name__=='__main__':unittest.main()


def test_wrong_pose_control_uses_other_real_emitter_when_foot_endpoints_coincide():
    import numpy as np
    from avengine.timeline.v7_phase2_motion_audio import wrong_registered_pose
    scene={'listener_m':[0,0,0],'sources':{
        'foot':{'trajectory':[{'time_s':0.,'position_m':[1,-1,0]},{'time_s':1.,'position_m':[1,-1,0]}]},
        'mouth':{'trajectory':[{'time_s':0.,'position_m':[1,0,0]},{'time_s':1.,'position_m':[1,0,0]}]},
        'ghost':{'native_visual_kind':'audio_only','trajectory':[{'time_s':0.,'position_m':[-1,0,0]},{'time_s':1.,'position_m':[-1,0,0]}]}}}
    endpoint,point,angle=wrong_registered_pose(scene,np.array([1,-1,0]),.5)
    assert endpoint=='mouth'
    assert np.array_equal(point,[1,0,0])
    assert 44<angle<46


def test_exact_phase2_native_reuse_rejects_changed_geometry_and_motion():
    from copy import deepcopy
    from avengine.timeline.v7_phase2_generation import exact_phase2_visual_identity
    a={'scene_id':'old','independent_parent_scene_id':'old','listener_m':[0,1,0],'motion_requests':{'human':{'action':'walk','speed':.15}}}
    b=deepcopy(a);b['scene_id']='new';b['independent_parent_scene_id']='new';b['reuse_qualified_phase2_native_parent']='/private/same-pilot/old'
    assert exact_phase2_visual_identity(a)==exact_phase2_visual_identity(b)
    b['listener_m'][0]=.01
    assert exact_phase2_visual_identity(a)!=exact_phase2_visual_identity(b)
    b=deepcopy(a);b['motion_requests']['human']['speed']=.16
    assert exact_phase2_visual_identity(a)!=exact_phase2_visual_identity(b)


def test_exact_native_reuse_only_omits_identical_animal_calls():
    from copy import deepcopy
    from avengine.timeline.v7_phase2_generation import exact_phase2_visual_reuse
    old={'scene_id':'old','listener_m':[0,1,0],'sources':{'h':{'species_id':'human'},'dog':{'species_id':'dog'}},
         'events':[{'endpoint_id':'h','start_tick':1},{'endpoint_id':'dog','start_tick':2},{'endpoint_id':'dog','start_tick':10}]}
    new=deepcopy(old);new['scene_id']='new';new['events']=new['events'][:2]
    assert exact_phase2_visual_reuse(old,new)
    new['events'][0]['start_tick']=3
    assert not exact_phase2_visual_reuse(old,new)
    new=deepcopy(old);new['events']=new['events'][:2];new['events'][1]['start_tick']=4
    assert not exact_phase2_visual_reuse(old,new)
    new=deepcopy(old);new['events']=new['events'][:1]
    assert not exact_phase2_visual_reuse(old,new)

def test_phase2_sampling_preserves_inactive_and_human_budget():
    from avengine.timeline.v7_registered_audio import phase2_ray_multiplier
    import numpy as np
    spec={'species_id':'dog','trajectory':[{'position_m':[0,0,0]},{'position_m':[1,0,0]}]}
    scene={'phase2_profile':True,'sources':{'dog':spec}}
    assert phase2_ray_multiplier(scene,{'dog':np.ones(20)})==9
    assert phase2_ray_multiplier(scene,{'dog':np.zeros(20)})==1
    spec['species_id']='human'
    assert phase2_ray_multiplier(scene,{'dog':np.ones(20)})==9
    scene['phase2_profile']=False
    assert phase2_ray_multiplier(scene,{'dog':np.ones(20)})==1


def _read_original_foot_child(vertices):
    from types import SimpleNamespace
    from unittest.mock import patch
    from avengine.capture.v7_phase2_native import NativePhase2Video
    from avengine.capture.v7_visible_native import NativeVisibleVideo
    class Child:
        def __init__(self): self.translation = np.zeros(3)
        def absolute_transformation(self):
            matrix = np.eye(4); matrix[:3, 3] = self.translation; return matrix
    entry = {'endpoint_id': 'person', 'declaration': {},
        'object': SimpleNamespace(root_scene_node=SimpleNamespace(create_child=Child)),
        'motion_episode': {'source_frames': [{'mouth_m': [0, 1.5, 0],
            'moving': True, 'feet': {'R': {'sole_local_m': [0, 0, 0],
                'in_contact': True, 'floor_delta_m': 0.}}}],
            'pose_glbs': ['hermetic_original_pose']}}
    video = object.__new__(NativePhase2Video)
    video.mn = SimpleNamespace(Vector3=lambda p: np.array(p))
    actual = {'body_to_world_m': np.eye(4).tolist(), 'native_object_id': 12}
    with patch.object(NativeVisibleVideo, '_apply_motion_entry', return_value=actual), \
            patch('avengine.capture.v7_visible_mesh.indexed_static_mesh',
                return_value=SimpleNamespace(vertices=np.array(vertices))):
        return video._apply_motion_entry(entry, 0)


def test_closest_buried_original_vertex_does_not_hide_valid_foot_surface():
    # The buried point is closer to the intended 25mm target, but cannot emit.
    actual = _read_original_foot_child([[0, 0.001, 0], [0.03, 0.012, 0]])
    child = actual['attached_emitter_readbacks'][0]
    assert child['original_posed_vertex_index'] == 1
    np.testing.assert_array_equal(child['position_m'], [0.03, 0.012, 0])
    assert child['readback_error_m'] == 0.


def test_no_original_surface_vertex_above_contact_keeps_rejection():
    import pytest
    with pytest.raises(ValueError, match='unchanged 50mm and 2mm gates'):
        _read_original_foot_child([[0, 0.001, 0], [0, 0.051, 0]])
