"""Checks that catch renderer routing, graphics accounting and false admission."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest


def load_tool(name):
    path=Path(__file__).resolve().parents[1]/'tools/rooms'/name
    spec=importlib.util.spec_from_file_location(name.removesuffix('.py'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_graphics_memory_is_counted_and_unavailable_is_preserved():
    guard=load_tool('v7_kujiale_ue_guard.py')
    xml='''<nvidia_smi_log><gpu><uuid>own-card</uuid><fb_memory_usage><free>17000 MiB</free><used>100 MiB</used></fb_memory_usage><processes>
    <process_info><pid>21</pid><type>C+G</type><process_name>UnrealEditor</process_name><used_memory>512 MiB</used_memory></process_info>
    <process_info><pid>22</pid><type>G</type><process_name>graphics</process_name><used_memory>N/A</used_memory></process_info>
    </processes></gpu></nvidia_smi_log>'''
    rows=guard.parse_gpu_xml(xml)
    assert rows[0]['processes'][0]['memory_mib']==512
    assert rows[0]['processes'][0]['type']=='C+G'
    assert rows[0]['processes'][1]['memory_mib'] is None


def test_unknown_renderer_fails_before_optional_runtime_import():
    from avengine.capture.native_continuous_backend import create_native_video
    with pytest.raises(ValueError,match='Unsupported native visual backend'):
        create_native_video({'input':{'visual_backend':'habitat_typo'}},{},{},[])


def test_ue_route_never_falls_back_to_habitat(monkeypatch):
    import sys,types
    from avengine.capture.native_continuous_backend import create_native_video
    native=types.ModuleType('avengine.capture.continuous_ue_video')
    def reject(*args,**kwargs):raise RuntimeError('native project unavailable')
    native.UENativeContinuousVideo=reject
    monkeypatch.setitem(sys.modules,native.__name__,native)
    with pytest.raises(RuntimeError,match='native project unavailable'):
        create_native_video({'input':{'visual_backend':'ue_spear'}},{},{},[])


def test_frozen_black_gate_accepts_boundary_and_rejects_next_pixel():
    check=load_tool('check_v7_kujiale_ue.py')
    rgb=np.full((10,100,3),80,dtype=np.uint8);rgb.reshape(-1,3)[:150]=7
    depth=np.ones((10,100),dtype=np.float32)
    params={'black_pixel_channel_lt':8,'black_fraction_max':.15}
    q,_=check.view_quality(rgb,depth,{'kind':'perspective'},{},params)
    assert q['status']=='pixel_checks_pass' and q['black_fraction']==.15
    rgb.reshape(-1,3)[150]=7
    q,_=check.view_quality(rgb,depth,{'kind':'perspective'},{},params)
    assert q['status']=='fail'
    # A channel exactly 8 is outside the frozen all-channels-below-8 gate.
    rgb.reshape(-1,3)[150]=[8,7,7]
    q,_=check.view_quality(rgb,depth,{'kind':'perspective'},{},params)
    assert q['status']=='pixel_checks_pass'


def test_native_geometry_cannot_be_translated_to_pass(tmp_path):
    check=load_tool('check_v7_kujiale_ue.py')
    source=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.]])
    build=tmp_path/'surface';build.mkdir()
    np.save(build/'vertices.npy',source)
    actual=(source[:,[0,2,1]]*100).astype('<f4');actual[:,0]+=10
    vp=tmp_path/'vertices.f32';fp=tmp_path/'triangles.u32'
    actual.tofile(vp);np.array([[0,1,2]],dtype='<u4').tofile(fp)
    audit={'native_render_geometry':{'world_vertices_ue_cm_f32':str(vp),'triangles_u32':str(fp)}}
    result=check.geometry_parity({'build_directory':str(tmp_path)},audit)
    assert result['status']=='fail'
    assert result['max_vertex_error_m']>.09
    assert result['alignment_fitting_used'] is False


def test_native_full_motion_rejects_obstacle_between_legal_endpoints():
    from avengine.capture.native_ue_trajectory import check_native_trajectories
    class Navigation:
        def snap_point(self,p):return np.asarray(p)
        def distance_to_closest_obstacle(self,p,r):return 0. if 1.4<float(p[0])<1.6 else r
    scene={'duration_s':2.,'sources':{'voice1':{'trajectory':[
        {'time_s':0.,'position_m':[1.,1.2,0.]},{'time_s':2.,'position_m':[2.,1.2,0.]}]}}}
    result=check_native_trajectories(scene,Navigation())
    assert result['status']=='fail' and result['trajectory_rewritten'] is False
    assert result['checks'][0]['minimum_obstacle_clearance_m']==0.


def test_native_full_motion_keeps_valid_input_track():
    from avengine.capture.native_ue_trajectory import check_native_trajectories
    class Navigation:
        def snap_point(self,p):return np.asarray(p)
        def distance_to_closest_obstacle(self,p,r):return r
    scene={'duration_s':2.,'sources':{'voice1':{'trajectory':[
        {'time_s':0.,'position_m':[1.,1.2,0.]},{'time_s':2.,'position_m':[2.,1.2,0.]}]}}}
    result=check_native_trajectories(scene,Navigation())
    assert result['status']=='pass' and result['checks'][0]['frame_count']==30
    assert scene['sources']['voice1']['trajectory'][1]['position_m']==[2.,1.2,0.]


def test_ue_native_depth_visibility_uses_same_renderer_and_depth_gate():
    from avengine.qa.pixel_visibility import compile_depth_pixel_visibility_truth
    context={'renderer_backend':'ue_spear','rgb_renderer_backend':'ue_spear',
        'camera_contract_id':'native','semantic_id_namespace':'native_depth','resolution_hw':[2,2],
        'frame_indices':[0],'camera_pose_ids':['actual-native-pose']}
    normal=np.array([[1.,2.],[3.,65504.]],dtype=np.float32)
    target=np.array([[1.,4.],[65504.,65504.]],dtype=np.float32)
    truth=compile_depth_pixel_visibility_truth(normal_depth_m_frames=[normal],
        target_only_depth_m_frames_by_instance={'a':[target]},semantic_ids_by_instance={'a':1001},
        normal_context={'pass_kind':'modal_scene',**context},target_only_contexts_by_instance={'a':{'pass_kind':'target_only','target_instance_id':'a',**context}},
        target_only_background_depth_m=65504.,absolute_tolerance_m=.01,relative_tolerance=.002)
    assert truth['renderer_backend']=='ue_spear'
    assert truth['depth_comparison']['absolute_tolerance_m']==.01
    assert truth['per_instance']['a']['frames'][0]['visible_pixels']==1


def test_empty_native_export_never_passes(tmp_path):
    check=load_tool('check_v7_kujiale_ue.py');(tmp_path/'surface').mkdir()
    np.save(tmp_path/'surface/vertices.npy',np.array([[0.,0.,0.]]))
    vp=tmp_path/'vertices.f32';fp=tmp_path/'triangles.u32';vp.touch();fp.touch()
    with pytest.raises(ValueError,match='empty'):
        check.geometry_parity({'build_directory':str(tmp_path)},
            {'native_render_geometry':{'world_vertices_ue_cm_f32':str(vp),'triangles_u32':str(fp)}})


@pytest.mark.parametrize('forward,expected',[
    ([1.,0.,0.],[1.,0.,0.]),([0.,0.,1.],[0.,0.,1.]),
    ([-1.,0.,0.],[-1.,0.,0.]),([0.,0.,-1.],[0.,0.,-1.])])
def test_ue_human_anatomical_forward_matches_cardinal_world_directions(forward,expected):
    from avengine.capture.continuous_ue_video import UENativeContinuousVideo
    actor=UENativeContinuousVideo.__new__(UENativeContinuousVideo)
    class Navigation:
        def snap_point(self,p):return np.asarray(p)
    actor.pf=Navigation();actor.floor_offset=0.
    actor.scene={'duration_s':30.,'listener_m':forward,
        'sources':{'voice1':{'trajectory':[{'time_s':0.,'position_m':[0.,1.2,0.]},
                                         {'time_s':30.,'position_m':[0.,1.2,0.]}]}}}
    actor.asset_profiles={'blue':{'timeline':{'walk_phase_period_frames':16}}}
    state=actor._state({'endpoint_id':'voice1','asset_id':'blue'},'source1',0)
    q=state['root_transform']['rotation_xyzw']
    import math
    native_yaw=-2*math.atan2(q[1],q[3])+math.pi/2 # UE imported human's measured +Y forward.
    observed=np.array([math.cos(native_yaw),0.,math.sin(native_yaw)])
    np.testing.assert_allclose(observed,expected,atol=1e-12)


def test_orthographic_native_camera_distance_is_not_a_missing_floor(tmp_path,monkeypatch):
    pytest.importorskip('embreex')
    from avengine.capture import native_ue_qa as qa
    vertices=np.array([[-5.,0.,-5.],[5.,0.,-5.],[5.,0.,5.],[-5.,0.,5.]])
    v=tmp_path/'v.f32';f=tmp_path/'f.u32'
    (vertices[:,[0,2,1]]*100).astype('<f4').tofile(v)
    np.array([[0,1,2],[0,2,3]],dtype='<u4').tofile(f)
    audit={'native_render_geometry':{'world_vertices_ue_cm_f32':str(v),'triangles_u32':str(f),
        'sections':[{'triangle_count':2,'material_path':'diffuse'}]}}
    monkeypatch.setattr(qa,'source_materials',lambda _:{'diffuse':{'default_native_material':False,
        'missing_expected_texture':False,'glass_hint':False,'native_opaque':True,'source_status':'preview_surface'}})
    view={'kind':'orthographic','position_m':[0.,2.,0.],'pitch_deg':-90.,'yaw_deg':-90.,'ortho_width_cm':400.}
    xx,yy=np.meshgrid(np.arange(30),np.arange(20))
    origins,rays,_=qa.view_rays(view,xx.ravel()*4,yy.ravel()*4,120,80)
    points=origins+2*rays
    depth=np.linalg.norm(points-view['position_m'],axis=1).reshape(20,30)
    room={'floor_polygon':{'type':'Polygon','coordinates':[[[-4,-4],[4,-4],[4,4],[-4,4],[-4,-4]]]}}
    result=qa.depth_geometry_check(audit,view,depth,room)
    assert result['status']=='pass'
    assert result['depth_definition']=='camera_origin_distance'
    assert result['missing_solid_face_fraction']==0.
    assert result['physical_depth_model_median_errors_m']['camera_origin_distance']<1e-6
    assert result['alignment_fitting_used'] is False
