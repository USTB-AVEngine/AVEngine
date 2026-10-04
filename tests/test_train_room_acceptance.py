"""Room scope and strict positive-area retention regressions."""
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

pytestmark=pytest.mark.fast_unit

from tools.acoustics.train_room_acceptance import probe_plan,cached_match,strict_zero_scene

def test_bad_neighbor_room_does_not_veto_good_room():
    from tools.acoustics.accept_houses import classify
    import copy
    criteria=dict(leak_fail_above=.15,leak_pass_max=.05,minimum_listeners=2,minimum_rirs=6,minimum_valid_rt_fraction=.7,foa_angle_fail_deg=30,foa_angle_pass_deg=15,distance_exponent_pass=[.7,1.3])
    good=dict(leakage=[dict(room_label='good',escape_fraction=.039)],rirs=[dict(listener_m=[i%2,1.5,0],status='valid',rt60_mid_s=.5,direct_delay_error_ms=0,tail_truncated=False) for i in range(8)],foa_checks=[dict(check=dict(angle_deg=1)) for i in range(3)],distance_attenuation=[dict(status='valid',pressure_exponent=1)],known_exposure=False)
    whole=copy.deepcopy(good);whole['leakage'].append(dict(room_label='bad',escape_fraction=.359))
    classify(whole,criteria);classify(good,criteria)
    assert whole['status']=='fail' and good['status']=='pass'

def test_strict_zero_mask_keeps_float32_roundoff_positive_area():
    from tools.acoustics.derive_zero_area_glb import zero_surface_mask
    # Cross32 cancels these products; cross64 on the exact float32 values
    # retains a positive physical area. Strict mode must preserve the face.
    v=np.array([[0,0,0],[1,1.0000001192092896,0],[1.0000001192092896,1.000000238418579,0]],dtype=np.float32)
    faces=np.array([[0,1,2]])
    ordinary,_=zero_surface_mask(v,faces)
    strict,receipt=zero_surface_mask(v,faces,strict=True)
    assert bool(ordinary[0]) and not bool(strict[0])
    assert receipt['native_float32_zero_with_double_residue']==0

def test_every_room_has_eight_probes_at_two_listeners():
    room={'placement':{'camera_m':[0.,1.5,0.],'source_1_m':[3.,1.2,0.],'source_2_m':[3.,1.2,2.]}}
    probes=probe_plan(room)
    assert len(probes)==8
    assert len({tuple(listener) for name,listener,source in probes})==2
    assert len({name for name,listener,source in probes})==8
    assert all(np.linalg.norm(source-listener)>0 for name,listener,source in probes)

def test_cached_waveform_requires_identical_source_and_listener():
    m={'listener_m':[0.,1.,0.],'source_m':[2.,1.,0.]}
    assert cached_match(m,[0.,1.,0.],[2.,1.,0.])
    assert not cached_match(m,[0.4,1.,0.],[2.,1.,0.])
    assert not cached_match(m,[0.,1.,0.],[2.,1.,.4])

def test_strict_native_geometry_retains_tiny_positive_area(tmp_path):
    from dataclasses import dataclass
    @dataclass
    class Scene:
        objects:tuple
        manifest_path:Path
        material_categories:tuple
        triangle_count_by_material:dict
    vertices=np.array([[0,0,0],[1,0,0],[0,1e-12,0],[2,0,0]],dtype=np.float32)
    faces=np.array([[0,1,2],[0,1,3]],dtype=np.uint32)
    scene=Scene(({'object_id':'test','vertices':vertices,'triangles':faces,'triangle_material_ids':np.array([0,0],dtype=np.uint32)},),tmp_path/'source.json',('wall',),{'wall':2})
    result,receipt=strict_zero_scene(scene,tmp_path/'derived')
    assert receipt['removed_faces']==1 and receipt['maximum_removed_area_m2']==0
    np.testing.assert_array_equal(result.objects[0]['triangles'],[[0,1,2]])
    np.testing.assert_array_equal(scene.objects[0]['triangles'],faces)
    assert (tmp_path/'derived/native_arrays.npz').is_file()

def test_positive_faces_are_left_for_the_actual_native_uploader(tmp_path):
    from dataclasses import dataclass
    @dataclass
    class Scene:
        objects:tuple
        manifest_path:Path
        material_categories:tuple
        triangle_count_by_material:dict
    v=np.array([[0,0,0],[10000,10000.0009765625,0],[10000.0009765625,10000.001953125,0]],dtype=np.float32)
    faces=np.array([[0,1,2]],dtype=np.uint32)
    assert np.all(np.cross(v[1]-v[0],v[2]-v[0])==0)
    assert np.any(np.cross((v[1]-v[0]).astype(float),(v[2]-v[0]).astype(float))!=0)
    scene=Scene(({'object_id':'positive','vertices':v,'triangles':faces,'triangle_material_ids':np.array([0],dtype=np.uint32)},),tmp_path/'source.json',('wall',),{'wall':1})
    result,receipt=strict_zero_scene(scene,tmp_path/'derived')
    assert receipt['removed_faces']==0
    np.testing.assert_array_equal(result.objects[0]['triangles'],faces)

def test_resume_checks_command_and_start_time_of_owned_child():
    import subprocess,sys
    from tools.acoustics.run_train_room_visuals import owned_child_alive
    command=[sys.executable,'-c','import time; time.sleep(5)']
    proc=subprocess.Popen(command)
    try:
        ticks=Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split()[19]
        record=dict(pid=proc.pid,argv=command,proc_start_ticks=ticks)
        assert owned_child_alive(record)
        assert not owned_child_alive(dict(record,argv=command+['wrong']))
        assert not owned_child_alive(dict(record,proc_start_ticks=str(int(ticks)+1)))
    finally:proc.terminate();proc.wait()


def test_cpu_visual_worker_memory_bound_prevents_shared_host_exhaustion():
    import subprocess,sys
    source="from tools.acoustics.train_room_visuals import memory_limit; import numpy as np; limit=memory_limit(); assert limit <= 16*1024**3; \ntry: np.empty((20*1024**3,),dtype='u1')\nexcept MemoryError: pass\nelse: raise AssertionError('unbounded worker address space')"
    subprocess.run([sys.executable,'-c',source],check=True)


def test_cpu_overhead_shares_only_byte_identical_rgb_textures():
    from PIL import Image
    from tools.rooms.room_selection.cpu_overhead import texture_pixels
    a=Image.new('RGBA',(16,16),(13,45,78,255));b=a.copy();cache={}
    first=texture_pixels(a,cache);second=texture_pixels(b,cache)
    assert first is second and len(cache)==1
    np.testing.assert_array_equal(first,np.asarray(a.convert('RGB')))
    b.putpixel((0,0),(14,45,78,255))
    changed=texture_pixels(b,cache)
    assert changed is not first and len(cache)==2
    np.testing.assert_array_equal(changed,np.asarray(b.convert('RGB')))
