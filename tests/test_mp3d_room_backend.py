import struct
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

from tools.rooms.room_screening.mp3d_geometry import load_mp3d_scene, read_house
from tools.rooms.room_selection.geometry import floor_windows
from tools.rooms.room_selection.cpu_overhead import raster, CPUOverhead
from tools.rooms.room_selection.media import black_metric

pytestmark = pytest.mark.fast_unit

def fixture_scene(tmp_path):
    scene=tmp_path/'sample';scene.mkdir()
    # Tiny metadata bbox deliberately cannot own the floor measurement.
    house='C 0 0 floor 0 floor\nO 0 7 0\nR 7 0 0 0 b 0 0 2 0 0 2 1 1 4 2 0 0 0 0\n'
    (scene/'sample.house').write_text(house)
    verts=[(0.,0.,2.),(4.,0.,2.),(4.,3.,2.),(0.,3.,2.)]
    faces=[(0,1,2),(0,2,3)]
    header=b'ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face 2\nproperty list uchar int vertex_indices\nproperty int object_id\nend_header\n'
    payload=header+b''.join(struct.pack('<fffBBB',*v,100,100,100) for v in verts)+b''.join(struct.pack('<Biiii',3,*f,0) for f in faces)
    (scene/'sample_semantic.ply').write_bytes(payload)
    return scene

def test_z_up_floor_identity_and_scope_survive_adapter(tmp_path):
    scene=fixture_scene(tmp_path);original=(scene/'sample_semantic.ply').read_bytes()
    mesh=load_mp3d_scene(scene)
    layers=floor_windows(mesh.ground[7],{'floor_height_separation_m':.3,'projection_precision_m':.001})
    assert len(layers)==1
    assert layers[0]['floor_y_m']==2
    assert layers[0]['geometry'].area==pytest.approx(12)
    assert layers[0]['geometry'].bounds==(0.,-3.,4.,0.)
    assert mesh.annotations[0]['region_id']==7
    assert mesh.regions[7]['bbox_xz_m']==[[0.,-1.],[1.,0.]]
    assert (scene/'sample_semantic.ply').read_bytes()==original

def test_conflicting_instance_region_is_rejected(tmp_path):
    p=tmp_path/'scan.house'
    p.write_text('C 0 0 floor 0 floor\nO 0 7 0\nO 0 8 0\nR 7 0 0 0 b 0 0 2 0 0 2 1 1 4 2 0 0 0 0\nR 8 0 0 0 b 0 0 2 0 0 2 1 1 4 2 0 0 0 0\n')
    with pytest.raises(ValueError,match='conflicting MP3D'):read_house(p)

def test_cpu_overhead_keeps_black_texture_and_unknown_texture_distinct():
    # A floor square in Habitat XZ should fill the viewport without a Y/Z swap.
    triangles=np.array([[[0,2,0],[4,2,0],[4,2,4]],[[0,2,0],[4,2,4],[0,2,4]]],float)
    uv=np.zeros((2,3,2));texture=np.zeros((2,2,3),np.uint8)
    solid=np.empty((0,3),np.uint8)
    rgb=np.full((16,16,3),255,np.uint8);height=np.full((16,16),-np.inf);known=np.ones((16,16),bool)
    raster(triangles,uv,texture,solid,4.,2.,2.,1.7,3.5,rgb,height,known)
    assert np.all(rgb==0) and np.all(known)
    unknown=np.ones((16,16),bool);height[:]=-np.inf
    raster(triangles,uv,np.empty((0,0,3),np.uint8),solid,4.,2.,2.,1.7,3.5,rgb,height,unknown)
    assert not np.any(unknown)

def test_overhead_ceiling_does_not_hide_floor_holes():
    # Ceiling at Y=5 is outside the explicit camera-height slab.
    t=np.array([[[0,5,0],[4,5,0],[4,5,4]],[[0,5,0],[4,5,4],[0,5,4]]],float)
    rgb=np.zeros((16,16,3),np.uint8);height=np.full((16,16),-np.inf);known=np.ones((16,16),bool)
    raster(t,np.zeros((2,3,2)),np.full((1,1,3),255,np.uint8),np.empty((0,3),np.uint8),4.,2.,2.,1.7,3.5,rgb,height,known)
    assert np.all(rgb==0) and not np.isfinite(height).any()


def test_cpu_camera_mask_and_black_pixels_agree_on_translated_scope(tmp_path):
    # Half a translated room has measured white scan geometry; the other half
    # is a true scan hole. Camera metadata must project both halves correctly.
    renderer=CPUOverhead.__new__(CPUOverhead)
    renderer.basis_diagnostics=[];renderer.basis_error=None
    triangles=np.array([[[12,2,-7],[14,2,-7],[14,2,-3]],[[12,2,-7],[14,2,-3],[12,2,-3]]],float)
    renderer.items=[(triangles,np.zeros((2,3,2)),np.full((1,1,3),255,np.uint8),np.empty((0,3),np.uint8))]
    p={'floor_height_separation_m':.3,'camera_height_m':1.5,'black_pixel_channel_lt':8}
    scope=box(12,-7,16,-3)
    entry,record,image,diag=renderer.render(scope,2,tmp_path/'overhead.png',p,size=512)
    result=black_metric(scope,2,entry,record,image,p)
    assert result['status']=='measured'
    assert result['black_fraction']==pytest.approx(.5,abs=.005)
    assert diag['black_measurement_usable']
    assert diag['covered_pixels_in_scope']>0
