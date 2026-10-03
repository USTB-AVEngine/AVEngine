"""Analytic source-identity and coordinate tests for the MP3D adapter."""
import numpy as np
import pytest
import yaml
from pathlib import Path
from shapely.geometry import shape,box
from tools.rooms.room_selection.mp3d import read_house,load_scene
from tools.rooms.room_selection.measurements import region_geometry,layer_furniture
from tools.rooms.room_screening.geometry import classify_category

pytestmark = pytest.mark.fast_unit


def house_text():
    return """ASCII 1.1
L 0 0 - 0 0 0 0 0 0 0 0 0 0
R 0 0 0 0 b 1 1 0 0 0 0 1 2 2 2 0 0 0 0
C 5 3 floor 2 floor 0 0 0 0 0
C 8 4 table 3 table 0 0 0 0 0
O 42 0 5 0 0 0
O 7 0 8 0 0 0
S 0 0 0 F 0 0 0
V 0 0 F 0 0 0
V 1 0 F 2 0 0
V 2 0 F 2 2 0
V 3 0 F 0 2 0
P native_pano 0 0 0 1 2 1.5 0 0 0
"""


def parameters():
    spec=yaml.safe_load((Path(__file__).resolve().parents[2]/"tools/rooms/room_selection/thresholds.yaml").read_text())
    return {k:v["value"] for k,v in spec["parameters"].items()}


def test_house_rotation_native_surface_and_acquisition_identity(tmp_path):
    path=tmp_path/"scene.house";path.write_text(house_text())
    ann,regions,levels=read_house(path)
    assert ann[42]["category"]=="floor" and ann[42]["role"]==classify_category("floor")
    assert regions[0]["bbox_xz_m"]==[[0.0,-2.0],[1.0,0.0]]
    assert shape(regions[0]["native_floor_polygon"]).area==pytest.approx(4)
    assert regions[0]["native_viewpoints"][0]["camera_m"]==[1.0,1.5,-2.0]
    assert levels[0]["floor_y_m"]==0


def test_face_object_ids_and_floor_not_clipped_to_native_bbox(tmp_path):
    from plyfile import PlyData,PlyElement
    scene=tmp_path/"analytic";scene.mkdir();(scene/"analytic.house").write_text(house_text())
    vertices=np.array([(0,0,0),(2,0,0),(2,2,0),(0,2,0),(0.4,0.4,1),(0.9,0.4,1),(0.4,0.9,1)],dtype=[("x","f4"),("y","f4"),("z","f4")])
    faces=np.empty(4,dtype=[("vertex_indices","i4",(3,)),("object_id","i4")])
    faces["vertex_indices"]=[[0,1,2],[0,2,3],[4,5,6],[0,1,2]];faces["object_id"]=[42,42,7,999]
    PlyData([PlyElement.describe(vertices,"vertex"),PlyElement.describe(faces,"face")],text=False).write(scene/"analytic_semantic.ply")
    mesh=load_scene(scene);layers,members,bounds=region_geometry(mesh,0,parameters())
    assert members[42]["instance_id"]==42
    assert mesh.diagnostics["unmapped_faces"]==1
    assert len(layers)==1 and layers[0]["floor_y_m"]==0
    g=layers[0]["geometry"];assert g.area==pytest.approx(4)
    native=box(0,-2,1,0);assert g.difference(native).area==pytest.approx(2)
    assert bounds==[[0.0,-2.0],[2.0,0.0]]
    furniture=layer_furniture(mesh,0,0,1.5)
    assert len(furniture)==1 and furniture[0]["geometry"].area==pytest.approx(.125,abs=1e-6)


def test_basis_y_up_flag_preserves_native_bottom_left_orientation():
    import struct
    from tools.rooms.room_selection.demo_textures import decode_rgba_payload
    # An asymmetric two-row signal distinguishes the storage convention;
    # colour/alpha bytes remain exact in either branch.
    red=bytes([255,0,0,255]);blue=bytes([0,0,255,127])
    top_down,flag=decode_rgba_payload(struct.pack("<III",1,2,0)+red+blue)
    assert not flag and list(top_down.getdata())==[(255,0,0,255),(0,0,255,127)]
    bottom_up,flag=decode_rgba_payload(struct.pack("<III",1,2,1)+blue+red)
    assert flag and list(bottom_up.getdata())==[(255,0,0,255),(0,0,255,127)]


def test_colour_conflict_uses_source_legacy_box_only_for_display():
    from tools.rooms.room_selection.demo_render import house_layers, visual_bbox
    row = {"room_label": "R1", "floors": [],
           "native_region": {"ambiguous_colours": ["DD5C8F"]},
           "legacy_room": {"bbox_xz_m": [[1, -3], [8, 7]], "floor_y_m": 0},
           "metrics": None, "stage1": {"status": "review"}}
    assert visual_bbox(row) == ([[1, -3], [8, 7]], "legacy_reference")
    # The colour-conflict metadata has no bbox. The display must avoid both
    # a KeyError and treating the legacy registration box as measured floor.
    layers = house_layers({"family": "hm3d", "rows": [row]})
    assert len(layers) == 1 and layers[0]["geometry"].area == 70
    assert row["metrics"] is None and row["floors"] == []
    assert row["stage1"]["status"] == "review"



def test_reference_camera_aim_does_not_look_straight_down_at_room_centre():
    from tools.rooms.room_selection.demo_render import reference_target
    camera = [2., 1.5, 1.5]
    target = reference_target(camera, box(0, 0, 4, 3), 0., {"source_height_m": 1.2})
    assert np.linalg.norm(np.asarray(target)[[0, 2]]-np.asarray(camera)[[0, 2]]) > 2
    assert target[1] == 1.2
