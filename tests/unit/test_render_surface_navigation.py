"""The shared builder preserves surface orientation and renderer routing."""
import importlib.util
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('render_nav_builder', ROOT/'tools/rooms/build_render_surface_navigation.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_shared_surface_selection_does_not_depend_on_room_name(tmp_path):
    mesh = tmp_path/'surface.glb'
    mesh.write_bytes(b'input is only resolved by this test')
    room = {'room_id': 'arbitrary_room', 'coordinate_system': {'up_axis': '+Y', 'linear_unit': 'meter'},
            'assets': [{'role': 'render_surface_mesh', 'path': 'surface.glb'}]}
    stage = builder.stage_for_manifest(room, tmp_path/'room.json')
    assert stage['collision_asset'] == stage['render_asset'] == str(mesh)
    assert stage['up'] == [0, 1, 0]
    room['coordinate_system']['up_axis'] = '+Z'
    with pytest.raises(ValueError, match='orientation'):
        builder.stage_for_manifest(room, tmp_path/'room.json')


def test_explicit_scan_orientation_uses_the_render_surface(tmp_path):
    stage_path = tmp_path/'scan.stage_config.json'
    stage_path.write_text(json.dumps({'render_asset': 'scan.glb', 'collision_asset': 'old_proxy.glb',
                                     'up': [0, 0, 1], 'front': [0, 1, 0]}))
    stage = builder.stage_for_manifest({}, tmp_path/'room.json', stage_path)
    assert stage['render_asset'] == stage['collision_asset'] == str(tmp_path/'scan.glb')
    assert stage['up'] == [0, 0, 1]
    assert stage['front'] == [0, 1, 0]


def test_native_navmesh_selection_preserves_the_declared_renderer():
    from avengine.capture.qa_plan_adapters import planning_adapter_for_room, selected_scene_reference
    for family, renderer in (('apartment', 'ue_spear'), ('kujiale', 'ue_spear'),
                             ('hm3d', 'habitat'), ('mp3d', 'habitat')):
        package = {'schema': 'avengine_qa_room_package_v1', 'room_id': 'arbitrary_room',
                   'family': family, 'renderer': renderer, 'walkable_space': {'kind': 'habitat_navmesh'}}
        assert planning_adapter_for_room({}, package) == 'habitat_native_navmesh'
        assert selected_scene_reference(package)['renderer'] == renderer
