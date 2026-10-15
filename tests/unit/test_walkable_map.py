"""Geometry/cache contracts; independent native admission remains separate."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from shapely.geometry import box
from avengine.rooms.walkable_map import WalkableMap, VERSION, SPECIES_RADII_M, LAYER_KEYS, source_identity, load_room_map
from avengine.rooms.walkable_map_builder import projected_clearance
from avengine.assets.roster_motion.reflect_navmesh import native_free_predicate

pytestmark = pytest.mark.fast_unit


def fixture_map(tmp_path, blocked=()):
    source = tmp_path/'stage.glb'
    source.write_bytes(b'fixture')
    a = np.ones((12, 12), bool)
    for z, x in blocked:
        a[z, x] = False
    meta = {'version': VERSION, 'room_id': 'room', 'species_radii_m': SPECIES_RADII_M,
            'source_identity': source_identity([source])}
    return WalkableMap([-.3, -.3], .05, 0., {s: a.copy() for s in SPECIES_RADII_M}, meta, np.zeros_like(a, float))


def save(m, path):
    np.savez(path, version=VERSION, origin_xz_m=m.origin, resolution_m=m.resolution,
             floor_reference_y_m=m.floor_reference, floor_y_m=m.floor_y,
             metadata_json=json.dumps(m.metadata), **{LAYER_KEYS[s]: a for s, a in m.layers.items()})


def test_negative_coordinates_bounds_and_nonfinite_are_blocked(tmp_path):
    m = fixture_map(tmp_path, [(0, 0)])
    assert not m.contains([-.299, -.299])
    assert m.contains([-.224, -.224])
    assert not m.contains([-.301, 0])
    assert not m.contains([.300001, 0])
    assert not m.contains([np.nan, 0])
    assert not m.contains([0, np.inf])


def test_supercover_rejects_thin_crossing_and_diagonal_corner(tmp_path):
    m = fixture_map(tmp_path, [(5, 5), (6, 5)])
    assert m.contains([-.22, -.025]) and m.contains([.22, -.025])
    assert not m.segment_free([-.22, -.025], [.22, -.025])
    # Endpoints are free but diagonal touches the corner of a blocked cell.
    assert m.contains([-.075, -.025]) and m.contains([-.025, -.075])
    assert not m.segment_free([-.075, -.025], [-.025, -.075])
    assert m.segment_free([-.225, -.225], [-.125, -.225])


def test_read_cache_rejects_stale_source_and_wrong_room(tmp_path, monkeypatch):
    m = fixture_map(tmp_path)
    path = tmp_path/'walkable_map_v1.npz'
    save(m, path)
    assert WalkableMap.load(path).contains([0, 0])
    with pytest.raises(ValueError, match='another room'):
        load_room_map('different', path=path)
    (tmp_path/'stage.glb').write_bytes(b'changed geometry')
    with pytest.raises(ValueError, match='changed'):
        WalkableMap.load(path)
    monkeypatch.setenv('AVENGINE_WALKABLE_MAP_ROOT', str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_room_map('missing')


def test_map_first_then_native_dynamic_and_camera(tmp_path):
    m = fixture_map(tmp_path, [(5, 5)])
    calls = []
    pf = SimpleNamespace(snap_point=lambda p: calls.append(p) or p,
                         distance_to_closest_obstacle=lambda p, r: .8)
    free = native_free_predicate(pf, box(-2, -2, 2, 2), 0., [2, 0, 2],
                                 walkable_map=m, obstacles=[box(.1, .1, .2, .2)])
    assert not free([-.025, -.025]) and not calls
    assert free([-.2, -.2]) and calls
    assert not free([.15, .15])
    assert not free.segment_is_free([-.2, -.025], [.2, -.025])
    close_camera = native_free_predicate(pf, box(-2, -2, 2, 2), 0., [0, 0, 0], walkable_map=m)
    assert not close_camera([-.2, -.2])


def test_all_height_projection_keeps_vertical_and_very_thin_triangles():
    # A wall in between conventional horizontal ray heights, degenerate in 2D.
    vertices = np.array([[0, .411, -1], [0, .419, 1], [0, .421, -1],
                         [-1, 0, -1], [1, 0, -1], [0, 0, 1],
                         [-1, 3, -1], [1, 3, -1], [0, 3, 1]])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    d, stats = projected_clearance(vertices, faces, 0., (-2, -2, 2, 2), np.array([[.1, 0], [.36, 0], [1., 0]]))
    assert d[0] == pytest.approx(.1)
    assert d[1] == pytest.approx(.36)
    assert np.isinf(d[2])
    assert stats['selected_original_nonfloor_triangles'] == 1
    assert stats['removed_original_ground_triangles'] == 1


def test_species_masks_must_be_nested(tmp_path):
    m = fixture_map(tmp_path)
    m.layers['cat'][0, 0] = False
    with pytest.raises(ValueError, match='nested'):
        WalkableMap(m.origin, .05, 0., m.layers, m.metadata)


def test_navmesh_vertical_offset_does_not_replace_the_actual_floor(tmp_path, monkeypatch):
    from avengine.assets.roster_motion import episode
    from avengine.assets import v7_phase2_floor
    from avengine.rooms.walkable_map_builder import build_room_map
    vertices = np.array([[-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2]], float)
    mesh = SimpleNamespace(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]))
    class Ray:
        MAX_QUERY_POINTS = 32
        def many(self, points):
            return np.zeros(len(points))
    class Floor:
        def __init__(self, *args):
            self.original_ray = Ray()
            self.native_ray_count = 0
            self.native_ray_agreements = []
        def __call__(self, point):
            self.native_ray_count += 1
            self.native_ray_agreements.append(0.)
            return 0.
    monkeypatch.setattr(episode, 'PhysicalFloor', Floor)
    monkeypatch.setattr(v7_phase2_floor, 'original_native_stage_mesh', lambda *a: (mesh, {}))
    stage, nav = tmp_path/'stage.glb', tmp_path/'stage.navmesh'
    stage.write_bytes(b'fixture'); nav.write_bytes(b'fixture')
    house = {'house': 'fixture', 'family': 'hm3d', 'input': {'visual_source': str(stage), 'navmesh_source': str(nav)}}
    room = {'room_label': 'room', 'measured_floor_y_m': 0., 'floor_polygon': box(-1, -1, 1, 1).__geo_interface__}
    pf = SimpleNamespace(snap_point=lambda p: np.asarray(p)+[0, .14, 0], distance_to_closest_obstacle=lambda p, r: 1.)
    result = build_room_map(SimpleNamespace(pathfinder=pf), house, room, code_commit='fixture', output=tmp_path/'map.npz')
    assert result.layers['human'].any()
    assert result.metadata['native_support_rays'] > 0
    assert WalkableMap.load(tmp_path/'map.npz').contains([0, 0])


def test_acceptance_manifest_configures_shared_room_cache(tmp_path, monkeypatch):
    from tools.timeline.run_v7_acceptance import configure_walkable_maps
    # setenv first so monkeypatch records and restores the original state after configure_walkable_maps
    # writes os.environ directly (otherwise the map root leaks into later tests).
    for key in ('AVENGINE_WALKABLE_MAP_ROOT', 'AVENGINE_REQUIRE_WALKABLE_MAP'):
        monkeypatch.setenv(key, 'unset'); monkeypatch.delenv(key)
    root = tmp_path/'maps'; target = root/'room'/'walkable_map_v1.npz'
    target.parent.mkdir(parents=True)
    m = fixture_map(tmp_path, [(5, 5)]); save(m, target)
    configure_walkable_maps({'walkable_map_root': str(root)})
    result = load_room_map('room', floor_reference=0.)
    assert result.path == str(target)
    assert not result.contains([-.025, -.025])
    with pytest.raises(FileNotFoundError):
        load_room_map('uncomputed')


def test_unconfigured_legacy_and_invalid_manifest_root(tmp_path, monkeypatch):
    from tools.timeline.run_v7_acceptance import configure_walkable_maps
    # setenv first so monkeypatch records and restores the original state after configure_walkable_maps
    # writes os.environ directly (otherwise the map root leaks into later tests).
    for key in ('AVENGINE_WALKABLE_MAP_ROOT', 'AVENGINE_REQUIRE_WALKABLE_MAP'):
        monkeypatch.setenv(key, 'unset'); monkeypatch.delenv(key)
    configure_walkable_maps({})
    assert load_room_map('room') is None
    with pytest.raises(FileNotFoundError, match='cache root is missing'):
        configure_walkable_maps({'walkable_map_root': str(tmp_path/'missing')})


def test_free_run_preserves_thin_wall_detection_and_clear_prefix_length(tmp_path):
    from avengine.assets.roster_motion.reflect_walk import _free_run
    pf = SimpleNamespace(snap_point=lambda p: p, distance_to_closest_obstacle=lambda p, r: .8)
    m = fixture_map(tmp_path, [(5, 5)])
    free = native_free_predicate(pf, box(-2, -2, 2, 2), 0., [2, 0, 2], walkable_map=m)
    # Both endpoints of the first .2m probe are free, with a blocked cell between.
    assert free([-.175, -.025]) and free([.025, -.025])
    assert _free_run(free, np.array([-.175, -.025]), 0., step=.2, max_m=.4) == 0.
    assert _free_run(free, np.array([-.225, -.225]), 0., step=.1, max_m=.8) == pytest.approx(.5)
