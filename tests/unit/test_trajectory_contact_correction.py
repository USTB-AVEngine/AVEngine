import numpy as np
import pytest

from avengine.assets import qualification_geometry as qg
from avengine.rooms import conditioned_sampler as cs
from avengine.rooms.furniture_layout import clock_config


def _surface(cx, height, *, half_x=0.35, half_z=0.35):
    base = np.asarray([
        [cx - half_x, height, -half_z],
        [cx + half_x, height, -half_z],
        [cx + half_x, height, half_z],
        [cx - half_x, height, half_z],
    ], dtype=float)
    return base, np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def _non_flat_room():
    vertices = []
    triangles = []
    for cx, height in ((0.0, 0.5), (1.0, 0.8), (2.0, 1.1), (3.0, 1.4)):
        points, faces = _surface(cx, height)
        start = len(vertices)
        vertices.extend(points.tolist())
        triangles.extend((faces + start).tolist())
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int64)


def _foot_contact(actor_id, root_y, sole_y, extent):
    return {
        'actor_id': actor_id,
        'measurement': 'measured',
        'inputs': {'retained_native_readback': True},
        'frames': [{
            'frame_index': 0,
            'root_world_position_m': [0.0, root_y, 0.0],
            'sole_world_y_m': sole_y,
            'sole_footprint_extent_m': list(extent),
        }],
    }


def _actors():
    return [
        {
            'entity_instance_id': 'source1',
            'actor_id': 'source1',
            'source_slot_id': 'source1',
            'asset_id': 'human_blue',
            'entity_class': 'articulated_human',
            'timeline': {
                'local_anatomical_forward_axis': [1.0, 0.0, 0.0],
            },
            'emitter_binding': {'emitter_offset_m': [0.0, 1.6, 0.0]},
        },
        {
            'entity_instance_id': 'source2',
            'actor_id': 'source2',
            'source_slot_id': 'source2',
            'asset_id': 'human_green',
            'entity_class': 'articulated_human',
            'timeline': {
                'local_anatomical_forward_axis': [1.0, 0.0, 0.0],
            },
            'emitter_binding': {'emitter_offset_m': [0.0, 1.5, 0.0]},
        },
    ]


def _contact_document():
    identity = {
        'episode_id': 'fixture_hm3d_episode_v0',
        'room_id': 'hm3d_fixture_room',
        'scene_id': 'hm3d_fixture_room',
        'world_id': None,
    }
    foot_path = None
    return {
        'schema': 'avengine_c01r4_trajectory_contact_correction_v1',
        'query_support_under_footprint': 'per_trajectory_sample',
        'identity': identity,
        'room_visual_geometry': {
            'scene_glb': '/fixture/room.glb',
            'scene_dataset_config': '/fixture/room.scene_dataset_config.json',
            'geometry_version': 'fixture-hm3d-v1',
            'identity': identity,
        },
        'navigation_floor': {
            'floor_height_m': 0.0,
            'source': '/fixture/floor_reference.json',
            'navmesh': '/fixture/room.navmesh',
            'identity': identity,
        },
        'support_query': {
            'grid': 3,
            'coverage_fraction': 0.9,
            'support_statistic': 'support_top_q95_m',
        },
        'controls': [
            {
                'actor_id': 'source1',
                'asset_id': 'human_blue',
                'control': 'native_measured_sole',
                'foot_contact': _foot_contact(
                    'source1', 2.0, 1.9, [0.4, 0.4]
                ),
                'footprint_extent_m': [0.4, 0.4],
            },
            {
                'actor_id': 'source2',
                'asset_id': 'human_green',
                'control': 'native_measured_sole',
                'foot_contact': _foot_contact(
                    'source2', 2.0, 1.8, [0.4, 0.4]
                ),
                'footprint_extent_m': [0.4, 0.4],
            },
        ],
    }


def _install_mesh_loader(monkeypatch):
    vertices, triangles = _non_flat_room()
    calls = []

    def load_room_triangles(scene_glb, *, scene_dataset_config=None, **_):
        calls.append((scene_glb, scene_dataset_config))
        return vertices, triangles, {
            'frame': 'habitat_world',
            'vertex_count': len(vertices),
            'triangle_count': len(triangles),
            'source_sha256': 'fixture-mesh',
        }

    monkeypatch.setattr(qg, 'load_room_triangles', load_room_triangles)
    return calls


def _paths():
    return [
        np.asarray([
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ], dtype=float),
        np.asarray([
            [3.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
        ], dtype=float),
    ]


def test_trajectory_contact_queries_non_flat_support_per_point_and_caches(monkeypatch):
    calls = _install_mesh_loader(monkeypatch)
    paths = _paths()
    report = cs.apply_contact_correction(
        _actors(), paths,
        {'contact_correction': _contact_document()},
        room={'room_id': 'hm3d_fixture_room'},
    )

    assert calls == [
        ('/fixture/room.glb', '/fixture/room.scene_dataset_config.json'),
    ]
    assert report['per_trajectory_point'] is True
    assert report['query_count'] == 4
    assert report['cache_hit_count'] == 2
    assert np.allclose(paths[0][:, 1], [0.6, 0.9, 1.2])
    assert np.allclose(paths[1][:, 1], [1.6, 1.6, 1.6])
    source1 = report['applied'][0]
    assert source1['support_level_height_series_m'] == pytest.approx(
        [0.5, 0.8, 1.1]
    )
    assert source1['corrected_root_height_series_m'] == pytest.approx(
        [0.6, 0.9, 1.2]
    )
    assert all(
        row['prediction'] == 'cpu_prediction'
        and row['native_verification'] == 'not_run'
        for row in source1['support_query_evidence']
    )
    assert report['geometry']['identity']['world_id'] is None


def test_sample_routes_derives_emitter_rotation_and_body_from_corrected_points(monkeypatch):
    _install_mesh_loader(monkeypatch)
    original_paths = _paths()

    class NativeFixtureSpace:
        metadata = {
            'authority': 'fixture_retained_native_route_bank',
            'floor_height_m': 2.0,
        }

        def route_bank(self):
            return ({'route_id': 'source1'}, {'route_id': 'source2'})

        def bounds(self):
            return np.asarray([[-1.0, 1.0, -1.0], [4.0, 3.0, 1.0]])

    def retained_routes(*_args, **_kwargs):
        return [path.copy() for path in original_paths], {
            'authority': 'fixture_retained_native_route_bank',
            'selected_floor_height_m': 2.0,
        }

    monkeypatch.setattr(cs, '_native_routes', retained_routes)
    profile = {
        'anchor_indices': [0],
        'speech_motion': 'speaker_moving',
        'competitor_motion': 'still',
        'contact_correction': _contact_document(),
    }
    paths, rotations, moving, emitters, bodies, metadata = cs.sample_routes(
        NativeFixtureSpace(), _actors(), profile,
        clock_config(frame_count=3, frame_rate_hz=15, sample_rate_hz=16000),
        np.random.default_rng(9),
    )

    assert np.allclose(paths[0, :, 1], [0.6, 0.9, 1.2])
    assert np.allclose(paths[1, :, 1], [1.6, 1.6, 1.6])
    assert np.allclose(emitters[0, :, 1], paths[0, :, 1] + 1.6)
    assert np.allclose(bodies[0, :, 1], paths[0, :, 1] + 1.28)
    assert np.allclose(emitters[1, :, 1], paths[1, :, 1] + 1.5)
    assert np.allclose(bodies[1, :, 1], paths[1, :, 1] + 1.2)
    assert rotations.shape == (2, 3, 4)
    assert moving.shape == (2, 3)
    assert metadata['contact_correction']['per_trajectory_point'] is True


def test_trajectory_contact_refuses_to_infer_navigation_floor_from_root(monkeypatch):
    _install_mesh_loader(monkeypatch)
    document = _contact_document()
    del document['navigation_floor']
    with pytest.raises(cs.CandidateFailure, match='navigation_floor_registration'):
        cs.apply_contact_correction(
            _actors(), _paths(),
            {'contact_correction': document},
        )

def test_static_cpu_contact_offset_is_usable_without_fresh_native_capture():
    actor = _actors()[0]
    paths = [np.tile(np.asarray([0.0, 9.0, 0.0]), (2, 1))]
    profile = {
        'contact_correction': {
            'schema': 'avengine_c05_root_contact_correction_v1',
            'controls': [{
                'actor_id': 'source1',
                'asset_id': 'human_blue',
                'control': 'cpu_measured_sole',
                'contact_offset': {
                    'measurement': 'measured',
                    'evidence_kind': 'cpu_reconstruction_from_baked_clips',
                    'root_above_contact_m': 0.001965,
                },
                'correction': {
                    'support_level_height_m': 0.5,
                    'prediction': 'cpu_prediction',
                },
            }],
        },
    }

    report = cs.apply_contact_correction([actor], paths, profile)

    assert np.allclose(paths[0][:, 1], 0.501965)
    assert report['applied'][0]['evidence_kind'] == (
        'cpu_reconstruction_from_baked_clips'
    )

