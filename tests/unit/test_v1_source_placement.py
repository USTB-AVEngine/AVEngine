from __future__ import annotations

import numpy as np
import pytest
from avengine.rooms.source_placement import (
    SourcePlacementError,
    bind_support_catalog_requests,
    plan_source_placement,
    SOURCE_PLACEMENT_SCHEMA,
    apply_room_collision,
    parse_support_surfaces,
    plan_static_source_placements,
    room_collision_report,
    support_catalog_surface_kinds,
)

def registry_record(surface='floor', normal=(0.0,1.0,0.0)):
    basis_v = [0.0,1.0,0.0] if tuple(normal)==(0.0,0.0,1.0) else ([0.0,0.0,1.0] if tuple(normal)==(0.0,-1.0,0.0) else [0.0,0.0,-1.0])
    return {'schema':'avengine_source_asset_runtime_registry_v1','assets':[{'asset_id':'fixture_device','revision':'fixture_v1','entity_class':'rigid_object','default_emitter_anchor_id':'speaker','emitter_anchors':[{'anchor_id':'speaker','anchor_type':'object_speaker','offset_m':[0.1,0.2,0.0],'offset_space':'final_scaled_asset_root'}],'runtime_backends':{'habitat':{'resting_pose':{'attachment_surface':surface,'base_plane_offset_m':0.1,'plane_normal_m':list(normal),'plane_basis_u_m':[1.0,0.0,0.0],'plane_basis_v_m':basis_v,'footprint_extent_m':[0.4,0.2],'height_m':0.5,'measured_plane':surface+'_base'}}}}]}

def visual_geometry(normal=(0.0,1.0,0.0)):
    if tuple(normal) == (0.0, 0.0, 1.0):
        vertices=[[-1.0,-0.2,0.0],[1.0,-0.2,0.0],[1.0,1.8,0.0],[-1.0,1.8,0.0]]
    else:
        vertices=[[-1.0,0.8,-1.0],[1.0,0.8,-1.0],[1.0,0.8,1.0],[-1.0,0.8,1.0]]
    return {'authority':'visual_geometry','geometry_id':'room_visual_fixture_v1','source_ref':'fixture://hm3d/tabletop','vertices_m':vertices,'triangles':[[0,1,2],[0,2,3]]}

def surface(kind='tabletop',normal=(0.0,1.0,0.0)):
    basis_v=[0.0,1.0,0.0] if tuple(normal)==(0.0,0.0,1.0) else ([0.0,0.0,1.0] if tuple(normal)==(0.0,-1.0,0.0) else [0.0,0.0,-1.0])
    return {'support_surfaces':[{'surface_id':'table_01','surface_kind':kind,'room_id':'room_fixture','origin_m':[0.0,0.8,0.0],'normal_m':list(normal),'basis_u_m':[1.0,0.0,0.0],'basis_v_m':basis_v,'bounds_u_m':[-1.0,1.0],'bounds_v_m':[-1.0,1.0],'geometry_ref':{'authority':'visual_geometry','geometry_id':'room_visual_fixture_v1','triangle_indices':[0,1]}}]}

def request(surface_id='table_01',index=0): return {'instance_id':'source1','asset_id':'fixture_device','support_surface_id':surface_id,'candidate_index':index,'yaw_deg':0.0}
def config(max_candidates=4): return {'normal_tolerance_deg':1.0,'plane_tolerance_m':1e-6,'candidate_search':{'grid_step_m':0.25,'max_candidates':max_candidates,'edge_margin_m':0.05}}

def test_tabletop_planning_uses_support_plane_and_emitter_transform():
    result=plan_source_placement(registry_record(),{'room_id':'room_fixture'},surface(),visual_geometry(),request(),config=config())
    assert result['status']=='planned'; assert result['qualification_status']=='not_run'; assert result['native_execution']=='not_run'; assert result['support_identity']['surface_id']=='table_01'; assert result['root_transform']['translation_m'][1]==pytest.approx(0.7); assert result['emitter_transform']['position_m'][1]==pytest.approx(0.9)

def test_candidate_lookup_is_bounded_and_footprint_aware():
    result=plan_source_placement(registry_record(),{'room_id':'room_fixture'},surface(),visual_geometry(),request(index=2),config=config(max_candidates=3)); assert result['candidate']['count']==3
    with pytest.raises(SourcePlacementError) as caught: plan_source_placement(registry_record(),{'room_id':'room_fixture'},surface(),visual_geometry(),request(index=3),config=config(max_candidates=3))
    assert caught.value.code=='candidate_index_invalid'

def test_missing_support_surface_rejects_without_floor_fallback():
    with pytest.raises(SourcePlacementError) as caught: plan_source_placement(registry_record(),{'room_id':'room_fixture'},{},visual_geometry(),request(),config=config())
    assert caught.value.code=='support_surface_data_missing'

def test_acoustic_proxy_is_never_visual_support():
    bad={'support_surfaces':[{**surface()['support_surfaces'][0],'geometry_ref':{'authority':'acoustic_proxy','geometry_id':'room_visual_fixture_v1','triangle_indices':[0]}}]}
    with pytest.raises(SourcePlacementError) as caught: plan_source_placement(registry_record(),{'room_id':'room_fixture'},bad,visual_geometry(),request(),config=config())
    assert caught.value.code=='acoustic_proxy_not_visual_geometry'

def test_missing_registry_plane_data_rejects_instead_of_converting_tilt():
    record=registry_record(); del record['assets'][0]['runtime_backends']['habitat']['resting_pose']['plane_normal_m']
    with pytest.raises(SourcePlacementError) as caught: plan_source_placement(record,{'room_id':'room_fixture'},surface(),visual_geometry(),request(),config=config())
    assert caught.value.code=='missing_registry_plane_normal'

def test_wall_and_ceiling_work_when_explicit_data_exists():
    for kind,normal in (('wall',(0.0,0.0,1.0)),('ceiling',(0.0,-1.0,0.0))):
        result=plan_source_placement(registry_record(surface=kind,normal=normal),{'room_id':'room_fixture'},surface(kind,normal),visual_geometry(normal),request(),config=config()); assert result['status']=='planned'; assert result['support_identity']['surface_kind']==kind

def test_batch_api_keeps_rejection_per_instance():
    result=plan_static_source_placements(registry_record(),{'room_id':'room_fixture'},surface(),visual_geometry(),[request(),{'instance_id':'source2','asset_id':'fixture_device','support_surface_id':'missing'}],config=config())
    assert result['status']=='partial'; assert [row['status'] for row in result['instances']]==['planned','rejected']; assert result['instances'][1]['reason']['code']=='support_surface_unresolved'

def test_non_static_asset_rejects():
    record=registry_record(); record['assets'][0]['entity_class']='articulated_animal'
    with pytest.raises(SourcePlacementError) as caught: plan_source_placement(record,{'room_id':'room_fixture'},surface(),visual_geometry(),request(),config=config())
    assert caught.value.code=='static_asset_required'

def test_asset_geometry_can_supply_measured_mount_footprint_and_plane():
    record=registry_record(surface='wall', normal=(0.0, 0.0, 1.0))
    pose=record['assets'][0]['runtime_backends']['habitat']['resting_pose']
    del pose['footprint_extent_m']
    del pose['plane_normal_m']
    del pose['plane_basis_u_m']
    del pose['plane_basis_v_m']
    req=request()
    req['asset_geometry']={
        'source_ref':'fixture://asset/finalized.glb',
        'base_plane_offset_m':0.2,
        'footprint_extent_m':[0.4,0.2],
        'plane_normal_m':[0.0,0.0,1.0],
        'plane_basis_u_m':[1.0,0.0,0.0],
        'plane_basis_v_m':[0.0,1.0,0.0],
    }
    result=plan_source_placement(record,{'room_id':'room_fixture'},surface('wall',(0.0,0.0,1.0)),visual_geometry((0.0,0.0,1.0)),req,config=config())
    assert result['status']=='planned'
    assert result['asset_resting_pose']['base_plane_offset_m']==pytest.approx(0.2)
    assert result['asset_resting_pose']['footprint_extent_m']==pytest.approx([0.4,0.2])


def _measured_fixture_geometry():
    return {
        'source_ref': 'fixture://asset/finalized.glb',
        'bounds_min_m': [-0.2, 0.0, -0.1],
        'bounds_max_m': [0.2, 0.5, 0.1],
    }


def _joint_config():
    return {
        'normal_tolerance_deg': 1.0,
        'plane_tolerance_m': 1e-6,
        'min_inter_instance_gap_m': 0.0,
        'candidate_search': {
            'grid_step_m': 0.1,
            'max_candidates': 8,
            'edge_margin_m': 0.05,
        },
    }


def test_joint_batch_reselects_implicit_candidate_and_keeps_room_collision_unknown():
    first = request()
    first['asset_geometry'] = _measured_fixture_geometry()
    second = request()
    second.pop('candidate_index')
    second['instance_id'] = 'source2'
    second['asset_geometry'] = _measured_fixture_geometry()
    result = plan_static_source_placements(
        registry_record(),
        {'room_id': 'room_fixture'},
        surface(),
        visual_geometry(),
        [first, second],
        config=_joint_config(),
    )
    assert result['status'] == 'planned'
    rows = result['instances']
    assert [row['status'] for row in rows] == ['planned', 'planned']
    assert rows[0]['candidate']['selected_index'] == 0
    assert rows[1]['candidate']['selected_index'] == 2
    assert rows[1]['candidate']['selection_mode'] == 'bounded_joint_nonoverlap'
    assert rows[1]['candidate']['rejected_indices'] == [0, 1]
    assert rows[1]['clearance']['inter_instance_aabb']['status'] == 'pass'
    assert rows[1]['clearance']['room_collision']['status'] == 'not_run'
    assert result['joint_candidate_selection']['room_collision'] == 'not_run'


def test_joint_batch_rejects_explicit_overlapping_candidate_without_silent_reselection():
    first = request()
    first['asset_geometry'] = _measured_fixture_geometry()
    second = request()
    second['instance_id'] = 'source2'
    second['candidate_index'] = 0
    second['asset_geometry'] = _measured_fixture_geometry()
    result = plan_static_source_placements(
        registry_record(),
        {'room_id': 'room_fixture'},
        surface(),
        visual_geometry(),
        [first, second],
        config=_joint_config(),
    )
    assert result['status'] == 'partial'
    assert result['instances'][0]['status'] == 'planned'
    rejected = result['instances'][1]
    assert rejected['status'] == 'rejected'
    assert rejected['reason']['code'] == 'explicit_candidate_conflict'
    assert rejected['candidate']['requested_index'] == 0
    assert rejected['candidate']['selected_index'] is None
    assert rejected['clearance']['inter_instance_aabb']['status'] == 'fail'
    assert rejected['clearance']['room_collision']['status'] == 'not_run'


def test_existing_planned_rows_are_checked_by_real_world_aabb():
    first = request()
    first['asset_geometry'] = _measured_fixture_geometry()
    existing = plan_source_placement(
        registry_record(),
        {'room_id': 'room_fixture'},
        surface(),
        visual_geometry(),
        first,
        config=_joint_config(),
    )
    second = request()
    second.pop('candidate_index')
    second['instance_id'] = 'source2'
    second['asset_geometry'] = _measured_fixture_geometry()
    result = plan_static_source_placements(
        registry_record(),
        {'room_id': 'room_fixture'},
        surface(),
        visual_geometry(),
        [second],
        config=_joint_config(),
        existing_placements=[existing],
    )
    assert result['status'] == 'planned'
    assert result['instances'][0]['candidate']['selected_index'] == 2
    assert result['joint_candidate_selection']['existing_placement_count'] == 1


def test_planned_transform_is_so3_and_quaternion_matches_emitter():
    for kind, normal in (('tabletop', (0.0, 1.0, 0.0)), ('wall', (0.0, 0.0, 1.0))):
        result = plan_source_placement(
            registry_record(surface=kind, normal=normal),
            {'room_id': 'room_fixture'},
            surface(kind, normal),
            visual_geometry(normal),
            request(),
            config=config(),
        )
        matrix = np.asarray(result['root_transform']['matrix_row_major'], dtype=float).reshape(4, 4)[:3, :3]
        quaternion = np.asarray(result['root_transform']['rotation_xyzw'], dtype=float)
        reconstructed = np.asarray([
            [1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2),
             2 * (quaternion[0] * quaternion[1] - quaternion[2] * quaternion[3]),
             2 * (quaternion[0] * quaternion[2] + quaternion[1] * quaternion[3])],
            [2 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
             1 - 2 * (quaternion[0] ** 2 + quaternion[2] ** 2),
             2 * (quaternion[1] * quaternion[2] - quaternion[0] * quaternion[3])],
            [2 * (quaternion[0] * quaternion[2] - quaternion[1] * quaternion[3]),
             2 * (quaternion[1] * quaternion[2] + quaternion[0] * quaternion[3]),
             1 - 2 * (quaternion[0] ** 2 + quaternion[1] ** 2)],
        ])
        assert np.linalg.det(matrix) == pytest.approx(1.0)
        assert np.max(np.abs(matrix.T @ matrix - np.eye(3))) < 1e-12
        assert np.linalg.norm(quaternion) == pytest.approx(1.0)
        assert np.max(np.abs(reconstructed - matrix)) < 1e-12
        emitter = np.asarray(result['emitter_transform']['position_m'], dtype=float)
        root = np.asarray(result['root_transform']['translation_m'], dtype=float)
        offset = np.asarray(result['emitter']['offset_m'], dtype=float)
        assert emitter == pytest.approx(root + matrix @ offset)


def test_reflected_asset_basis_is_rejected_before_transform():
    record = registry_record()
    pose = record['assets'][0]['runtime_backends']['habitat']['resting_pose']
    del pose['plane_normal_m']
    del pose['plane_basis_u_m']
    del pose['plane_basis_v_m']
    req = request()
    req['asset_geometry'] = {
        **_measured_fixture_geometry(),
        'plane_normal_m': [0.0, 1.0, 0.0],
        'plane_basis_u_m': [1.0, 0.0, 0.0],
        'plane_basis_v_m': [0.0, 0.0, 1.0],
    }
    with pytest.raises(SourcePlacementError) as caught:
        plan_source_placement(
            record,
            {'room_id': 'room_fixture'},
            surface(),
            visual_geometry(),
            req,
            config=config(),
        )
    assert caught.value.code == 'registry_plane_basis_not_right_handed'


def support_catalog_fixture():
    """A support catalog shaped like the real T06 one: surfaces plus measurements."""
    return {
        'schema': 'avengine_support_surface_catalog_v1',
        'status': 'research_candidate',
        'room': {'room_id': 'room_fixture'},
        'layout': surface(),
        'visual_geometry': visual_geometry(),
        'asset_visual_geometry_measurements': {
            'fixture_device': {
                'asset_id': 'fixture_device',
                'support_kind': 'tabletop',
                'measured_from': 'finalized.glb',
                'bounds_min_m': [-0.2, 0.0, -0.1],
                'bounds_max_m': [0.2, 0.5, 0.1],
                'footprint_extent_m': [0.4, 0.2],
                'plane_normal_m': [0.0, 1.0, 0.0],
                'plane_basis_u_m': [1.0, 0.0, 0.0],
                'plane_basis_v_m': [0.0, 0.0, -1.0],
                'base_plane_offset_m': 0.1,
            },
        },
    }


def test_catalog_binder_supplies_the_measured_geometry_the_planner_needs():
    record = registry_record()
    del record['assets'][0]['runtime_backends']['habitat']['resting_pose']['plane_normal_m']
    catalog = support_catalog_fixture()
    with pytest.raises(SourcePlacementError) as caught:
        plan_source_placement(record, {'room_id': 'room_fixture'}, surface(),
                              visual_geometry(), request(), config=config())
    assert caught.value.code == 'missing_registry_plane_normal'
    bound = bind_support_catalog_requests([request()], catalog, registry=record)
    assert bound[0]['asset_geometry']['plane_normal_m'] == [0.0, 1.0, 0.0]
    assert bound[0]['asset_revision'] == 'fixture_v1'
    result = plan_source_placement(record, {'room_id': 'room_fixture'}, surface(),
                                   visual_geometry(), bound[0], config=config())
    assert result['status'] == 'planned'


def test_catalog_binder_does_not_overwrite_an_explicit_request_geometry():
    catalog = support_catalog_fixture()
    explicit = dict(request())
    explicit['asset_geometry'] = {'plane_normal_m': [0.0, 1.0, 0.0], 'marker': 'caller'}
    bound = bind_support_catalog_requests([explicit], catalog)
    assert bound[0]['asset_geometry']['marker'] == 'caller'


def test_catalog_binder_reports_an_unmeasured_asset_instead_of_guessing():
    catalog = support_catalog_fixture()
    catalog['asset_visual_geometry_measurements'] = {'other_asset': {'bounds_min_m': [0, 0, 0]}}
    with pytest.raises(SourcePlacementError) as caught:
        bind_support_catalog_requests([request()], catalog)
    assert caught.value.code == 'asset_geometry_unmeasured'
    unbound = bind_support_catalog_requests([request()], catalog, strict=False)
    assert 'asset_geometry' not in unbound[0]


def test_catalog_surface_kinds_expose_the_legal_mount_surface_per_asset():
    assert support_catalog_surface_kinds(support_catalog_fixture()) == {'fixture_device': 'tabletop'}


def test_static_rigid_pose_is_one_legal_transform_for_every_frame():
    """A static source has no per-frame pose, so the single planned transform has
    to be a proper rigid motion and has to be reproducible call after call."""
    catalog = support_catalog_fixture()
    bound = bind_support_catalog_requests([request()], catalog, registry=registry_record())
    first = plan_source_placement(registry_record(), {'room_id': 'room_fixture'}, surface(),
                                  visual_geometry(), bound[0], config=config())
    second = plan_source_placement(registry_record(), {'room_id': 'room_fixture'}, surface(),
                                   visual_geometry(), bound[0], config=config())
    assert first['root_transform'] == second['root_transform']
    matrix = np.asarray(first['root_transform']['matrix_row_major'], dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    quaternion = np.asarray(first['root_transform']['rotation_xyzw'], dtype=float)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.isfinite(matrix))
    # the same rigid motion carries the emitter, frame after frame
    emitter_local = np.asarray(first['emitter']['offset_m'], dtype=float)
    expected = rotation @ emitter_local + matrix[:3, 3]
    assert np.allclose(expected, first['emitter_transform']['position_m'], atol=1e-9)
    for _ in range(150):
        assert first['emitter_transform']['position_m'] == second['emitter_transform']['position_m']


def room_mesh():
    """A 4x4 m floor slab at y=0 plus a wall panel at x=1."""
    vertices = [
        [-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0],
        [1.0, 0.0, -2.0], [1.0, 0.0, 2.0], [1.0, 3.0, 2.0], [1.0, 3.0, -2.0],
    ]
    triangles = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    return {"representation": "real_surface_mesh", "source": "test_fixture",
            "source_to_common_frame": "identity",
            "coordinate_frame": {"up_axis": "+Y", "handedness": "right"},
            "vertices": vertices, "triangles": triangles}


def placed(instance_id, low, high, *, status="planned"):
    return {"instance_id": instance_id, "asset_id": "fixture_device", "status": status,
            "support_identity": {"surface_id": "table_01", "surface_kind": "tabletop"},
            "asset_bounds": {"world_aabb_min_m": list(low), "world_aabb_max_m": list(high)},
            "clearance": {"status": "partial",
                          "inter_instance_aabb": {"status": "pass", "checked_against": ["other"],
                                                  "overlap_conflicts": []},
                          "room_collision": {"status": "not_run", "reason": "not supplied"}}}


def test_a_placement_clear_of_the_room_mesh_passes_the_collision_query():
    rows = [placed("source1", [-0.5, 0.5, -0.5], [-0.3, 0.7, -0.3])]
    report = room_collision_report(rows, room_geometry=room_mesh(),
                                   config={"penetration_tolerance_m": 0.01}, room_id="room")
    assert report["status"] == "pass"
    assert report["rows"][0]["intersecting_triangle_count"] == 0
    assert report["geometry"]["representation"] == "real_surface_mesh"
    assert report["geometry"]["source_to_common_frame"] == "identity"


def test_a_placement_inside_the_wall_fails_with_the_triangles_that_reach_in():
    rows = [placed("source1", [0.9, 0.5, -0.2], [1.2, 0.8, 0.2])]
    report = room_collision_report(rows, room_geometry=room_mesh(),
                                   config={"penetration_tolerance_m": 0.01}, room_id="room")
    assert report["status"] == "fail"
    row = report["rows"][0]
    assert row["status"] == "fail"
    assert row["intersecting_triangle_count"] > 0
    assert row["intersecting_triangle_indices"]


def test_resting_contact_on_a_surface_is_not_read_as_a_collision():
    """The base face sits exactly on the floor slab; only the tolerance separates
    resting contact from penetration."""
    rows = [placed("source1", [-0.5, 0.0, -0.5], [-0.3, 0.2, -0.3])]
    touching = room_collision_report(rows, room_geometry=room_mesh(),
                                     config={"penetration_tolerance_m": 0.0}, room_id="room")
    assert touching["rows"][0]["status"] == "fail"
    resting = room_collision_report(rows, room_geometry=room_mesh(),
                                    config={"penetration_tolerance_m": 0.01}, room_id="room")
    assert resting["rows"][0]["status"] == "pass"


def test_a_deeply_sunk_placement_still_fails_at_the_same_tolerance():
    rows = [placed("source1", [-0.5, -0.05, -0.5], [-0.3, 0.2, -0.3])]
    report = room_collision_report(rows, room_geometry=room_mesh(),
                                   config={"penetration_tolerance_m": 0.01}, room_id="room")
    assert report["rows"][0]["status"] == "fail"


def test_a_proxy_mesh_is_refused_rather_than_used_to_decide():
    geometry = {**room_mesh(), "representation": "acoustic_proxy"}
    report = room_collision_report([placed("source1", [-0.5, 0.5, -0.5], [-0.3, 0.7, -0.3])],
                                   room_geometry=geometry,
                                   config={"penetration_tolerance_m": 0.01}, room_id="room")
    assert report["status"] == "not_run"
    assert "proxy" in report["rows"][0]["reason"]


def test_without_room_geometry_every_row_stays_not_run():
    report = room_collision_report([placed("source1", [-0.5, 0.5, -0.5], [-0.3, 0.7, -0.3])],
                                   room_geometry={}, config={"penetration_tolerance_m": 0.01})
    assert report["status"] == "not_run"
    assert report["rows"][0]["status"] == "not_run"


def test_a_rejected_row_is_not_given_a_collision_verdict():
    report = room_collision_report([placed("source1", [0, 0, 0], [1, 1, 1], status="rejected")],
                                   room_geometry=room_mesh(),
                                   config={"penetration_tolerance_m": 0.01})
    assert report["rows"][0]["status"] == "not_run"


def test_an_asset_smaller_than_the_tolerance_is_not_silently_cleared():
    rows = [placed("source1", [-0.5, 0.5, -0.5], [-0.49, 0.51, -0.49])]
    report = room_collision_report(rows, room_geometry=room_mesh(),
                                   config={"penetration_tolerance_m": 0.05}, room_id="room")
    assert report["rows"][0]["status"] == "not_run"
    assert "interior box" in report["rows"][0]["reason"]


def test_a_negative_tolerance_is_refused():
    with pytest.raises(SourcePlacementError) as caught:
        room_collision_report([], room_geometry=room_mesh(),
                              config={"penetration_tolerance_m": -0.01})
    assert caught.value.code == "penetration_tolerance_invalid"


def test_apply_room_collision_folds_the_verdict_into_the_plan():
    plan = {"schema": SOURCE_PLACEMENT_SCHEMA, "status": "planned",
            "instances": [placed("source1", [-0.5, 0.5, -0.5], [-0.3, 0.7, -0.3]),
                          placed("source2", [0.9, 0.5, -0.2], [1.2, 0.8, 0.2])]}
    report = room_collision_report(plan["instances"], room_geometry=room_mesh(),
                                   config={"penetration_tolerance_m": 0.01}, room_id="room")
    merged = apply_room_collision(plan, report)
    first, second = merged["instances"]
    assert first["clearance"]["room_collision"]["status"] == "pass"
    assert first["clearance"]["status"] == "pass"
    assert second["clearance"]["room_collision"]["status"] == "fail"
    assert second["clearance"]["status"] == "fail"
    assert merged["room_collision"]["status"] == "fail"
    # the original plan is not mutated
    assert plan["instances"][0]["clearance"]["room_collision"]["status"] == "not_run"


def measured_surface_fixture(derivation="native_metric_depth_backprojection_plane_fit"):
    """A surface fitted from its own capture, with no index into a shared array."""
    row = dict(surface()['support_surfaces'][0])
    row.pop('geometry_ref')
    row['surface_id'] = 'table_measured'
    row['source'] = {'geometry_derivation': derivation,
                     'capture_root': '/capture/attempt_01/capture',
                     'visual_depth_source': '/capture/attempt_01/capture/depth.npy',
                     'frame_index': 0}
    return {'support_surfaces': [row]}


def test_a_surface_fitted_from_its_own_capture_is_accepted_and_planned():
    layout = measured_surface_fixture()
    parsed = parse_support_surfaces(layout, room_id='room_fixture')
    assert parsed[0]['geometry_ref']['authority'] == 'visual_geometry'
    assert parsed[0]['geometry_ref']['triangle_indices'] is None
    assert parsed[0]['geometry_ref']['cross_check'] == 'own_capture_fit_no_shared_triangle_array'
    result = plan_source_placement(
        registry_record(), {'room_id': 'room_fixture'}, layout, visual_geometry(),
        request(surface_id='table_measured'), config=config())
    assert result['status'] == 'planned'
    evidence = next(item for item in result['planning_evidence']
                    if item['kind'] == 'visual_geometry')
    assert evidence['status'] == 'observed_on_its_own_capture'
    assert evidence['surface_geometry_cross_check'] == 'own_capture_fit_no_shared_triangle_array'
    assert evidence['surface_visual_depth_source'].endswith('depth.npy')


def test_a_surface_fitted_on_acoustic_geometry_is_still_refused():
    layout = measured_surface_fixture(derivation='acoustic_proxy_plane_fit')
    with pytest.raises(SourcePlacementError) as caught:
        parse_support_surfaces(layout, room_id='room_fixture')
    assert caught.value.code == 'acoustic_proxy_not_visual_geometry'


def test_a_surface_naming_no_visual_capture_is_refused():
    layout = measured_surface_fixture()
    layout['support_surfaces'][0]['source'] = {'geometry_derivation': 'hand_authored'}
    with pytest.raises(SourcePlacementError) as caught:
        parse_support_surfaces(layout, room_id='room_fixture')
    assert caught.value.code == 'visual_support_geometry_required'


def test_a_surface_with_triangle_indices_is_still_cross_checked():
    layout = surface()
    layout['support_surfaces'][0]['bounds_u_m'] = [-0.05, 0.05]
    with pytest.raises(SourcePlacementError) as caught:
        plan_source_placement(registry_record(), {'room_id': 'room_fixture'}, layout,
                              visual_geometry(), request(), config=config())
    assert caught.value.code == 'support_surface_bounds_mismatch'


def test_a_support_normal_offset_seats_the_asset_on_the_surface_that_is_there():
    """A fitted plane runs through a slab, not along its face. The request states
    how far the measured surface sits from the fit, along the support normal."""
    flat = plan_source_placement(registry_record(), {'room_id': 'room_fixture'}, surface(),
                                 visual_geometry(), request(), config=config())
    seated_request = dict(request())
    seated_request['surface_normal_offset_m'] = 0.02
    seated_request['surface_normal_offset_source'] = 'measured slab top under this footprint'
    seated = plan_source_placement(registry_record(), {'room_id': 'room_fixture'}, surface(),
                                   visual_geometry(), seated_request, config=config())
    assert seated['root_transform']['translation_m'][1] == pytest.approx(
        flat['root_transform']['translation_m'][1] + 0.02)
    assert seated['emitter_transform']['position_m'][1] == pytest.approx(
        flat['emitter_transform']['position_m'][1] + 0.02)
    # only the height moves; the asset keeps its footprint and its rotation
    assert seated['root_transform']['rotation_xyzw'] == flat['root_transform']['rotation_xyzw']
    assert seated['asset_resting_pose']['footprint_extent_m'] == flat['asset_resting_pose']['footprint_extent_m']
    assert seated['support_identity']['surface_normal_offset_m'] == 0.02
    evidence = next(item for item in seated['planning_evidence']
                    if item['kind'] == 'support_normal_offset')
    assert evidence['status'] == 'observed'
    assert evidence['source'] == 'measured slab top under this footprint'


def test_a_negative_offset_seats_an_asset_that_the_fit_left_floating():
    seated_request = dict(request())
    seated_request['surface_normal_offset_m'] = -0.015
    seated = plan_source_placement(registry_record(), {'room_id': 'room_fixture'}, surface(),
                                   visual_geometry(), seated_request, config=config())
    flat = plan_source_placement(registry_record(), {'room_id': 'room_fixture'}, surface(),
                                 visual_geometry(), request(), config=config())
    assert seated['root_transform']['translation_m'][1] < flat['root_transform']['translation_m'][1]


def test_no_offset_leaves_the_placement_exactly_where_it_was():
    flat = plan_source_placement(registry_record(), {'room_id': 'room_fixture'}, surface(),
                                 visual_geometry(), request(), config=config())
    assert flat['support_identity']['surface_normal_offset_m'] == 0.0
    assert 'none' in flat['support_identity']['surface_normal_offset_basis']
    evidence = next(item for item in flat['planning_evidence']
                    if item['kind'] == 'support_normal_offset')
    assert evidence['status'] == 'not_applied'


def test_the_batch_planner_carries_the_offset_into_every_candidate_trial():
    rows = [{**request(), 'surface_normal_offset_m': 0.02}]
    rows[0].pop('candidate_index')
    result = plan_static_source_placements(
        registry_record(), {'room_id': 'room_fixture'}, surface(), visual_geometry(),
        rows, config=config())
    assert result['instances'][0]['status'] == 'planned'
    assert result['instances'][0]['support_identity']['surface_normal_offset_m'] == 0.02
