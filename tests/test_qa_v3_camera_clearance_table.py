"""Camera clearance table geometry, proven on an analytic room.

No engine, no real scene: an axis-aligned box room with two obstacles is
ray-cast analytically into the four 90-degree faces and, independently, into
the production camera.  The re-projection from the faces must reproduce the
direct render and its clearance verdict, the left/right convention must match
UE yaw, and no-hit pixels must never count as blockage.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from build_qa_v3_camera_clearance_table import (  # noqa: E402
    parse_args,
    yaw_summaries,
)
from camera_clearance import (  # noqa: E402
    FACE_YAWS_DEG,
    NO_HIT_M,
    CameraClearanceError,
    VirtualCamera,
    blocked_column_fraction,
    clean_depth,
    face_ray_directions,
    min_pool,
    point_key,
    sample_cube_radial,
    yaw_bin_index,
)
from preflight_camera_clearance_depth import (  # noqa: E402
    clearance_statistics,
    target_band_rows,
)

CAMERA_HEIGHT_M = 1.47
# room: walls at +-6 m, floor 1.47 m below the camera, ceiling 1.5 m above
PLANES = [(0, 6.0), (0, -6.0), (1, 6.0), (1, -6.0), (2, -CAMERA_HEIGHT_M), (2, 1.5)]
# a kitchen island 0.8-1.4 m straight ahead (+X), 1.2 m tall (top 0.27 m below
# the lens): it fills most of the target band rows, so the column medians the
# preflight metric uses fall below 1.5 m, while the eye band stays clear
SOFA = (np.array([0.8, -0.8, -CAMERA_HEIGHT_M]), np.array([1.4, 0.8, -0.27]))
# a tall thin pillar ahead-right (+X, +Y): tests the left/right convention
PILLAR = (np.array([2.0, 1.5, -CAMERA_HEIGHT_M]), np.array([2.2, 1.7, 1.5]))
BOXES = [SOFA, PILLAR]


def _raycast_t(dirs: np.ndarray) -> np.ndarray:
    dirs = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    t = np.full(dirs.shape[:-1], np.inf)
    for axis, value in PLANES:
        tt = value / dirs[..., axis]
        t = np.where((tt > 0) & (tt < t), tt, t)
    for lo, hi in BOXES:
        t1 = lo / dirs
        t2 = hi / dirs
        tmin = np.minimum(t1, t2).max(axis=-1)
        tmax = np.maximum(t1, t2).min(axis=-1)
        hit = (tmax >= tmin) & (tmin > 0)
        t = np.where(hit & (tmin < t), tmin, t)
    return t


def _synthetic_faces(width=256, height=384) -> np.ndarray:
    faces = []
    for yaw in FACE_YAWS_DEG:
        dirs = face_ray_directions(width, height, yaw)
        # the engine depth buffer is the radial distance to the surface
        faces.append(_raycast_t(dirs).astype(np.float32))
    return np.stack(faces)


def _direct_depth(camera: VirtualCamera, yaw_deg: float) -> np.ndarray:
    theta = np.deg2rad(yaw_deg + camera.alpha_deg)
    elev = np.deg2rad(camera.elev_deg)
    dirs = np.stack([np.cos(elev) * np.cos(theta), np.cos(elev) * np.sin(theta),
                     np.sin(elev)], axis=-1)
    return _raycast_t(dirs).astype(np.float32)


@pytest.fixture(scope="module")
def faces():
    return _synthetic_faces()


def test_face_rays_follow_ue_yaw_convention():
    dirs = face_ray_directions(64, 64, 0.0)
    centre = dirs[32, 32]
    assert centre[0] > 0.99                       # face 0 looks along +X
    assert dirs[32, 60][1] > 0.3                  # right half of the image is +Y
    assert dirs[60, 32][2] < -0.3                 # bottom rows look down
    east = face_ray_directions(64, 64, 90.0)[32, 32]
    assert east[1] > 0.99                         # face 1 looks along +Y


def test_reprojection_reproduces_a_direct_render(faces):
    camera = VirtualCamera(105.0, 320, 180)
    for yaw in (0.0, 27.0, 133.0, 250.5):
        direct = _direct_depth(camera, yaw)
        reprojected = camera.reproject_depth(faces, yaw)
        assert np.isfinite(reprojected).all()
        rel = np.abs(reprojected - direct) / direct
        assert np.median(rel) < 0.01, yaw
        assert np.percentile(rel, 95) < 0.08, yaw


def test_obstacle_on_the_right_lands_in_the_right_columns(faces):
    camera = VirtualCamera(105.0, 320, 180)
    planar = camera.reproject_depth(faces, 0.0)
    # along the horizon row only the pillar is closer than the walls (the island
    # top sits below eye height); it stands 2.0-2.2 m ahead at bearing ~37
    # degrees, i.e. right of centre
    horizon = planar[camera.height // 2]
    columns = np.where(horizon < 3.0)[0]
    assert columns.size > 0
    assert columns.min() > camera.width // 2
    # turned towards it, it sits in the centre columns
    facing = camera.reproject_depth(faces, math.degrees(math.atan2(1.6, 2.1)))
    centre = np.where(facing[camera.height // 2] < 3.0)[0]
    assert abs(centre.mean() - camera.width / 2) < camera.width * 0.05


def test_clearance_verdict_matches_the_preflight_statistics(faces):
    camera = VirtualCamera(105.0, 640, 360)
    rows = target_band_rows(360, hfov_deg=105.0, aspect=640 / 360,
                            camera_height_m=CAMERA_HEIGHT_M, target_height_m=0.5,
                            distance_range_m=(2.5, 10.0))
    for yaw, expect_blocked in ((0.0, True), (180.0, False), (90.0, False)):
        direct = _direct_depth(camera, yaw)
        stats = clearance_statistics(direct, (1.0, 1.5, 2.5), hfov_deg=105.0,
                                     camera_height_m=CAMERA_HEIGHT_M,
                                     target_height_m=0.5,
                                     target_distance_range_m=(2.5, 10.0))
        preflight_fraction = stats["target_band_blocked_column_fraction"]["1.5"]
        table_fraction = blocked_column_fraction(
            camera.reproject_depth(faces, yaw), rows, 1.5)
        assert abs(table_fraction - preflight_fraction) < 0.03, yaw
        assert (preflight_fraction > 0.3) is expect_blocked, yaw
        # the island top is below eye height: the eye band stays clear
        assert stats["eye_band_blocked_column_fraction"]["1.5"] == 0.0


def test_yaw_summaries_change_with_heading(faces):
    camera = VirtualCamera(105.0, 320, 180)
    yaws = [i * 2.0 for i in range(180)]
    summary = yaw_summaries(faces, camera=camera, camera_height_m=CAMERA_HEIGHT_M,
                            yaws_deg=yaws, nears_m=(1.0, 1.5, 2.5),
                            target_heights_m=(0.5, 1.0, 1.7),
                            target_distance_range_m=(2.5, 10.0))
    target = summary["target_band_blocked_column_fraction"]
    assert target.shape == (3, 3, 180)
    dog_15 = target[0, 1]                    # 0.5 m target, 1.5 m near
    assert dog_15[yaw_bin_index(0.0, 2.0)] > 0.3
    assert dog_15[yaw_bin_index(180.0, 2.0)] == 0.0
    # a taller target band sits higher in the frame, so the island blocks less of it
    assert target[2, 1, yaw_bin_index(0.0, 2.0)] <= dog_15[yaw_bin_index(0.0, 2.0)]
    # nothing at eye height within 1.5 m (the pillar stands 2 m away)
    assert summary["eye_band_blocked_column_fraction"][1].max() == 0.0


def test_no_hit_pixels_are_kept_as_sentinel_and_never_block(faces):
    holed = faces.copy()
    holed[0, :, 100:140] = NO_HIT_M
    camera = VirtualCamera(105.0, 320, 180)
    planar = camera.reproject_depth(holed, 0.0)
    assert (planar == NO_HIT_M).any()
    rows = target_band_rows(180, hfov_deg=105.0, aspect=320 / 180,
                            camera_height_m=CAMERA_HEIGHT_M, target_height_m=0.5,
                            distance_range_m=(2.5, 10.0))
    only_sentinel = np.full_like(planar, NO_HIT_M)
    assert blocked_column_fraction(only_sentinel, rows, 1.5) == 0.0
    raw = np.array([[np.nan, -1.0, 0.0, 2000.0, 3.0]], dtype=np.float32)
    assert clean_depth(raw).tolist() == [[NO_HIT_M, NO_HIT_M, NO_HIT_M, NO_HIT_M, 3.0]]


def test_directions_outside_the_ring_coverage_are_nan(faces):
    radial = sample_cube_radial(faces, np.array([45.0]), np.array([-80.0]))
    assert np.isnan(radial).all()
    inside = sample_cube_radial(faces, np.array([45.0]), np.array([-30.0]))
    assert np.isfinite(inside).all()


def test_min_pool_keeps_the_nearest_depth():
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    pooled = min_pool(depth, 2)
    assert pooled.tolist() == [[0.0, 2.0], [8.0, 10.0]]
    with pytest.raises(CameraClearanceError):
        min_pool(depth, 3)


def test_point_key_and_yaw_bins_are_stable():
    assert point_key((12.34, -5.06)) == point_key((12.3, -5.1))
    assert point_key((0.0, 0.0)) == "0.0,0.0"
    assert yaw_bin_index(359.5, 2.0) == 0
    assert yaw_bin_index(-2.0, 2.0) == 179
    assert yaw_bin_index(90.9, 2.0) == 45


def test_cli_validates_its_grids():
    common = ["--scene-config", "s.json", "--stage-root", "r",
              "--spear-executable", "e", "--output", "/tmp/out"]
    args = parse_args(common)
    assert args.near_m == [1.0, 1.5, 2.5]
    assert args.target_heights_m == [0.5, 1.0, 1.7]
    with pytest.raises(SystemExit):
        parse_args(common + ["--yaw-step-deg", "7"])
    with pytest.raises(SystemExit):
        parse_args(common + ["--verdict-near-m", "2.0"])
    with pytest.raises(SystemExit):
        parse_args(common + ["--store-downsample", "3"])


# ---------------------------------------------------------------------------
# reader (the solver side)
# ---------------------------------------------------------------------------

def _write_synthetic_table(root, faces, *, points=((100.0, -250.0), (312.5, 40.0)),
                           heights=(1.47,), yaw_step=2.0):
    import json
    from camera_clearance import point_key
    root.mkdir()
    (root / "faces").mkdir()
    camera = VirtualCamera(105.0, 320, 180)
    yaws = [i * yaw_step for i in range(int(360 / yaw_step))]
    nears = (1.0, 1.5, 2.5)
    target_heights = (0.5, 1.0, 1.7)
    target = np.zeros((len(points), len(heights), 3, 3, len(yaws)), np.float32)
    eye = np.zeros((len(points), len(heights), 3, len(yaws)), np.float32)
    stored = []
    index_rows = []
    for pi in range(len(points)):
        for hi, height in enumerate(heights):
            summary = yaw_summaries(faces, camera=camera, camera_height_m=height,
                                    yaws_deg=yaws, nears_m=nears,
                                    target_heights_m=target_heights,
                                    target_distance_range_m=(2.5, 10.0))
            target[pi, hi] = summary["target_band_blocked_column_fraction"]
            eye[pi, hi] = summary["eye_band_blocked_column_fraction"]
            stored.append(np.stack([min_pool(f, 2) for f in faces]).astype(np.float16))
            index_rows.append((pi, hi))
    np.savez_compressed(root / "faces" / "shard_0000.npz", radial_m=np.stack(stored),
                        point_index=np.asarray([p for p, _ in index_rows], np.int32),
                        height_index=np.asarray([h for _, h in index_rows], np.int32))
    np.savez_compressed(root / "summaries.npz",
                        target_band_blocked_column_fraction=target.astype(np.float16),
                        eye_band_blocked_column_fraction=eye.astype(np.float16),
                        clear_default_rule=target[:, :, 0, 1, :] <= 0.2,
                        points_xy_cm=np.asarray(points, np.float32),
                        camera_heights_m=np.asarray(heights, np.float32),
                        yaws_deg=np.asarray(yaws, np.float32),
                        nears_m=np.asarray(nears, np.float32),
                        target_heights_m=np.asarray(target_heights, np.float32),
                        point_seconds=np.zeros((len(points), len(heights)), np.float32))
    (root / "camera_clearance_table.json").write_text(json.dumps({
        "schema": "qa_v3_camera_clearance_table_v1", "scene_id": "synthetic_room",
        "code": {"revision": "0" * 40, "dirty": False},
        "summaries": {"path": "summaries.npz", "yaw_step_deg": yaw_step},
        "faces": {"shards": [{"path": "faces/shard_0000.npz", "count": len(stored)}]},
        "points": {"keys": [point_key(xy) for xy in points]},
        "stage": {"native_map": "/Game/synthetic"}}))
    return root


PARAMS_RULE = {"CAMERA_CLEARANCE_TARGET_HEIGHT_M": 0.5, "CAMERA_CLEARANCE_NEAR_M": 1.5,
               "CAMERA_CLEARANCE_BLOCKED_FRACTION_MAX": 0.2}


def test_reader_answers_clear_and_blocked_yaws_from_the_table(tmp_path, faces):
    from camera_clearance import CameraClearanceTable, rule_from_params
    root = _write_synthetic_table(tmp_path / "table", faces)
    table = CameraClearanceTable.load(root)
    rule = rule_from_params(PARAMS_RULE, table)
    assert table.scene_id == "synthetic_room"
    assert table.identity["points"] == 2
    # facing the island (+X) is blocked, facing away is clear; 0.1 cm jitter
    # in the solver's point still resolves to the same table row
    assert table.blocked_fraction((100.04, -249.96), 1.47, 0.0, rule) > 0.3
    assert table.is_clear((100.0, -250.0), 1.47, 180.0, rule)
    assert not table.is_clear((100.0, -250.0), 1.4715, 1.0, rule)
    mask = table.clear_yaw_mask((312.5, 40.0), 1.47, rule)
    assert mask.shape == (180,) and mask.any() and not mask.all()
    assert table.points_with_clear_yaw(1.47, rule).tolist() == [True, True]
    assert table.missing_points([(100.0, -250.0), (7.0, 7.0)]) == ["7.0,7.0"]


def test_reader_fails_closed_on_uncovered_point_height_or_rule(tmp_path, faces):
    from camera_clearance import (CameraClearanceError, CameraClearanceTable,
                                  ClearanceRule, rule_from_params)
    root = _write_synthetic_table(tmp_path / "table", faces)
    table = CameraClearanceTable.load(root)
    rule = ClearanceRule(0.5, 1.5, 0.2)
    with pytest.raises(CameraClearanceError, match="not covered"):
        table.blocked_fraction((7.0, 7.0), 1.47, 0.0, rule)
    with pytest.raises(CameraClearanceError, match="camera height"):
        table.blocked_fraction((100.0, -250.0), 1.8, 0.0, rule)
    with pytest.raises(CameraClearanceError, match="near distance"):
        table.blocked_fraction((100.0, -250.0), 1.47, 0.0, ClearanceRule(0.5, 2.0, 0.2))
    with pytest.raises(CameraClearanceError, match="target height"):
        rule_from_params(dict(PARAMS_RULE, CAMERA_CLEARANCE_TARGET_HEIGHT_M=0.6), table)
    with pytest.raises(CameraClearanceError, match="lack camera clearance keys"):
        rule_from_params({"CAMERA_CLEARANCE_NEAR_M": 1.5}, table)
    with pytest.raises(CameraClearanceError, match="no index"):
        CameraClearanceTable.load(tmp_path / "missing")


def test_reader_sight_lines_hit_the_stored_geometry(tmp_path, faces):
    from camera_clearance import CameraClearanceTable, NO_HIT_M
    root = _write_synthetic_table(tmp_path / "table", faces)
    table = CameraClearanceTable.load(root)
    stored = table.faces((100.0, -250.0), 1.47)
    assert stored.shape == (4, 192, 128) and stored.dtype == np.float32
    # the pillar stands 2.0-2.2 m ahead-right; along its bearing at eye height
    # the first obstacle is the pillar, elsewhere it is the far wall
    bearing = math.degrees(math.atan2(1.6, 2.1))
    hit = table.first_obstacle_m((100.0, -250.0), 1.47, np.array([bearing, 180.0]),
                                 np.array([0.0, 0.0]))
    assert 2.4 < hit[0] < 2.9
    assert hit[1] > 5.5
    assert hit[1] < NO_HIT_M


def test_fallback_heights_come_from_params_only():
    from camera_clearance import CameraClearanceError, fallback_heights_from_params
    assert fallback_heights_from_params({}) == []
    assert fallback_heights_from_params({"CAMERA_HEIGHT_FALLBACK_M": 1.8}) == [1.8]
    assert fallback_heights_from_params({"CAMERA_HEIGHT_FALLBACK_M": [1.8, 2.0]}) == [1.8, 2.0]
    with pytest.raises(CameraClearanceError):
        fallback_heights_from_params({"CAMERA_HEIGHT_FALLBACK_M": [-1.0]})


# ---------------------------------------------------------------------------
# visibility prediction on the analytic room
# ---------------------------------------------------------------------------

CAM = (100.0, -250.0)          # table point (UE cm); the room is centred on the camera
DOG = {"height_m": 0.5, "length_m": 0.8}
HUMAN = {"height_m": 1.7, "length_m": 0.5}


def _actor_at(bearing_deg, distance_m):
    return (CAM[0] + 100.0 * distance_m * math.cos(math.radians(bearing_deg)),
            CAM[1] + 100.0 * distance_m * math.sin(math.radians(bearing_deg)))


def _predict(table, actor_xy, body, others=()):
    from visibility_prediction import predict_point_visibility
    return predict_point_visibility(table, camera_xy_cm=CAM, camera_height_m=1.47,
                                    ground_z_cm=0.0, actor_xy_cm=actor_xy, body=body,
                                    others=others)


def test_prediction_sees_the_island_hide_a_dog_but_not_a_standing_human(tmp_path, faces):
    from camera_clearance import CameraClearanceTable
    table = CameraClearanceTable.load(_write_synthetic_table(tmp_path / "table", faces))
    dog_behind_island = _predict(table, _actor_at(0.0, 3.0), DOG)
    assert dog_behind_island["predicted_visible_fraction"] == 0.0
    assert dog_behind_island["blocked_by_scene"] == 9
    human_behind_island = _predict(table, _actor_at(0.0, 3.0), HUMAN)
    assert 0.0 < human_behind_island["predicted_visible_fraction"] < 1.0
    dog_in_the_open = _predict(table, _actor_at(180.0, 3.0), DOG)
    assert dog_in_the_open["predicted_visible_fraction"] == 1.0
    assert dog_in_the_open["known_fraction"] == 1.0
    behind_pillar = _predict(table, _actor_at(math.degrees(math.atan2(1.6, 2.1)), 3.5), DOG)
    assert behind_pillar["predicted_visible_fraction"] < 0.5


def test_prediction_counts_another_actor_as_an_occluder(tmp_path, faces):
    from camera_clearance import CameraClearanceTable
    table = CameraClearanceTable.load(_write_synthetic_table(tmp_path / "table", faces))
    clear = _predict(table, _actor_at(180.0, 4.0), DOG)
    # a dog just in front of the target cuts the low sight lines; a dog half
    # way to the camera does not, because the lens looks down over its back
    blocked = _predict(table, _actor_at(180.0, 4.0), DOG,
                       others=[(_actor_at(180.0, 3.6), DOG)])
    over_the_back = _predict(table, _actor_at(180.0, 4.0), DOG,
                             others=[(_actor_at(180.0, 2.0), DOG)])
    assert clear["predicted_visible_fraction"] == 1.0
    assert blocked["predicted_visible_fraction"] < 1.0
    assert blocked["blocked_by_actor"] > 0
    assert over_the_back["blocked_by_actor"] == 0
    aside = _predict(table, _actor_at(180.0, 4.0), DOG,
                     others=[(_actor_at(150.0, 3.6), DOG)])
    assert aside["blocked_by_actor"] == 0


def test_predicted_tiers_follow_the_pixel_join_ladder():
    from visibility_prediction import predicted_tier
    assert predicted_tier(False, 1.0) == "out_of_view"
    assert predicted_tier(True, None) == "unknown"
    assert predicted_tier(True, 0.0) == "hidden"
    assert predicted_tier(True, 0.1) == "heavy"
    assert predicted_tier(True, 0.3) == "medium"
    assert predicted_tier(True, 0.9) == "light"


def test_timeline_prediction_and_statistics(tmp_path, faces):
    from camera_clearance import CameraClearanceTable
    from visibility_prediction import predict_timeline, timeline_statistics
    table = CameraClearanceTable.load(_write_synthetic_table(tmp_path / "table", faces))
    # the target walks from behind the island (bearing 0) round to the open
    # side (bearing 180) over 10 frames; the other actor stays in the open
    target = [_actor_at(180.0 * k / 9, 3.0) for k in range(10)]
    other = [_actor_at(240.0, 3.0)] * 10
    result = predict_timeline(table, camera_xy_cm=CAM, camera_height_m=1.47,
                              camera_yaw_deg=180.0, hfov_deg=105.0, ground_z_cm=0.0,
                              routes_by_slot={"source1": target, "source2": other},
                              bodies_by_slot={"source1": DOG, "source2": DOG})
    rows = result["slots"]["source1"]["per_frame"]
    assert [r["frame"] for r in rows] == list(range(10))
    assert rows[0]["in_fov"] is False and rows[0]["tier"] == "out_of_view"
    assert rows[-1]["in_fov"] is True and rows[-1]["tier"] == "light"
    assert all(r["tier"] in ("light", "medium", "heavy", "hidden", "out_of_view", "unknown")
               for r in rows)
    stats = timeline_statistics(rows, instants={"anchor": 1, "query": 9})
    assert stats["frames_evaluated"] == 10
    assert stats["visible_near_instant"]["query"] is True
    assert stats["visible_near_instant"]["anchor"] is False
    assert stats["hidden_frames_before_instant"]["anchor"] >= 2
    assert stats["hidden_frames_before_instant"]["query"] == 0
    assert stats["never_visible"] is False
    assert 0.0 < stats["visible_frames_fraction"] < 1.0
