from __future__ import annotations

import numpy as np
import pytest

from avengine.capture.mixed_capture import (
    MixedCaptureError,
    capture_human_beagle_paths,
)
from avengine.dataset.visual_review import (
    AssetBoundVisualReviewError,
    _require_explicit_runtime_root,
    _sensor_rig_readback_errors,
)
from tools.dataset.build_asset_bound_visual_reviews import (
    AssetBoundReviewError,
    _capture_arrays,
    _heading_xz,
    _world_centers,
)


def _world_matrices() -> np.ndarray:
    matrices = np.broadcast_to(np.eye(4), (75, 2, 4, 4)).copy()
    matrices[:, 0, 0, 3] = np.arange(75, dtype=np.float64)
    matrices[:, 1, 2, 3] = -2.0
    return matrices


def test_review_centers_and_headings_are_reconstructed_from_asset_root_space() -> None:
    matrices = _world_matrices()
    centers = _world_centers(matrices, ((0.0, 1.61, 0.0), (0.312, 0.252, 0.0)))
    headings = _heading_xz(matrices, ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))

    assert centers.shape == (75, 2, 3)
    assert centers[0, 0] == pytest.approx((0.0, 1.61, 0.0))
    assert centers[74, 0] == pytest.approx((74.0, 1.61, 0.0))
    assert centers[0, 1] == pytest.approx((0.312, 0.252, -2.0))
    assert np.array_equal(headings["source1"], np.tile((0.0, 1.0), (75, 1)))
    assert np.array_equal(headings["source2"], np.tile((1.0, 0.0), (75, 1)))


def test_review_rejects_malformed_capture_transforms() -> None:
    with pytest.raises(AssetBoundReviewError, match="actor transforms"):
        _world_centers(np.eye(4), ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)))


def test_review_rejects_capture_from_a_different_asset_instance(tmp_path) -> None:
    capture = tmp_path / "capture"
    arrays = capture / "arrays"
    arrays.mkdir(parents=True)
    (capture / "evidence.json").write_text(
        """{
  "status": "pass",
  "research_only": true,
  "actors": [
    {"actor_class": "human", "asset_id": "human_expected"},
    {"actor_class": "cat", "asset_id": "cat_wrong_instance"}
  ]
}\n""",
        encoding="utf-8",
    )

    with pytest.raises(AssetBoundReviewError, match="asset IDs"):
        _capture_arrays(
            capture,
            expected_actor_classes=("human", "cat"),
            expected_asset_ids=("human_expected", "cat_expected"),
        )


def test_mixed_capture_refuses_a_secondary_actor_that_would_reuse_human_identity() -> None:
    with pytest.raises(MixedCaptureError, match="identity fields"):
        capture_human_beagle_paths(
            room_manifest_path="room.json",
            m1_request_path="m1.json",
            human_runtime_glb_path="human.glb",
            beagle_animal_manifest_path="animal.json",
            beagle_m2_request_path="m2.json",
            human_root_path_m=(),
            beagle_root_path_m=(),
            output_dir="output",
            secondary_actor_id="human0",
        )


def test_mixed_capture_requires_an_explicit_human_asset_identity() -> None:
    with pytest.raises(MixedCaptureError, match="identity fields"):
        capture_human_beagle_paths(
            room_manifest_path="room.json",
            m1_request_path="m1.json",
            human_runtime_glb_path="human.glb",
            beagle_animal_manifest_path="animal.json",
            beagle_m2_request_path="m2.json",
            human_root_path_m=(),
            beagle_root_path_m=(),
            output_dir="output",
            human_asset_id="",
        )


def test_sensor_rig_readback_checks_agent_camera_listener_and_all_sensors() -> None:
    expected = {
        "translation_m": [1.0, 1.5, -2.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    listener = {
        "translation_m": [1.01, 1.5, -2.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    errors = _sensor_rig_readback_errors(
        expected_world_from_rig=expected,
        snapshot={
            "agent": expected,
            "sensors": {
                "rgb0": expected,
                "depth0": expected,
                "listener0": listener,
            },
        },
        rgb_sensor_uuid="rgb0",
        listener_uuid="listener0",
    )

    assert errors["agent"] == pytest.approx(0.0)
    assert errors["camera"] == pytest.approx(0.0)
    assert errors["listener"] == pytest.approx(0.01)
    assert errors["all_sensors"] == pytest.approx(0.01)


def test_sensor_rig_readback_fails_closed_when_listener_is_missing() -> None:
    expected = {
        "translation_m": [1.0, 1.5, -2.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    with pytest.raises(
        AssetBoundVisualReviewError,
        match="readback is incomplete",
    ):
        _sensor_rig_readback_errors(
            expected_world_from_rig=expected,
            snapshot={
                "agent": expected,
                "sensors": {"rgb0": expected},
            },
            rgb_sensor_uuid="rgb0",
            listener_uuid="listener0",
        )


def test_direct_review_requires_an_explicit_runtime_root() -> None:
    with pytest.raises(AssetBoundVisualReviewError, match="explicit runtime_root"):
        _require_explicit_runtime_root(None)


def test_direct_review_rejects_a_missing_runtime_root(tmp_path) -> None:
    with pytest.raises(AssetBoundVisualReviewError, match="missing"):
        _require_explicit_runtime_root(tmp_path / "absent")


def test_direct_review_rejects_a_git_checkout_runtime_root(tmp_path) -> None:
    checkout = tmp_path / "habitat-sim-AVEngine"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "data").mkdir()
    with pytest.raises(AssetBoundVisualReviewError, match="Git"):
        _require_explicit_runtime_root(checkout / "data")


def test_direct_review_accepts_an_explicit_non_checkout_runtime_root(tmp_path) -> None:
    runtime = tmp_path / "runtime-root"
    runtime.mkdir()
    assert _require_explicit_runtime_root(runtime) == runtime.resolve()
