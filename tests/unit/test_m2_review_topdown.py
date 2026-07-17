from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest

import avengine.m2.review_topdown as review_topdown
from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
import tools.m2.compose_topdown_review as compose_topdown_cli
from avengine.m2.review_topdown import (
    TopdownReviewError,
    compose_topdown_review,
    encode_review_rgb_frames,
    habitat_xz_to_navmesh_pixel,
    render_topdown_panel,
    semantic_object_footprint_from_obb,
    verify_topdown_review_evidence,
)


BOUNDS = ([-2.0, -0.5, -4.0], [2.0, 2.5, 4.0])
_STRICT_CORE_CAPTURE_ERRORS = review_topdown._strict_core_capture_errors


@pytest.fixture(autouse=True)
def _accept_minimal_core_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most focused tests use a tiny core fixture; dispatch is tested separately."""

    monkeypatch.setattr(
        review_topdown, "_strict_core_capture_errors", lambda _path, _inputs: []
    )


def _transform(
    x: float,
    z: float,
    *,
    rotation_xyzw: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "translation_m": [x, 0.0, z],
        "rotation_xyzw": rotation_xyzw or [0.0, 0.0, 0.0, 1.0],
    }


def _records() -> list[dict[str, Any]]:
    return [
        {"frame_index": index, "world_from_actor": _transform(-0.8 + index * 0.8, 0.5)}
        for index in range(3)
    ]


def _room_request() -> dict[str, Any]:
    return {
        "request_id": "room_review_fixture",
        "room_id": "room_review_fixture",
        "primary_camera_rig": {
            "rig_id": "camera_rig_0",
            "view_id": "view0",
            "world_from_rig": _transform(0.0, 2.5),
            "shared_calibration": {
                "projection": "pinhole",
                "hfov_degrees": 90.0,
            },
        },
        "sources": [
            {
                "source_id": "source0",
                "world_from_source": _transform(-1.5, -2.0),
            },
            {
                "source_id": "source1",
                "world_from_source": _transform(1.4, 2.0),
            },
        ],
        "qa_views": [
            {
                "qa_id": "navmesh_topdown",
                "kind": "topdown",
                "meters_per_pixel": 0.05,
                "height_m": 0.02,
            }
        ],
    }


def _navmesh() -> np.ndarray:
    value = np.zeros((16, 8), dtype=np.uint8)
    value[2:14, 1:7] = 1
    value[6:10, 3:5] = 0
    return value


def _rgb_stack() -> np.ndarray:
    rgb = np.zeros((3, 48, 64, 4), dtype=np.uint8)
    rgb[..., 3] = 255
    rgb[0, ..., :3] = (20, 40, 60)
    rgb[1, ..., :3] = (30, 50, 70)
    rgb[2, ..., :3] = (40, 60, 80)
    return rgb


def _navmesh_metadata() -> dict[str, Any]:
    return {
        "qa_id": "navmesh_topdown",
        "meters_per_pixel": 0.05,
        "height_m": 0.02,
        "navigable_pixel_count": int(np.count_nonzero(_navmesh())),
        "room_id": "room_review_fixture",
    }


def _semantic_footprint(
    *, object_id: str = "0_0_42", category: str = "chair"
) -> dict[str, Any]:
    local_to_world = np.eye(4, dtype=np.float64)
    local_to_world[0, 0] = 0.45
    local_to_world[1, 1] = 0.6
    local_to_world[2, 2] = 0.35
    local_to_world[:3, 3] = [0.0, 0.6, 0.0]
    result = semantic_object_footprint_from_obb(
        object_id=object_id,
        category=category,
        local_to_world=local_to_world,
    )
    assert result is not None
    return result


def _write_verifier_inputs(root: Path, rgb: np.ndarray) -> dict[str, Path]:
    root.mkdir(parents=True)
    core = root / "core_capture_evidence.json"
    room_manifest = root / "room_manifest.json"
    room = root / "room_request.json"
    navmesh = root / "room.navmesh"
    rgb_array = root / "rgb.npy"
    write_json(core, {"frames": _records()})
    write_json(room_manifest, {"room_id": "room_review_fixture"})
    write_json(room, _room_request())
    navmesh.write_bytes(b"fixture raw Habitat navmesh")
    np.save(rgb_array, rgb, allow_pickle=False)
    return {
        "core_capture_evidence": core,
        "room_manifest": room_manifest,
        "room_request": room,
        "navmesh": navmesh,
        "rgb_array": rgb_array,
    }


def _compose_verifiable_review(
    root: Path, *, with_semantic_object: bool = False
) -> tuple[Path, dict[str, Path]]:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("strict top-down verifier requires FFmpeg tools")
    rgb = _rgb_stack()
    artifacts = _write_verifier_inputs(root / "inputs", rgb)
    footprints: list[dict[str, Any]] = []
    if with_semantic_object:
        descriptor = root / "inputs" / "scene.house"
        write_json(descriptor, {"fixture": "semantic descriptor"})
        artifacts["semantic_descriptor"] = descriptor
        footprints = [_semantic_footprint()]

    output = root / "review"
    compose_topdown_review(
        rgb_frames=rgb,
        frame_records=_records(),
        room_camera_request=_room_request(),
        navmesh_binary_map=_navmesh(),
        navmesh_bounds=BOUNDS,
        output_dir=output,
        input_artifacts=artifacts,
        semantic_object_footprints=footprints,
        navmesh_metadata=_navmesh_metadata(),
    )
    return output / "topdown_review_evidence.json", artifacts


def test_habitat_xz_mapping_matches_raw_topdown_grid_orientation() -> None:
    assert habitat_xz_to_navmesh_pixel(
        [-2.0, 0.0, -4.0], navmesh_shape_hw=(9, 5), bounds=BOUNDS
    ) == pytest.approx((0.0, 0.0))
    assert habitat_xz_to_navmesh_pixel(
        [0.0, 0.0, 0.0], navmesh_shape_hw=(9, 5), bounds=BOUNDS
    ) == pytest.approx((2.0, 4.0))
    assert habitat_xz_to_navmesh_pixel(
        [2.0, 0.0, 4.0], navmesh_shape_hw=(9, 5), bounds=BOUNDS
    ) == pytest.approx((4.0, 8.0))
    assert habitat_xz_to_navmesh_pixel(
        [-3.0, -6.0], navmesh_shape_hw=(9, 5), bounds=BOUNDS
    ) == pytest.approx((-1.0, -2.0))


def test_topdown_panel_draws_navmesh_camera_actor_trajectory_and_sources() -> None:
    first = render_topdown_panel(
        _navmesh(),
        bounds=BOUNDS,
        frame_records=_records(),
        room_camera_request=_room_request(),
        frame_index=0,
        panel_size_wh=(160, 120),
    )
    last = render_topdown_panel(
        _navmesh(),
        bounds=BOUNDS,
        frame_records=_records(),
        room_camera_request=_room_request(),
        frame_index=2,
        panel_size_wh=(160, 120),
    )

    assert first.shape == (120, 160, 3)
    assert first.dtype == np.uint8
    assert not np.array_equal(first, last)
    colors = {tuple(color) for color in first.reshape(-1, 3)}
    assert (195, 201, 207) in colors  # navigable navmesh
    assert (48, 53, 59) in colors  # blocked or unknown navmesh
    assert (46, 154, 255) in colors  # camera marker
    assert (69, 196, 118) in colors  # first source anchor
    assert (180, 91, 214) in colors  # second source anchor
    assert (255, 142, 45) in colors  # actor marker


def test_descriptor_object_footprint_uses_transformed_obb_corners_not_origin() -> None:
    local_to_world = np.eye(4, dtype=np.float64)
    local_to_world[0, 0] = 0.6
    local_to_world[1, 1] = 1.0
    local_to_world[2, 2] = 0.3
    local_to_world[:3, 3] = [-8.0, 1.0, -3.0]

    footprint = semantic_object_footprint_from_obb(
        object_id="0_0_5",
        category="bed",
        local_to_world=local_to_world,
    )

    assert footprint is not None
    assert footprint["bounds_min_xz_m"] == pytest.approx([-8.6, -3.3])
    assert footprint["bounds_max_xz_m"] == pytest.approx([-7.4, -2.7])
    assert all(
        point[0] < -7.0 and point[1] < -2.0 for point in footprint["polygon_xz_m"]
    )
    assert (
        semantic_object_footprint_from_obb(
            object_id="0_0_0",
            category="wall",
            local_to_world=local_to_world,
        )
        is None
    )


def test_topdown_panel_draws_descriptor_object_without_changing_nav_semantics() -> None:
    panel = render_topdown_panel(
        _navmesh(),
        bounds=BOUNDS,
        frame_records=_records(),
        room_camera_request=_room_request(),
        frame_index=0,
        semantic_object_footprints=[_semantic_footprint()],
        panel_size_wh=(160, 120),
    )

    colors = {tuple(color) for color in panel.reshape(-1, 3)}
    assert (151, 105, 62) in colors  # descriptor object outline
    assert (195, 201, 207) in colors  # navigability remains a separate layer
    assert (48, 53, 59) in colors  # blocked/unknown remains a separate layer


def test_actor_heading_changes_panel_without_moving_actor() -> None:
    identity_record = [{"frame_index": 0, "world_from_actor": _transform(0.0, 0.0)}]
    turn_record = [
        {
            "frame_index": 0,
            "world_from_actor": _transform(
                0.0, 0.0, rotation_xyzw=[0.0, 1.0, 0.0, 0.0]
            ),
        }
    ]
    identity = render_topdown_panel(
        _navmesh(),
        bounds=BOUNDS,
        frame_records=identity_record,
        room_camera_request=_room_request(),
        frame_index=0,
        panel_size_wh=(160, 120),
        trusted_actor_local_forward_axis=(1.0, 0.0, 0.0),
        trusted_actor_forward_axis_source="unit_fixture",
    )
    turned = render_topdown_panel(
        _navmesh(),
        bounds=BOUNDS,
        frame_records=turn_record,
        room_camera_request=_room_request(),
        frame_index=0,
        panel_size_wh=(160, 120),
        trusted_actor_local_forward_axis=(1.0, 0.0, 0.0),
        trusted_actor_forward_axis_source="unit_fixture",
    )

    assert not np.array_equal(identity, turned)


def test_actor_heading_uses_positive_x_trajectory_for_identity_frames() -> None:
    matrices = tuple(
        review_topdown._actor_matrix(record, index=index)
        for index, record in enumerate(_records())
    )

    headings, binding = review_topdown._actor_headings_xz(
        matrices,
        trusted_local_forward_axis=None,
        trusted_forward_axis_source=None,
    )

    assert all(heading == pytest.approx([1.0, 0.0]) for heading in headings)
    assert binding["policy"] == review_topdown.ACTOR_HEADING_TRAJECTORY_POLICY
    assert binding["idle_policy"] == review_topdown.ACTOR_IDLE_HEADING_POLICY


def test_horse_like_yaw_cannot_reverse_positive_x_trajectory_heading() -> None:
    sine = float(np.sin(np.pi / 4.0))
    cosine = float(np.cos(np.pi / 4.0))
    records = [
        {
            "frame_index": index,
            "world_from_actor": _transform(
                -0.8 + index * 0.8,
                0.5,
                rotation_xyzw=[0.0, sine, 0.0, cosine],
            ),
        }
        for index in range(3)
    ]
    matrices = tuple(
        review_topdown._actor_matrix(record, index=index)
        for index, record in enumerate(records)
    )

    headings, _binding = review_topdown._actor_headings_xz(
        matrices,
        trusted_local_forward_axis=None,
        trusted_forward_axis_source=None,
    )

    assert all(heading == pytest.approx([1.0, 0.0]) for heading in headings)


def test_compose_topdown_review_is_hash_bound_and_qa_only(tmp_path: Path) -> None:
    rgb = _rgb_stack()
    artifacts = _write_verifier_inputs(tmp_path / "inputs", rgb)
    encoded_frames: list[np.ndarray] = []

    def fake_encoder(frames: Any, destination: Path, *, fps: int) -> int:
        assert fps == 15
        encoded_frames.extend(np.asarray(frame).copy() for frame in frames)
        destination.write_bytes(b"fixture review-only mp4")
        return len(encoded_frames)

    output = tmp_path / "review"
    evidence = compose_topdown_review(
        rgb_frames=rgb,
        frame_records=_records(),
        room_camera_request=_room_request(),
        navmesh_binary_map=_navmesh(),
        navmesh_bounds=BOUNDS,
        output_dir=output,
        navmesh_metadata=_navmesh_metadata(),
        input_artifacts=artifacts,
        review_video_encode=fake_encoder,
    )

    assert len(encoded_frames) == 3
    assert all(frame.shape == (48, 112, 3) for frame in encoded_frames)
    assert evidence["review_only"] is True
    assert evidence["qa_only"] is True
    assert evidence["formal_view"] is False
    assert evidence["qualification_claim"] is False
    assert evidence["view_id"] is None
    assert evidence["formal_view_ids"] == []
    assert evidence["formal_capture_modified"] is False
    assert evidence["sensor_view_created"] is False
    assert evidence["qa_policy"]["formal_view"] is False
    assert evidence["qa_policy"]["qualification_claim"] is False
    assert evidence["qa_policy"]["view_id"] is None
    assert (
        evidence["qa_policy"]["navmesh_semantics"]
        == "binary_navigability_not_object_identity"
    )
    assert evidence["qa_policy"]["descriptor_semantics_not_object_detection"] is True
    assert evidence["qa_policy"]["semantic_scene_access_created_sensor"] is False
    assert evidence["inputs"]["source_anchors"]["count"] == 2
    semantic = evidence["inputs"]["semantic_object_footprints"]
    assert semantic["count"] == 0
    assert semantic["objects"] == []
    assert semantic["canonical_content_sha256"] == canonical_json_sha256([])
    assert semantic["semantic_descriptor_artifact"] is None
    assert evidence["encoder"]["png_staging"] is False
    assert evidence["layout"]["encoded_size_wh"] == [112, 48]
    focus = evidence["layout"]["focus"]
    assert focus["policy"] == "camera_actor_trajectory_sources_aabb_v1"
    assert focus["margin_m"] == 1.0
    assert focus["minimum_span_m"] == 4.0
    assert focus["effective_bounds_min_xyz"] == pytest.approx([-2.0, -0.5, -3.0])
    assert focus["effective_bounds_max_xyz"] == pytest.approx([2.0, 2.5, 3.5])
    assert focus["navmesh_roi_rc_exclusive"] != [0, 0, 16, 8]

    video = output / evidence["output"]["video"]["path"]
    saved_navmesh = output / evidence["output"]["navmesh_binary_map"]["path"]
    assert video.is_file()
    assert saved_navmesh.is_file()
    assert np.array_equal(np.load(saved_navmesh, allow_pickle=False), _navmesh())
    assert evidence["output"]["video"]["sha256"] == sha256_file(video)
    artifact = evidence["inputs"]["artifacts"]["core_capture_evidence"]
    assert artifact["sha256"] == sha256_file(artifacts["core_capture_evidence"])
    declared_hash = evidence.pop("evidence_content_sha256")
    assert declared_hash == canonical_json_sha256(evidence)

    saved = load_json(output / "topdown_review_evidence.json")
    assert saved["evidence_content_sha256"] == declared_hash
    errors = verify_topdown_review_evidence(output / "topdown_review_evidence.json")
    assert "top-down review encoder contract is invalid" in errors
    assert any("output.video is not a probeable MP4" in error for error in errors)
    assert not list(output.rglob("*.png"))


def _rehash_evidence(value: dict[str, Any]) -> None:
    value.pop("evidence_content_sha256", None)
    value["evidence_content_sha256"] = canonical_json_sha256(value)


def _refresh_input_artifact(evidence: dict[str, Any], name: str, path: Path) -> None:
    record = evidence["inputs"]["artifacts"][name]
    record["byte_size"] = path.stat().st_size
    record["sha256"] = sha256_file(path)


def _refresh_output_artifact(evidence: dict[str, Any], name: str, path: Path) -> None:
    record = evidence["output"][name]
    record["byte_size"] = path.stat().st_size
    record["sha256"] = sha256_file(path)


def test_strict_verifier_accepts_real_h264_review(tmp_path: Path) -> None:
    evidence_path, _artifacts = _compose_verifiable_review(tmp_path)

    assert verify_topdown_review_evidence(evidence_path) == []


def test_verifier_rejects_claim_and_canonical_hash_tampering(tmp_path: Path) -> None:
    evidence_path, _artifacts = _compose_verifiable_review(tmp_path)
    evidence = load_json(evidence_path)
    evidence["formal_view"] = True
    write_json(evidence_path, evidence)

    errors = verify_topdown_review_evidence(evidence_path)

    assert "top-down review evidence content hash differs" in errors
    assert "top-down review QA-only claim is invalid" in errors


def test_verifier_rejects_output_and_input_byte_tampering(tmp_path: Path) -> None:
    evidence_path, artifacts = _compose_verifiable_review(tmp_path)
    evidence = load_json(evidence_path)
    video = evidence_path.parent / evidence["output"]["video"]["path"]
    video.write_bytes(video.read_bytes() + b"tamper")
    artifacts["room_request"].write_bytes(artifacts["room_request"].read_bytes() + b" ")

    errors = verify_topdown_review_evidence(evidence_path)

    assert "output.video artifact bytes changed" in errors
    assert "inputs.artifacts.room_request artifact bytes changed" in errors


def test_verifier_rehashes_semantic_footprints_and_descriptor_artifact(
    tmp_path: Path,
) -> None:
    evidence_path, artifacts = _compose_verifiable_review(
        tmp_path, with_semantic_object=True
    )
    evidence = load_json(evidence_path)
    semantic = evidence["inputs"]["semantic_object_footprints"]
    assert semantic["count"] == 1
    assert semantic["semantic_descriptor_artifact"] == "semantic_descriptor"
    assert verify_topdown_review_evidence(evidence_path) == []

    semantic["objects"][0]["category"] = "sofa"
    _rehash_evidence(evidence)
    write_json(evidence_path, evidence)
    errors = verify_topdown_review_evidence(evidence_path)
    assert "semantic object footprints canonical hash differs" in errors
    assert "top-down review evidence content hash differs" not in errors

    # Restore the evidence, then independently tamper with the bound descriptor.
    evidence_path, artifacts = _compose_verifiable_review(
        tmp_path / "descriptor_tamper", with_semantic_object=True
    )
    artifacts["semantic_descriptor"].write_bytes(
        artifacts["semantic_descriptor"].read_bytes() + b"tamper"
    )
    errors = verify_topdown_review_evidence(evidence_path)
    assert "inputs.artifacts.semantic_descriptor artifact bytes changed" in errors


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            "output_escape",
            "output.video escapes the top-down evidence directory",
        ),
        (
            "relative_input",
            "inputs.artifacts.core_capture_evidence input path must be absolute",
        ),
    ],
)
def test_verifier_enforces_output_confinement_and_absolute_inputs(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    evidence_path, _artifacts = _compose_verifiable_review(tmp_path)
    evidence = load_json(evidence_path)
    if mutation == "output_escape":
        escaped = evidence_path.parent.parent / "escaped.mp4"
        escaped.write_bytes(b"escaped review")
        evidence["output"]["video"] = {
            "path": "../escaped.mp4",
            "byte_size": escaped.stat().st_size,
            "sha256": sha256_file(escaped),
        }
    else:
        evidence["inputs"]["artifacts"]["core_capture_evidence"]["path"] = (
            "core_capture_evidence.json"
        )
    _rehash_evidence(evidence)
    write_json(evidence_path, evidence)

    errors = verify_topdown_review_evidence(evidence_path)

    assert expected_error in errors
    assert "top-down review evidence content hash differs" not in errors


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            "core_frames",
            "frame records canonical hash differs from core capture",
        ),
        (
            "room_request",
            "room camera request canonical hash differs",
        ),
        (
            "rgb_array",
            "RGB stack hash differs from rgb_array NPY",
        ),
    ],
)
def test_verifier_independently_recomputes_source_bindings(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    evidence_path, artifacts = _compose_verifiable_review(tmp_path)
    evidence = load_json(evidence_path)
    if mutation == "core_frames":
        core = load_json(artifacts["core_capture_evidence"])
        core["frames"][0]["world_from_actor"]["translation_m"][0] += 0.25
        write_json(artifacts["core_capture_evidence"], core)
        artifact_name = "core_capture_evidence"
    elif mutation == "room_request":
        room = load_json(artifacts["room_request"])
        room["sources"][0]["world_from_source"]["translation_m"][0] += 0.25
        write_json(artifacts["room_request"], room)
        artifact_name = "room_request"
    else:
        rgb = np.load(artifacts["rgb_array"], allow_pickle=False).copy()
        rgb[0, 0, 0, 0] ^= np.uint8(1)
        np.save(artifacts["rgb_array"], rgb, allow_pickle=False)
        artifact_name = "rgb_array"
    _refresh_input_artifact(evidence, artifact_name, artifacts[artifact_name])
    _rehash_evidence(evidence)
    write_json(evidence_path, evidence)

    errors = verify_topdown_review_evidence(evidence_path)

    assert expected_error in errors
    assert "top-down review evidence content hash differs" not in errors
    assert not any("artifact bytes changed" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("timeline", "output.video frame count differs from timeline"),
        ("layout", "top-down review layout size formula differs"),
        ("navmesh", "navmesh binary map hash differs from saved NPY"),
        ("semantic", "inputs.semantic_object_footprints binding is invalid"),
        ("actor_heading", "actor heading binding differs from core trajectory"),
        ("camera_policy", "top-down review QA-only claim is invalid"),
    ],
)
def test_verifier_rejects_self_rehashed_contract_mutations(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    evidence_path, _artifacts = _compose_verifiable_review(tmp_path)
    evidence = load_json(evidence_path)
    if mutation == "timeline":
        evidence["timeline"]["frame_count"] += 1
    elif mutation == "layout":
        evidence["layout"]["content_size_wh"][0] += 2
    elif mutation == "navmesh":
        evidence["inputs"]["navmesh_binary_map"]["content_sha256"] = "0" * 64
    elif mutation == "semantic":
        evidence["inputs"]["semantic_object_footprints"]["unexpected"] = True
    elif mutation == "actor_heading":
        evidence["inputs"]["actor_heading"]["canonical_content_sha256"] = "0" * 64
    else:
        evidence["qa_policy"]["camera_heading_policy"] = "untrusted_camera_axis"
    _rehash_evidence(evidence)
    write_json(evidence_path, evidence)

    errors = verify_topdown_review_evidence(evidence_path)

    assert expected_error in errors
    assert "top-down review evidence content hash differs" not in errors


def test_verifier_rejects_truncated_video_even_after_rehash(
    tmp_path: Path,
) -> None:
    evidence_path, _artifacts = _compose_verifiable_review(tmp_path)
    evidence = load_json(evidence_path)
    video = evidence_path.parent / evidence["output"]["video"]["path"]
    payload = video.read_bytes()
    video.write_bytes(payload[: max(32, len(payload) // 2)])
    _refresh_output_artifact(evidence, "video", video)
    _rehash_evidence(evidence)
    write_json(evidence_path, evidence)

    errors = verify_topdown_review_evidence(evidence_path)

    assert any(
        marker in error
        for error in errors
        for marker in (
            "output.video is not a probeable MP4",
            "output.video failed full decode",
            "output.video stream metadata is invalid",
        )
    )
    assert "top-down review evidence content hash differs" not in errors


def test_verifier_rejects_valid_black_video_even_after_rehash(
    tmp_path: Path,
) -> None:
    evidence_path, _artifacts = _compose_verifiable_review(tmp_path)
    evidence = load_json(evidence_path)
    video = evidence_path.parent / evidence["output"]["video"]["path"]
    width, height = evidence["layout"]["encoded_size_wh"]
    black_video = evidence_path.parent / "black_replacement.mp4"
    black_frames = (
        np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(evidence["timeline"]["frame_count"])
    )
    encode_review_rgb_frames(
        black_frames,
        black_video,
        fps=evidence["timeline"]["frame_rate_hz"],
    )
    black_video.replace(video)
    _refresh_output_artifact(evidence, "video", video)
    _rehash_evidence(evidence)
    write_json(evidence_path, evidence)

    errors = verify_topdown_review_evidence(evidence_path)

    assert any(
        error.startswith("output.video pixels differ from rebuilt composite frames")
        for error in errors
    )
    assert "top-down review evidence content hash differs" not in errors


def test_topdown_verifier_dispatches_local_tr_core_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path, artifacts = _compose_verifiable_review(tmp_path)
    evidence = load_json(evidence_path)
    core = load_json(artifacts["core_capture_evidence"])
    core["schema"] = compose_topdown_cli.LOCAL_TR_REVIEW_EVIDENCE_SCHEMA
    core["evidence_content_sha256"] = "0" * 64
    write_json(artifacts["core_capture_evidence"], core)
    _refresh_input_artifact(
        evidence, "core_capture_evidence", artifacts["core_capture_evidence"]
    )
    _rehash_evidence(evidence)
    write_json(evidence_path, evidence)
    monkeypatch.setattr(
        review_topdown,
        "_strict_core_capture_errors",
        _STRICT_CORE_CAPTURE_ERRORS,
    )

    errors = verify_topdown_review_evidence(evidence_path)

    assert any(
        error.startswith("core capture strict verification:")
        and (
            "evidence content SHA-256 mismatch" in error
            or "local-TR verifier rejected malformed evidence" in error
        )
        for error in errors
    )
    assert "top-down review evidence content hash differs" not in errors


def test_compose_cli_rejects_tampered_local_tr_core_before_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()
    write_json(
        capture / "evidence.json",
        {
            "schema": compose_topdown_cli.LOCAL_TR_REVIEW_EVIDENCE_SCHEMA,
            "review_only": True,
            "qualification_claim": False,
            "formal_view_ids": [],
            "review_view_ids": ["view0"],
            "sensor_contract": {"view_id": "view0"},
            "frames": _records(),
            "evidence_content_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(
        compose_topdown_cli, "verify_saved_capture_arrays", lambda _value, _root: []
    )

    with pytest.raises(
        TopdownReviewError, match="local-TR capture verification failed"
    ):
        compose_topdown_cli._verified_capture(capture)


def test_compose_removes_video_when_encoder_count_is_wrong(tmp_path: Path) -> None:
    rgb = np.zeros((3, 32, 32, 3), dtype=np.uint8)

    def short_encoder(frames: Any, destination: Path, *, fps: int) -> int:
        list(frames)
        destination.write_bytes(b"incomplete")
        return 2

    output = tmp_path / "bad_review"
    with pytest.raises(TopdownReviewError, match="frame count differs"):
        compose_topdown_review(
            rgb_frames=rgb,
            frame_records=_records(),
            room_camera_request=_room_request(),
            navmesh_binary_map=_navmesh(),
            navmesh_bounds=BOUNDS,
            output_dir=output,
            review_video_encode=short_encoder,
        )

    assert not (output / "rgb_topdown_review.mp4").exists()
    assert not (output / "navmesh_binary.npy").exists()
    assert not (output / "topdown_review_evidence.json").exists()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are unavailable",
)
def test_raw_review_encoder_streams_rgb_without_png_staging(tmp_path: Path) -> None:
    frames = [
        np.full((16, 18, 3), (index * 70, 25, 190), dtype=np.uint8)
        for index in range(3)
    ]
    output = tmp_path / "streamed.mp4"

    assert encode_review_rgb_frames(frames, output, fps=3) == 3

    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,pix_fmt,nb_read_frames",
                "-of",
                "json",
                str(output),
            ],
            text=True,
        )
    )["streams"][0]
    assert probe == {
        "width": 18,
        "height": 16,
        "pix_fmt": "yuv420p",
        "nb_read_frames": "3",
    }
    assert not list(tmp_path.glob("*.png"))
    assert not list(tmp_path.glob(".streamed.*.mp4"))


def test_encoder_rejects_shape_drift_and_cleans_partial_output(tmp_path: Path) -> None:
    frames = [
        np.zeros((16, 18, 3), dtype=np.uint8),
        np.zeros((18, 18, 3), dtype=np.uint8),
    ]
    output = tmp_path / "bad.mp4"

    with pytest.raises(TopdownReviewError, match="shape changed"):
        encode_review_rgb_frames(frames, output, fps=3)

    assert not output.exists()
    assert not list(tmp_path.glob(".bad.*.mp4"))


def test_encoder_race_does_not_replace_or_delete_concurrent_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "raced.mp4"
    sentinel = b"concurrent writer sentinel"

    def raced_frames() -> Any:
        output.write_bytes(sentinel)
        yield np.zeros((16, 18, 3), dtype=np.uint8)

    with pytest.raises(TopdownReviewError, match="appeared during encoding"):
        encode_review_rgb_frames(raced_frames(), output, fps=3)

    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(".raced.*.mp4"))


def test_evidence_race_preserves_concurrent_sentinel_and_cleans_owned_outputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "review"
    evidence_path = output / "topdown_review_evidence.json"
    sentinel = b"concurrent evidence sentinel"

    def raced_encoder(frames: Any, destination: Path, *, fps: int) -> int:
        materialized = list(frames)
        destination.write_bytes(b"owned fake video")
        evidence_path.write_bytes(sentinel)
        return len(materialized)

    with pytest.raises(TopdownReviewError, match="JSON output already exists"):
        compose_topdown_review(
            rgb_frames=_rgb_stack(),
            frame_records=_records(),
            room_camera_request=_room_request(),
            navmesh_binary_map=_navmesh(),
            navmesh_bounds=BOUNDS,
            output_dir=output,
            navmesh_metadata=_navmesh_metadata(),
            review_video_encode=raced_encoder,
        )

    assert evidence_path.read_bytes() == sentinel
    assert not (output / "rgb_topdown_review.mp4").exists()
    assert not (output / "navmesh_binary.npy").exists()
