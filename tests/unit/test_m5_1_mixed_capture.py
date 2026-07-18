from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m2.glb import extract_actions, extract_skins, parse_glb
from avengine.m2.glb_write import build_glb
from avengine.m5_1.human_runtime import (
    ARMATURE_NODE_NAME,
    HEAD_LINK_NAME,
    MOUTH_LINK_NAME,
    SYNTHETIC_ROOT_NAME,
    promote_rocketbox_skin_ancestors,
)
from avengine.m5_1.mixed_capture import (
    MixedCaptureError,
    _actor_heading_evidence,
    _beagle_anatomical_forward_binding,
    _continuous_beagle_walk_states,
    _human_anatomical_forward_binding,
    capture_legacy_route,
    trajectory_world_matrices,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _append(binary: bytearray, values: np.ndarray) -> tuple[int, int]:
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    payload = np.ascontiguousarray(values).tobytes(order="C")
    binary.extend(payload)
    return offset, len(payload)


def _rocketbox_shape_fixture() -> bytes:
    binary = bytearray()
    inverse_offset, inverse_length = _append(
        binary,
        np.tile(np.eye(4, dtype=np.dtype("<f4")).T.reshape(1, 16), (80, 1)),
    )
    times_offset, times_length = _append(
        binary, np.asarray([0.0, 1.0], dtype=np.dtype("<f4"))
    )
    rotations_offset, rotations_length = _append(
        binary,
        np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.dtype("<f4")),
    )

    names = ["Bip01 Pelvis", HEAD_LINK_NAME, MOUTH_LINK_NAME]
    names.extend(f"Bip01 Fixture {index:02d}" for index in range(77))
    nodes: list[dict[str, object]] = []
    for index, name in enumerate(names):
        node: dict[str, object] = {"name": name}
        if index + 1 < len(names):
            node["children"] = [index + 1]
        nodes.append(node)
    nodes.append({"name": ARMATURE_NODE_NAME, "children": [0]})
    document: dict[str, object] = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [80]}],
        "nodes": nodes,
        "skins": [
            {
                "name": ARMATURE_NODE_NAME,
                "joints": list(range(80)),
                "inverseBindMatrices": 0,
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": inverse_offset, "byteLength": inverse_length},
            {"buffer": 0, "byteOffset": times_offset, "byteLength": times_length},
            {
                "buffer": 0,
                "byteOffset": rotations_offset,
                "byteLength": rotations_length,
            },
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 80, "type": "MAT4"},
            {"bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC4"},
        ],
        "animations": [
            {
                "name": name,
                "samplers": [
                    {"input": 1, "output": 2, "interpolation": "LINEAR"}
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 80, "path": "rotation"}}
                ],
            }
            for name in ("Walking", "Standing_Idle")
        ],
    }
    return build_glb(document, binary)


def test_promote_rocketbox_appends_zero_weight_ancestors_without_reindexing() -> None:
    source = parse_glb(_rocketbox_shape_fixture())
    before = extract_skins(source)[0]
    promoted = parse_glb(promote_rocketbox_skin_ancestors(source))
    after = extract_skins(promoted)[0]

    assert tuple(joint.node_index for joint in after.joints[:80]) == tuple(
        joint.node_index for joint in before.joints
    )
    assert [joint.name for joint in after.joints[-2:]] == [
        ARMATURE_NODE_NAME,
        SYNTHETIC_ROOT_NAME,
    ]
    assert after.skeleton_node_index == after.joints[-1].node_index
    assert len(after.inverse_bind_matrices or ()) == 82
    assert [action.name for action in extract_actions(promoted)] == ["Walking", "Idle"]
    assert all(
        any(
            channel.target_node_name == SYNTHETIC_ROOT_NAME
            and channel.target_path == "rotation"
            for channel in action.channels
        )
        for action in extract_actions(promoted)
    )


@pytest.mark.parametrize(
    "local_forward_axis",
    [
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
)
def test_trajectory_world_matrices_preserve_points_and_align_asset_forward(
    local_forward_axis: tuple[float, float, float],
) -> None:
    points = np.column_stack(
        (
            np.linspace(-1.0, 2.0, 270),
            np.full(270, 0.271),
            np.linspace(3.0, -4.0, 270),
        )
    )
    matrices = trajectory_world_matrices(
        points, local_forward_axis=local_forward_axis
    )
    expected_forward = points[-1] - points[0]
    expected_forward[1] = 0.0
    expected_forward /= np.linalg.norm(expected_forward)

    assert matrices.shape == (270, 4, 4)
    assert np.array_equal(matrices[:, :3, 3], points)
    assert np.allclose(
        matrices[:, :3, :3] @ np.asarray(local_forward_axis), expected_forward
    )
    assert np.allclose(np.linalg.det(matrices[:, :3, :3]), 1.0)


def test_trajectory_world_matrices_reject_vertical_or_implicit_forward_axis() -> None:
    points = np.column_stack(
        (
            np.linspace(0.0, 1.0, 270),
            np.zeros(270),
            np.zeros(270),
        )
    )
    with pytest.raises(TypeError, match="local_forward_axis"):
        trajectory_world_matrices(points)  # type: ignore[call-arg]
    with pytest.raises(MixedCaptureError, match="must be horizontal"):
        trajectory_world_matrices(
            points, local_forward_axis=(0.0, 1.0, 0.0)
        )


def test_actor_heading_evidence_retains_every_frame_and_fails_closed() -> None:
    points = np.column_stack(
        (
            np.linspace(0.0, 2.0, 270),
            np.full(270, 0.271),
            np.zeros(270),
        )
    )
    matrices = trajectory_world_matrices(
        points, local_forward_axis=(0.0, 0.0, 1.0)
    )
    evidence = _actor_heading_evidence(
        actor_id="human0",
        points_m=points,
        actor_world_matrices=matrices,
        local_forward_axis=(0.0, 0.0, 1.0),
        binding_source={"kind": "unit_test", "declared_axis": "+Z"},
    )

    assert evidence["status"] == "pass"
    assert evidence["gate"]["all_frames_passed"] is True
    assert len(evidence["frames"]) == 270
    assert all(frame["passed"] for frame in evidence["frames"])
    assert evidence["frames"][137]["path_tangent_world"] == [1.0, 0.0, 0.0]

    wrong = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 270, axis=0)
    wrong[:, :3, 3] = points
    with pytest.raises(MixedCaptureError, match="heading gate failed"):
        _actor_heading_evidence(
            actor_id="human0",
            points_m=points,
            actor_world_matrices=wrong,
            local_forward_axis=(0.0, 0.0, 1.0),
            binding_source={"kind": "unit_test", "declared_axis": "+Z"},
        )


def test_human_forward_binding_reads_hash_closed_runtime_metadata(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema": "avengine_m5_1_rocketbox_human_runtime_v1",
        "status": "pass",
        "anatomical_frame": {
            "actor_up_axis": "+Y",
            "actor_forward_axis": "+Z",
            "source": "fixture_rest_pose_head_to_mjaw_axis",
        },
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    path = tmp_path / "human_runtime_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    axis, source = _human_anatomical_forward_binding(path)

    assert axis == (0.0, 0.0, 1.0)
    assert source["declared_axis"] == "+Z"
    assert source["sha256"] == sha256_file(path)


def test_beagle_forward_binding_reads_manifest_bound_animation_qa(
    tmp_path: Path,
) -> None:
    qa = {
        "schema": "avengine_m2_animation_qa_v1",
        "status": "pass",
        "semantic_terminal_motion": {
            "actor_up_axis": "+Y",
            "source_facing_axis_in_actor_frame": "+X",
        },
    }
    qa_path = tmp_path / "animation.json"
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    asset = {
        "files": [
            {
                "role": "animation_qa",
                "path": qa_path.name,
                "byte_size": qa_path.stat().st_size,
                "sha256": sha256_file(qa_path),
            }
        ]
    }

    axis, source = _beagle_anatomical_forward_binding(
        asset=asset, asset_manifest_path=tmp_path / "asset_manifest.json"
    )

    assert axis == (1.0, 0.0, 0.0)
    assert source["declared_axis"] == "+X"
    assert source["sha256"] == sha256_file(qa_path)

    qa_path.write_text(json.dumps({**qa, "status": "fail"}), encoding="utf-8")
    with pytest.raises(MixedCaptureError, match="bytes differ"):
        _beagle_anatomical_forward_binding(
            asset=asset, asset_manifest_path=tmp_path / "asset_manifest.json"
        )


@dataclass(frozen=True)
class _State:
    action_id: str


def test_beagle_selection_accepts_only_one_continuous_45_state_walk_block() -> None:
    states = tuple([_State("idle")] * 15 + [_State("walk")] * 45 + [_State("idle")] * 15)
    indices, selected = _continuous_beagle_walk_states(states)
    assert indices == tuple(range(15, 60))
    assert len(selected) == 45
    assert all(state.action_id == "walk" for state in selected)

    split = tuple([_State("walk")] * 22 + [_State("idle")] + [_State("walk")] * 23)
    with pytest.raises(MixedCaptureError, match="continuous 45-state walk block"):
        _continuous_beagle_walk_states(split)


def test_legacy_wrapper_consumes_manifest_habitat_paths_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_path = REPOSITORY_ROOT / "examples/m5_1/legacy_apartment/route_manifest.json"
    route = json.loads(route_path.read_text(encoding="utf-8"))
    sentinel = object()
    retained: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> object:
        retained.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "avengine.m5_1.mixed_capture.capture_human_beagle_paths", fake_capture
    )
    result = capture_legacy_route(
        route_manifest_path=route_path,
        room_manifest_path="room.json",
        m1_request_path="request.json",
        human_runtime_glb_path="human.glb",
        beagle_animal_manifest_path="beagle.json",
        beagle_m2_request_path="beagle_request.json",
        output_dir="output",
    )

    assert result is sentinel
    assert retained["human_root_path_m"] == route["routes"]["human_path"][
        "habitat_trajectory_m"
    ]
    assert retained["beagle_root_path_m"] == route["routes"]["dog_path"][
        "habitat_trajectory_m"
    ]
    assert retained["require_legacy_camera"] is True
    assert retained["route_provenance"]["path_consumption"] == (
        "verbatim_manifest_routes_habitat_trajectory_m"
    )
