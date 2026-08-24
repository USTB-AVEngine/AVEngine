from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import load_json
from tools.m5 import describe_mp3d_viewpoint_tracks as descriptor


ROOM_ID = "habitat_mp3d_example_room"
VIEWPOINT_ID = "mp3d_view_01"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(tmp_path: Path, *, frames: int = 75, probe_status: str = "pass"):
    bank_root = tmp_path / "bank"
    request_path = bank_root / "requests" / f"{VIEWPOINT_ID}.json"
    transform = {
        "translation_m": [0.0, 1.5, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    request = {
        "schema": "avengine_m1_capture_request_v1",
        "request_id": "request-view-01",
        "room_id": ROOM_ID,
        "primary_camera_rig": {
            "rig_id": "camera_rig_0",
            "view_id": "view0",
            "world_from_rig": transform,
        },
        "listener": {
            "listener_id": "listener0",
            "attached_to": "camera_rig_0",
            "rig_from_listener": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    }
    _write_json(request_path, request)
    bank_path = bank_root / "viewpoint_bank.json"
    _write_json(
        bank_path,
        {
            "research_only": True,
            "episode_counted": False,
            "room_id": ROOM_ID,
            "viewpoints": [{"viewpoint_id": VIEWPOINT_ID}],
            "m1_capture_request_paths": {VIEWPOINT_ID: f"requests/{VIEWPOINT_ID}.json"},
        },
    )

    actor0 = np.zeros((frames, 3), dtype=np.float64)
    actor1 = np.zeros((frames, 3), dtype=np.float64)
    actor0[:, 2] = -4.0
    actor1[:, 2] = -5.0
    if frames >= 75:
        actor0[20:60, 0] = np.linspace(-1.0, 1.0, 40)
        actor0[60:, 0] = 1.0
        actor1[15:55, 0] = np.linspace(1.0, -1.0, 40)
        actor1[55:, 0] = -1.0
    route_path = tmp_path / "route.json"
    _write_json(
        route_path,
        {
            "research_only": True,
            "episode_counted": False,
            "provenance": {"new_request_room_id": ROOM_ID},
            "current_visual_realization": {
                "actor_ids": ["actor0", "actor1"],
                "actor_skin_root_paths_m": {
                    "actor0": actor0.tolist(),
                    "actor1": actor1.tolist(),
                },
            },
        },
    )
    probe_root = tmp_path / "probes"
    receipt_path = probe_root / VIEWPOINT_ID / "receipt.json"
    _write_json(
        receipt_path,
        {
            "status": probe_status,
            "room_id": ROOM_ID,
            "request_id": "request-view-01",
            "navigation": {"passed": True, "expected_floor_passed": True},
            "camera_listener_rig": {
                "listener_co_located": True,
                "world_from_rig": transform,
            },
            "topdown": {"required": True, "passed": True},
            "observations": {
                "rgb": {"statistics": {"color_standard_deviation": 1.0}},
                "depth": {"statistics": {"finite_positive_fraction": 1.0}},
                "semantic": {"statistics": {"unique_id_count": 3}},
            },
        },
    )
    return bank_path, route_path, probe_root, actor0, actor1


def _args(tmp_path: Path, *, frames: int = 75, probe_status: str = "pass"):
    bank, route, probes, actor0, actor1 = _inputs(
        tmp_path, frames=frames, probe_status=probe_status
    )
    output = tmp_path / "output"
    args = descriptor.parse_args(
        [
            "--viewpoint-bank",
            str(bank),
            "--route-explanation",
            str(route),
            "--probe-root",
            str(probes),
            "--output",
            str(output),
        ]
    )
    return args, output, actor0, actor1


def test_run_preserves_exact_mp3d_tracks_and_pending_visibility(tmp_path: Path) -> None:
    args, output, actor0, actor1 = _args(tmp_path)

    assert descriptor.run(args) == output
    result = load_json(output / "viewpoint_raw_track_descriptors.json")

    assert result["candidate_filtering"] == "none"
    assert result["raw_legal_candidate_catalog_preserved"] is True
    assert result["viewpoint_count"] == 1
    assert result["track_authority"]["semantics"] == (
        "visual_actor_skin_root_not_audio_emitter"
    )
    track = result["viewpoints"][0]["track_descriptor"]
    assert track["track_provenance"] == (
        "caller_supplied_per_frame_positions_no_interpolation"
    )
    assert track["frame_count"] == 75
    assert track["axis_mapping"]["horizontal_axis_indices"] == [0, 2]
    assert track["axis_mapping"]["vertical_axis_index"] == 1
    assert track["frames"][19]["source1"]["position_xyz"] == actor0[19].tolist()
    assert track["frames"][20]["source1"]["position_xyz"] == actor0[20].tolist()
    assert track["frames"][54]["source2"]["position_xyz"] == actor1[54].tolist()
    assert track["frames"][55]["source2"]["position_xyz"] == actor1[55].tolist()
    assert track["source_identity_binding"] == {
        "source1_id": "actor0",
        "source2_id": "actor1",
        "identity_preserved_through_crossing_or_swap": True,
    }
    assert track["native_projection_visibility"] == {
        "status": "pending_same_renderer_native_execution",
        "projection_by_frame": None,
        "in_frame_by_frame": None,
        "occlusion_by_frame": None,
        "visible_pixels_by_frame": None,
    }
    assert track["legality"]["status"] == "not_evaluated_by_track_descriptor"
    assert track["difficulty"]["candidate_filtering"] == "disabled"
    assert result["viewpoints"][0]["native_static_probe"]["status"] == "pass"
    assert (output / "README.md").is_file()

    with pytest.raises(
        descriptor.MP3DTrackDescriptorError, match="refusing to replace output"
    ):
        descriptor.run(args)


def test_run_rejects_nonpassing_native_probe(tmp_path: Path) -> None:
    args, output, _actor0, _actor1 = _args(tmp_path, probe_status="fail")

    with pytest.raises(
        descriptor.MP3DTrackDescriptorError, match="native probe does not match"
    ):
        descriptor.run(args)

    assert not output.exists()


def test_run_rejects_non_75_frame_route_without_output(tmp_path: Path) -> None:
    args, output, _actor0, _actor1 = _args(tmp_path, frames=74)

    with pytest.raises(descriptor.MP3DTrackDescriptorError, match=r"finite \[75,3\]"):
        descriptor.run(args)

    assert not output.exists()


def test_bank_request_path_cannot_escape_bank_root(tmp_path: Path) -> None:
    args, output, _actor0, _actor1 = _args(tmp_path)
    bank = json.loads(args.viewpoint_bank.read_text(encoding="utf-8"))
    bank["m1_capture_request_paths"][VIEWPOINT_ID] = "../route.json"
    args.viewpoint_bank.write_text(json.dumps(bank), encoding="utf-8")

    with pytest.raises(
        descriptor.MP3DTrackDescriptorError, match="stay under the bank root"
    ):
        descriptor.run(args)

    assert not output.exists()
