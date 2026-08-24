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


def _inputs(
    tmp_path: Path,
    *,
    frames: int = 75,
    probe_status: str = "pass",
    with_dynamic: bool = False,
):
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
            "topdown": {
                "reports": [{"qa_id": "navmesh_topdown", "navigable_pixel_count": 10}],
                "checks": [
                    {
                        "check_id": "qa_navmesh_topdown",
                        "status": "pass",
                    }
                ],
            },
            "observations": {
                "rgb": {"statistics": {"color_standard_deviation": 1.0}},
                "depth": {"statistics": {"finite_positive_fraction": 1.0}},
                "semantic": {"statistics": {"unique_id_count": 3}},
            },
        },
    )
    dynamic_roots = []
    runtime_sources = None
    runtime_visibility = None
    if with_dynamic:
        dynamic_root = tmp_path / "dynamic"
        runtime_sources = np.stack((actor0, actor1), axis=1)
        runtime_sources[:, :, 1] += 0.4
        runtime_visibility = np.stack(
            (
                np.arange(frames, dtype=np.int64),
                np.arange(frames, dtype=np.int64)[::-1],
            ),
            axis=1,
        )
        arrays = dynamic_root / "arrays"
        arrays.mkdir(parents=True)
        np.save(arrays / "source_positions_m.npy", runtime_sources)
        np.save(arrays / "semantic_visibility_pixels.npy", runtime_visibility)
        _write_json(
            dynamic_root / "research_receipt.json",
            {
                "status": "research_only",
                "research_only": True,
                "episode_counted": False,
                "inputs": {"m1_request": str(request_path.resolve())},
                "selected_room": {"room_id": ROOM_ID},
                "capture": {"native_habitat_started": True, "frame_count": frames},
                "actors": {
                    "actor_ids": ["actor0", "actor1"],
                    "source_ids": ["source0", "source1"],
                },
                "artifacts": {
                    "source_positions_m": "arrays/source_positions_m.npy",
                    "semantic_visibility_pixels": (
                        "arrays/semantic_visibility_pixels.npy"
                    ),
                },
                "visibility": {
                    "candidate_filtering": "disabled",
                    "per_actor": {
                        "actor0": {
                            "per_frame_visible_pixels": runtime_visibility[
                                :, 0
                            ].tolist()
                        },
                        "actor1": {
                            "per_frame_visible_pixels": runtime_visibility[
                                :, 1
                            ].tolist()
                        },
                    },
                },
            },
        )
        dynamic_roots.append(dynamic_root)
    return (
        bank_path,
        route_path,
        probe_root,
        actor0,
        actor1,
        dynamic_roots,
        runtime_sources,
        runtime_visibility,
    )


def _args(
    tmp_path: Path,
    *,
    frames: int = 75,
    probe_status: str = "pass",
    with_dynamic: bool = False,
):
    bank, route, probes, actor0, actor1, dynamics, sources, visibility = _inputs(
        tmp_path,
        frames=frames,
        probe_status=probe_status,
        with_dynamic=with_dynamic,
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
            *[
                value
                for root in dynamics
                for value in ("--dynamic-capture-dir", str(root))
            ],
            "--output",
            str(output),
        ]
    )
    return args, output, actor0, actor1, sources, visibility


def test_run_preserves_exact_mp3d_tracks_and_pending_visibility(tmp_path: Path) -> None:
    args, output, actor0, actor1, _sources, _visibility = _args(tmp_path)

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
    normalized_topdown = result["viewpoints"][0]["native_static_probe"]["topdown"]
    assert normalized_topdown["required"] is True
    assert normalized_topdown["passed"] is True
    assert normalized_topdown["summary_fields_derived_from_reports_checks"] is True
    assert (output / "README.md").is_file()

    with pytest.raises(
        descriptor.MP3DTrackDescriptorError, match="refusing to replace output"
    ):
        descriptor.run(args)


def test_run_binds_runtime_emitter_tracks_and_native_visible_pixels(
    tmp_path: Path,
) -> None:
    args, output, _actor0, _actor1, sources, visibility = _args(
        tmp_path, with_dynamic=True
    )

    descriptor.run(args)
    result = load_json(output / "viewpoint_raw_track_descriptors.json")
    assert result["artifact_status"] == (
        "explicit_actor_root_and_runtime_emitter_tracks_described"
    )
    runtime = result["viewpoints"][0]["runtime_emitter_track"]
    assert runtime["runtime_binding_status"] == "pass"
    assert runtime["source_ids"] == ["source0", "source1"]
    track = runtime["track_descriptor"]
    assert [frame["source1"]["position_xyz"] for frame in track["frames"]] == (
        sources[:, 0, :].tolist()
    )
    assert [frame["source2"]["position_xyz"] for frame in track["frames"]] == (
        sources[:, 1, :].tolist()
    )
    assert [frame["source1"]["visible_pixels"] for frame in track["frames"]] == (
        visibility[:, 0].tolist()
    )
    assert track["native_projection_visibility"] == {
        "status": "native_visible_pixels_recorded_projection_occlusion_pending",
        "projection_by_frame": None,
        "in_frame_by_frame": None,
        "occlusion_by_frame": None,
        "visible_pixels_by_frame": {
            "source0": visibility[:, 0].tolist(),
            "source1": visibility[:, 1].tolist(),
        },
    }


def test_run_rejects_nonpassing_native_probe(tmp_path: Path) -> None:
    args, output, *_unused = _args(tmp_path, probe_status="fail")

    with pytest.raises(
        descriptor.MP3DTrackDescriptorError, match="native probe does not match"
    ):
        descriptor.run(args)

    assert not output.exists()


def test_run_rejects_non_75_frame_route_without_output(tmp_path: Path) -> None:
    args, output, *_unused = _args(tmp_path, frames=74)

    with pytest.raises(descriptor.MP3DTrackDescriptorError, match=r"finite \[75,3\]"):
        descriptor.run(args)

    assert not output.exists()


def test_bank_request_path_cannot_escape_bank_root(tmp_path: Path) -> None:
    args, output, *_unused = _args(tmp_path)
    bank = json.loads(args.viewpoint_bank.read_text(encoding="utf-8"))
    bank["m1_capture_request_paths"][VIEWPOINT_ID] = "../route.json"
    args.viewpoint_bank.write_text(json.dumps(bank), encoding="utf-8")

    with pytest.raises(
        descriptor.MP3DTrackDescriptorError, match="stay under the bank root"
    ):
        descriptor.run(args)

    assert not output.exists()
