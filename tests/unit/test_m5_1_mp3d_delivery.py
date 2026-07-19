from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avengine.m5_1.mp3d_delivery import (
    MP3D_REQUIRED_GATE_IDS,
    build_mp3d_overlay_tracks,
    listener_yaw_degrees,
    render_mp3d_topdown_frames,
    source_program_reuse_record,
    validate_room_visual_gate,
)
from avengine.m5_1.delivery import M51DeliveryError, source_actor_binding_record
from avengine.contracts.json_io import load_json
from avengine.m5_1.source_contracts import load_source_manifest


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPOSITORY / "examples/m5_1/legacy_apartment/source_manifest.json"
ROUTE_MANIFEST = REPOSITORY / "examples/m5_1/mp3d_articulated_review/route_manifest.json"


def _mp3d_bindings(
    human_semantic_id: int = 73000, dog_semantic_id: int = 73001
) -> dict[str, object]:
    source = load_source_manifest(SOURCE_MANIFEST)
    route = load_json(ROUTE_MANIFEST)
    route["semantic_ids"] = {
        "human0": human_semantic_id,
        "dog0": dog_semantic_id,
    }
    capture = {
        "actors": [
            {
                "actor_id": "human0",
                "actor_class": "human",
                "emitter_link": "Bip01 MJaw",
                "semantic_id": human_semantic_id,
            },
            {
                "actor_id": "dog0",
                "actor_class": "dog",
                "emitter_link": "beagle Xtra Mouth",
                "semantic_id": dog_semantic_id,
            },
        ]
    }
    return source_actor_binding_record(source, route, capture, room_family="mp3d")


def test_program_reuse_explicitly_excludes_legacy_spatial_assertions() -> None:
    source = load_source_manifest(SOURCE_MANIFEST)
    record = source_program_reuse_record(source)
    assert record["applicability"] == "taxonomy_event_timing_and_audio_program_only"
    assert record["legacy_spatial_trajectory_applicable"] is False
    assert record["legacy_observer_applicable"] is False
    assert record["legacy_source_and_clip_flags_applicable"] is False
    assert record["legacy_visual_provenance_applicable"] is False
    assert [item["source_id"] for item in record["sources"]] == [
        "source0",
        "source1",
    ]
    assert len(record["event_overlap_windows"]) == 3
    assert "visual_asset" not in str(record["sources"])
    assert "migration" not in str(record["sources"])
    assert len(record["record_content_sha256"]) == 64


def test_listener_yaw_uses_habitat_y_up_xyzw_quaternion() -> None:
    assert listener_yaw_degrees((0.0, 0.0, 0.0, 1.0)) == 0.0
    half = np.deg2rad(45.0)
    assert np.isclose(
        listener_yaw_degrees((0.0, np.sin(half), 0.0, np.cos(half))),
        90.0,
    )


def test_mp3d_tracks_use_capture_semantics_and_only_room_local_review_flags() -> None:
    source = load_source_manifest(SOURCE_MANIFEST)
    anchors = np.zeros((270, 3, 3), dtype=np.float64)
    anchors[:, 1] = (-4.6, 1.5, -3.0)
    anchors[:, 2] = (-3.7, 0.4, -3.0)
    semantic = np.zeros((270, 6, 8), dtype=np.uint32)
    semantic[:, 1:3, 1:3] = 73000
    semantic[:, 3:5, 5:7] = 73001
    bindings = _mp3d_bindings()
    gate = {
        "status": "pass",
        "qualification_claim": False,
        "gate_count": 14,
        "passed_gate_count": 14,
    }
    tracks = build_mp3d_overlay_tracks(
        source,
        anchor_positions_m=anchors,
        semantic_frames=semantic,
        clearance_m={"human0": np.full(270, 0.6), "dog0": np.full(270, 0.3)},
        gate_evidence=gate,
        source_actor_bindings=bindings,
    )
    assert tuple(track.source_id for track in tracks) == ("source0", "source1")
    assert tracks[0].current_event_by_frame[75] == "event_human_speech_001"
    assert tracks[1].current_event_by_frame[90] == "event_beagle_bark_001"
    assert tracks[0].true_flags == ("center_navmesh_pass", "visible_all_frames")
    assert "crosses_azimuth_zero" not in tracks[0].true_flags
    assert np.array_equal(tracks[0].main_marker_xy[0], [1.5, 1.5])
    assert np.array_equal(tracks[1].main_marker_xy[0], [5.5, 3.5])


def test_mp3d_gate_requires_exact_frozen_schema_and_fourteen_ids() -> None:
    route = load_json(ROUTE_MANIFEST)
    gate = {
        "schema": "avengine_m5_1_mp3d_mixed_visual_gate_v1",
        "status": "pass",
        "qualification_claim": False,
        "route_id": route["route_id"],
        "gate_count": 14,
        "passed_gate_count": 14,
        "gates": [
            {"gate_id": gate_id, "status": "pass"}
            for gate_id in sorted(MP3D_REQUIRED_GATE_IDS)
        ],
    }
    assert len(validate_room_visual_gate(gate, route, room_family="mp3d")) == 14

    gate["gates"][0]["gate_id"] = "invented_gate"
    with pytest.raises(M51DeliveryError, match="frozen 14"):
        validate_room_visual_gate(gate, route, room_family="mp3d")


def test_real_navmesh_topdown_renderer_has_exact_review_shape_and_progress() -> None:
    navmesh = np.zeros((40, 80), dtype=np.uint8)
    navmesh[5:35, 10:70] = 1
    human = np.linspace((-4.6, 0.07, -2.7), (-4.6, 0.07, -3.8), 3)
    dog = np.linspace((-3.7, 0.07, -2.7), (-3.7, 0.07, -3.8), 3)
    kwargs = dict(
        navmesh_binary_map=navmesh,
        navmesh_bounds_m=((-8.0, -1.0, -6.0), (0.0, 3.0, 2.0)),
        actor_center_paths_m={"human0": human, "dog0": dog},
        listener_position_m=(-4.15, 1.57, -1.25),
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        clearance_m={"human0": np.full(3, 0.6), "dog0": np.full(3, 0.3)},
        shared_island_id=1,
        source_actor_bindings=_mp3d_bindings(),
        size_wh=(640, 480),
    )
    first = render_mp3d_topdown_frames(**kwargs)
    second = render_mp3d_topdown_frames(**kwargs)
    assert first.shape == (3, 480, 640, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[-1])
    assert np.count_nonzero(first) > 0
