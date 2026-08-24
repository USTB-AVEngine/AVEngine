from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avengine.camera_placement import (
    DEFAULT_YAW_DEGREES,
    camera_height_variants,
    iter_declared_floor_camera_placements,
)
from avengine.contracts.json_io import load_json
from tools.m5 import build_mp3d_camera_placement_catalog as builder


ROOM_ID = "unknown_room_instance_007"


def test_declared_floor_iterator_is_target_agnostic_and_deterministic() -> None:
    samples = [[2.0, 0.1, 3.0], [1.0, 0.1, 2.0], [2.0, 0.1, 3.0]]
    heights = camera_height_variants(1.5)

    first = tuple(
        iter_declared_floor_camera_placements(
            samples,
            camera_center_heights_m=heights,
        )
    )
    second = tuple(
        iter_declared_floor_camera_placements(
            list(reversed(samples)),
            camera_center_heights_m=heights,
        )
    )

    assert first == second
    assert len(first) == 2 * 3 * 12
    assert {item.yaw_deg for item in first} == set(DEFAULT_YAW_DEGREES)
    assert {item.height_id for item in first} == {"low", "base", "high"}
    assert all("semantic" not in key for key in first[0].to_record())


def _write_inputs(tmp_path: Path):
    room = tmp_path / "room.json"
    request = tmp_path / "request.json"
    floor = tmp_path / "floor.json"
    room.write_text("{}", encoding="utf-8")
    request.write_text("{}", encoding="utf-8")
    floor.write_text(
        json.dumps(
            {
                "authority": "live_habitat_pathfinder_declared_navmesh_export",
                "camera_floor_samples_m": [
                    [2.0, 0.1, 3.0],
                    [1.0, 0.1, 2.0],
                ],
                "coordinate_space": "habitat_world_xyz_m",
                "episode_counted": False,
                "native_habitat_floor_export": True,
                "research_only": True,
                "room_id": ROOM_ID,
            }
        ),
        encoding="utf-8",
    )
    return room, request, floor


def _args(tmp_path: Path):
    room, request, floor = _write_inputs(tmp_path)
    output = tmp_path / "catalog"
    return (
        builder.parse_args(
            [
                "--room-manifest",
                str(room),
                "--base-m1-request",
                str(request),
                "--camera-floor-samples",
                str(floor),
                "--output",
                str(output),
            ]
        ),
        output,
        floor,
    )


def test_builder_writes_complete_unfiltered_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, output, _floor = _args(tmp_path)
    inputs = SimpleNamespace(
        room={
            "room_id": ROOM_ID,
            "navigation": {"agent_height_m": 1.5},
        },
        request={"request_id": "base-request"},
    )
    monkeypatch.setattr(builder, "load_and_validate_inputs", lambda *_args: inputs)
    monkeypatch.setattr(builder, "_current_mp3d_room_error", lambda _inputs: None)

    assert builder.run(args) == output
    record = load_json(output / "camera_placement_catalog.json")

    assert record["room_id"] == ROOM_ID
    assert record["candidate_filtering"] == "none"
    assert record["studio_used"] is False
    assert record["input_floor_sample_count"] == 2
    assert record["canonical_floor_sample_count"] == 2
    assert record["placement_count"] == 2 * 3 * 12
    assert record["yaw_degrees"] == list(DEFAULT_YAW_DEGREES)
    assert all(item["semantic_room_label"] is None for item in record["placements"])
    assert all(item["semantic_target"] is None for item in record["placements"])
    assert all(
        item["source_path_join_status"] == "pending" for item in record["placements"]
    )
    assert (output / "README.md").is_file()

    with pytest.raises(
        builder.MP3DCameraCatalogError, match="refusing to replace output"
    ):
        builder.run(args)


def test_builder_rejects_floor_export_from_another_room(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, output, floor = _args(tmp_path)
    value = json.loads(floor.read_text(encoding="utf-8"))
    value["room_id"] = "different_room"
    floor.write_text(json.dumps(value), encoding="utf-8")
    inputs = SimpleNamespace(
        room={
            "room_id": ROOM_ID,
            "navigation": {"agent_height_m": 1.5},
        },
        request={"request_id": "base-request"},
    )
    monkeypatch.setattr(builder, "load_and_validate_inputs", lambda *_args: inputs)
    monkeypatch.setattr(builder, "_current_mp3d_room_error", lambda _inputs: None)

    with pytest.raises(
        builder.MP3DCameraCatalogError, match="does not match one native research room"
    ):
        builder.run(args)

    assert not output.exists()
