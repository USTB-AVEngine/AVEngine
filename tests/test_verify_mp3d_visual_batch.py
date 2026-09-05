from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from verify_mp3d_visual_batch import (  # noqa: E402
    MP3DVisualVerificationError,
    verify_point,
)


def _capture(root: Path) -> None:
    root.mkdir()
    np.save(root / "rgb.npy", np.zeros((2, 1, 2, 3), dtype=np.uint8))
    np.save(root / "depth.npy", np.ones((2, 1, 2), dtype=np.float32))
    semantic = np.asarray([[[11, 0]], [[11, 0]]], dtype=np.int32)
    np.save(root / "semantic.npy", semantic)
    identity = {
        "translation_m": [0.0, 1.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    records = []
    for index in range(2):
        records.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "camera_readback": {
                    "world_time_seconds": 0.0,
                    "agent": identity,
                },
                "actor_readbacks": [
                    {
                        "actor_id": "a1",
                        "source_slot_id": "source1",
                        "source_endpoint_id": "e1",
                        "semantic_id": 11,
                        "planned_route_center_m": [100.0, 0.0, 0.0],
                        "world_from_skin_root": identity,
                        "emitter_world_position_m": [1.0 + index, 0.0, 0.0],
                        "semantic_pixel_count": 1,
                    },
                    {
                        "actor_id": "a2",
                        "source_slot_id": "source2",
                        "source_endpoint_id": "e2",
                        "semantic_id": 12,
                        "planned_route_center_m": [100.0, 0.0, 0.0],
                        "world_from_skin_root": identity,
                        "emitter_world_position_m": [2.0 + index, 0.0, 0.0],
                        "semantic_pixel_count": 0,
                    },
                ],
                "source_positions_m": [[1.0 + index, 0.0, 0.0], [2.0 + index, 0.0, 0.0]],
            }
        )
    (root / "frame_records.json").write_text(
        json.dumps({"source_endpoint_ids": ["e1", "e2"], "frames": records}),
        encoding="utf-8",
    )
    (root / "research_receipt.json").write_text(
        json.dumps(
            {
                "schema": "avengine_mp3d_multi_actor_native_capture_v1",
                "artifact_role": "observed_native_habitat_capture",
                "research_only": True,
                "episode_counted": False,
                "capture": {
                    "frame_count": 2,
                    "frame_rate_hz": 15,
                    "ticks_per_frame": 3200,
                    "native_habitat_started": True,
                },
                "actors": [
                    {"source_slot_id": "source1", "source_endpoint_id": "e1"},
                    {"source_slot_id": "source2", "source_endpoint_id": "e2"},
                ],
                "artifacts": {
                    "rgb": "rgb.npy",
                    "depth": "depth.npy",
                    "semantic": "semantic.npy",
                },
            }
        ),
        encoding="utf-8",
    )


def test_verifier_uses_observed_emitter_positions_and_array_truth(tmp_path: Path):
    point = tmp_path / "point"
    _capture(point)
    result = verify_point("point", point)
    assert result["status"] == "pass"
    assert result["frame_count"] == 2
    records = json.loads((point / "frame_records.json").read_text())
    records["frames"][0]["source_positions_m"][0] = [100.0, 0.0, 0.0]
    (point / "frame_records.json").write_text(json.dumps(records))
    with np.testing.assert_raises(MP3DVisualVerificationError):
        verify_point("point", point)


def test_receipt_m1_is_checked_and_world_time_may_advance(tmp_path: Path):
    point = tmp_path / "point"
    _capture(point)
    m1 = tmp_path / "m1.json"
    m1.write_text(json.dumps({
        "primary_camera_rig": {
            "world_from_rig": {
                "translation_m": [0.0, 1.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        }
    }))
    receipt_path = point / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"] = {"m1_request": str(m1)}
    receipt_path.write_text(json.dumps(receipt))
    records_path = point / "frame_records.json"
    records = json.loads(records_path.read_text())
    records["frames"][1]["camera_readback"]["world_time_seconds"] = 0.1
    records_path.write_text(json.dumps(records))
    assert verify_point("point", point)["status"] == "pass"

    records["frames"][1]["camera_readback"]["agent"]["translation_m"][0] = 0.1
    records_path.write_text(json.dumps(records))
    with pytest.raises(MP3DVisualVerificationError, match="declared M1 rig"):
        verify_point("point", point)


def test_verifier_accepts_one_configured_actor(tmp_path: Path):
    point = tmp_path / "point"
    _capture(point)
    records_path = point / "frame_records.json"
    records = json.loads(records_path.read_text())
    records["source_endpoint_ids"] = ["e1"]
    for frame in records["frames"]:
        frame["actor_readbacks"] = frame["actor_readbacks"][:1]
        frame["source_positions_m"] = frame["source_positions_m"][:1]
    records_path.write_text(json.dumps(records))

    receipt_path = point / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["actors"] = receipt["actors"][:1]
    receipt_path.write_text(json.dumps(receipt))

    result = verify_point("point", point)
    assert result["status"] == "pass"
    assert result["actor_count"] == 1



def test_verifier_binds_each_frame_actor_to_receipt_endpoint(tmp_path: Path):
    point = tmp_path / "point"
    _capture(point)
    records_path = point / "frame_records.json"
    records = json.loads(records_path.read_text())
    for frame in records["frames"]:
        first, second = frame["actor_readbacks"]
        first["source_endpoint_id"], second["source_endpoint_id"] = (
            second["source_endpoint_id"],
            first["source_endpoint_id"],
        )
    records_path.write_text(json.dumps(records))

    with pytest.raises(
        MP3DVisualVerificationError,
        match="source_endpoint_id differs from the receipt",
    ):
        verify_point("point", point)



def test_verifier_rejects_symlinked_capture_root(tmp_path: Path):
    point = tmp_path / "point"
    _capture(point)
    alias = tmp_path / "point-link"
    alias.symlink_to(point, target_is_directory=True)

    with pytest.raises(
        MP3DVisualVerificationError,
        match="root must not be a symlink",
    ):
        verify_point("point", alias)
