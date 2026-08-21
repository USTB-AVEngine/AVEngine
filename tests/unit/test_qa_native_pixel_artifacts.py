from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/finalize_native_pixel_artifacts.py"
SPEC = importlib.util.spec_from_file_location("finalize_native_pixel_artifacts", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _fixture(tmp_path: Path, *, extra_mask_key: bool = False) -> Path:
    depth = tmp_path / "depth.npz"
    np.savez_compressed(
        depth,
        normal_depth_m=np.ones((1, 2, 3), dtype=np.float16),
        target_only_source1_depth_m=np.ones((1, 2, 3), dtype=np.float16),
        target_only_source2_depth_m=np.ones((1, 2, 3), dtype=np.float16),
    )
    masks = tmp_path / "masks.npz"
    arrays = {
        "depth_derived_modal_semantic": np.zeros((1, 2, 3), dtype=np.uint8),
        "modal_visible_source1": np.zeros((1, 2, 3), dtype=bool),
        "modal_visible_source2": np.zeros((1, 2, 3), dtype=bool),
        "target_only_source1": np.zeros((1, 2, 3), dtype=np.uint8),
        "target_only_source2": np.zeros((1, 2, 3), dtype=np.uint8),
    }
    if extra_mask_key:
        arrays["uncontracted"] = np.zeros((1,), dtype=np.uint8)
    np.savez_compressed(masks, **arrays)
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000000.png").write_bytes(b"png")
    descriptors = tmp_path / "descriptors.json"
    descriptors.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": TOOL.SCHEMA,
                "status": "pass",
                "artifacts": {
                    "metric_depth": str(depth),
                    "pixel_masks": str(masks),
                    "rgb_frames": str(frames),
                    "object_id_descriptors": str(descriptors),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_finalize_records_every_file_and_frame_inventory(tmp_path: Path) -> None:
    manifest_path = _fixture(tmp_path)
    result = TOOL.finalize(manifest_path)

    assert set(result["sha256"]) == {
        "metric_depth",
        "pixel_masks",
        "object_id_descriptors",
    }
    frame_record = result["artifact_records"]["rgb_frames"]
    assert frame_record["kind"] == "directory"
    assert frame_record["file_count"] == 1
    assert len(frame_record["inventory_root_sha256"]) == 64


def test_finalize_rejects_extra_mask_array_key(tmp_path: Path) -> None:
    manifest_path = _fixture(tmp_path, extra_mask_key=True)

    with pytest.raises(RuntimeError, match="exact contracted keys"):
        TOOL.finalize(manifest_path)
