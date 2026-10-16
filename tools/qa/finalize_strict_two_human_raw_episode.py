#!/usr/bin/env python3
"""CPU-only finalizer for one atomically published strict full75 raw spool."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path(__file__).with_name("run_strict_two_human_full75_room_batch.py")
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "strict2h_room_batch_contract_finalizer", CONTRACT_PATH
)
if CONTRACT_SPEC is None or CONTRACT_SPEC.loader is None:
    raise RuntimeError(f"cannot import batch contract: {CONTRACT_PATH}")
CONTRACT = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = CONTRACT
CONTRACT_SPEC.loader.exec_module(CONTRACT)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_native_backend() -> Any:
    path = REPOSITORY / "tools/qa/capture_spear_native_pixel_episode.py"
    _require(path.is_file(), f"single-Episode native backend missing: {path}")
    spec = importlib.util.spec_from_file_location(
        "strict2h_existing_native_capture_backend", path
    )
    _require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _memmap(spool: Path, name: str) -> np.memmap:
    contract = CONTRACT.RAW_MEMMAP_CONTRACT[name]
    path = spool / name
    return np.memmap(
        path,
        mode="r",
        dtype=np.dtype(contract["dtype"]),
        shape=tuple(contract["shape"]),
        order="C",
    )


def _validate_runtime(spool: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    readbacks = _load(spool / "runtime_readbacks.json")
    _require(len(readbacks.get("normal", [])) == 75, "normal runtime count drift")
    target = readbacks.get("target_only")
    _require(
        isinstance(target, Mapping)
        and set(target) == {"source1", "source2"}
        and all(len(target[slot]) == 75 for slot in target),
        "target-only runtime count drift",
    )
    assets = _load(spool / "runtime_asset_readbacks.json")
    _require(
        assets.get("status") == "pass"
        and assets.get("frame_index") == 74
        and set(assets.get("per_instance", {})) == {"source1", "source2"},
        "live runtime asset closure drift",
    )
    required = {
        "blueprint",
        "skeletal_mesh",
        "skeleton",
        "stable_actor_tag",
        "standing_idle",
        "current_action",
        "emitter_native_readback",
    }
    for source_slot, record in assets["per_instance"].items():
        _require(record.get("status") == "pass", f"{source_slot}: live status drift")
        _require(
            required.issubset(record), f"{source_slot}: live readback keys missing"
        )
        _require(
            all(record[key].get("status") == "pass" for key in required),
            f"{source_slot}: live readback failed",
        )
    return readbacks, assets


def _pixel_gate(
    truth: Mapping[str, Any],
    *,
    target_slot: str,
    target_side: str,
    speech: tuple[int, int],
) -> dict[str, Any]:
    distractor_slot = "source2" if target_slot == "source1" else "source1"
    target_frames = truth["per_instance"][target_slot]["frames"]
    distractor_frames = truth["per_instance"][distractor_slot]["frames"]
    _require(len(target_frames) == len(distractor_frames) == 75, "pixel count drift")
    start, end = speech
    target_speech = [
        item for item in target_frames if start <= int(item["frame_index"]) <= end
    ]
    _require(len(target_speech) == end - start + 1, "speech pixel truth incomplete")
    minimum_target_fraction = min(
        float(item["visible_fraction"]) for item in target_speech
    )
    minimum_distractor_fraction = min(
        float(item["visible_fraction"]) for item in distractor_frames
    )
    minimum_target_pixels = min(int(item["visible_pixels"]) for item in target_speech)
    minimum_distractor_pixels = min(
        int(item["visible_pixels"]) for item in distractor_frames
    )
    _require(minimum_target_fraction >= 0.8, "target visibility below 0.8")
    _require(minimum_distractor_fraction >= 0.5, "distractor visibility below 0.5")
    _require(minimum_target_pixels >= 5000, "target visible-pixel gate failed")
    _require(minimum_distractor_pixels >= 5000, "distractor visible-pixel gate failed")
    target_x = [
        float(item["target_centroid_xy_px"][0]) / 1280.0 for item in target_frames
    ]
    distractor_x = [
        float(item["target_centroid_xy_px"][0]) / 1280.0 for item in distractor_frames
    ]
    if target_side == "left":
        _require(max(target_x) < 0.48 and min(distractor_x) > 0.52, "left/right drift")
    else:
        _require(min(target_x) > 0.52 and max(distractor_x) < 0.48, "right/left drift")
    for owner, values in (("target", target_speech), ("distractor", distractor_frames)):
        for item in values:
            x0, y0, x1, y1 = [int(value) for value in item["target_bbox_xyxy_px"]]
            _require(
                x0 >= 1 and y0 >= 1 and x1 <= 1278 and y1 <= 718,
                f"{owner} bbox touches the frame edge",
            )
    return {
        "status": "pass",
        "target_source_slot": target_slot,
        "distractor_source_slot": distractor_slot,
        "target_side": target_side,
        "minimum_target_visible_fraction_during_speech": minimum_target_fraction,
        "minimum_distractor_visible_fraction": minimum_distractor_fraction,
        "minimum_target_visible_pixels_during_speech": minimum_target_pixels,
        "minimum_distractor_visible_pixels": minimum_distractor_pixels,
    }


def _artifact(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"artifact missing: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def finalize(
    *,
    batch_request: Path,
    expected_batch_sha256: str,
    episode_id: str,
    expected_binding_sha256: str,
    raw_ready: Path,
    attempt_root: Path,
    output: Path,
) -> Path:
    batch = CONTRACT.resolve_request(batch_request)
    _require(
        batch.request_sha256 == expected_batch_sha256, "batch request digest drift"
    )
    matches = [item for item in batch.episodes if item.episode_id == episode_id]
    _require(len(matches) == 1, "Episode selection is not unique")
    episode = matches[0]
    _require(
        episode.bindings["binding_sha256"] == expected_binding_sha256,
        "Episode binding digest drift",
    )
    _require(output.resolve() == episode.output_root, "Episode output root drift")
    _require(
        attempt_root.resolve() == raw_ready.resolve().parents[1],
        "raw receipt/attempt root drift",
    )
    CONTRACT.validate_raw_ready_receipt(raw_ready, batch=batch, episode=episode)
    spool = raw_ready.parent
    context = _load(spool / "capture_context.json")
    _require(context.get("episode_id") == episode_id, "raw context Episode drift")
    _require(context.get("native_map") == batch.native_map, "raw context map drift")
    _require(context.get("frame_indices") == list(range(75)), "raw context frame drift")
    _require(
        isinstance(context.get("camera_pose_ids"), list)
        and len(context["camera_pose_ids"]) == 75,
        "raw camera pose identity drift",
    )

    backend = _load_native_backend()
    sys.path.insert(0, str(REPOSITORY / "src"))
    from avengine.qa.pixel_visibility import (
        PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        compile_depth_pixel_visibility_truth,
    )

    normal_depth = _memmap(spool, "normal_depth_m.f16le")
    target_depths = {
        "source1": _memmap(spool, "target_only_source1_depth_m.f16le"),
        "source2": _memmap(spool, "target_only_source2_depth_m.f16le"),
    }
    normal_object_ids = _memmap(spool, "normal_object_ids.u32le")
    readbacks, assets = _validate_runtime(spool)
    common_context = {
        "renderer_backend": "spear_unreal_native_apartment",
        "rgb_renderer_backend": "spear_unreal_native_apartment",
        "camera_contract_id": "lead_a_native_spear_bp_camera_sensor_v1",
        "semantic_id_namespace": "lead_a_native_spear_metric_depth_instances_v1",
        "resolution_hw": [720, 1280],
        "frame_indices": list(range(75)),
        "camera_pose_ids": context["camera_pose_ids"],
    }
    semantic_ids = {"source1": 1, "source2": 2}
    truth = compile_depth_pixel_visibility_truth(
        normal_depth_m_frames=normal_depth,
        target_only_depth_m_frames_by_instance=target_depths,
        semantic_ids_by_instance=semantic_ids,
        normal_context={"pass_kind": "modal_scene", **common_context},
        target_only_contexts_by_instance={
            slot: {
                "pass_kind": "target_only",
                "target_instance_id": slot,
                **common_context,
            }
            for slot in semantic_ids
        },
        target_only_background_depth_m=backend.TARGET_ONLY_BACKGROUND_DEPTH_M,
        absolute_tolerance_m=backend.ABSOLUTE_TOLERANCE_M,
        relative_tolerance=backend.RELATIVE_TOLERANCE,
    )
    _require(
        truth.get("authority") == PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        "pixel truth authority drift",
    )
    pixel_gate = _pixel_gate(
        truth,
        target_slot=episode.target_source_slot,
        target_side=episode.target_side,
        speech=episode.speech_frame_window_inclusive,
    )
    modal_masks, target_masks = backend._derive_masks(
        normal_depths=normal_depth,
        target_depths_by_instance=target_depths,
        semantic_ids=semantic_ids,
    )
    alignment = backend._maximum_readback_drift(
        readbacks["normal"], readbacks["target_only"]
    )
    _require(
        alignment["maximum_location_drift_cm"] == 0.0
        and alignment["maximum_rotation_drift_deg"] == 0.0,
        "normal/target-only runtime drift",
    )

    final_root = attempt_root / "finalized_output"
    final_root.mkdir(exist_ok=False)
    depth_path = final_root / "metric_depth_native.npz"
    np.savez_compressed(
        depth_path,
        normal_depth_m=normal_depth,
        target_only_source1_depth_m=target_depths["source1"],
        target_only_source2_depth_m=target_depths["source2"],
    )
    object_path = final_root / "normal_object_ids_uint32.npz"
    np.savez_compressed(object_path, normal_object_ids=normal_object_ids)
    mask_path = final_root / "native_pixel_masks_depth_authority_v1.npz"
    np.savez_compressed(
        mask_path,
        depth_derived_modal_semantic=modal_masks,
        modal_visible_source1=modal_masks == 1,
        modal_visible_source2=modal_masks == 2,
        target_only_source1=target_masks["source1"],
        target_only_source2=target_masks["source2"],
    )
    truth_path = final_root / "pixel_visibility_truth.json"
    CONTRACT._atomic_write_json(truth_path, truth)

    rgb_root = spool / "rgb_frames"
    visual_path = final_root / "native_rgb_visual_only.mp4"
    muxed_path = final_root / "native_rgb_binaural.mp4"
    backend._run_checked(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "15",
            "-start_number",
            "0",
            "-i",
            str(rgb_root / "frame_%06d.png"),
            "-frames:v",
            "75",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "15",
            str(visual_path),
        ]
    )
    backend._run_checked(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(visual_path),
            "-i",
            str(episode.audio_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "2",
            "-shortest",
            str(muxed_path),
        ]
    )
    probe = backend._ffprobe(muxed_path)
    video_streams = [
        item for item in probe.get("streams", []) if item.get("codec_type") == "video"
    ]
    audio_streams = [
        item for item in probe.get("streams", []) if item.get("codec_type") == "audio"
    ]
    _require(len(video_streams) == len(audio_streams) == 1, "mux stream closure drift")
    video = video_streams[0]
    audio = audio_streams[0]
    _require(
        int(video.get("width", -1)) == 1280
        and int(video.get("height", -1)) == 720
        and video.get("r_frame_rate") == "15/1",
        "video geometry/frame-rate drift",
    )
    _require(
        int(audio.get("channels", -1)) == 2
        and int(audio.get("sample_rate", -1)) == 16000,
        "muxed audio contract drift",
    )
    depth_values = np.asarray(normal_depth)
    _require(
        np.isfinite(depth_values).all() and (depth_values > 0).all(),
        "normal metric depth contains invalid values",
    )
    artifacts = {
        "metric_depth": _artifact(depth_path),
        "normal_object_ids": _artifact(object_path),
        "pixel_masks": _artifact(mask_path),
        "pixel_truth": _artifact(truth_path),
        "visual_video": _artifact(visual_path),
        "binaural_video": _artifact(muxed_path),
        "runtime_readbacks": _artifact(spool / "runtime_readbacks.json"),
        "runtime_asset_readbacks": _artifact(spool / "runtime_asset_readbacks.json"),
        "authoritative_audio": _artifact(episode.audio_wav),
        "raw_ready": _artifact(raw_ready),
    }
    manifest = {
        "schema": "avengine_native_strict_two_human_raw_finalization_manifest_v1",
        "status": "pass",
        "episode_id": episode_id,
        "native_map": batch.native_map,
        "input_binding_sha256": expected_binding_sha256,
        "raw_receipt_sha256": _sha256(raw_ready),
        "capture_contract": {
            "normal_rgb_frames": 75,
            "normal_metric_depth_frames": 75,
            "source1_target_only_depth_frames": 75,
            "source2_target_only_depth_frames": 75,
            "normal_runtime_readbacks": 75,
            "target_only_runtime_readbacks": 150,
            "live_asset_readback": True,
        },
        "runtime_alignment": alignment,
        "live_asset_readbacks": assets,
        "pixel_gate": pixel_gate,
        "audio_binding": episode.bindings["audio_contract"],
        "acoustic_binding": episode.bindings["acoustics"],
        "motion_realism_binding": episode.bindings["motion_realism"],
        "metric_depth": {
            "dtype": str(normal_depth.dtype),
            "shape": list(normal_depth.shape),
            "minimum_m": float(np.min(normal_depth)),
            "maximum_m": float(np.max(normal_depth)),
        },
        "ffprobe": probe,
        "artifacts": artifacts,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "ground_contact_release_qualified": False,
        "motion_realism_release_qualified": batch.request[
            "motion_realism_release_qualified"
        ],
    }
    manifest_path = final_root / "manifest.json"
    CONTRACT._atomic_write_json(manifest_path, manifest)
    final_receipt = {
        "schema": CONTRACT.FINAL_RECEIPT_SCHEMA,
        "status": "pass",
        "episode_id": episode_id,
        "batch_request_sha256": expected_batch_sha256,
        "input_binding_sha256": expected_binding_sha256,
        "capture_contract": manifest["capture_contract"],
        "raw_ready": str(raw_ready.resolve()),
        "raw_ready_sha256": _sha256(raw_ready),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "finalized_output": str(final_root.resolve()),
        "formal_episode_count": 0,
        "qualification_claim": False,
        "ground_contact_release_qualified": False,
        "motion_realism_release_qualified": batch.request[
            "motion_realism_release_qualified"
        ],
    }
    final_receipt_path = output / "FINAL_READY.json"
    _require(not final_receipt_path.exists(), "FINAL_READY must be new")
    CONTRACT._atomic_write_json_new(final_receipt_path, final_receipt)
    return final_receipt_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-request", type=Path, required=True)
    parser.add_argument("--batch-request-sha256", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--input-binding-sha256", required=True)
    parser.add_argument("--raw-ready", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    path = finalize(
        batch_request=args.batch_request.resolve(),
        expected_batch_sha256=args.batch_request_sha256,
        episode_id=args.episode_id,
        expected_binding_sha256=args.input_binding_sha256,
        raw_ready=args.raw_ready.resolve(),
        attempt_root=args.attempt_root.resolve(),
        output=args.output.resolve(),
    )
    print(f"STRICT_TWO_HUMAN_RAW_EPISODE_FINALIZE_OK receipt={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
