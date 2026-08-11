#!/usr/bin/env python3
"""Fail-closed finalizer for one strict two-human 75-frame native canary."""

from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _wav_contract(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"binaural WAV missing: {path}")
    with wave.open(str(path), "rb") as stream:
        record = {
            "channel_count": stream.getnchannels(),
            "sample_rate_hz": stream.getframerate(),
            "sample_count": stream.getnframes(),
            "sample_width_bytes": stream.getsampwidth(),
        }
    _require(
        record
        == {
            "channel_count": 2,
            "sample_rate_hz": 16000,
            "sample_count": 80000,
            "sample_width_bytes": 2,
        },
        f"binaural WAV contract drift: {record}",
    )
    return record


def _validate_acoustics(canary: dict[str, Any]) -> dict[str, Any]:
    evidence = canary["acoustic_evidence"]
    plan_path = Path(evidence["exact_rir_plan"])
    cache_path = Path(evidence["rir_cache"])
    delivery_path = Path(evidence["binaural_delivery"])
    plan = _load(plan_path)
    cache = _load(cache_path)
    delivery = _load(delivery_path)
    _require(len(plan.get("jobs", [])) == 2, "exact RIR plan must contain two jobs")
    _require(
        cache.get("status") == "pass"
        and cache.get("full_plan_complete") is True
        and cache.get("selected_job_count") == 2,
        "exact RIR cache is incomplete",
    )
    _require(
        delivery.get("status") == "pass"
        and delivery.get("episode_count") == 1
        and delivery.get("qualification_claim") is False,
        "binaural delivery boundary drift",
    )
    wav = _wav_contract(Path(canary["audio_wav"]))
    return {
        "status": "pass_exact_two_source_rir_target_only_binaural",
        "rir_job_count": 2,
        **wav,
        "target_active": True,
        "distractor_silent": True,
    }


def _validate_npz(capture_root: Path) -> dict[str, Any]:
    depth_path = capture_root / "metric_depth_native.npz"
    mask_path = capture_root / "native_pixel_masks_depth_authority_v1.npz"
    object_path = capture_root / "normal_object_ids_uint32.npz"
    _require(depth_path.is_file() and mask_path.is_file() and object_path.is_file(), "native arrays missing")
    with np.load(depth_path) as arrays:
        _require(
            set(arrays.files)
            == {
                "normal_depth_m",
                "target_only_source1_depth_m",
                "target_only_source2_depth_m",
            },
            "metric-depth array keys drift",
        )
        for key in arrays.files:
            value = arrays[key]
            _require(value.shape == (75, 720, 1280), f"{key}: shape drift")
            _require(value.dtype == np.float16, f"{key}: dtype drift")
            _require(np.isfinite(value).all(), f"{key}: nonfinite depth")
            _require((value > 0).all(), f"{key}: nonpositive depth")
    with np.load(mask_path) as arrays:
        required = {
            "depth_derived_modal_semantic",
            "modal_visible_source1",
            "modal_visible_source2",
            "target_only_source1",
            "target_only_source2",
        }
        _require(set(arrays.files) == required, "mask array keys drift")
        for key in required:
            _require(arrays[key].shape == (75, 720, 1280), f"{key}: shape drift")
        _require(
            np.all(np.count_nonzero(arrays["target_only_source1"], axis=(1, 2)) > 0)
            and np.all(np.count_nonzero(arrays["target_only_source2"], axis=(1, 2)) > 0),
            "target-only footprint absent in one or more frames",
        )
    with np.load(object_path) as arrays:
        _require(
            arrays["normal_object_ids"].shape == (75, 720, 1280)
            and arrays["normal_object_ids"].dtype == np.uint32,
            "normal object-ID array drift",
        )
    return {
        "status": "pass",
        "normal_rgb_frame_count": 75,
        "metric_depth_frame_count": 75,
        "source1_target_only_frame_count": 75,
        "source2_target_only_frame_count": 75,
    }


def _validate_pixels(canary: dict[str, Any], capture_root: Path) -> dict[str, Any]:
    truth = _load(capture_root / "pixel_visibility_truth.json")
    _require(truth.get("status") == "pass", "pixel truth status drift")
    _require(truth.get("frame_indices") == list(range(75)), "pixel frame index drift")
    _require(truth.get("resolution_hw") == [720, 1280], "pixel resolution drift")
    target_frames = truth["per_instance"]["source1"]["frames"]
    distractor_frames = truth["per_instance"]["source2"]["frames"]
    _require(len(target_frames) == len(distractor_frames) == 75, "pixel frame count drift")
    start, end = [int(value) for value in canary["speech_frame_window_inclusive"]]
    target_speech = [item for item in target_frames if start <= int(item["frame_index"]) <= end]
    _require(len(target_speech) == end - start + 1, "target speech-frame truth incomplete")
    minimum_target_fraction = min(float(item["visible_fraction"]) for item in target_speech)
    minimum_distractor_fraction = min(float(item["visible_fraction"]) for item in distractor_frames)
    minimum_target_pixels = min(int(item["visible_pixels"]) for item in target_speech)
    minimum_distractor_pixels = min(int(item["visible_pixels"]) for item in distractor_frames)
    _require(minimum_target_fraction >= 0.8, "target visibility below 0.8 during speech")
    _require(minimum_distractor_fraction >= 0.5, "distractor visibility below 0.5")
    _require(minimum_target_pixels >= 5000 and minimum_distractor_pixels >= 5000, "visible pixel minimum failed")
    target_side = canary["target_side"]
    target_x = [float(item["target_centroid_xy_px"][0]) / 1280.0 for item in target_frames]
    distractor_x = [
        float(item["target_centroid_xy_px"][0]) / 1280.0
        for item in distractor_frames
    ]
    if target_side == "left":
        _require(max(target_x) < 0.48 and min(distractor_x) > 0.52, "left/right identity drift")
    else:
        _require(min(target_x) > 0.52 and max(distractor_x) < 0.48, "right/left identity drift")
    for owner, frames in (("target", target_speech), ("distractor", distractor_frames)):
        for item in frames:
            x0, y0, x1, y1 = [int(value) for value in item["target_bbox_xyxy_px"]]
            _require(
                x0 >= 1 and y0 >= 1 and x1 <= 1278 and y1 <= 718,
                f"{owner} bbox touches frame edge at f{item['frame_index']}",
            )
    return {
        "status": "pass",
        "target_side": target_side,
        "minimum_target_visible_fraction_during_speech": minimum_target_fraction,
        "minimum_distractor_visible_fraction": minimum_distractor_fraction,
        "minimum_target_visible_pixels_during_speech": minimum_target_pixels,
        "minimum_distractor_visible_pixels": minimum_distractor_pixels,
    }


def _validate_runtime(capture_root: Path) -> dict[str, Any]:
    readbacks = _load(capture_root / "runtime_readbacks.json")
    _require(len(readbacks["normal"]) == 75, "normal runtime readback frame count drift")
    _require(
        set(readbacks["target_only"]) == {"source1", "source2"}
        and all(len(value) == 75 for value in readbacks["target_only"].values()),
        "target-only runtime readback frame count drift",
    )
    assets = _load(capture_root / "runtime_asset_readbacks.json")
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
        "emitter_native_readback",
    }
    for instance_id, record in assets["per_instance"].items():
        _require(record.get("status") == "pass", f"{instance_id}: runtime asset status drift")
        _require(required.issubset(record), f"{instance_id}: live gate keys missing")
        _require(
            all(record[key].get("status") == "pass" for key in required),
            f"{instance_id}: one or more live gates failed",
        )
    return {
        "status": "pass",
        "normal_frame_readback_count": 75,
        "target_only_frame_readback_count": 150,
        "live_asset_readback_frame": 74,
        "live_asset_instance_count": 2,
    }


def _validate_video(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"video missing: {path}")
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,r_frame_rate,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    _require(
        stream["nb_frames"] == "75"
        and stream["r_frame_rate"] == "15/1"
        and stream["width"] == 1280
        and stream["height"] == 720,
        f"video stream contract drift: {stream}",
    )
    return stream


def finalize(
    *,
    canary_plan_path: Path,
    canary_index: int,
    capture_root: Path,
    launch_receipt_path: Path,
    output: Path,
) -> Path:
    plan = _load(canary_plan_path)
    matches = [
        item for item in plan["canaries"] if int(item["canary_index"]) == canary_index
    ]
    _require(len(matches) == 1, "canary index must resolve exactly once")
    canary = matches[0]
    launch = _load(launch_receipt_path)
    _require(
        launch.get("status") == "pass"
        and launch.get("capture_process_exit_code") == 0
        and launch.get("physical_gpu_index") == 1
        and launch.get("graphics_adapter_argument") == 1
        and launch.get("forbidden_physical_gpu_indices_used") == [],
        "GPU1 launch receipt failed",
    )
    manifest = _load(capture_root / "manifest.json")
    _require(
        manifest.get("status") == "pass"
        and manifest.get("scenario_id") == canary["episode_id"],
        "capture manifest Episode/status drift",
    )
    frame_contract = manifest["frame_contract"]
    _require(
        frame_contract["frame_count"] == 75
        and frame_contract["formal_episode_frame_count"] == 75
        and frame_contract["captured_frame_indices"] == list(range(75))
        and len(frame_contract["camera_pose_ids"]) == 75,
        "full75 frame contract failed",
    )
    _require(
        Path(manifest["audio"]["authoritative_wav"]).resolve()
        == Path(canary["audio_wav"]).resolve(),
        "authoritative audio lineage drift",
    )
    _require(
        manifest["runtime_alignment"]["maximum_location_drift_cm"] == 0.0
        and manifest["runtime_alignment"]["maximum_rotation_drift_deg"] == 0.0,
        "normal/target-only runtime alignment drift",
    )
    rgb_frames = sorted((capture_root / "rgb_frames").glob("frame_*.png"))
    _require(
        len(rgb_frames) == 75
        and [path.name for path in rgb_frames]
        == [f"frame_{index:06d}.png" for index in range(75)],
        "normal RGB sequence incomplete",
    )
    acoustics = _validate_acoustics(canary)
    arrays = _validate_npz(capture_root)
    pixels = _validate_pixels(canary, capture_root)
    runtime = _validate_runtime(capture_root)
    visual_video = _validate_video(capture_root / "native_rgb_visual_only.mp4")
    av_video = _validate_video(capture_root / "native_rgb_binaural.mp4")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "avengine_native_strict_two_human_full75_canary_finalization_v1",
        "status": "pass",
        "episode_id": canary["episode_id"],
        "canary_index": canary_index,
        "full75_canary_pass": True,
        "captured_frame_count": 75,
        "duration_seconds": 5,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "gpu": {
            "physical_index": 1,
            "forbidden_indices_used": [],
            "capture_process_exit_code": 0,
        },
        "acoustics": acoustics,
        "native_arrays": arrays,
        "pixels": pixels,
        "runtime": runtime,
        "video": {"visual_only": visual_video, "binaural": av_video},
        "artifacts": {
            "capture_manifest": str((capture_root / "manifest.json").resolve()),
            "normal_rgb_frames": str((capture_root / "rgb_frames").resolve()),
            "metric_depth": str((capture_root / "metric_depth_native.npz").resolve()),
            "target_only_masks": str((capture_root / "native_pixel_masks_depth_authority_v1.npz").resolve()),
            "pixel_visibility_truth": str((capture_root / "pixel_visibility_truth.json").resolve()),
            "runtime_readbacks": str((capture_root / "runtime_readbacks.json").resolve()),
            "runtime_asset_readbacks": str((capture_root / "runtime_asset_readbacks.json").resolve()),
            "binaural_video": str((capture_root / "native_rgb_binaural.mp4").resolve()),
            "binaural_wav": str(Path(canary["audio_wav"]).resolve()),
            "gpu_launch_receipt": str(launch_receipt_path.resolve()),
        },
    }
    result_path = output / "finalization.json"
    _write(result_path, result)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--canary-index", type=int, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        canary_plan_path=args.canary_plan.resolve(),
        canary_index=args.canary_index,
        capture_root=args.capture_root.resolve(),
        launch_receipt_path=args.launch_receipt.resolve(),
        output=args.output.resolve(),
    )
    print(f"STRICT_TWO_HUMAN_FULL75_CANARY_OK finalization={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
