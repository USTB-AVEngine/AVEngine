#!/usr/bin/env python3
"""Append text-selective GT/predicted DoA trajectories to a real review video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from avengine_v43.hdf5_data import Hdf5QueryBank
from avengine_v43.inference_visualization import (
    FRAME_RATE_HZ,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    circular_error_deg,
    normalize_360,
    render_panel_frame,
)
from avengine_v43.labels import LegacyV4AudioError, native_azimuth_to_bin360
from run_training_smoke import _assert_output_contract, _load_model
from train import SCHEMA as TRAINING_SCHEMA
from train import _cache_text_embeddings, _is_clap_state_key


VISUALIZATION_SCHEMA = "avengine_v43_binaural360_inference_visualization_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-h5", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clap-checkpoint", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=20_260_723)
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise LegacyV4AudioError(f"JSON root must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _sample_from_index(index: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    samples = index.get("samples")
    if not isinstance(samples, list):
        raise LegacyV4AudioError("dataset index lacks samples")
    matches = [
        sample
        for sample in samples
        if isinstance(sample, Mapping) and sample.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise LegacyV4AudioError(
            f"dataset index must contain sample exactly once: {sample_id}"
        )
    sample = matches[0]
    if sample.get("split") != "test":
        raise LegacyV4AudioError("inference review sample must come from test split")
    if sample.get("both_sources_active") is not True:
        raise LegacyV4AudioError("review sample must contain two active sources")
    return sample


def _probe_video(path: Path) -> Mapping[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    streams = value.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if (
        not isinstance(video, Mapping)
        or int(video.get("width", -1)) != 1280
        or int(video.get("height", -1)) != PANEL_HEIGHT
        or int(video.get("nb_frames", -1)) != 75
        or video.get("avg_frame_rate") != "15/1"
        or not isinstance(audio, Mapping)
        or int(audio.get("channels", -1)) != 2
    ):
        raise LegacyV4AudioError(
            "source video must be 1280x480, 75 frames at 15 fps, with 2ch audio"
        )
    return {
        "video_codec": video.get("codec_name"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_count": int(video["nb_frames"]),
        "frame_rate": video["avg_frame_rate"],
        "audio_codec": audio.get("codec_name"),
        "audio_channels": int(audio["channels"]),
        "duration_seconds": float(value["format"]["duration"]),
    }


def _load_inference_checkpoint(
    *,
    torch: Any,
    model: Any,
    path: Path,
) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != TRAINING_SCHEMA:
        raise LegacyV4AudioError("checkpoint schema is invalid")
    state = payload.get("model_state_without_clap")
    if not isinstance(state, Mapping):
        raise LegacyV4AudioError("checkpoint lacks model_state_without_clap")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_clap = [
        key for key in incompatible.missing_keys if not _is_clap_state_key(key)
    ]
    if unexpected or missing_non_clap:
        raise LegacyV4AudioError(
            "checkpoint model keys are incompatible: "
            f"unexpected={unexpected[:5]}, missing={missing_non_clap[:5]}"
        )
    return payload


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _progress(message: str) -> None:
    print(f"[v43-inference-review] {message}", flush=True)


def _encode_and_mux(
    *,
    frame_root: Path,
    source_video: Path,
    panel_video: Path,
    output_video: Path,
) -> None:
    _run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(int(FRAME_RATE_HZ)),
            "-i",
            str(frame_root / "frame_%04d.png"),
            "-frames:v",
            "75",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(panel_video),
        ]
    )
    _run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_video),
            "-i",
            str(panel_video),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-frames:v",
            "75",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
    )


def main() -> int:
    args = _arguments()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output_root}")
    output_root.mkdir(parents=True)

    dataset_h5 = args.dataset_h5.resolve()
    dataset_index_path = args.dataset_index.resolve()
    checkpoint_path = args.checkpoint.resolve()
    clap_checkpoint = args.clap_checkpoint.resolve()
    source_video = args.source_video.resolve()
    for path in (
        dataset_h5,
        dataset_index_path,
        checkpoint_path,
        clap_checkpoint,
        source_video,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    _progress("validating source video and dataset-index binding")
    source_probe = _probe_video(source_video)
    sample = _sample_from_index(_load_json(dataset_index_path), args.sample_id)
    episode_id = str(sample.get("episode_id", ""))
    if source_video.parent.name != episode_id:
        raise LegacyV4AudioError(
            "source video folder does not match dataset-index episode_id"
        )

    _progress("reading one two-query mixture and its 75-frame labels from HDF5")
    with Hdf5QueryBank(dataset_h5, preload_mixtures=False) as bank:
        refs = [
            query
            for query in bank.queries("test")
            if query.sample_id == args.sample_id
        ]
        refs.sort(key=lambda query: query.source_index)
        if len(refs) != 2 or [query.source_index for query in refs] != [0, 1]:
            raise LegacyV4AudioError(
                "HDF5 test split must contain source1/source2 queries"
            )
        mixtures = bank.read_mixtures(refs)
        targets = np.stack(
            [native_azimuth_to_bin360(query.azimuth_deg) for query in refs]
        )
        captions = [query.caption for query in refs]

        _progress(f"loading model and frozen CLAP on {args.device}")
        torch, model, device, model_load_seconds, model_audit = _load_model(
            clap_checkpoint=clap_checkpoint,
            device_name=args.device,
            seed=args.seed,
        )
        checkpoint_payload = _load_inference_checkpoint(
            torch=torch,
            model=model,
            path=checkpoint_path,
        )
        text_cache = _cache_text_embeddings(
            torch=torch,
            model=model,
            device=device,
            queries_by_split={"test": refs},
        )
        mixture_tensor = torch.from_numpy(mixtures).to(device)
        text_embeddings = torch.stack([text_cache[caption] for caption in captions])
        null_cues = torch.zeros(
            2,
            1,
            dtype=mixture_tensor.dtype,
            device=device,
        )
        _progress("running both text queries in one GPU batch")
        model.eval()
        torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            separated, doa, cardinality = model(
                mixture_tensor,
                null_cues,
                text_embeddings,
            )
        torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        _assert_output_contract(
            torch=torch,
            separated=separated,
            doa=doa,
            cardinality=cardinality,
            batch_size=2,
        )
        predictions = torch.argmax(doa, dim=-1).cpu().numpy().astype(np.float64)
        cardinality_prediction = (
            torch.argmax(cardinality, dim=-1).cpu().numpy().astype(int)
        )

    targets = normalize_360(targets)
    predictions = normalize_360(predictions)
    errors = circular_error_deg(predictions, targets)
    query_difference_rate = float(np.mean(predictions[0] != predictions[1]))
    result = {
        "schema": VISUALIZATION_SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "sample_id": args.sample_id,
        "episode_id": episode_id,
        "split": "test",
        "source_video": str(source_video),
        "source_video_probe": dict(source_probe),
        "dataset_h5": str(dataset_h5),
        "dataset_index": str(dataset_index_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_epoch_zero_based": checkpoint_payload.get("best_epoch"),
        "checkpoint_best_validation_mae_deg": checkpoint_payload.get(
            "best_validation_mae_deg"
        ),
        "model": dict(model_audit),
        "captions": captions,
        "asset_ids_by_source_slot": dict(
            sample.get("asset_ids_by_source_slot", {})
        ),
        "frame_rate_hz": FRAME_RATE_HZ,
        "frame_count": int(targets.shape[1]),
        "native_azimuth_convention": {
            "front_deg": 0,
            "right_deg": 90,
            "rear_deg": 180,
            "left_deg": 270,
        },
        "target_azimuth_deg": targets.tolist(),
        "predicted_azimuth_deg": predictions.tolist(),
        "absolute_circular_error_deg": errors.tolist(),
        "metrics": {
            "per_query_mean_absolute_error_deg": [
                float(np.mean(errors[index])) for index in range(2)
            ],
            "per_query_median_absolute_error_deg": [
                float(np.median(errors[index])) for index in range(2)
            ],
            "overall_mean_absolute_error_deg": float(np.mean(errors)),
            "overall_median_absolute_error_deg": float(np.median(errors)),
            "overall_p90_absolute_error_deg": float(np.percentile(errors, 90)),
            "query_prediction_difference_rate": query_difference_rate,
            "cardinality_accuracy_for_one_target": float(
                np.mean(cardinality_prediction == 1)
            ),
        },
        "timing_seconds": {
            "model_load": model_load_seconds,
            "two_query_inference": inference_seconds,
        },
    }
    inference_json = output_root / "inference.json"
    _atomic_write_json(inference_json, result)

    _progress("rendering 75 DoA review-panel frames")
    frame_root = output_root / "panel_frames"
    for frame_index in range(targets.shape[1]):
        render_panel_frame(
            frame_index=frame_index,
            targets_deg=targets,
            predictions_deg=predictions,
            captions=captions,
            sample_id=args.sample_id,
            output_path=frame_root / f"frame_{frame_index:04d}.png",
        )
    panel_video = output_root / "doa_prediction_panel.mp4"
    output_video = output_root / "ue_topdown_binaural_with_doa_prediction.mp4"
    _progress("encoding panel and muxing it beside RGB+Topdown with original audio")
    _encode_and_mux(
        frame_root=frame_root,
        source_video=source_video,
        panel_video=panel_video,
        output_video=output_video,
    )
    result["outputs"] = {
        "inference_json": str(inference_json),
        "panel_video": str(panel_video),
        "review_video": str(output_video),
        "panel_width": PANEL_WIDTH,
        "panel_height": PANEL_HEIGHT,
        "review_width": 1280 + PANEL_WIDTH,
        "review_height": PANEL_HEIGHT,
    }
    result["review_video_probe"] = dict(_probe_video_for_output(output_video))
    _atomic_write_json(inference_json, result)
    _progress("completed and passed final video readback")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _probe_video_for_output(path: Path) -> Mapping[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    value = json.loads(
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    video = next(
        stream for stream in value["streams"] if stream.get("codec_type") == "video"
    )
    audio = next(
        stream for stream in value["streams"] if stream.get("codec_type") == "audio"
    )
    if (
        int(video["width"]) != 1280 + PANEL_WIDTH
        or int(video["height"]) != PANEL_HEIGHT
        or int(video["nb_frames"]) != 75
        or video["avg_frame_rate"] != "15/1"
        or int(audio["channels"]) != 2
    ):
        raise LegacyV4AudioError("final review video readback failed")
    return {
        "video_codec": video.get("codec_name"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_count": int(video["nb_frames"]),
        "frame_rate": video["avg_frame_rate"],
        "audio_codec": audio.get("codec_name"),
        "audio_channels": int(audio["channels"]),
        "duration_seconds": float(value["format"]["duration"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
