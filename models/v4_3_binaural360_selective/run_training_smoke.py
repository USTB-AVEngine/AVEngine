#!/usr/bin/env python3
"""Run one real from-scratch v4_3 selective binaural-360 training smoke."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from avengine_v43.labels import (
    LegacyV4AudioError,
    caption_for_asset,
    circular_gaussian_targets,
    deterministic_split_samples,
    label_tracks_for_source,
    native_azimuth_to_bin360,
)


SAMPLE_RATE_HZ = 16_000
DURATION_SECONDS = 5.0
SAMPLE_COUNT = 80_000
OUTPUT_FRAME_COUNT = 75
OUTPUT_AZIMUTH_BINS = 360
SOURCE_SLOTS = ("source1", "source2")
SCHEMA = "avengine_v43_text_selective_binaural360_training_smoke_v1"


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--rir-plan", type=Path, required=True)
    parser.add_argument("--clap-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--offset", type=_nonnegative_int, default=0)
    parser.add_argument("--limit", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--train-steps", type=_positive_int, default=1)
    parser.add_argument("--learning-rate", type=_positive_float, default=1.0e-3)
    parser.add_argument("--seed", type=_nonnegative_int, default=20_260_723)
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise LegacyV4AudioError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_soundfile():
    try:
        import soundfile
    except ImportError as exc:
        raise RuntimeError("the model environment must provide soundfile") from exc
    return soundfile


def _read_mixture(path: Path) -> np.ndarray:
    soundfile = _load_soundfile()
    if not path.is_file():
        raise FileNotFoundError(path)
    samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE_HZ or samples.shape != (SAMPLE_COUNT, 2):
        raise LegacyV4AudioError(
            f"{path} must be exactly {SAMPLE_COUNT}x2 at {SAMPLE_RATE_HZ} Hz"
        )
    mixture = np.ascontiguousarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(mixture)))
    if not math.isfinite(peak) or peak <= 0.0:
        raise LegacyV4AudioError(f"{path} has no finite positive energy")
    mixture /= peak + 1.0e-8
    if not np.all(np.isfinite(mixture)):
        raise LegacyV4AudioError(f"{path} normalization produced non-finite data")
    return mixture


def _trajectory_lookup(
    trajectory_bank: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    episodes = trajectory_bank.get("episodes")
    if not isinstance(episodes, list):
        raise LegacyV4AudioError("trajectory bank lacks episodes")
    result: dict[str, Mapping[str, Any]] = {}
    for episode in episodes:
        episode_id = episode.get("episode_id") if isinstance(episode, Mapping) else None
        if not isinstance(episode_id, str) or episode_id in result:
            raise LegacyV4AudioError("trajectory episode IDs are invalid")
        result[episode_id] = episode
    return result


def _audio_root(dataset_index: Mapping[str, Any]) -> Path:
    roots = dataset_index.get("roots")
    value = roots.get("audio_batch_root") if isinstance(roots, Mapping) else None
    if not isinstance(value, str):
        raise LegacyV4AudioError("dataset index lacks roots.audio_batch_root")
    result = Path(value)
    if not result.is_dir():
        raise FileNotFoundError(result)
    return result


def _build_queries(
    *,
    samples: Sequence[Mapping[str, Any]],
    dataset_index: Mapping[str, Any],
    trajectory_bank: Mapping[str, Any],
    rir_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    listener_position = rir_plan.get("listener_position_m")
    listener_orientation = rir_plan.get("listener_orientation_wxyz")
    if not isinstance(listener_position, list) or not isinstance(
        listener_orientation, list
    ):
        raise LegacyV4AudioError("RIR plan lacks the listener pose")
    frame_rate = trajectory_bank.get("frame_rate_hz")
    if not isinstance(frame_rate, (int, float)) or frame_rate <= 0:
        raise LegacyV4AudioError("trajectory frame rate is invalid")

    episodes = _trajectory_lookup(trajectory_bank)
    audio_root = _audio_root(dataset_index)
    queries: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample.get("sample_id")
        episode_id = sample.get("episode_id")
        audio_path = sample.get("audio_path")
        assets = sample.get("asset_ids_by_source_slot")
        if (
            not isinstance(sample_id, str)
            or not isinstance(episode_id, str)
            or episode_id not in episodes
            or not isinstance(audio_path, str)
            or not isinstance(assets, Mapping)
        ):
            raise LegacyV4AudioError("sample binding is invalid")
        mixture = _read_mixture(audio_root / audio_path)
        paths = episodes[episode_id].get("source_center_paths_m")
        if not isinstance(paths, Mapping):
            raise LegacyV4AudioError(f"{episode_id} lacks source-center paths")

        sample_queries: list[dict[str, Any]] = []
        for source_slot in SOURCE_SLOTS:
            asset_id = assets.get(source_slot)
            positions = paths.get(source_slot)
            if not isinstance(asset_id, str) or positions is None:
                raise LegacyV4AudioError(
                    f"{episode_id} lacks the {source_slot} binding"
                )
            labels = label_tracks_for_source(
                positions,
                source_frame_rate_hz=float(frame_rate),
                target_duration_seconds=DURATION_SECONDS,
                target_frame_count=OUTPUT_FRAME_COUNT,
                listener_position_m=listener_position,
                listener_orientation_wxyz=listener_orientation,
            )
            sample_queries.append(
                {
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "split": sample["split"],
                    "motion_case": sample["motion_case"],
                    "source_slot_id": source_slot,
                    "asset_id": asset_id,
                    "caption": caption_for_asset(asset_id),
                    "target_cardinality": 1,
                    "mixture": mixture,
                    "azimuth_deg": np.asarray(
                        labels["native_360_azimuth_deg"],
                        dtype=np.float64,
                    ),
                }
            )
        if len({value["caption"] for value in sample_queries}) != 2:
            raise LegacyV4AudioError(
                f"{sample_id} has ambiguous text-only source queries"
            )
        queries.extend(sample_queries)
    return queries


def _load_model(
    *,
    clap_checkpoint: Path,
    device_name: str,
    seed: int,
):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the model environment must provide PyTorch") from exc
    if not clap_checkpoint.is_file():
        raise FileNotFoundError(clap_checkpoint)
    if not device_name.startswith("cuda:") or not torch.cuda.is_available():
        raise LegacyV4AudioError("an explicit CUDA device is required")
    device = torch.device(device_name)
    if device.index is None or device.index >= torch.cuda.device_count():
        raise LegacyV4AudioError(f"CUDA device is unavailable: {device_name}")

    from avengine_v43.model import TPEech_Progressive_Refinement

    started = time.perf_counter()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = TPEech_Progressive_Refinement(
        in_channels=256,
        out_channels=64,
        hidden_channels=128,
        gpuid=device,
        kernel_size=40,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=True,
        num_layers=6,
        K=80,
        num_spks=2,
        max_text_length=512,
        fuse_type="CLAP",
        clap_checkpoint=clap_checkpoint.resolve(),
    ).to(device)
    model.chunk_size = SAMPLE_COUNT
    input_stft_frames = SAMPLE_COUNT // model.hop_length + 1
    old_time_align = model.doa_estimator.time_align
    model.doa_estimator.time_align = torch.nn.Linear(
        input_stft_frames,
        OUTPUT_FRAME_COUNT,
        device=device,
        dtype=old_time_align.weight.dtype,
    )
    old_head = model.doa_estimator.fc_out
    model.doa_estimator.fc_out = torch.nn.Linear(
        old_head.in_features,
        OUTPUT_AZIMUTH_BINS,
        bias=old_head.bias is not None,
        device=device,
        dtype=old_head.weight.dtype,
    )

    class NullCueEncoder(torch.nn.Module):
        def forward(self, cue):
            return torch.zeros(
                cue.shape[0],
                512,
                dtype=cue.dtype,
                device=cue.device,
            )

    class NullCueProjection(torch.nn.Module):
        def forward(self, cue_embedding):
            return torch.zeros(
                cue_embedding.shape[0],
                256,
                dtype=cue_embedding.dtype,
                device=cue_embedding.device,
            )

    model.cue_encoder = NullCueEncoder()
    model.cue_fc = NullCueProjection()
    for parameter in model.CLAP.parameters():
        parameter.requires_grad = False
    model.CLAP.eval()
    model.eval()
    torch.cuda.synchronize(device)
    audit = {
        "family": "v4_3_new_IPD_Enhancer",
        "model": "TPEech_Progressive_Refinement",
        "localization_initialization": "from_scratch",
        "old_localization_checkpoint_loaded": False,
        "frozen_pretrained_component": "LAION_CLAP_text_encoder",
        "clap_checkpoint_path": str(clap_checkpoint.resolve()),
        "audio_cue_branch": "fixed_zero_null_branch",
        "input_stft_frame_count": input_stft_frames,
        "output_frame_count": OUTPUT_FRAME_COUNT,
        "output_azimuth_bins": OUTPUT_AZIMUTH_BINS,
        "deterministic_algorithms": True,
        "tf32_enabled": False,
        "seed": seed,
        "device": str(device),
    }
    return torch, model, device, time.perf_counter() - started, audit


def _assert_output_contract(
    *,
    torch: Any,
    separated: Any,
    doa: Any,
    cardinality: Any,
    batch_size: int,
) -> None:
    if (
        tuple(separated.shape) != (batch_size, SAMPLE_COUNT, 2)
        or tuple(doa.shape)
        != (batch_size, OUTPUT_FRAME_COUNT, OUTPUT_AZIMUTH_BINS)
        or tuple(cardinality.shape) != (batch_size, OUTPUT_FRAME_COUNT, 3)
        or not torch.isfinite(separated).all()
        or not torch.isfinite(doa).all()
        or not torch.isfinite(cardinality).all()
    ):
        raise LegacyV4AudioError("the 5s/text-only/360 output contract failed")


def _circular_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = np.abs(predicted - target)
    return np.minimum(difference, 360.0 - difference)


def _run_model(
    *,
    torch: Any,
    model: Any,
    device: Any,
    queries: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    results: list[dict[str, Any]] = []
    model.eval()
    total_started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(queries), batch_size)):
            batch = queries[start : start + batch_size]
            mixtures = torch.from_numpy(
                np.stack([value["mixture"] for value in batch])
            ).to(device)
            null_cues = torch.zeros(
                len(batch),
                1,
                dtype=mixtures.dtype,
                device=device,
            )
            captions = [str(value["caption"]) for value in batch]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            separated, doa, cardinality = model(mixtures, null_cues, captions)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            _assert_output_contract(
                torch=torch,
                separated=separated,
                doa=doa,
                cardinality=cardinality,
                batch_size=len(batch),
            )
            predicted = torch.argmax(doa, dim=-1).detach().cpu().numpy()
            predicted_cardinality = (
                torch.argmax(cardinality, dim=-1).detach().cpu().numpy()
            )
            for index, query in enumerate(batch):
                target = native_azimuth_to_bin360(query["azimuth_deg"])
                prediction = predicted[index].astype(np.float64)
                error = _circular_error(prediction, target)
                results.append(
                    {
                        key: query[key]
                        for key in (
                            "sample_id",
                            "episode_id",
                            "split",
                            "motion_case",
                            "source_slot_id",
                            "asset_id",
                            "caption",
                        )
                    }
                    | {
                        "ground_truth_azimuth_deg": target.tolist(),
                        "predicted_azimuth_bin": prediction.astype(int).tolist(),
                        "absolute_error_deg": error.tolist(),
                        "mean_absolute_error_deg": float(np.mean(error)),
                        "predicted_cardinality": predicted_cardinality[index]
                        .astype(int)
                        .tolist(),
                        "forward_batch_seconds": elapsed,
                        "batch_index": batch_index,
                    }
                )
    return results, time.perf_counter() - total_started


def _aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = np.asarray(
        [
            value
            for result in results
            for value in result["absolute_error_deg"]
        ],
        dtype=np.float64,
    )
    predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    targets: dict[str, list[np.ndarray]] = defaultdict(list)
    for result in results:
        sample_id = str(result["sample_id"])
        predictions[sample_id].append(
            np.asarray(result["predicted_azimuth_bin"], dtype=np.float64)
        )
        targets[sample_id].append(
            np.asarray(result["ground_truth_azimuth_deg"], dtype=np.float64)
        )
    difference_rates: list[float] = []
    prediction_separations: list[float] = []
    target_separations: list[float] = []
    for sample_id, values in predictions.items():
        if len(values) != 2 or len(targets[sample_id]) != 2:
            raise LegacyV4AudioError(
                f"{sample_id} does not have exactly two target queries"
            )
        difference_rates.append(float(np.mean(values[0] != values[1])))
        prediction_separations.append(
            float(np.mean(_circular_error(values[0], values[1])))
        )
        target_separations.append(
            float(np.mean(_circular_error(targets[sample_id][0], targets[sample_id][1])))
        )
    return {
        "query_count": len(results),
        "frame_count": int(errors.size),
        "mean_absolute_error_deg": float(np.mean(errors)),
        "median_absolute_error_deg": float(np.median(errors)),
        "p90_absolute_error_deg": float(np.percentile(errors, 90)),
        "mean_query_prediction_difference_rate": float(
            np.mean(difference_rates)
        ),
        "mean_predicted_source_pair_separation_deg": float(
            np.mean(prediction_separations)
        ),
        "mean_ground_truth_source_pair_separation_deg": float(
            np.mean(target_separations)
        ),
    }


def _train(
    *,
    torch: Any,
    model: Any,
    device: Any,
    queries: Sequence[Mapping[str, Any]],
    batch_size: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise LegacyV4AudioError("the from-scratch model has no trainable parameters")
    head = model.doa_estimator.fc_out
    head_before = head.weight.detach().clone()
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()
    model.CLAP.eval()
    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    doa_losses: list[float] = []
    cardinality_losses: list[float] = []
    gradient_norms: list[float] = []
    parameter_with_gradient_count = 0
    started = time.perf_counter()
    for step in range(steps):
        indices = [
            (step * batch_size + offset) % len(queries)
            for offset in range(batch_size)
        ]
        batch = [queries[index] for index in indices]
        mixtures = torch.from_numpy(
            np.stack([value["mixture"] for value in batch])
        ).to(device)
        null_cues = torch.zeros(
            len(batch),
            1,
            dtype=mixtures.dtype,
            device=device,
        )
        captions = [str(value["caption"]) for value in batch]
        targets = torch.from_numpy(
            np.stack(
                [
                    circular_gaussian_targets(value["azimuth_deg"])
                    for value in batch
                ]
            )
        ).to(device)
        target_cardinality = torch.ones(
            len(batch) * OUTPUT_FRAME_COUNT,
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        separated, doa, cardinality = model(mixtures, null_cues, captions)
        _assert_output_contract(
            torch=torch,
            separated=separated,
            doa=doa,
            cardinality=cardinality,
            batch_size=len(batch),
        )
        doa_loss = torch.nn.functional.binary_cross_entropy(doa, targets)
        cardinality_loss = torch.nn.functional.nll_loss(
            torch.log(cardinality.clamp_min(1.0e-8)).reshape(-1, 3),
            target_cardinality,
        )
        loss = 100.0 * doa_loss + cardinality_loss
        if not torch.isfinite(loss):
            raise LegacyV4AudioError("training loss is non-finite")
        loss.backward()
        squared_norm = torch.zeros((), device=device)
        parameter_with_gradient_count = 0
        for parameter in trainable:
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise LegacyV4AudioError("model gradient is non-finite")
            parameter_with_gradient_count += parameter.numel()
            squared_norm += torch.sum(torch.square(parameter.grad))
        if parameter_with_gradient_count == 0:
            raise LegacyV4AudioError("the model produced no gradients")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        doa_losses.append(float(doa_loss.detach().cpu()))
        cardinality_losses.append(float(cardinality_loss.detach().cpu()))
        gradient_norms.append(float(torch.sqrt(squared_norm).detach().cpu()))

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    head_change = float(
        torch.max(torch.abs(head.weight.detach() - head_before)).cpu()
    )
    if head_change <= 0.0:
        raise LegacyV4AudioError("the 360-degree head did not update")
    return {
        "status": "pass",
        "scope": "all_non_clap_trainable_v43_parameters",
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "losses": losses,
        "doa_losses": doa_losses,
        "cardinality_losses": cardinality_losses,
        "gradient_norms": gradient_norms,
        "maximum_absolute_head_weight_change": head_change,
        "elapsed_seconds": elapsed,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable
        ),
        "parameter_with_gradient_count": parameter_with_gradient_count,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "wet_stems_required": False,
        "objective": "100*circular_doa_bce + target_cardinality_nll",
    }


def main() -> int:
    args = _arguments()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output_root}")

    dataset_index = _load_json(args.dataset_index)
    trajectory_bank = _load_json(args.trajectory_bank)
    rir_plan = _load_json(args.rir_plan)
    samples = deterministic_split_samples(
        dataset_index,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
    )
    preparation_started = time.perf_counter()
    queries = _build_queries(
        samples=samples,
        dataset_index=dataset_index,
        trajectory_bank=trajectory_bank,
        rir_plan=rir_plan,
    )
    preparation_seconds = time.perf_counter() - preparation_started
    torch, model, device, model_load_seconds, model_audit = _load_model(
        clap_checkpoint=args.clap_checkpoint,
        device_name=args.device,
        seed=args.seed,
    )
    pre_results, pre_seconds = _run_model(
        torch=torch,
        model=model,
        device=device,
        queries=queries,
        batch_size=args.batch_size,
    )
    training = _train(
        torch=torch,
        model=model,
        device=device,
        queries=queries,
        batch_size=args.batch_size,
        steps=args.train_steps,
        learning_rate=args.learning_rate,
    )
    post_results, post_seconds = _run_model(
        torch=torch,
        model=model,
        device=device,
        queries=queries,
        batch_size=args.batch_size,
    )
    pre_metrics = _aggregate(pre_results)
    post_metrics = _aggregate(post_results)

    output_root.mkdir(parents=True)
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "outcome": "forward_loss_backward_parameter_update_contract_pass",
        "performance_gate_status": "not_run",
        "performance_gate_reason": (
            "a from-scratch smoke update does not measure convergence"
        ),
        "audio_only": True,
        "visual_media_read": False,
        "input_contract": {
            "duration_seconds": DURATION_SECONDS,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": SAMPLE_COUNT,
            "channel_layout": "native_hrtf_binaural_left_right",
            "crop": "none",
            "queries_per_mixture": 2,
            "query_mode": "text_only",
            "sources_simultaneously_active": 2,
        },
        "output_contract": {
            "frame_count": OUTPUT_FRAME_COUNT,
            "azimuth_bins": OUTPUT_AZIMUTH_BINS,
            "angle_convention": (
                "listener-local circular: front=0, right=90, rear=180, left=270"
            ),
        },
        "model": model_audit,
        "selection": {
            "split": args.split,
            "offset": args.offset,
            "sample_count": len(samples),
            "sample_ids": [str(value["sample_id"]) for value in samples],
        },
        "timing_seconds": {
            "input_preparation": preparation_seconds,
            "model_load": model_load_seconds,
            "pre_training_inference": pre_seconds,
            "training": training["elapsed_seconds"],
            "post_training_inference": post_seconds,
        },
        "pre_training_metrics": pre_metrics,
        "training": training,
        "post_training_metrics": post_metrics,
        "queries": post_results,
    }
    _write_json(output_root / "results.json", result)
    print(
        json.dumps(
            {
                "outcome": result["outcome"],
                "model": model_audit,
                "training": training,
                "pre_training_metrics": pre_metrics,
                "post_training_metrics": post_metrics,
                "results": str(output_root / "results.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
