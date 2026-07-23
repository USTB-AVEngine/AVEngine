#!/usr/bin/env python3
"""Train the isolated v4_3 text-selective binaural-360 model."""

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
    circular_gaussian_targets,
    native_azimuth_to_bin360,
)
from avengine_v43.hdf5_data import (
    EXPECTED_SPLIT_COUNTS,
    Hdf5QueryBank,
    QueryRef,
)
from run_training_smoke import (
    OUTPUT_AZIMUTH_BINS,
    OUTPUT_FRAME_COUNT,
    SAMPLE_COUNT,
    _assert_output_contract,
    _load_json,
    _load_model,
)


SCHEMA = "avengine_v43_text_selective_binaural360_training_v1"


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


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-h5", type=Path, required=True)
    parser.add_argument("--clap-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--epochs", type=_positive_int, default=20)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--validation-batch-size", type=_positive_int, default=16)
    parser.add_argument("--learning-rate", type=_positive_float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=0.0)
    parser.add_argument("--save-every-steps", type=_positive_int, default=100)
    parser.add_argument("--seed", type=_nonnegative_int, default=20_260_723)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument(
        "--max-global-steps",
        type=_positive_int,
        help="development/resume verification stop; omit for full training",
    )
    return parser.parse_args()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _cache_text_embeddings(
    *,
    torch: Any,
    model: Any,
    device: Any,
    queries_by_split: Mapping[str, Sequence[QueryRef]],
) -> dict[str, Any]:
    captions = sorted(
        {
            query.caption
            for queries in queries_by_split.values()
            for query in queries
        }
    )
    model.CLAP.eval()
    with torch.inference_mode():
        embeddings = model.text_encoder(captions).to(device)
    if embeddings.shape != (len(captions), 512) or not torch.isfinite(
        embeddings
    ).all():
        raise LegacyV4AudioError("CLAP text embedding cache is invalid")
    return {
        caption: embeddings[index].detach()
        for index, caption in enumerate(captions)
    }


def _batch_tensors(
    *,
    torch: Any,
    device: Any,
    bank: Hdf5QueryBank,
    batch: Sequence[QueryRef],
    text_cache: Mapping[str, Any],
    with_targets: bool,
) -> dict[str, Any]:
    mixtures = torch.from_numpy(bank.read_mixtures(batch)).to(device)
    text_embeddings = torch.stack(
        [text_cache[value.caption] for value in batch]
    )
    result = {
        "mixtures": mixtures,
        "null_cues": torch.zeros(
            len(batch),
            1,
            dtype=mixtures.dtype,
            device=device,
        ),
        "text_embeddings": text_embeddings,
    }
    if with_targets:
        result["doa_targets"] = torch.from_numpy(
            np.stack(
                [
                    circular_gaussian_targets(value.azimuth_deg)
                    for value in batch
                ]
            )
        ).to(device)
        result["cardinality_targets"] = torch.ones(
            len(batch) * OUTPUT_FRAME_COUNT,
            dtype=torch.long,
            device=device,
        )
    return result


def _losses(
    *,
    torch: Any,
    doa: Any,
    cardinality: Any,
    doa_targets: Any,
    cardinality_targets: Any,
) -> tuple[Any, Any, Any]:
    doa_loss = torch.nn.functional.binary_cross_entropy(doa, doa_targets)
    cardinality_loss = torch.nn.functional.nll_loss(
        torch.log(cardinality.clamp_min(1.0e-8)).reshape(-1, 3),
        cardinality_targets,
    )
    return 100.0 * doa_loss + cardinality_loss, doa_loss, cardinality_loss


def _circular_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = np.abs(predicted - target)
    return np.minimum(difference, 360.0 - difference)


def _evaluate(
    *,
    torch: Any,
    model: Any,
    device: Any,
    bank: Hdf5QueryBank,
    queries: Sequence[QueryRef],
    text_cache: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    errors: list[float] = []
    predictions_by_sample: dict[str, list[np.ndarray]] = defaultdict(list)
    targets_by_sample: dict[str, list[np.ndarray]] = defaultdict(list)
    weighted_loss = 0.0
    weighted_doa_loss = 0.0
    weighted_cardinality_loss = 0.0
    correct_cardinality = 0
    cardinality_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            tensors = _batch_tensors(
                torch=torch,
                device=device,
                bank=bank,
                batch=batch,
                text_cache=text_cache,
                with_targets=True,
            )
            separated, doa, cardinality = model(
                tensors["mixtures"],
                tensors["null_cues"],
                tensors["text_embeddings"],
            )
            _assert_output_contract(
                torch=torch,
                separated=separated,
                doa=doa,
                cardinality=cardinality,
                batch_size=len(batch),
            )
            loss, doa_loss, cardinality_loss = _losses(
                torch=torch,
                doa=doa,
                cardinality=cardinality,
                doa_targets=tensors["doa_targets"],
                cardinality_targets=tensors["cardinality_targets"],
            )
            weighted_loss += float(loss.cpu()) * len(batch)
            weighted_doa_loss += float(doa_loss.cpu()) * len(batch)
            weighted_cardinality_loss += (
                float(cardinality_loss.cpu()) * len(batch)
            )
            predicted = torch.argmax(doa, dim=-1).cpu().numpy()
            predicted_cardinality = torch.argmax(cardinality, dim=-1)
            correct_cardinality += int(
                torch.sum(predicted_cardinality == 1).cpu()
            )
            cardinality_count += predicted_cardinality.numel()
            for index, query in enumerate(batch):
                target = native_azimuth_to_bin360(query.azimuth_deg)
                prediction = predicted[index].astype(np.float64)
                errors.extend(_circular_error(prediction, target).tolist())
                sample_id = query.sample_id
                predictions_by_sample[sample_id].append(prediction)
                targets_by_sample[sample_id].append(target)
    torch.cuda.synchronize(device)

    query_difference_rates: list[float] = []
    predicted_separations: list[float] = []
    target_separations: list[float] = []
    for sample_id, predictions in predictions_by_sample.items():
        targets = targets_by_sample[sample_id]
        if len(predictions) != 2 or len(targets) != 2:
            raise LegacyV4AudioError(
                f"{sample_id} does not contain two comparable target queries"
            )
        query_difference_rates.append(
            float(np.mean(predictions[0] != predictions[1]))
        )
        predicted_separations.append(
            float(np.mean(_circular_error(predictions[0], predictions[1])))
        )
        target_separations.append(
            float(np.mean(_circular_error(targets[0], targets[1])))
        )
    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "sample_count": len(predictions_by_sample),
        "query_count": len(queries),
        "frame_count": int(error_array.size),
        "objective_loss": weighted_loss / len(queries),
        "doa_bce": weighted_doa_loss / len(queries),
        "cardinality_nll": weighted_cardinality_loss / len(queries),
        "circular_mean_absolute_error_deg": float(np.mean(error_array)),
        "circular_median_absolute_error_deg": float(np.median(error_array)),
        "circular_p90_absolute_error_deg": float(
            np.percentile(error_array, 90)
        ),
        "cardinality_accuracy": correct_cardinality / cardinality_count,
        "mean_query_prediction_difference_rate": float(
            np.mean(query_difference_rates)
        ),
        "mean_predicted_source_pair_separation_deg": float(
            np.mean(predicted_separations)
        ),
        "mean_ground_truth_source_pair_separation_deg": float(
            np.mean(target_separations)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _is_clap_state_key(key: str) -> bool:
    return key.startswith("CLAP.") or ".CLAP." in key


def _model_state_without_clap(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not _is_clap_state_key(key)
    }


def _checkpoint_payload(
    *,
    torch: Any,
    device: Any,
    model: Any,
    optimizer: Any,
    run_identity: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    epoch: int,
    next_batch_index: int,
    global_step: int,
    best_validation_mae_deg: float,
    best_epoch: int | None,
    partial_epoch_loss_sum: float,
    partial_epoch_loss_count: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model_state_without_clap": _model_state_without_clap(model),
        "optimizer_state": optimizer.state_dict(),
        "run_identity": dict(run_identity),
        "history": list(history),
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "global_step": global_step,
        "best_validation_mae_deg": best_validation_mae_deg,
        "best_epoch": best_epoch,
        "partial_epoch_loss_sum": partial_epoch_loss_sum,
        "partial_epoch_loss_count": partial_epoch_loss_count,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
    }


def _save_checkpoint(
    *,
    torch: Any,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    *,
    torch: Any,
    path: Path,
    device: Any,
    model: Any,
    optimizer: Any,
    run_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise LegacyV4AudioError("resume checkpoint schema is invalid")
    if payload.get("run_identity") != dict(run_identity):
        raise LegacyV4AudioError("resume checkpoint run identity does not match")
    state = payload.get("model_state_without_clap")
    if not isinstance(state, Mapping):
        raise LegacyV4AudioError("resume checkpoint lacks model state")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_clap = [
        key
        for key in incompatible.missing_keys
        if not _is_clap_state_key(key)
    ]
    if unexpected or missing_non_clap:
        raise LegacyV4AudioError(
            "resume checkpoint model keys are incompatible: "
            f"unexpected={unexpected[:5]}, missing={missing_non_clap[:5]}"
        )
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_rng_state = payload.get("cuda_rng_state")
    if cuda_rng_state is None:
        legacy_states = payload.get("cuda_rng_state_all")
        if (
            not isinstance(legacy_states, Sequence)
            or device.index is None
            or device.index >= len(legacy_states)
        ):
            raise LegacyV4AudioError("resume checkpoint lacks CUDA RNG state")
        cuda_rng_state = legacy_states[device.index]
    torch.cuda.set_rng_state(cuda_rng_state.cpu(), device)
    return payload


def _summary(
    *,
    torch: Any,
    device: Any,
    status: str,
    run_identity: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    data_preparation_seconds: float,
    model_load_seconds: float,
    history: Sequence[Mapping[str, Any]],
    global_step: int,
    next_epoch: int,
    next_batch_index: int,
    best_validation_mae_deg: float,
    best_epoch: int | None,
    test_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "research_only": True,
        "qualification_claim": False,
        "run_identity": dict(run_identity),
        "model": dict(model_audit),
        "split_contract": EXPECTED_SPLIT_COUNTS,
        "query_counts": {
            split: count * 2 for split, count in EXPECTED_SPLIT_COUNTS.items()
        },
        "global_step": global_step,
        "next_epoch": next_epoch,
        "next_batch_index": next_batch_index,
        "best_validation_mae_deg": (
            best_validation_mae_deg
            if math.isfinite(best_validation_mae_deg)
            else None
        ),
        "best_epoch": best_epoch,
        "history": list(history),
        "test_metrics": dict(test_metrics) if test_metrics is not None else None,
        "timing_seconds": {
            "data_preparation": data_preparation_seconds,
            "model_load": model_load_seconds,
        },
        "gpu_memory_bytes": {
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        },
    }


def main() -> int:
    args = _arguments()
    output_root = args.output_root.resolve()
    resume_path = args.resume_from.resolve() if args.resume_from else None
    if resume_path is None:
        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError(f"refusing to replace output: {output_root}")
        output_root.mkdir(parents=True)
    elif not output_root.is_dir():
        raise FileNotFoundError(output_root)

    dataset_h5_path = args.dataset_h5.resolve()
    clap_checkpoint_path = args.clap_checkpoint.resolve()
    run_identity = {
        "dataset_h5": str(dataset_h5_path),
        "clap_checkpoint": str(clap_checkpoint_path),
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "input_sample_count": SAMPLE_COUNT,
        "output_frame_count": OUTPUT_FRAME_COUNT,
        "output_azimuth_bins": OUTPUT_AZIMUTH_BINS,
        "query_mode": "text_only",
        "deterministic_algorithms": True,
        "tf32_enabled": False,
    }
    config_path = output_root / "run_config.json"
    if resume_path is None:
        _atomic_write_json(
            config_path,
            {
                "schema": SCHEMA,
                "run_identity": run_identity,
                "epochs_requested": args.epochs,
                "save_every_steps": args.save_every_steps,
            },
        )
    else:
        existing_config = _load_json(config_path)
        if existing_config.get("run_identity") != run_identity:
            raise LegacyV4AudioError("resume run_config identity does not match")

    preparation_started = time.perf_counter()
    bank = Hdf5QueryBank(dataset_h5_path, preload_mixtures=True)
    queries_by_split = {
        split: bank.queries(split) for split in EXPECTED_SPLIT_COUNTS
    }
    data_preparation_seconds = time.perf_counter() - preparation_started

    torch, model, device, model_load_seconds, model_audit = _load_model(
        clap_checkpoint=clap_checkpoint_path,
        device_name=args.device,
        seed=args.seed,
    )
    text_cache = _cache_text_embeddings(
        torch=torch,
        model=model,
        device=device,
        queries_by_split=queries_by_split,
    )
    torch.cuda.reset_peak_memory_stats(device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=float(args.weight_decay),
    )

    epoch = 0
    next_batch_index = 0
    global_step = 0
    best_validation_mae_deg = math.inf
    best_epoch: int | None = None
    history: list[Mapping[str, Any]] = []
    partial_epoch_loss_sum = 0.0
    partial_epoch_loss_count = 0
    if resume_path is not None:
        payload = _load_checkpoint(
            torch=torch,
            path=resume_path,
            device=device,
            model=model,
            optimizer=optimizer,
            run_identity=run_identity,
        )
        epoch = int(payload["epoch"])
        next_batch_index = int(payload["next_batch_index"])
        global_step = int(payload["global_step"])
        best_validation_mae_deg = float(payload["best_validation_mae_deg"])
        best_epoch = payload["best_epoch"]
        history = list(payload["history"])
        partial_epoch_loss_sum = float(payload["partial_epoch_loss_sum"])
        partial_epoch_loss_count = int(payload["partial_epoch_loss_count"])
        if (
            args.max_global_steps is not None
            and args.max_global_steps <= global_step
        ):
            raise LegacyV4AudioError(
                "--max-global-steps must exceed the resumed global step"
            )

    latest_path = output_root / "latest.pt"
    best_path = output_root / "best.pt"
    stop_requested = False
    while epoch < args.epochs:
        model.train()
        model.CLAP.eval()
        train_queries = queries_by_split["train"]
        order = np.random.default_rng(args.seed + epoch).permutation(
            len(train_queries)
        )
        batch_count = math.ceil(len(order) / args.batch_size)
        if next_batch_index < 0 or next_batch_index >= batch_count:
            raise LegacyV4AudioError("resume next_batch_index is out of range")
        epoch_started = time.perf_counter()
        for batch_index in range(next_batch_index, batch_count):
            index_slice = order[
                batch_index
                * args.batch_size : (batch_index + 1)
                * args.batch_size
            ]
            batch = [train_queries[int(index)] for index in index_slice]
            tensors = _batch_tensors(
                torch=torch,
                device=device,
                bank=bank,
                batch=batch,
                text_cache=text_cache,
                with_targets=True,
            )
            optimizer.zero_grad(set_to_none=True)
            separated, doa, cardinality = model(
                tensors["mixtures"],
                tensors["null_cues"],
                tensors["text_embeddings"],
            )
            _assert_output_contract(
                torch=torch,
                separated=separated,
                doa=doa,
                cardinality=cardinality,
                batch_size=len(batch),
            )
            loss, _, _ = _losses(
                torch=torch,
                doa=doa,
                cardinality=cardinality,
                doa_targets=tensors["doa_targets"],
                cardinality_targets=tensors["cardinality_targets"],
            )
            if not torch.isfinite(loss):
                raise LegacyV4AudioError("training loss is non-finite")
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None and not torch.isfinite(
                    parameter.grad
                ).all():
                    raise LegacyV4AudioError("training gradient is non-finite")
            optimizer.step()
            partial_epoch_loss_sum += float(loss.detach().cpu())
            partial_epoch_loss_count += 1
            global_step += 1
            completed_epoch = batch_index + 1 == batch_count
            next_batch_index = batch_index + 1

            if (
                global_step % args.save_every_steps == 0
                and not completed_epoch
            ):
                _save_checkpoint(
                    torch=torch,
                    path=latest_path,
                    payload=_checkpoint_payload(
                        torch=torch,
                        device=device,
                        model=model,
                        optimizer=optimizer,
                        run_identity=run_identity,
                        history=history,
                        epoch=epoch,
                        next_batch_index=next_batch_index,
                        global_step=global_step,
                        best_validation_mae_deg=best_validation_mae_deg,
                        best_epoch=best_epoch,
                        partial_epoch_loss_sum=partial_epoch_loss_sum,
                        partial_epoch_loss_count=partial_epoch_loss_count,
                    ),
                )
                print(
                    json.dumps(
                        {
                            "status": "training_in_progress",
                            "epoch": epoch,
                            "global_step": global_step,
                            "batch_index": batch_index + 1,
                            "batch_count": batch_count,
                            "latest_objective_loss": float(loss.detach().cpu()),
                        }
                    ),
                    flush=True,
                )
            if (
                args.max_global_steps is not None
                and global_step >= args.max_global_steps
            ):
                stop_requested = True
                if not completed_epoch:
                    _save_checkpoint(
                        torch=torch,
                        path=latest_path,
                        payload=_checkpoint_payload(
                            torch=torch,
                            device=device,
                            model=model,
                            optimizer=optimizer,
                            run_identity=run_identity,
                            history=history,
                            epoch=epoch,
                            next_batch_index=next_batch_index,
                            global_step=global_step,
                            best_validation_mae_deg=best_validation_mae_deg,
                            best_epoch=best_epoch,
                            partial_epoch_loss_sum=partial_epoch_loss_sum,
                            partial_epoch_loss_count=partial_epoch_loss_count,
                        ),
                    )
                    summary = _summary(
                        torch=torch,
                        device=device,
                        status="stopped_after_requested_global_steps",
                        run_identity=run_identity,
                        model_audit=model_audit,
                        data_preparation_seconds=data_preparation_seconds,
                        model_load_seconds=model_load_seconds,
                        history=history,
                        global_step=global_step,
                        next_epoch=epoch,
                        next_batch_index=next_batch_index,
                        best_validation_mae_deg=best_validation_mae_deg,
                        best_epoch=best_epoch,
                        test_metrics=None,
                    )
                    _atomic_write_json(output_root / "training_summary.json", summary)
                    print(json.dumps(summary, ensure_ascii=False, indent=2))
                    return 0
                break

        torch.cuda.synchronize(device)
        validation = _evaluate(
            torch=torch,
            model=model,
            device=device,
            bank=bank,
            queries=queries_by_split["validation"],
            text_cache=text_cache,
            batch_size=args.validation_batch_size,
        )
        if partial_epoch_loss_count != batch_count:
            raise LegacyV4AudioError(
                "resumed epoch loss accounting does not cover every batch"
            )
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "mean_train_objective_loss": (
                partial_epoch_loss_sum / partial_epoch_loss_count
            ),
            "training_seconds": time.perf_counter() - epoch_started,
            "validation": validation,
        }
        history.append(epoch_record)
        validation_mae = float(validation["circular_mean_absolute_error_deg"])
        if validation_mae < best_validation_mae_deg:
            best_validation_mae_deg = validation_mae
            best_epoch = epoch
            _save_checkpoint(
                torch=torch,
                path=best_path,
                payload=_checkpoint_payload(
                    torch=torch,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    run_identity=run_identity,
                    history=history,
                    epoch=epoch + 1,
                    next_batch_index=0,
                    global_step=global_step,
                    best_validation_mae_deg=best_validation_mae_deg,
                    best_epoch=best_epoch,
                    partial_epoch_loss_sum=0.0,
                    partial_epoch_loss_count=0,
                ),
            )
        epoch += 1
        next_batch_index = 0
        partial_epoch_loss_sum = 0.0
        partial_epoch_loss_count = 0
        _save_checkpoint(
            torch=torch,
            path=latest_path,
            payload=_checkpoint_payload(
                torch=torch,
                device=device,
                model=model,
                optimizer=optimizer,
                run_identity=run_identity,
                history=history,
                epoch=epoch,
                next_batch_index=0,
                global_step=global_step,
                best_validation_mae_deg=best_validation_mae_deg,
                best_epoch=best_epoch,
                partial_epoch_loss_sum=0.0,
                partial_epoch_loss_count=0,
            ),
        )
        summary = _summary(
            torch=torch,
            device=device,
            status=(
                "stopped_after_requested_global_steps"
                if stop_requested
                else "training_in_progress"
            ),
            run_identity=run_identity,
            model_audit=model_audit,
            data_preparation_seconds=data_preparation_seconds,
            model_load_seconds=model_load_seconds,
            history=history,
            global_step=global_step,
            next_epoch=epoch,
            next_batch_index=0,
            best_validation_mae_deg=best_validation_mae_deg,
            best_epoch=best_epoch,
            test_metrics=None,
        )
        _atomic_write_json(output_root / "training_summary.json", summary)
        if stop_requested:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

    best_payload = _load_checkpoint(
        torch=torch,
        path=best_path,
        device=device,
        model=model,
        optimizer=optimizer,
        run_identity=run_identity,
    )
    if int(best_payload["best_epoch"]) != int(best_epoch):
        raise LegacyV4AudioError("best checkpoint identity is inconsistent")
    test_metrics = _evaluate(
        torch=torch,
        model=model,
        device=device,
        bank=bank,
        queries=queries_by_split["test"],
        text_cache=text_cache,
        batch_size=args.validation_batch_size,
    )
    summary = _summary(
        torch=torch,
        device=device,
        status="pass",
        run_identity=run_identity,
        model_audit=model_audit,
        data_preparation_seconds=data_preparation_seconds,
        model_load_seconds=model_load_seconds,
        history=history,
        global_step=global_step,
        next_epoch=epoch,
        next_batch_index=0,
        best_validation_mae_deg=best_validation_mae_deg,
        best_epoch=best_epoch,
        test_metrics=test_metrics,
    )
    _atomic_write_json(output_root / "training_summary.json", summary)
    bank.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
