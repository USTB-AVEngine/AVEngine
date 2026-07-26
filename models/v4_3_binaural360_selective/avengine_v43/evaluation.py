"""Pure metric helpers for held-out binaural-360 evaluation reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AZIMUTH_REGIONS = ("front", "right", "rear", "left")
EVALUATION_REPORT_SCHEMA = "avengine_v43_binaural360_slice_evaluation_v1"
EVALUATION_PRODUCER_SCHEMA = "avengine_v43_evaluation_producer_v1"
EVALUATION_PRODUCER_CODE_FILES = frozenset(
    {
        "evaluate_dataset.py",
        "avengine_v43/evaluation.py",
        "avengine_v43/hdf5_data.py",
        "avengine_v43/labels.py",
        "avengine_v43/publication.py",
        "run_training_smoke.py",
        "train.py",
        "visualize_inference.py",
        "avengine_v43/model.py",
    }
)
EVALUATION_PRODUCER_RUNTIME_FIELDS = frozenset(
    {
        "python_implementation",
        "python_version",
        "python_executable",
        "platform",
        "numpy_version",
        "torch_version",
        "torch_cuda_runtime_version",
    }
)
MOTION_CASE_LABELS = frozenset(
    {
        "both_moving",
        "source1_moving_source2_static",
        "source1_static_source2_moving",
        "static_static",
    }
)
SOURCE_MOTION_LABELS = frozenset({"moving", "static"})
CAPTION_LABELS = frozenset({"cat meowing", "dog barking", "human speech"})
COMPARISON_SLICE_LABELS = {
    "motion_case": MOTION_CASE_LABELS,
    "source_motion": SOURCE_MOTION_LABELS,
    "caption": CAPTION_LABELS,
    "target_region": frozenset(AZIMUTH_REGIONS),
}


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_producer(value: Any) -> bool:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "code", "runtime"}
        or value.get("schema") != EVALUATION_PRODUCER_SCHEMA
        or not isinstance(value.get("code"), Mapping)
        or set(value["code"]) != EVALUATION_PRODUCER_CODE_FILES
        or not isinstance(value.get("runtime"), Mapping)
        or set(value["runtime"]) != EVALUATION_PRODUCER_RUNTIME_FIELDS
    ):
        return False
    if not all(
        isinstance(record, Mapping)
        and set(record) == {"sha256"}
        and _valid_sha256(record.get("sha256"))
        for record in value["code"].values()
    ):
        return False
    return all(
        isinstance(runtime_value, str) and bool(runtime_value)
        for runtime_value in value["runtime"].values()
    )


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and bool(np.isfinite(value))
    )


def _contains_only_finite_numbers(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(
            _contains_only_finite_numbers(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return all(_contains_only_finite_numbers(item) for item in value)
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    return True


def _valid_overall_metric_ranges(overall: Mapping[str, Any]) -> bool:
    angle_metrics = (
        "mean_absolute_error_deg",
        "median_absolute_error_deg",
        "p90_absolute_error_deg",
        "uniform_target_region_macro_mean_absolute_error_deg",
    )
    rate_metrics = ("error_over_45deg_rate", "error_over_90deg_rate")
    fractions = overall.get("target_region_frame_fractions")
    return (
        all(
            _is_finite_number(overall.get(name))
            and 0.0 <= float(overall[name]) <= 180.0
            for name in angle_metrics
        )
        and all(
            _is_finite_number(overall.get(name))
            and 0.0 <= float(overall[name]) <= 1.0
            for name in rate_metrics
        )
        and isinstance(fractions, Mapping)
        and set(fractions) == set(AZIMUTH_REGIONS)
        and all(
            _is_finite_number(value) and 0.0 <= float(value) <= 1.0
            for value in fractions.values()
        )
        and bool(
            np.isclose(
                sum(float(value) for value in fractions.values()),
                1.0,
                rtol=0.0,
                atol=1.0e-9,
            )
        )
    )


def _validated_slice_rows(
    report: Mapping[str, Any],
    *,
    owner: str,
    slice_name: str,
    expected_labels: frozenset[str],
    expected_sample_count: int,
    expected_frame_count: int,
    expected_query_count: int,
) -> dict[str, Mapping[str, Any]]:
    slices = report.get("slices")
    raw_rows = slices.get(slice_name) if isinstance(slices, Mapping) else None
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{owner} evaluation report lacks {slice_name} slice")
    rows: dict[str, Mapping[str, Any]] = {}
    frame_count = 0
    query_count = 0
    for raw in raw_rows:
        if (
            not isinstance(raw, Mapping)
            or not _contains_only_finite_numbers(raw)
            or any(not isinstance(key, str) for key in raw)
            or any(
                key != "label" and not _is_finite_number(value)
                for key, value in raw.items()
            )
            or any(
                not 0.0 <= float(value) <= 180.0
                for key, value in raw.items()
                if key.endswith("_deg")
            )
            or any(
                not 0.0 <= float(value) <= 1.0
                for key, value in raw.items()
                if key.endswith("_rate")
            )
        ):
            raise ValueError(f"{owner} {slice_name} slice row is invalid")
        label = raw.get("label")
        if not isinstance(label, str) or not label or label in rows:
            raise ValueError(
                f"{owner} {slice_name} slice labels are invalid or duplicated"
            )
        row_frame_count = raw.get("frame_count")
        mean_error = raw.get("mean_absolute_error_deg")
        if (
            not isinstance(row_frame_count, (int, np.integer))
            or isinstance(row_frame_count, (bool, np.bool_))
            or int(row_frame_count) <= 0
            or not _is_finite_number(mean_error)
            or not 0.0 <= float(mean_error) <= 180.0
        ):
            raise ValueError(f"{owner} {slice_name} slice row is invalid")
        frame_count += int(row_frame_count)
        row_query_count = raw.get("query_count")
        if slice_name != "target_region":
            row_sample_count = raw.get("sample_count")
            if (
                not isinstance(row_query_count, (int, np.integer))
                or isinstance(row_query_count, (bool, np.bool_))
                or int(row_query_count) <= 0
                or int(row_frame_count) != int(row_query_count) * 75
                or not isinstance(row_sample_count, (int, np.integer))
                or isinstance(row_sample_count, (bool, np.bool_))
                or not 1 <= int(row_sample_count) <= expected_sample_count
            ):
                raise ValueError(f"{owner} {slice_name} query count is invalid")
            query_count += int(row_query_count)
        rows[label] = raw
    if set(rows) != expected_labels:
        raise ValueError(f"{owner} {slice_name} slice labels differ")
    if frame_count != expected_frame_count or (
        slice_name != "target_region"
        and query_count != expected_query_count
    ):
        raise ValueError(f"{owner} {slice_name} slice is incomplete")
    return rows


def normalize_360(values: Any) -> np.ndarray:
    result = np.mod(np.asarray(values, dtype=np.float64), 360.0)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("azimuth values must be finite and non-empty")
    return result


def circular_error_deg(predicted: Any, target: Any) -> np.ndarray:
    predicted_array = normalize_360(predicted)
    target_array = normalize_360(target)
    if predicted_array.shape != target_array.shape:
        raise ValueError("prediction and target shapes differ")
    difference = np.abs(predicted_array - target_array)
    return np.minimum(difference, 360.0 - difference)


def azimuth_region(values: Any) -> np.ndarray:
    """Map AVEngine azimuths to front/right/rear/left quadrants."""

    signed = (normalize_360(values) + 180.0) % 360.0 - 180.0
    result = np.full(signed.shape, "front", dtype="<U5")
    result[(signed >= 45.0) & (signed < 135.0)] = "right"
    result[(signed >= 135.0) | (signed < -135.0)] = "rear"
    result[(signed >= -135.0) & (signed < -45.0)] = "left"
    return result


def source_motion_label(motion_case: str, source_index: int) -> str:
    """Return moving/static without assuming human/dog source roles."""

    normalized = str(motion_case).casefold().replace("-", "_")
    if source_index not in (0, 1):
        raise ValueError("source_index must be 0 or 1")
    if "both_moving" in normalized:
        return "moving"
    if "static_static" in normalized:
        return "static"
    moving_token = ("source1_moving", "source2_moving")[source_index]
    static_token = ("source1_static", "source2_static")[source_index]
    if moving_token in normalized:
        return "moving"
    if static_token in normalized:
        return "static"
    return "unknown"


def summarize_errors(errors: Any) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("errors must be finite and non-empty")
    return {
        "frame_count": int(values.size),
        "mean_absolute_error_deg": float(np.mean(values)),
        "median_absolute_error_deg": float(np.median(values)),
        "p90_absolute_error_deg": float(np.percentile(values, 90)),
        "p95_absolute_error_deg": float(np.percentile(values, 95)),
        "maximum_absolute_error_deg": float(np.max(values)),
        "error_over_15deg_rate": float(np.mean(values > 15.0)),
        "error_over_45deg_rate": float(np.mean(values > 45.0)),
        "error_over_90deg_rate": float(np.mean(values > 90.0)),
    }


def summarize_query_records(
    records: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[group_key])].append(record)
    result: list[dict[str, Any]] = []
    for label in sorted(grouped):
        group = grouped[label]
        errors = np.concatenate(
            [np.asarray(value["errors_deg"], dtype=np.float64) for value in group]
        )
        result.append(
            {
                "label": label,
                "sample_count": len({str(value["sample_id"]) for value in group}),
                "query_count": len(group),
                **summarize_errors(errors),
            }
        )
    return result


def summarize_target_regions(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    errors_by_region: dict[str, list[np.ndarray]] = defaultdict(list)
    front_rear_by_region: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        targets = normalize_360(record["targets_deg"])
        predictions = normalize_360(record["predictions_deg"])
        errors = circular_error_deg(predictions, targets)
        target_regions = azimuth_region(targets)
        predicted_regions = azimuth_region(predictions)
        for region in AZIMUTH_REGIONS:
            mask = target_regions == region
            if not np.any(mask):
                continue
            errors_by_region[region].append(errors[mask])
            front_rear = (
                ((target_regions == "front") & (predicted_regions == "rear"))
                | ((target_regions == "rear") & (predicted_regions == "front"))
            )
            front_rear_by_region[region].append(front_rear[mask])
    result: list[dict[str, Any]] = []
    for region in AZIMUTH_REGIONS:
        if region not in errors_by_region:
            continue
        errors = np.concatenate(errors_by_region[region])
        confusions = np.concatenate(front_rear_by_region[region])
        result.append(
            {
                "label": region,
                **summarize_errors(errors),
                "front_rear_confusion_rate": float(np.mean(confusions)),
            }
        )
    return result


def error_histogram(
    records: Iterable[Mapping[str, Any]], *, bin_width_deg: int = 5
) -> dict[str, list[Any]]:
    if bin_width_deg <= 0 or 180 % bin_width_deg:
        raise ValueError("bin_width_deg must divide 180")
    errors = np.concatenate(
        [np.asarray(record["errors_deg"], dtype=np.float64) for record in records]
    )
    edges = np.arange(0.0, 180.0 + bin_width_deg, bin_width_deg)
    counts, _ = np.histogram(errors, bins=edges)
    return {
        "edges_deg": edges.tolist(),
        "counts": counts.astype(int).tolist(),
    }


def compare_evaluation_reports(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a compact same-checkpoint room-generalization comparison."""

    report_context: dict[str, dict[str, Any]] = {}
    validated_slices: dict[
        str, dict[str, dict[str, Mapping[str, Any]]]
    ] = {}
    for owner, report in (("baseline", baseline), ("candidate", candidate)):
        inputs = report.get("inputs")
        producer = report.get("producer")
        overall = report.get("overall")
        sample_count = report.get("sample_count")
        query_count = report.get("query_count")
        provenance = report.get("provenance")
        if (
            report.get("schema") != EVALUATION_REPORT_SCHEMA
            or report.get("status") != "pass"
            or report.get("split") != "test"
            or report.get("research_only") is not True
            or report.get("qualification_claim") is not False
            or not isinstance(overall, Mapping)
            or not _contains_only_finite_numbers(overall)
            or not _valid_overall_metric_ranges(overall)
            or not isinstance(inputs, Mapping)
            or not _valid_producer(producer)
            or not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 1
            or not isinstance(query_count, int)
            or isinstance(query_count, bool)
            or query_count != sample_count * 2
            or not isinstance(provenance, Mapping)
            or provenance.get("schema")
            != "avengine_v43_evaluation_provenance_v1"
            or provenance.get("source_caption_authority")
            != "hdf5_query_bank_cross_checked_with_dataset_index"
            or provenance.get("motion_case_authority")
            != (
                "hdf5_input_identity.trajectory_bank_cross_checked_with_"
                "dataset_index"
            )
            or not isinstance(provenance.get("trajectory_bank"), str)
            or not provenance["trajectory_bank"]
            or not _valid_sha256(
                provenance.get("trajectory_bank_sha256")
            )
            or not isinstance(
                provenance.get("dataset_index_identity_exact_match"), bool
            )
            or not isinstance(
                provenance.get("legacy_training_identity_compatibility"),
                bool,
            )
        ):
            raise ValueError(f"{owner} evaluation report is invalid")
        overall_frame_count = overall.get("frame_count")
        if (
            not isinstance(overall_frame_count, int)
            or isinstance(overall_frame_count, bool)
            or overall_frame_count != query_count * 75
        ):
            raise ValueError(f"{owner} evaluation overall frame count is invalid")
        validated_slices[owner] = {
            slice_name: _validated_slice_rows(
                report,
                owner=owner,
                slice_name=slice_name,
                expected_labels=expected_labels,
                expected_sample_count=sample_count,
                expected_frame_count=overall_frame_count,
                expected_query_count=query_count,
            )
            for slice_name, expected_labels in COMPARISON_SLICE_LABELS.items()
        }
        context = {
            "training_room_id": report.get("training_room_id"),
            "evaluation_room_id": report.get("evaluation_room_id"),
            "evaluation_regime": report.get("evaluation_regime"),
            "dataset_h5": inputs.get("dataset_h5"),
            "dataset_h5_sha256": inputs.get("dataset_h5_sha256"),
            "dataset_index": inputs.get("dataset_index"),
            "dataset_index_sha256": inputs.get("dataset_index_sha256"),
            "training_dataset_h5": inputs.get("training_dataset_h5"),
            "training_dataset_h5_sha256": inputs.get(
                "training_dataset_h5_sha256"
            ),
            "training_dataset_index": inputs.get("training_dataset_index"),
            "training_dataset_index_sha256": inputs.get(
                "training_dataset_index_sha256"
            ),
            "checkpoint": inputs.get("checkpoint"),
            "checkpoint_sha256": inputs.get("checkpoint_sha256"),
            "clap_checkpoint": inputs.get("clap_checkpoint"),
            "clap_checkpoint_sha256": inputs.get("clap_checkpoint_sha256"),
            "producer": producer,
            "provenance": provenance,
        }
        string_keys = (
            "training_room_id",
            "evaluation_room_id",
            "evaluation_regime",
            "dataset_h5",
            "dataset_index",
            "training_dataset_h5",
            "training_dataset_index",
            "checkpoint",
            "clap_checkpoint",
        )
        if any(
            not isinstance(context[key], str) or not context[key]
            for key in string_keys
        ) or any(
            not _valid_sha256(context[key])
            for key in (
                "dataset_h5_sha256",
                "dataset_index_sha256",
                "training_dataset_h5_sha256",
                "training_dataset_index_sha256",
                "checkpoint_sha256",
                "clap_checkpoint_sha256",
            )
        ):
            raise ValueError(f"{owner} evaluation report context is invalid")
        report_context[owner] = context

    baseline_context = report_context["baseline"]
    candidate_context = report_context["candidate"]
    if (
        baseline_context["evaluation_regime"] != "same_room_held_out"
        or baseline_context["evaluation_room_id"]
        != baseline_context["training_room_id"]
        or baseline_context["dataset_h5"]
        != baseline_context["training_dataset_h5"]
        or baseline_context["dataset_h5_sha256"]
        != baseline_context["training_dataset_h5_sha256"]
        or baseline_context["dataset_index"]
        != baseline_context["training_dataset_index"]
        or baseline_context["dataset_index_sha256"]
        != baseline_context["training_dataset_index_sha256"]
    ):
        raise ValueError("baseline must be a same_room_held_out evaluation")
    if (
        baseline_context["provenance"].get("evaluation_dataset_identity_mode")
        != "legacy_training_identity_compatibility"
        or baseline_context["provenance"].get(
            "dataset_index_identity_exact_match"
        )
        is not False
        or baseline_context["provenance"].get(
            "legacy_training_identity_compatibility"
        )
        is not True
    ):
        raise ValueError("baseline provenance mode is invalid")
    if (
        candidate_context["evaluation_regime"] != "cross_room_zero_shot"
        or candidate_context["evaluation_room_id"]
        == candidate_context["training_room_id"]
        or candidate_context["dataset_h5"]
        == candidate_context["training_dataset_h5"]
        or candidate_context["dataset_h5_sha256"]
        == candidate_context["training_dataset_h5_sha256"]
        or candidate_context["dataset_index"]
        == candidate_context["training_dataset_index"]
        or candidate_context["dataset_index_sha256"]
        == candidate_context["training_dataset_index_sha256"]
    ):
        raise ValueError("candidate must be a cross_room_zero_shot evaluation")
    if (
        candidate_context["provenance"].get(
            "evaluation_dataset_identity_mode"
        )
        != "evaluation_hdf5_exact_index_identity"
        or candidate_context["provenance"].get(
            "dataset_index_identity_exact_match"
        )
        is not True
        or candidate_context["provenance"].get(
            "legacy_training_identity_compatibility"
        )
        is not False
    ):
        raise ValueError("candidate provenance mode is invalid")
    if (
        baseline_context["training_room_id"]
        != candidate_context["training_room_id"]
    ):
        raise ValueError("evaluation reports must share one training room")
    for key, label in (
        ("training_dataset_h5", "training HDF5"),
        ("training_dataset_h5_sha256", "training HDF5 SHA256"),
        ("training_dataset_index", "training dataset index"),
        ("training_dataset_index_sha256", "training dataset index SHA256"),
    ):
        if baseline_context[key] != candidate_context[key]:
            raise ValueError(f"evaluation reports must share one {label}")

    baseline_checkpoint = baseline_context["checkpoint"]
    candidate_checkpoint = candidate_context["checkpoint"]
    if baseline_checkpoint != candidate_checkpoint:
        raise ValueError("evaluation reports must use the same checkpoint")
    checkpoint_sha256 = baseline_context["checkpoint_sha256"]
    if checkpoint_sha256 != candidate_context["checkpoint_sha256"]:
        raise ValueError("evaluation reports must use the same checkpoint SHA256")
    baseline_clap = baseline_context["clap_checkpoint"]
    candidate_clap = candidate_context["clap_checkpoint"]
    if baseline_clap != candidate_clap:
        raise ValueError("evaluation reports must use the same CLAP checkpoint")
    clap_sha256 = baseline_context["clap_checkpoint_sha256"]
    if clap_sha256 != candidate_context["clap_checkpoint_sha256"]:
        raise ValueError("evaluation reports must use the same CLAP SHA256")
    if baseline_context["producer"] != candidate_context["producer"]:
        raise ValueError("evaluation reports must use the same producer identity")
    metric_names = (
        "mean_absolute_error_deg",
        "median_absolute_error_deg",
        "p90_absolute_error_deg",
        "error_over_45deg_rate",
        "error_over_90deg_rate",
        "uniform_target_region_macro_mean_absolute_error_deg",
    )
    metrics = {}
    for name in metric_names:
        baseline_value = baseline["overall"].get(name)
        candidate_value = candidate["overall"].get(name)
        if not _is_finite_number(baseline_value) or not _is_finite_number(
            candidate_value
        ):
            raise ValueError(f"evaluation reports lack numeric {name}")
        metrics[name] = {
            "baseline": float(baseline_value),
            "candidate": float(candidate_value),
            "delta": float(candidate_value) - float(baseline_value),
        }

    slice_comparisons = {}
    for slice_name, expected_labels in COMPARISON_SLICE_LABELS.items():
        baseline_rows = validated_slices["baseline"][slice_name]
        candidate_rows = validated_slices["candidate"][slice_name]
        rows = []
        for label in sorted(expected_labels):
            before = float(baseline_rows[label]["mean_absolute_error_deg"])
            after = float(candidate_rows[label]["mean_absolute_error_deg"])
            rows.append(
                {
                    "label": label,
                    "baseline_mean_absolute_error_deg": before,
                    "candidate_mean_absolute_error_deg": after,
                    "delta_mean_absolute_error_deg": after - before,
                    "baseline_frame_count": int(baseline_rows[label]["frame_count"]),
                    "candidate_frame_count": int(candidate_rows[label]["frame_count"]),
                }
            )
        slice_comparisons[slice_name] = rows
    return {
        "schema": "avengine_v43_room_generalization_comparison_v1",
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "training_room_id": baseline_context["training_room_id"],
        "baseline_room_id": baseline_context["evaluation_room_id"],
        "candidate_room_id": candidate_context["evaluation_room_id"],
        "checkpoint": baseline_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "clap_checkpoint": baseline_clap,
        "clap_checkpoint_sha256": clap_sha256,
        "producer": baseline_context["producer"],
        "metrics": metrics,
        "slices": slice_comparisons,
    }


__all__ = [
    "AZIMUTH_REGIONS",
    "CAPTION_LABELS",
    "COMPARISON_SLICE_LABELS",
    "EVALUATION_PRODUCER_CODE_FILES",
    "EVALUATION_PRODUCER_RUNTIME_FIELDS",
    "EVALUATION_REPORT_SCHEMA",
    "MOTION_CASE_LABELS",
    "SOURCE_MOTION_LABELS",
    "azimuth_region",
    "circular_error_deg",
    "compare_evaluation_reports",
    "error_histogram",
    "normalize_360",
    "source_motion_label",
    "summarize_errors",
    "summarize_query_records",
    "summarize_target_regions",
]
