from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine_v43.evaluation import (
    COMPARISON_SLICE_LABELS,
    EVALUATION_PRODUCER_CODE_FILES,
    EVALUATION_PRODUCER_RUNTIME_FIELDS,
    EVALUATION_PRODUCER_SCHEMA,
    azimuth_region,
    circular_error_deg,
    compare_evaluation_reports,
    source_motion_label,
    summarize_errors,
    summarize_target_regions,
)
from avengine_v43.hdf5_data import (
    EVALUATION_BUILDER_CODE_FILES,
    EVALUATION_BUILDER_PRODUCER_SCHEMA,
    EVALUATION_HDF5_SCHEMA,
    HDF5_SCHEMA,
    EvaluationQueryBank,
    QueryRef,
    open_query_bank,
)
from avengine_v43.labels import LegacyV4AudioError
from avengine_v43.publication import atomic_publish_directory
from build_evaluation_hdf5 import (
    BUILDER_CODE_FILES,
    BUILDER_PRODUCER_SCHEMA,
    NATIVE_DELIVERY_SCHEMA,
    NATIVE_OUTPUT_CLOSURE_SCHEMA,
    _builder_producer_identity,
    _load_h5py,
    _publish_pair_no_replace,
    _sha256_file,
    _validate_native_output,
)
from evaluate_dataset import (
    EVALUATION_INDEX_SCHEMA,
    TRAINING_INDEX_SCHEMA,
    _resolve_evaluation_context,
    _validate_evaluation_index_header,
    _validated_sample_metadata,
    _validated_split_sample_episode_ids,
)


def _producer(seed: str = "a"):
    return {
        "schema": EVALUATION_PRODUCER_SCHEMA,
        "code": {
            name: {"sha256": seed * 64}
            for name in EVALUATION_PRODUCER_CODE_FILES
        },
        "runtime": {
            name: f"{name}-test"
            for name in EVALUATION_PRODUCER_RUNTIME_FIELDS
        },
    }


def _builder_producer(seed: str = "a"):
    return {
        "schema": EVALUATION_BUILDER_PRODUCER_SCHEMA,
        "code": {
            name: {"sha256": seed * 64}
            for name in EVALUATION_BUILDER_CODE_FILES
        },
        "runtime": {
            "python_implementation": "CPython",
            "python_version": "3.10.15",
            "python_executable": "/tmp/python",
            "platform": "test-platform",
            "numpy_version": "2.0.0",
            "h5py_version": "3.13.0",
            "soundfile_version": "0.13.1",
            "libsndfile_version": "1.2.2",
        },
    }


def _evaluation_report(
    room_id: str,
    *,
    mean: float = 10.0,
    macro: float = 11.0,
    regime: str | None = None,
    training_room_id: str = "apartment_0000",
):
    sample_count = 4
    query_count = sample_count * 2
    frame_count = query_count * 75

    def rows(slice_name, labels):
        ordered = sorted(labels)
        result = []
        query_counts = [
            query_count // len(ordered)
            + (1 if index < query_count % len(ordered) else 0)
            for index in range(len(ordered))
        ]
        if slice_name == "target_region":
            frame_counts = [
                frame_count // len(ordered)
                + (1 if index < frame_count % len(ordered) else 0)
                for index in range(len(ordered))
            ]
        else:
            frame_counts = [count * 75 for count in query_counts]
        for index, label in enumerate(ordered):
            row = {
                "label": label,
                "sample_count": sample_count,
                "frame_count": frame_counts[index],
                "mean_absolute_error_deg": mean,
            }
            if slice_name != "target_region":
                row["query_count"] = query_counts[index]
            result.append(row)
        return result

    same_room = room_id == training_room_id
    return {
        "schema": "avengine_v43_binaural360_slice_evaluation_v1",
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "split": "test",
        "sample_count": sample_count,
        "query_count": query_count,
        "training_room_id": training_room_id,
        "evaluation_room_id": room_id,
        "evaluation_regime": regime or (
            "same_room_held_out" if same_room else "cross_room_zero_shot"
        ),
        "inputs": {
            "dataset_h5": (
                "/tmp/training.h5" if same_room else "/tmp/evaluation.h5"
            ),
            "dataset_h5_sha256": "4" * 64 if same_room else "5" * 64,
            "dataset_index": (
                "/tmp/training-index.json"
                if same_room
                else "/tmp/evaluation-index.json"
            ),
            "dataset_index_sha256": "6" * 64 if same_room else "7" * 64,
            "training_dataset_h5": "/tmp/training.h5",
            "training_dataset_h5_sha256": "4" * 64,
            "training_dataset_index": "/tmp/training-index.json",
            "training_dataset_index_sha256": "6" * 64,
            "checkpoint": "/tmp/best.pt",
            "checkpoint_sha256": "1" * 64,
            "clap_checkpoint": "/tmp/clap.pt",
            "clap_checkpoint_sha256": "2" * 64,
        },
        "producer": _producer(),
        "provenance": {
            "schema": "avengine_v43_evaluation_provenance_v1",
            "evaluation_dataset_identity_mode": (
                "legacy_training_identity_compatibility"
                if same_room
                else "evaluation_hdf5_exact_index_identity"
            ),
            "dataset_index_identity_exact_match": not same_room,
            "legacy_training_identity_compatibility": same_room,
            "source_caption_authority": (
                "hdf5_query_bank_cross_checked_with_dataset_index"
            ),
            "motion_case_authority": (
                "hdf5_input_identity.trajectory_bank_cross_checked_with_"
                "dataset_index"
            ),
            "trajectory_bank": "/tmp/trajectory.json",
            "trajectory_bank_sha256": "9" * 64,
        },
        "overall": {
            "frame_count": frame_count,
            "mean_absolute_error_deg": mean,
            "median_absolute_error_deg": mean,
            "p90_absolute_error_deg": mean,
            "error_over_45deg_rate": 0.1,
            "error_over_90deg_rate": 0.05,
            "uniform_target_region_macro_mean_absolute_error_deg": macro,
            "target_region_frame_fractions": {
                "front": 0.25,
                "right": 0.25,
                "rear": 0.25,
                "left": 0.25,
            },
        },
        "slices": {
            slice_name: rows(slice_name, labels)
            for slice_name, labels in COMPARISON_SLICE_LABELS.items()
        },
    }


def test_circular_error_preserves_front_rear_distinction():
    result = circular_error_deg([359.0, 0.0, 180.0], [1.0, 180.0, 0.0])
    assert result.tolist() == [2.0, 180.0, 180.0]


def test_azimuth_regions_follow_avengine_listener_convention():
    result = azimuth_region([0, 44.9, 45, 134.9, 135, 225, 315, 359])
    assert result.tolist() == [
        "front", "front", "right", "right", "rear", "left", "front", "front"
    ]


@pytest.mark.parametrize(
    ("motion_case", "source_index", "expected"),
    [
        ("static_static", 0, "static"),
        ("recombined_both_moving", 1, "moving"),
        ("source1_moving_source2_static", 0, "moving"),
        ("source1_moving_source2_static", 1, "static"),
        ("source1_static_source2_moving", 0, "static"),
        ("source1_static_source2_moving", 1, "moving"),
    ],
)
def test_source_motion_stays_generic(motion_case, source_index, expected):
    assert source_motion_label(motion_case, source_index) == expected


def test_error_summary_and_regions_report_tail_failures():
    summary = summarize_errors([0, 1, 2, 180])
    assert summary["mean_absolute_error_deg"] == pytest.approx(45.75)
    assert summary["error_over_90deg_rate"] == pytest.approx(0.25)
    regions = summarize_target_regions(
        [
            {
                "targets_deg": np.asarray([0, 90, 180, 270]),
                "predictions_deg": np.asarray([180, 90, 0, 270]),
                "errors_deg": np.asarray([180, 0, 180, 0]),
            }
        ]
    )
    by_label = {row["label"]: row for row in regions}
    assert by_label["front"]["front_rear_confusion_rate"] == 1.0
    assert by_label["rear"]["front_rear_confusion_rate"] == 1.0
    assert by_label["right"]["mean_absolute_error_deg"] == 0.0


def test_variable_size_evaluation_bank(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "evaluation.h5"
    string_type = h5py.string_dtype(encoding="utf-8")
    builder_producer = _builder_producer()
    with h5py.File(path, "w") as file:
        file.attrs["schema"] = EVALUATION_HDF5_SCHEMA
        file.attrs["completed_sample_count"] = 2
        file.attrs["room_id"] = "kujiale_0020"
        file.attrs["research_only"] = True
        file.attrs["qualification_claim"] = False
        file.attrs["split"] = "test"
        file.attrs["sample_rate_hz"] = 16_000
        file.attrs["duration_seconds"] = 5.0
        file.attrs["source_count"] = 2
        file.attrs["output_frame_count"] = 75
        file.attrs["input_identity_json"] = json.dumps(
            {
                "dataset_index": str(tmp_path / "index.json"),
                "trajectory_bank": str(tmp_path / "trajectory.json"),
                "trajectory_bank_sha256": "1" * 64,
                "rir_plan": str(tmp_path / "rir.json"),
                "rir_plan_sha256": "2" * 64,
                "builder_producer_identity": builder_producer,
            }
        )
        file.attrs["builder_producer_identity_json"] = json.dumps(
            builder_producer
        )
        file.attrs["caption_table_json"] = '["cat meowing", "dog barking"]'
        file.create_dataset("mixture", data=np.ones((2, 80_000, 2), np.float32))
        file.create_dataset("azimuth_deg", data=np.zeros((2, 2, 75), np.float32))
        file.create_dataset("caption_id", data=np.asarray([[0, 1], [1, 0]], np.uint8))
        file.create_dataset(
            "sample_id", data=np.asarray(["sample0", "sample1"], dtype=object), dtype=string_type
        )
        file.create_dataset(
            "episode_id",
            data=np.asarray(["episode0", "episode1"], dtype=object),
            dtype=string_type,
        )
    with EvaluationQueryBank(path) as bank:
        assert len(bank.queries("test")) == 4
        assert bank.read_mixtures(bank.queries("test")[:2]).shape == (2, 80_000, 2)
        assert bank.identity()["room_id"] == "kujiale_0020"
        assert len(bank.identity()["sample_episode_sequence_sha256"]) == 64
        assert (
            bank.identity()["builder_producer_identity"]
            == builder_producer
        )
    with open_query_bank(path) as bank:
        assert [query.caption for query in bank.queries("test")[:2]] == [
            "cat meowing",
            "dog barking",
        ]


def test_evaluation_bank_fails_closed_on_room_and_episode_identity(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "invalid-evaluation.h5"
    string_type = h5py.string_dtype(encoding="utf-8")
    builder_producer = _builder_producer()
    with h5py.File(path, "w") as file:
        file.attrs["schema"] = EVALUATION_HDF5_SCHEMA
        file.attrs["completed_sample_count"] = 1
        file.attrs["room_id"] = "kujiale_0020"
        file.attrs["research_only"] = True
        file.attrs["qualification_claim"] = False
        file.attrs["split"] = "test"
        file.attrs["sample_rate_hz"] = 16_000
        file.attrs["duration_seconds"] = 5.0
        file.attrs["source_count"] = 2
        file.attrs["output_frame_count"] = 75
        file.attrs["input_identity_json"] = json.dumps(
            {
                "dataset_index": str(tmp_path / "index.json"),
                "trajectory_bank": str(tmp_path / "trajectory.json"),
                "trajectory_bank_sha256": "1" * 64,
                "rir_plan": str(tmp_path / "rir.json"),
                "rir_plan_sha256": "2" * 64,
                "builder_producer_identity": builder_producer,
            }
        )
        file.attrs["builder_producer_identity_json"] = json.dumps(
            builder_producer
        )
        file.attrs["caption_table_json"] = '["cat meowing", "dog barking"]'
        file.create_dataset("mixture", shape=(1, 80_000, 2), dtype=np.float32)
        file.create_dataset("azimuth_deg", shape=(1, 2, 75), dtype=np.float32)
        file.create_dataset("caption_id", data=np.asarray([[0, 1]], np.uint8))
        file.create_dataset(
            "sample_id",
            data=np.asarray(["sample0"], dtype=object),
            dtype=string_type,
        )
    with pytest.raises(LegacyV4AudioError, match="episode_id"):
        EvaluationQueryBank(path)
    with h5py.File(path, "r+") as file:
        file.create_dataset(
            "episode_id",
            data=np.asarray(["episode0"], dtype=object),
            dtype=string_type,
        )
        file.attrs["room_id"] = ""
    with pytest.raises(LegacyV4AudioError, match="room_id"):
        EvaluationQueryBank(path)
    with h5py.File(path, "r+") as file:
        file.attrs["room_id"] = "kujiale_0020"
        file.attrs["builder_producer_identity_json"] = json.dumps(
            _builder_producer("b")
        )
    with pytest.raises(LegacyV4AudioError, match="input identity"):
        EvaluationQueryBank(path)


def test_room_comparison_requires_same_checkpoint_and_reports_deltas():
    result = compare_evaluation_reports(
        _evaluation_report("apartment_0000", mean=10.0, macro=11.0),
        _evaluation_report("kujiale_0020", mean=15.0, macro=17.0),
    )
    assert result["metrics"]["mean_absolute_error_deg"]["delta"] == 5.0
    assert result["baseline_room_id"] == "apartment_0000"
    assert result["training_room_id"] == "apartment_0000"
    with pytest.raises(ValueError, match="same checkpoint"):
        changed = _evaluation_report("kujiale_0020", mean=15.0, macro=17.0)
        changed["inputs"]["checkpoint"] = "/tmp/other.pt"
        compare_evaluation_reports(
            _evaluation_report("apartment_0000", mean=10.0, macro=11.0),
            changed,
        )


def test_room_comparison_rejects_incompatible_evaluation_context():
    baseline = _evaluation_report(
        "apartment_0000", regime="same_room_held_out"
    )
    candidate = _evaluation_report(
        "kujiale_0020", regime="cross_room_zero_shot"
    )
    invalid_regime = dict(candidate, evaluation_regime="same_room_held_out")
    with pytest.raises(ValueError, match="cross_room_zero_shot"):
        compare_evaluation_reports(baseline, invalid_regime)

    different_training_room = dict(candidate, training_room_id="other_training_room")
    with pytest.raises(ValueError, match="share one training room"):
        compare_evaluation_reports(baseline, different_training_room)

    same_room_candidate = dict(candidate, evaluation_room_id="apartment_0000")
    with pytest.raises(ValueError, match="cross_room_zero_shot"):
        compare_evaluation_reports(baseline, same_room_candidate)

    different_clap = dict(candidate)
    different_clap["inputs"] = {
        **candidate["inputs"],
        "clap_checkpoint": "/tmp/other-clap.pt",
    }
    with pytest.raises(ValueError, match="same CLAP checkpoint"):
        compare_evaluation_reports(baseline, different_clap)

    different_checkpoint_hash = dict(candidate)
    different_checkpoint_hash["inputs"] = {
        **candidate["inputs"],
        "checkpoint_sha256": "3" * 64,
    }
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        compare_evaluation_reports(baseline, different_checkpoint_hash)

    different_producer = dict(candidate, producer=_producer("b"))
    with pytest.raises(ValueError, match="producer identity"):
        compare_evaluation_reports(baseline, different_producer)

    different_training_hash = dict(candidate)
    different_training_hash["inputs"] = {
        **candidate["inputs"],
        "training_dataset_h5_sha256": "8" * 64,
    }
    with pytest.raises(ValueError, match="training HDF5 SHA256"):
        compare_evaluation_reports(baseline, different_training_hash)

    same_hdf5_path = copy.deepcopy(candidate)
    same_hdf5_path["inputs"]["dataset_h5"] = candidate["inputs"][
        "training_dataset_h5"
    ]
    with pytest.raises(ValueError, match="cross_room_zero_shot"):
        compare_evaluation_reports(baseline, same_hdf5_path)

    same_hdf5_content = copy.deepcopy(candidate)
    same_hdf5_content["inputs"]["dataset_h5_sha256"] = candidate["inputs"][
        "training_dataset_h5_sha256"
    ]
    with pytest.raises(ValueError, match="cross_room_zero_shot"):
        compare_evaluation_reports(baseline, same_hdf5_content)

    same_index_path = copy.deepcopy(candidate)
    same_index_path["inputs"]["dataset_index"] = candidate["inputs"][
        "training_dataset_index"
    ]
    with pytest.raises(ValueError, match="cross_room_zero_shot"):
        compare_evaluation_reports(baseline, same_index_path)

    same_index_content = copy.deepcopy(candidate)
    same_index_content["inputs"]["dataset_index_sha256"] = candidate["inputs"][
        "training_dataset_index_sha256"
    ]
    with pytest.raises(ValueError, match="cross_room_zero_shot"):
        compare_evaluation_reports(baseline, same_index_content)


@pytest.mark.parametrize("value", [True, np.nan, np.inf, -np.inf])
def test_room_comparison_rejects_boolean_or_nonfinite_overall(value):
    baseline = _evaluation_report("apartment_0000")
    candidate = _evaluation_report("kujiale_0020")
    candidate["overall"]["mean_absolute_error_deg"] = value
    with pytest.raises(ValueError, match="candidate evaluation report is invalid"):
        compare_evaluation_reports(baseline, candidate)


def test_room_comparison_rejects_incomplete_or_ambiguous_slices():
    baseline = _evaluation_report("apartment_0000")

    missing = _evaluation_report("kujiale_0020")
    missing["slices"].pop("caption")
    with pytest.raises(ValueError, match="lacks caption slice"):
        compare_evaluation_reports(baseline, missing)

    duplicate = _evaluation_report("kujiale_0020")
    duplicate["slices"]["source_motion"].append(
        copy.deepcopy(duplicate["slices"]["source_motion"][0])
    )
    with pytest.raises(ValueError, match="duplicated"):
        compare_evaluation_reports(baseline, duplicate)

    wrong_labels = _evaluation_report("kujiale_0020")
    wrong_labels["slices"]["motion_case"][0]["label"] = "invented_case"
    with pytest.raises(ValueError, match="labels differ"):
        compare_evaluation_reports(baseline, wrong_labels)

    incomplete = _evaluation_report("kujiale_0020")
    incomplete["slices"]["caption"][0]["query_count"] -= 1
    incomplete["slices"]["caption"][0]["frame_count"] -= 75
    with pytest.raises(ValueError, match="slice is incomplete"):
        compare_evaluation_reports(baseline, incomplete)


@pytest.mark.parametrize("value", [True, np.nan, np.inf, -np.inf, -0.1, 180.1])
def test_room_comparison_rejects_boolean_or_nonfinite_slice_metrics(value):
    baseline = _evaluation_report("apartment_0000")
    candidate = _evaluation_report("kujiale_0020")
    candidate["slices"]["target_region"][0][
        "mean_absolute_error_deg"
    ] = value
    with pytest.raises(ValueError, match="target_region slice row is invalid"):
        compare_evaluation_reports(baseline, candidate)


def test_room_comparison_rejects_nonnumeric_slice_count():
    baseline = _evaluation_report("apartment_0000")
    candidate = _evaluation_report("kujiale_0020")
    candidate["slices"]["caption"][0]["sample_count"] = "four"
    with pytest.raises(ValueError, match="caption slice row is invalid"):
        compare_evaluation_reports(baseline, candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean_absolute_error_deg", -0.1),
        ("median_absolute_error_deg", 180.1),
        ("p90_absolute_error_deg", -1.0),
        ("uniform_target_region_macro_mean_absolute_error_deg", 181.0),
        ("error_over_45deg_rate", -0.01),
        ("error_over_90deg_rate", 1.01),
    ],
)
def test_room_comparison_rejects_out_of_range_overall_metrics(field, value):
    baseline = _evaluation_report("apartment_0000")
    candidate = _evaluation_report("kujiale_0020")
    candidate["overall"][field] = value
    with pytest.raises(ValueError, match="candidate evaluation report is invalid"):
        compare_evaluation_reports(baseline, candidate)


def test_room_comparison_rejects_invalid_target_region_fractions():
    baseline = _evaluation_report("apartment_0000")

    missing_region = _evaluation_report("kujiale_0020")
    missing_region["overall"]["target_region_frame_fractions"].pop("left")
    with pytest.raises(ValueError, match="candidate evaluation report is invalid"):
        compare_evaluation_reports(baseline, missing_region)

    wrong_sum = _evaluation_report("kujiale_0020")
    wrong_sum["overall"]["target_region_frame_fractions"]["front"] = 0.5
    with pytest.raises(ValueError, match="candidate evaluation report is invalid"):
        compare_evaluation_reports(baseline, wrong_sum)


def test_room_comparison_rejects_fake_producer_and_claim_flags():
    baseline = _evaluation_report("apartment_0000")

    missing_code = _evaluation_report("kujiale_0020")
    missing_code["producer"]["code"].pop(
        next(iter(EVALUATION_PRODUCER_CODE_FILES))
    )
    with pytest.raises(ValueError, match="candidate evaluation report is invalid"):
        compare_evaluation_reports(baseline, missing_code)

    extra_runtime = _evaluation_report("kujiale_0020")
    extra_runtime["producer"]["runtime"]["invented_runtime"] = "fake"
    with pytest.raises(ValueError, match="candidate evaluation report is invalid"):
        compare_evaluation_reports(baseline, extra_runtime)

    qualification_claim = _evaluation_report("kujiale_0020")
    qualification_claim["qualification_claim"] = True
    with pytest.raises(ValueError, match="candidate evaluation report is invalid"):
        compare_evaluation_reports(baseline, qualification_claim)


def _write_training_context(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    training_index = tmp_path / "training_index.json"
    training_index.write_text(
        json.dumps(
            {
                "schema": TRAINING_INDEX_SCHEMA,
                "status": "pass",
                "room_id": "apartment_0000",
            }
        ),
        encoding="utf-8",
    )
    training_h5 = tmp_path / "training.h5"
    with h5py.File(training_h5, "w") as file:
        file.attrs["schema"] = HDF5_SCHEMA
        file.attrs["input_identity_json"] = json.dumps(
            {"dataset_index": str(training_index)}
        )
    clap = tmp_path / "clap.pt"
    clap.write_bytes(b"clap")
    checkpoint = {
        "run_identity": {
            "dataset_h5": str(training_h5),
            "clap_checkpoint": str(clap),
        }
    }
    return training_h5.resolve(), training_index.resolve(), clap.resolve(), checkpoint


def _cross_room_index(input_identity):
    return {
        "schema": EVALUATION_INDEX_SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "room_id": "kujiale_0020",
        "sample_count": 1,
        "split_sample_counts": {"test": 1},
        "source_slots": ["source1", "source2"],
        "input_identity": input_identity,
        "samples": [
            {
                "sample_id": "sample0",
                "episode_id": "episode0",
                "split": "test",
            }
        ],
    }


def test_evaluation_context_derives_same_and_cross_room_from_artifacts(tmp_path):
    training_h5, training_index, clap, checkpoint = _write_training_context(
        tmp_path
    )
    same_index = {
        "schema": TRAINING_INDEX_SCHEMA,
        "status": "pass",
        "room_id": "apartment_0000",
    }
    same = _resolve_evaluation_context(
        evaluation_h5=training_h5,
        evaluation_index_path=training_index,
        evaluation_index=same_index,
        bank_identity={
            "schema": HDF5_SCHEMA,
            "input_identity": {"dataset_index": str(training_index)},
        },
        checkpoint=checkpoint,
        clap_checkpoint=clap,
        declared_training_room_id="apartment_0000",
        declared_evaluation_room_id="apartment_0000",
    )
    assert same["evaluation_regime"] == "same_room_held_out"

    evaluation_h5 = tmp_path / "evaluation.h5"
    evaluation_h5.write_bytes(b"h5-placeholder")
    evaluation_index = tmp_path / "evaluation_index.json"
    evaluation_identity = {"dataset_index": str(evaluation_index.resolve())}
    evaluation_index_value = _cross_room_index(evaluation_identity)
    evaluation_index.write_text(
        json.dumps(evaluation_index_value),
        encoding="utf-8",
    )
    cross = _resolve_evaluation_context(
        evaluation_h5=evaluation_h5.resolve(),
        evaluation_index_path=evaluation_index.resolve(),
        evaluation_index=evaluation_index_value,
        bank_identity={
            "schema": EVALUATION_HDF5_SCHEMA,
            "room_id": "kujiale_0020",
            "input_identity": evaluation_identity,
        },
        checkpoint=checkpoint,
        clap_checkpoint=clap,
        declared_training_room_id=None,
        declared_evaluation_room_id=None,
    )
    assert same["evaluation_dataset_identity_mode"] == (
        "legacy_training_identity_compatibility"
    )
    assert same["dataset_index_identity_exact_match"] is False
    assert cross["training_room_id"] == "apartment_0000"
    assert cross["evaluation_regime"] == "cross_room_zero_shot"
    assert cross["evaluation_dataset_identity_mode"] == (
        "evaluation_hdf5_exact_index_identity"
    )
    assert cross["dataset_index_identity_exact_match"] is True


def test_evaluation_context_rejects_cross_room_identity_and_header_tamper(
    tmp_path,
):
    training_h5, _, clap, checkpoint = _write_training_context(tmp_path)
    evaluation_h5 = tmp_path / "evaluation.h5"
    evaluation_h5.write_bytes(b"h5-placeholder")
    evaluation_index = tmp_path / "evaluation_index.json"
    hdf5_identity = {
        "dataset_index": str(evaluation_index.resolve()),
        "trajectory_bank": "/tmp/bound-trajectory.json",
    }
    tampered_index = _cross_room_index(
        {
            **hdf5_identity,
            "trajectory_bank": "/tmp/tampered-trajectory.json",
        }
    )
    evaluation_index.write_text(
        json.dumps(tampered_index),
        encoding="utf-8",
    )
    with pytest.raises(LegacyV4AudioError, match="input_identity"):
        _resolve_evaluation_context(
            evaluation_h5=evaluation_h5.resolve(),
            evaluation_index_path=evaluation_index.resolve(),
            evaluation_index=tampered_index,
            bank_identity={
                "schema": EVALUATION_HDF5_SCHEMA,
                "room_id": "kujiale_0020",
                "input_identity": hdf5_identity,
            },
            checkpoint=checkpoint,
            clap_checkpoint=clap,
            declared_training_room_id=None,
            declared_evaluation_room_id=None,
        )

    exact_index = {**tampered_index, "input_identity": hdf5_identity}
    evaluation_index.write_text(json.dumps(exact_index), encoding="utf-8")
    with pytest.raises(LegacyV4AudioError, match="room_id"):
        _resolve_evaluation_context(
            evaluation_h5=evaluation_h5.resolve(),
            evaluation_index_path=evaluation_index.resolve(),
            evaluation_index=exact_index,
            bank_identity={
                "schema": EVALUATION_HDF5_SCHEMA,
                "room_id": "tampered_room_header",
                "input_identity": hdf5_identity,
            },
            checkpoint=checkpoint,
            clap_checkpoint=clap,
            declared_training_room_id=None,
            declared_evaluation_room_id=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("research_only", False),
        ("qualification_claim", True),
        ("sample_count", 2),
        ("split_sample_counts", {"test": 2}),
        ("source_slots", ["source2", "source1"]),
    ],
)
def test_evaluation_index_header_rejects_tamper(field, value):
    index = _cross_room_index({"dataset_index": "/tmp/evaluation-index.json"})
    index[field] = value
    with pytest.raises(LegacyV4AudioError, match="header is invalid"):
        _validate_evaluation_index_header(index)


def test_split_sample_episode_validation_filters_full_bank_in_exact_order():
    samples = {
        "test0": {"episode_id": "episode0"},
        "test1": {"episode_id": "episode1"},
    }
    queries = [
        QueryRef(1, 0, "test0", "cat meowing", np.zeros(75)),
        QueryRef(1, 1, "test0", "dog barking", np.zeros(75)),
        QueryRef(2, 0, "test1", "human speech", np.zeros(75)),
        QueryRef(2, 1, "test1", "cat meowing", np.zeros(75)),
    ]
    result = _validated_split_sample_episode_ids(
        samples=samples,
        queries=queries,
        bank_sample_ids=("train0", "test0", "test1"),
        bank_episode_ids=("train_episode", "episode0", "episode1"),
        bank_schema=HDF5_SCHEMA,
    )
    assert result == {"test0": "episode0", "test1": "episode1"}

    reversed_samples = {
        "test1": samples["test1"],
        "test0": samples["test0"],
    }
    with pytest.raises(LegacyV4AudioError, match="query sample/source order"):
        _validated_split_sample_episode_ids(
            samples=reversed_samples,
            queries=queries,
            bank_sample_ids=("train0", "test0", "test1"),
            bank_episode_ids=("train_episode", "episode0", "episode1"),
            bank_schema=HDF5_SCHEMA,
        )

    with pytest.raises(LegacyV4AudioError, match="sample/episode order"):
        _validated_split_sample_episode_ids(
            samples=samples,
            queries=queries,
            bank_sample_ids=("train0", "test0", "test1"),
            bank_episode_ids=("train_episode", "tampered", "episode1"),
            bank_schema=HDF5_SCHEMA,
        )


def _sample_metadata_fixture(tmp_path: Path):
    trajectory_path = tmp_path / "trajectory_bank.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema": "avengine_room_trajectory_bank_v2",
                "episodes": [
                    {
                        "episode_id": "episode0",
                        "motion_case": "both_moving",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    samples = {
        "sample0": {
            "sample_id": "sample0",
            "episode_id": "episode0",
            "split": "test",
            "motion_case": "both_moving",
            "both_sources_active": True,
            "source_classes": {
                "source1": "cat meowing",
                "source2": "dog barking",
            },
        }
    }
    queries = [
        QueryRef(
            sample_index=0,
            source_index=0,
            sample_id="sample0",
            caption="cat meowing",
            azimuth_deg=np.zeros(75, dtype=np.float32),
        ),
        QueryRef(
            sample_index=0,
            source_index=1,
            sample_id="sample0",
            caption="dog barking",
            azimuth_deg=np.zeros(75, dtype=np.float32),
        ),
    ]
    bank_identity = {
        "schema": EVALUATION_HDF5_SCHEMA,
        "input_identity": {
            "trajectory_bank": str(trajectory_path.resolve()),
            "trajectory_bank_sha256": _sha256_file(trajectory_path),
        },
    }
    return trajectory_path, samples, queries, bank_identity


def test_sample_slice_metadata_is_derived_from_bound_hdf5_and_trajectory(
    tmp_path,
):
    _, samples, queries, bank_identity = _sample_metadata_fixture(tmp_path)
    metadata, trajectory_identity = _validated_sample_metadata(
        samples=samples,
        queries=queries,
        sample_episode_ids={"sample0": "episode0"},
        bank_identity=bank_identity,
    )
    assert metadata["sample0"]["motion_case"] == "both_moving"
    assert metadata["sample0"]["captions_by_source_slot"] == {
        "source1": "cat meowing",
        "source2": "dog barking",
    }
    assert metadata["sample0"]["ordered_source_pair"] == (
        "cat meowing|dog barking"
    )
    assert trajectory_identity["trajectory_bank_sha256"] == (
        bank_identity["input_identity"]["trajectory_bank_sha256"]
    )


def test_sample_slice_metadata_rejects_index_caption_order_and_motion_tamper(
    tmp_path,
):
    _, samples, queries, bank_identity = _sample_metadata_fixture(tmp_path)

    swapped_index = copy.deepcopy(samples)
    swapped_index["sample0"]["source_classes"] = {
        "source1": "dog barking",
        "source2": "cat meowing",
    }
    with pytest.raises(LegacyV4AudioError, match="HDF5 captions"):
        _validated_sample_metadata(
            samples=swapped_index,
            queries=queries,
            sample_episode_ids={"sample0": "episode0"},
            bank_identity=bank_identity,
        )

    swapped_queries = [
        QueryRef(
            sample_index=query.sample_index,
            source_index=query.source_index,
            sample_id=query.sample_id,
            caption=queries[1 - query.source_index].caption,
            azimuth_deg=query.azimuth_deg,
        )
        for query in queries
    ]
    with pytest.raises(LegacyV4AudioError, match="HDF5 captions"):
        _validated_sample_metadata(
            samples=samples,
            queries=swapped_queries,
            sample_episode_ids={"sample0": "episode0"},
            bank_identity=bank_identity,
        )

    tampered_motion = copy.deepcopy(samples)
    tampered_motion["sample0"]["motion_case"] = "static_static"
    with pytest.raises(LegacyV4AudioError, match="binding is invalid"):
        _validated_sample_metadata(
            samples=tampered_motion,
            queries=queries,
            sample_episode_ids={"sample0": "episode0"},
            bank_identity=bank_identity,
        )


def test_sample_slice_metadata_rejects_query_structure_and_trajectory_tamper(
    tmp_path,
):
    trajectory_path, samples, queries, bank_identity = _sample_metadata_fixture(
        tmp_path
    )

    with pytest.raises(LegacyV4AudioError, match="binding is invalid"):
        _validated_sample_metadata(
            samples=samples,
            queries=queries[:1],
            sample_episode_ids={"sample0": "episode0"},
            bank_identity=bank_identity,
        )

    duplicate_slot = [
        queries[0],
        QueryRef(
            sample_index=0,
            source_index=0,
            sample_id="sample0",
            caption="dog barking",
            azimuth_deg=np.zeros(75, dtype=np.float32),
        ),
    ]
    with pytest.raises(LegacyV4AudioError, match="source slot is duplicated"):
        _validated_sample_metadata(
            samples=samples,
            queries=duplicate_slot,
            sample_episode_ids={"sample0": "episode0"},
            bank_identity=bank_identity,
        )

    trajectory_path.write_text(
        json.dumps(
            {
                "schema": "avengine_room_trajectory_bank_v2",
                "episodes": [
                    {
                        "episode_id": "episode0",
                        "motion_case": "static_static",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LegacyV4AudioError, match="SHA256 differs"):
        _validated_sample_metadata(
            samples=samples,
            queries=queries,
            sample_episode_ids={"sample0": "episode0"},
            bank_identity=bank_identity,
        )


@pytest.mark.parametrize(
    ("training_declaration", "evaluation_declaration", "message"),
    [
        ("other_room", "kujiale_0020", "--training-room-id"),
        ("apartment_0000", "other_room", "--evaluation-room-id"),
    ],
)
def test_evaluation_context_rejects_cli_room_declaration_mismatch(
    tmp_path,
    training_declaration,
    evaluation_declaration,
    message,
):
    training_h5, _, clap, checkpoint = _write_training_context(tmp_path)
    evaluation_h5 = tmp_path / "evaluation.h5"
    evaluation_h5.write_bytes(b"h5-placeholder")
    evaluation_index = tmp_path / "evaluation_index.json"
    evaluation_index.write_text("{}", encoding="utf-8")
    with pytest.raises(LegacyV4AudioError, match=message):
        _resolve_evaluation_context(
            evaluation_h5=evaluation_h5.resolve(),
            evaluation_index_path=evaluation_index.resolve(),
            evaluation_index={"status": "pass", "room_id": "kujiale_0020"},
            bank_identity={
                "schema": EVALUATION_HDF5_SCHEMA,
                "room_id": "kujiale_0020",
                "input_identity": {"dataset_index": str(evaluation_index)},
            },
            checkpoint=checkpoint,
            clap_checkpoint=clap,
            declared_training_room_id=training_declaration,
            declared_evaluation_room_id=evaluation_declaration,
        )


def test_evaluation_context_rejects_checkpoint_clap_mismatch(tmp_path):
    training_h5, _, _, checkpoint = _write_training_context(tmp_path)
    evaluation_h5 = tmp_path / "evaluation.h5"
    evaluation_h5.write_bytes(b"h5-placeholder")
    evaluation_index = tmp_path / "evaluation_index.json"
    evaluation_index.write_text("{}", encoding="utf-8")
    other_clap = tmp_path / "other-clap.pt"
    other_clap.write_bytes(b"other")
    with pytest.raises(LegacyV4AudioError, match="run_identity"):
        _resolve_evaluation_context(
            evaluation_h5=evaluation_h5.resolve(),
            evaluation_index_path=evaluation_index.resolve(),
            evaluation_index={"status": "pass", "room_id": "kujiale_0020"},
            bank_identity={
                "schema": EVALUATION_HDF5_SCHEMA,
                "room_id": "kujiale_0020",
                "input_identity": {"dataset_index": str(evaluation_index)},
            },
            checkpoint=checkpoint,
            clap_checkpoint=other_clap.resolve(),
            declared_training_room_id=None,
            declared_evaluation_room_id=None,
        )


def test_pair_publication_rolls_back_h5_if_index_publish_fails(tmp_path):
    staged_h5 = tmp_path / ".dataset.h5.incomplete"
    staged_index = tmp_path / ".index.json.staging"
    output_h5 = tmp_path / "dataset.h5"
    output_index = tmp_path / "index.json"
    staged_h5.write_bytes(b"h5")
    staged_index.write_bytes(b"index-new")
    output_index.write_bytes(b"index-existing")
    with pytest.raises(FileExistsError):
        _publish_pair_no_replace(
            staged_h5=staged_h5,
            output_h5=output_h5,
            staged_index=staged_index,
            output_index=output_index,
        )
    assert not output_h5.exists()
    assert output_index.read_bytes() == b"index-existing"


def test_directory_publication_is_atomic_and_no_replace(tmp_path):
    staging = tmp_path / ".report.staging"
    output = tmp_path / "report"
    staging.mkdir()
    (staging / "evaluation.json").write_text("{}", encoding="utf-8")
    assert atomic_publish_directory(staging, output) == output
    assert not staging.exists()
    assert (output / "evaluation.json").is_file()

    second_staging = tmp_path / ".report.second.staging"
    second_staging.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        atomic_publish_directory(second_staging, output)
    assert second_staging.is_dir()


def test_builder_producer_identity_binds_result_changing_code_and_runtime():
    identity = _builder_producer_identity(_load_h5py())
    assert identity["schema"] == BUILDER_PRODUCER_SCHEMA
    assert set(identity["code"]) == set(BUILDER_CODE_FILES)
    assert all(
        len(record["sha256"]) == 64
        for record in identity["code"].values()
    )
    assert identity["runtime"]["numpy_version"] == np.__version__
    assert identity["runtime"]["h5py_version"]
    assert identity["runtime"]["soundfile_version"]
    assert identity["runtime"]["libsndfile_version"]


def _native_delivery(*, files):
    return {
        "schema": NATIVE_DELIVERY_SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "both_sources_active": True,
        "mixture_is_exact_persisted_source_stem_sum": True,
        "source_slots": ["source1", "source2"],
        "sample_count": 1,
        "input_closure": {},
        "output_closure": {
            "schema": NATIVE_OUTPUT_CLOSURE_SCHEMA,
            "status": "pass",
            "sample_count": 1,
            "wave_file_count": 3,
            "wave_sidecar_count": 3,
            "audio_artifact_file_count": 6,
            "audio_artifact_hashes_bound_by": "samples.json",
            "files": files,
        },
    }


def test_native_output_validation_accepts_hash_bound_schema(tmp_path):
    trajectory = tmp_path / "trajectory_bank.json"
    rir = tmp_path / "rir_job_plan.json"
    trajectory.write_text('{"trajectory":true}', encoding="utf-8")
    rir.write_text('{"rir":true}', encoding="utf-8")
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    artifact_names = (
        "mixture.wav",
        "mixture.wav.json",
        "source1.wav",
        "source1.wav.json",
        "source2.wav",
        "source2.wav.json",
    )
    for name in artifact_names:
        (audio_root / name).write_bytes(name.encode("utf-8"))
    input_closure = {
        "schema": "avengine_room_evaluation_binaural_input_closure_v1",
        "status": "pass",
        "files": {
            "trajectory_bank.json": {"sha256": _sha256_file(trajectory)},
            "rir_job_plan.json": {"sha256": _sha256_file(rir)},
        },
    }
    sample = {
        "sample_id": "episode0__v00",
        "episode_id": "episode0",
        "ordinal": 0,
        "split": "test",
        "both_sources_active": True,
        "mixture_is_exact_persisted_source_stem_sum": True,
        "source_classes": {
            "source1": "cat meowing",
            "source2": "dog barking",
        },
        "audio_path": "mixture.wav",
        "audio_sidecar_path": "mixture.wav.json",
        "audio_sha256": _sha256_file(audio_root / "mixture.wav"),
        "audio_sidecar_sha256": _sha256_file(
            audio_root / "mixture.wav.json"
        ),
        "source_stems": {
            slot: {
                "audio_path": f"{slot}.wav",
                "audio_sidecar_path": f"{slot}.wav.json",
                "audio_sha256": _sha256_file(audio_root / f"{slot}.wav"),
                "sidecar_sha256": _sha256_file(
                    audio_root / f"{slot}.wav.json"
                ),
            }
            for slot in ("source1", "source2")
        },
    }
    documents = {
        "input_closure.json": input_closure,
        "samples.json": {
            "schema": "avengine_room_evaluation_binaural_samples_v1",
            "status": "pass",
            "sample_count": 1,
            "input_closure": input_closure,
            "samples": [sample],
        },
        "dry_audio_classes.json": {"status": "pass"},
        "timing.json": {"status": "pass"},
    }
    output_files = {}
    for name, value in documents.items():
        path = audio_root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        output_files[name] = {
            "path": name,
            "byte_size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    delivery = _native_delivery(files=output_files)
    delivery["input_closure"] = input_closure
    (audio_root / "delivery.json").write_text(
        json.dumps(delivery), encoding="utf-8"
    )
    identity, samples, audio_paths = _validate_native_output(
        audio_root.resolve(),
        trajectory_path=trajectory.resolve(),
        rir_path=rir.resolve(),
    )
    assert identity["native_output_closure"]["wave_file_count"] == 3
    assert samples[0]["sample_id"] == "episode0__v00"
    assert audio_paths["episode0__v00"] == (audio_root / "mixture.wav").resolve()


def test_native_output_validation_rejects_missing_and_tampered_closure(tmp_path):
    trajectory = tmp_path / "trajectory_bank.json"
    rir = tmp_path / "rir_job_plan.json"
    trajectory.write_text("{}", encoding="utf-8")
    rir.write_text("{}", encoding="utf-8")
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    delivery = _native_delivery(files={})
    delivery.pop("output_closure")
    (audio_root / "delivery.json").write_text(
        json.dumps(delivery), encoding="utf-8"
    )
    with pytest.raises(LegacyV4AudioError, match="output_closure"):
        _validate_native_output(
            audio_root.resolve(),
            trajectory_path=trajectory.resolve(),
            rir_path=rir.resolve(),
        )

    files = {}
    for name in (
        "input_closure.json",
        "samples.json",
        "dry_audio_classes.json",
        "timing.json",
    ):
        path = audio_root / name
        path.write_text("{}", encoding="utf-8")
        files[name] = {
            "path": name,
            "byte_size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    (audio_root / "delivery.json").write_text(
        json.dumps(_native_delivery(files=files)), encoding="utf-8"
    )
    (audio_root / "samples.json").write_text("[]", encoding="utf-8")
    with pytest.raises(LegacyV4AudioError, match="SHA256 differs"):
        _validate_native_output(
            audio_root.resolve(),
            trajectory_path=trajectory.resolve(),
            rir_path=rir.resolve(),
        )
