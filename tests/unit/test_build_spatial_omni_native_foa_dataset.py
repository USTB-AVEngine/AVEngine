from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.m4.audio import write_float32_wav
from avengine.m4.spatial import rlr_foa_wav_metadata
from tools.qa import build_spatial_omni_native_foa_dataset as builder


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _native_metadata(**overrides) -> dict:
    metadata = {
        **rlr_foa_wav_metadata(),
        "audio_role": "dynamic_wet_mixture",
        "source_ids": ["source1", "source2"],
        "summation_order": ["source1", "source2"],
        "hrtf_used": False,
        "resampling": "not_applied",
        "amplitude_normalization": "not_applied",
        "limiter": "not_applied",
    }
    metadata.update(overrides)
    return metadata


def _write_point_audio(
    audio_root: Path,
    point_id: str,
    *,
    channels: int = 4,
    sample_rate: int = 16_000,
    sample_count: int = 80_000,
    metadata: dict | None = None,
) -> Path:
    path = audio_root / point_id / "audio/foa/mixture.wav"
    samples = np.zeros((channels, sample_count), dtype=np.float32)
    write_float32_wav(
        path,
        samples,
        sample_rate,
        metadata=_native_metadata() if metadata is None else metadata,
    )
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    clips = tmp_path / "clips"
    audio = tmp_path / "audio"
    clips.mkdir()
    specs = {
        "p1": {"point_id": "p1", "twin_of": None, "offscreen_candidate": False},
        "p1t": {"point_id": "p1t", "twin_of": "p1", "offscreen_candidate": False},
        "p2": {"point_id": "p2", "twin_of": None, "offscreen_candidate": False},
        "p3": {"point_id": "p3", "twin_of": None, "offscreen_candidate": False},
        "p4": {"point_id": "p4", "twin_of": None, "offscreen_candidate": True},
    }
    for point_id, spec in specs.items():
        _write_json(inputs / point_id / "spec.json", spec)
        _write_point_audio(audio, point_id)
    for point_id in ("p1", "p1t", "p2", "p4"):
        (clips / f"{point_id}.mp4").write_bytes(b"fixture clip")
    questions = {
        "questions": [
            {
                "question_id": "q-binary",
                "point_id": "p1",
                "type_id": "QV2-T4-SIDE",
                "question_en": "Was the first sound left or right?",
                "options": ["left", "right"],
                "answer": "left",
                "modality_expectation": "audio_sufficient_control",
            },
            {
                "question_id": "q-twin",
                "point_id": "p1t",
                "type_id": "QV2-T4-SIDE",
                "question_en": "Was the twin sound left or right?",
                "options": ["left", "right"],
                "answer": "right",
                "modality_expectation": "audio_sufficient_control",
            },
            {
                "question_id": "q-numeric",
                "point_id": "p2",
                "type_id": "QV2-T5-DOA",
                "format": "numeric_banded",
                "question_en": "What was the source azimuth?",
                "answer_numeric": -22.5,
                "unit": "degrees",
                "scoring_bands": [15.0, 30.0],
                "modality_expectation": "audio_required",
            },
            {
                "question_id": "q-refusal",
                "point_id": "p3",
                "question_en": "What color was the unseen shirt?",
                "options": ["blue", "cannot answer from audio"],
                "answer": "cannot answer from audio",
                "type_id": "QV2-TR-REFUSAL",
                "answer_arity": "nonbinary",
                "refusal_gold": True,
                "provided_modalities": ["audio"],
                "withheld_modalities": ["video"],
                "video_path": "/must/not/leak.mp4",
                "modality_expectation": "must_refuse_audio_only_visual_attribute",
            },
            {
                "question_id": "q-offscreen",
                "point_id": "p4",
                "type_id": "QV2-T1-OFFSCREEN",
                "question_en": "Which source sounded off screen?",
                "options": ["source1", "source2"],
                "answer": "source2",
                "modality_expectation": "dual_required",
            },
        ]
    }
    questions_path = tmp_path / "questions.json"
    _write_json(questions_path, questions)
    return {
        "questions": questions_path,
        "inputs": inputs,
        "clips": clips,
        "audio": audio,
        "output": tmp_path / "dataset",
    }


def _read_rows(output: Path) -> list[dict]:
    rows = []
    for split in ("train", "valid"):
        rows.extend(
            json.loads(line)
            for line in (output / f"qa/{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    return rows


def test_native_foa_dataset_preserves_twin_split_and_question_types(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = builder.build_dataset(
        questions_path=fixture["questions"],
        inputs_root=fixture["inputs"],
        clips_dir=fixture["clips"],
        audio_root=fixture["audio"],
        output=fixture["output"],
        valid_fraction=0.5,
        seed=7,
        generated_at="2026-08-24T00:00:00+00:00",
    )

    rows = _read_rows(output)
    by_id = {row["qa_id"]: row for row in rows}
    assert by_id["q-binary"]["split"] == by_id["q-twin"]["split"]
    assert by_id["q-binary"]["answer_format"] == "mcq"
    numeric = by_id["q-numeric"]
    assert numeric["answer_format"] == "numeric_banded"
    assert numeric["answer_numeric"] == -22.5
    assert numeric["unit"] == "degrees"
    assert numeric["scoring_bands"] == [15.0, 30.0]
    refusal = by_id["q-refusal"]
    assert refusal["is_refusal"] is True
    assert refusal["input_modalities"] == ["audio"]
    assert refusal["modality_mask"] == {"audio": True, "video": False}
    assert refusal["refusal_gold"] is True
    assert refusal["provided_modalities"] == ["audio"]
    assert refusal["withheld_modalities"] == ["video"]
    assert "video_path" not in refusal
    assert "/must/not/leak.mp4" not in json.dumps(refusal)
    for question_id in ("q-binary", "q-twin", "q-numeric", "q-offscreen"):
        assert Path(by_id[question_id]["video_path"]).is_file()

    for row in rows:
        audio_path = Path(row["audio_path"])
        assert audio_path.name == "mixture.wav"
        assert audio_path.is_file()
        assert audio_path.with_suffix(".wav.json").is_file()
        assert row["audio_format"]["format_id"] == "rlr_foa_acn_n3d_world_v1"
        assert row["audio_format"]["raw_channel_order"] == ["W", "Y", "Z", "X"]

    provenance = json.loads(
        (output / "DATASET_PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert provenance["status"] == "research_only"
    assert provenance["research_only"] is True
    assert provenance["episode_counted"] is False
    assert provenance["formal_dataset_count"] == 0
    assert provenance["audio_provenance"]["source"] == "native_rlr_foa"
    assert "foaish" in provenance["audio_provenance"]["fallbacks"]
    assert provenance["trainer_compatibility"]["status"] == "blocked"
    assert not list(tmp_path.glob(".dataset.staging-*"))


@pytest.mark.parametrize(
    ("channels", "sample_rate", "sample_count", "match"),
    [
        (2, 16_000, 80_000, r"shape \[4,80000\]"),
        (4, 48_000, 80_000, "16000 Hz"),
        (4, 16_000, 79_999, r"shape \[4,80000\]"),
    ],
)
def test_native_foa_rejects_wrong_wave_layout(
    tmp_path: Path,
    channels: int,
    sample_rate: int,
    sample_count: int,
    match: str,
) -> None:
    root = tmp_path / "point"
    _write_point_audio(
        tmp_path,
        "point",
        channels=channels,
        sample_rate=sample_rate,
        sample_count=sample_count,
    )
    with pytest.raises(builder.SpatialOmniDatasetError, match=match):
        builder.validate_native_foa_point(root, point_id="point")


def test_native_foa_rejects_wrong_spatial_format_and_foaish_fallback(
    tmp_path: Path,
) -> None:
    wrong = _native_metadata()
    wrong["spatial_format"] = dict(wrong["spatial_format"])
    wrong["spatial_format"]["format_id"] = "foaish_mid_side_v0"
    _write_point_audio(tmp_path, "wrong", metadata=wrong)
    with pytest.raises(builder.SpatialOmniDatasetError, match="not native RLR FOA"):
        builder.validate_native_foa_point(tmp_path / "wrong", point_id="wrong")

    foaish = tmp_path / "foaish" / "audio/foaish"
    foaish.mkdir(parents=True)
    (foaish / "mixture.wav").write_bytes(b"pseudo")
    with pytest.raises(builder.SpatialOmniDatasetError, match="fallbacks are forbidden"):
        builder.validate_native_foa_point(tmp_path / "foaish", point_id="foaish")


def test_real_refusal_shape_must_have_consistent_modality_declarations() -> None:
    row = {
        "type_id": "QV2-TR-REFUSAL",
        "refusal_gold": True,
        "provided_modalities": ["audio"],
        "withheld_modalities": ["video"],
    }
    assert builder._is_refusal(row) is True

    inconsistent = dict(row, refusal_gold=False)
    with pytest.raises(
        builder.SpatialOmniDatasetError,
        match="modality declarations disagree.*refusal_gold",
    ):
        builder._is_refusal(inconsistent)

    with pytest.raises(builder.SpatialOmniDatasetError, match="must use QV2-TR-REFUSAL"):
        builder._is_refusal({"is_refusal": True})


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"hrtf_used": True}, "hrtf_used"),
        ({"resampling": "applied"}, "resampling"),
        ({"amplitude_normalization": "applied"}, "amplitude_normalization"),
        ({"limiter": "applied"}, "limiter"),
    ],
)
def test_native_foa_rejects_processing_metadata(
    tmp_path: Path, override: dict, match: str
) -> None:
    _write_point_audio(tmp_path, "point", metadata=_native_metadata(**override))
    with pytest.raises(builder.SpatialOmniDatasetError, match=match):
        builder.validate_native_foa_point(tmp_path / "point", point_id="point")


def test_native_foa_rejects_mixture_source_order_drift(tmp_path: Path) -> None:
    metadata = _native_metadata(summation_order=["source2", "source1"])
    _write_point_audio(tmp_path, "point", metadata=metadata)
    with pytest.raises(builder.SpatialOmniDatasetError, match="summation_order"):
        builder.validate_native_foa_point(tmp_path / "point", point_id="point")


def test_non_refusal_offscreen_row_requires_clip(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["clips"] / "p4.mp4").unlink()
    with pytest.raises(builder.SpatialOmniDatasetError, match="p4 lacks clip"):
        builder.build_dataset(
            questions_path=fixture["questions"],
            inputs_root=fixture["inputs"],
            clips_dir=fixture["clips"],
            audio_root=fixture["audio"],
            output=fixture["output"],
            valid_fraction=0.5,
        )
    assert not fixture["output"].exists()


def test_output_is_atomic_and_refuses_to_clobber(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["output"].mkdir()
    sentinel = fixture["output"] / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(builder.SpatialOmniDatasetError, match="refusing to clobber"):
        builder.build_dataset(
            questions_path=fixture["questions"],
            inputs_root=fixture["inputs"],
            clips_dir=fixture["clips"],
            audio_root=fixture["audio"],
            output=fixture["output"],
            valid_fraction=0.5,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".dataset.staging-*"))
