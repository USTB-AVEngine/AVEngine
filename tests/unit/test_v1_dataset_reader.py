"""Contracts for the portable V1 QA dataset reader.

These are hermetic: they build a small export root by hand, so they prove the
reader's own boundary. They do not stand in for reading a real native delivery,
which is exercised separately against retained media.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from avengine.dataset import qa_dataset_reader as reader_module
from avengine.dataset.qa_dataset_reader import (
    DATASET_INDEX_SCHEMA,
    PRIVATE_INDEX_SCHEMA,
    QaDatasetReadError,
    open_qa_dataset,
    probe_media,
)


pytestmark = pytest.mark.fast_unit


BINAURAL_DECLARATION = {
    "layout_type": "binaural",
    "layout_id": "rlr_binaural_lr_v1",
    "channel_count": 2,
    "channel_labels": ["left", "right"],
    "channel_order": "not_applicable",
    "normalization": "not_applicable",
    "coordinate_frame": "listener_local",
    "sample_rate_hz": 16000,
}
FOA_DECLARATION = {
    "layout_type": "ambisonics",
    "layout_id": "rlr_foa_acn_n3d_world_v1",
    "channel_count": 4,
    "channel_labels": ["W", "Y", "Z", "X"],
    "channel_order": "ACN",
    "normalization": "N3D",
    "coordinate_frame": "avengine_world",
    "sample_rate_hz": 16000,
}


def _write_wav(
    path: Path, *, channels: int, frames: int, rate: int = 16000, marker: int = 0
) -> None:
    payload = struct.pack("<f", float(marker)) * channels * frames
    fmt = struct.pack("<HHIIHH", 3, channels, rate, rate * channels * 4, channels * 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def _write_mp4(path: Path, *, seconds: float, marker: bytes = b"\x00") -> None:
    ftyp = struct.pack(">I", 16) + b"ftypisom" + b"isom"
    mvhd_body = (
        b"\x00\x00\x00\x00"  # version 0 + flags
        + struct.pack(">I", 0)  # creation
        + struct.pack(">I", 0)  # modification
        + struct.pack(">I", 1000)  # timescale
        + struct.pack(">I", int(seconds * 1000))  # duration
    )
    mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
    moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ftyp + moov + marker)


def _question_set(episode_id: str, *, truth: str) -> dict:
    other = "green" if truth == "blue" else "blue"
    return {
        "schema": "avengine_qa_unified_question_v1_set",
        "episode_id": episode_id,
        "items": [
            {
                "schema": "avengine_qa_unified_question_v1",
                "status": "pass",
                "qa_id": "QA-02",
                "question_id": f"qa_02__{episode_id}__target_source1",
                "model_input": {
                    "mcq": {
                        "question_en": "Who made the sound?",
                        "question_zh": "谁发声了？",
                        "options": [
                            {"option": "A", "label_en": "blue top", "label_zh": "蓝色上衣"},
                            {"option": "B", "label_en": "green top", "label_zh": "绿色上衣"},
                        ],
                    },
                    "open": {
                        "question_en": "Who made the sound?",
                        "question_zh": "谁发声了？",
                    },
                },
                "form_status": {"mcq": {"status": "pass"}, "open": {"status": "pass"}},
                "forms": {
                    "mcq": {
                        "question_en": "Who made the sound?",
                        "answer_type": "choice",
                        "options": [
                            {"value": "blue", "label_en": "blue top", "label_zh": "蓝色上衣",
                             "allow_value": False},
                            {"value": "green", "label_en": "green top", "label_zh": "绿色上衣",
                             "allow_value": False},
                        ],
                        "gold": {
                            "correct_index": 0 if truth == "blue" else 1,
                            "value": truth,
                        },
                    },
                    "open": {
                        "question_en": "Who made the sound?",
                        "answer_type": "closed_set",
                        "truth": truth,
                        "classes": {
                            "blue": ["blue top", "蓝色上衣"],
                            "green": ["green top", "绿色上衣"],
                        },
                    },
                },
                "truth": {"value": truth, "answer_type": "closed_set"},
            },
            {
                "schema": "avengine_qa_unified_question_v1",
                "status": "deferred",
                "qa_id": "QA-06",
                "question_id": f"qa_06__{episode_id}__deferred",
                "forms": {},
            },
        ],
        "angle_followups": [
            {
                "schema": "avengine_qa_unified_question_v1",
                "status": "pass",
                "qa_id": "QA-02",
                "question_id": f"qa_02__{episode_id}__angle",
                "model_input": {
                    "open": {
                        "question_en": "At what angle?",
                        "question_zh": "在什么角度？",
                    }
                },
                "form_status": {"open": {"status": "pass"}},
                "forms": {
                    "open": {
                        "question_en": "At what angle?",
                        "answer_type": "angle_deg",
                        "truth": 30.0,
                        "convention": "right_positive",
                        "scoring_mode": "continuous",
                    }
                },
                "truth": {"value": 30.0, "answer_type": "angle_deg"},
            }
        ],
        "deferred": [{"qa_id": "QA-06", "code": "not_applicable"}],
    }


SAMPLES = (("sample_a", "episode_a", "blue", "mp3d"), ("sample_b", "episode_b", "green", "hm3d"))


@pytest.fixture
def export_root(tmp_path: Path) -> Path:
    root = tmp_path / "export"
    samples = []
    private_records = []
    for index, (sample_id, episode_id, truth, room) in enumerate(SAMPLES, start=1):
        audio = f"media/audio_{index}.wav"
        video = f"media/video_{index}.mp4"
        _write_wav(root / audio, channels=2, frames=160_000, marker=index)
        _write_mp4(root / video, seconds=10.0, marker=bytes([index]))
        questions_path = f"private/questions/{sample_id}.json"
        (root / questions_path).parent.mkdir(parents=True, exist_ok=True)
        (root / questions_path).write_text(
            json.dumps(_question_set(episode_id, truth=truth)), encoding="utf-8"
        )
        question_set = _question_set(episode_id, truth=truth)
        main = question_set["items"][0]
        angle = question_set["angle_followups"][0]
        samples.append(
            {
                "sample_id": sample_id,
                "record_kind": "episode",
                "room_family": room,
                "split": "train" if index == 1 else "test",
                "calibration": {
                    "projection": "pinhole",
                    "width_px": 1280,
                    "height_px": 720,
                    "status": "public",
                    "time": {"duration_seconds": 10.0, "frame_count": 150},
                },
                "media": {"video": {"preview": video}, "audio": {"binaural": {"path": audio}}},
                "questions": [
                    {
                        "question_id": f"question_{index:06d}",
                        "qa_id": "QA-02",
                        "kind": "main",
                        "forms": ["mcq", "open"],
                        "prompt": main["model_input"],
                        "required_modalities": None,
                    },
                    {
                        "question_id": f"question_{index + 100:06d}",
                        "qa_id": "QA-02",
                        "kind": "angle_followup",
                        "forms": ["open"],
                        "prompt": angle["model_input"],
                        "required_modalities": None,
                    },
                ],
            }
        )
        private_records.append(
            {
                "sample_id": sample_id,
                "record_kind": "episode",
                "group_id": None,
                "member_id": None,
                "world_id": f"engine_world_{room}_{index}",
                "world_key": f"world_{index:04d}",
                "episode_id": episode_id,
                "core_sample_id": None,
                "questions_path": questions_path,
                "question_id_map": {
                    f"question_{index:06d}": main["question_id"],
                    f"question_{index + 100:06d}": angle["question_id"],
                },
            }
        )
    index_doc = {
        "schema": DATASET_INDEX_SCHEMA,
        "status": "pass",
        "dataset_version": "v1",
        "requested_qa_ids": ["QA-02", "QA-06"],
        "audio_layouts": {"binaural": BINAURAL_DECLARATION},
        "observation_protocols": {"full_episode_av": {"audio": "full_episode"}},
        "counts": {"sample_count": 2, "world_count": 2},
        "splits": {"train": ["sample_a"], "test": ["sample_b"]},
        "samples": samples,
    }
    (root / "public").mkdir(parents=True, exist_ok=True)
    (root / "public/dataset_index.json").write_text(
        json.dumps(index_doc), encoding="utf-8"
    )
    (root / "private/gold_index.json").write_text(
        json.dumps(
            {
                "schema": PRIVATE_INDEX_SCHEMA,
                "status": "pass",
                "dataset_version": "v1",
                "core_bundle": None,
                "records": private_records,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_stdlib_probes_read_the_real_media_headers(export_root: Path) -> None:
    audio = probe_media(export_root / "media/audio_1.wav")
    assert audio["channel_count"] == 2
    assert audio["sample_rate_hz"] == 16_000
    assert audio["frame_count"] == 160_000
    assert audio["duration_s"] == pytest.approx(10.0)
    assert audio["encoding"] == "pcm_float"

    video = probe_media(export_root / "media/video_1.mp4")
    assert video["duration_s"] == pytest.approx(10.0)


def test_reader_opens_from_the_root_alone_and_lists_by_selection(export_root: Path) -> None:
    reader = open_qa_dataset(export_root)

    assert reader.list_samples() == ["sample_a", "sample_b"]
    assert reader.list_samples(split="train") == ["sample_a"]
    assert reader.list_samples(room_family="hm3d") == ["sample_b"]
    assert reader.list_samples(qa_id="QA-02") == ["sample_a", "sample_b"]
    # QA-06 exists only as a deferred requirement, so it is not a valid question.
    assert reader.list_samples(qa_id="QA-06") == []
    assert reader.audio_layouts() == ("binaural",)


def test_model_input_carries_no_gold_engine_identity_or_intervention(export_root: Path) -> None:
    reader = open_qa_dataset(export_root)
    payload = reader.model_input("sample_a", form="mcq", language="en")

    text = json.dumps(payload)
    for leaked in ("truth", "gold", "correct_index", "allow_value", "engine_world",
                   "questions_path", "episode_a"):
        assert leaked not in text
    assert "world_key" not in payload
    assert reader.private().grouping("sample_a")["world_key"] == "world_0001"
    assert payload["items"][0]["options"] == [
        {"option": "A", "label": "blue top"},
        {"option": "B", "label": "green top"},
    ]
    # The angle follow-up has no mcq form, so an mcq request must not list it.
    assert [item["kind"] for item in payload["items"]] == ["main"]

    open_payload = reader.model_input("sample_a", form="open", language="zh")
    assert [item["kind"] for item in open_payload["items"]] == ["main", "angle_followup"]
    assert open_payload["items"][0]["question"] == "谁发声了？"


def test_model_input_refuses_a_leaking_public_index(export_root: Path) -> None:
    document = json.loads((export_root / "public/dataset_index.json").read_text())
    document["samples"][0]["questions"][0]["prompt"]["mcq"]["gold"] = {"correct_index": 0}
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))

    with pytest.raises(QaDatasetReadError, match="private field 'gold' reached"):
        open_qa_dataset(export_root)


def test_media_selection_checks_the_declaration_against_the_real_file(export_root: Path) -> None:
    reader = open_qa_dataset(export_root)
    media = reader.media("sample_a", audio_layout="binaural")

    assert media["audio_relative_path"] == "media/audio_1.wav"
    assert media["audio_declaration"] == BINAURAL_DECLARATION
    assert media["audio_probe"]["channel_count"] == 2
    assert media["video_probe"]["duration_s"] == pytest.approx(10.0)

    # A declaration that disagrees with the delivered channel count fails closed.
    document = json.loads((export_root / "public/dataset_index.json").read_text())
    document["audio_layouts"]["binaural"]["channel_count"] = 4
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))
    with pytest.raises(QaDatasetReadError, match="channels but the declaration says"):
        open_qa_dataset(export_root).media("sample_a")


def test_an_undeclared_layout_is_refused_rather_than_assumed(export_root: Path) -> None:
    reader = open_qa_dataset(export_root)

    with pytest.raises(QaDatasetReadError, match="is not declared by this export"):
        reader.layout_declaration("ambisonics")
    with pytest.raises(QaDatasetReadError, match="is not declared by this export"):
        reader.media("sample_a", audio_layout="ambisonics")


def test_an_incomplete_ambisonics_declaration_is_refused(export_root: Path) -> None:
    document = json.loads((export_root / "public/dataset_index.json").read_text())
    partial = dict(FOA_DECLARATION)
    partial.pop("normalization")
    partial.pop("coordinate_frame")
    document["audio_layouts"]["ambisonics"] = partial
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))

    reader = open_qa_dataset(export_root)
    with pytest.raises(QaDatasetReadError, match="declaration is incomplete"):
        reader.layout_declaration("ambisonics")


def test_an_attached_layout_is_another_track_not_another_world(export_root: Path) -> None:
    _write_wav(export_root / "attachments/sample_a/ambisonics/mixture.wav",
               channels=4, frames=160_000)
    document = json.loads((export_root / "public/dataset_index.json").read_text())
    document["audio_layouts"]["ambisonics"] = FOA_DECLARATION
    document["samples"][0]["media"]["audio"]["ambisonics"] = {
        "path": "attachments/sample_a/ambisonics/mixture.wav",
        "attached_view_of": "binaural",
    }
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))

    reader = open_qa_dataset(export_root)
    assert reader.list_samples(audio_layout="ambisonics") == ["sample_a"]
    media = reader.media("sample_a", audio_layout="ambisonics")
    assert media["attached_view_of"] == "binaural"
    assert media["audio_declaration"]["normalization"] == "N3D"
    assert media["audio_declaration"]["coordinate_frame"] == "avengine_world"
    assert media["audio_probe"]["channel_count"] == 4
    # Same clock as the binaural track, and no extra sample, world or question.
    assert media["audio_probe"]["frame_count"] == 160_000
    assert reader.counts["sample_count"] == 2
    assert len(reader.questions("sample_a", kinds=("main",))) == 1


def test_absolute_and_escaping_references_are_refused(export_root: Path) -> None:
    reader = open_qa_dataset(export_root)

    with pytest.raises(QaDatasetReadError, match="must be relative to the export root"):
        reader.resolve("/data/elsewhere/audio.wav")
    with pytest.raises(QaDatasetReadError, match="is not confined"):
        reader.resolve("../outside.wav")

    document = json.loads((export_root / "public/dataset_index.json").read_text())
    document["samples"][0]["media"]["audio"]["binaural"] = {
        "path": str(export_root / "media/audio_1.wav")
    }
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))
    with pytest.raises(QaDatasetReadError, match="absolute media references"):
        open_qa_dataset(export_root).verify_self_contained()


def test_world_accounting_rejects_a_duplicated_media_payload(export_root: Path) -> None:
    reader = open_qa_dataset(export_root)
    result = reader.verify_world_accounting()
    assert result["world_key_count"] == 2
    assert result["declared_world_count"] == 2
    assert result["duplicate_media_payloads"] == 0

    # Export the same world twice under a second name.
    duplicate = export_root / "media/video_1_again.mp4"
    duplicate.write_bytes((export_root / "media/video_1.mp4").read_bytes())
    document = json.loads((export_root / "public/dataset_index.json").read_text())
    document["samples"][1]["media"]["video"]["preview"] = "media/video_1_again.mp4"
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))

    with pytest.raises(QaDatasetReadError, match="same media payload more than once"):
        open_qa_dataset(export_root).verify_world_accounting()


def test_coverage_counts_valid_questions_and_reports_the_gaps(export_root: Path) -> None:
    coverage = open_qa_dataset(export_root).coverage()

    assert coverage["sample_count"] == 2
    assert coverage["distinct_world_keys"] == 2
    assert coverage["question_kind_counts"] == {"main": 2, "angle_followup": 2}
    # One item with both forms is one question; the form counts are a breakdown.
    assert coverage["form_counts"] == {"mcq": 2, "open": 4}
    assert coverage["by_qa_id"]["QA-02"]["valid_main_questions"] == 2
    assert coverage["by_qa_id"]["QA-02"]["valid_angle_followups"] == 2
    assert coverage["by_qa_id"]["QA-02"]["distinct_worlds_with_main"] == 2
    assert coverage["qa_ids_without_valid_main_question"] == ["QA-06"]


def test_private_labels_need_the_separate_interface(export_root: Path) -> None:
    reader = open_qa_dataset(export_root)
    assert not hasattr(reader, "gold")
    assert "world_id" not in reader.sample("sample_a")

    private = reader.private()
    assert private.grouping("sample_a")["world_id"] == "engine_world_mp3d_1"
    assert private.grouping("sample_a")["episode_id"] == "episode_a"
    assert private.groups() == {}
    assert private.question_id_map("sample_a")["question_000001"].startswith("qa_02__episode_a")


def test_gold_replay_separates_right_from_wrong_answers(export_root: Path) -> None:
    private = open_qa_dataset(export_root).private()

    assert [row["gold_answer"] for row in private.gold("sample_a", form="mcq")] == ["A"]
    assert [row["gold_answer"] for row in private.gold("sample_b", form="mcq")] == ["B"]

    for form in ("mcq", "open"):
        smoke = private.gold_replay_smoke(form=form)
        assert smoke["gold"]["metrics"]["accuracy"] == pytest.approx(1.0)
        assert smoke["wrong"]["metrics"]["accuracy"] < 1.0
        assert smoke["separates_correct_from_wrong"] is True


def test_a_wrong_answer_scores_zero_and_a_missing_answer_keeps_the_denominator(
    export_root: Path,
) -> None:
    private = open_qa_dataset(export_root).private()
    gold = private.gold_answers(form="mcq")

    exact = private.score(gold, form="mcq")
    assert exact["metrics"] == {
        "question_count": 2,
        "answered_count": 2,
        "missing_answer_count": 0,
        "correct_count": 2,
        "accuracy": pytest.approx(1.0),
        "mean_score": pytest.approx(1.0),
    }

    flipped = {"sample_a": {"question_000001": "B"}, "sample_b": {"question_000002": "A"}}
    wrong = private.score(flipped, form="mcq")
    assert wrong["metrics"]["correct_count"] == 0
    assert wrong["metrics"]["answered_count"] == 2

    partial = private.score({"sample_a": gold["sample_a"]}, form="mcq")
    assert partial["metrics"]["question_count"] == 2
    assert partial["metrics"]["answered_count"] == 1
    assert partial["metrics"]["missing_answer_count"] == 1
    assert partial["metrics"]["accuracy"] == pytest.approx(0.5)


def test_angle_followups_score_separately_from_main_questions(export_root: Path) -> None:
    private = open_qa_dataset(export_root).private()
    result = private.score(private.gold_answers(form="open"), form="open")

    assert result["question_kind_counts"] == {"main": 2, "angle_followup": 2}
    assert result["metrics"]["question_count"] == 4
    assert result["metrics"]["accuracy"] == pytest.approx(1.0)


def test_group_relation_scoring_is_not_applicable_without_a_core_bundle(
    export_root: Path,
) -> None:
    result = open_qa_dataset(export_root).private().score_group_relations(form="mcq")

    assert result["status"] == "not_applicable"
    assert "no core group bundle" in result["reason"]


def test_split_selection_and_unknown_names(export_root: Path) -> None:
    reader = open_qa_dataset(export_root, config={"split": "test"})
    assert reader.list_samples() == ["sample_b"]

    with pytest.raises(QaDatasetReadError, match="is not declared"):
        reader.list_samples(split="calibration")
    with pytest.raises(QaDatasetReadError, match="unknown reader configuration key"):
        open_qa_dataset(export_root, config={"layout": "binaural"})
    with pytest.raises(QaDatasetReadError, match="unsupported answer form"):
        open_qa_dataset(export_root, config={"form": "essay"})


def test_a_private_record_is_required_for_every_public_sample(export_root: Path) -> None:
    document = json.loads((export_root / "private/gold_index.json").read_text())
    document["records"] = document["records"][:1]
    (export_root / "private/gold_index.json").write_text(json.dumps(document))

    with pytest.raises(QaDatasetReadError, match="no record for samples"):
        open_qa_dataset(export_root).private()


def test_a_wrong_schema_or_missing_index_is_refused(tmp_path: Path, export_root: Path) -> None:
    with pytest.raises(QaDatasetReadError, match="dataset root is not a directory"):
        open_qa_dataset(tmp_path / "absent")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(QaDatasetReadError, match="public dataset index is missing"):
        open_qa_dataset(empty)

    document = json.loads((export_root / "public/dataset_index.json").read_text())
    document["schema"] = "something_else_v1"
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))
    with pytest.raises(QaDatasetReadError, match="public index schema is not"):
        open_qa_dataset(export_root)


def test_riff_probe_refuses_a_non_wave_payload(tmp_path: Path) -> None:
    bogus = tmp_path / "not_audio.wav"
    bogus.write_bytes(b"RIFFxxxxNOPEmore")
    with pytest.raises(QaDatasetReadError, match="not a RIFF/WAVE file"):
        reader_module.riff_probe(bogus)


def test_public_payload_list_covers_every_media_reference(export_root: Path) -> None:
    document = json.loads((export_root / "public/dataset_index.json").read_text())
    document["public_payload"] = {
        "paths": [
            "public/dataset_index.json",
            "media/audio_1.wav",
            "media/video_1.mp4",
            "media/audio_2.wav",
            "media/video_2.mp4",
        ]
    }
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))

    reader = open_qa_dataset(export_root)
    assert len(reader.public_payload_paths()) == 5
    isolation = reader.verify_private_isolation()
    assert isolation["status"] == "pass"
    assert isolation["media_references_covered"] == 4
    assert isolation["public_json_files_scanned"] == 1

    document["public_payload"]["paths"].remove("media/audio_2.wav")
    (export_root / "public/dataset_index.json").write_text(json.dumps(document))
    with pytest.raises(QaDatasetReadError, match="omits media the reader needs"):
        open_qa_dataset(export_root).verify_private_isolation()


def test_a_public_payload_list_is_required_before_claiming_isolation(
    export_root: Path,
) -> None:
    reader = open_qa_dataset(export_root)

    with pytest.raises(QaDatasetReadError, match="declares no public_payload path list"):
        reader.public_payload_paths()


def test_public_sample_world_key_is_rejected_as_grouping_leak(export_root: Path) -> None:
    path = export_root / "public/dataset_index.json"
    value = json.loads(path.read_text())
    value["samples"][0]["world_key"] = "leaked_group"
    path.write_text(json.dumps(value))
    with pytest.raises(QaDatasetReadError, match="world_key"):
        open_qa_dataset(export_root)


def test_public_only_inputs_work_without_private_world_joins(export_root: Path) -> None:
    (export_root / "private").rename(export_root / "retained_private_for_test")
    reader = open_qa_dataset(export_root)
    assert reader.list_samples() == ["sample_a", "sample_b"]
    assert "world_key" not in reader.model_input("sample_a")
    assert reader.coverage()["distinct_world_keys"] is None
    assert reader.verify_world_accounting()["status"] == "not_run"
