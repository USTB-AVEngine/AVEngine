"""Harvest planning: no audio stored twice, no speaker in both splits.

Both properties are the kind that fail quietly. A duplicated recording
produces a library that looks twice as large as it is and a question
with no answer; a speaker appearing in training and evaluation produces
a score that is partly memory. Neither shows up as an error at the time.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from avengine.assets.sound_harvest import (
    HarvestError,
    Speaker,
    flac_duration_seconds,
    parse_vctk_speakers,
    plan_fsd50k_harvest,
    plan_vctk_roster,
    plan_vctk_utterances,
    sidecar_for_speech,
)

MAPPING = {
    "target_clips_per_class": 2,
    "entries": [
        {"event_class": "dog_bark", "fsd50k_labels": ["Bark", "Dog"]},
        {"event_class": "alarm_beep", "fsd50k_labels": ["Alarm"]},
        {"event_class": "alarm_clock", "fsd50k_labels": ["Alarm"]},
        {"event_class": "placeholder", "fsd50k_labels": []},
    ],
}
POOL = (
    [{"fname": f"b{i}", "labels": "Bark"} for i in range(5)]
    + [{"fname": f"d{i}", "labels": "Dog"} for i in range(5)]
    + [{"fname": f"a{i}", "labels": "Alarm"} for i in range(3)]
)


def test_one_recording_is_never_stored_twice() -> None:
    plan = plan_fsd50k_harvest(MAPPING, POOL)
    ids = [pick["fsd50k_id"] for pick in plan["picks"]]
    assert len(ids) == len(set(ids)), "同一段音频被选了两次"
    # alarm_beep and alarm_clock draw from the same label and must not
    # both copy the same file
    beep = {p["fsd50k_id"] for p in plan["picks"] if p["event_class"] == "alarm_beep"}
    clock = {p["fsd50k_id"] for p in plan["picks"] if p["event_class"] == "alarm_clock"}
    assert not (beep & clock)
    assert not any(p["event_class"] == "placeholder" for p in plan["picks"])


def test_a_rerun_adds_nothing_and_tops_up_after_a_raised_target() -> None:
    first = plan_fsd50k_harvest(MAPPING, POOL)
    present = {p["fsd50k_id"]: [p["event_class"]] for p in first["picks"]}
    again = plan_fsd50k_harvest(MAPPING, POOL, already_present=present)
    assert again["picks"] == []

    bigger = plan_fsd50k_harvest(
        MAPPING, POOL, already_present=present, target_per_class=4
    )
    assert bigger["picks"], "提高目标后应该继续补"
    assert all(p["fsd50k_id"] not in present for p in bigger["picks"])


def test_a_class_that_cannot_be_filled_is_reported_not_hidden() -> None:
    plan = plan_fsd50k_harvest(MAPPING, POOL, target_per_class=10)
    short = {row["event_class"] for row in plan["shortfalls"]}
    # only three Alarm clips exist, shared between two classes
    assert "alarm_beep" in short or "alarm_clock" in short


def test_the_plan_is_deterministic() -> None:
    assert plan_fsd50k_harvest(MAPPING, POOL) == plan_fsd50k_harvest(MAPPING, POOL)


SPEAKER_INFO = "ID  AGE  GENDER  ACCENTS  REGION\n" + "\n".join(
    f"p{200 + i}  {20 + i}  {'F' if i % 2 == 0 else 'M'}  English  Region"
    for i in range(40)
)


def test_speakers_never_appear_in_both_splits() -> None:
    speakers = parse_vctk_speakers(SPEAKER_INFO)
    assert len(speakers) == 40
    roster = plan_vctk_roster(speakers, train_per_gender=3, eval_per_gender=2)
    train = {s.speaker_id for s in roster if s.split == "train"}
    evaluation = {s.speaker_id for s in roster if s.split == "eval"}
    assert not (train & evaluation), "同一个说话人同时出现在训练和考试里"
    assert len(train) == 6 and len(evaluation) == 4
    for split, count in (("train", 3), ("eval", 2)):
        for gender in ("F", "M"):
            assert sum(
                1 for s in roster if s.split == split and s.gender == gender
            ) == count, "男女没配平"


def test_a_roster_larger_than_the_corpus_fails_loudly() -> None:
    with pytest.raises(HarvestError, match="不够"):
        plan_vctk_roster(parse_vctk_speakers(SPEAKER_INFO), train_per_gender=50)


def test_shared_sentences_come_first_and_are_marked() -> None:
    speaker = Speaker("p225", "F", "English", "train")
    available = {f"{i:03d}": 3.0 for i in range(1, 60)}
    available["050"] = 0.4      # too short
    available["051"] = 30.0     # too long
    picked = plan_vctk_utterances(speaker, available, per_speaker=30)

    assert len(picked) == 30
    assert [p["sentence_id"] for p in picked[:5]] == ["001", "002", "003", "004", "005"]
    assert all(p["controlled_content"] for p in picked[:23])
    assert not any(p["controlled_content"] for p in picked[23:])
    assert "050" not in {p["sentence_id"] for p in picked}
    assert "051" not in {p["sentence_id"] for p in picked}


def test_speech_sidecar_carries_the_answer_and_the_split() -> None:
    speaker = Speaker("p225", "F", "English", "eval")
    pick = plan_vctk_utterances(speaker, {"001": 2.0}, per_speaker=1)[0]
    sidecar = sidecar_for_speech(pick, "  Please call Stella.\n")
    assert sidecar["transcript"] == "Please call Stella."
    assert sidecar["gender"] == "F" and sidecar["split"] == "eval"
    assert sidecar["controlled_content"] is True
    assert sidecar["event_classes"] == ["speech_playback"]
    assert sidecar["dry"] is True and sidecar["language"] == "en"


def test_flac_duration_is_read_from_the_header(tmp_path: Path) -> None:
    """A synthetic STREAMINFO: 48 kHz, 96000 samples, so two seconds."""

    sample_rate, total = 48000, 96000
    bits = (sample_rate << 44) | (1 << 41) | (15 << 36) | total
    block = b"\x00" * 10 + bits.to_bytes(8, "big") + b"\x00" * 16
    path = tmp_path / "probe.flac"
    path.write_bytes(
        b"fLaC" + bytes([0x80]) + struct.pack(">I", len(block))[1:] + block
    )
    assert abs(flac_duration_seconds(path) - 2.0) < 1e-6

    (tmp_path / "not.flac").write_bytes(b"RIFFnope")
    assert flac_duration_seconds(tmp_path / "not.flac") is None
