"""Plan what to pull from the local corpora into the dry-sound library.

Collecting material at scale is a download job, and download jobs do not
belong to people. Both corpora this module plans against already sit on
the machine - a 42,745-clip sound-effect set and a 111-speaker speech
corpus - so the work is selection, not acquisition, and selection is
arithmetic once someone has said which label means which of our event
classes. That mapping is the one human step; everything below is the
machine part.

Two properties are load-bearing and therefore explicit here rather than
left to the calling script:

**One recording is stored once.** A clip that serves both "alarm beep"
and "alarm clock" gets one file whose sidecar names both classes. Two
copies would let the question miner hand two sound sources in the same
room the same waveform and then ask which of them is sounding, and that
question has no answer.

**Speakers do not straddle the train and evaluation splits.** A model
that heard a voice during training recognises it later for the wrong
reason, so the split is drawn over speakers, not over utterances, and
the two sides are disjoint by construction. VCTK also hands us something
better than we could build: its first twenty-three sentences are the
same text for every speaker, which is a ready-made controlled set where
content is held constant and only the voice varies.
"""

from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# VCTK's elicitation paragraph and rainbow passage: identical text for
# every speaker, verified across four speakers on the delivered corpus.
VCTK_COMMON_SENTENCE_IDS = tuple(f"{index:03d}" for index in range(1, 24))
VCTK_MIN_SECONDS = 1.5
VCTK_MAX_SECONDS = 6.0
FSD50K_MIN_SECONDS = 0.5
FSD50K_MAX_SECONDS = 20.0

# These are optional source metadata fields. They are copied only when an
# explicit sidecar supplies them; no field is reconstructed from a filename or
# the human-readable source string.
SPEECH_METADATA_FIELDS = (
    "speaker_id",
    "utterance_id",
    "transcript",
    "split",
)


def speech_metadata_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit speech provenance fields without inventing labels."""

    if not isinstance(value, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    for key in SPEECH_METADATA_FIELDS:
        if key not in value or value[key] is None:
            continue
        raw = value[key]
        if not isinstance(raw, str) or not raw.strip():
            continue
        metadata[key] = raw
    return metadata


class HarvestError(ValueError):
    pass


def flac_duration_seconds(path: Path) -> float | None:
    """Duration from the FLAC STREAMINFO block, without decoding audio.

    Selecting a few hundred utterances out of eighty thousand should not
    cost eighty thousand decodes; the header carries sample rate and
    total sample count, which is all the filter needs.
    """

    with path.open("rb") as handle:
        if handle.read(4) != b"fLaC":
            return None
        header = handle.read(4)
        if len(header) < 4:
            return None
        length = int.from_bytes(header[1:4], "big")
        block = handle.read(length)
        if (header[0] & 0x7F) != 0 or len(block) < 18:
            return None
    # bits 80..99 sample rate, bits 100..135 total samples
    bits = int.from_bytes(block[10:18], "big")
    sample_rate = (bits >> 44) & 0xFFFFF
    total_samples = bits & 0xFFFFFFFFF
    if not sample_rate or not total_samples:
        return None
    return total_samples / sample_rate


# --------------------------------------------------------------------------
# sound effects


@dataclass(frozen=True)
class EffectPick:
    event_class: str
    fsd50k_id: str
    label: str
    relative_dir: str


def plan_fsd50k_harvest(
    mapping: Mapping[str, Any],
    pool_rows: Sequence[Mapping[str, str]],
    *,
    already_present: Mapping[str, Sequence[str]] | None = None,
    target_per_class: int | None = None,
    seed: int = 20260901,
) -> dict[str, Any]:
    """Choose which pool clips fill which class, without repeating audio.

    ``already_present`` maps an FSD50K id already in the library to the
    event classes it currently serves, so a re-run adds nothing it
    already added and an id reused by a second class is recorded as an
    extra class on the existing file rather than a second copy.
    """

    present = {str(k): list(v) for k, v in (already_present or {}).items()}
    by_label: dict[str, list[str]] = {}
    for row in pool_rows:
        by_label.setdefault(str(row["labels"]), []).append(str(row["fname"]))

    target = int(target_per_class or mapping.get("target_clips_per_class") or 20)
    picks: list[EffectPick] = []
    extra_classes: dict[str, list[str]] = {}
    shortfalls: list[dict[str, Any]] = []
    claimed: set[str] = set()

    for entry in mapping.get("entries") or []:
        event_class = str(entry["event_class"])
        labels = [str(label) for label in entry.get("fsd50k_labels") or []]
        if not labels:
            continue
        have = sum(1 for classes in present.values() if event_class in classes)
        wanted = max(0, target - have)
        if wanted == 0:
            continue

        candidates: list[tuple[str, str]] = []
        for label in labels:
            for fsd_id in by_label.get(label, []):
                candidates.append((fsd_id, label))
        # Deterministic order that is not simply "lowest id first": the
        # pool is ordered by upload, and taking its head would bias every
        # class toward one era of one contributor.
        rng = random.Random(f"{seed}:{event_class}")
        rng.shuffle(candidates)

        taken = 0
        for fsd_id, label in candidates:
            if taken >= wanted:
                break
            if fsd_id in present:
                if event_class not in present[fsd_id]:
                    present[fsd_id].append(event_class)
                    extra_classes.setdefault(fsd_id, []).append(event_class)
                    taken += 1
                continue
            if fsd_id in claimed:
                continue
            claimed.add(fsd_id)
            present[fsd_id] = [event_class]
            picks.append(
                EffectPick(event_class, fsd_id, label, f"{event_class}/fsd50k_{fsd_id}")
            )
            taken += 1
        if taken < wanted:
            shortfalls.append(
                {
                    "event_class": event_class,
                    "wanted": wanted,
                    "found": taken,
                    "labels": labels,
                }
            )

    return {
        "target_per_class": target,
        "picks": [pick.__dict__ for pick in picks],
        "extra_classes_for_existing": extra_classes,
        "shortfalls": shortfalls,
    }


# --------------------------------------------------------------------------
# speech


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    gender: str
    accent: str
    split: str


def parse_vctk_speakers(info_text: str) -> list[Speaker]:
    speakers: list[Speaker] = []
    for line in info_text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4 or not parts[0].startswith("p"):
            continue
        speakers.append(Speaker(parts[0], parts[2].upper(), parts[3], "unassigned"))
    return speakers


def plan_vctk_roster(
    speakers: Sequence[Speaker],
    *,
    train_per_gender: int = 8,
    eval_per_gender: int = 4,
    seed: int = 20260901,
) -> list[Speaker]:
    """A gender-balanced roster whose train and eval speakers are disjoint.

    Disjoint by speakers rather than by utterances, because a voice heard
    in training is recognised at evaluation for the wrong reason. Both
    sides carry equal numbers of each gender so a model cannot score by
    guessing the more common one.
    """

    chosen: list[Speaker] = []
    for gender in ("F", "M"):
        pool = sorted(
            (s for s in speakers if s.gender == gender),
            key=lambda s: s.speaker_id,
        )
        needed = train_per_gender + eval_per_gender
        if len(pool) < needed:
            raise HarvestError(
                f"{gender} 说话人只有 {len(pool)} 个,不够 {needed} 个"
            )
        rng = random.Random(f"{seed}:{gender}")
        rng.shuffle(pool)
        for index, speaker in enumerate(pool[:needed]):
            split = "train" if index < train_per_gender else "eval"
            chosen.append(
                Speaker(speaker.speaker_id, speaker.gender, speaker.accent, split)
            )
    return sorted(chosen, key=lambda s: (s.split, s.gender, s.speaker_id))


def plan_vctk_utterances(
    speaker: Speaker,
    available: Mapping[str, float],
    *,
    per_speaker: int = 25,
    common_sentence_ids: Iterable[str] = VCTK_COMMON_SENTENCE_IDS,
    minimum_seconds: float = VCTK_MIN_SECONDS,
    maximum_seconds: float = VCTK_MAX_SECONDS,
    seed: int = 20260901,
) -> list[dict[str, Any]]:
    """Pick this speaker's utterances: the shared text first, then their own.

    ``available`` maps sentence id to duration in seconds. The shared
    sentences come first and are marked, because a set where every
    speaker says the same words is the only way to ask which of two
    people is speaking without the words themselves giving it away.
    """

    usable = {
        sentence_id: seconds
        for sentence_id, seconds in available.items()
        if minimum_seconds <= seconds <= maximum_seconds
    }
    common = [s for s in common_sentence_ids if s in usable]
    rest = sorted(set(usable) - set(common))
    rng = random.Random(f"{seed}:{speaker.speaker_id}")
    rng.shuffle(rest)

    picked: list[dict[str, Any]] = []
    for sentence_id in common + rest:
        if len(picked) >= per_speaker:
            break
        picked.append(
            {
                "speaker_id": speaker.speaker_id,
                "gender": speaker.gender,
                "accent": speaker.accent,
                "split": speaker.split,
                "sentence_id": sentence_id,
                "duration_s": round(usable[sentence_id], 3),
                "controlled_content": sentence_id in set(common_sentence_ids),
            }
        )
    return picked


def sidecar_for_speech(pick: Mapping[str, Any], transcript: str) -> dict[str, Any]:
    return {
        "event_classes": ["speech_playback"],
        "source": (
            f"VCTK-Corpus 0.92, speaker {pick['speaker_id']}, "
            f"sentence {pick['sentence_id']}"
        ),
        "dry": True,
        "transcript": transcript.strip(),
        "speaker_id": pick["speaker_id"],
        "utterance_id": pick["sentence_id"],
        "gender": pick["gender"],
        "accent": pick["accent"],
        "split": pick["split"],
        "controlled_content": bool(pick["controlled_content"]),
        "language": "en",
        "notes": (
            "半消声室录音,天然干声;转写为数据集原文,可直接作为标准答案"
        ),
    }


def sidecar_for_effect(pick: Mapping[str, Any], event_classes: Sequence[str]) -> dict:
    return {
        "event_classes": list(event_classes),
        "source": f"FSD50K dataset, file {pick['fsd50k_id']}.wav, label {pick['label']}",
        "dry": True,
        "notes": "自动采集自 FSD50K 单标签子集(一条录音只挂一个标签)",
    }


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
