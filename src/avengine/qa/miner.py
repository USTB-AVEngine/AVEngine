"""Mining-first simple question generation over compiled fact tables.

Questions are mined from predicates that already hold in a fact table;
nothing is staged or re-simulated. Every emitted question carries the
evidence values it was mined from, the gate margins that make its answer
unambiguous, and an honest modality note. Certification (axis-1 twins,
off-screen certificates) is attached in a later phase and is recorded here
as ``pending``.

v1 scope (continuous dual-source audio, no visibility facts yet):
``Q-ATTR``, ``Q-ACT``, ``Q-CNT``, ``Q-CMP``, ``Q-SRC`` (sounding-pair
form) and ``Q-AVREL`` are mineable. ``Q-LOC-EGO`` needs the P1 modal
visibility pass and ``Q-LOC-ALLO`` needs the furniture OBB snapshot; both
are reported as deferred rather than silently skipped.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

MINER_SCHEMA = "avengine_qa_simple_question_set_v1"

MOVING_FRACTION_CLEAR = 0.95
STATIC_FRACTION_CLEAR = 0.05
CMP_DISTANCE_MARGIN_M = 0.5
CMP_SIGN_CONSISTENCY = 0.9
AVREL_SIGN_CONSISTENCY = 0.9
AVREL_MIN_ABS_AZIMUTH_DEG = 15.0
AVREL_MAX_ABS_AZIMUTH_DEG = 165.0

DEFERRED_TYPES = {
    "Q-LOC-EGO": "needs the P1 modal/amodal visibility pass",
    "Q-LOC-ALLO": "needs the furniture OBB anchor snapshot",
}

_VOCALIZATION_EN = {"dog": "barking", "cat": "meowing", "human": "speaking"}
_SPECIES_EN = {"dog": "dog", "cat": "cat", "human": "person"}
_SPECIES_ZH = {"dog": "狗", "cat": "猫", "human": "人"}
_VOCALIZATION_ZH = {"dog": "在叫", "cat": "在叫", "human": "在说话"}
_COAT_EN = {
    "black_white": "black and white",
    "yellow": "yellow",
    "standard_tricolor": "tricolor",
    "ruddy": "ruddy brown",
    "black": "black",
    "white": "white",
    "tabby": "tabby",
}
_COAT_ZH = {
    "black_white": "黑白色",
    "yellow": "黄色",
    "standard_tricolor": "三色",
    "ruddy": "红棕色",
    "black": "黑色",
    "white": "白色",
    "tabby": "虎斑",
}
_COAT_DOMAINS = {
    "dog": ["black_white", "yellow", "standard_tricolor", "black"],
    "cat": ["ruddy", "black", "white", "tabby"],
}
_SPECIES_PAIR_OPTIONS = [
    ("dog", "cat"),
    ("dog", "human"),
    ("cat", "human"),
    ("dog", "dog"),
]


class QAMinerError(ValueError):
    """A fact table violates an assumption the miner relies on."""


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _stable_shuffle(values: Sequence[Any], *keys: str) -> list[Any]:
    return sorted(values, key=lambda value: _stable_digest(*keys, str(value)))


def _instances(fact_table: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    instances = fact_table.get("instances")
    if not isinstance(instances, Sequence) or not instances:
        raise QAMinerError("fact table lacks instances")
    return list(instances)


def _track(fact_table: Mapping[str, Any], instance_id: str) -> Mapping[str, Any]:
    track = fact_table["tracks"]["instances"].get(instance_id)
    if not isinstance(track, Mapping):
        raise QAMinerError(f"fact table lacks a track for {instance_id!r}")
    return track


def _unique_species_instance(
    fact_table: Mapping[str, Any], species_id: str
) -> Mapping[str, Any] | None:
    matches = [
        instance
        for instance in _instances(fact_table)
        if instance.get("species_id") == species_id
    ]
    return matches[0] if len(matches) == 1 else None


def _species_phrase(species_id: str) -> tuple[str, str]:
    return (
        f"the {_SPECIES_EN[species_id]} that is {_VOCALIZATION_EN[species_id]}",
        f"{_VOCALIZATION_ZH[species_id]}的那只{_SPECIES_ZH[species_id]}"
        if species_id != "human"
        else "在说话的那个人",
    )


def _question(
    *,
    type_id: str,
    episode_id: str,
    qualifier: str,
    question_en: str,
    question_zh: str,
    options: Sequence[Mapping[str, str]],
    answer_value: str,
    evidence: Mapping[str, Any],
    modality_note: str,
) -> dict[str, Any]:
    question_id = f"{type_id.lower().replace('-', '_')}__{episode_id}__{qualifier}"
    ordered = _stable_shuffle(
        [dict(option) for option in options], question_id, "options"
    )
    answer_index = [option["value"] for option in ordered].index(answer_value)
    return {
        "question_id": question_id,
        "type_id": type_id,
        "episode_id": episode_id,
        "question_en": question_en,
        "question_zh": question_zh,
        "options": ordered,
        "answer_index": answer_index,
        "answer_value": answer_value,
        "evidence": dict(evidence),
        "modality_note": modality_note,
        "certification": {"status": "pending", "axis1_twin": "planned_P1prime"},
    }


def _coat_options(species_id: str) -> list[dict[str, str]]:
    domain = _COAT_DOMAINS.get(species_id)
    if not domain:
        return []
    return [
        {"value": coat, "label_en": _COAT_EN[coat], "label_zh": _COAT_ZH[coat]}
        for coat in domain
    ]


def mine_q_attr(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Coat colour of the uniquely-identified vocalizing animal."""

    questions: list[dict[str, Any]] = []
    for instance in _instances(fact_table):
        species = instance.get("species_id")
        coat = instance.get("attributes", {}).get("coat_value")
        if species not in _COAT_DOMAINS or not coat:
            continue
        if _unique_species_instance(fact_table, species) is None:
            continue
        options = _coat_options(species)
        if coat not in {option["value"] for option in options}:
            continue
        phrase_en, phrase_zh = _species_phrase(species)
        questions.append(
            _question(
                type_id="Q-ATTR",
                episode_id=fact_table["episode_id"],
                qualifier=instance["instance_id"],
                question_en=f"What is the coat colour of {phrase_en}?",
                question_zh=f"{phrase_zh}是什么毛色？",
                options=options,
                answer_value=coat,
                evidence={
                    "instance_id": instance["instance_id"],
                    "asset_id": instance["asset_id"],
                    "coat_value": coat,
                    "sound_event_ids": [
                        event["event_id"]
                        for event in fact_table["sound_events"]
                        if event["source_slot_id"] == instance["instance_id"]
                    ],
                },
                modality_note=(
                    "single instance of this species in scene; audio identifies "
                    "the species, vision supplies the coat; same-species "
                    "distractor pairs land in P2'"
                ),
            )
        )
    return questions


def mine_q_act(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Walking versus staying still, only when the whole window is clear."""

    questions: list[dict[str, Any]] = []
    options = [
        {"value": "moving", "label_en": "moving", "label_zh": "在走动"},
        {"value": "static", "label_en": "staying still", "label_zh": "静止不动"},
    ]
    for instance in _instances(fact_table):
        species = instance.get("species_id")
        if species not in _SPECIES_EN:
            continue
        if _unique_species_instance(fact_table, species) is None:
            continue
        track = _track(fact_table, instance["instance_id"])
        moving = track["moving"]
        fraction = sum(1 for value in moving if value) / len(moving)
        if STATIC_FRACTION_CLEAR < fraction < MOVING_FRACTION_CLEAR:
            continue
        answer = "moving" if fraction >= MOVING_FRACTION_CLEAR else "static"
        phrase_en, phrase_zh = _species_phrase(species)
        questions.append(
            _question(
                type_id="Q-ACT",
                episode_id=fact_table["episode_id"],
                qualifier=instance["instance_id"],
                question_en=f"Is {phrase_en} moving or staying still?",
                question_zh=f"{phrase_zh}在走动还是静止不动？",
                options=options,
                answer_value=answer,
                evidence={
                    "instance_id": instance["instance_id"],
                    "moving_frame_fraction": fraction,
                    "moving_threshold_mps": track["moving_threshold_mps"],
                },
                modality_note=(
                    "answerable from vision alone once the instance is "
                    "localized; audio identifies which instance is asked about"
                ),
            )
        )
    return questions


def mine_q_cnt(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Scene-level species counts, explicitly including off-screen sources."""

    questions: list[dict[str, Any]] = []
    options = [
        {"value": str(count), "label_en": str(count), "label_zh": str(count)}
        for count in range(4)
    ]
    species_counts: dict[str, int] = {}
    for instance in _instances(fact_table):
        species = instance.get("species_id")
        species_counts[species] = species_counts.get(species, 0) + 1
    for species in sorted(_SPECIES_EN):
        count = species_counts.get(species, 0)
        plural_en = {"dog": "dogs", "cat": "cats", "human": "people"}[species]
        questions.append(
            _question(
                type_id="Q-CNT",
                episode_id=fact_table["episode_id"],
                qualifier=species,
                question_en=(
                    f"Counting off-screen sound sources too, how many "
                    f"{plural_en} are in the scene?"
                ),
                question_zh=f"算上画面外的声源，场景里一共有几{'个' if species == 'human' else '只'}{_SPECIES_ZH[species]}？",
                options=options,
                answer_value=str(count),
                evidence={"species_id": species, "instance_count": count},
                modality_note=(
                    "scene truth from the instance table; visibility split "
                    "(on-screen versus off-screen variants) lands in P1"
                ),
            )
        )
    return questions


def mine_q_cmp(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Which of the two sources stays closer to the listener."""

    instances = _instances(fact_table)
    if len(instances) != 2:
        return []
    first, second = instances
    if first.get("species_id") == second.get("species_id"):
        return []
    track_a = _track(fact_table, first["instance_id"])
    track_b = _track(fact_table, second["instance_id"])
    distance_a = np.asarray(track_a["doa"]["distance_m"], dtype=np.float64)
    distance_b = np.asarray(track_b["doa"]["distance_m"], dtype=np.float64)
    closer_a_fraction = float(np.mean(distance_a < distance_b))
    margin = abs(float(np.mean(distance_a)) - float(np.mean(distance_b)))
    if margin < CMP_DISTANCE_MARGIN_M:
        return []
    if CMP_SIGN_CONSISTENCY > closer_a_fraction > 1.0 - CMP_SIGN_CONSISTENCY:
        return []
    closer = first if closer_a_fraction >= CMP_SIGN_CONSISTENCY else second
    options = [
        {
            "value": instance["species_id"],
            "label_en": _SPECIES_EN[instance["species_id"]],
            "label_zh": _SPECIES_ZH[instance["species_id"]],
        }
        for instance in instances
    ]
    return [
        _question(
            type_id="Q-CMP",
            episode_id=fact_table["episode_id"],
            qualifier="closer",
            question_en=(
                f"Throughout the clip, which is closer to you: the "
                f"{_SPECIES_EN[first['species_id']]} or the "
                f"{_SPECIES_EN[second['species_id']]}?"
            ),
            question_zh=(
                f"整段中，{_SPECIES_ZH[first['species_id']]}和"
                f"{_SPECIES_ZH[second['species_id']]}谁离你更近？"
            ),
            options=options,
            answer_value=closer["species_id"],
            evidence={
                "mean_distance_m": {
                    first["instance_id"]: float(np.mean(distance_a)),
                    second["instance_id"]: float(np.mean(distance_b)),
                },
                "closer_fraction": max(closer_a_fraction, 1.0 - closer_a_fraction),
                "mean_margin_m": margin,
            },
            modality_note=(
                "distance is an audio-dominant cue at fixed listener pose; "
                "numeric twin Q-NUM-DIST shares this evidence"
            ),
        )
    ]


def mine_q_src(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Which pair of source kinds is sounding (both sources are active)."""

    species = sorted(
        event["sound_class"]["species_id"] for event in fact_table["sound_events"]
    )
    if len(species) != 2:
        return []
    pair = tuple(species)
    options = []
    answer_value = None
    for candidate in _SPECIES_PAIR_OPTIONS:
        value = "+".join(candidate)
        label_en = " and ".join(_SPECIES_EN[item] for item in candidate)
        label_zh = "和".join(_SPECIES_ZH[item] for item in candidate)
        options.append({"value": value, "label_en": label_en, "label_zh": label_zh})
        if tuple(sorted(candidate)) == pair:
            answer_value = value
    if answer_value is None:
        raise QAMinerError(f"unexpected sounding pair {pair!r}")
    return [
        _question(
            type_id="Q-SRC",
            episode_id=fact_table["episode_id"],
            qualifier="pair",
            question_en="Which two kinds of sound sources are sounding in this scene?",
            question_zh="这个场景里正在发声的是哪两种声源？",
            options=options,
            answer_value=answer_value,
            evidence={"sounding_species": list(pair)},
            modality_note=(
                "both sources are continuously active in v1; single-event "
                "attribution (which one just called) lands with P2 "
                "intermittent windows"
            ),
        )
    ]


def mine_q_avrel(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Left/right of a species' sound, gated away from front/rear ambiguity."""

    questions: list[dict[str, Any]] = []
    options = [
        {"value": "left", "label_en": "from the left", "label_zh": "左侧"},
        {"value": "right", "label_en": "from the right", "label_zh": "右侧"},
    ]
    for instance in _instances(fact_table):
        species = instance.get("species_id")
        if species not in _SPECIES_EN:
            continue
        if _unique_species_instance(fact_table, species) is None:
            continue
        track = _track(fact_table, instance["instance_id"])
        azimuth = np.asarray(track["doa"]["azimuth_deg"], dtype=np.float64)
        right_fraction = float(np.mean(azimuth > 0.0))
        if AVREL_SIGN_CONSISTENCY > right_fraction > 1.0 - AVREL_SIGN_CONSISTENCY:
            continue
        median_abs = float(np.median(np.abs(azimuth)))
        if not AVREL_MIN_ABS_AZIMUTH_DEG <= median_abs <= AVREL_MAX_ABS_AZIMUTH_DEG:
            continue
        answer = "right" if right_fraction >= AVREL_SIGN_CONSISTENCY else "left"
        sound_en = {
            "dog": "the dog's barking",
            "cat": "the cat's meowing",
            "human": "the person's voice",
        }[species]
        sound_zh = {"dog": "狗叫声", "cat": "猫叫声", "human": "说话声"}[species]
        questions.append(
            _question(
                type_id="Q-AVREL",
                episode_id=fact_table["episode_id"],
                qualifier=instance["instance_id"],
                question_en=f"Is {sound_en} coming from your left or your right?",
                question_zh=f"{sound_zh}来自你的左侧还是右侧？",
                options=options,
                answer_value=answer,
                evidence={
                    "instance_id": instance["instance_id"],
                    "sign_consistency": max(right_fraction, 1.0 - right_fraction),
                    "median_abs_azimuth_deg": median_abs,
                    "azimuth_convention": fact_table["listener"][
                        "azimuth_convention"
                    ],
                },
                modality_note=(
                    "azimuth sign is an audio cue over the full 360-degree "
                    "field; the off-screen certified variant lands in P1"
                ),
            )
        )
    return questions


_MINERS: dict[str, Callable[[Mapping[str, Any]], list[dict[str, Any]]]] = {
    "Q-ATTR": mine_q_attr,
    "Q-ACT": mine_q_act,
    "Q-CNT": mine_q_cnt,
    "Q-CMP": mine_q_cmp,
    "Q-SRC": mine_q_src,
    "Q-AVREL": mine_q_avrel,
}


def mine_fact_table(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    if fact_table.get("schema") != "avengine_qa_fact_table_v1":
        raise QAMinerError("input is not an avengine_qa_fact_table_v1 document")
    questions: list[dict[str, Any]] = []
    for miner in _MINERS.values():
        questions.extend(miner(fact_table))
    return questions


def answer_histogram(questions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    histogram: dict[str, dict[str, int]] = {}
    for question in questions:
        by_type = histogram.setdefault(question["type_id"], {})
        answer = question["answer_value"]
        by_type[answer] = by_type.get(answer, 0) + 1
    return histogram


def balance_answer_histogram(
    questions: Sequence[Mapping[str, Any]], *, seed: str
) -> list[dict[str, Any]]:
    """Downsample each type so every observed answer is equally frequent."""

    by_type_answer: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for question in questions:
        key = (question["type_id"], question["answer_value"])
        by_type_answer.setdefault(key, []).append(question)
    minimum_by_type: dict[str, int] = {}
    for (type_id, _), bucket in by_type_answer.items():
        current = minimum_by_type.get(type_id)
        minimum_by_type[type_id] = (
            len(bucket) if current is None else min(current, len(bucket))
        )
    balanced: list[dict[str, Any]] = []
    for (type_id, answer), bucket in sorted(
        by_type_answer.items(), key=lambda item: item[0]
    ):
        keep = minimum_by_type[type_id]
        ordered = sorted(
            bucket, key=lambda question: _stable_digest(seed, question["question_id"])
        )
        balanced.extend(dict(question) for question in ordered[:keep])
    return sorted(balanced, key=lambda question: question["question_id"])
