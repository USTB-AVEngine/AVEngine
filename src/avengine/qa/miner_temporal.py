"""Temporal (B-group) and numeric question mining over declared event windows.

Only fact tables whose sound events carry ``declared_intermittent_program_v1``
windows are eligible: the declared window is ground truth by construction, so
temporal anchors are exact ticks. Every question carries the anchoring event
ids and the gate margins that make its answer unambiguous. MCQ questions use
``format: "mcq"``; numeric questions use ``format: "numeric_banded"`` with
their scoring bands recorded instead of options.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from avengine.qa.intermittent import WINDOW_AUTHORITY
from avengine.qa.miner import QAMinerError, _question as _mcq_question

TIME_BASE_HZ = 48_000
MOTION_CLEAR_FRACTION = 0.9
AT_MIN_ABS_AZIMUTH_DEG = 15.0
ORDER_MARGIN_TICKS = 14_400  # 0.3 s
NUM_TIME_BANDS_S = (0.3, 1.0)

_SPECIES_EN = {"dog": "dog", "cat": "cat", "human": "person"}
_SPECIES_ZH = {"dog": "狗", "cat": "猫", "human": "人"}
_CALL_EN = {"dog": "bark", "cat": "meow", "human": "utterance"}
_CALL_ZH = {"dog": "狗叫", "cat": "猫叫", "human": "说话声"}
_ORDINAL_EN = ("first", "second", "third", "fourth")
_ORDINAL_ZH = ("第一", "第二", "第三", "第四")


def _events_by_slot(fact_table: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    events: dict[str, list[Mapping[str, Any]]] = {}
    for event in fact_table["sound_events"]:
        if event.get("window_authority") != WINDOW_AUTHORITY:
            raise QAMinerError(
                "temporal mining requires declared intermittent windows"
            )
        events.setdefault(event["source_slot_id"], []).append(event)
    for slot_events in events.values():
        slot_events.sort(key=lambda event: event["start_tick"])
    return events


def _instance(fact_table: Mapping[str, Any], slot_id: str) -> Mapping[str, Any]:
    for instance in fact_table["instances"]:
        if instance["instance_id"] == slot_id:
            return instance
    raise QAMinerError(f"instance {slot_id!r} missing from fact table")


def _unique_species(fact_table: Mapping[str, Any]) -> dict[str, str]:
    """species_id -> slot, only for species appearing exactly once."""

    counts: dict[str, list[str]] = {}
    for instance in fact_table["instances"]:
        counts.setdefault(instance["species_id"], []).append(
            instance["instance_id"]
        )
    return {
        species: slots[0] for species, slots in counts.items() if len(slots) == 1
    }


def _numeric_question(
    *,
    type_id: str,
    episode_id: str,
    qualifier: str,
    question_en: str,
    question_zh: str,
    answer_numeric: float,
    unit: str,
    bands: Sequence[float],
    evidence: Mapping[str, Any],
    modality_note: str,
) -> dict[str, Any]:
    question_id = f"{type_id.lower().replace('-', '_')}__{episode_id}__{qualifier}"
    return {
        "question_id": question_id,
        "type_id": type_id,
        "episode_id": episode_id,
        "format": "numeric_banded",
        "question_en": question_en,
        "question_zh": question_zh,
        "answer_numeric": float(answer_numeric),
        "unit": unit,
        "scoring_bands": [float(band) for band in bands],
        "evidence": dict(evidence),
        "modality_note": modality_note,
        "certification": {"status": "pending", "axis1_twin": "planned_P1prime"},
    }


def mine_q_num_cnt(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """How many times did the <species> call in the whole clip?"""

    questions = []
    events = _events_by_slot(fact_table)
    for species, slot in sorted(_unique_species(fact_table).items()):
        slot_events = events.get(slot)
        if not slot_events:
            continue
        count = len(slot_events)
        options = [
            {"value": str(value), "label_en": str(value), "label_zh": str(value)}
            for value in range(1, 5)
        ]
        question = _mcq_question(
            type_id="Q-NUM-CNT",
            episode_id=fact_table["episode_id"],
            qualifier=species,
            question_en=(
                f"How many separate {_CALL_EN[species]}s did the "
                f"{_SPECIES_EN[species]} make during the clip?"
            ),
            question_zh=f"整段里一共有几声{_CALL_ZH[species]}？",
            options=options,
            answer_value=str(count),
            evidence={
                "source_slot_id": slot,
                "event_ids": [event["event_id"] for event in slot_events],
            },
            modality_note=(
                "event count is declared truth; audible as separated bursts"
            ),
        )
        question["format"] = "mcq"
        questions.append(question)
    return questions


def mine_q_num_time(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """At what time did the first <species> call start?"""

    questions = []
    events = _events_by_slot(fact_table)
    for species, slot in sorted(_unique_species(fact_table).items()):
        slot_events = events.get(slot)
        if not slot_events:
            continue
        first = slot_events[0]
        start_seconds = first["start_tick"] / TIME_BASE_HZ
        questions.append(
            _numeric_question(
                type_id="Q-NUM-TIME",
                episode_id=fact_table["episode_id"],
                qualifier=species,
                question_en=(
                    f"At what time (in seconds from the start) does the first "
                    f"{_CALL_EN[species]} of the {_SPECIES_EN[species]} begin?"
                ),
                question_zh=f"第一声{_CALL_ZH[species]}开始于第几秒？",
                answer_numeric=round(start_seconds, 3),
                unit="seconds",
                bands=NUM_TIME_BANDS_S,
                evidence={
                    "source_slot_id": slot,
                    "event_id": first["event_id"],
                    "start_tick": first["start_tick"],
                },
                modality_note="event onset is declared truth at integer ticks",
            )
        )
    return questions


def _motion_mode_in_frames(
    fact_table: Mapping[str, Any], slot_id: str, start_frame: int, end_frame: int
) -> str | None:
    moving = fact_table["tracks"]["instances"][slot_id]["moving"]
    window = moving[start_frame:end_frame]
    if not window:
        return None
    fraction = sum(1 for value in window if value) / len(window)
    if fraction >= MOTION_CLEAR_FRACTION:
        return "moving"
    if fraction <= 1.0 - MOTION_CLEAR_FRACTION:
        return "static"
    return None


_MOTION_OPTIONS = [
    {"value": "moving", "label_en": "moving", "label_zh": "在走动"},
    {"value": "static", "label_en": "staying still", "label_zh": "静止不动"},
]
_SIDE_OPTIONS = [
    {"value": "left", "label_en": "on your left", "label_zh": "左侧"},
    {"value": "right", "label_en": "on your right", "label_zh": "右侧"},
]


def mine_q_temp_during(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """During the k-th call of species A, was species B moving or still?"""

    questions = []
    events = _events_by_slot(fact_table)
    unique = _unique_species(fact_table)
    for species_a, slot_a in sorted(unique.items()):
        for species_b, slot_b in sorted(unique.items()):
            if slot_a == slot_b:
                continue
            for index, event in enumerate(events.get(slot_a, [])[:4]):
                mode = _motion_mode_in_frames(
                    fact_table, slot_b, event["start_frame"], event["end_frame"]
                )
                if mode is None:
                    continue
                question = _mcq_question(
                    type_id="Q-TEMP-DURING",
                    episode_id=fact_table["episode_id"],
                    qualifier=f"{species_a}{index}_{species_b}",
                    question_en=(
                        f"During the {_ORDINAL_EN[index]} {_CALL_EN[species_a]}, "
                        f"is the {_SPECIES_EN[species_b]} moving or staying still?"
                    ),
                    question_zh=(
                        f"{_ORDINAL_ZH[index]}声{_CALL_ZH[species_a]}期间，"
                        f"{_SPECIES_ZH[species_b]}在走动还是静止？"
                    ),
                    options=_MOTION_OPTIONS,
                    answer_value=mode,
                    evidence={
                        "anchor_event_id": event["event_id"],
                        "window_frames": [event["start_frame"], event["end_frame"]],
                        "observed_slot": slot_b,
                    },
                    modality_note=(
                        "audio anchors the window, vision answers the motion "
                        "state inside it"
                    ),
                )
                question["format"] = "mcq"
                questions.append(question)
    return questions


def mine_q_temp_at(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """When the k-th call of species A starts, which side is species B on?"""

    questions = []
    events = _events_by_slot(fact_table)
    unique = _unique_species(fact_table)
    for species_a, slot_a in sorted(unique.items()):
        for species_b, slot_b in sorted(unique.items()):
            if slot_a == slot_b:
                continue
            azimuth = fact_table["tracks"]["instances"][slot_b]["doa"]["azimuth_deg"]
            for index, event in enumerate(events.get(slot_a, [])[:4]):
                frame = min(event["start_frame"], len(azimuth) - 1)
                value = azimuth[frame]
                if abs(value) < AT_MIN_ABS_AZIMUTH_DEG or abs(value) > 180.0 - AT_MIN_ABS_AZIMUTH_DEG:
                    continue
                question = _mcq_question(
                    type_id="Q-TEMP-AT",
                    episode_id=fact_table["episode_id"],
                    qualifier=f"{species_a}{index}_{species_b}",
                    question_en=(
                        f"When the {_ORDINAL_EN[index]} {_CALL_EN[species_a]} "
                        f"starts, is the {_SPECIES_EN[species_b]} on your left "
                        f"or your right?"
                    ),
                    question_zh=(
                        f"{_ORDINAL_ZH[index]}声{_CALL_ZH[species_a]}开始时，"
                        f"{_SPECIES_ZH[species_b]}在你的左侧还是右侧？"
                    ),
                    options=_SIDE_OPTIONS,
                    answer_value="right" if value > 0 else "left",
                    evidence={
                        "anchor_event_id": event["event_id"],
                        "anchor_frame": frame,
                        "azimuth_deg": value,
                    },
                    modality_note=(
                        "audio anchors the instant; the answer needs the other "
                        "instance's direction at that instant"
                    ),
                )
                question["format"] = "mcq"
                questions.append(question)
    return questions


def mine_q_temp_between(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Between two consecutive calls of species A, was species B moving?"""

    questions = []
    events = _events_by_slot(fact_table)
    unique = _unique_species(fact_table)
    for species_a, slot_a in sorted(unique.items()):
        slot_events = events.get(slot_a, [])
        for species_b, slot_b in sorted(unique.items()):
            if slot_a == slot_b:
                continue
            for index in range(len(slot_events) - 1):
                gap_start = slot_events[index]["end_frame"]
                gap_end = slot_events[index + 1]["start_frame"]
                if gap_end - gap_start < 3:
                    continue
                mode = _motion_mode_in_frames(fact_table, slot_b, gap_start, gap_end)
                if mode is None:
                    continue
                question = _mcq_question(
                    type_id="Q-TEMP-BETWEEN",
                    episode_id=fact_table["episode_id"],
                    qualifier=f"{species_a}{index}_{species_b}",
                    question_en=(
                        f"Between the {_ORDINAL_EN[index]} and "
                        f"{_ORDINAL_EN[index + 1]} {_CALL_EN[species_a]}s, is the "
                        f"{_SPECIES_EN[species_b]} moving or staying still?"
                    ),
                    question_zh=(
                        f"{_ORDINAL_ZH[index]}声和{_ORDINAL_ZH[index + 1]}声"
                        f"{_CALL_ZH[species_a]}之间，{_SPECIES_ZH[species_b]}"
                        f"在走动还是静止？"
                    ),
                    options=_MOTION_OPTIONS,
                    answer_value=mode,
                    evidence={
                        "gap_between_event_ids": [
                            slot_events[index]["event_id"],
                            slot_events[index + 1]["event_id"],
                        ],
                        "gap_frames": [gap_start, gap_end],
                        "observed_slot": slot_b,
                    },
                    modality_note=(
                        "the silent gap is only locatable by listening; vision "
                        "answers the motion state inside it"
                    ),
                )
                question["format"] = "mcq"
                questions.append(question)
    return questions


def mine_q_temp_order(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Whose first call starts earlier?"""

    events = _events_by_slot(fact_table)
    unique = _unique_species(fact_table)
    if len(unique) != 2:
        return []
    (species_a, slot_a), (species_b, slot_b) = sorted(unique.items())
    events_a = events.get(slot_a)
    events_b = events.get(slot_b)
    if not events_a or not events_b:
        return []
    start_a = events_a[0]["start_tick"]
    start_b = events_b[0]["start_tick"]
    if abs(start_a - start_b) < ORDER_MARGIN_TICKS:
        return []
    winner = species_a if start_a < start_b else species_b
    options = [
        {
            "value": species,
            "label_en": f"the {_SPECIES_EN[species]}'s",
            "label_zh": _SPECIES_ZH[species],
        }
        for species in (species_a, species_b)
    ]
    question = _mcq_question(
        type_id="Q-TEMP-ORDER",
        episode_id=fact_table["episode_id"],
        qualifier="first_call",
        question_en=(
            f"Whose first call comes earlier: the {_SPECIES_EN[species_a]}'s "
            f"or the {_SPECIES_EN[species_b]}'s?"
        ),
        question_zh=(
            f"{_SPECIES_ZH[species_a]}和{_SPECIES_ZH[species_b]}，"
            f"谁先发出第一声？"
        ),
        options=options,
        answer_value=winner,
        evidence={
            "first_start_ticks": {species_a: start_a, species_b: start_b},
            "margin_ticks": abs(start_a - start_b),
        },
        modality_note=(
            "audio-temporal ordering; margin gated at 0.3 s to stay "
            "perceptually unambiguous"
        ),
    )
    question["format"] = "mcq"
    return [question]


_TEMPORAL_MINERS: dict[str, Callable[[Mapping[str, Any]], list[dict[str, Any]]]] = {
    "Q-NUM-CNT": mine_q_num_cnt,
    "Q-NUM-TIME": mine_q_num_time,
    "Q-TEMP-DURING": mine_q_temp_during,
    "Q-TEMP-AT": mine_q_temp_at,
    "Q-TEMP-BETWEEN": mine_q_temp_between,
    "Q-TEMP-ORDER": mine_q_temp_order,
}


def mine_temporal_fact_table(fact_table: Mapping[str, Any]) -> list[dict[str, Any]]:
    if fact_table.get("schema") != "avengine_qa_fact_table_v1":
        raise QAMinerError("input is not an avengine_qa_fact_table_v1 document")
    questions: list[dict[str, Any]] = []
    for miner in _TEMPORAL_MINERS.values():
        questions.extend(miner(fact_table))
    return questions
