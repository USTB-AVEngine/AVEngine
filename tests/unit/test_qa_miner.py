from __future__ import annotations

import pytest

from avengine.qa.fact_table import compile_episode_fact_table
from avengine.qa.miner import (
    QAMinerError,
    answer_histogram,
    balance_answer_histogram,
    mine_fact_table,
    mine_q_avrel,
    mine_q_cmp,
)

FRAME_COUNT = 75
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _registry() -> dict:
    def asset(asset_id: str, species: str, coat, offset, forward):
        realized = {
            "size": "medium",
            "body_build": "standard",
            "life_stage": "adult",
        }
        if coat is not None:
            realized["coat_profile"] = {
                "profile_id": f"{species}_coat_v1",
                "value": coat,
            }
        return {
            "asset_id": asset_id,
            "revision": "test_v1",
            "display_label": species.title(),
            "entity_class": "articulated_animal",
            "identity": {"species_id": species, "breed_id": f"{species}_breed"},
            "realized_attributes": realized,
            "timeline": {"local_anatomical_forward_axis": forward},
            "default_emitter_anchor_id": "muzzle",
            "emitter_anchors": [
                {
                    "anchor_id": "muzzle",
                    "anchor_type": "muzzle",
                    "offset_m": offset,
                    "offset_space": "final_scaled_asset_root",
                }
            ],
            "admission_state": "research",
        }

    return {
        "registry_id": "test_registry",
        "revision": "test_rev",
        "assets": [
            asset("asset_dog", "dog", "black_white", [0.4, 0.6, 0.0], [1, 0, 0]),
            asset("asset_cat", "cat", "ruddy", [0.3, 0.15, 0.0], [1, 0, 0]),
        ],
    }


def _fact_table(*, cat_x_start: float = 1.0, cat_z: float = 1.0) -> dict:
    """Static dog on the left-front; cat walking +X on the given z plane."""

    dog_root = [-2.0, 0.5, -2.0]
    dog_emitter = [-2.0, 1.1, -2.4]
    cat_roots = [[cat_x_start + 0.02 * i, 0.3, cat_z] for i in range(FRAME_COUNT)]
    cat_emitters = [
        [cat_x_start + 0.02 * i - 0.3, 0.45, cat_z] for i in range(FRAME_COUNT)
    ]
    bank_episode = {
        "episode_id": "miner_episode_0000",
        "motion_case": "source1_static_source2_moving",
        "source_center_paths_m": {
            "source1": [list(dog_emitter) for _ in range(FRAME_COUNT)],
            "source2": cat_emitters,
        },
        "source_root_paths_m": {
            "source1": [list(dog_root) for _ in range(FRAME_COUNT)],
            "source2": cat_roots,
        },
    }
    sample_entry = {
        "episode_id": "miner_episode_0000",
        "asset_ids_by_source_slot": {"source1": "asset_dog", "source2": "asset_cat"},
        "audio": {
            "channel_count": 2,
            "sample_rate_hz": 16000,
            "sample_count": 80000,
            "peak_absolute": 0.05,
            "mixture": {"path": "miner_episode_0000__v00.wav", "audio_sha256": SHA_A},
        },
    }
    dry = {
        slot: {
            "variant_index": 0,
            "record": {"input": {"path": f"/dry/{slot}.wav", "sha256": SHA_B},
                       "linear_gain": 0.1},
        }
        for slot in ("source1", "source2")
    }
    return compile_episode_fact_table(
        bank_header={
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": 15,
            "seconds_per_episode": 5.0,
            "source_slots": ["source1", "source2"],
        },
        bank_episode=bank_episode,
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        sample_entry=sample_entry,
        dry_variants_by_slot=dry,
        registry=_registry(),
        anchors=[
            {
                "anchor_id": "marker_front",
                "kind": "marker",
                "position_m": [0.0, 0.5, -4.0],
                "yaw_deg": None,
            }
        ],
        room={"room_capsule_id": "test_room", "revision": "v1"},
        rir_cache_request_identity_sha256=SHA_C,
        provenance_inputs=[
            {"role": "trajectory_bank", "path": "/x/bank.json", "sha256": SHA_B}
        ],
    )


def test_mine_fact_table_yields_expected_types_and_answers() -> None:
    questions = mine_fact_table(_fact_table())
    by_type: dict[str, list[dict]] = {}
    for question in questions:
        by_type.setdefault(question["type_id"], []).append(question)

    assert sorted(by_type) == ["Q-ACT", "Q-ATTR", "Q-AVREL", "Q-CMP", "Q-CNT", "Q-SRC"]

    attr_answers = {
        question["evidence"]["instance_id"]: question["answer_value"]
        for question in by_type["Q-ATTR"]
    }
    assert attr_answers == {"source1": "black_white", "source2": "ruddy"}

    act_answers = {
        question["evidence"]["instance_id"]: question["answer_value"]
        for question in by_type["Q-ACT"]
    }
    assert act_answers == {"source1": "static", "source2": "moving"}

    cnt_answers = {
        question["evidence"]["species_id"]: question["answer_value"]
        for question in by_type["Q-CNT"]
    }
    assert cnt_answers == {"dog": "1", "cat": "1", "human": "0"}

    (cmp_question,) = by_type["Q-CMP"]
    assert cmp_question["answer_value"] == "cat"
    assert cmp_question["evidence"]["mean_margin_m"] >= 0.5

    (src_question,) = by_type["Q-SRC"]
    assert src_question["answer_value"] == "dog+cat"

    avrel_answers = {
        question["evidence"]["instance_id"]: question["answer_value"]
        for question in by_type["Q-AVREL"]
    }
    assert avrel_answers["source1"] == "left"
    assert avrel_answers["source2"] == "right"

    for question in questions:
        assert question["options"][question["answer_index"]]["value"] == (
            question["answer_value"]
        )
        assert question["certification"]["status"] == "pending"


def test_avrel_gate_skips_front_crossing_and_near_front_sources() -> None:
    # Cat crosses the front axis: azimuth sign flips, so no Q-AVREL for it.
    crossing = _fact_table(cat_x_start=-0.4, cat_z=-1.0)
    avrel = mine_q_avrel(crossing)
    assert [q["evidence"]["instance_id"] for q in avrel] == ["source1"]


def test_cmp_gate_requires_distance_margin() -> None:
    # Cat mean distance close to the dog's: margin below the 0.5 m gate.
    near_tie = _fact_table(cat_x_start=-3.35, cat_z=0.6)
    assert mine_q_cmp(near_tie) == []


def test_miner_rejects_non_fact_table_input() -> None:
    with pytest.raises(QAMinerError):
        mine_fact_table({"schema": "something_else"})


def test_balance_answer_histogram_is_deterministic_and_uniform() -> None:
    def question(type_id: str, answer: str, index: int) -> dict:
        return {
            "question_id": f"{type_id}__{answer}__{index}",
            "type_id": type_id,
            "answer_value": answer,
        }

    questions = (
        [question("Q-ACT", "moving", i) for i in range(6)]
        + [question("Q-ACT", "static", i) for i in range(2)]
        + [question("Q-CNT", "1", i) for i in range(3)]
    )
    balanced = balance_answer_histogram(questions, seed="s1")
    histogram = answer_histogram(balanced)
    assert histogram["Q-ACT"] == {"moving": 2, "static": 2}
    assert histogram["Q-CNT"] == {"1": 3}
    again = balance_answer_histogram(questions, seed="s1")
    assert [q["question_id"] for q in again] == [q["question_id"] for q in balanced]
    different_seed = balance_answer_histogram(questions, seed="s2")
    assert answer_histogram(different_seed)["Q-ACT"] == {"moving": 2, "static": 2}
