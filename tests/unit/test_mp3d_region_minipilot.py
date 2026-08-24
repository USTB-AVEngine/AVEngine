"""CPU tests for Habitat-native MP3D four-point QA design."""

import pytest

from avengine.qa.answer_balance import build_answer_balance_report
from tools.qa import build_mp3d_region_minipilot as pilot


def test_listener_local_azimuth_uses_positive_right() -> None:
    listener = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert pilot.listener_local_azimuth_deg(
        listener, [1.0, 0.0, -1.0]
    ) == pytest.approx(45.0)
    assert pilot.listener_local_azimuth_deg(
        listener, [-1.0, 0.0, -1.0]
    ) == pytest.approx(-45.0)


def test_four_motion_cases_balance_ta_and_t7() -> None:
    questions = []
    for motion_case, (first_answer, second_answer) in pilot.MOTION_ANSWERS.items():
        base = {"point_id": motion_case, "motion_case": motion_case}
        questions.extend(
            [
                pilot._motion_question(
                    base=base,
                    type_id="QV2-TA-MOTION",
                    source_slot="source1",
                    english_ordinal="first",
                    chinese_ordinal="一",
                    answer=first_answer,
                ),
                pilot._motion_question(
                    base=base,
                    type_id="QV2-T7-DURING",
                    source_slot="source2",
                    english_ordinal="second",
                    chinese_ordinal="二",
                    answer=second_answer,
                ),
            ]
        )
    report = build_answer_balance_report(questions)
    assert report["status"] == "pass"
    assert report["by_type"]["QV2-TA-MOTION"]["answer_key_counts"] == {"A": 2, "B": 2}
    assert report["by_type"]["QV2-T7-DURING"]["answer_key_counts"] == {"A": 2, "B": 2}


def test_t5_question_retains_banded_numeric_authority() -> None:
    row = pilot._t5_question(
        base={"point_id": "p0"},
        source_slot="source2",
        event_frame=17,
        azimuth=-1.234,
    )
    assert row["type_id"] == "QV2-T5-DOA"
    assert row["answer_numeric"] == -1.2
    assert row["scoring"]["full_credit_within_deg"] == 15.0
    assert row["scoring"]["partial_credit_within_deg"] == 30.0
    assert row["gold_authority"] == "habitat_native_listener_and_emitter_link_readback"
