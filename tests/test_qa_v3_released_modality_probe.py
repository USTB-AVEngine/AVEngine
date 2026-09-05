"""Focused tests for released-media shortcut probe inputs and scoring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from build_qa_v3_released_probe_items import (  # noqa: E402
    _validate_media_clock,
    build,
)
from probe_released_modality_shortcuts import run  # noqa: E402


PARAMS = {"THETA_FULL": 15.0, "THETA_HALF": 30.0,
          "T_FULL": 0.3, "T_HALF": 1.0}
TOOL = TOOLS / "build_qa_v3_released_probe_items.py"


def _released_fixture(tmp_path, open_block, *, point="card8_001",
                      selection_extra=None):
    facts, audio, media = tmp_path / "facts", tmp_path / "audio", tmp_path / "media"
    (facts / point).mkdir(parents=True)
    (audio / point / "audio/binaural").mkdir(parents=True)
    (media / point).mkdir(parents=True)
    (audio / point / "audio/binaural/mixture.wav").write_bytes(b"wav")
    (media / point / "video_only.mp4").write_bytes(b"mp4")
    fact = {
        "profile_id": "card8",
        "episode_id": f"episode-{point}",
        "target_first": True,
        "mcq": {
            "stem": "when?",
            "options_space": ["a", "b"],
            "truth_option": "a",
        },
        "open": open_block,
    }
    (facts / point / "fact_record.json").write_text(
        json.dumps(fact), encoding="utf-8")
    selection = {
        "selected": [{"point_id": point, "profile_id": "card8"}],
    }
    if selection_extra:
        selection.update(selection_extra)
    return selection, facts, audio, media, point


def test_text_probe_reports_empirical_not_universal_claim():
    items = []
    for index in range(8):
        label = "left" if index % 2 == 0 else "right"
        items.append({
            "question_id": f"q{index}", "group_id": f"q{index}",
            "profile_id": "p", "form": "mcq",
            "task_type": "classification", "question": f"token {label}",
            "options": ["left", "right"], "truth": label,
        })
    result = run(items, "text", PARAMS, folds=4)
    assert result["records"][0]["accuracy"] >= 0.75
    assert result["qualification_claim"] is False
    assert "does not prove" in result["boundary"]


def test_probe_output_embeds_executed_scoring_params_and_fails_closed():
    items = [{
        "question_id": f"q{index}", "group_id": f"q{index}",
        "profile_id": "p", "form": "mcq", "task_type": "classification",
        "question": "same", "options": ["a", "b"], "truth": "a",
    } for index in range(4)]
    result = run(items, "text", PARAMS, folds=2)
    assert result["scoring_params"]["T_FULL"] == 0.3
    assert result["scoring_params"]["T_HALF"] == 1.0
    assert result["scoring_params"]["time_certification_policy"] == \
        "strict_full_credit_only"
    assert result["scoring_params"]["T_FULL_status"] == \
        "unspecified_treat_as_placeholder"
    import pytest
    with pytest.raises(ValueError, match="T_FULL"):
        run(items, "text", {k: v for k, v in PARAMS.items() if k != "T_FULL"},
            folds=2)


def test_numeric_time_probe_uses_strict_scorer():
    items = []
    for index in range(6):
        items.append({
            "question_id": f"q{index}", "group_id": f"q{index}",
            "profile_id": "card8", "form": "open",
            "task_type": "numeric_time", "question": "same",
            "options": [], "truth": 1.0 + 0.5 * index,
        })
    result = run(items, "text", PARAMS, folds=3)
    record = result["records"][0]
    assert 0.0 <= record["mean_scorer_score"] <= 1.0
    assert 0.0 <= record["empirical_constant_baseline"] <= 1.0


def _transcript_items():
    return [
        {
            "question_id": f"speech_{index}",
            "group_id": f"speech_{index}",
            "profile_id": "speech",
            "form": "open",
            "task_type": "transcript_wer",
            "question": "What was said?",
            "options": [],
            "truth": truth,
        }
        for index, truth in enumerate(
            ("Hello, world!", "One two", "Third sentence"))
    ]


def _transcript_params():
    return {
        "TRANSCRIPT_NORMALIZATION": {
            "unicode_form": "NFKC",
            "casefold": True,
            "punctuation": "remove",
        }
    }


def test_transcript_probe_scores_explicit_free_text_without_accuracy(tmp_path):
    items = _transcript_items()
    predictions = {
        "speech_0": " hello world ",
        "speech_1": "one two",
        "speech_2": "third sentence",
    }
    result = run(
        items, "text", _transcript_params(), folds=3,
        predictions=predictions,
    )
    record = result["records"][0]
    assert record["task_type"] == "transcript_wer"
    assert record["status"] == "research_probe_complete"
    assert record["n"] == 3
    assert record["mean_wer"] == pytest.approx(0.0)
    assert record["mean_scorer_score"] == pytest.approx(1.0)
    assert record["exact_match_rate"] == pytest.approx(1.0)
    assert "accuracy" not in record
    assert result["scoring_params"]["TRANSCRIPT_NORMALIZATION"] ==         _transcript_params()["TRANSCRIPT_NORMALIZATION"]


def test_transcript_probe_without_predictions_stays_pending():
    result = run(_transcript_items(), "text", _transcript_params(), folds=3)
    record = result["records"][0]
    assert record["status"] == "pending_predictions"
    assert "mean_wer" not in record


def test_transcript_probe_cli_consumes_a_fixed_prediction_file(tmp_path):
    items_path = tmp_path / "items.json"
    params_path = tmp_path / "params.json"
    predictions_path = tmp_path / "predictions.json"
    output_path = tmp_path / "probe.json"
    items = _transcript_items()
    items_path.write_text(json.dumps(items), encoding="utf-8")
    params_path.write_text(json.dumps(_transcript_params()), encoding="utf-8")
    predictions_path.write_text(json.dumps({
        "speech_0": "HELLO WORLD",
        "speech_1": "one two",
        "speech_2": "third sentence",
    }), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(TOOLS / "probe_released_modality_shortcuts.py"),
            "--items", str(items_path),
            "--modality", "text",
            "--params", str(params_path),
            "--predictions", str(predictions_path),
            "--output", str(output_path),
        ],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(completed.stdout)["groups"] == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["records"][0]["task_type"] == "transcript_wer"
    assert result["records"][0]["mean_scorer_score"] == pytest.approx(1.0)
    assert result["predictions_source"]["kind"] == "explicit_free_text"


def test_transcript_wer_is_not_emitted_as_classification(tmp_path):
    selection, facts, audio, media, _ = _released_fixture(
        tmp_path,
        {
            "stem": "What was said?",
            "truth_value": "hello",
            "scoring": "transcript_wer",
        },
    )
    rows = build(selection, facts, audio, media)
    open_row = next(row for row in rows if row["form"] == "open")
    assert open_row["task_type"] == "transcript_wer"


def test_item_builder_uses_only_released_paths_and_question_gold(tmp_path):
    selection, facts, audio, media, point = _released_fixture(
        tmp_path,
        {
            "stem": "seconds?",
            "truth_value": 2.4,
            "scoring": "absolute_time",
            "certification_policy": "strict_full_credit_only",
        },
    )
    rows = build(selection, facts, audio, media)
    assert len(rows) == 2
    assert rows[0]["question_id"] == f"{point}__mcq"
    assert rows[1]["question_id"] == f"{point}__open"
    assert rows[0]["question_id"] != rows[1]["question_id"]
    assert rows[0]["group_id"] == rows[1]["group_id"] == f"episode-{point}"
    assert rows[0]["point_id"] == rows[1]["point_id"] == point
    assert rows[0]["episode_id"] == rows[1]["episode_id"] == f"episode-{point}"
    assert "scene_id" not in rows[0]
    assert rows[1]["task_type"] == "numeric_time"
    assert rows[1]["certification_policy"] == "strict_full_credit_only"
    forbidden = {
        "timeline", "dry", "rir", "engine_frame_note",
        "azimuth_deg_engine_frame", "azimuth_interval_engine_frame",
        "query_window_seconds",
    }
    assert all(forbidden.isdisjoint(row) for row in rows)


def test_interval_angle_keeps_authoritative_interval_and_dcase_convention(tmp_path):
    selection, facts, audio, media, point = _released_fixture(
        tmp_path,
        {
            "stem": "angle?",
            "truth_value": 175.0,
            "truth_interval_deg": [172.0, 178.0],
            "truth_value_note": "midpoint of truth_interval_deg",
            "unit": "deg",
            "scoring": "circular_deg_interval",
            "convention": "dcase_foa_left_positive",
            "certification_policy": "strict_full_credit_only",
        },
        point="card1F_001",
    )
    rows = build(selection, facts, audio, media)
    open_row = next(row for row in rows if row["form"] == "open")
    assert open_row["task_type"] == "numeric_angle"
    assert open_row["truth"] == 175.0
    assert open_row["truth_interval_deg"] == [172.0, 178.0]
    assert open_row["convention"] == "dcase_foa_left_positive"
    assert open_row["certification_policy"] == "strict_full_credit_only"
    assert open_row["question_id"] == f"{point}__open"


def test_answer_form_selection_filters_single_form_and_explicit_wins(tmp_path):
    selection, facts, audio, media, _ = _released_fixture(
        tmp_path,
        {
            "stem": "seconds?",
            "truth_value": 2.4,
            "scoring": "absolute_time",
        },
        selection_extra={"ANSWER_FORMS_DEFAULT": ["open"]},
    )
    rows = build(selection, facts, audio, media)
    assert [row["form"] for row in rows] == ["open"]

    rows = build(
        selection,
        facts,
        audio,
        media,
        answer_forms=["mcq"],
        params={"ANSWER_FORMS_DEFAULT": ["open"]},
    )
    assert [row["form"] for row in rows] == ["mcq"]

    selection.pop("ANSWER_FORMS_DEFAULT")
    rows = build(
        selection,
        facts,
        audio,
        media,
        params={"ANSWER_FORMS_DEFAULT": ["open"]},
    )
    assert [row["form"] for row in rows] == ["open"]


def test_cli_accepts_explicit_form_and_params(tmp_path):
    selection, facts, audio, media, point = _released_fixture(
        tmp_path,
        {
            "stem": "seconds?",
            "truth_value": 2.4,
            "scoring": "absolute_time",
        },
    )
    selection_path = tmp_path / "selection.json"
    params_path = tmp_path / "params.json"
    output_path = tmp_path / "items.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    params_path.write_text(
        json.dumps({"ANSWER_FORMS_DEFAULT": ["mcq", "open"]}),
        encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(TOOL),
            "--selection-manifest", str(selection_path),
            "--facts-root", str(facts),
            "--audio-root", str(audio),
            "--media-root", str(media),
            "--output", str(output_path),
            "--params", str(params_path),
            "--answer-form", "open",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["record_count"] == 1
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert rows[0]["question_id"] == f"{point}__open"


def test_numeric_probe_keeps_strict_policy_and_authoritative_interval():
    import probe_released_modality_shortcuts as probe_module
    params = {"THETA_FULL": 15.0, "THETA_HALF": 30.0, "T_FULL": 0.5, "T_HALF": 1.0,
              "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only"}
    _, scores = probe_module._numeric_score([20.0], [0.0], "numeric_angle", params)
    assert scores == [0.0]
    _, scores = probe_module._numeric_score(
        [20.0], [0.0], "numeric_angle", params,
        [{"truth_interval_deg": [10.0, 20.0], "convention": "dcase_foa_left_positive",
          "certification_policy": "strict_full_credit_only"}])
    assert scores == [1.0]
    assert probe_module.scoring_snapshot(params)["angle_policy"] == "strict_full_credit_only"



def test_requested_form_only_reads_its_fact_block(tmp_path):
    selection, facts, audio, media, point = _released_fixture(
        tmp_path,
        {
            "stem": "seconds?",
            "truth_value": 2.4,
            "scoring": "absolute_time",
        },
    )
    fact_path = facts / point / "fact_record.json"
    fact = json.loads(fact_path.read_text())
    fact.pop("open")
    fact_path.write_text(json.dumps(fact), encoding="utf-8")
    rows = build(selection, facts, audio, media, answer_forms=["mcq"])
    assert [row["form"] for row in rows] == ["mcq"]

    selection, facts, audio, media, point = _released_fixture(
        tmp_path / "open-only",
        {
            "stem": "seconds?",
            "truth_value": 2.4,
            "scoring": "absolute_time",
        },
    )
    fact_path = facts / point / "fact_record.json"
    fact = json.loads(fact_path.read_text())
    fact.pop("mcq")
    fact_path.write_text(json.dumps(fact), encoding="utf-8")
    rows = build(selection, facts, audio, media, answer_forms=["open"])
    assert [row["form"] for row in rows] == ["open"]


def test_scene_and_point_components_are_unambiguous_and_duplicates_refused(tmp_path):
    facts = tmp_path / "facts"
    audio = tmp_path / "audio"
    media = tmp_path / "media"
    selected = []
    for point, scene in (("b", "a__"), ("__b", "a")):
        (facts / point).mkdir(parents=True)
        (audio / point / "audio/binaural").mkdir(parents=True)
        (media / point).mkdir(parents=True)
        (audio / point / "audio/binaural/mixture.wav").write_bytes(b"wav")
        (media / point / "video_only.mp4").write_bytes(b"mp4")
        (facts / point / "fact_record.json").write_text(
            json.dumps({
                "scene_id": scene,
                "profile_id": "p",
                "episode_id": "e",
                "mcq": {"stem": "m", "options_space": ["a"],
                        "truth_option": "a"},
                "open": {"stem": "o", "truth_value": 1.0,
                         "scoring": "absolute_time"},
            }),
            encoding="utf-8",
        )
        selected.append({"point_id": point})

    rows = build({"selected": selected}, facts, audio, media)
    question_ids = [row["question_id"] for row in rows]
    assert len(question_ids) == len(set(question_ids)) == 4
    assert all(question_id.startswith("qa3:") for question_id in question_ids)

    with pytest.raises(ValueError, match="duplicate question_id"):
        build({"selected": [selected[0], selected[0]]}, facts, audio, media)


@pytest.mark.parametrize(
    ("theta_full", "theta_half"),
    [(31.0, 30.0), (float("nan"), 30.0)],
)
def test_probe_rejects_invalid_angle_tolerances_for_classification(
    theta_full, theta_half,
):
    items = [{
        "question_id": f"q{index}", "group_id": f"q{index}",
        "profile_id": "p", "form": "mcq", "task_type": "classification",
        "question": "same", "options": ["a", "b"], "truth": "a",
    } for index in range(4)]
    with pytest.raises(ValueError, match="angle tolerances|finite"):
        run(
            items, "text",
            {**PARAMS, "THETA_FULL": theta_full, "THETA_HALF": theta_half},
            folds=2,
        )



def test_item_builder_consumes_assembler_candidates_and_pilot_media(tmp_path):
    fact_path = tmp_path / "source" / "fact_record.json"
    fact_path.parent.mkdir(parents=True)
    fact_path.write_text(json.dumps({
        "scene_id": "room_a",
        "point_id": "card12_001",
        "profile_id": "card12",
        "episode_id": "episode_1",
        "answer_forms": ["open"],
        "open": {"stem": "which?", "truth_value": "a",
                 "scoring": "closed_set"},
    }))
    pilot_id = "pilot:6:room_a6:card125:001"
    audio = tmp_path / "audio" / pilot_id / "audio/binaural"
    media = tmp_path / "media" / pilot_id
    audio.mkdir(parents=True)
    media.mkdir(parents=True)
    (audio / "mixture.wav").write_bytes(b"wav")
    (media / "video_only.mp4").write_bytes(b"mp4")
    selection = {
        "answer_forms": ["open"],
        "rooms": {
            "room_a": {
                "profiles": {
                    "card12": {
                        "status": "selected",
                        "answer_forms": ["open"],
                        "candidates": [{
                            "pilot_id": pilot_id,
                            "source_point_id": "card12_001",
                            "artifacts": {"fact": str(fact_path)},
                        }],
                    },
                },
            },
        },
    }
    rows = build(selection, audio_root=tmp_path / "audio",
                 media_root=tmp_path / "media")
    assert len(rows) == 1
    assert rows[0]["form"] == "open"
    assert rows[0]["pilot_id"] == pilot_id
    assert rows[0]["scene_id"] == "room_a"
    assert rows[0]["point_id"] == "card12_001"
    assert rows[0]["episode_id"] == "episode_1"
    assert rows[0]["audio"].endswith(
        f"/{pilot_id}/audio/binaural/mixture.wav")
    assert rows[0]["video"].endswith(f"/{pilot_id}/video_only.mp4")


def test_released_clock_rejects_inconsistent_audio_duration(tmp_path):
    wav = tmp_path / "mixture.wav"
    video = tmp_path / "video_only.mp4"
    sf.write(wav, np.zeros((32000, 2), dtype=np.float32), 16000, subtype="FLOAT")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=2x2:r=15:d=2",
            "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", "15", str(video),
        ],
        check=True,
    )
    clock = {
        "frame_count": 30, "frame_rate_hz": 15,
        "sample_rate_hz": 16000, "sample_count": 32000,
    }
    report = _validate_media_clock(
        {"frame_clock": clock}, wav, video, owner="clock-test")
    assert report["clip_seconds"] == pytest.approx(2.0)
    assert report["audio_media_duration_seconds"] == pytest.approx(2.0)

    point_dir = tmp_path / "point"
    point_dir.mkdir()
    point_clock = {key: value for key, value in clock.items()
                   if key != "frame_rate_hz"}
    point_clock["video_fps"] = 15
    point_dir.joinpath("audio_program.json").write_text(json.dumps({
        "timeline": point_clock,
    }), encoding="utf-8")
    point_local_report = _validate_media_clock(
        {}, wav, video, owner="point-local-clock-test", point_dir=point_dir)
    assert point_local_report["frame_count"] == 30
    with pytest.raises(ValueError, match="released audio duration"):
        _validate_media_clock(
            {"frame_clock": {**clock, "frame_count": 45}},
            wav, video, owner="clock-test",
        )


def _transcript_group(profile, prefix):
    truths = ("Hello, world!", "One two", "Third sentence")
    return [
        {
            "question_id": f"{prefix}_{index}",
            "group_id": f"{prefix}_{index}",
            "profile_id": profile,
            "form": "open",
            "task_type": "transcript_wer",
            "question": "What was said?",
            "options": [],
            "truth": truth,
        }
        for index, truth in enumerate(truths)
    ]


def test_transcript_predictions_are_validated_globally_then_sliced_per_group():
    items = _transcript_group("speech_a", "a")
    items += _transcript_group("speech_b", "b")
    predictions = {
        row["question_id"]: row["truth"].lower()
        for row in items
    }
    result = run(
        items, "text", _transcript_params(), folds=3,
        predictions=predictions,
    )
    records = {
        record["profile_id"]: record
        for record in result["records"]
    }
    assert set(records) == {"speech_a", "speech_b"}
    assert all(record["status"] == "research_probe_complete"
               and record["n"] == 3
               and record["mean_wer"] == pytest.approx(0.0)
               for record in records.values())


def test_transcript_prediction_missing_and_extra_ids_are_checked_across_groups():
    items = _transcript_group("speech_a", "a")
    items += _transcript_group("speech_b", "b")
    predictions = {
        row["question_id"]: row["truth"]
        for row in items
        if row["profile_id"] == "speech_a"
    }
    predictions["unexpected"] = "extra"
    with pytest.raises(
        ValueError,
        match=r"missing=.*b_0.*extra=.*unexpected",
    ):
        run(
            items, "text", _transcript_params(), folds=3,
            predictions=predictions,
        )


def test_mixed_transcript_and_classification_only_consumes_transcript_predictions():
    transcript = _transcript_group("speech", "speech")
    classification = [
        {
            "question_id": f"class_{index}",
            "group_id": f"class_{index}",
            "profile_id": "visual",
            "form": "mcq",
            "task_type": "classification",
            "question": f"token {label}",
            "options": ["left", "right"],
            "truth": label,
        }
        for index, label in enumerate(("left", "right", "left", "right"))
    ]
    predictions = {
        row["question_id"]: row["truth"]
        for row in transcript
    }
    result = run(
        transcript + classification,
        "text", _transcript_params(), folds=2,
        predictions=predictions,
    )
    records = {
        record["task_type"]: record
        for record in result["records"]
    }
    assert records["transcript_wer"]["status"] == "research_probe_complete"
    assert "accuracy" not in records["transcript_wer"]
    assert records["classification"]["status"] == "research_probe_complete"
    with pytest.raises(ValueError, match=r"extra=.*class_0"):
        run(
            transcript + classification,
            "text", _transcript_params(), folds=2,
            predictions={**predictions, "class_0": "left"},
        )
