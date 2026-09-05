from __future__ import annotations

import json
from pathlib import Path
import sys
import wave

from avengine.dataset.model_evaluation import (
    build_spatial_omni_benchmark_command,
    prepare_qwen25_omni_pilot_gold,
    prepare_qwen25_omni_pilot_inputs,
    prepare_whisper_review_request,
    prepare_spatial_omni_qa_root,
)


def _wav(path: Path, channels: int) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * channels * 4)
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_stereo_export_is_explicitly_not_run(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "stereo.wav", 2)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    inputs = _jsonl(tmp_path / "questions.jsonl", [{
        "question_id": "q1", "question": "Who?", "audio_path": str(audio),
        "video_path": str(video),
    }])
    answers = _jsonl(tmp_path / "answers.jsonl", [{
        "question_id": "q1", "truth": "blue",
    }])
    manifest = prepare_spatial_omni_qa_root(
        model_inputs_path=inputs,
        answers_path=answers,
        output_root=tmp_path / "runtime",
    )
    assert manifest["status"] == "not_run"
    assert manifest["unsupported_question_ids"] == ["q1"]
    assert "4-channel FOA" in manifest["reason"]
    assert not (tmp_path / "runtime" / "qa" / "test.jsonl").exists()


def test_foa_join_and_benchmark_command_are_private_and_explicit(
    tmp_path: Path,
) -> None:
    audio = _wav(tmp_path / "foa.wav", 4)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    inputs = _jsonl(tmp_path / "questions.jsonl", [{
        "question_id": "q1", "episode_id": "e1", "question": "Who?",
        "audio_path": str(audio), "video_path": str(video),
    }])
    answers = _jsonl(tmp_path / "answers.jsonl", [{
        "question_id": "q1", "truth": "blue",
    }])
    manifest = prepare_spatial_omni_qa_root(
        model_inputs_path=inputs,
        answers_path=answers,
        output_root=tmp_path / "runtime",
    )
    assert manifest["status"] == "pass"
    joined = json.loads(
        (tmp_path / "runtime" / "qa" / "test.jsonl").read_text().strip()
    )
    assert joined["answer"] == "blue"
    assert joined["audio_path"] == str(audio)
    assert manifest["private_runtime_input"] is True

    runtime = tmp_path / "runtime-prefix"
    (runtime / "scripts").mkdir(parents=True)
    (runtime / "bin").mkdir()
    (runtime / "scripts" / "batch_bench_so_qa.py").write_text("", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    command = build_spatial_omni_benchmark_command(
        runtime_prefix=runtime,
        checkpoint_path=checkpoint,
        qa_root=tmp_path / "runtime" / "qa",
        output_dir=tmp_path / "bench",
        device="cuda:2",
        max_samples=1,
        python_executable=sys.executable,
    )
    assert command[0] == str(Path(sys.executable).resolve())
    assert "--checkpoint-path" in command
    assert command[-2:] == ["--max-samples", "1"]


def test_qwen25_pilot_preserves_stereo_and_emits_answer_free_variants(
    tmp_path: Path, monkeypatch
) -> None:
    audio = _wav(tmp_path / "stereo.wav", 2)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    inputs = _jsonl(tmp_path / "questions.jsonl", [{
        "question_id": "q1", "episode_id": "e1", "question_en": "Who?",
        "options": ["blue", "green"], "audio_path": str(audio),
        "video_path": str(video),
    }, {
        "question_id": "q-open", "question": "What did they say?",
        "options": [], "audio_path": str(audio), "video_path": str(video),
    }])

    def fake_mux(*, destination, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"muxed-full-av")

    monkeypatch.setattr(
        "avengine.dataset.model_evaluation._mux_stereo_video", fake_mux
    )
    manifest = prepare_qwen25_omni_pilot_inputs(
        model_inputs_path=inputs,
        output_root=tmp_path / "pilot",
    )
    assert manifest["status"] == "ready"
    assert manifest["counts"] == {
        "source_questions": 2,
        "runnable_questions": 1,
        "model_inputs": 3,
        "skipped_questions": 1,
    }
    payload = json.loads(
        (tmp_path / "pilot" / "model_inputs.json").read_text()
    )
    assert payload["input_count"] == 3
    assert all("answer" not in row and "truth" not in row for row in payload["items"])
    assert all(row["sample_id"] == "q1" for row in payload["items"])
    assert (tmp_path / "pilot" / "media" / "audio_only" / "00000-q1.wav").stat().st_ino == audio.stat().st_ino
    assert "downmix_to_mono_16k" in manifest["audio_contract"]["model_preprocessing"]


def test_whisper_request_keeps_only_windowed_speech_and_onset_items(
    tmp_path: Path,
) -> None:
    audio = _wav(tmp_path / "stereo.wav", 2)
    inputs = _jsonl(tmp_path / "questions.jsonl", [
        {
            "question_id": "q12",
            "question_en": "What did they say?",
            "audio_path": str(audio),
            "input": {"media_clock": {
                "sample_rate_hz": 16_000, "sample_count": 16_000,
                "clip_seconds": 1.0,
            }},
        },
        {
            "question_id": "q19",
            "question_en": "When did speech start?",
            "audio_path": str(audio),
            "input": {"media_clock": {
                "sample_rate_hz": 16_000, "sample_count": 16_000,
                "clip_seconds": 1.0,
            }},
        },
        {
            "question_id": "q04",
            "question_en": "Was it left or right?",
            "audio_path": str(audio),
            "input": {"media_clock": {
                "sample_rate_hz": 16_000, "sample_count": 16_000,
                "clip_seconds": 1.0,
            }},
        },
    ])
    answers = _jsonl(tmp_path / "answers.jsonl", [
        {
            "question_id": "q12", "status": "pass",
            "answer_type": "transcript_wer", "truth": "hello",
            "evidence": {"event": {"start_s": 0.2, "end_s": 0.6}},
        },
        {
            "question_id": "q19", "status": "pass",
            "answer_type": "time_s", "truth": 0.2,
            "evidence": {"first_event": {"start_s": 0.2, "end_s": 0.6}},
        },
        {
            "question_id": "q04", "status": "pass",
            "answer_type": "closed_set", "truth": "left", "evidence": {},
        },
    ])
    model = tmp_path / "small.en.pt"
    model.write_bytes(b"model")
    manifest = prepare_whisper_review_request(
        model_inputs_path=inputs,
        answers_path=answers,
        output_path=tmp_path / "whisper_request.json",
        model_path=model,
    )
    assert manifest["status"] == "ready"
    request = json.loads((tmp_path / "whisper_request.json").read_text())
    assert [item["id"] for item in request["items"]] == ["q12", "q19"]
    assert request["items"][0]["window_seconds"] == [0.2, 0.6]
    assert request["items"][1]["window_seconds"] == [0.0, 1.0]
    assert all("truth" not in item and "answer" not in item for item in request["items"])
    assert any(
        row["question_id"] == "q04"
        and row["reason"] == "spatial_or_non_speech_question"
        for row in manifest["skipped_questions"]
    )


def test_qwen25_gold_uses_private_truth_without_touching_inputs(
    tmp_path: Path,
) -> None:
    inputs = _jsonl(tmp_path / "questions.jsonl", [{
        "question_id": "q1",
        "question": "Which color?",
        "options": ["blue", "green"],
        "audio_path": str(_wav(tmp_path / "stereo.wav", 2)),
        "video_path": str(tmp_path / "video.mp4"),
    }])
    Path(tmp_path / "video.mp4").write_bytes(b"video")
    answers = _jsonl(tmp_path / "answers.jsonl", [{
        "question_id": "q1", "status": "pass", "truth": "blue",
        "truth_label": "blue", "required_modalities": ["video", "binaural_audio"],
    }])
    output = tmp_path / "gold.json"
    manifest = prepare_qwen25_omni_pilot_gold(
        model_inputs_path=inputs, answers_path=answers, output_path=output
    )
    assert manifest["status"] == "ready"
    gold = json.loads(output.read_text())
    assert gold["items"][0]["answer_index"] == 0
    assert gold["items"][0]["answer_value"] == "blue"
    assert "truth" not in json.loads(inputs.read_text().splitlines()[0])


def test_qwen25_gold_prefers_private_mcq_correct_index(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "stereo.wav", 2)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    inputs = _jsonl(tmp_path / "questions.jsonl", [{
        "question_id": "qa_22__episode__whole_clip",
        "question": "How many entities and speakers?",
        "options": ["2 entities, 2 speaking", "2 entities, 1 speaking"],
        "audio_path": str(audio),
        "video_path": str(video),
    }])
    answers = _jsonl(tmp_path / "answers.jsonl", [{
        "question_id": "qa_22__episode__whole_clip",
        "status": "pass",
        "truth": [2, 2],
        "truth_label": "2, 2",
        "forms": {
            "mcq": {
                "options": [
                    {"value": "2|2", "label_en": "2 entities, 2 speaking"},
                    {"value": "2|1", "label_en": "2 entities, 1 speaking"},
                ],
                "gold": {"correct_index": 0, "value": "2|2"},
            },
        },
    }])
    output = tmp_path / "gold.json"
    manifest = prepare_qwen25_omni_pilot_gold(
        model_inputs_path=inputs,
        answers_path=answers,
        output_path=output,
    )
    assert manifest["counts"]["gold_questions"] == 1
    gold = json.loads(output.read_text())
    assert gold["items"][0]["answer_index"] == 0
    assert gold["items"][0]["answer_value"] == "2 entities, 2 speaking"
    assert gold["items"][0]["scoring_source"] == "private_forms.mcq.gold"
