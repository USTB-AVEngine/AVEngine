from __future__ import annotations

import json
import wave
from pathlib import Path

from tools.qa.adapt_four_speaker_research_qa import build


ACTORS = [
    ("speaker_blue", "blue", "s_blue", "Blue sentence."),
    ("speaker_pink", "pink", "s_pink", "Pink sentence."),
    ("speaker_green", "green", "s_green", "Green sentence."),
    ("speaker_white", "white", "s_white", "White sentence."),
]


def _write_wav(path: Path, count: int = 1600) -> None:
    payload = (b"\x01\x00" * count)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(payload)


def _inputs(tmp_path: Path, *, with_audio: bool = True) -> tuple[Path, Path, Path | None, Path | None]:
    bindings = []
    for actor, color, sound, transcript in ACTORS:
        wav = tmp_path / f"{sound}.wav"
        _write_wav(wav)
        bindings.append(
            {
                "actor_id": actor,
                "sound_asset_id": sound,
                "speaker_id": actor,
                "gender": "M",
                "color": color,
                "transcript": transcript,
                "path": str(wav),
            }
        )
    binding_path = tmp_path / "voice_binding.json"
    binding_path.write_text(json.dumps(bindings), encoding="utf-8")

    frame_count = 30
    readback = {
        "clock": {
            "frame_count": frame_count,
            "frame_rate_hz": 15.0,
            "sample_rate_hz": 16000,
            "sample_count": 16000,
        },
        "camera": [
            {"location_cm": [0.0, 120.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0]}
            for _ in range(frame_count)
        ],
        "emitters": {
            actor: [
                {"location_cm": [float(index), 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0]}
                for index in range(frame_count)
            ]
            for actor, *_ in ACTORS
        },
    }
    readback_path = tmp_path / "frame_readbacks.json"
    readback_path.write_text(json.dumps(readback), encoding="utf-8")

    if not with_audio:
        return readback_path, binding_path, None, None

    audio_events = []
    report_events = []
    cursor = 100
    for index, (actor, color, sound, transcript) in enumerate(ACTORS):
        start = cursor
        end = start + 1600
        event_id = f"event_{index}"
        audio_events.append(
            {
                "event_id": event_id,
                "source_endpoint_id": actor,
                "sound_asset_id": sound,
                "start_sample": start,
                "end_sample_exclusive": end,
            }
        )
        report_events.append(
            {
                "event_id": event_id,
                "actor_id": actor,
                "sound_asset_id": sound,
                "transcript": transcript,
                "start_sample": start,
                "end_sample_exclusive": end,
                "pcm_output_nonzero_interval": [start, end],
            }
        )
        cursor = end + 100
    audio_path = tmp_path / "audio_program.json"
    audio_path.write_text(json.dumps({"events": audio_events}), encoding="utf-8")
    report_path = tmp_path / "research_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "research",
                "complete_sentences_preserved": True,
                "events": report_events,
            }
        ),
        encoding="utf-8",
    )
    return readback_path, binding_path, audio_path, report_path


def test_missing_native_audio_is_not_run(tmp_path: Path) -> None:
    readback, binding, audio, report = _inputs(tmp_path, with_audio=False)
    result = build(
        frame_readbacks=readback,
        voice_binding=binding,
        audio_program=audio,
        research_report=report,
        output=tmp_path / "out",
    )
    assert result["status"] == "not_run"
    validation = json.loads((tmp_path / "out" / "input_validation_report.json").read_text())
    assert validation["voice_binding"]["status"] == "pass"
    assert validation["frame_readbacks"]["status"] == "pass"
    assert validation["audio"]["status"] == "not_run"


def test_reuses_question_spec_for_order_card13_and_card14(tmp_path: Path) -> None:
    readback, binding, audio, report = _inputs(tmp_path)
    result = build(
        frame_readbacks=readback,
        voice_binding=binding,
        audio_program=audio,
        research_report=report,
        output=tmp_path / "out",
        require_all_speaker_visible=False,
    )
    assert result["status"] == "research_only"
    cards = {item["card"]: item for item in result["samples"]}
    assert cards["speaker_order"]["evaluation"]["status"] == "pass"
    assert cards["card13"]["evaluation"]["status"] == "pass"
    assert cards["card14"]["evaluation"]["status"] == "pass"
    assert cards["card13"]["evaluation"]["answer"]["value"] == "Blue sentence."
    assert cards["card14"]["evaluation"]["answer"]["value"] == "blue"


def test_visibility_condition_defers_only_card13_card14(tmp_path: Path) -> None:
    readback, binding, audio, report = _inputs(tmp_path)
    truth = {
        "per_instance": {
            actor: {
                "frames": [
                    {
                        "frame_index": 0,
                        "state": "visible_clear" if index == 0 else "out_of_view",
                    }
                ]
            }
            for index, (actor, *_rest) in enumerate(ACTORS)
        }
    }
    truth_path = tmp_path / "pixel_visibility_truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    result = build(
        frame_readbacks=readback,
        voice_binding=binding,
        audio_program=audio,
        research_report=report,
        pixel_visibility_truth=truth_path,
        output=tmp_path / "out",
        require_all_speaker_visible=True,
    )
    assert result["status"] == "research_only"
    assert any(item["card"] == "speaker_order" for item in result["samples"])
    assert {item["card"] for item in result["deferred_samples"]} == {"card13", "card14"}
