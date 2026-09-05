from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest

from avengine.qa.dataset_export import (
    DatasetExportError,
    build_dataset_records,
    export_dataset,
)


def _wav(path: Path, *, channels: int, frames: int = 80, rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * channels * frames)


def _fixture(tmp_path: Path):
    root = tmp_path / "pipeline"
    point = root / "pairs/room-a/card7/audio/point-1"
    binaural = point / "audio/binaural/mixture.wav"
    foa = point / "audio/foa/mixture.wav"
    _wav(binaural, channels=2)
    _wav(foa, channels=4)
    receipt = {
        "execution_variant": "main",
        "audio": {
            "layouts": ["binaural", "ambisonics"],
            "by_layout": {
                "binaural": {
                    "layout_type": "binaural",
                    "output_directory": "binaural",
                    "channel_count": 2,
                    "channel_labels": ["left", "right"],
                    "sample_rate_hz": 16_000,
                    "sample_count": 80,
                },
                "ambisonics": {
                    "layout_type": "ambisonics",
                    "output_directory": "foa",
                    "channel_count": 4,
                    "channel_labels": ["W", "Y", "Z", "X"],
                    "sample_rate_hz": 16_000,
                    "sample_count": 80,
                },
            },
        },
    }
    (point / "research_receipt.json").write_text(json.dumps(receipt))
    video = root / "pairs/room-a/card7/media/point-1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture-video")
    common = {
        "group_id": "group-1",
        "point_id": "point-1",
        "episode_id": "episode-1",
        "profile_id": "card7",
        "audio": str(binaural),
        "video": str(video),
        "scene_id": "room-a",
        "media_clock": {
            "frame_count": 2,
            "frame_rate_hz": 400.0,
            "clip_seconds": 0.005,
            "sample_rate_hz": 16_000,
            "sample_count": 80,
        },
        "task_type": "classification",
    }
    items = [
        {
            **common,
            "question_id": "q-mcq",
            "form": "mcq",
            "question": "Which source?",
            "options": ["first", "second"],
            "truth": "first",
        },
        {
            **common,
            "question_id": "q-open",
            "form": "open",
            "question": "Name the source.",
            "options": [],
            "truth": "first",
            "certification_policy": "strict_full_credit_only",
        },
    ]
    released = root / "questions/released_items.json"
    released.parent.mkdir(parents=True)
    released.write_text(json.dumps(items))
    (root / "pipeline_manifest.json").write_text(json.dumps({
        "source": {"repository": "/server/AVEngine", "git_commit": "fixture"},
        "failures": [],
        "shortfalls": [],
    }))
    probe = lambda _path: {
        "frame_count": 2,
        "frame_rate_hz": 400.0,
        "duration_seconds": 0.005,
    }
    return root, released, items, probe


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_export_separates_answers_and_declares_all_receipt_layouts(tmp_path: Path):
    root, released, _items, probe = _fixture(tmp_path)
    output = tmp_path / "export"
    manifest = export_dataset(
        pipeline_root=root,
        released_items_path=released,
        output_root=output,
        layouts=["binaural,ambisonics"],
        video_probe=probe,
    )
    public = [json.loads(line) for line in (output / "public/questions.jsonl").read_text().splitlines()]
    private = [json.loads(line) for line in (output / "private/answers.jsonl").read_text().splitlines()]
    assert [row["question_id"] for row in public] == ["q-mcq", "q-open"]
    assert [row["question_id"] for row in public] == [row["question_id"] for row in private]
    assert set(public[0]["audio_by_layout"]) == {"binaural", "ambisonics"}
    assert public[0]["audio_by_layout"]["binaural"]["channel_count"] == 2
    assert public[0]["audio_by_layout"]["ambisonics"]["channel_count"] == 4
    assert public[0]["audio_by_layout"]["ambisonics"]["channel_labels"] == ["W", "Y", "Z", "X"]
    assert private[0]["truth"] == "first"
    assert private[1]["scoring"]["certification_policy"] == "strict_full_credit_only"
    assert not ({"truth", "answer", "gold", "answer_key", "fact", "facts"} & set(_all_keys(public)))
    assert manifest["counts"]["questions"] == 2
    assert manifest["layouts"]["observed"] == ["ambisonics", "binaural"]


def test_build_rejects_duplicate_ids_and_missing_requested_layout(tmp_path: Path):
    root, _released, items, probe = _fixture(tmp_path)
    with pytest.raises(DatasetExportError, match="duplicate question_id"):
        build_dataset_records(
            [items[0], items[0]], pipeline_root=root, video_probe=probe
        )
    with pytest.raises(DatasetExportError, match="missing requested layouts"):
        build_dataset_records(
            items, pipeline_root=root, layouts=["unknown"], video_probe=probe
        )


def test_export_rejects_audio_clock_mismatch_and_no_clobber(tmp_path: Path):
    root, released, items, probe = _fixture(tmp_path)
    items[0]["media_clock"]["sample_count"] = 81
    released.write_text(json.dumps(items))
    with pytest.raises(DatasetExportError, match="clock differs"):
        export_dataset(
            pipeline_root=root,
            released_items_path=released,
            output_root=tmp_path / "bad",
            video_probe=probe,
        )
    assert not (tmp_path / "bad").exists()

    root, released, _items, probe = _fixture(tmp_path / "second")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(DatasetExportError, match="refusing to overwrite"):
        export_dataset(
            pipeline_root=root,
            released_items_path=released,
            output_root=existing,
            video_probe=probe,
        )
