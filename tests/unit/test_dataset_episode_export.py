from __future__ import annotations

import gzip
import os
import json
from pathlib import Path
import wave

import pytest

from avengine.dataset import episode_export
from avengine.dataset.episode_export import (
    EpisodeExportError,
    build_episode_export_records,
    export_episode_bundle,
)


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _wav(path: Path, *, frames: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 2 * frames)
    return path


def _request(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    shared = source / "room" / "room.glb"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"shared-room")
    evidence_base = source / "evidence" / "facts.json"
    evidence_extended = source / "evidence" / "masks.npz"
    evidence_base.parent.mkdir(parents=True)
    evidence_base.write_bytes(b"base-evidence")
    evidence_extended.write_bytes(b"extended-evidence")
    audio = _wav(source / "media" / "episode.wav")
    video = source / "media" / "episode.mp4"
    video.write_bytes(b"video-master")
    plan1 = _json(source / "plan-1.json", {"schema": "plan", "clock": {
        "frame_count": 2, "frame_rate_hz": 2.0, "sample_count": 8,
        "sample_rate_hz": 16_000,
    }})
    plan2 = _json(source / "plan-2.json", {"schema": "plan-2"})
    actual1 = _json(source / "actual-1.json", {"schema": "actual", "status": "pass"})
    actual2 = _json(source / "actual-2.json", {"schema": "actual-2", "status": "pass"})
    qa1 = _json(source / "qa-1.json", {"samples": [
        {"question_id": "q1", "catalog_id": "QA-03",
         "question": "Who spoke first?", "evaluation": {
             "status": "pass", "question": "Who spoke first?",
             "answer": {"value": "blue"},
         }},
        {"question_id": "q2", "catalog_id": "QA-12",
         "question": "What did they say?", "evaluation": {
             "status": "pass", "question": "What did they say?",
             "answer": {"value": "hello"},
         }},
    ]})
    qa2 = _json(source / "qa-2.json", {"samples": [
        {"question_id": "q3", "catalog_id": "QA-02",
         "question": "What color?", "evaluation": {
             "status": "pass", "question": "What color?",
             "answer": {"value": "green"},
         }},
    ]})
    cache = source / "cache" / "rir.npy"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"numeric-cache")
    request = {
        "schema": "avengine_episode_export_request_v1",
        "evidence_retention": {"mode": "minimal", "include_extended": False},
        "model_evaluations": {
            "model_id": "local-test",
            "runtime_prefix": "/runtime",
            "api_key": "must-not-leak",
            "results": [{
                "question_id": "q1", "prediction": "blue", "score": 1.0,
                "sampling": [0, 1, 2],
            }],
        },
        "rooms": [
            {
                "room_id": "room-a",
                "shared_resources": [{"role": "visual_room", "path": str(shared)}],
                "episodes": [{
                    "episode_id": "episode-a",
                    "plan": str(plan1),
                    "actual": str(actual1),
                    "media": {"video_master": str(video), "stereo_wav": str(audio)},
                    "evidence": [
                        {"role": "facts", "path": str(evidence_base), "retention": "base"},
                        {"role": "masks", "path": str(evidence_extended), "retention": "extended"},
                    ],
                    "qa": [str(qa1)],
                    "cache": [{"role": "numeric_rir", "path": str(cache), "peak_bytes": 99}],
                }],
            },
            {
                "room_id": "room-b",
                "shared_resources": [{"role": "visual_room", "path": str(shared)}],
                "episodes": [{
                    "episode_id": "episode-b",
                    "plan": str(plan2),
                    "actual": str(actual2),
                    "media": {"video_master": str(video), "stereo_wav": str(audio)},
                    "qa": [str(qa2)],
                }],
            },
        ],
    }
    return _json(tmp_path / "request.json", request)


def test_export_keeps_shared_episode_and_qa_lineage_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        episode_export,
        "probe_video",
        lambda _path, ffprobe="ffprobe": {
            "frame_count": 2, "frame_rate_hz": 2.0, "duration_seconds": 1.0
        },
    )
    request = _request(tmp_path)
    output = tmp_path / "bundle"
    manifest = export_episode_bundle(
        request_path=request, output_root=output, gzip_json=True
    )
    assert manifest["counts"] == {
        "rooms": 2, "episodes": 2, "questions": 3, "answers": 3,
        "verified_unique_questions": 3,
        "catalog_ids": {"QA-02": 1, "QA-03": 1, "QA-12": 1},
        "models": 1,
    }
    assert (output / "manifest.json").is_file()
    assert (output / "qa/questions.jsonl.gz").is_file()
    public = [
        json.loads(line)
        for line in gzip.open(output / "qa/questions.jsonl.gz", "rt", encoding="utf-8")
    ]
    private = [
        json.loads(line)
        for line in gzip.open(output / "qa/answers.jsonl.gz", "rt", encoding="utf-8")
    ]
    assert [row["question_id"] for row in public] == ["q1", "q2", "q3"]
    assert [row["question_id"] for row in private] == ["q1", "q2", "q3"]
    assert [row["pair_id"] for row in public] == ["q1", "q2", "q3"]
    assert all("truth" not in row and "answer" not in row for row in public)
    assert [row["truth"] for row in private] == ["blue", "hello", "green"]
    assert all(row["episode_id"] in {"episode-a", "episode-b"} for row in public)
    model_rows = [
        json.loads(line)
        for line in gzip.open(
            output / "evaluation/model_results.jsonl.gz", "rt", encoding="utf-8"
        )
    ]
    assert model_rows[0]["raw_answer"] == "blue"
    assert model_rows[0]["score"] == 1.0
    assert model_rows[0]["request"]["api_key"] == "<redacted>"
    assert "sampling" not in model_rows[0].get("result_fields", {})
    assert any(row["status"] == "not_run" for row in model_rows)
    storage = json.loads(gzip.open(output / "storage.json.gz", "rt", encoding="utf-8").read())
    assert storage["shared_room_bytes"] == len(b"shared-room")
    assert storage["cache_bytes"] == len(b"numeric-cache")
    assert storage["cache_peak_bytes"] == 99
    assert storage["verified_unique_question_count"] == 3
    assert storage["amortized_bytes_per_verified_unique_question"] is not None
    episodes = [
        json.loads(line)
        for line in gzip.open(output / "episodes.jsonl.gz", "rt", encoding="utf-8")
    ]
    assert episodes[0]["evidence"]["omitted"][0]["status"] == "omitted_by_policy"
    assert episodes[0]["cache"]["lifecycle_owner"] == "audio_agent"
    assert episodes[0]["media"]["stereo_wav"]["lossless"] is True
    assert "sha256" not in episodes[0]["media"]["video_master"]
    assert "sha256" not in episodes[0]["lineage"]["actual"]["reference"]


def test_numeric_cache_role_rejects_geometry_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        episode_export,
        "probe_video",
        lambda _path, ffprobe="ffprobe": {
            "frame_count": 2, "frame_rate_hz": 2.0, "duration_seconds": 1.0
        },
    )
    request = _request(tmp_path)
    payload = json.loads(request.read_text())
    geometry = tmp_path / "rir_00.obj"
    geometry.write_bytes(b"geometry")
    payload["rooms"][0]["episodes"][0]["cache"][0]["path"] = str(geometry)
    bad_request = _json(tmp_path / "bad-request.json", payload)
    with pytest.raises(EpisodeExportError, match="geometry"):
        export_episode_bundle(request_path=bad_request, output_root=tmp_path / "bad")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(EpisodeExportError, match="overwrite"):
        export_episode_bundle(request_path=request, output_root=existing)


def test_unified_question_set_keeps_deferred_catalog_rows_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        episode_export,
        "probe_video",
        lambda _path, ffprobe="ffprobe": {
            "frame_count": 2, "frame_rate_hz": 2.0, "duration_seconds": 1.0
        },
    )
    request = _request(tmp_path)
    payload = json.loads(request.read_text())
    unified = tmp_path / "source" / "unified_questions.json"
    _json(unified, {
        "schema": "avengine_qa_unified_question_set_v1",
        "items": [{
            "schema": "avengine_qa_unified_question_v1",
            "status": "pass",
            "qa_id": "QA-01",
            "question_id": "qa_01__episode_a__target",
            "question": {"en": "Did the actor speak?", "zh": "发声了吗？"},
            "model_input": {"mcq": {"options": [
                {"value": "yes", "label_en": "yes"},
                {"value": "no", "label_en": "no"},
            ]}},
            "truth": {
                "answer_type": "closed_set",
                "value": "yes",
                "label": "yes",
                "evidence": {"event_ids": ["e1"]},
            },
            "evidence": {"event_ids": ["e1"]},
        }],
        "deferred": [{
            "qa_id": "QA-04",
            "status": "deferred",
            "code": "front_dead_zone",
            "detail": "source is inside the left/right dead zone",
        }],
    })
    payload["rooms"][0]["episodes"][0]["qa"] = [str(unified)]
    payload["rooms"] = payload["rooms"][:1]
    updated = _json(tmp_path / "unified-request.json", payload)
    records = build_episode_export_records(
        json.loads(updated.read_text()),
        request_dir=tmp_path,
        video_probe_fn=lambda _path: {
            "frame_count": 2, "frame_rate_hz": 2.0, "duration_seconds": 1.0
        },
    )
    assert len(records["model_inputs"]) == 1
    assert len(records["answers"]) == 2
    assert records["answers"][0]["truth"] == "yes"
    assert records["answers"][1]["status"] == "not_run"
    assert records["answers"][1]["truth"] is None
    assert records["qa_coverage"]["catalog_status_counts"] == {
        "QA-01": {"pass": 1}, "QA-04": {"not_run": 1}
    }


def test_required_extended_evidence_is_kept_under_minimal_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        episode_export,
        "probe_video",
        lambda _path, ffprobe="ffprobe": {
            "frame_count": 2, "frame_rate_hz": 2.0, "duration_seconds": 1.0
        },
    )
    request = _request(tmp_path)
    payload = json.loads(request.read_text())
    payload["rooms"][0]["episodes"][0]["evidence"][1]["required"] = True
    updated = _json(tmp_path / "required-request.json", payload)
    records = build_episode_export_records(
        json.loads(updated.read_text()),
        request_dir=tmp_path,
        video_probe_fn=lambda _path: {
            "frame_count": 2, "frame_rate_hz": 2.0, "duration_seconds": 1.0
        },
    )
    evidence = records["episodes"][0]["evidence"]
    assert evidence["omitted"] == []
    assert evidence["included"][-1]["required"] is True


def test_unified_forms_and_coverage_survive_private_boundary(tmp_path: Path) -> None:
    request = _request(tmp_path)
    payload = json.loads(request.read_text())
    unified = tmp_path / "source" / "forms-and-coverage.json"
    source_question_id = "qa_01__episode_a__source1"
    _json(unified, {
        "schema": "avengine_qa_unified_question_set_v1",
        "coverage": [{
            "qa_id": "QA-01",
            "question_id": source_question_id,
            "required_modalities": ["video", "binaural_audio"],
            "requirements": {
                "required_modalities": ["video", "binaural_audio"],
                "observation_window": [3, 9],
            },
        }],
        "items": [{
            "schema": "avengine_qa_unified_question_v1",
            "status": "pass",
            "qa_id": "QA-01",
            "question_id": source_question_id,
            "question": {"en": "Which actor made a sound?", "zh": "哪个个体发声？"},
            "model_input": {
                "mcq": {
                    "question_en": "Which actor made a sound?",
                    "options": [
                        {"option": "A", "label_en": "green actor"},
                        {"option": "B", "label_en": "blue actor"},
                    ],
                },
                "open": {"question_en": "Which actor made a sound?"},
            },
            "forms": {
                "mcq": {
                    "answer_type": "choice",
                    "options": [
                        {"value": "green", "label_en": "green actor"},
                        {"value": "blue", "label_en": "blue actor"},
                    ],
                    "gold": {"correct_index": 0, "value": "green"},
                },
                "open": {
                    "answer_type": "closed_set",
                    "classes": {"green": ["green actor"], "blue": ["blue actor"]},
                    "truth": "green",
                    "observation_window": [3, 9],
                },
            },
            "form_status": {
                "mcq": {"status": "pass"},
                "open": {"status": "pass"},
            },
            "truth": {
                "answer_type": "closed_set",
                "value": "green",
                "label": "green actor",
                "mcq_value": "green",
                "source": "native_engine_readbacks",
            },
            "evidence": {"observation_window": [3, 9], "query_frame": 6},
        }],
    })
    payload["rooms"] = payload["rooms"][:1]
    payload["rooms"][0]["episodes"][0]["qa"] = [str(unified)]
    payload.pop("model_evaluations", None)
    updated = _json(tmp_path / "forms-request.json", payload)
    records = build_episode_export_records(
        json.loads(updated.read_text()),
        request_dir=tmp_path,
        video_probe_fn=lambda _path: {
            "frame_count": 2, "frame_rate_hz": 2.0, "duration_seconds": 1.0
        },
    )
    public = records["model_inputs"][0]
    private = records["answers"][0]
    assert "source_question_id" not in public
    assert "evidence" not in public and "truth" not in public
    assert "source1" not in json.dumps(public, ensure_ascii=False)
    assert public["question_id"] == "episode-a__QA-01__item-0001"
    assert set(public["model_input"]) == {"mcq", "open"}
    assert public["model_input"]["mcq"]["options"][0]["label_en"] == "green actor"
    assert public["model_input"]["open"]["question_en"] == "Which actor made a sound?"
    assert private["question_id"] == public["question_id"]
    assert private["source_question_id"] == source_question_id
    assert private["forms"]["mcq"]["gold"]["correct_index"] == 0
    assert private["forms"]["open"]["answer_type"] == "closed_set"
    assert private["form_status"]["mcq"]["status"] == "pass"
    assert private["required_modalities"] == ["video", "binaural_audio"]
    assert private["requirements"]["observation_window"] == [3, 9]
    assert private["observation"]["observation_window"] == [3, 9]
    assert private["observation"]["query_frame"] == 6


def test_storage_counts_unique_inode_for_nested_and_hardlink(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "room.bin").write_bytes(b"room-payload")
    video = shared / "episode.mp4"
    video.write_bytes(b"video-payload")
    audio = _wav(shared / "episode.wav")
    plan = _json(
        shared / "plan.json",
        {
            "schema": "plan",
            "clock": {
                "frame_count": 2,
                "frame_rate_hz": 2.0,
                "sample_count": 8,
                "sample_rate_hz": 16_000,
            },
        },
    )
    actual = _json(shared / "actual.json", {"schema": "actual", "status": "pass"})
    qa = _json(
        shared / "qa.json",
        {
            "items": [
                {
                    "question_id": "q1",
                    "qa_id": "QA-01",
                    "question": "Who?",
                    "evaluation": {
                        "status": "pass",
                        "question": "Who?",
                        "answer": {"value": "blue"},
                    },
                }
            ]
        },
    )
    hardlink = tmp_path / "room-hardlink.bin"
    os.link(shared / "room.bin", hardlink)
    request = {
        "schema": "avengine_episode_export_request_v1",
        "evidence_retention": {
            "mode": "extended",
            "include_extended": True,
            "preserve_required": True,
        },
        "rooms": [
            {
                "room_id": "room-a",
                "shared_resources": [
                    {"role": "room_directory", "path": str(shared)},
                    {"role": "room_hardlink", "path": str(hardlink)},
                ],
                "episodes": [
                    {
                        "episode_id": "episode-a",
                        "plan": str(plan),
                        "actual": str(actual),
                        "media": {
                            "video_master": str(video),
                            "stereo_wav": str(audio),
                        },
                        "qa": str(qa),
                    }
                ],
            }
        ],
    }
    records = build_episode_export_records(
        request,
        request_dir=tmp_path,
        video_probe_fn=lambda _path: {
            "frame_count": 2,
            "frame_rate_hz": 2.0,
            "duration_seconds": 1.0,
        },
    )
    expected = 0
    seen = set()
    for path in shared.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key not in seen:
            seen.add(key)
            expected += stat.st_size
    assert records["storage"]["shared_room_bytes"] == expected
    assert records["storage"]["permanent_bytes"] == 0
    assert records["storage"]["persistent_bytes_including_shared_room"] == expected
    assert records["storage"]["directory_file_overlap_counted_once"] is True
    assert records["storage"]["hardlink_payload_counted_once"] is True


def test_cache_lifecycle_allows_cleaned_payload_missing(tmp_path: Path) -> None:
    request = _request(tmp_path)
    payload = json.loads(request.read_text())
    payload["rooms"][0]["episodes"][0].pop("cache", None)
    cache_root = tmp_path / "cleaned-cache"
    cache_root.mkdir()
    metadata = cache_root / "manifest.json"
    metadata.write_text('{"status":"cleaned"}\n', encoding="utf-8")
    missing_numeric = cache_root / "sequence.npz"
    cleanup = _json(
        tmp_path / "cleanup_record.json",
        {
            "schema": "avengine_dynamic_rir_cache_cleanup_v2",
            "status": "pass",
            "cache_root": str(cache_root),
            "numeric_payload_files": [str(missing_numeric)],
            "numeric_payload_lifecycle": "numeric_payload_cleaned",
            "storage": {
                "cache_bytes_before": 100,
                "numeric_payload_bytes_before": 90,
                "numeric_payload_bytes_cleared": 90,
                "peak_bytes": 100,
                "preserved_metadata_bytes": metadata.stat().st_size,
            },
        },
    )
    payload["cache_lifecycle_records"] = [
        {"role": "numeric_rir_cleanup", "path": str(cleanup)}
    ]
    updated = _json(tmp_path / "lifecycle-request.json", payload)
    records = build_episode_export_records(
        json.loads(updated.read_text()),
        request_dir=tmp_path,
        video_probe_fn=lambda _path: {
            "frame_count": 2,
            "frame_rate_hz": 2.0,
            "duration_seconds": 1.0,
        },
    )
    lifecycle = records["storage"]["cache_lifecycle"]
    assert lifecycle["status"] == "pass"
    assert lifecycle["cache_peak_bytes"] == 100
    assert lifecycle["numeric_payload_bytes_current"] == 0
    assert lifecycle["metadata_bytes_current"] == metadata.stat().st_size
    assert lifecycle["records"][0]["numeric_payload_files_present"] == 0
    assert lifecycle["records"][0]["numeric_payload_lifecycle"] == "numeric_payload_cleaned"
    assert records["storage"]["cache_bytes"] == 0
    assert records["storage"]["numeric_rir_cache_status"] == "generated_cache"


def test_batch_evidence_manifest_and_inode_dedup(tmp_path: Path) -> None:
    request_path = _request(tmp_path)
    payload = json.loads(request_path.read_text())
    probe = lambda _path: {
        "frame_count": 2,
        "frame_rate_hz": 2.0,
        "duration_seconds": 1.0,
    }
    baseline = build_episode_export_records(
        payload, request_dir=tmp_path, video_probe_fn=probe
    )
    batch_file = tmp_path / "batch-diagnostic.json"
    batch_file.write_bytes(b"batch-diagnostic")
    batch_alias = tmp_path / "batch-diagnostic-hardlink.json"
    os.link(batch_file, batch_alias)
    payload["batch_evidence"] = [
        {"role": "batch_diagnostic", "path": str(batch_file), "required": True},
        {"role": "batch_diagnostic_alias", "path": str(batch_alias), "required": True},
    ]
    updated = _json(tmp_path / "batch-request.json", payload)
    records = build_episode_export_records(
        json.loads(updated.read_text()),
        request_dir=tmp_path,
        video_probe_fn=probe,
    )
    assert len(records["batch_evidence"]) == 2
    assert all(ref["required"] is True for ref in records["batch_evidence"])
    assert records["storage"]["permanent_bytes"] == (
        baseline["storage"]["permanent_bytes"] + batch_file.stat().st_size
    )
    output = tmp_path / "batch-bundle"
    manifest = export_episode_bundle(
        request_path=updated, output_root=output, gzip_json=False,
        video_probe_fn=probe,
    )
    assert len(manifest["batch_evidence"]) == 2
    assert manifest["batch_evidence"][0]["required"] is True
