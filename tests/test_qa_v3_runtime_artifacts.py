"""Tests for the declarative QA runtime artifact compatibility layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.qa.runtime_artifacts import (
    RuntimeArtifactError,
    load_runtime_artifacts,
    registered_pixel_consumer,
)


def _write(path: Path, value: object = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _point(tmp_path: Path) -> Path:
    point = tmp_path / "point"
    point.mkdir()
    for name in (
        "actor_selection.json",
        "timeline.json",
        "actor_selection_gateB.json",
        "timeline_gateB.json",
        "timeline_segment2.json",
    ):
        _write(point / name)
    return point


def test_legacy_candidate_gets_main_segment_and_release_compatibility(tmp_path: Path) -> None:
    point = _point(tmp_path)
    result = load_runtime_artifacts(point, {"profile_id": "card7"})
    assert result["legacy_visual_compatibility"] is True
    assert result["legacy_release_compatibility"] is True
    assert [row["id"] for row in result["visual_variants"]] == ["main"]
    assert [row["id"] for row in result["segments"]] == ["segment1"]
    assert result["visual_variants"][0]["actor_selection"] == point / "actor_selection.json"
    assert result["segments"][0]["timeline"] == point / "timeline.json"
    assert result["release_media"][0]["release"] is True
    assert result["release_media"][0]["audio_variant"] == "main"


def test_extended_descriptions_keep_extra_fields_and_resolve_segment2(tmp_path: Path) -> None:
    point = _point(tmp_path)
    fact = {
        "visual_variants": [
            {
                "id": "main",
                "kind": "qa_v3_current_apartment_visual",
                "actor_selection": "actor_selection.json",
                "timeline": "timeline.json",
                "release": True,
                "future_field": {"kept": True},
            },
            {
                "id": "gateB",
                "kind": "qa_v3_current_apartment_visual",
                "actor_selection": "actor_selection_gateB.json",
                "timeline": "timeline_gateB.json",
                "release": False,
            },
        ],
        "segments": [
            {"id": "segment1", "variant": "main"},
            {
                "id": "segment2",
                "variant": "gateB",
                "actor_selection": "actor_selection_gateB.json",
                "timeline": "timeline_segment2.json",
            },
        ],
        "pixel_evidence": [{"id": "main", "kind": "qa_v3_extended_pixel"}],
        "release_media": [
            {"id": "main", "variant": "main", "segment": "segment1", "release": True},
            {"id": "gateB", "variant": "gateB", "segment": "segment2", "release": False},
        ],
    }
    result = load_runtime_artifacts(point, fact)
    assert result["legacy_visual_compatibility"] is False
    assert result["visual_variants"][0]["future_field"] == {"kept": True}
    assert result["segments"][1]["timeline"] == point / "timeline_segment2.json"
    assert result["segments"][1]["variant"] == "gateB"
    assert result["pixel_evidence"][0]["pixel_truth"] if "pixel_truth" in result["pixel_evidence"][0] else True
    assert [row["id"] for row in result["release_media"]] == ["main", "gateB"]


def test_mapping_form_and_duplicate_ids_are_supported_and_checked(tmp_path: Path) -> None:
    point = _point(tmp_path)
    result = load_runtime_artifacts(
        point,
        {
            "visual_variants": {
                "main": {
                    "actor_selection": "actor_selection.json",
                    "timeline": "timeline.json",
                },
            },
            "segments": {"segment1": {"variant": "main"}},
        },
    )
    assert result["visual_variants"][0]["id"] == "main"
    assert result["segments"][0]["id"] == "segment1"
    with pytest.raises(RuntimeArtifactError, match="duplicate"):
        load_runtime_artifacts(
            point,
            {
                "visual_variants": [
                    {"id": "main", "actor_selection": "actor_selection.json", "timeline": "timeline.json"},
                    {"id": "main", "actor_selection": "actor_selection_gateB.json", "timeline": "timeline_gateB.json"},
                ],
            },
        )


def test_visual_paths_cannot_escape_candidate_and_unknown_pixel_kind_is_not_executable(
    tmp_path: Path,
) -> None:
    point = _point(tmp_path)
    with pytest.raises(RuntimeArtifactError, match="escapes"):
        load_runtime_artifacts(
            point,
            {
                "visual_variants": [{
                    "id": "main",
                    "actor_selection": "../outside.json",
                    "timeline": "timeline.json",
                }],
            },
        )
    result = load_runtime_artifacts(
        point,
        {"pixel_evidence": [{"id": "future", "kind": "future_consumer"}]},
    )
    assert result["pixel_evidence"][0]["kind"] == "future_consumer"
    assert registered_pixel_consumer("qa_v3_extended_pixel").name == "join_qa_v3_extended_pixel.py"
    assert registered_pixel_consumer("future_consumer") is None


def test_capture_resolver_consumes_declared_segment(tmp_path: Path) -> None:
    from tools.qa.run_qa_v3_capture_batch import resolve_capture_inputs

    point = _point(tmp_path)
    fact = {
        "visual_variants": [{
            "id": "main",
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
        }],
        "segments": [{
            "id": "later",
            "variant": "main",
            "timeline": "timeline_segment2.json",
        }],
    }
    (point / "fact_record.json").write_text(json.dumps(fact), encoding="utf-8")
    selection, timeline, description = resolve_capture_inputs(
        point, descriptor_id="later", descriptor_kind="segment"
    )
    assert selection == point / "actor_selection.json"
    assert timeline == point / "timeline_segment2.json"
    assert description == point / "fact_record.json"


def test_release_media_references_must_exist(tmp_path: Path) -> None:
    point = _point(tmp_path)
    with pytest.raises(RuntimeArtifactError, match="unknown segment"):
        load_runtime_artifacts(point, {
            "release_media": [{
                "id": "bad",
                "variant": "main",
                "segment": "missing",
                "release": True,
            }]
        })


def test_extended_descriptor_generation_uses_profile_capabilities(tmp_path: Path) -> None:
    from tools.qa.design_qa_v3_extended_profile import _runtime_descriptions

    point = tmp_path / "candidate"
    point.mkdir()
    for name in (
        "timeline.json",
        "timeline_segment2.json",
        "actor_selection.json",
        "actor_selection_gateB.json",
        "timeline_gateB.json",
    ):
        _write(point / name)
    result = _runtime_descriptions(
        {
            "segment_count": 2,
            "pixel_consumer_kind": "qa_v3_extended_pixel",
            "runtime_consumer_status": "pending_cross_segment_consumer",
        },
        point,
    )
    assert [row["id"] for row in result["segments"]] == ["segment1", "segment2"]
    assert [row["segment"] for row in result["release_media"]] == ["segment1", "segment2"]
    assert all(row["release"] is True for row in result["release_media"])
    assert result["release_media"][0]["audio_variant"] == "main"
    assert result["release_media"][1]["audio_variant"] is None
    assert result["release_media"][1]["status"] == "pending_audio_consumer"
    assert result["pixel_evidence"][0]["kind"] == "qa_v3_extended_pixel"
    assert result["runtime_consumer_status"] == "pending_cross_segment_consumer"



def test_release_media_rejects_malformed_audio_variant(tmp_path: Path):
    point = _point(tmp_path)
    fact = {
        "release_media": [{
            "id": "segment1",
            "variant": "main",
            "segment": "segment1",
            "kind": "qa_v3_review_clip",
            "release": True,
            "audio_variant": {"unexpected": "mapping"},
        }]
    }
    with pytest.raises(RuntimeArtifactError, match="audio_variant"):
        load_runtime_artifacts(point, fact)



def test_declared_release_without_audio_variant_does_not_infer_main(
    tmp_path: Path,
) -> None:
    point = _point(tmp_path)
    fact = {
        "visual_variants": [{
            "id": "main",
            "kind": "qa_v3_current_apartment_visual",
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
        }],
        "segments": [
            {"id": "segment1", "variant": "main"},
            {
                "id": "segment2",
                "variant": "main",
                "timeline": "timeline_segment2.json",
            },
        ],
        "release_media": [{
            "id": "segment2",
            "variant": "main",
            "segment": "segment2",
            "kind": "qa_v3_review_clip",
            "release": True,
        }],
    }

    result = load_runtime_artifacts(point, fact)

    assert result["release_media"][0]["audio_variant"] is None


def test_pixel_producer_binding_frames_must_be_inside_timeline_count(tmp_path: Path) -> None:
    point = _point(tmp_path)
    _write(point / "timeline.json", {"render": {"frame_count": 75}, "frames": [{}] * 75})
    with pytest.raises(RuntimeArtifactError, match="0 <= frame < frame_count"):
        load_runtime_artifacts(point, {
            "pixel_producers": [{
                "id": "main",
                "kind": "qa_v3_timeline_native_pixel",
                "actor_selection": "actor_selection.json",
                "timeline": "timeline.json",
                "binding_frames": [75],
            }]
        })
    result = load_runtime_artifacts(point, {
        "pixel_producers": [{
            "id": "main",
            "kind": "qa_v3_timeline_native_pixel",
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
            "binding_frames": [0, 74],
        }]
    })
    assert result["pixel_producers"][0]["binding_frames"] == [0, 74]
    assert result["pixel_producers"][0]["frame_count"] == 75
