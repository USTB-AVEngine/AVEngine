from __future__ import annotations

from typing import Any, Mapping

import pytest

from avengine.m6.exporter_interface import (
    ExporterInterfaceError,
    ReadOnlyEvidenceBundle,
    TaskExporter,
)


def _index() -> dict[str, Any]:
    return {
        "evidence_bundle_id": "controlled_canary_run_01",
        "room_id": "blender_custom_two_zone_v1",
        "entity_ids": ["beagle_0", "beagle_1"],
        "source_endpoint_ids": ["beagle_0_muzzle", "beagle_1_muzzle"],
        "event_ids": ["bark_event_0"],
        "artifacts": [
            {"role": "source_manifest", "uri": "repo://run/source_manifest.json", "sha256": "1" * 64},
            {"role": "timeline", "uri": "repo://run/timeline.json", "sha256": "2" * 64},
        ],
    }


def test_evidence_view_is_read_only_and_does_not_require_dense_motion_or_qa() -> None:
    source = _index()
    view = ReadOnlyEvidenceBundle(source)
    source["entity_ids"].append("late_mutation")
    assert view["entity_ids"] == ("beagle_0", "beagle_1")
    assert "world_velocity" not in view
    assert "question" not in view
    with pytest.raises(TypeError):
        view["room_id"] = "other"  # type: ignore[index]
    with pytest.raises(AttributeError):
        view["entity_ids"].append("other")


def test_artifact_lookup_exposes_only_role_uri_and_hash() -> None:
    view = ReadOnlyEvidenceBundle(_index())
    artifact = view.artifact("timeline")
    assert artifact.uri == "repo://run/timeline.json"
    assert artifact.sha256 == "2" * 64
    with pytest.raises(KeyError):
        view.artifact("selecttsl_labels")


def test_future_exporter_protocol_can_read_evidence_without_simulator() -> None:
    class IdentityExporter:
        exporter_id = "identity_exporter"
        exporter_version = "v1"

        def export(
            self,
            evidence: ReadOnlyEvidenceBundle,
            specification: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            return {
                "source_endpoint_ids": evidence["source_endpoint_ids"],
                "timeline_sha256": evidence.artifact("timeline").sha256,
                "specification": dict(specification),
            }

    exporter = IdentityExporter()
    assert isinstance(exporter, TaskExporter)
    result = exporter.export(ReadOnlyEvidenceBundle(_index()), {"format": "future_task"})
    assert result["timeline_sha256"] == "2" * 64
    assert result["source_endpoint_ids"] == (
        "beagle_0_muzzle",
        "beagle_1_muzzle",
    )


def test_evidence_index_requires_canonical_ids_and_hashes() -> None:
    value = _index()
    value["source_endpoint_ids"] = list(reversed(value["source_endpoint_ids"]))
    with pytest.raises(ExporterInterfaceError, match="canonical stable IDs"):
        ReadOnlyEvidenceBundle(value)

