"""Ordinary identity checks for one current Kujiale full-home room input.

The check prevents independently loaded metadata, source and visual profiles
from silently naming different UE maps or USD stages.  Room type is retained
only as annotation and is not part of camera-candidate generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from avengine.optional_backends.interioragent_kujiale import (
    PROFILE_SCHEMA as VISUAL_PROFILE_SCHEMA,
)
from avengine.optional_backends.residential_episode import (
    PROFILE_SCHEMA as SOURCE_PROFILE_SCHEMA,
    SCENE_METADATA_SCHEMA,
)


def _required_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{owner} must be a non-empty string")
    return value.strip()


def validate_kujiale_scene_binding(
    *,
    metadata: Mapping[str, Any],
    source_profile: Mapping[str, Any],
    visual_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check ordinary identities across one room and current UE scene."""

    if metadata.get("schema") != SCENE_METADATA_SCHEMA:
        raise RuntimeError(f"scene metadata schema must be {SCENE_METADATA_SCHEMA}")
    if source_profile.get("schema") != SOURCE_PROFILE_SCHEMA:
        raise RuntimeError(f"source profile schema must be {SOURCE_PROFILE_SCHEMA}")
    if visual_profile.get("schema_version") != VISUAL_PROFILE_SCHEMA:
        raise RuntimeError(f"visual profile schema must be {VISUAL_PROFILE_SCHEMA}")

    dataset_ids = {
        _required_string(metadata.get("dataset_id"), owner="metadata.dataset_id"),
        _required_string(
            source_profile.get("dataset_id"), owner="source_profile.dataset_id"
        ),
        _required_string(
            visual_profile.get("dataset_id"), owner="visual_profile.dataset_id"
        ),
    }
    if len(dataset_ids) != 1:
        raise RuntimeError(
            "metadata, source profile and visual profile dataset_id differ"
        )
    scene_ids = {
        _required_string(metadata.get("scene_id"), owner="metadata.scene_id"),
        _required_string(
            source_profile.get("scene_id"), owner="source_profile.scene_id"
        ),
        _required_string(
            visual_profile.get("scene_id"), owner="visual_profile.scene_id"
        ),
    }
    if len(scene_ids) != 1:
        raise RuntimeError(
            "metadata, source profile and visual profile scene_id differ"
        )

    source_map = _required_string(
        source_profile.get("map_path"), owner="source_profile.map_path"
    )
    visual_map = _required_string(
        visual_profile.get("map_path"), owner="visual_profile.map_path"
    )
    if source_map != visual_map:
        raise RuntimeError("source profile and visual profile map_path differ")
    if not source_map.startswith("/Game/"):
        raise RuntimeError("bound map_path must be a /Game asset path")

    source_reference = _required_string(
        metadata.get("source_reference"), owner="metadata.source_reference"
    )
    expected_stage_name = _required_string(
        visual_profile.get("source_stage_name"),
        owner="visual_profile.source_stage_name",
    )
    if Path(expected_stage_name).name != expected_stage_name:
        raise RuntimeError("visual_profile.source_stage_name must be a filename")
    observed_stage_name = Path(source_reference).name
    if observed_stage_name != expected_stage_name:
        raise RuntimeError("metadata and visual profile source stage differ")

    room_scope = _required_string(
        metadata.get("room_scope"), owner="metadata.room_scope"
    )
    selected_scopes = visual_profile.get("selected_scopes")
    if not isinstance(selected_scopes, list) or room_scope not in selected_scopes:
        raise RuntimeError("metadata room_scope is absent from visual profile scopes")

    return {
        "dataset_id": next(iter(dataset_ids)),
        "scene_id": next(iter(scene_ids)),
        "room_id": _required_string(metadata.get("room_id"), owner="metadata.room_id"),
        "room_scope": room_scope,
        "map_path": source_map,
        "source_stage_name": observed_stage_name,
        "source_reference": source_reference,
    }


__all__ = ["validate_kujiale_scene_binding"]
