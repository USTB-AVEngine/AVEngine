"""A deliberately thin, read-only boundary for future task exporters.

This is not an episode schema and does not define QA, SelectTSL bins, dense
velocity tracks, or a target source.  It only makes stable IDs and immutable
artifact hashes available so a later exporter can derive task data without
re-running Habitat.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExporterInterfaceError(ValueError):
    pass


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class EvidenceArtifactRef:
    role: str
    uri: str
    sha256: str


class ReadOnlyEvidenceBundle(Mapping[str, Any]):
    """Immutable view over a canary evidence index, not the evidence payloads."""

    _REQUIRED = {
        "evidence_bundle_id",
        "room_id",
        "entity_ids",
        "source_endpoint_ids",
        "event_ids",
        "artifacts",
    }

    def __init__(self, value: Mapping[str, Any]):
        detached = deepcopy(dict(value))
        missing = sorted(self._REQUIRED - set(detached))
        if missing:
            raise ExporterInterfaceError(f"missing evidence index fields: {missing}")
        for field in ("evidence_bundle_id", "room_id"):
            if not isinstance(detached[field], str) or _STABLE_ID.fullmatch(detached[field]) is None:
                raise ExporterInterfaceError(f"{field} must be a stable ID")
        for field in ("entity_ids", "source_endpoint_ids", "event_ids"):
            values = detached[field]
            if (
                not isinstance(values, list)
                or values != sorted(set(values))
                or any(not isinstance(item, str) or _STABLE_ID.fullmatch(item) is None for item in values)
            ):
                raise ExporterInterfaceError(f"{field} must be canonical stable IDs")
        artifacts = detached["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ExporterInterfaceError("artifacts must be a non-empty list")
        roles: list[str] = []
        for item in artifacts:
            if not isinstance(item, Mapping):
                raise ExporterInterfaceError("artifact references must be mappings")
            role = item.get("role")
            uri = item.get("uri")
            sha256 = item.get("sha256")
            if not isinstance(role, str) or _STABLE_ID.fullmatch(role) is None:
                raise ExporterInterfaceError("artifact role must be a stable ID")
            if not isinstance(uri, str) or not uri:
                raise ExporterInterfaceError("artifact uri must be non-empty")
            if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
                raise ExporterInterfaceError("artifact sha256 must be lowercase SHA-256")
            roles.append(role)
        if roles != sorted(set(roles)):
            raise ExporterInterfaceError("artifact roles must be unique and canonical")
        self._value = _deep_freeze(detached)

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def artifact(self, role: str) -> EvidenceArtifactRef:
        for item in self._value["artifacts"]:
            if item["role"] == role:
                return EvidenceArtifactRef(
                    role=item["role"], uri=item["uri"], sha256=item["sha256"]
                )
        raise KeyError(f"evidence artifact role is unavailable: {role}")


@runtime_checkable
class TaskExporter(Protocol):
    """Future deterministic derivative exporter; M6 supplies no implementation."""

    exporter_id: str
    exporter_version: str

    def export(
        self,
        evidence: ReadOnlyEvidenceBundle,
        specification: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

