from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from avengine.m6.rooms import (
    ResourceResolution,
    find_acoustic_representation,
    resolve_room_resources,
    room_revision_key,
)


@dataclass(frozen=True)
class AcousticRepresentationResolution:
    representation_id: str
    status: str
    path: Path | None
    build_mode: str
    producer: str | None
    reason: str | None
    input_resources: tuple[str, ...]


@dataclass(frozen=True)
class ProviderRoomResolution:
    room_key: str
    provider_id: str
    status: str
    resources: Mapping[str, ResourceResolution]
    dimension_statuses: Mapping[str, str]
    blockers: tuple[str, ...]


def _aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_run"
    for candidate in ("fail", "blocked", "not_run"):
        if candidate in statuses:
            return candidate
    return "pass"


class RoomProvider:
    """Data-driven room source adapter.

    Providers resolve declared dependencies and expose build plans. They do not
    silently execute a native runtime or compiler, so an unavailable historical
    package remains blocked/not_run instead of becoming an inferred pass.
    """

    provider_id = ""

    def records(self, registry: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            record
            for record in registry["records"]
            if record["provider_id"] == self.provider_id
        )

    def enumerate_local_rooms(
        self,
        registry: Mapping[str, Any],
        *,
        repository_root: str | Path,
        environment: Mapping[str, str] | None = None,
        verify_hash: bool = True,
    ) -> tuple[ProviderRoomResolution, ...]:
        return tuple(
            self.resolve_room(
                record,
                repository_root=repository_root,
                environment=environment,
                verify_hash=verify_hash,
            )
            for record in self.records(registry)
        )

    def resolve_room(
        self,
        record: Mapping[str, Any],
        *,
        repository_root: str | Path,
        environment: Mapping[str, str] | None = None,
        verify_hash: bool = True,
    ) -> ProviderRoomResolution:
        if record["provider_id"] != self.provider_id:
            raise ValueError(
                f"provider {self.provider_id!r} cannot resolve record owned by "
                f"{record['provider_id']!r}"
            )
        resolutions = resolve_room_resources(
            record,
            repository_root=repository_root,
            environment=environment,
            verify_hash=verify_hash,
        )

        dimension_statuses: dict[str, str] = {}
        all_required_statuses: list[str] = []
        blockers: list[str] = []
        dimensions = {
            dimension
            for resource in record["resources"]
            for dimension in resource["required_for"]
        }
        for dimension in sorted(dimensions):
            resource_ids = [
                resource["resource_id"]
                for resource in record["resources"]
                if dimension in resource["required_for"]
            ]
            statuses = [resolutions[resource_id].status for resource_id in resource_ids]
            dimension_statuses[dimension] = _aggregate_status(statuses)
            all_required_statuses.extend(statuses)
            for resource_id in resource_ids:
                resolution = resolutions[resource_id]
                if resolution.status != "pass":
                    blockers.append(
                        f"{dimension}:{resource_id}:{resolution.status}:"
                        f"{resolution.reason or 'no reason recorded'}"
                    )

        return ProviderRoomResolution(
            room_key=room_revision_key(record),
            provider_id=self.provider_id,
            status=_aggregate_status(all_required_statuses),
            resources=resolutions,
            dimension_statuses=dimension_statuses,
            blockers=tuple(blockers),
        )

    def acoustic_representation(
        self,
        record: Mapping[str, Any],
        representation_id: str,
        *,
        repository_root: str | Path,
        environment: Mapping[str, str] | None = None,
        verify_hash: bool = True,
    ) -> AcousticRepresentationResolution:
        if record["provider_id"] != self.provider_id:
            raise ValueError(
                f"provider {self.provider_id!r} cannot inspect record owned by "
                f"{record['provider_id']!r}"
            )
        representation = find_acoustic_representation(record, representation_id)
        if (
            representation["geometry_kind"] == "debug_aabb_proxy"
            and representation["role"] != "diagnostic_only"
        ):
            return AcousticRepresentationResolution(
                representation_id,
                "fail",
                None,
                representation["build_mode"],
                representation.get("producer"),
                "AABB geometry cannot be an acoustic authority",
                tuple(representation["input_resource_ids"]),
            )

        resolutions = resolve_room_resources(
            record,
            repository_root=repository_root,
            environment=environment,
            verify_hash=verify_hash,
        )
        input_ids = tuple(representation["input_resource_ids"])
        input_statuses = [resolutions[resource_id].status for resource_id in input_ids]
        input_status = _aggregate_status(input_statuses)
        if input_status in {"fail", "blocked", "not_run"}:
            reasons = [
                f"{resource_id}: {resolutions[resource_id].reason}"
                for resource_id in input_ids
                if resolutions[resource_id].status != "pass"
            ]
            return AcousticRepresentationResolution(
                representation_id,
                input_status,
                None,
                representation["build_mode"],
                representation.get("producer"),
                "; ".join(reasons),
                input_ids,
            )

        resource_id = representation.get("resource_id")
        output_resolution = resolutions.get(resource_id) if resource_id else None
        build_mode = representation["build_mode"]
        if build_mode == "reference":
            if output_resolution is None:
                return AcousticRepresentationResolution(
                    representation_id,
                    "fail",
                    None,
                    build_mode,
                    None,
                    "reference mode requires a resource_id",
                    input_ids,
                )
            return AcousticRepresentationResolution(
                representation_id,
                output_resolution.status,
                output_resolution.path,
                build_mode,
                None,
                output_resolution.reason,
                input_ids,
            )
        if build_mode in {"compile", "derive"}:
            producer = representation.get("producer")
            if not producer:
                return AcousticRepresentationResolution(
                    representation_id,
                    "fail",
                    None,
                    build_mode,
                    None,
                    f"{build_mode} mode requires an explicit producer",
                    input_ids,
                )
            if output_resolution is not None:
                if output_resolution.status == "pass":
                    return AcousticRepresentationResolution(
                        representation_id,
                        "pass",
                        output_resolution.path,
                        build_mode,
                        producer,
                        None,
                        input_ids,
                    )
                if output_resolution.status in {"fail", "blocked"}:
                    return AcousticRepresentationResolution(
                        representation_id,
                        output_resolution.status,
                        None,
                        build_mode,
                        producer,
                        output_resolution.reason,
                        input_ids,
                    )
            return AcousticRepresentationResolution(
                representation_id,
                "not_run",
                None,
                build_mode,
                producer,
                f"declared producer has not materialized a verified output: {producer}",
                input_ids,
            )
        return AcousticRepresentationResolution(
            representation_id,
            "blocked",
            None,
            build_mode,
            representation.get("producer"),
            representation.get("blocker", "acoustic representation is unavailable"),
            input_ids,
        )


class BlenderCustomRoomProvider(RoomProvider):
    provider_id = "blender_custom"


class ReplicaCADRoomProvider(RoomProvider):
    provider_id = "replica_cad"


class LegacyUEApartmentRoomProvider(RoomProvider):
    provider_id = "legacy_ue_apartment"


class Matterport3DRoomProvider(RoomProvider):
    provider_id = "matterport3d"


PROVIDER_TYPES = {
    provider_type.provider_id: provider_type
    for provider_type in (
        BlenderCustomRoomProvider,
        ReplicaCADRoomProvider,
        LegacyUEApartmentRoomProvider,
        Matterport3DRoomProvider,
    )
}


def provider_for_id(provider_id: str) -> RoomProvider:
    try:
        return PROVIDER_TYPES[provider_id]()
    except KeyError as error:
        raise KeyError(f"no M6 room provider registered for {provider_id!r}") from error


def providers_from_registry(registry: Mapping[str, Any]) -> tuple[RoomProvider, ...]:
    provider_ids = sorted({record["provider_id"] for record in registry["records"]})
    return tuple(provider_for_id(provider_id) for provider_id in provider_ids)
