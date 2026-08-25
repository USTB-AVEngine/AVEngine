from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import load_json
from avengine.security.path_policy import PathPolicyError, WorkspacePathPolicy


ROOM_REGISTRY_SCHEMA = "avengine_m6_room_registry_v1"
ROOM_PROVIDER_IDS = {
    "blender_custom",
    "replica_cad",
    "legacy_ue_apartment",
    "matterport3d",
}
RESOURCE_STATUS_VALUES = {"pass", "fail", "blocked", "not_run"}
ACOUSTIC_AUTHORITY_ROLES = {
    "production_authority",
    "raw_source",
    "derived_proxy",
}

_ENV_TEMPLATE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}/(.+)$")


class RoomContractError(ValueError):
    """Raised when a room registry or room record is not safe to consume."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class ResourceResolution:
    resource_id: str
    status: str
    path: Path | None
    reason: str | None
    sha256: str | None


def _schema_path(filename: str) -> Path:
    source_path = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed_path = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    return source_path if source_path.is_file() else installed_path


def _json_schema_errors(value: Any, filename: str) -> list[str]:
    schema = load_json(_schema_path(filename))
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def room_revision_key(record: Mapping[str, Any]) -> str:
    return f"{record['room_id']}@{record['revision']}"


def _resource_index(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {resource["resource_id"]: resource for resource in record["resources"]}


def _representation_index(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        representation["representation_id"]: representation
        for representation in record["acoustic_representations"]
    }


def validate_room_registry(value: Any) -> list[str]:
    """Validate the v1 schema and the cross-record room invariants.

    This validator intentionally does not touch the filesystem. Availability and
    artifact hashes are resolved by a provider at execution time.
    """

    errors = _json_schema_errors(value, "m6_room_registry_v1.schema.json")
    if errors or not isinstance(value, dict):
        return errors

    records = value.get("records", [])
    revision_keys: set[str] = set()
    for record_index, record in enumerate(records):
        prefix = f"records[{record_index}]"
        if not isinstance(record, dict):
            continue
        key = room_revision_key(record)
        if key in revision_keys:
            errors.append(f"{prefix}: duplicate room revision key {key!r}")
        revision_keys.add(key)

        provider_id = record.get("provider_id")
        if provider_id not in ROOM_PROVIDER_IDS:
            errors.append(f"{prefix}: unsupported provider_id {provider_id!r}")

        resources = _resource_index(record)
        if len(resources) != len(record.get("resources", [])):
            errors.append(f"{prefix}: resource_id values must be unique")

        representations = _representation_index(record)
        if len(representations) != len(record.get("acoustic_representations", [])):
            errors.append(f"{prefix}: acoustic representation IDs must be unique")

        for resource in record.get("resources", []):
            location = resource.get("location", {})
            kind = location.get("kind")
            if kind == "repository_relative":
                raw_path = Path(location.get("path", ""))
                if raw_path.is_absolute():
                    errors.append(
                        f"{prefix}.{resource['resource_id']}: repository path must be relative"
                    )
            elif kind == "environment_template":
                template = location.get("path_template", "")
                match = _ENV_TEMPLATE.fullmatch(template)
                if match is None:
                    errors.append(
                        f"{prefix}.{resource['resource_id']}: invalid environment path template"
                    )
                elif match.group(1) != location.get("environment_variable"):
                    errors.append(
                        f"{prefix}.{resource['resource_id']}: template variable must match "
                        "environment_variable"
                    )

        for representation in record.get("acoustic_representations", []):
            representation_id = representation["representation_id"]
            resource_id = representation.get("resource_id")
            if resource_id is not None and resource_id not in resources:
                errors.append(
                    f"{prefix}.{representation_id}: unknown resource_id {resource_id!r}"
                )
            for input_resource_id in representation.get("input_resource_ids", []):
                if input_resource_id not in resources:
                    errors.append(
                        f"{prefix}.{representation_id}: unknown input resource "
                        f"{input_resource_id!r}"
                    )
            derived_from = representation.get("derived_from")
            if derived_from is not None and derived_from not in representations:
                errors.append(
                    f"{prefix}.{representation_id}: unknown derived_from {derived_from!r}"
                )
            if representation["geometry_kind"] == "debug_aabb_proxy":
                if representation["role"] != "diagnostic_only":
                    errors.append(
                        f"{prefix}.{representation_id}: AABB geometry is diagnostic-only"
                    )
                if record["admission_state"] == "admitted":
                    errors.append(
                        f"{prefix}.{representation_id}: admitted rooms cannot use an AABB "
                        "acoustic representation"
                    )
            if representation["role"] == "raw_source" and not representation["immutable"]:
                errors.append(
                    f"{prefix}.{representation_id}: raw acoustic sources must be immutable"
                )
            if representation["role"] == "derived_proxy" and derived_from is None:
                errors.append(
                    f"{prefix}.{representation_id}: a derived proxy must declare derived_from"
                )

        for report_ref in record.get("qualification_reports", []):
            representation_id = report_ref.get("acoustic_representation_id")
            if representation_id is not None and representation_id not in representations:
                errors.append(
                    f"{prefix}: qualification report references unknown acoustic "
                    f"representation {representation_id!r}"
                )
            if Path(report_ref.get("path", "")).is_absolute():
                errors.append(
                    f"{prefix}: qualification report paths must be repository-relative"
                )

        if provider_id == "matterport3d":
            raw = [
                item for item in record["acoustic_representations"]
                if item["role"] == "raw_source"
            ]
            derived = [
                item for item in record["acoustic_representations"]
                if item["role"] == "derived_proxy"
            ]
            if len(raw) != 1 or not derived:
                errors.append(
                    f"{prefix}: Matterport3D records require one immutable raw source and "
                    "at least one separately declared derived proxy"
                )
            elif any(
                item.get("derived_from") != raw[0]["representation_id"]
                for item in derived
            ):
                errors.append(
                    f"{prefix}: every Matterport3D derived proxy must reference its raw "
                    "source"
                )

        lineage = record.get("lineage", {})
        if lineage.get("acoustic_profile_id") == lineage.get("room_geometry_id"):
            errors.append(
                f"{prefix}: acoustic profile and room geometry are separate lineage axes"
            )

    return errors


def load_room_registry(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = load_json(source)
    errors = validate_room_registry(value)
    if errors:
        raise RoomContractError(errors)
    return value


def index_room_records(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    errors = validate_room_registry(registry)
    if errors:
        raise RoomContractError(errors)
    return {room_revision_key(record): record for record in registry["records"]}


def find_room_record(
    registry: Mapping[str, Any], room_id: str, revision: str | None = None
) -> Mapping[str, Any]:
    matches = [record for record in registry["records"] if record["room_id"] == room_id]
    if revision is not None:
        matches = [record for record in matches if record["revision"] == revision]
    if len(matches) != 1:
        suffix = f" revision {revision!r}" if revision is not None else ""
        raise KeyError(f"expected exactly one room {room_id!r}{suffix}, found {len(matches)}")
    return matches[0]


def find_resource(record: Mapping[str, Any], resource_id: str) -> Mapping[str, Any]:
    resources = _resource_index(record)
    try:
        return resources[resource_id]
    except KeyError as error:
        raise KeyError(
            f"room {room_revision_key(record)} has no resource {resource_id!r}"
        ) from error


def find_acoustic_representation(
    record: Mapping[str, Any], representation_id: str
) -> Mapping[str, Any]:
    representations = _representation_index(record)
    try:
        return representations[representation_id]
    except KeyError as error:
        raise KeyError(
            f"room {room_revision_key(record)} has no acoustic representation "
            f"{representation_id!r}"
        ) from error


def resolve_resource(
    resource: Mapping[str, Any],
    *,
    repository_root: str | Path,
    environment: Mapping[str, str] | None = None,
    verify_hash: bool = True,
) -> ResourceResolution:
    """Resolve one declared resource without silently inventing local defaults."""

    env = dict(os.environ if environment is None else environment)
    resource_id = str(resource["resource_id"])
    location = resource["location"]
    kind = location["kind"]

    if kind == "generated":
        return ResourceResolution(
            resource_id=resource_id,
            status="not_run",
            path=None,
            reason=(
                f"generated resource has not been produced; run {location['producer']}"
            ),
            sha256=None,
        )
    if kind == "unavailable":
        return ResourceResolution(
            resource_id=resource_id,
            status="blocked",
            path=None,
            reason=location["blocker"],
            sha256=None,
        )

    if kind == "repository_relative":
        raw_root = Path(repository_root)
        raw_candidate = location["path"]
    elif kind == "environment_template":
        variable = location["environment_variable"]
        raw_root = env.get(variable)
        if not raw_root:
            return ResourceResolution(
                resource_id,
                "blocked",
                None,
                f"required environment variable {variable} is not set",
                None,
            )
        match = _ENV_TEMPLATE.fullmatch(location["path_template"])
        if match is None or match.group(1) != variable:
            return ResourceResolution(
                resource_id, "fail", None, "invalid environment path template", None
            )
        raw_candidate = match.group(2)
    else:
        return ResourceResolution(
            resource_id, "fail", None, f"unsupported resource location kind {kind!r}", None
        )

    try:
        policy = WorkspacePathPolicy.from_roots([raw_root])
    except (FileNotFoundError, PathPolicyError) as error:
        return ResourceResolution(
            resource_id,
            "blocked",
            None,
            f"declared resource root is unavailable: {error}",
            None,
        )
    expected_hash = resource.get("sha256")
    try:
        candidate = policy.resolve_input(
            raw_candidate,
            owner=f"M6 room resource {resource_id}",
            kind="file",
            expected_sha256=expected_hash if verify_hash else None,
        )
    except PathPolicyError as error:
        message = str(error)
        status = "blocked" if "does not exist" in message else "fail"
        return ResourceResolution(
            resource_id,
            status,
            None,
            message,
            None,
        )
    return ResourceResolution(
        resource_id,
        "pass",
        candidate,
        None,
        expected_hash if verify_hash else None,
    )


def resolve_room_resources(
    record: Mapping[str, Any],
    *,
    repository_root: str | Path,
    environment: Mapping[str, str] | None = None,
    verify_hash: bool = True,
) -> dict[str, ResourceResolution]:
    return {
        resource["resource_id"]: resolve_resource(
            resource,
            repository_root=repository_root,
            environment=environment,
            verify_hash=verify_hash,
        )
        for resource in record["resources"]
    }
