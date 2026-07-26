"""Data-driven room and source-asset runtime profile registries.

The M6 registries own stable room/entity identity.  This module adds the
separate, replaceable execution information needed by a concrete runtime:
Timeline shape/forward metadata, measured emitter anchors, UE content paths
and room-backend scene selection.  A dry sound remains an independent sound
asset and is intentionally not embedded in a visual source profile.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import load_json


SOURCE_ASSET_RUNTIME_REGISTRY_SCHEMA = (
    "avengine_source_asset_runtime_registry_v1"
)
ROOM_RUNTIME_PROFILE_REGISTRY_SCHEMA = (
    "avengine_room_runtime_profile_registry_v1"
)

_SOURCE_SCHEMA_FILE = "source_asset_runtime_registry_v1.schema.json"
_ROOM_SCHEMA_FILE = "room_runtime_profile_registry_v1.schema.json"
_SOURCE_DEFAULT_FILE = "source_asset_runtime_profiles.json"
_ROOM_DEFAULT_FILE = "room_runtime_profiles.json"


class RuntimeProfileError(ValueError):
    """A runtime profile registry or exact selection is invalid."""

    def __init__(self, errors: Sequence[str] | str):
        self.errors = (
            (errors,)
            if isinstance(errors, str)
            else tuple(str(error) for error in errors)
        )
        super().__init__("; ".join(self.errors))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path(filename: str) -> Path:
    source = _repository_root() / "schemas" / filename
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(f"AVEngine runtime schema is unavailable: {filename}")
    return path


def _default_data_path(filename: str) -> Path:
    source = _repository_root() / "examples" / "runtime" / filename
    installed = (
        Path(sys.prefix) / "share" / "avengine" / "runtime_profiles" / filename
    )
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(
            f"AVEngine default runtime registry is unavailable: {filename}"
        )
    return path


def default_source_asset_runtime_registry_path() -> Path:
    return _default_data_path(_SOURCE_DEFAULT_FILE)


def default_room_runtime_profile_registry_path() -> Path:
    return _default_data_path(_ROOM_DEFAULT_FILE)


def _schema_errors(value: Any, filename: str) -> list[str]:
    schema = load_json(_schema_path(filename))
    errors: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def _finite_vector(value: Any, *, length: int, owner: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeProfileError(
            f"{owner} must contain exactly {length} finite numbers"
        )
    if len(value) != length:
        raise RuntimeProfileError(
            f"{owner} must contain exactly {length} finite numbers"
        )
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RuntimeProfileError(f"{owner}[{index}] must be finite")
        number = float(item)
        if not math.isfinite(number):
            raise RuntimeProfileError(f"{owner}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def validate_source_asset_runtime_registry(value: Any) -> list[str]:
    """Validate schema plus cross-record source/runtime invariants."""

    errors = _schema_errors(value, _SOURCE_SCHEMA_FILE)
    if errors or not isinstance(value, Mapping):
        return errors

    records = value.get("assets", ())
    keys: set[tuple[str, str]] = set()
    asset_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        prefix = f"assets[{index}]"
        key = (str(record.get("asset_id")), str(record.get("revision")))
        if key in keys:
            errors.append(f"{prefix}: repeated asset ID/revision {key!r}")
        keys.add(key)
        asset_id = str(record.get("asset_id"))
        if asset_id in asset_ids:
            errors.append(
                f"{prefix}: v1 requires one selected runtime revision per asset ID"
            )
        asset_ids.add(asset_id)

        anchors = record.get("emitter_anchors", ())
        anchor_ids = [
            anchor.get("anchor_id")
            for anchor in anchors
            if isinstance(anchor, Mapping)
        ]
        if len(anchor_ids) != len(set(anchor_ids)):
            errors.append(f"{prefix}: emitter anchor IDs must be unique")
        if record.get("default_emitter_anchor_id") not in anchor_ids:
            errors.append(f"{prefix}: default emitter anchor does not resolve")

        timeline = record.get("timeline")
        if isinstance(timeline, Mapping):
            try:
                axis = _finite_vector(
                    timeline.get("local_anatomical_forward_axis"),
                    length=3,
                    owner=f"{prefix}.timeline.local_anatomical_forward_axis",
                )
            except RuntimeProfileError as error:
                errors.extend(error.errors)
            else:
                norm = math.sqrt(sum(component * component for component in axis))
                if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
                    errors.append(
                        f"{prefix}: local anatomical forward axis must be unit length"
                    )
                if math.hypot(axis[0], axis[2]) <= 1.0e-12:
                    errors.append(
                        f"{prefix}: local anatomical forward axis must be horizontal"
                    )

    for alias, reference in value.get("aliases", {}).items():
        if not isinstance(reference, Mapping):
            continue
        key = (str(reference.get("asset_id")), str(reference.get("revision")))
        if key not in keys:
            errors.append(f"aliases.{alias}: exact asset revision does not resolve")
    return errors


def validate_room_runtime_profile_registry(value: Any) -> list[str]:
    """Validate schema plus exact default/profile selection invariants."""

    errors = _schema_errors(value, _ROOM_SCHEMA_FILE)
    if errors or not isinstance(value, Mapping):
        return errors
    profile_ids: list[str] = []
    for index, profile in enumerate(value.get("profiles", ())):
        if not isinstance(profile, Mapping):
            continue
        profile_id = str(profile.get("profile_id"))
        profile_ids.append(profile_id)
        if profile.get("backend_id") == "spear_unreal":
            map_path = profile.get("scene", {}).get("map_path")
            if not isinstance(map_path, str) or not map_path.startswith("/Game/"):
                errors.append(
                    f"profiles[{index}]: SPEAR/UE map_path must start with /Game/"
                )
        if profile.get("backend_id") == "habitat_native":
            map_path = profile.get("scene", {}).get("map_path")
            if (
                not isinstance(map_path, str)
                or map_path.startswith("/Game/")
                or not map_path.endswith(".json")
            ):
                errors.append(
                    f"profiles[{index}]: habitat_native map_path must reference "
                    "a room manifest JSON, not a UE map"
                )
    if len(profile_ids) != len(set(profile_ids)):
        errors.append("room runtime profile IDs must be unique")
    if value.get("default_profile_id") not in profile_ids:
        errors.append("default_profile_id does not resolve")
    return errors


def _load_validated(path: str | Path, validator: Any) -> dict[str, Any]:
    value = load_json(path)
    errors = validator(value)
    if errors:
        raise RuntimeProfileError(errors)
    return value


def load_source_asset_runtime_registry(path: str | Path) -> dict[str, Any]:
    return _load_validated(path, validate_source_asset_runtime_registry)


def load_room_runtime_profile_registry(path: str | Path) -> dict[str, Any]:
    return _load_validated(path, validate_room_runtime_profile_registry)


def load_default_source_asset_runtime_registry() -> dict[str, Any]:
    return load_source_asset_runtime_registry(
        default_source_asset_runtime_registry_path()
    )


def load_default_room_runtime_profile_registry() -> dict[str, Any]:
    return load_room_runtime_profile_registry(
        default_room_runtime_profile_registry_path()
    )


def source_asset_runtime_index(
    registry: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    errors = validate_source_asset_runtime_registry(registry)
    if errors:
        raise RuntimeProfileError(errors)
    return {str(record["asset_id"]): record for record in registry["assets"]}


def resolve_source_asset_runtime_profile(
    registry: Mapping[str, Any],
    asset_id: str,
    revision: str | None = None,
) -> Mapping[str, Any]:
    try:
        record = source_asset_runtime_index(registry)[asset_id]
    except KeyError as error:
        raise RuntimeProfileError(f"unregistered source asset: {asset_id!r}") from error
    if revision is not None and record["revision"] != revision:
        raise RuntimeProfileError(
            f"source asset revision does not resolve: {asset_id}@{revision}"
        )
    return record


def resolve_source_asset_alias(
    registry: Mapping[str, Any], alias: str
) -> Mapping[str, Any]:
    errors = validate_source_asset_runtime_registry(registry)
    if errors:
        raise RuntimeProfileError(errors)
    try:
        reference = registry["aliases"][alias]
    except KeyError as error:
        raise RuntimeProfileError(f"unknown source asset alias: {alias!r}") from error
    return resolve_source_asset_runtime_profile(
        registry,
        str(reference["asset_id"]),
        str(reference["revision"]),
    )


def source_timeline_profiles(
    registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return detached Timeline-facing profiles indexed by concrete asset ID."""

    result: dict[str, dict[str, Any]] = {}
    for asset_id, record in source_asset_runtime_index(registry).items():
        timeline = record["timeline"]
        result[asset_id] = {
            "revision": record["revision"],
            "template_id": timeline["template_id"],
            "body_plan_id": timeline["body_plan_id"],
            "local_anatomical_forward_axis": tuple(
                float(value)
                for value in timeline["local_anatomical_forward_axis"]
            ),
            "walk_phase_period_frames": int(
                timeline["walk_phase_period_frames"]
            ),
            "idle_action_id": timeline["idle_action_id"],
            "walking_action_id": timeline["walking_action_id"],
            "display_label": record["display_label"],
            "identity": deepcopy(dict(record["identity"])),
            "realized_attributes": deepcopy(dict(record["realized_attributes"])),
            "geometry": deepcopy(dict(record["geometry"])),
        }
    return result


def spear_actor_bindings(
    registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return detached SPEAR/UE bindings for every compatible source asset."""

    result: dict[str, dict[str, Any]] = {}
    for asset_id, record in source_asset_runtime_index(registry).items():
        raw = record["runtime_backends"].get("spear_unreal")
        if not isinstance(raw, Mapping):
            continue
        binding = deepcopy(dict(raw))
        binding["asset_revision"] = record["revision"]
        result[asset_id] = binding
    return result


def build_asset_emitter_binding(
    registry: Mapping[str, Any],
    *,
    source_slot_id: str,
    asset_id: str,
    anchor_id: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Materialize one source-slot binding from measured asset-local metadata."""

    if source_slot_id not in {"source1", "source2"}:
        raise RuntimeProfileError("source slot must be source1 or source2")
    record = resolve_source_asset_runtime_profile(registry, asset_id, revision)
    selected_anchor_id = anchor_id or str(record["default_emitter_anchor_id"])
    matches = [
        anchor
        for anchor in record["emitter_anchors"]
        if anchor["anchor_id"] == selected_anchor_id
    ]
    if len(matches) != 1:
        raise RuntimeProfileError(
            f"asset {asset_id!r} has no unique emitter anchor {selected_anchor_id!r}"
        )
    anchor = matches[0]
    return {
        "source_slot_id": source_slot_id,
        "asset_id": asset_id,
        "asset_revision": record["revision"],
        "semantic_anchor_id": selected_anchor_id,
        "emitter_offset_m": deepcopy(list(anchor["offset_m"])),
        "local_anatomical_forward_axis": deepcopy(
            list(record["timeline"]["local_anatomical_forward_axis"])
        ),
        "offset_space": anchor["offset_space"],
    }


def room_runtime_profile_index(
    registry: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    errors = validate_room_runtime_profile_registry(registry)
    if errors:
        raise RuntimeProfileError(errors)
    return {
        str(profile["profile_id"]): profile for profile in registry["profiles"]
    }


def resolve_room_runtime_profile(
    registry: Mapping[str, Any],
    profile_id: str | None = None,
) -> Mapping[str, Any]:
    selected = str(profile_id or registry.get("default_profile_id", ""))
    try:
        return room_runtime_profile_index(registry)[selected]
    except KeyError as error:
        raise RuntimeProfileError(
            f"unknown room runtime profile: {selected!r}"
        ) from error


def validate_room_runtime_links(
    runtime_registry: Mapping[str, Any],
    room_registry: Mapping[str, Any],
) -> list[str]:
    """Check that every runtime room profile references one M6 room revision."""

    errors = validate_room_runtime_profile_registry(runtime_registry)
    if errors:
        return errors
    registry_id = room_registry.get("registry_id")
    records = {
        (record.get("room_id"), record.get("revision"))
        for record in room_registry.get("records", ())
        if isinstance(record, Mapping)
    }
    for index, profile in enumerate(runtime_registry["profiles"]):
        reference = profile["room_ref"]
        if reference["registry_id"] != registry_id:
            errors.append(
                f"profiles[{index}].room_ref.registry_id does not match room registry"
            )
        if (reference["room_id"], reference["revision"]) not in records:
            errors.append(
                f"profiles[{index}].room_ref does not resolve an exact room revision"
            )
    return errors


__all__ = [
    "ROOM_RUNTIME_PROFILE_REGISTRY_SCHEMA",
    "SOURCE_ASSET_RUNTIME_REGISTRY_SCHEMA",
    "RuntimeProfileError",
    "build_asset_emitter_binding",
    "default_room_runtime_profile_registry_path",
    "default_source_asset_runtime_registry_path",
    "load_default_room_runtime_profile_registry",
    "load_default_source_asset_runtime_registry",
    "load_room_runtime_profile_registry",
    "load_source_asset_runtime_registry",
    "resolve_room_runtime_profile",
    "resolve_source_asset_alias",
    "resolve_source_asset_runtime_profile",
    "room_runtime_profile_index",
    "source_asset_runtime_index",
    "source_timeline_profiles",
    "spear_actor_bindings",
    "validate_room_runtime_links",
    "validate_room_runtime_profile_registry",
    "validate_source_asset_runtime_registry",
]
