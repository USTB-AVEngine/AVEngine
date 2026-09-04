"""Data-driven room and source-asset runtime profile registries.

The M6 registries own stable room/entity identity.  This module adds the
separate, replaceable execution information needed by a concrete runtime:
Timeline shape/forward metadata, measured emitter anchors, UE content paths
and room-backend scene selection.  A dry sound remains an independent sound
asset and is intentionally not embedded in a visual source profile.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.appearance.contracts import CANONICAL_DOMAINS, COAT_PROFILE_DOMAINS
from avengine.contracts.json_io import canonical_json_sha256, load_json


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


@lru_cache(maxsize=None)
def _validator(filename: str) -> Draft202012Validator:
    """Compile one shipped schema per process.

    The schemas are shipped data that cannot change while a tool runs, but
    every call used to re-read the file and rebuild the validator, which for a
    schema with references is where the time goes.  Measured 2026-09-03: a
    two-cell card17 design spent 1.75 s of its 3.05 s rebuilding this validator
    28 times over the same registry (57 percent of the run), while every
    sha256 in that run together came to 0.13 percent.  The checks are
    unchanged; only the compilation is shared.
    """
    return Draft202012Validator(load_json(_schema_path(filename)))


def _schema_errors(value: Any, filename: str) -> list[str]:
    errors: list[str] = []
    for error in sorted(
        _validator(filename).iter_errors(value),
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


def _validate_local_basis(
    value: Mapping[str, Any],
    *,
    owner: str,
    expected_forward: Sequence[float] | None = None,
) -> list[str]:
    """Validate one explicit right-handed orthonormal asset-local basis."""

    errors: list[str] = []
    axes: dict[str, tuple[float, ...]] = {}
    for field in ("forward_axis", "up_axis", "right_axis"):
        try:
            axis = _finite_vector(
                value.get(field),
                length=3,
                owner=f"{owner}.{field}",
            )
        except RuntimeProfileError as error:
            errors.extend(error.errors)
            continue
        norm = math.sqrt(sum(component * component for component in axis))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            errors.append(f"{owner}.{field} must be unit length")
        axes[field] = axis
    if len(axes) != 3:
        return errors

    forward = axes["forward_axis"]
    up = axes["up_axis"]
    right = axes["right_axis"]
    for left_name, left, right_name, other in (
        ("forward_axis", forward, "up_axis", up),
        ("forward_axis", forward, "right_axis", right),
        ("up_axis", up, "right_axis", right),
    ):
        dot = sum(a * b for a, b in zip(left, other, strict=True))
        if not math.isclose(dot, 0.0, rel_tol=0.0, abs_tol=1.0e-9):
            errors.append(
                f"{owner}.{left_name} and {right_name} must be orthogonal"
            )

    forward_cross_up = (
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    )
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-9)
        for observed, expected in zip(forward_cross_up, right, strict=True)
    ):
        errors.append(f"{owner} must satisfy forward cross up equals right")

    if expected_forward is not None:
        try:
            declared_forward = _finite_vector(
                expected_forward,
                length=3,
                owner=f"{owner}.expected_forward",
            )
        except RuntimeProfileError as error:
            errors.extend(error.errors)
        else:
            if any(
                not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-9)
                for observed, expected in zip(
                    forward, declared_forward, strict=True
                )
            ):
                errors.append(
                    f"{owner}.forward_axis must match the Timeline "
                    "local anatomical forward axis"
                )
    return errors


def _asset_bound_artifacts(lineage: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return every exact external artifact referenced by one lineage block."""

    result: list[Mapping[str, Any]] = []
    source_asset = lineage.get("source_asset_v2")
    if isinstance(source_asset, Mapping):
        for field in ("record", "registry"):
            artifact = source_asset.get(field)
            if isinstance(artifact, Mapping):
                result.append(artifact)
    geometry = lineage.get("geometry")
    if isinstance(geometry, Mapping):
        for field in (
            "raw_pixel3d_glb",
            "tokenrig_input_glb",
            "runtime_glb",
            "repair_evidence",
        ):
            artifact = geometry.get(field)
            if isinstance(artifact, Mapping):
                result.append(artifact)
    for field in (
        "tokenrig_animation_closure",
        "ue_asset_bound_import_evidence",
        "ue_runtime_readback_evidence",
        "emitter_measurement_evidence",
    ):
        artifact = lineage.get(field)
        if isinstance(artifact, Mapping):
            result.append(artifact)
    admission = lineage.get("admission")
    if isinstance(admission, Mapping):
        result.extend(
            artifact
            for artifact in admission.get("evidence", ())
            if isinstance(artifact, Mapping)
        )
    return result


# Validated registries, keyed by the content that was validated.
#
# Every lookup used to re-validate the whole registry: source_asset_runtime_index
# and resolve_source_asset_alias both call the validator, and resolving one
# asset goes through the index.  Measured 2026-09-03 on a two-cell card17
# design: 28 validations of the same 14-asset registry, 1.75 s of a 3.05 s run
# (57 percent), while every sha256 in that run together came to 0.13 percent.
# Compiling the schema once did not help - the cost is the traversal - so the
# result is remembered per content instead.  Hashing the document to key it
# costs about a millisecond against sixty for a validation, which is the whole
# point: a hash here buys time rather than spending it.  Errors are returned as
# a fresh list so a caller may still mutate its copy.
_VALIDATED_SOURCE_REGISTRIES: dict[str, tuple[str, ...]] = {}
_VALIDATED_CACHE_LIMIT = 8


def _content_key(value: Any) -> str | None:
    """Content hash of a registry document, or None when it cannot be hashed."""
    try:
        return canonical_json_sha256(value)
    except (TypeError, ValueError):
        return None


def validate_source_asset_runtime_registry(value: Any) -> list[str]:
    """Validate schema plus cross-record source/runtime invariants.

    The result is memoized on the document's content hash, so validating the
    same registry twice in one process costs a hash instead of a full pass.
    """
    key = _content_key(value)
    if key is not None:
        remembered = _VALIDATED_SOURCE_REGISTRIES.get(key)
        if remembered is not None:
            return list(remembered)
    errors = _validate_source_asset_runtime_registry_uncached(value)
    if key is not None:
        if len(_VALIDATED_SOURCE_REGISTRIES) >= _VALIDATED_CACHE_LIMIT:
            _VALIDATED_SOURCE_REGISTRIES.pop(
                next(iter(_VALIDATED_SOURCE_REGISTRIES)), None)
        _VALIDATED_SOURCE_REGISTRIES[key] = tuple(errors)
    return errors


def _validate_source_asset_runtime_registry_uncached(value: Any) -> list[str]:
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

        identity = record.get("identity")
        attributes = record.get("realized_attributes")
        if isinstance(identity, Mapping) and isinstance(attributes, Mapping):
            coat = attributes.get("coat_profile")
            if isinstance(coat, Mapping):
                registry_key = (
                    str(identity.get("species_id")),
                    str(identity.get("breed_id")),
                    str(coat.get("profile_id")),
                )
                domain = COAT_PROFILE_DOMAINS.get(registry_key)
                if domain is None:
                    errors.append(
                        f"{prefix}: coat profile {registry_key!r} is not "
                        "registered in the appearance contract; a "
                        "namespaced-looking profile_id is not registration"
                    )
                elif coat.get("value") not in domain:
                    errors.append(
                        f"{prefix}: coat value {coat.get('value')!r} is outside "
                        f"the registered domain {domain}"
                    )

        generation = record.get("generation_request_attributes")
        if generation is not None and isinstance(generation, Mapping):
            for axis in ("size", "body_build", "life_stage"):
                sampled = generation.get(axis)
                if sampled is not None and sampled not in CANONICAL_DOMAINS[axis]:
                    errors.append(
                        f"{prefix}: generation_request_attributes.{axis} "
                        f"{sampled!r} is outside the canonical domain"
                    )

        timeline = record.get("timeline")
        timeline_forward: tuple[float, ...] | None = None
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
                timeline_forward = axis
                norm = math.sqrt(sum(component * component for component in axis))
                if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
                    errors.append(
                        f"{prefix}: local anatomical forward axis must be unit length"
                    )
                if math.hypot(axis[0], axis[2]) <= 1.0e-12:
                    errors.append(
                        f"{prefix}: local anatomical forward axis must be horizontal"
                    )

            idle_action_id = timeline.get("idle_action_id")
            walking_action_id = timeline.get("walking_action_id")
            if idle_action_id == walking_action_id:
                errors.append(
                    f"{prefix}: Idle and Walking action IDs must be distinct"
                )

        anchors_by_id = {
            str(anchor.get("anchor_id")): anchor
            for anchor in anchors
            if isinstance(anchor, Mapping)
        }
        for anchor_id, anchor in anchors_by_id.items():
            basis = anchor.get("local_basis")
            if isinstance(basis, Mapping):
                errors.extend(
                    _validate_local_basis(
                        basis,
                        owner=f"{prefix}.emitter_anchors[{anchor_id}].local_basis",
                        expected_forward=timeline_forward,
                    )
                )

        spear = record.get("runtime_backends", {}).get("spear_unreal")
        if isinstance(spear, Mapping):
            actor_scale = spear.get("actor_scale")
            if actor_scale is not None and (
                isinstance(actor_scale, bool)
                or not isinstance(actor_scale, (int, float))
                or not math.isfinite(float(actor_scale))
                or float(actor_scale) <= 0.0
            ):
                errors.append(
                    f"{prefix}.runtime_backends.spear_unreal.actor_scale "
                    "must be a positive finite number"
                )

            action_paths = spear.get("animation_paths_by_action_id")
            if isinstance(action_paths, Mapping) and isinstance(timeline, Mapping):
                expected_action_paths = {
                    str(timeline.get("idle_action_id")): spear.get(
                        "idle_animation"
                    ),
                    str(timeline.get("walking_action_id")): spear.get(
                        "walking_animation"
                    ),
                }
                if dict(action_paths) != expected_action_paths:
                    errors.append(
                        f"{prefix}: animation_paths_by_action_id must exactly "
                        "bind the Timeline Idle and Walking action IDs"
                    )

        lineage = record.get("asset_bound_lineage")
        admission_state = record.get("admission_state")
        if admission_state == "formal" and not isinstance(lineage, Mapping):
            errors.append(
                f"{prefix}: formal admission requires exact asset-bound lineage"
            )
        if not isinstance(lineage, Mapping):
            continue

        if record.get("geometry", {}).get("mesh_authority") != (
            "generated_pixel3d_target_native"
        ):
            errors.append(
                f"{prefix}: SPEAR source_asset_v2 lineage is only valid for "
                "generated Pixel3D target-native geometry"
            )
        if lineage.get("runtime_asset_id") != record.get("asset_id"):
            errors.append(
                f"{prefix}: asset-bound lineage runtime_asset_id does not match"
            )
        if lineage.get("runtime_revision") != record.get("revision"):
            errors.append(
                f"{prefix}: asset-bound lineage runtime_revision does not match"
            )

        geometry_lineage = lineage.get("geometry")
        if isinstance(geometry_lineage, Mapping):
            lineage_kind = geometry_lineage.get("lineage_kind")
            raw = geometry_lineage.get("raw_pixel3d_glb")
            tokenrig_input = geometry_lineage.get("tokenrig_input_glb")
            if geometry_lineage.get("runtime_mesh_uri") != record.get(
                "geometry", {}
            ).get("source_mesh_uri"):
                errors.append(
                    f"{prefix}: asset-bound runtime_mesh_uri does not match "
                    "geometry.source_mesh_uri"
                )
            if isinstance(raw, Mapping) and isinstance(tokenrig_input, Mapping):
                same_bytes = raw.get("sha256") == tokenrig_input.get("sha256")
                if (
                    lineage_kind == "unchanged_pixel3d_geometry"
                    and not same_bytes
                ):
                    errors.append(
                        f"{prefix}: unchanged Pixel3D geometry must retain the "
                        "raw GLB SHA-256"
                    )
                if (
                    lineage_kind == "bounded_same_pixel3d_mesh_repair"
                    and same_bytes
                ):
                    errors.append(
                        f"{prefix}: bounded repair must bind a byte-distinct "
                        "TokenRig input GLB"
                    )

        if not isinstance(spear, Mapping):
            errors.append(
                f"{prefix}: exact asset-bound lineage requires a SPEAR/UE binding"
            )
        else:
            if (
                isinstance(spear.get("actor_scale"), bool)
                or not isinstance(spear.get("actor_scale"), (int, float))
                or not math.isfinite(float(spear["actor_scale"]))
                or float(spear["actor_scale"]) <= 0.0
            ):
                errors.append(
                    f"{prefix}: exact asset-bound lineage requires positive "
                    "finite SPEAR/UE actor_scale"
                )
            for field in (
                "blueprint_class_path",
                "skeletal_mesh_path",
                "idle_animation",
                "walking_animation",
            ):
                path = spear.get(field)
                if not isinstance(path, str) or not path.startswith("/Game/"):
                    errors.append(
                        f"{prefix}: exact asset-bound {field} must be an "
                        "explicit /Game/ object path"
                    )
            if not isinstance(
                spear.get("animation_paths_by_action_id"), Mapping
            ):
                errors.append(
                    f"{prefix}: exact asset-bound lineage requires "
                    "animation_paths_by_action_id"
                )

        missing_basis = [
            anchor_id
            for anchor_id, anchor in anchors_by_id.items()
            if not isinstance(anchor.get("local_basis"), Mapping)
        ]
        if missing_basis:
            errors.append(
                f"{prefix}: exact asset-bound emitter anchors lack local basis: "
                f"{sorted(missing_basis)}"
            )

        source_asset_v2 = lineage.get("source_asset_v2")
        admission = lineage.get("admission")
        expected_source_states = {
            "research": "research_candidate",
            "qualified": "research_candidate",
            "formal": "formal_dataset_asset",
            "unavailable": "technical_spike_only",
            "rejected": "rejected",
        }
        if (
            isinstance(source_asset_v2, Mapping)
            and source_asset_v2.get("state_classification")
            != expected_source_states.get(str(admission_state))
        ):
            errors.append(
                f"{prefix}: source_asset_v2 state_classification does not "
                "match runtime admission_state"
            )
        if isinstance(admission, Mapping):
            if admission.get("state") != admission_state:
                errors.append(
                    f"{prefix}: asset-bound admission state does not match"
                )
            formal_authorized = admission.get(
                "formal_dataset_registration_authorized"
            )
            if formal_authorized is not (admission_state == "formal"):
                errors.append(
                    f"{prefix}: formal dataset registration authorization "
                    "must be true exactly for formal admission"
                )

        if admission_state == "formal":
            for artifact in _asset_bound_artifacts(lineage):
                path = PurePosixPath(str(artifact.get("path", "")))
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "tmp" in {part.lower() for part in path.parts}
                ):
                    errors.append(
                        f"{prefix}: formal lineage artifacts must use immutable "
                        "relative non-tmp paths"
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
    binding = {
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
    if isinstance(anchor.get("local_basis"), Mapping):
        binding["local_basis"] = deepcopy(dict(anchor["local_basis"]))
    return binding


def build_exact_asset_bound_runtime_binding(
    registry: Mapping[str, Any],
    *,
    source_slot_id: str,
    asset_id: str,
    anchor_id: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Materialize one complete exact Pixel3D/SPEAR/UE runtime binding.

    Historical research profiles may omit the exact closure and continue to
    resolve through the legacy helpers.  This helper deliberately fails closed
    unless scale, Mesh/action object paths, emitter basis, source_asset_v2
    lineage and admission evidence are all present.
    """

    record = resolve_source_asset_runtime_profile(registry, asset_id, revision)
    lineage = record.get("asset_bound_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeProfileError(
            f"source asset {asset_id!r} has no exact asset-bound lineage"
        )
    spear = record.get("runtime_backends", {}).get("spear_unreal")
    if not isinstance(spear, Mapping):
        raise RuntimeProfileError(
            f"source asset {asset_id!r} has no exact SPEAR/UE binding"
        )
    emitter = build_asset_emitter_binding(
        registry,
        source_slot_id=source_slot_id,
        asset_id=asset_id,
        anchor_id=anchor_id,
        revision=revision,
    )
    if not isinstance(emitter.get("local_basis"), Mapping):
        raise RuntimeProfileError(
            f"source asset {asset_id!r} emitter has no exact local basis"
        )
    timeline = record["timeline"]
    return {
        "schema": "avengine_exact_asset_bound_runtime_binding_v1",
        "source_slot_id": source_slot_id,
        "asset_id": asset_id,
        "asset_revision": record["revision"],
        "actor_scale": float(spear["actor_scale"]),
        "emitter": emitter,
        "timeline": {
            "template_id": timeline["template_id"],
            "body_plan_id": timeline["body_plan_id"],
            "local_anatomical_forward_axis": deepcopy(
                list(timeline["local_anatomical_forward_axis"])
            ),
            "animation_paths_by_action_id": deepcopy(
                dict(spear["animation_paths_by_action_id"])
            ),
        },
        "spear_unreal": deepcopy(dict(spear)),
        "asset_bound_lineage": deepcopy(dict(lineage)),
        "admission_state": record["admission_state"],
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
    "build_exact_asset_bound_runtime_binding",
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
