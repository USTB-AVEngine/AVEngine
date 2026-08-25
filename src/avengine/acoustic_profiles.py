"""Room-identity acoustic profile selection over one common RLR solver.

This module deliberately does not depend on a visual runtime.  It joins an
exact M6 room revision and its acoustic-profile lineage to material inputs,
one acoustic representation/resource, and production/reference RLR settings.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
)
from avengine.acoustics.contracts import validate_package_manifest


ACOUSTIC_PROFILE_REGISTRY_SCHEMA = "avengine_acoustic_profile_registry_v1"
ACOUSTIC_PROFILE_SELECTION_SCHEMA = "avengine_acoustic_profile_selection_v1"
SOLVER_BACKEND_ID = "rlr_audio_propagation"

_SCHEMA_FILE = "acoustic_profile_registry_v1.schema.json"
_DEFAULT_FILE = "acoustic_profiles.json"
_ENV_TEMPLATE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}/(.+)$")


class AcousticProfileError(ValueError):
    """An acoustic registry, binding, path, or exact selection is invalid."""

    def __init__(self, errors: Sequence[str] | str):
        self.errors = (
            (errors,)
            if isinstance(errors, str)
            else tuple(str(error) for error in errors)
        )
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class AcousticProfileSelection:
    """Resolved acoustic inputs for one exact room revision."""

    solver_backend_id: str
    profile: Mapping[str, Any]
    binding: Mapping[str, Any]
    room_record: Mapping[str, Any]
    representation: Mapping[str, Any]
    resource: Mapping[str, Any]
    acoustic_package_manifest_path: Path | None
    material_paths: Mapping[str, Path | None]
    simulation_request_path: Path
    reference_simulation_request_path: Path | None
    verification_status: str
    acoustic_profile_registry_sha256: str
    room_registry_sha256: str
    _path_records: Mapping[str, Mapping[str, Any]] = field(
        repr=False, compare=False
    )

    def simulation_path(self, simulation_profile: str = "production") -> Path:
        if simulation_profile == "production":
            return self.simulation_request_path
        if simulation_profile == "reference":
            if self.reference_simulation_request_path is None:
                raise AcousticProfileError(
                    f"profile {self.profile['profile_id']!r} has no reference "
                    "simulation request"
                )
            return self.reference_simulation_request_path
        raise AcousticProfileError(
            "simulation_profile must be 'production' or 'reference'"
        )

    def receipt(
        self, simulation_profile: str = "production"
    ) -> dict[str, Any]:
        """Return a JSON-safe, hash-bound selection record.

        With ``verify_paths=False`` this remains useful to visual-only callers,
        but every path record and the receipt explicitly remain ``not_verified``.
        """

        self.simulation_path(simulation_profile)
        paths = deepcopy(dict(self._path_records))
        paths["selected_simulation_request"] = deepcopy(
            paths[f"simulation_{simulation_profile}"]
        )
        value: dict[str, Any] = {
            "schema": ACOUSTIC_PROFILE_SELECTION_SCHEMA,
            "solver_backend_id": self.solver_backend_id,
            "verification_status": self.verification_status,
            "room_ref": deepcopy(dict(self.binding["room_ref"])),
            "lineage_acoustic_profile_id": self.binding[
                "lineage_acoustic_profile_id"
            ],
            "room_provider_id": self.room_record["provider_id"],
            "profile_ref": deepcopy(dict(self.binding["profile_ref"])),
            "binding_id": self.binding["binding_id"],
            "acoustic_representation_id": self.representation[
                "representation_id"
            ],
            "acoustic_resource_id": self.resource["resource_id"],
            "geometry_resource_id": self.binding["geometry_resource_id"],
            "simulation_profile": simulation_profile,
            "acoustic_profile_registry_sha256": (
                self.acoustic_profile_registry_sha256
            ),
            "room_registry_sha256": self.room_registry_sha256,
            "paths": paths,
        }
        value["selection_content_sha256"] = canonical_json_sha256(value)
        return value


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    source = _repository_root() / "schemas" / _SCHEMA_FILE
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / _SCHEMA_FILE
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(
            f"AVEngine acoustic profile schema is unavailable: {_SCHEMA_FILE}"
        )
    return path


def default_acoustic_profile_registry_path() -> Path:
    source = _repository_root() / "examples" / "runtime" / _DEFAULT_FILE
    installed = (
        Path(sys.prefix) / "share" / "avengine" / "runtime_profiles" / _DEFAULT_FILE
    )
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(
            f"AVEngine default acoustic profile registry is unavailable: {_DEFAULT_FILE}"
        )
    return path


def _schema_errors(value: Any) -> list[str]:
    schema = load_json(_schema_path())
    errors: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def _validate_path_reference(
    reference: Mapping[str, Any], *, owner: str
) -> list[str]:
    errors: list[str] = []
    kind = reference.get("kind")
    if kind == "repository_relative":
        path = Path(str(reference.get("path", "")))
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"{owner}: repository path must be relative and confined")
    elif kind == "environment_template":
        variable = reference.get("environment_variable")
        template = reference.get("path_template")
        match = _ENV_TEMPLATE.fullmatch(str(template))
        if match is None or match.group(1) != variable:
            errors.append(
                f"{owner}: path_template must begin with the declared "
                "environment variable"
            )
    return errors


def validate_acoustic_profile_registry(value: Any) -> list[str]:
    """Validate schema plus uniqueness and profile-local invariants."""

    errors = _schema_errors(value)
    if errors or not isinstance(value, Mapping):
        return errors

    profiles: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, profile in enumerate(value.get("profiles", ())):
        if not isinstance(profile, Mapping):
            continue
        key = (str(profile.get("profile_id")), str(profile.get("revision")))
        if key in profiles:
            errors.append(f"profiles[{index}]: duplicate exact profile {key!r}")
        profiles[key] = profile
        if profile.get("solver_backend_id") != value.get("solver_backend_id"):
            errors.append(
                f"profiles[{index}]: solver_backend_id differs from registry"
            )
        roles: set[str] = set()
        for resource_index, material in enumerate(
            profile.get("material_binding", {}).get("resources", ())
        ):
            if not isinstance(material, Mapping):
                continue
            role = str(material.get("role"))
            if role in roles:
                errors.append(
                    f"profiles[{index}].material_binding.resources"
                    f"[{resource_index}]: duplicate role {role!r}"
                )
            roles.add(role)
            path = material.get("path")
            if isinstance(path, Mapping):
                errors.extend(
                    _validate_path_reference(
                        path,
                        owner=(
                            f"profiles[{index}].material_binding.resources"
                            f"[{resource_index}].path"
                        ),
                    )
                )
        for name, path in profile.get("simulation", {}).items():
            if isinstance(path, Mapping):
                errors.extend(
                    _validate_path_reference(
                        path, owner=f"profiles[{index}].simulation.{name}"
                    )
                )

    binding_ids: set[str] = set()
    binding_keys: set[tuple[str, str, str, str]] = set()
    bound_profiles: set[tuple[str, str]] = set()
    for index, binding in enumerate(value.get("bindings", ())):
        if not isinstance(binding, Mapping):
            continue
        binding_id = str(binding.get("binding_id"))
        if binding_id in binding_ids:
            errors.append(f"bindings[{index}]: duplicate binding_id {binding_id!r}")
        binding_ids.add(binding_id)
        room_ref = binding.get("room_ref", {})
        key = (
            str(room_ref.get("registry_id")),
            str(room_ref.get("room_id")),
            str(room_ref.get("revision")),
            str(binding.get("lineage_acoustic_profile_id")),
        )
        if key in binding_keys:
            errors.append(f"bindings[{index}]: duplicate exact room/lineage binding")
        binding_keys.add(key)
        profile_ref = binding.get("profile_ref", {})
        profile_key = (
            str(profile_ref.get("profile_id")),
            str(profile_ref.get("revision")),
        )
        bound_profiles.add(profile_key)
        if profile_key not in profiles:
            errors.append(
                f"bindings[{index}].profile_ref does not resolve an exact profile"
            )
        if profile_ref.get("profile_id") != binding.get(
            "lineage_acoustic_profile_id"
        ):
            errors.append(
                f"bindings[{index}]: profile_id must equal the room lineage "
                "acoustic_profile_id"
            )
        package = binding.get("acoustic_package_manifest")
        if isinstance(package, Mapping):
            errors.extend(
                _validate_path_reference(
                    package,
                    owner=f"bindings[{index}].acoustic_package_manifest",
                )
            )

    for profile_key in profiles:
        if profile_key not in bound_profiles:
            errors.append(f"profile {profile_key!r} has no room binding")
    return errors


def load_acoustic_profile_registry(path: str | Path) -> dict[str, Any]:
    value = load_json(path)
    errors = validate_acoustic_profile_registry(value)
    if errors:
        raise AcousticProfileError(errors)
    return value


def load_default_acoustic_profile_registry() -> dict[str, Any]:
    return load_acoustic_profile_registry(default_acoustic_profile_registry_path())


def validate_acoustic_profile_links(
    registry: Mapping[str, Any], room_registry: Mapping[str, Any]
) -> list[str]:
    """Validate exact room, lineage, representation, and resource links."""

    errors = validate_acoustic_profile_registry(registry)
    if errors:
        return errors
    room_registry_id = room_registry.get("registry_id")
    records = {
        (record.get("room_id"), record.get("revision")): record
        for record in room_registry.get("records", ())
        if isinstance(record, Mapping)
    }
    profiles = {
        (profile["profile_id"], profile["revision"]): profile
        for profile in registry["profiles"]
    }
    for index, binding in enumerate(registry["bindings"]):
        room_ref = binding["room_ref"]
        if room_ref["registry_id"] != room_registry_id:
            errors.append(
                f"bindings[{index}].room_ref.registry_id does not match room registry"
            )
            continue
        room = records.get((room_ref["room_id"], room_ref["revision"]))
        if room is None:
            errors.append(
                f"bindings[{index}].room_ref does not resolve an exact room revision"
            )
            continue
        if room.get("lineage", {}).get("acoustic_profile_id") != binding[
            "lineage_acoustic_profile_id"
        ]:
            errors.append(
                f"bindings[{index}] does not match room lineage.acoustic_profile_id"
            )
        profile_ref = binding["profile_ref"]
        profile = profiles[(profile_ref["profile_id"], profile_ref["revision"])]
        provider_id = room.get("provider_id")
        source_matches = [
            candidate
            for candidate in profiles.values()
            if provider_id in candidate["source_selection"]["provider_ids"]
        ]
        if len(source_matches) != 1:
            errors.append(
                f"bindings[{index}] room provider_id does not select exactly "
                "one acoustic profile"
            )
        elif (
            source_matches[0]["profile_id"],
            source_matches[0]["revision"],
        ) != (profile["profile_id"], profile["revision"]):
            errors.append(
                f"bindings[{index}] source-selected acoustic profile differs "
                "from binding.profile_ref"
            )
        representations = {
            item.get("representation_id"): item
            for item in room.get("acoustic_representations", ())
            if isinstance(item, Mapping)
        }
        representation = representations.get(
            binding["acoustic_representation_id"]
        )
        if representation is None:
            errors.append(
                f"bindings[{index}].acoustic_representation_id does not resolve"
            )
            continue
        resources = {
            item.get("resource_id"): item
            for item in room.get("resources", ())
            if isinstance(item, Mapping)
        }
        resource = resources.get(binding["acoustic_resource_id"])
        if resource is None:
            errors.append(f"bindings[{index}].acoustic_resource_id does not resolve")
        if representation.get("resource_id") != binding["acoustic_resource_id"]:
            errors.append(
                f"bindings[{index}]: representation resource differs from "
                "acoustic_resource_id"
            )
        geometry_resource_id = binding["geometry_resource_id"]
        geometry_resource = resources.get(geometry_resource_id)
        if geometry_resource is None:
            errors.append(
                f"bindings[{index}].geometry_resource_id does not resolve"
            )
        elif not isinstance(geometry_resource.get("sha256"), str):
            errors.append(
                f"bindings[{index}].geometry_resource_id must select a "
                "sha256-bound resource"
            )
        if geometry_resource_id not in representation.get(
            "input_resource_ids", ()
        ):
            errors.append(
                f"bindings[{index}].geometry_resource_id is not an input of "
                "the selected acoustic representation"
            )
    return errors


def select_acoustic_profile_for_room_source(
    registry: Mapping[str, Any],
    room_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select one profile from a dataset/provider identity, or fail closed."""

    errors = validate_acoustic_profile_registry(registry)
    if errors:
        raise AcousticProfileError(errors)
    provider_id = room_record.get("provider_id")
    if not isinstance(provider_id, str) or not provider_id:
        raise AcousticProfileError(
            "room record must declare provider_id for source profile selection"
        )
    matches = [
        profile
        for profile in registry["profiles"]
        if provider_id in profile["source_selection"]["provider_ids"]
    ]
    if len(matches) != 1:
        raise AcousticProfileError(
            "room provider_id must resolve exactly one acoustic profile: "
            f"provider_id={provider_id!r}, match_count={len(matches)}"
        )
    profile = matches[0]
    lineage_profile_id = room_record.get("lineage", {}).get(
        "acoustic_profile_id"
    )
    if (
        isinstance(lineage_profile_id, str)
        and lineage_profile_id
        and lineage_profile_id != profile["profile_id"]
    ):
        raise AcousticProfileError(
            "source-selected profile differs from "
            "room lineage.acoustic_profile_id"
        )
    return profile


def _resolve_path(
    reference: Mapping[str, Any],
    *,
    owner: str,
    repository_root: Path,
    environment: Mapping[str, str],
    verify: bool,
) -> tuple[Path | None, dict[str, Any]]:
    kind = reference["kind"]
    if kind == "repository_relative":
        raw = Path(reference["path"])
        if raw.is_absolute() or ".." in raw.parts:
            raise AcousticProfileError(
                f"{owner}: repository path must be relative and confined"
            )
        root = repository_root.resolve()
        path = (root / raw).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise AcousticProfileError(
                f"{owner}: repository path escapes repository root"
            ) from error
    else:
        variable = reference["environment_variable"]
        match = _ENV_TEMPLATE.fullmatch(reference["path_template"])
        if match is None or match.group(1) != variable:
            raise AcousticProfileError(f"{owner}: invalid environment path template")
        raw_root = environment.get(variable)
        if not raw_root:
            if verify:
                raise AcousticProfileError(
                    f"{owner}: required environment variable {variable} is not set"
                )
            return None, {
                "declared": deepcopy(dict(reference)),
                "resolved_path": None,
                "verification_status": "not_verified",
                "exists": None,
                "size_bytes": None,
                "sha256": None,
            }
        root = Path(raw_root).expanduser().resolve()
        path = (root / match.group(2)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise AcousticProfileError(
                f"{owner}: environment path escapes declared root"
            ) from error

    expected = reference.get("sha256")
    actual: str | None = None
    size: int | None = None
    exists: bool | None = None
    if verify:
        if not path.is_file():
            raise AcousticProfileError(f"{owner}: file does not exist: {path}")
        exists = True
        size = path.stat().st_size
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise AcousticProfileError(
                f"{owner}: sha256 mismatch: expected {expected}, observed {actual}"
            )
    return path, {
        "declared": deepcopy(dict(reference)),
        "resolved_path": str(path),
        "verification_status": "verified" if verify else "not_verified",
        "exists": exists,
        "size_bytes": size,
        "sha256": actual,
    }


def _validate_acoustic_package_identity(
    path: Path,
    *,
    room_ref: Mapping[str, Any],
    room_record: Mapping[str, Any],
    profile: Mapping[str, Any],
    representation: Mapping[str, Any],
    resource: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    try:
        manifest = load_json(path)
    except (OSError, ValueError) as error:
        raise AcousticProfileError(
            f"acoustic package manifest is not readable JSON: {path}: {error}"
        ) from error
    manifest_errors = validate_package_manifest(manifest)
    if manifest_errors:
        raise AcousticProfileError(
            [
                f"acoustic package manifest contract: {error}"
                for error in manifest_errors
            ]
        )

    source_room = manifest["source_room"]
    if source_room["room_id"] != room_ref["room_id"]:
        raise AcousticProfileError(
            "acoustic package source_room.room_id does not match exact room_ref"
        )
    if representation.get("resource_id") != resource.get("resource_id"):
        raise AcousticProfileError(
            "selected acoustic representation does not own the selected resource"
        )

    room_resources = {
        item.get("resource_id"): item
        for item in room_record.get("resources", ())
        if isinstance(item, Mapping)
    }
    geometry_resource_id = binding["geometry_resource_id"]
    geometry_resource = room_resources.get(geometry_resource_id)
    if (
        not isinstance(geometry_resource, Mapping)
        or geometry_resource_id not in representation.get(
            "input_resource_ids", ()
        )
        or not isinstance(geometry_resource.get("sha256"), str)
    ):
        raise AcousticProfileError(
            "selected geometry_resource_id is not a sha256-bound input of the "
            "acoustic representation"
        )
    if source_room["geometry_asset_sha256"] != geometry_resource["sha256"]:
        raise AcousticProfileError(
            "acoustic package source_room.geometry_asset_sha256 does not match "
            "the binding's exact geometry_resource_id"
        )

    room_manifest_hashes = {
        item["sha256"]
        for item in room_resources.values()
        if item.get("resource_type") == "room_manifest"
        and isinstance(item.get("sha256"), str)
    }
    if (
        room_manifest_hashes
        and source_room["manifest_sha256"] not in room_manifest_hashes
    ):
        raise AcousticProfileError(
            "acoustic package source_room.manifest_sha256 does not match any "
            "declared room_manifest resource sha256"
        )

    package_profile_binding = manifest["materials"].get(
        "acoustic_profile_binding"
    )
    if not isinstance(package_profile_binding, Mapping):
        raise AcousticProfileError(
            "acoustic package omits materials.acoustic_profile_binding"
        )
    expected_profile_ref = binding["profile_ref"]
    if (
        package_profile_binding.get("profile_id")
        != expected_profile_ref["profile_id"]
        or package_profile_binding.get("profile_revision")
        != expected_profile_ref["revision"]
    ):
        raise AcousticProfileError(
            "acoustic package material profile identity does not match "
            "binding.profile_ref"
        )
    material_binding = profile["material_binding"]
    if package_profile_binding.get("adapter_id") != material_binding[
        "adapter_id"
    ]:
        raise AcousticProfileError(
            "acoustic package material adapter does not match selected profile"
        )
    expected_material_resources = {
        item["role"]: item["path"].get("sha256")
        for item in material_binding["resources"]
    }
    if any(value is None for value in expected_material_resources.values()):
        raise AcousticProfileError(
            "verified acoustic profiles require sha256-bound material resources"
        )
    observed_resource_items = [
        item
        for item in package_profile_binding.get("resources", ())
        if isinstance(item, Mapping)
    ]
    observed_material_resources = {
        item.get("role"): item.get("sha256")
        for item in observed_resource_items
    }
    if (
        len(observed_material_resources) != len(observed_resource_items)
        or observed_material_resources != expected_material_resources
    ):
        raise AcousticProfileError(
            "acoustic package material resource identities do not match "
            "the selected profile"
        )


def acoustic_profile_package_binding(
    selection: AcousticProfileSelection,
) -> dict[str, Any]:
    """Build the exact profile identity embedded in a generated M3 package."""

    material_binding = selection.profile["material_binding"]
    resources: list[dict[str, str]] = []
    for item in material_binding["resources"]:
        expected_sha256 = item["path"].get("sha256")
        if not isinstance(expected_sha256, str):
            raise AcousticProfileError(
                "profiled package compilation requires sha256-bound material "
                f"resource {item['role']!r}"
            )
        resources.append(
            {
                "role": item["role"],
                "sha256": expected_sha256,
            }
        )
    return {
        "schema": "avengine_m3_acoustic_profile_binding_v1",
        "profile_id": selection.profile["profile_id"],
        "profile_revision": selection.profile["revision"],
        "adapter_id": material_binding["adapter_id"],
        "resources": resources,
    }


def verify_acoustic_profile_material_inputs(
    selection: AcousticProfileSelection,
) -> dict[str, dict[str, Any]]:
    """Verify every selected material input before profile-driven compilation."""

    records: dict[str, dict[str, Any]] = {}
    for item in selection.profile["material_binding"]["resources"]:
        role = item["role"]
        path = selection.material_paths.get(role)
        if path is None:
            raise AcousticProfileError(
                f"material resource {role!r} did not resolve to a path"
            )
        if not path.is_file():
            raise AcousticProfileError(
                f"material resource {role!r} does not exist: {path}"
            )
        expected_sha256 = item["path"].get("sha256")
        if not isinstance(expected_sha256, str):
            raise AcousticProfileError(
                "profiled package compilation requires sha256-bound material "
                f"resource {role!r}"
            )
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise AcousticProfileError(
                f"material resource {role!r} sha256 mismatch: expected "
                f"{expected_sha256}, observed {observed_sha256}"
            )
        records[role] = {
            "path": str(path.resolve()),
            "byte_size": path.stat().st_size,
            "sha256": observed_sha256,
        }
    return records


def verify_acoustic_package_for_selection(
    selection: AcousticProfileSelection,
    manifest_path: str | Path,
) -> None:
    """Verify one generated package against an already exact room selection."""

    _validate_acoustic_package_identity(
        Path(manifest_path).resolve(),
        room_ref=selection.binding["room_ref"],
        room_record=selection.room_record,
        profile=selection.profile,
        representation=selection.representation,
        resource=selection.resource,
        binding=selection.binding,
    )


def resolve_acoustic_profile(
    registry: Mapping[str, Any],
    room_registry: Mapping[str, Any],
    room_ref: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    verify_paths: bool = True,
) -> AcousticProfileSelection:
    """Resolve one exact room+lineage binding and all declared input paths."""

    link_errors = validate_acoustic_profile_links(registry, room_registry)
    if link_errors:
        raise AcousticProfileError(link_errors)
    expected_ref_fields = {"registry_id", "room_id", "revision"}
    if set(room_ref) != expected_ref_fields:
        raise AcousticProfileError(
            "room_ref must contain exactly registry_id, room_id, and revision"
        )
    if room_ref["registry_id"] != room_registry.get("registry_id"):
        raise AcousticProfileError("room_ref.registry_id does not match room registry")

    room_matches = [
        record
        for record in room_registry["records"]
        if record["room_id"] == room_ref["room_id"]
        and record["revision"] == room_ref["revision"]
    ]
    if len(room_matches) != 1:
        raise AcousticProfileError(
            f"room_ref does not resolve exactly one room revision: {dict(room_ref)!r}"
        )
    room = room_matches[0]
    lineage_profile_id = room["lineage"]["acoustic_profile_id"]
    binding_matches = [
        binding
        for binding in registry["bindings"]
        if binding["room_ref"] == room_ref
        and binding["lineage_acoustic_profile_id"] == lineage_profile_id
    ]
    if len(binding_matches) != 1:
        raise AcousticProfileError(
            "room_ref plus room lineage.acoustic_profile_id does not resolve "
            "exactly one acoustic binding"
        )
    binding = binding_matches[0]

    profile_matches = [
        profile
        for profile in registry["profiles"]
        if profile["profile_id"] == binding["profile_ref"]["profile_id"]
        and profile["revision"] == binding["profile_ref"]["revision"]
    ]
    if len(profile_matches) != 1:
        raise AcousticProfileError("binding profile_ref does not resolve exactly once")
    profile = profile_matches[0]
    representation = next(
        item
        for item in room["acoustic_representations"]
        if item["representation_id"] == binding["acoustic_representation_id"]
    )
    resource = next(
        item
        for item in room["resources"]
        if item["resource_id"] == binding["acoustic_resource_id"]
    )

    root = Path(repository_root) if repository_root is not None else _repository_root()
    env = dict(os.environ if environment is None else environment)
    path_records: dict[str, Mapping[str, Any]] = {}
    package_path, path_records["acoustic_package_manifest"] = _resolve_path(
        binding["acoustic_package_manifest"],
        owner="acoustic_package_manifest",
        repository_root=root,
        environment=env,
        verify=verify_paths,
    )
    material_paths: dict[str, Path | None] = {}
    for material in profile["material_binding"]["resources"]:
        role = material["role"]
        material_paths[role], path_records[f"material_{role}"] = _resolve_path(
            material["path"],
            owner=f"material resource {role}",
            repository_root=root,
            environment=env,
            verify=verify_paths,
        )
    production_path, path_records["simulation_production"] = _resolve_path(
        profile["simulation"]["production"],
        owner="production simulation request",
        repository_root=root,
        environment=env,
        verify=verify_paths,
    )
    if production_path is None:
        raise AcousticProfileError("production simulation request did not resolve")
    reference_path: Path | None = None
    if "reference" in profile["simulation"]:
        reference_path, path_records["simulation_reference"] = _resolve_path(
            profile["simulation"]["reference"],
            owner="reference simulation request",
            repository_root=root,
            environment=env,
            verify=verify_paths,
        )
    else:
        path_records["simulation_reference"] = {
            "declared": None,
            "resolved_path": None,
            "verification_status": (
                "verified" if verify_paths else "not_verified"
            ),
            "exists": None,
            "size_bytes": None,
            "sha256": None,
        }

    if verify_paths:
        assert package_path is not None
        _validate_acoustic_package_identity(
            package_path,
            room_ref=room_ref,
            room_record=room,
            profile=profile,
            representation=representation,
            resource=resource,
            binding=binding,
        )

    return AcousticProfileSelection(
        solver_backend_id=SOLVER_BACKEND_ID,
        profile=profile,
        binding=binding,
        room_record=room,
        representation=representation,
        resource=resource,
        acoustic_package_manifest_path=package_path,
        material_paths=material_paths,
        simulation_request_path=production_path,
        reference_simulation_request_path=reference_path,
        verification_status="verified" if verify_paths else "not_verified",
        acoustic_profile_registry_sha256=canonical_json_sha256(registry),
        room_registry_sha256=canonical_json_sha256(room_registry),
        _path_records=path_records,
    )


__all__ = [
    "ACOUSTIC_PROFILE_REGISTRY_SCHEMA",
    "ACOUSTIC_PROFILE_SELECTION_SCHEMA",
    "SOLVER_BACKEND_ID",
    "AcousticProfileError",
    "AcousticProfileSelection",
    "acoustic_profile_package_binding",
    "default_acoustic_profile_registry_path",
    "load_acoustic_profile_registry",
    "load_default_acoustic_profile_registry",
    "resolve_acoustic_profile",
    "select_acoustic_profile_for_room_source",
    "verify_acoustic_package_for_selection",
    "verify_acoustic_profile_material_inputs",
    "validate_acoustic_profile_links",
    "validate_acoustic_profile_registry",
]
