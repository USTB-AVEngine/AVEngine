"""Compile exact room/profile selections into the common M3 package contract.

SoundSpaces, Habitat semantic scenes, and SPEAR/UE exports differ only in how
their geometry labels or material slots are translated into RLR materials.
This module selects that adapter from an exact acoustic-profile registry
binding, invokes the existing M3 compiler, and verifies the resulting package
against the selected room geometry and material resource hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from avengine.acoustic_profiles import (
    AcousticProfileError,
    AcousticProfileSelection,
    acoustic_profile_package_binding,
    resolve_acoustic_profile,
    verify_acoustic_package_for_selection,
    verify_acoustic_profile_material_inputs,
)
from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
)
from avengine.m1.contracts import validate_room_manifest
from avengine.acoustics.compiler import (
    AcousticSceneCompileError,
    compile_mp3d_soundspaces_research_scene,
    compile_visual_slot_semantic_research_scene,
)
from avengine.acoustics.contracts import load_and_validate_acoustic_scene_package


PROFILED_COMPILE_RECEIPT_SCHEMA = (
    "avengine_m3_profiled_acoustic_scene_compile_receipt_v1"
)

_SOUNDSPACES_ROUTE = (
    "soundspaces2_public",
    "soundspaces2_mp3d_semantic_labels_v1",
)
_HABITAT_ROUTE = (
    "habitat_scene",
    "habitat_visual_material_slots_residential_v1",
)
_SPEAR_ROUTE = (
    "spear_ue_authored",
    "spear_ue_material_slot_authored_v1",
)


@dataclass(frozen=True)
class ProfiledAcousticSceneCompileResult:
    """A package plus the exact profile-selection evidence used to build it."""

    manifest_path: Path
    coverage_path: Path
    compiler_route: str
    selection: AcousticProfileSelection
    room_manifest_record: Mapping[str, Any]
    material_input_records: Mapping[str, Mapping[str, Any]]
    simulation_input_records: Mapping[str, Mapping[str, Any]]

    def receipt(self) -> dict[str, Any]:
        manifest = load_json(self.manifest_path)
        value: dict[str, Any] = {
            "schema": PROFILED_COMPILE_RECEIPT_SCHEMA,
            "status": "pass",
            "solver_backend_id": self.selection.solver_backend_id,
            "compiler_route": self.compiler_route,
            "room_ref": dict(self.selection.binding["room_ref"]),
            "lineage_acoustic_profile_id": self.selection.binding[
                "lineage_acoustic_profile_id"
            ],
            "profile_ref": dict(self.selection.binding["profile_ref"]),
            "binding_id": self.selection.binding["binding_id"],
            "acoustic_representation_id": self.selection.representation[
                "representation_id"
            ],
            "acoustic_resource_id": self.selection.resource["resource_id"],
            "geometry_resource_id": self.selection.binding[
                "geometry_resource_id"
            ],
            "origin": dict(self.selection.profile["origin"]),
            "room_manifest": dict(self.room_manifest_record),
            "material_inputs": {
                role: dict(record)
                for role, record in self.material_input_records.items()
            },
            "simulation_inputs": {
                name: dict(record)
                for name, record in self.simulation_input_records.items()
            },
            "package": {
                "manifest_path": str(self.manifest_path),
                "manifest_byte_size": self.manifest_path.stat().st_size,
                "manifest_sha256": sha256_file(self.manifest_path),
                "package_id": manifest["package_id"],
                "package_content_sha256": manifest[
                    "package_content_sha256"
                ],
                "acoustic_profile_binding": dict(
                    manifest["materials"]["acoustic_profile_binding"]
                ),
            },
            "coverage": {
                "path": str(self.coverage_path),
                "byte_size": self.coverage_path.stat().st_size,
                "sha256": sha256_file(self.coverage_path),
            },
        }
        value["receipt_content_sha256"] = canonical_json_sha256(value)
        return value


def _verify_room_manifest(
    selection: AcousticProfileSelection,
    room_manifest_path: Path,
) -> dict[str, Any]:
    try:
        room = load_json(room_manifest_path)
    except (OSError, ValueError) as error:
        raise AcousticSceneCompileError(
            f"selected room manifest is not readable JSON: {error}"
        ) from error
    errors = validate_room_manifest(room)
    if errors:
        raise AcousticSceneCompileError(
            "selected room manifest is invalid: " + "; ".join(errors)
        )
    room_ref = selection.binding["room_ref"]
    if room["room_id"] != room_ref["room_id"]:
        raise AcousticSceneCompileError(
            "selected room manifest room_id does not match exact room_ref"
        )
    registry_coordinate = selection.room_record.get("coordinate_system", {})
    manifest_coordinate = room.get("coordinate_system", {})
    coordinate_fields = (
        "handedness",
        "up_axis",
        "forward_axis",
        "linear_unit",
    )
    if any(
        manifest_coordinate.get(field) != registry_coordinate.get(field)
        for field in coordinate_fields
    ):
        raise AcousticSceneCompileError(
            "selected room manifest coordinate system differs from exact room "
            "registry record"
        )

    observed_sha256 = sha256_file(room_manifest_path)
    declared = [
        item
        for item in selection.room_record.get("resources", ())
        if item.get("resource_type") == "room_manifest"
    ]
    if len(declared) > 1:
        raise AcousticSceneCompileError(
            "exact room registry record declares multiple room_manifest resources"
        )
    if declared:
        expected_sha256 = declared[0].get("sha256")
        if not isinstance(expected_sha256, str):
            raise AcousticSceneCompileError(
                "declared room_manifest resource has no sha256"
            )
        if observed_sha256 != expected_sha256:
            raise AcousticSceneCompileError(
                "selected room manifest sha256 differs from exact room registry "
                f"record: expected {expected_sha256}, observed {observed_sha256}"
            )
        registry_binding = {
            "status": "pass",
            "resource_id": declared[0]["resource_id"],
            "declared_sha256": expected_sha256,
        }
    else:
        registry_binding = {
            "status": "not_declared",
            "reason": (
                "room identity is closed by room_id plus the exact compiled "
                "geometry_resource_id hash"
            ),
        }
    return {
        "path": str(room_manifest_path),
        "byte_size": room_manifest_path.stat().st_size,
        "sha256": observed_sha256,
        "room_id": room["room_id"],
        "registry_binding": registry_binding,
    }


def _verify_simulation_inputs(
    selection: AcousticProfileSelection,
) -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {
        "production": selection.simulation_request_path,
    }
    if selection.reference_simulation_request_path is not None:
        paths["reference"] = selection.reference_simulation_request_path
    records: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise AcousticProfileError(
                f"{name} simulation request does not exist: {path}"
            )
        declared = selection.profile["simulation"][name].get("sha256")
        if not isinstance(declared, str):
            raise AcousticProfileError(
                f"{name} simulation request is not sha256-bound"
            )
        observed = sha256_file(path)
        if observed != declared:
            raise AcousticProfileError(
                f"{name} simulation request sha256 mismatch: expected "
                f"{declared}, observed {observed}"
            )
        records[name] = {
            "path": str(path.resolve()),
            "byte_size": path.stat().st_size,
            "sha256": observed,
        }
    return records


def _single_material_path(
    selection: AcousticProfileSelection,
    records: Mapping[str, Mapping[str, Any]],
    *,
    expected_role: str,
) -> Path:
    if set(records) != {expected_role}:
        raise AcousticProfileError(
            f"adapter {selection.profile['material_binding']['adapter_id']!r} "
            f"requires exactly material role {expected_role!r}, observed "
            f"{sorted(records)!r}"
        )
    return Path(records[expected_role]["path"])


def compile_profiled_acoustic_scene(
    selection: AcousticProfileSelection,
    *,
    room_manifest: str | Path,
    output: str | Path,
    seed: int = 917,
    package_id: str | None = None,
    probe_origins: Sequence[Sequence[float]] | None = None,
    probe_direction_count: int = 32,
    environment: Mapping[str, str] | None = None,
) -> ProfiledAcousticSceneCompileResult:
    """Compile one already-selected profile and verify its exact package identity."""

    room_manifest_path = Path(room_manifest).resolve()
    room_manifest_record = _verify_room_manifest(selection, room_manifest_path)
    material_records = verify_acoustic_profile_material_inputs(selection)
    simulation_records = _verify_simulation_inputs(selection)
    package_binding = acoustic_profile_package_binding(selection)
    origin_kind = selection.profile["origin"]["kind"]
    adapter_id = selection.profile["material_binding"]["adapter_id"]
    route_key = (origin_kind, adapter_id)
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AcousticSceneCompileError(f"output already exists: {destination}")
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.profiled-staging-",
            dir=destination.parent,
        )
    ).resolve()
    staging_root.chmod(0o700)
    compile_output = staging_root / "package"

    try:
        if route_key == _SOUNDSPACES_ROUTE:
            material_path = _single_material_path(
                selection,
                material_records,
                expected_role="soundspaces2_public_material_config",
            )
            origin = selection.profile["origin"]
            manifest_path, coverage_path = (
                compile_mp3d_soundspaces_research_scene(
                    room_manifest=room_manifest_path,
                    material_config=material_path,
                    output=compile_output,
                    database_id=selection.profile["profile_id"],
                    version=selection.profile["revision"],
                    source_description=(
                        f"{origin['project']}; source_revision="
                        f"{origin['source_revision']}"
                    ),
                    source_uri=origin.get("citation"),
                    package_id=package_id,
                    probe_origins=probe_origins,
                    probe_direction_count=probe_direction_count,
                    environment=environment,
                )
            )
            compiler_route = "soundspaces2_mp3d_public_materials_to_m3_v1"
        elif route_key in {_HABITAT_ROUTE, _SPEAR_ROUTE}:
            expected_role = (
                "habitat_visual_material_rules"
                if route_key == _HABITAT_ROUTE
                else "spear_ue_authored_material_rules"
            )
            material_path = _single_material_path(
                selection,
                material_records,
                expected_role=expected_role,
            )
            manifest_path, coverage_path = (
                compile_visual_slot_semantic_research_scene(
                    room_manifest=room_manifest_path,
                    material_rules=material_path,
                    output=compile_output,
                    seed=seed,
                    transform_profile="identity_y_up",
                    transform_reviewed=True,
                    package_id=package_id,
                    probe_origins=probe_origins,
                    probe_direction_count=probe_direction_count,
                    environment=environment,
                    acoustic_profile_binding=package_binding,
                )
            )
            compiler_route = (
                "habitat_visual_slots_to_m3_v1"
                if route_key == _HABITAT_ROUTE
                else "spear_ue_export_visual_slots_to_m3_v1"
            )
        else:
            raise AcousticProfileError(
                "no fail-closed M3 compiler route for acoustic profile "
                f"origin/adapter {route_key!r}"
            )

        manifest_path = Path(manifest_path).resolve()
        coverage_path = Path(coverage_path).resolve()
        package = load_and_validate_acoustic_scene_package(manifest_path)
        observed_binding = package.manifest["materials"].get(
            "acoustic_profile_binding"
        )
        if observed_binding != package_binding:
            raise AcousticSceneCompileError(
                "compiled package acoustic_profile_binding differs from exact "
                "registry selection"
            )
        verify_acoustic_package_for_selection(selection, manifest_path)
        os.rename(manifest_path.parent, destination)
        shutil.rmtree(staging_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return ProfiledAcousticSceneCompileResult(
        manifest_path=destination / manifest_path.name,
        coverage_path=destination / coverage_path.name,
        compiler_route=compiler_route,
        selection=selection,
        room_manifest_record=room_manifest_record,
        material_input_records=material_records,
        simulation_input_records=simulation_records,
    )


def compile_registered_acoustic_scene(
    acoustic_profile_registry: Mapping[str, Any],
    room_registry: Mapping[str, Any],
    room_ref: Mapping[str, Any],
    *,
    room_manifest: str | Path,
    output: str | Path,
    repository_root: str | Path,
    environment: Mapping[str, str] | None = None,
    seed: int = 917,
    package_id: str | None = None,
    probe_origins: Sequence[Sequence[float]] | None = None,
    probe_direction_count: int = 32,
) -> ProfiledAcousticSceneCompileResult:
    """Select by exact scene identity, compile, and close the package binding."""

    selection = resolve_acoustic_profile(
        acoustic_profile_registry,
        room_registry,
        room_ref,
        repository_root=repository_root,
        environment=environment,
        verify_paths=False,
    )
    return compile_profiled_acoustic_scene(
        selection,
        room_manifest=room_manifest,
        output=output,
        seed=seed,
        package_id=package_id,
        probe_origins=probe_origins,
        probe_direction_count=probe_direction_count,
        environment=environment,
    )


__all__ = [
    "PROFILED_COMPILE_RECEIPT_SCHEMA",
    "ProfiledAcousticSceneCompileResult",
    "compile_profiled_acoustic_scene",
    "compile_registered_acoustic_scene",
]
