"""Prepare a real-surface authored room for the existing AVEngine M3/RLR compiler.

This adapter consumes a room build directory and its authored GLB/USD/anchor
metadata. It creates the existing room-package input shape, writes an explicit
material-assumption table, and calls the existing GLB research compiler. The
material table is a controlled research assumption; visual PBR names are never
treated as measured acoustic truth.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.acoustics.compiler import compile_explicit_glb_research_scene
from avengine.acoustics.gltf import ExpandedGltfScene, extract_triangle_scene
from avengine.rooms.contracts import validate_room_manifest


_IDENTITY_MATRIX = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _first_existing(paths: Sequence[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _metadata_path(build_dir: Path) -> Path | None:
    return _first_existing(
        (
            build_dir / "room_manifest.json",
            build_dir / "room_handoff.json",
            build_dir / "qa" / "build_report.json",
        )
    )


def _source_glb(build_dir: Path, override: str | None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file() or path.suffix.casefold() != ".glb":
            raise ValueError(f"source GLB does not exist: {path}")
        return path

    candidates: list[Path] = []
    for directory in (build_dir / "visual", build_dir / "ue_import", build_dir):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*.glb")))
    candidates = [
        path
        for path in candidates
        if "collision" not in path.stem.casefold()
    ]
    unique = sorted(set(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise ValueError(
            "build must resolve exactly one non-collision source GLB; "
            f"found={[str(path) for path in unique]}"
        )
    return unique[0]


def _source_usd(build_dir: Path, override: str | None) -> Path | None:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"source USD does not exist: {path}")
        return path
    candidates = sorted(
        {
            *build_dir.rglob("*.usd"),
            *build_dir.rglob("*.usda"),
        }
    )
    return candidates[0].resolve() if candidates else None


def _anchor_path(build_dir: Path) -> Path | None:
    return _first_existing(
        (
            build_dir / "functional_anchors.json",
            build_dir / "metadata" / "functional_anchors.json",
        )
    )


def _room_id(metadata: Mapping[str, Any], build_dir: Path) -> str:
    value = metadata.get("room_id") or metadata.get("room_spec_id")
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"room metadata has no room_id/room_spec_id: {build_dir}")


def _source_revision(metadata: Mapping[str, Any], build_dir: Path) -> str:
    for key in ("source_revision", "room_spec_id", "room_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return f"{value}_{build_dir.name}"
    return build_dir.name


def _canonical_anchor(value: Sequence[float], coordinate_note: str) -> list[float]:
    if len(value) != 3:
        raise ValueError(f"anchor position must have three values: {value!r}")
    source = [float(item) for item in value]
    if "+Z up" in coordinate_note or "authoring_up" in coordinate_note:
        return [source[0], source[2], -source[1]]
    return source


def _load_anchor_data(
    path: Path | None,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], str]:
    if path is None:
        return {}, [], "no functional anchor file supplied"
    raw = _load_json(path)
    coordinate_note = str(raw.get("coordinate_system", ""))
    anchors_raw = raw.get("anchors", {})
    anchors: dict[str, list[float]] = {}
    if isinstance(anchors_raw, Mapping):
        for name, value in anchors_raw.items():
            if (
                isinstance(name, str)
                and isinstance(value, list)
                and len(value) == 3
                and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
            ):
                anchors[name] = _canonical_anchor(value, coordinate_note)
    seats: list[dict[str, Any]] = []
    raw_seats = raw.get("seat_points", [])
    if isinstance(raw_seats, list):
        for item in raw_seats:
            if not isinstance(item, Mapping):
                continue
            anchor_id = item.get("anchor_id")
            position = item.get("position_m")
            if not isinstance(anchor_id, str) or not isinstance(position, list) or len(position) != 3:
                continue
            seat = dict(item)
            seat["position_source_m"] = [float(value) for value in position]
            seat["position_canonical_m"] = _canonical_anchor(position, coordinate_note)
            seats.append(seat)
    return anchors, seats, coordinate_note


def _connectivity_pairs(anchors: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    candidates = [
        (name, list(map(float, value)))
        for name, value in anchors.items()
        if "view" not in name.casefold()
    ]
    if len(candidates) < 2:
        candidates = [(name, list(map(float, value))) for name, value in anchors.items()]
    if len(candidates) < 2:
        return []
    return [
        {
            "pair_id": f"anchor_{index:02d}_{left[0]}_to_{right[0]}",
            "start_m": left[1],
            "end_m": right[1],
        }
        for index, (left, right) in enumerate(zip(candidates, candidates[1:]))
    ][:8]


def _opening_documents(metadata: Mapping[str, Any]) -> list[dict[str, str]]:
    openings: list[dict[str, str]] = []
    for key in ("exterior_openings", "openings"):
        raw = metadata.get(key)
        if not isinstance(raw, list):
            continue
        for index, value in enumerate(raw):
            if not isinstance(value, Mapping):
                continue
            opening_id = value.get("opening_id") or value.get("link_id")
            if not isinstance(opening_id, str) or not opening_id:
                opening_id = f"{key}_{index:03d}"
            kind = str(value.get("kind", "other")).casefold()
            mapped_kind = (
                "door"
                if "door" in kind
                else "window"
                if "window" in kind or "glazing" in kind
                else "archway"
                if "opening" in kind or "arch" in kind
                else "other"
            )
            openings.append(
                {
                    "opening_id": opening_id,
                    "kind": mapped_kind,
                    "description": "Authored build opening declaration; acoustic geometry is the source GLB.",
                }
            )
    raw_links = metadata.get("links")
    if isinstance(raw_links, list):
        for index, value in enumerate(raw_links):
            if not isinstance(value, Mapping):
                continue
            link_id = value.get("link_id") or f"link_{index:03d}"
            if any(item["opening_id"] == link_id for item in openings):
                continue
            openings.append(
                {
                    "opening_id": str(link_id),
                    "kind": "archway",
                    "description": "Authored room link declaration; acoustic geometry is the source GLB.",
                }
            )
    count = metadata.get("exterior_opening_count")
    if not openings and isinstance(count, int) and count > 0:
        openings = [
            {
                "opening_id": f"build_report_exterior_opening_{index:03d}",
                "kind": "other",
                "description": "Opening count reported by authored build; exact topology is not supplied.",
            }
            for index in range(count)
        ]
    return openings


def _surface_self_hit_check(scene: ExpandedGltfScene) -> dict[str, Any]:
    for triangle in scene.triangles:
        points = scene.vertices[triangle].astype(np.float64)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length <= 1.0e-8:
            continue
        direction = normal / length
        centroid = points.mean(axis=0)
        epsilon = 0.02
        return {
            "check_id": "authored_surface_self_hit_control",
            "origin_m": (centroid + direction * epsilon).tolist(),
            "direction": (-direction).tolist(),
            "expectation": "hit_within_m",
            "distance_m": 2.0 * epsilon,
        }
    raise ValueError("source GLB contains no non-degenerate triangle")


def _classify_material(name: str, profile: Mapping[str, Any]) -> str:
    lowered = name.casefold()
    classes = profile["classes"]
    for class_name, definition in classes.items():
        if class_name == profile["default_class"]:
            continue
        tokens = definition.get("tokens", [])
        if any(str(token).casefold() in lowered for token in tokens):
            return class_name
    return str(profile["default_class"])


def _interleave(values: Sequence[float], bands: Sequence[float]) -> list[float]:
    result: list[float] = []
    for frequency, value in zip(bands, values):
        result.extend((float(frequency), float(value)))
    return result


def _material_documents(
    room_id: str,
    scene: ExpandedGltfScene,
    profile: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    names = sorted(set(scene.triangle_source_material_names))
    room_token = re.sub(r"[^a-z0-9]+", "_", room_id.casefold()).strip("_")[:24] or "room"
    bands = [float(value) for value in profile["bands_hz"]]
    entries: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    classifications: dict[str, str] = {}
    for index, source_name in enumerate(names):
        class_name = _classify_material(source_name, profile)
        definition = profile["classes"][class_name]
        category = f"avm_{room_token}_slot_{index:03d}_{class_name}"
        material_key = category
        entries.append(
            {
                "source_material_name": source_name,
                "material_id": index,
                "category_name": category,
                "material_key": material_key,
                "mapping_source": (
                    f"Explicit authored assumption class={class_name}; "
                    "visual PBR is not acoustic truth"
                ),
                "mapping_confidence": 0.25,
                "human_override": True,
                "randomized": False,
                "fallback": False,
            }
        )
        materials.append(
            {
                "material_key": material_key,
                "name": f"AVEngine authored assumption {class_name} {index:03d}",
                "labels": [category],
                "absorption": [float(value) for value in definition["absorption"]],
                "scattering": [float(value) for value in definition["scattering"]],
                "transmission": [float(value) for value in definition["transmission"]],
                "damping": [float(value) for value in definition["damping"]],
                "density": float(definition["density"]),
                "speed": float(definition["speed"]),
                "source": (
                    "Authored-room class assumption for research RLR exercise; "
                    "not measured physical material truth"
                ),
                "confidence": 0.25,
            }
        )
        classifications[source_name] = class_name
    mapping = {
        "schema": "avengine_m3_acoustic_material_mapping_v1",
        "mapping_id": f"{room_id}_authored_explicit_slots_v1",
        "room_id": room_id,
        "mapping_source_kind": "explicit_author_slot",
        "source_to_canonical": {
            "matrix_row_major": list(_IDENTITY_MATRIX),
            "source": "Authored GLB export is already right-handed +Y-up metres with -Z forward.",
            "reviewed": True,
        },
        "entries": entries,
    }
    database = {
        "schema": "avengine_m3_acoustic_material_database_v1",
        "database_id": f"{room_id}_authored_assumption_v1",
        "version": "1",
        "bands_hz": bands,
        "coefficient_units": {
            "absorption": "fraction_of_incident_sound_pressure",
            "scattering": "fraction_of_incident_sound_pressure",
            "transmission": "fraction_of_incident_sound_pressure",
            "damping": "decibels_per_meter",
            "density": "kilograms_per_cubic_meter",
            "speed": "meters_per_second",
        },
        "provenance": {
            "source": str(profile["source_note"]),
            "confidence": 0.25,
            "material_semantics": str(profile["material_semantics"]),
            "intended_use": str(profile.get("intended_use", "research_compiler_diagnostics")),
        },
        "materials": materials,
    }
    return mapping, database, classifications


def _room_manifest(
    *,
    room_id: str,
    metadata: Mapping[str, Any],
    build_dir: Path,
    source_glb: Path,
    source_usd: Path | None,
    source_manifest: Path | None,
    dataset_descriptor: Path,
    navmesh_descriptor: Path,
    anchor_path: Path | None,
    scene: ExpandedGltfScene,
    anchors: Mapping[str, Sequence[float]],
    coordinate_note: str,
) -> dict[str, Any]:
    assets: list[dict[str, str]] = [
        {"role": "render_surface_mesh", "path": str(source_glb)},
        {"role": "scene_dataset_config", "path": str(dataset_descriptor)},
        {"role": "navmesh", "path": str(navmesh_descriptor)},
    ]
    if source_usd is not None:
        assets.append({"role": "authored_usd_source", "path": str(source_usd)})
    if source_manifest is not None:
        assets.append({"role": "authored_build_metadata", "path": str(source_manifest)})
    if anchor_path is not None:
        assets.append({"role": "functional_anchors", "path": str(anchor_path)})
    return {
        "schema": "avengine_room_package_v1",
        "room_id": room_id,
        "room_kind": "external_usd_real_surface",
        "geometry_representation": "real_surface_mesh",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "scene": {
            "scene_id_kind": "handle",
            "scene_id": room_id,
            "dataset_config_path": str(dataset_descriptor),
            "navmesh_path": str(navmesh_descriptor),
            "navmesh_policy": "recompute_if_missing",
            "load_semantic_mesh": False,
            "enable_physics": False,
        },
        "assets": assets,
        "semantics": {
            "interpretation": (
                "Authored room source GLB surface materials plus declared "
                "functional anchors; no acoustic inference from visual PBR."
            ),
            "id_to_label": {},
        },
        "navigation": {
            "agent_height_m": 1.5,
            "agent_radius_m": 0.2,
            "include_static_objects": False,
        },
        "openings": _opening_documents(metadata),
        "connectivity_pairs": _connectivity_pairs(anchors),
        "ray_checks": [_surface_self_hit_check(scene)],
        "acoustics": {
            "status": "deferred_to_m3",
            "reason": (
                "M3/RLR package prepared from authored surface GLB; "
                "material assumptions remain research-placeholder inputs."
            ),
        },
        "provenance": {
            "source": (
                "AVEngine multi-home authored room build; source GLB is a "
                "real surface mesh and visual PBR is not acoustic truth."
            ),
            "source_revision": _source_revision(metadata, build_dir),
            "build_directory": str(build_dir),
            "source_manifest": str(source_manifest) if source_manifest else None,
            "source_usd": str(source_usd) if source_usd else None,
            "functional_anchor_coordinate_system": coordinate_note,
        },
        "surface_audit": {
            "method": (
                "Source authored GLB real surface mesh; no debug AABB proxy; "
                "visual PBR names are mapped only through an explicit research table."
            ),
            "aabb_proxy": False,
            "source_glb": str(source_glb),
            "triangle_count": int(len(scene.triangles)),
            "material_count": len(set(scene.triangle_source_material_names)),
            "usd_source": str(source_usd) if source_usd else None,
        },
    }


def prepare(
    *,
    build_dir: str | Path,
    output: str | Path,
    source_glb: str | None = None,
    source_usd: str | None = None,
    material_profile: str | Path | None = None,
    package_id: str | None = None,
) -> dict[str, Any]:
    build = Path(build_dir).expanduser().resolve()
    if not build.is_dir():
        raise ValueError(f"build directory does not exist: {build}")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"fresh output required: {destination}")
    metadata_path = _metadata_path(build)
    if metadata_path is None:
        raise ValueError(f"no authored room metadata found under {build}")
    metadata = _load_json(metadata_path)
    room_id = _room_id(metadata, build)
    geometry_path = _source_glb(build, source_glb)
    usd_path = _source_usd(build, source_usd)
    anchor_path = _anchor_path(build)
    anchor_map, seat_points, coordinate_note = _load_anchor_data(anchor_path)
    scene = extract_triangle_scene(geometry_path)
    profile_path = (
        Path(material_profile).expanduser().resolve()
        if material_profile
        else Path(__file__).resolve().parents[2]
        / "examples"
        / "acoustics"
        / "authored_room_material_assumptions_v1.json"
    )
    profile = _load_json(profile_path)
    mapping, database, classifications = _material_documents(room_id, scene, profile)

    destination.mkdir(parents=True)
    source_inputs = destination / "source_inputs"
    source_inputs.mkdir()
    dataset_descriptor = source_inputs / "scene_dataset_descriptor.json"
    navmesh_descriptor = source_inputs / "navmesh_descriptor.json"
    dataset_descriptor.write_text(
        json.dumps(
            {
                "schema": "avengine_authored_room_scene_descriptor_v1",
                "status": "research_candidate",
                "room_id": room_id,
                "source_glb": str(geometry_path),
                "source_usd": str(usd_path) if usd_path else None,
                "build_directory": str(build),
                "note": "Descriptor for acoustic package provenance; not a native navigation scene config.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    navmesh_descriptor.write_text(
        json.dumps(
            {
                "schema": "avengine_authored_room_navmesh_descriptor_v1",
                "status": "not_run",
                "room_id": room_id,
                "reason": "No native navmesh is claimed by this acoustics preparation step; AVEngine/SPEAR owns navigation.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    room = _room_manifest(
        room_id=room_id,
        metadata=metadata,
        build_dir=build,
        source_glb=geometry_path,
        source_usd=usd_path,
        source_manifest=metadata_path,
        dataset_descriptor=dataset_descriptor,
        navmesh_descriptor=navmesh_descriptor,
        anchor_path=anchor_path,
        scene=scene,
        anchors=anchor_map,
        coordinate_note=coordinate_note,
    )
    room_path = source_inputs / "room_manifest.json"
    mapping_path = source_inputs / "material_mapping.json"
    database_path = source_inputs / "material_database.json"
    _write_json(room_path, room)
    _write_json(mapping_path, mapping)
    _write_json(database_path, database)
    room_errors = validate_room_manifest(room)
    if room_errors:
        raise ValueError("generated room manifest is invalid: " + "; ".join(room_errors))
    package_path = destination / "package"
    manifest_path = compile_explicit_glb_research_scene(
        room_manifest=room_path,
        material_mapping=mapping_path,
        material_database=database_path,
        output=package_path,
        package_id=package_id or f"{room_id}_authored_rlr_research_v1",
    )
    report = {
        "schema": "avengine_authored_room_acoustic_preparation_v1",
        "status": "pass",
        "room_id": room_id,
        "build_directory": str(build),
        "source_glb": str(geometry_path),
        "source_usd": str(usd_path) if usd_path else None,
        "source_manifest": str(metadata_path),
        "functional_anchors": str(anchor_path) if anchor_path else None,
        "seat_points": seat_points,
        "material_profile": str(profile_path),
        "material_classifications": classifications,
        "material_semantics": profile["material_semantics"],
        "physical_acoustic_material_claim": False,
        "package_manifest": str(manifest_path),
        "claim_boundary": (
            "Real authored GLB geometry compiled through the existing M3/RLR "
            "compiler. Material coefficients are explicit research-placeholder "
            "assumptions; no native navigation or physical-material admission."
        ),
    }
    _write_json(destination / "preparation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-glb", type=str)
    parser.add_argument("--source-usd", type=str)
    parser.add_argument("--material-profile", type=Path)
    parser.add_argument("--package-id", type=str)
    args = parser.parse_args()
    report = prepare(
        build_dir=args.build_dir,
        output=args.output,
        source_glb=args.source_glb,
        source_usd=args.source_usd,
        material_profile=args.material_profile,
        package_id=args.package_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
