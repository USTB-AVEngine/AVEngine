from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
)
from avengine.m3.materials import (
    MATERIAL_QUALIFICATION_CLAIM,
    MaterialContractError,
    compile_materials,
    production_admission_errors,
    validate_material_database,
    validate_material_mapping,
)


PACKAGE_SCHEMA = "avengine_acoustic_scene_package_v1"
CANARY_REQUEST_SCHEMA = "avengine_m3_acoustic_canary_request_v1"
COMPILE_EVIDENCE_SCHEMA = "avengine_m3_compile_evidence_v1"

_SCHEMA_FILES = {
    PACKAGE_SCHEMA: "acoustic_scene_package_v1.schema.json",
    "avengine_m3_acoustic_material_mapping_v1": (
        "m3_acoustic_material_mapping_v1.schema.json"
    ),
    "avengine_m3_acoustic_material_database_v1": (
        "m3_acoustic_material_database_v1.schema.json"
    ),
    CANARY_REQUEST_SCHEMA: "m3_acoustic_canary_request_v1.schema.json",
    COMPILE_EVIDENCE_SCHEMA: "m3_compile_evidence_v1.schema.json",
}


class AcousticSceneContractError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class ImmutableFileSnapshot:
    """One path read exactly once for size/hash and every subsequent parse."""

    path: Path
    payload: bytes
    byte_size: int
    sha256: str


def read_immutable_file_snapshot(
    path: str | Path,
    *,
    cache: dict[Path, ImmutableFileSnapshot] | None = None,
) -> ImmutableFileSnapshot:
    resolved = Path(path).resolve()
    if cache is not None and resolved in cache:
        return cache[resolved]
    payload = resolved.read_bytes()
    snapshot = ImmutableFileSnapshot(
        path=resolved,
        payload=payload,
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    if cache is not None:
        cache[resolved] = snapshot
    return snapshot


@dataclass(frozen=True)
class ValidatedAcousticScenePackage:
    manifest_path: Path
    package_root: Path
    manifest: dict[str, Any]
    manifest_file_sha256: str
    manifest_byte_size: int
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_material_ids: np.ndarray
    material_categories: dict[str, Any]
    material_categories_path: Path
    material_categories_bytes: bytes
    rlr_material_database: dict[str, Any]
    rlr_material_database_path: Path
    rlr_material_database_bytes: bytes
    source_material_mapping: dict[str, Any]
    source_material_database: dict[str, Any]
    qa_reports: dict[str, dict[str, Any]]

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    @property
    def object_count(self) -> int:
        return len(self.manifest["objects"])

    @property
    def material_category_count(self) -> int:
        return len(self.material_categories["categories"])

    @property
    def category_triangle_counts(self) -> dict[str, int]:
        return {
            category["category_name"]: int(
                np.count_nonzero(
                    self.triangle_material_ids == int(category["material_id"])
                )
            )
            for category in self.material_categories["categories"]
        }


def _schema_path(schema_name: str) -> Path:
    filename = _SCHEMA_FILES[schema_name]
    source = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(f"AVEngine schema is unavailable: {filename}")
    return path


def json_schema_errors(value: Any, schema_name: str) -> list[str]:
    schema = load_json(_schema_path(schema_name))
    errors: list[str] = []
    validator = Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def _all_numbers_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    return False


def validate_mapping_document(
    mapping: Mapping[str, Any], *, room_id: str | None = None
) -> list[str]:
    return [
        *json_schema_errors(mapping, "avengine_m3_acoustic_material_mapping_v1"),
        *validate_material_mapping(mapping, room_id=room_id),
    ]


def validate_material_database_document(database: Mapping[str, Any]) -> list[str]:
    return [
        *json_schema_errors(
            database, "avengine_m3_acoustic_material_database_v1"
        ),
        *validate_material_database(database),
    ]


def validate_canary_request(request: Mapping[str, Any]) -> list[str]:
    errors = json_schema_errors(request, CANARY_REQUEST_SCHEMA)
    if not _all_numbers_finite(request):
        errors.append("canary request contains a non-finite number")
    listener = request.get("listener")
    source = request.get("source")
    if isinstance(listener, Mapping) and isinstance(source, Mapping):
        try:
            listener_position = np.asarray(listener["position_m"], dtype=np.float64)
            source_position = np.asarray(source["position_m"], dtype=np.float64)
            if np.linalg.norm(listener_position - source_position) <= 1e-6:
                errors.append("canary source and listener positions must be distinct")
        except (KeyError, TypeError, ValueError):
            pass
    return errors


def _record_snapshot(
    package_root: Path,
    record: Mapping[str, Any],
    owner: str,
    errors: list[str],
    *,
    cache: dict[Path, ImmutableFileSnapshot],
) -> ImmutableFileSnapshot | None:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"{owner}.path must be a non-empty relative path")
        return None
    declared = Path(raw_path)
    if declared.is_absolute() or ".." in declared.parts:
        errors.append(f"{owner}.path must be confined to the package")
        return None
    path = (package_root / declared).resolve()
    try:
        path.relative_to(package_root)
    except ValueError:
        errors.append(f"{owner}.path or symlink escapes the package")
        return None
    try:
        snapshot = read_immutable_file_snapshot(path, cache=cache)
    except OSError as exc:
        errors.append(f"{owner} is missing or unreadable: {raw_path}: {exc}")
        return None
    if record.get("byte_size") != snapshot.byte_size:
        errors.append(f"{owner}.byte_size does not match {raw_path}")
    if record.get("sha256") != snapshot.sha256:
        errors.append(f"{owner}.sha256 does not match {raw_path}")
    return snapshot


def _manifest_file_records(manifest: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any]]] = []
    arrays = manifest.get("arrays")
    if isinstance(arrays, Mapping):
        for name in ("vertices", "triangles", "triangle_material_ids"):
            record = arrays.get(name)
            if isinstance(record, Mapping):
                records.append((f"arrays.{name}", record))
    materials = manifest.get("materials")
    if isinstance(materials, Mapping):
        for name in (
            "source_mapping",
            "source_database",
            "categories",
            "rlr_database",
        ):
            record = materials.get(name)
            if isinstance(record, Mapping):
                records.append((f"materials.{name}", record))
    qa = manifest.get("qa")
    if isinstance(qa, Mapping):
        for name in (
            "geometry_report",
            "material_coverage",
            "ray_leakage",
            "compiler_source_to_package_parity",
        ):
            record = qa.get(name)
            if isinstance(record, Mapping):
                records.append((f"qa.{name}", record))
    debug = manifest.get("debug_mesh")
    if isinstance(debug, Mapping):
        records.append(("debug_mesh", debug))
    return records


def validate_package_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors = json_schema_errors(manifest, PACKAGE_SCHEMA)
    if not _all_numbers_finite(manifest):
        errors.append("package manifest contains a non-finite number")
    if manifest.get("schema") != PACKAGE_SCHEMA:
        errors.append(f"manifest.schema must be {PACKAGE_SCHEMA!r}")
    content_hash = manifest.get("package_content_sha256")
    try:
        actual_hash = canonical_json_sha256(
            {
                key: value
                for key, value in manifest.items()
                if key != "package_content_sha256"
            }
        )
    except (TypeError, ValueError):
        actual_hash = None
    if content_hash != actual_hash:
        errors.append("package_content_sha256 does not match canonical manifest content")

    geometry = manifest.get("geometry")
    if isinstance(geometry, Mapping):
        if (
            manifest.get("package_mode") == "production"
            and geometry.get("representation") == "debug_aabb_proxy"
        ):
            errors.append("production package cannot use debug AABB geometry")
        transform = geometry.get("source_to_canonical")
        if isinstance(transform, Mapping):
            if (
                manifest.get("package_mode") == "production"
                and transform.get("reviewed") is not True
            ):
                errors.append(
                    "production geometry.source_to_canonical must be explicitly reviewed"
                )
            matrix = transform.get("matrix_row_major")
            if isinstance(matrix, list) and len(matrix) == 16:
                matrix_array = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
                if not np.allclose(matrix_array[3], [0, 0, 0, 1], atol=1e-9):
                    errors.append("geometry.source_to_canonical must be affine")
                if abs(float(np.linalg.det(matrix_array[:3, :3]))) <= 1e-12:
                    errors.append("geometry.source_to_canonical must be nonsingular")

    objects = manifest.get("objects")
    if isinstance(objects, list):
        object_ids = [
            item.get("object_id") for item in objects if isinstance(item, Mapping)
        ]
        if len(object_ids) != len(set(object_ids)):
            errors.append("package object_id values must be unique")
    compiler = manifest.get("compiler")
    if isinstance(compiler, Mapping) and isinstance(compiler.get("components"), Mapping):
        if compiler.get("implementation_sha256") != canonical_json_sha256(
            compiler["components"]
        ):
            errors.append("compiler implementation_sha256 is not its component hash")
    materials = manifest.get("materials")
    if isinstance(materials, Mapping):
        semantics = materials.get("material_semantics")
        expected_claim = MATERIAL_QUALIFICATION_CLAIM.get(semantics)
        if expected_claim is None or materials.get("qualification_claim") != expected_claim:
            errors.append(
                "materials.qualification_claim does not match material_semantics"
            )
        if manifest.get("package_mode") == "production" and semantics not in {
            "controlled_canary",
            "reviewed_physical",
        }:
            errors.append(
                "production materials must claim controlled_canary or reviewed_physical semantics"
            )
        if (
            manifest.get("package_mode") == "production"
            and materials.get("mapping_source_kind") != "explicit_author_slot"
        ):
            errors.append(
                "production materials.mapping_source_kind must be explicit_author_slot"
            )
    return errors


def _load_npy(
    snapshot: ImmutableFileSnapshot,
    record: Mapping[str, Any],
    *,
    expected_dtype: str,
    owner: str,
    errors: list[str],
) -> np.ndarray | None:
    try:
        value = np.load(io.BytesIO(snapshot.payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        errors.append(f"{owner} is not a valid non-pickle NPY file: {exc}")
        return None
    if value.dtype.str != expected_dtype:
        errors.append(f"{owner} dtype must be {expected_dtype}, got {value.dtype.str}")
    if not value.flags.c_contiguous:
        errors.append(f"{owner} must be C-contiguous")
    declared_shape = record.get("shape")
    if not isinstance(declared_shape, list) or list(value.shape) != declared_shape:
        errors.append(f"{owner} shape does not match its manifest descriptor")
    return value


def _load_json_object_snapshot(
    snapshot: ImmutableFileSnapshot,
    *,
    owner: str,
    errors: list[str],
) -> dict[str, Any]:
    try:
        value = json.loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{owner} is not valid UTF-8 JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{owner} JSON root must be an object")
        return {}
    return value


def _validate_object_ranges(
    manifest: Mapping[str, Any], triangles: np.ndarray, errors: list[str]
) -> None:
    objects = manifest.get("objects")
    geometry = manifest.get("geometry")
    if not isinstance(objects, list) or not isinstance(geometry, Mapping):
        return
    expected_vertex = 0
    expected_triangle = 0
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            continue
        if item.get("vertex_offset") != expected_vertex:
            errors.append(f"objects[{index}] vertex range is not contiguous")
        if item.get("triangle_offset") != expected_triangle:
            errors.append(f"objects[{index}] triangle range is not contiguous")
        vertex_count = item.get("vertex_count")
        triangle_count = item.get("triangle_count")
        if not isinstance(vertex_count, int) or not isinstance(triangle_count, int):
            continue
        start = expected_triangle
        stop = start + triangle_count
        object_triangles = triangles[start:stop]
        minimum = expected_vertex
        maximum = expected_vertex + vertex_count
        if object_triangles.size and (
            int(object_triangles.min()) < minimum
            or int(object_triangles.max()) >= maximum
        ):
            errors.append(f"objects[{index}] triangles escape its vertex range")
        world_from_object = item.get("world_from_object")
        if isinstance(world_from_object, list) and len(world_from_object) == 16:
            if not np.allclose(
                np.asarray(world_from_object, dtype=np.float64).reshape(4, 4),
                np.eye(4),
                atol=1e-9,
            ):
                errors.append(
                    f"objects[{index}].world_from_object must be identity when transforms are baked"
                )
        expected_vertex += vertex_count
        expected_triangle += triangle_count
    if expected_vertex != geometry.get("vertex_count"):
        errors.append("object vertex ranges do not cover geometry.vertex_count")
    if expected_triangle != geometry.get("triangle_count"):
        errors.append("object triangle ranges do not cover geometry.triangle_count")


def _validate_material_files(
    categories: Mapping[str, Any],
    rlr_database: Mapping[str, Any],
    material_ids: np.ndarray,
    manifest: Mapping[str, Any],
    errors: list[str],
) -> None:
    if categories.get("schema") != "avengine_acoustic_material_categories_v1":
        errors.append("material categories schema is invalid")
    raw_categories = categories.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        errors.append("material categories must be a non-empty array")
        return
    category_count = manifest.get("materials", {}).get("category_count")
    if category_count != len(raw_categories):
        errors.append("materials.category_count does not match material_categories.json")
    category_ids = [
        value.get("material_id")
        for value in raw_categories
        if isinstance(value, Mapping)
    ]
    if category_ids != list(range(len(raw_categories))):
        errors.append("material category IDs must be ordered and contiguous from zero")
    if any(
        not isinstance(value, Mapping) or value.get("fallback") is not False
        for value in raw_categories
    ):
        errors.append("all material categories must explicitly disable fallback")
    if material_ids.size:
        used = sorted(int(value) for value in np.unique(material_ids))
        if used != list(range(len(raw_categories))):
            errors.append("triangle material IDs must use every declared category exactly in range")

    rlr_materials = rlr_database.get("materials")
    if not isinstance(rlr_materials, list) or not rlr_materials:
        errors.append("RLR material database must contain materials")
        return
    all_labels: list[str] = []
    for material_index, material in enumerate(rlr_materials):
        prefix = f"RLR materials[{material_index}]"
        if not isinstance(material, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        expected_keys = {
            "name",
            "labels",
            "absorption",
            "scattering",
            "transmission",
            "damping",
            "density",
            "speed",
        }
        if set(material) != expected_keys:
            errors.append(f"{prefix} fields must be exactly {sorted(expected_keys)}")
        labels = material.get("labels")
        if not isinstance(labels, list) or not labels:
            errors.append(f"{prefix}.labels must be non-empty")
        else:
            for label in labels:
                if not isinstance(label, str) or not label or label != label.lower():
                    errors.append(f"{prefix}.labels must be non-empty lowercase strings")
                else:
                    all_labels.append(label)
        frequencies: list[float] | None = None
        for field in ("absorption", "scattering", "transmission", "damping"):
            curve = material.get(field)
            if not isinstance(curve, list) or len(curve) < 2 or len(curve) % 2:
                errors.append(f"{prefix}.{field} must be frequency/value pairs")
                continue
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in curve
            ):
                errors.append(f"{prefix}.{field} must contain finite numbers")
                continue
            current_frequencies = [float(value) for value in curve[0::2]]
            values = [float(value) for value in curve[1::2]]
            if any(value <= 0 for value in current_frequencies) or any(
                left >= right
                for left, right in zip(current_frequencies, current_frequencies[1:])
            ):
                errors.append(f"{prefix}.{field} frequencies must increase")
            if frequencies is None:
                frequencies = current_frequencies
            elif frequencies != current_frequencies:
                errors.append(f"{prefix} coefficient curves must use identical bands")
            if field == "damping":
                if any(value < 0 for value in values):
                    errors.append(f"{prefix}.damping values must be non-negative")
            elif any(not 0 <= value <= 1 for value in values):
                errors.append(f"{prefix}.{field} values must be in [0, 1]")
        for physical_field in ("density", "speed"):
            value = material.get(physical_field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                errors.append(
                    f"{prefix}.{physical_field} must be positive and finite"
                )
    if len(all_labels) != len(set(all_labels)):
        errors.append("RLR material labels must be globally unique")

    for index, category in enumerate(raw_categories):
        if not isinstance(category, Mapping):
            continue
        name = category.get("category_name")
        if not isinstance(name, str):
            continue
        scores: list[tuple[str, int]] = []
        for material in rlr_materials:
            if not isinstance(material, Mapping):
                continue
            material_name = str(material.get("name", ""))
            labels = material.get("labels")
            raw_labels = labels if isinstance(labels, list) else []
            score = sum(
                1
                for label in raw_labels
                if isinstance(label, str) and label.lower() in name.lower()
            )
            scores.append((material_name, score))
        highest = max((score for _, score in scores), default=0)
        winners = [material_name for material_name, score in scores if score == highest]
        if highest <= 0 or winners != [category.get("rlr_material_name")]:
            errors.append(
                f"category {index} does not uniquely match its declared RLR material"
            )


def _replay_material_source_inputs(
    *,
    manifest: Mapping[str, Any],
    mapping: Mapping[str, Any],
    database: Mapping[str, Any],
    categories: Mapping[str, Any],
    rlr_database: Mapping[str, Any],
    material_ids: np.ndarray,
    errors: list[str],
) -> None:
    """Recompile packaged material inputs instead of trusting derived files."""

    room_id = manifest.get("source_room", {}).get("room_id")
    mapping_errors = validate_mapping_document(
        mapping, room_id=room_id if isinstance(room_id, str) else None
    )
    database_errors = validate_material_database_document(database)
    errors.extend(f"source mapping replay: {error}" for error in mapping_errors)
    errors.extend(f"source database replay: {error}" for error in database_errors)
    materials_manifest = manifest.get("materials", {})
    if isinstance(materials_manifest, Mapping):
        if materials_manifest.get("mapping_sha256") != materials_manifest.get(
            "source_mapping", {}
        ).get("sha256"):
            errors.append("materials.mapping_sha256 does not bind source_mapping")
        if materials_manifest.get("database_source_sha256") != materials_manifest.get(
            "source_database", {}
        ).get("sha256"):
            errors.append(
                "materials.database_source_sha256 does not bind source_database"
            )
        if materials_manifest.get("mapping_id") != mapping.get("mapping_id"):
            errors.append("materials.mapping_id does not match source_mapping")
        if materials_manifest.get("mapping_source_kind") != mapping.get(
            "mapping_source_kind"
        ):
            errors.append(
                "materials.mapping_source_kind does not match source_mapping"
            )
        if materials_manifest.get("database_id") != database.get("database_id"):
            errors.append("materials.database_id does not match source_database")
        provenance = database.get("provenance", {})
        if materials_manifest.get("material_semantics") != provenance.get(
            "material_semantics"
        ):
            errors.append(
                "materials.material_semantics does not match source_database"
            )
        expected_claim = MATERIAL_QUALIFICATION_CLAIM.get(
            provenance.get("material_semantics")
        )
        if materials_manifest.get("qualification_claim") != expected_claim:
            errors.append(
                "materials.qualification_claim does not match source_database"
            )
    if manifest.get("package_mode") == "production":
        errors.extend(
            f"production material replay: {error}"
            for error in production_admission_errors(mapping, database)
        )
    if mapping_errors or database_errors or not isinstance(room_id, str):
        return
    try:
        compiled = compile_materials(mapping, database, room_id=room_id)
    except MaterialContractError as exc:
        errors.extend(f"material replay compile: {error}" for error in exc.errors)
        return
    if compiled.categories_document != categories:
        errors.append(
            "packaged material_categories.json does not equal source-input replay"
        )
    if compiled.rlr_database != rlr_database:
        errors.append(
            "packaged material_database.json does not equal source-input replay"
        )
    expected = np.full(material_ids.shape, np.iinfo(np.uint32).max, dtype="<u4")
    for index, item in enumerate(manifest.get("objects", [])):
        if not isinstance(item, Mapping):
            continue
        source_name = item.get("source_material_name")
        material_id = compiled.source_material_to_id.get(source_name)
        start = item.get("triangle_offset")
        count = item.get("triangle_count")
        if material_id is None:
            errors.append(
                f"objects[{index}].source_material_name has no replayed mapping"
            )
            continue
        if not isinstance(start, int) or not isinstance(count, int):
            continue
        stop = start + count
        if start < 0 or stop > len(expected):
            continue
        expected[start:stop] = material_id
    if not np.array_equal(expected, material_ids):
        errors.append(
            "triangle_material_ids.npy does not equal source mapping/object-range replay"
        )


def _validate_qa_reports(
    *,
    manifest: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    vertices: np.ndarray,
    triangles: np.ndarray,
    material_ids: np.ndarray,
    categories: Mapping[str, Any],
    debug_obj_payload: bytes,
    errors: list[str],
) -> None:
    from avengine.m3.qa import (
        array_sha256,
        debug_obj_array_parity_bytes,
        triangle_areas,
    )

    expected_schemas = {
        "geometry_report": "avengine_m3_geometry_report_v1",
        "material_coverage": "avengine_m3_material_coverage_v1",
        "ray_leakage": "avengine_m3_ray_leakage_v1",
        "compiler_source_to_package_parity": (
            "avengine_m3_compiler_source_to_package_parity_v1"
        ),
    }
    production = manifest.get("package_mode") == "production"
    for name, expected_schema in expected_schemas.items():
        report = reports.get(name)
        if not isinstance(report, Mapping):
            errors.append(f"QA report {name} is missing or invalid")
            continue
        if report.get("schema") != expected_schema:
            errors.append(f"QA report {name} schema is invalid")
        if not _all_numbers_finite(report):
            errors.append(f"QA report {name} contains non-finite numbers")
        if production and report.get("status") != "pass":
            errors.append(f"production QA report {name} must have status pass")

    geometry = reports.get("geometry_report", {})
    if isinstance(geometry, Mapping):
        if geometry.get("vertex_count") != len(vertices):
            errors.append("geometry_report vertex_count does not match arrays")
        if geometry.get("triangle_count") != len(triangles):
            errors.append("geometry_report triangle_count does not match arrays")
        if geometry.get("object_count") != len(manifest.get("objects", [])):
            errors.append("geometry_report object_count does not match manifest")
        if geometry.get("source_geometry_sha256") != manifest.get("source_room", {}).get(
            "geometry_asset_sha256"
        ):
            errors.append("geometry_report source geometry hash does not match manifest")
        hashes = geometry.get("array_hashes", {})
        if not isinstance(hashes, Mapping) or hashes.get("vertices") != array_sha256(
            vertices
        ):
            errors.append("geometry_report vertices hash does not match arrays")
        if not isinstance(hashes, Mapping) or hashes.get("triangles") != array_sha256(
            triangles
        ):
            errors.append("geometry_report triangles hash does not match arrays")
        topology = geometry.get("topology", {})
        if production and (
            not isinstance(topology, Mapping)
            or topology.get("degenerate_triangle_count") != 0
            or topology.get("duplicate_triangle_count") != 0
            or topology.get("per_object_boundary_edge_count_after_exact_weld") != 0
            or topology.get("per_object_nonmanifold_edge_count_after_exact_weld") != 0
            or geometry.get("aabb_proxy") is not False
        ):
            errors.append("production geometry_report topology/AABB gates do not pass")
    areas = triangle_areas(vertices, triangles)
    if not np.isfinite(areas).all():
        errors.append("package arrays contain non-finite triangle areas")
    if production and np.any(areas <= 0):
        errors.append("production package arrays contain zero-area triangles")

    coverage = reports.get("material_coverage", {})
    if isinstance(coverage, Mapping):
        if coverage.get("triangle_count") != len(triangles):
            errors.append("material_coverage triangle_count does not match arrays")
        if coverage.get("assigned_triangle_count") != len(triangles):
            errors.append("material_coverage does not assign every triangle")
        if coverage.get("coverage_fraction") != 1.0:
            errors.append("material_coverage coverage_fraction must be 1.0")
        if coverage.get("fallback_triangle_count") != 0:
            errors.append("material_coverage fallback_triangle_count must be zero")
        if coverage.get("unmatched_triangle_count") != 0:
            errors.append("material_coverage unmatched_triangle_count must be zero")
        raw_coverage_categories = coverage.get("categories")
        raw_categories = categories.get("categories")
        if isinstance(raw_coverage_categories, list) and isinstance(raw_categories, list):
            expected_counts = {
                int(category["material_id"]): int(
                    np.count_nonzero(material_ids == int(category["material_id"]))
                )
                for category in raw_categories
            }
            measured_counts = {
                int(category["material_id"]): int(category["triangle_count"])
                for category in raw_coverage_categories
                if isinstance(category, Mapping)
                and isinstance(category.get("material_id"), int)
                and isinstance(category.get("triangle_count"), int)
            }
            if measured_counts != expected_counts:
                errors.append("material_coverage per-category counts do not match arrays")
        else:
            errors.append("material_coverage categories must be an array")

    leakage = reports.get("ray_leakage", {})
    if isinstance(leakage, Mapping):
        checks = leakage.get("checks")
        if not isinstance(checks, list) or leakage.get("declared_check_count") != len(
            checks
        ):
            errors.append("ray_leakage declared_check_count does not match checks")
        elif production and (
            not checks or any(check.get("status") != "pass" for check in checks)
        ):
            errors.append("production ray_leakage must contain passing declared rays")

    parity = reports.get("compiler_source_to_package_parity", {})
    if isinstance(parity, Mapping):
        replayed_debug_obj = debug_obj_array_parity_bytes(
            debug_obj_payload, vertices, triangles
        )
        if parity.get("debug_obj_parity") != replayed_debug_obj:
            errors.append(
                "compiler_source_to_package_parity debug OBJ snapshot replay differs"
            )
        if parity.get("source_geometry_sha256") != manifest.get("source_room", {}).get(
            "geometry_asset_sha256"
        ):
            errors.append(
                "compiler_source_to_package_parity source hash does not match manifest"
            )
        if parity.get("package_vertex_count") != len(vertices) or parity.get(
            "package_triangle_count"
        ) != len(triangles):
            errors.append(
                "compiler_source_to_package_parity counts do not match arrays"
            )
        parity_hashes = parity.get("array_hashes", {})
        if (
            not isinstance(parity_hashes, Mapping)
            or parity_hashes.get("emitted_npy_vertices") != array_sha256(vertices)
            or parity_hashes.get("emitted_npy_triangles") != array_sha256(triangles)
            or parity_hashes.get("canonical_expected_vertices")
            != array_sha256(vertices)
            or parity_hashes.get("canonical_expected_triangles")
            != array_sha256(triangles)
        ):
            errors.append(
                "compiler_source_to_package_parity array hashes do not match arrays"
            )
        if production and (
            parity.get("expected_vertex_bytes_identical_to_npy") is not True
            or parity.get("expected_triangle_bytes_identical_to_npy") is not True
            or parity.get("bounds_identical_within_m") is not True
            or parity.get("visual_runtime_parity_claim") is not False
            or parity.get("debug_obj_parity", {}).get(
                "vertex_values_float32_roundtrip_identical"
            )
            is not True
            or parity.get("debug_obj_parity", {}).get("triangle_indices_identical")
            is not True
            or parity.get("debug_obj_parity", {}).get("malformed_line_count") != 0
        ):
            errors.append(
                "production compiler source-to-package parity gates do not pass"
            )


def load_and_validate_package(
    manifest_path: str | Path,
    *,
    manifest_snapshot: ImmutableFileSnapshot | None = None,
    snapshot_cache: dict[Path, ImmutableFileSnapshot] | None = None,
) -> ValidatedAcousticScenePackage:
    path = Path(manifest_path).resolve()
    cache = snapshot_cache if snapshot_cache is not None else {}
    if manifest_snapshot is None:
        try:
            manifest_snapshot = read_immutable_file_snapshot(path, cache=cache)
        except OSError as exc:
            raise AcousticSceneContractError(
                [f"package manifest is missing or unreadable: {exc}"]
            ) from exc
    elif manifest_snapshot.path.resolve() != path:
        raise AcousticSceneContractError(
            ["provided manifest snapshot path does not match manifest_path"]
        )
    else:
        cached = cache.get(path)
        if cached is not None and cached.payload != manifest_snapshot.payload:
            raise AcousticSceneContractError(
                ["snapshot cache contains conflicting package manifest bytes"]
            )
        cache[path] = manifest_snapshot
    manifest_errors: list[str] = []
    manifest = _load_json_object_snapshot(
        manifest_snapshot, owner="package manifest", errors=manifest_errors
    )
    if manifest_errors:
        raise AcousticSceneContractError(manifest_errors)
    errors = validate_package_manifest(manifest)
    root = path.parent.resolve()
    resolved: dict[str, ImmutableFileSnapshot] = {}
    for owner, record in _manifest_file_records(manifest):
        record_snapshot = _record_snapshot(
            root, record, owner, errors, cache=cache
        )
        if record_snapshot is not None:
            resolved[owner] = record_snapshot

    arrays = manifest.get("arrays", {})
    vertices = triangles = material_ids = None
    if "arrays.vertices" in resolved and isinstance(arrays, Mapping):
        vertices = _load_npy(
            resolved["arrays.vertices"],
            arrays["vertices"],
            expected_dtype="<f4",
            owner="arrays.vertices",
            errors=errors,
        )
    if "arrays.triangles" in resolved and isinstance(arrays, Mapping):
        triangles = _load_npy(
            resolved["arrays.triangles"],
            arrays["triangles"],
            expected_dtype="<u4",
            owner="arrays.triangles",
            errors=errors,
        )
    if "arrays.triangle_material_ids" in resolved and isinstance(arrays, Mapping):
        material_ids = _load_npy(
            resolved["arrays.triangle_material_ids"],
            arrays["triangle_material_ids"],
            expected_dtype="<u4",
            owner="arrays.triangle_material_ids",
            errors=errors,
        )

    geometry = manifest.get("geometry", {})
    if vertices is not None:
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            errors.append("vertices must have shape [N, 3]")
        if not np.isfinite(vertices).all():
            errors.append("vertices contain non-finite values")
        if len(vertices) != geometry.get("vertex_count"):
            errors.append("vertices length does not match geometry.vertex_count")
    if triangles is not None:
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            errors.append("triangles must have shape [T, 3]")
        elif vertices is not None and triangles.size and int(triangles.max()) >= len(vertices):
            errors.append("triangles contain an out-of-range vertex index")
        if len(triangles) != geometry.get("triangle_count"):
            errors.append("triangles length does not match geometry.triangle_count")
    if material_ids is not None:
        if material_ids.ndim != 1:
            errors.append("triangle_material_ids must have shape [T]")
        if triangles is not None and len(material_ids) != len(triangles):
            errors.append("triangle_material_ids length must equal triangle count")

    categories: dict[str, Any] = {}
    rlr_database: dict[str, Any] = {}
    source_mapping: dict[str, Any] = {}
    source_database: dict[str, Any] = {}
    if "materials.source_mapping" in resolved:
        source_mapping = _load_json_object_snapshot(
            resolved["materials.source_mapping"],
            owner="materials.source_mapping",
            errors=errors,
        )
    if "materials.source_database" in resolved:
        source_database = _load_json_object_snapshot(
            resolved["materials.source_database"],
            owner="materials.source_database",
            errors=errors,
        )
    if "materials.categories" in resolved:
        categories = _load_json_object_snapshot(
            resolved["materials.categories"],
            owner="materials.categories",
            errors=errors,
        )
    if "materials.rlr_database" in resolved:
        rlr_database = _load_json_object_snapshot(
            resolved["materials.rlr_database"],
            owner="materials.rlr_database",
            errors=errors,
        )
    if material_ids is not None and categories and rlr_database:
        _validate_material_files(categories, rlr_database, material_ids, manifest, errors)
        if source_mapping and source_database:
            _replay_material_source_inputs(
                manifest=manifest,
                mapping=source_mapping,
                database=source_database,
                categories=categories,
                rlr_database=rlr_database,
                material_ids=material_ids,
                errors=errors,
            )
    if triangles is not None:
        _validate_object_ranges(manifest, triangles, errors)

    qa_reports: dict[str, dict[str, Any]] = {}
    for name in (
        "geometry_report",
        "material_coverage",
        "ray_leakage",
        "compiler_source_to_package_parity",
    ):
        owner = f"qa.{name}"
        if owner in resolved:
            report = _load_json_object_snapshot(
                resolved[owner], owner=owner, errors=errors
            )
            if report:
                qa_reports[name] = report
    if (
        vertices is not None
        and triangles is not None
        and material_ids is not None
        and categories
        and "debug_mesh" in resolved
    ):
        _validate_qa_reports(
            manifest=manifest,
            reports=qa_reports,
            vertices=vertices,
            triangles=triangles,
            material_ids=material_ids,
            categories=categories,
            debug_obj_payload=resolved["debug_mesh"].payload,
            errors=errors,
        )
    if errors:
        raise AcousticSceneContractError(errors)
    assert vertices is not None
    assert triangles is not None
    assert material_ids is not None
    vertices.setflags(write=False)
    triangles.setflags(write=False)
    material_ids.setflags(write=False)
    return ValidatedAcousticScenePackage(
        manifest_path=path,
        package_root=root,
        manifest=manifest,
        manifest_file_sha256=manifest_snapshot.sha256,
        manifest_byte_size=manifest_snapshot.byte_size,
        vertices=vertices,
        triangles=triangles,
        triangle_material_ids=material_ids,
        material_categories=categories,
        material_categories_path=resolved["materials.categories"].path,
        material_categories_bytes=resolved["materials.categories"].payload,
        rlr_material_database=rlr_database,
        rlr_material_database_path=resolved["materials.rlr_database"].path,
        rlr_material_database_bytes=resolved["materials.rlr_database"].payload,
        source_material_mapping=source_mapping,
        source_material_database=source_database,
        qa_reports=qa_reports,
    )


def validate_package(path: str | Path) -> list[str]:
    try:
        load_and_validate_package(path)
    except AcousticSceneContractError as exc:
        return exc.errors
    except (OSError, ValueError) as exc:
        return [str(exc)]
    return []


def load_and_validate_acoustic_scene_package(
    manifest_path: str | Path,
    *,
    manifest_snapshot: ImmutableFileSnapshot | None = None,
    snapshot_cache: dict[Path, ImmutableFileSnapshot] | None = None,
) -> ValidatedAcousticScenePackage:
    """Stable runtime-facing alias for the strict package loader."""

    return load_and_validate_package(
        manifest_path,
        manifest_snapshot=manifest_snapshot,
        snapshot_cache=snapshot_cache,
    )
