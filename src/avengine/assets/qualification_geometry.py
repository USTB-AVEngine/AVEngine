"""Measured geometry observations for source-asset qualification.

:func:`avengine.dataset.source_asset_qualification.build_qualification_matrix`
derives every dimension status itself from observation catalogs.  This module
produces those catalogs by measuring the real artefacts:

* asset contact geometry from the registered GLB, read with the repository's
  own glTF reader (:func:`avengine.acoustics.gltf.extract_triangle_scene`);
* articulated foot support from the retained per-frame skinned-vertex
  grounding audit written by ``tools/assets/audit_habitat_mesh_grounding.py``;
* room support surfaces back-projected from retained native metric depth and
  semantic frames;
* room clearance from the room's own visual triangle mesh.

Nothing here decides whether an asset qualifies.  No function returns a
qualification status, a verdict or an eligibility flag, and every row names
the file it was measured from together with what it does not establish.  A
missing input produces a ``not_run`` observation naming the path or capability
that is absent, never a substituted value.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from avengine.acoustics.gltf import extract_triangle_scene

SCHEMA_GEOMETRY = "avengine_source_qualification_geometry_catalog_v1"
SCHEMA_SURFACES = "avengine_support_surface_catalog_v1"
SCHEMA_CLEARANCE = "avengine_source_placement_room_clearance_v1"

#: A contact plane counts as horizontal while its normal is within this angle
#: of world up.  Taken from the boundary-plane selection already used to build
#: the T06 support catalog, not chosen here.
HORIZONTAL_NORMAL_MAX_TILT_DEG = 20.0

#: Boundary-face band as a fraction of the extent along the tested axis.  Both
#: values reproduce the T06 asset measurement.
HORIZONTAL_BAND_FRACTION = 0.02
HORIZONTAL_BAND_FLOOR_M = 0.0015
MOUNT_BAND_FRACTION = 0.014
MOUNT_BAND_FLOOR_M = 0.001

#: Room contact tolerance.  Re-exported from the qualification module so the
#: clearance observation and the placement check share one number.
from avengine.dataset.source_asset_qualification import (  # noqa: E402
    DEFAULT_SUPPORT_PLANE_TOLERANCE_M as DEFAULT_CONTACT_TOLERANCE_M,
)

#: Why a clear bounding box is a sound result.  The asset's world AABB encloses
#: its mesh, so a box that no room triangle penetrates encloses a mesh that no
#: room triangle penetrates.  The converse does not hold, which is why an
#: overlapping box is reported as unresolved rather than as a failure.
_CLEAR_BOX_REASON = (
    "no room triangle reaches inside the placed world bounding box beyond the "
    "contact tolerance; the box encloses the asset mesh, so the mesh is clear too"
)

CONTACT_PLANE_HORIZONTAL_BASE = "horizontal_base"
CONTACT_PLANE_HORIZONTAL_TOP = "horizontal_top"
CONTACT_PLANE_VERTICAL_MOUNT = "vertical_mount"

#: Room surface kinds whose measured normal points up, i.e. the kinds a
#: horizontal-based asset could rest on.  Which one a given asset belongs to is
#: a room-level fact, not a mesh fact; see :func:`resolve_support_kind`.
UP_FACING_SURFACE_KINDS = ("floor", "tabletop", "shelf")


class QualificationGeometryError(ValueError):
    """An input needed for a geometry measurement is missing or malformed."""


def _grounding_module() -> Any:
    """Load the existing skinned-vertex grounding tool as a library.

    ``avengine.acoustics.gltf`` deliberately refuses a skinned mesh -- it reads
    static room geometry.  The skinning maths for articulated actors already
    exists in ``tools/assets/audit_habitat_mesh_grounding.py``, so this reuses
    that module rather than writing a second skinning implementation.  The path
    is derived by fixed index from the installed package, never by walking
    parent directories.
    """
    import importlib.util
    import sys

    import avengine

    if "_avengine_grounding_audit" in sys.modules:
        return sys.modules["_avengine_grounding_audit"]
    repo_root = Path(avengine.__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "assets" / "audit_habitat_mesh_grounding.py"
    if not tool_path.is_file():
        raise QualificationGeometryError(
            f"the skinned grounding tool is not available at {tool_path}"
        )
    spec = importlib.util.spec_from_file_location("_avengine_grounding_audit", tool_path)
    if spec is None or spec.loader is None:
        raise QualificationGeometryError(f"cannot load {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_avengine_grounding_audit"] = module
    spec.loader.exec_module(module)
    return module


def measure_skinned_rest_vertices(
    visual_glb: str | Path, *, joint_mapping: str | Path | Mapping[str, Any] | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the rest-pose skinned vertices of an articulated visual GLB.

    With ``joint_mapping`` supplied the vertices come back in the actor root
    frame, so they share a frame with the per-frame grounding audit; without it
    they stay in the GLB's own frame and the row says so.
    """
    tool = _grounding_module()
    path = Path(visual_glb)
    if not path.is_file():
        raise QualificationGeometryError(f"visual GLB does not exist: {path}")
    document = tool.load_glb(path)
    positions, joints, weights, inverse_bind, mesh_node_global, names, indices = tool._geometry(
        document
    )
    globals_by_node = tool._global_nodes(document)
    joint_matrices = {
        name: globals_by_node[int(node_index)]
        for name, node_index in zip(names, indices, strict=True)
    }
    frame = "gltf_scene_root"
    actor_from_skin_root = np.eye(4, dtype=np.float64)
    mapping_payload: Mapping[str, Any] | None = None
    if joint_mapping is not None:
        mapping_payload = (
            joint_mapping if isinstance(joint_mapping, Mapping) else _load_json(joint_mapping)
        )
        if mapping_payload.get("source_glb_sha256") not in (None, document.sha256):
            raise QualificationGeometryError(
                "joint mapping does not bind this visual GLB"
            )
        actor_from_skin_root = tool._matrix_from_mapping(mapping_payload["actor_from_skin_root"])
        frame = "actor_root"
    vertices = tool._skin_actor_vertices(
        positions=positions,
        joints=joints,
        weights=weights,
        inverse_bind=inverse_bind,
        mesh_node_global=mesh_node_global,
        actor_from_skin_root=actor_from_skin_root,
        joint_matrices=joint_matrices,
        skin_joint_names=names,
    )
    evidence = {
        "reader": "tools/assets/audit_habitat_mesh_grounding.py skinned-vertex path",
        "source_ref": str(path),
        "source_sha256": document.sha256,
        "vertex_count": int(len(vertices)),
        "skin_joint_count": len(names),
        "frame": frame,
        "joint_mapping_ref": None if joint_mapping is None else str(joint_mapping),
    }
    return vertices, evidence


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise QualificationGeometryError("cannot normalise a zero or non-finite vector")
    return vector / norm


def _floats(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).ravel()]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    """The value as a finite float, or None when it is neither."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vector(value: Any, length: int = 3) -> list[float] | None:
    """The value as a finite vector of the given length, or None."""
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        return None
    numbers = [_finite(item) for item in value]
    if any(item is None for item in numbers):
        return None
    return [float(item) for item in numbers]


def _not_run(reason: str, *, missing: Sequence[str] = (), **facts: Any) -> dict[str, Any]:
    """An observation that was not made, naming exactly what is absent.

    The word ``not_run`` here describes this measurement attempt.  It is not a
    dimension status: the qualification matrix derives those itself.
    """
    row: dict[str, Any] = {"measurement": "not_run", "reason": str(reason)}
    if missing:
        row["missing_inputs"] = [str(item) for item in missing]
    row.update(facts)
    return row


# --------------------------------------------------------------------------
# asset mesh geometry
# --------------------------------------------------------------------------

def load_asset_triangles(glb_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read one GLB into world-space triangles with the repository's reader."""
    path = Path(glb_path)
    if not path.is_file():
        raise QualificationGeometryError(f"asset GLB does not exist: {path}")
    scene = extract_triangle_scene(path)
    vertices = np.asarray(scene.vertices, dtype=float)
    triangles = np.asarray(scene.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(triangles) == 0:
        raise QualificationGeometryError(f"asset GLB has no triangles: {path}")
    evidence = {
        "reader": "avengine.acoustics.gltf.extract_triangle_scene",
        "source_ref": str(path),
        "source_sha256": scene.source_sha256,
        "source_byte_size": int(scene.source_byte_size),
        "node_instance_count": int(scene.source_node_instance_count),
        "primitive_count": int(scene.source_primitive_count),
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
    }
    return vertices, triangles, evidence


def triangle_data(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-triangle corner points, area and unit normal."""
    points = vertices[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = np.zeros_like(cross)
    nonzero = area > 1.0e-10
    normal[nonzero] = cross[nonzero] / (2.0 * area[nonzero, None])
    return points, area, normal


def boundary_plane_candidates(
    vertices: np.ndarray, triangles: np.ndarray
) -> list[dict[str, Any]]:
    """Measure the six axis-aligned boundary faces of a mesh.

    Each candidate records the faces lying in the boundary band, their area
    share of the whole mesh, the area-weighted outward normal and the residual
    of those faces about their own plane.  Selection happens in
    :func:`measure_rigid_contact_geometry`; this function only measures.
    """
    points, area, normals = triangle_data(vertices, triangles)
    total_area = float(area.sum())
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    extent = bounds_max - bounds_min
    centre = vertices.mean(axis=0)
    candidates: list[dict[str, Any]] = []
    for axis in range(3):
        for side in ("min", "max"):
            boundary = float(bounds_min[axis] if side == "min" else bounds_max[axis])
            band = max(MOUNT_BAND_FLOOR_M, float(extent[axis]) * MOUNT_BAND_FRACTION)
            if side == "max":
                face_mask = np.all(points[:, :, axis] >= boundary - band, axis=1)
            else:
                face_mask = np.all(points[:, :, axis] <= boundary + band, axis=1)
            face_mask = face_mask & (area > 1.0e-10)
            if not face_mask.any():
                continue
            selected = points[face_mask].reshape(-1, 3)
            plane_point = selected.mean(axis=0)
            face_normals = normals[face_mask].copy()
            toward_body = centre - points[face_mask].mean(axis=1)
            signs = np.where(np.einsum("ij,ij->i", face_normals, toward_body) >= 0.0, 1.0, -1.0)
            face_normals *= signs[:, None]
            normal = _unit(np.average(face_normals, axis=0, weights=area[face_mask]))
            residual = np.abs((selected - plane_point) @ normal)
            area_sum = float(area[face_mask].sum())
            tilt_deg = math.degrees(math.acos(min(1.0, abs(float(normal[1])))))
            candidates.append(
                {
                    "axis": int(axis),
                    "side": side,
                    "band_m": float(band),
                    "face_count": int(face_mask.sum()),
                    "area_m2": area_sum,
                    "area_share": area_sum / max(total_area, 1.0e-12),
                    "normal_m": _floats(normal),
                    "plane_point_m": _floats(plane_point),
                    "residual_q95_m": float(np.quantile(residual, 0.95)),
                    "normal_tilt_from_up_deg": tilt_deg,
                    "is_horizontal": tilt_deg <= HORIZONTAL_NORMAL_MAX_TILT_DEG
                    or tilt_deg >= 180.0 - HORIZONTAL_NORMAL_MAX_TILT_DEG,
                }
            )
    return candidates


def _plane_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, normal))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    basis_u = _unit(seed - normal * float(np.dot(seed, normal)))
    basis_v = _unit(np.cross(normal, basis_u))
    if float(np.dot(np.cross(basis_u, basis_v), normal)) < 0.0:
        basis_v = -basis_v
    return basis_u, basis_v


def classify_contact_plane(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Name the contact-plane class the mesh itself supports.

    The mesh says whether an asset presents a downward horizontal base, an
    upward horizontal face or a vertical mounting back.  It does not say which
    room surface the asset belongs on; :func:`resolve_support_kind` keeps that
    separate.
    """
    base = next(
        (row for row in candidates if row["axis"] == 1 and row["side"] == "min"), None
    )
    top = next(
        (row for row in candidates if row["axis"] == 1 and row["side"] == "max"), None
    )
    vertical = [row for row in candidates if not row["is_horizontal"]]
    best_vertical = max(vertical, key=lambda row: float(row["area_share"])) if vertical else None
    classes: list[str] = []
    if base is not None and base["is_horizontal"]:
        classes.append(CONTACT_PLANE_HORIZONTAL_BASE)
    if top is not None and top["is_horizontal"]:
        classes.append(CONTACT_PLANE_HORIZONTAL_TOP)
    if best_vertical is not None:
        classes.append(CONTACT_PLANE_VERTICAL_MOUNT)
    return {
        "measured_contact_plane_classes": classes,
        "horizontal_base_area_share": float(base["area_share"]) if base else None,
        "horizontal_base_residual_q95_m": float(base["residual_q95_m"]) if base else None,
        "largest_vertical_face_area_share": (
            float(best_vertical["area_share"]) if best_vertical else None
        ),
        "measurement_boundary": (
            "the mesh measures the orientation of a candidate contact plane; it does "
            "not establish which room surface class the asset belongs on"
        ),
    }


def resolve_support_kind(
    record: Mapping[str, Any],
    contact_classes: Sequence[str],
    *,
    prior_support_kind: str | None = None,
    prior_source_ref: str | None = None,
) -> dict[str, Any]:
    """Decide whether a support kind may be stated, and on what basis.

    A support kind is emitted only when something outside this measurement
    supplies it: a registry declaration, or a prior measured catalog.  Where
    the registry itself records the attachment surface as *assumed*, no kind is
    emitted -- an assumption is not an observation, and inventing one here
    would let the qualification matrix read a guess as a measurement.
    """
    pose = (
        (record.get("runtime_backends") or {}).get("habitat") or {}
    ).get("resting_pose") or {}
    declared = pose.get("attachment_surface")
    assumed = pose.get("attachment_surface_assumed")
    note = pose.get("attachment_surface_note")
    measured_from = pose.get("measured_from")

    consistent: bool | None = None
    if declared in ("floor", "tabletop", "shelf"):
        consistent = CONTACT_PLANE_HORIZONTAL_BASE in contact_classes
    elif declared == "wall":
        consistent = CONTACT_PLANE_VERTICAL_MOUNT in contact_classes
    elif declared == "ceiling":
        consistent = CONTACT_PLANE_HORIZONTAL_TOP in contact_classes

    result: dict[str, Any] = {
        "registry_attachment_surface": declared,
        "registry_attachment_surface_assumed": assumed,
        "registry_attachment_surface_note": note,
        "registry_resting_pose_measured_from": measured_from,
        "contact_class_consistent_with_registry": consistent,
    }
    if prior_support_kind:
        result.update(
            {
                "support_kind": str(prior_support_kind),
                "support_kind_basis": "prior_catalog_declared_intent",
                "support_kind_basis_ref": prior_source_ref,
                "support_kind_basis_note": (
                    "carried from a prior catalog where the kind was chosen by the "
                    "caller for the room it was planning; the mesh identifies a contact "
                    "plane orientation and does not identify this kind on its own"
                ),
            }
        )
        return result
    if assumed is True:
        result.update(
            {
                "support_kind": None,
                "support_kind_basis": "registry_assumed_not_declared",
                "support_kind_absent_reason": (
                    "the registry records attachment_surface as assumed, not declared, "
                    "and a horizontal base is compatible with both a floor and a "
                    "tabletop; no input observed here distinguishes them"
                ),
            }
        )
        return result
    if declared:
        result.update(
            {"support_kind": str(declared), "support_kind_basis": "registry_declared"}
        )
        return result
    result.update(
        {
            "support_kind": None,
            "support_kind_basis": "absent",
            "support_kind_absent_reason": "the registry declares no attachment surface",
        }
    )
    return result


def measure_rigid_contact_geometry(
    record: Mapping[str, Any],
    *,
    prior_support_kind: str | None = None,
    prior_source_ref: str | None = None,
) -> dict[str, Any]:
    """Measure one rigid asset's bounds, contact plane and footprint."""
    habitat = (record.get("runtime_backends") or {}).get("habitat") or {}
    glb_path = habitat.get("glb_path")
    if not glb_path:
        return _not_run(
            "the registry record carries no habitat glb_path",
            missing=["runtime_backends.habitat.glb_path"],
            asset_id=record.get("asset_id"),
        )
    vertices, triangles, evidence = load_asset_triangles(glb_path)
    pose = habitat.get("resting_pose") or {}
    candidates = boundary_plane_candidates(vertices, triangles)
    classification = classify_contact_plane(candidates)
    classes = classification["measured_contact_plane_classes"]
    kind = resolve_support_kind(
        record,
        classes,
        prior_support_kind=prior_support_kind,
        prior_source_ref=prior_source_ref,
    )

    declared = kind.get("support_kind") or kind.get("registry_attachment_surface")
    if declared == "wall":
        pool = [row for row in candidates if not row["is_horizontal"]]
        target = pose.get("mounting_plane_area_share")
        if pool and target:
            selected = min(
                pool,
                key=lambda row: abs(float(row["area_share"]) - float(target))
                / max(float(target), 1.0e-9)
                + float(row["residual_q95_m"]) / 0.005,
            )
            selection_rule = "registry mounting_plane_area_share plus planar residual"
        elif pool:
            selected = max(pool, key=lambda row: float(row["area_share"]))
            selection_rule = "largest vertical boundary face (registry declares no area share)"
        else:
            return _not_run(
                "no vertical boundary face exists on this mesh to mount against",
                missing=["a vertical boundary face"],
                asset_id=record.get("asset_id"),
                candidate_planes=candidates,
            )
    elif declared == "ceiling":
        selected = next(
            (row for row in candidates if row["axis"] == 1 and row["side"] == "max"), None
        )
        selection_rule = "top horizontal boundary face"
        if selected is None:
            return _not_run(
                "no top horizontal boundary face exists on this mesh",
                missing=["an upward horizontal boundary face"],
                asset_id=record.get("asset_id"),
            )
    else:
        selected = next(
            (row for row in candidates if row["axis"] == 1 and row["side"] == "min"), None
        )
        selection_rule = "bottom horizontal boundary face"
        if selected is None:
            return _not_run(
                "no bottom horizontal boundary face exists on this mesh",
                missing=["a downward horizontal boundary face"],
                asset_id=record.get("asset_id"),
            )

    plane_point = np.asarray(selected["plane_point_m"], dtype=float)
    face_normal = _unit(np.asarray(selected["normal_m"], dtype=float))
    if face_normal[1] < 0.0 and selected["axis"] == 1:
        face_normal = -face_normal
    #: For a horizontal contact the registry expresses the footprint in the
    #: axis-aligned convention -- "put the asset origin at floor height and
    #: apply yaw only" -- and records the base face's own tilt separately.  So
    #: the frame stays axis aligned and the measured face normal is reported as
    #: its own fact; projecting onto the tilted face instead would move the
    #: footprint out of the convention it is cross-checked against.
    if selected["axis"] == 1:
        normal = np.array([0.0, 1.0, 0.0])
        plane_frame = "world_axis_aligned"
    else:
        normal = face_normal
        plane_frame = "measured_face_normal"
    base_tilt_deg = math.degrees(math.acos(min(1.0, abs(float(face_normal[1])))))
    basis_u, basis_v = _plane_frame(normal)
    projected = (vertices - plane_point) @ np.column_stack((basis_u, basis_v))
    footprint = np.ptp(projected, axis=0)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)

    row: dict[str, Any] = {
        "asset_id": record.get("asset_id"),
        "entity_class": record.get("entity_class"),
        "geometry_authority": "asset_visual_geometry",
        "measured_from": Path(str(glb_path)).name,
        "source_ref": str(glb_path),
        "measurement_method": "registered GLB triangles, axis-aligned boundary-face plane",
        "node_transform_and_geometry": evidence,
        "bounds_min_m": _floats(bounds_min),
        "bounds_max_m": _floats(bounds_max),
        "measured_extent_m": _floats(bounds_max - bounds_min),
        "measured_height_m": float(bounds_max[1] - bounds_min[1]),
        "plane_origin_m": _floats(plane_point),
        "plane_normal_m": _floats(normal),
        "plane_basis_u_m": _floats(basis_u),
        "plane_basis_v_m": _floats(basis_v),
        "footprint_extent_m": _floats(footprint),
        "base_plane_offset_m": float(np.dot(plane_point, normal)),
        "measured_face_normal_m": _floats(face_normal),
        "measured_face_normal_tilt_deg": base_tilt_deg,
        "plane_frame": plane_frame,
        "support_plane_selection": {
            "axis": int(selected["axis"]),
            "side": selected["side"],
            "selection_rule": selection_rule,
            "face_count": int(selected["face_count"]),
            "selected_area_m2": float(selected["area_m2"]),
            "selected_area_share": float(selected["area_share"]),
            "residual_q95_m": float(selected["residual_q95_m"]),
            "registry_mounting_plane_area_share": pose.get("mounting_plane_area_share"),
        },
        "candidate_planes": candidates,
        "registry_cross_check": {
            "registry_height_m": pose.get("height_m"),
            "registry_footprint_extent_m": pose.get("footprint_extent_m"),
            "registry_base_plane_offset_m": pose.get("base_plane_offset_m"),
            "registry_measured_plane": pose.get("measured_plane"),
            "registry_base_normal_tilt_deg": pose.get("base_normal_tilt_deg"),
            "registry_contact_extent_m": pose.get("contact_extent_m"),
            "registry_how_to_place": pose.get("how_to_place"),
        },
        "measurement_boundary": (
            "visual mesh geometry only; no native render, physics support or room "
            "placement is established by this row"
        ),
    }
    row.update(classification)
    row.update(kind)
    if row.get("support_kind") is None:
        row.pop("support_kind", None)
    return row


def discover_articulated_package(glb_path: str | Path) -> dict[str, Any]:
    """Locate the retained package inputs that sit beside a visual GLB.

    Two package layouts are in use and both are real: the P12 layout keeps the
    visual mesh at ``<root>/package/visual.glb`` with its siblings at the root,
    and the older M2 dataset layout keeps ``visual.glb`` at the root with
    ``habitat/``, ``actions/`` and ``contacts/`` subdirectories.  Each input is
    resolved by trying its known locations, so an absent one is reported as that
    input rather than mistaken for the other layout.
    """
    visual = Path(glb_path)
    roots = [visual.parent.parent, visual.parent]
    candidates = {
        "joint_mapping": ("joint_mapping.json", "habitat/joint_mapping.json"),
        "actions_npz": ("actions.npz", "actions/idle.npz", "actions/actions.npz"),
        "contacts": ("contacts.json", "contacts/contact_phases.json"),
        "emitter_anchors": ("package/emitter_anchors.json", "emitter_anchors.json"),
        "habitat_probe": ("habitat_static_probe/probe.json", "habitat/probe.json"),
        "runtime_binding": (
            RUNTIME_BINDING_RELATIVE_PATH,
            "habitat/habitat_runtime_binding.json",
        ),
    }
    resolved: dict[str, Any] = {
        "visual_glb": str(visual) if visual.is_file() else None,
        "package_layout": "p12_package" if visual.parent.name == "package" else "m2_dataset",
        "package_root": str(visual.parent.parent if visual.parent.name == "package" else visual.parent),
    }
    for role, relative_paths in candidates.items():
        found = None
        for root in roots:
            for relative in relative_paths:
                probe = root / relative
                if probe.is_file():
                    found = str(probe)
                    break
            if found:
                break
        resolved[role] = found
    return resolved


def run_grounding_audit(record: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Run the existing skinned-vertex grounding audit for one actor.

    Returns the audit payload, or a ``not_run`` observation naming exactly which
    input or capability is missing.  The quadruped contact order is a gate in
    that tool, so a biped actor reports the gate instead of being forced
    through it.
    """
    habitat = (record.get("runtime_backends") or {}).get("habitat") or {}
    package = discover_articulated_package(habitat.get("glb_path") or "")
    missing = [
        key
        for key in ("visual_glb", "joint_mapping", "actions_npz", "emitter_anchors", "contacts")
        if not package.get(key)
    ]
    if missing:
        return _not_run(
            "the retained package does not carry every grounding audit input",
            missing=[f"{key} under {package['package_root']}" for key in missing],
            asset_id=record.get("asset_id"),
        )
    tool = _grounding_module()
    destination = Path(output_path)
    if destination.exists():
        return _load_json(destination)
    try:
        tool.audit(
            visual_glb=Path(package["visual_glb"]),
            actions_npz=Path(package["actions_npz"]),
            joint_mapping=Path(package["joint_mapping"]),
            anchors=Path(package["emitter_anchors"]),
            contacts=Path(package["contacts"]),
            output=destination,
        )
    except Exception as error:  # the tool raises its own error type
        return _not_run(
            f"the grounding audit refused this actor: {type(error).__name__}: {error}",
            missing=["a grounding audit that accepts this actor's contact set"],
            asset_id=record.get("asset_id"),
            contact_order=_contact_order(package.get("contacts")),
        )
    return _load_json(destination)


def _contact_order(contacts_path: str | None) -> list[str] | None:
    if not contacts_path:
        return None
    try:
        return list(_load_json(contacts_path).get("contact_order") or [])
    except (OSError, ValueError):
        return None


def measure_articulated_contact_geometry(
    record: Mapping[str, Any],
    *,
    grounding_audit: Mapping[str, Any] | str | Path | None = None,
    prior_support_kind: str | None = None,
) -> dict[str, Any]:
    """Measure one articulated actor's rest bounds and multi-frame foot support.

    The rest-pose bounds come from the registered visual GLB.  The foot support
    plane and the footprint come from the retained per-frame skinned-vertex
    grounding audit, so the footprint is the area the feet actually sweep over
    the animation, not a bounding box of the whole body.  Without that audit the
    footprint is reported ``not_run`` and no substitute is used.
    """
    habitat = (record.get("runtime_backends") or {}).get("habitat") or {}
    glb_path = habitat.get("glb_path")
    if not glb_path:
        return _not_run(
            "the registry record carries no habitat glb_path",
            missing=["runtime_backends.habitat.glb_path"],
            asset_id=record.get("asset_id"),
        )
    pose = habitat.get("resting_pose") or {}
    package = discover_articulated_package(glb_path)
    vertices, evidence = measure_skinned_rest_vertices(
        glb_path, joint_mapping=package.get("joint_mapping")
    )
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    kind = resolve_support_kind(
        record, [CONTACT_PLANE_HORIZONTAL_BASE], prior_support_kind=prior_support_kind
    )

    row: dict[str, Any] = {
        "asset_id": record.get("asset_id"),
        "entity_class": record.get("entity_class"),
        "geometry_authority": "asset_visual_geometry",
        "measured_from": Path(str(glb_path)).name,
        "source_ref": str(glb_path),
        "measurement_method": "registered visual GLB rest-pose skinned vertices",
        "node_transform_and_geometry": evidence,
        "package_inputs": package,
        "bounds_min_m": _floats(bounds_min),
        "bounds_max_m": _floats(bounds_max),
        "measured_extent_m": _floats(bounds_max - bounds_min),
        "measured_height_m": float(bounds_max[1] - bounds_min[1]),
        "plane_normal_m": [0.0, 1.0, 0.0],
        "plane_basis_u_m": [1.0, 0.0, 0.0],
        "plane_basis_v_m": [0.0, 0.0, -1.0],
        "registry_cross_check": {
            "registry_height_m": pose.get("height_m"),
            "registry_footprint_extent_m": pose.get("footprint_extent_m"),
            "registry_base_plane_offset_m": pose.get("base_plane_offset_m"),
            "registry_resting_pose_measured_from": pose.get("measured_from"),
            "registry_probe_path": pose.get("probe_path"),
        },
        "retained_native_probe": _probe_bounds_cross_check(
            pose.get("probe_path") or package.get("habitat_probe"),
            bounds_min,
            bounds_max,
        ),
        "measurement_boundary": (
            "bind-pose visual geometry and retained per-frame skinned foot anchors; "
            "no physics support, no contact force and no room placement"
        ),
    }
    row.update(kind)
    if row.get("support_kind") is None:
        row.pop("support_kind", None)

    anchor_footprint = measure_anchor_footprint_from_baked_actions(
        package, asset_id=record.get("asset_id")
    )
    row["anchor_footprint"] = anchor_footprint
    if anchor_footprint.get("measurement") == "measured":
        row["footprint_extent_m"] = anchor_footprint["contact_footprint_extent_m"]
        row["footprint_source"] = "declared contact anchors over the baked actions"

    if grounding_audit is None:
        row["articulated_support"] = _not_run(
            "no per-frame skinned grounding audit was supplied for this actor",
            missing=[
                "tools/assets/audit_habitat_mesh_grounding.py output for this asset"
            ],
        )
        if "footprint_extent_m" not in row:
            row["footprint_extent_absent_reason"] = (
                "an articulated footprint is the area the feet sweep across the "
                "animation; it is not derivable from a bind-pose bounding box"
            )
        return row

    audit = (
        grounding_audit
        if isinstance(grounding_audit, Mapping)
        else _load_json(grounding_audit)
    )
    if audit.get("measurement") == "not_run":
        row["articulated_support"] = dict(audit)
        if "footprint_extent_m" not in row:
            row["footprint_extent_absent_reason"] = (
                "an articulated footprint is the area the feet sweep across the "
                "animation; it is not derivable from a rest-pose bounding box"
            )
        return row
    support = summarise_articulated_support(audit)
    row["articulated_support"] = support
    if support.get("measurement") != "not_run":
        row.setdefault("footprint_extent_m", support["contact_footprint_extent_m"])
        row["plane_origin_m"] = [0.0, float(support["support_plane_y_actor_m"]), 0.0]
        row["base_plane_offset_m"] = float(support["support_plane_y_actor_m"])
    return row


def _probe_bounds_cross_check(
    probe_path: str | None, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> dict[str, Any]:
    """Compare measured rest bounds against the retained native probe readback.

    The probe's ``cumulative_bb`` was read back from a Habitat runtime in an
    earlier, separately retained run.  It is recorded here as an independent
    cross-check; the two frames need not coincide, so a difference is reported
    rather than resolved.
    """
    if not probe_path or not Path(probe_path).is_file():
        return _not_run(
            "no retained Habitat probe readback for this actor",
            missing=["habitat_static_probe/probe.json"],
        )
    probe = _load_json(probe_path)
    box = ((probe.get("runtime") or {}).get("cumulative_bb")) or {}
    low, high = box.get("min"), box.get("max")
    if not low or not high:
        return _not_run(
            "the retained probe carries no cumulative bounding box",
            missing=["runtime.cumulative_bb.min/max"],
            probe_path=str(probe_path),
        )
    probe_extent = np.asarray(high, dtype=float) - np.asarray(low, dtype=float)
    measured_extent = np.asarray(bounds_max) - np.asarray(bounds_min)
    return {
        "measurement": "measured",
        "probe_path": str(probe_path),
        "probe_schema": probe.get("schema"),
        "probe_cumulative_bb_min_m": _floats(low),
        "probe_cumulative_bb_max_m": _floats(high),
        "probe_extent_m": _floats(probe_extent),
        "measured_extent_m": _floats(measured_extent),
        "sorted_extent_difference_m": _floats(
            np.sort(measured_extent) - np.sort(probe_extent)
        ),
        "comparison_boundary": (
            "the probe box was read back from a separate Habitat run and need not "
            "share this frame or this pose, so a difference is a reported fact and "
            "not a disagreement to resolve here"
        ),
    }


def summarise_articulated_support(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a grounding audit to a foot support plane and a swept footprint."""
    actions = audit.get("actions")
    if not isinstance(actions, Mapping) or not actions:
        return _not_run(
            "the grounding audit carries no per-frame action records",
            missing=["actions{}.frame_records[]"],
        )
    contact_points: list[list[float]] = []
    frame_minima: list[float] = []
    per_action: dict[str, Any] = {}
    for action_id, action in actions.items():
        if not isinstance(action, Mapping):
            continue
        records = action.get("frame_records")
        if not isinstance(records, list) or not records:
            continue
        minima = [float(item["mesh_min_y_m"]) for item in records if "mesh_min_y_m" in item]
        frame_minima.extend(minima)
        action_points: list[list[float]] = []
        for item in records:
            positions = item.get("contact_joint_positions_m")
            if isinstance(positions, Mapping):
                action_points.extend(_floats(value) for value in positions.values())
        contact_points.extend(action_points)
        per_action[str(action_id)] = {
            "frame_count": len(records),
            "mesh_min_y_m": float(min(minima)) if minima else None,
            "mesh_max_y_m": action.get("mesh_max_y_m"),
            "contact_sample_count": len(action_points),
        }
    if not contact_points or not frame_minima:
        return _not_run(
            "the grounding audit carries no contact joint positions",
            missing=["actions{}.frame_records[].contact_joint_positions_m"],
        )
    points = np.asarray(contact_points, dtype=float)
    support_y = float(min(frame_minima))
    horizontal = points[:, [0, 2]]
    extent = np.ptp(horizontal, axis=0)
    return {
        "measurement": "measured",
        "method": "retained per-frame skinned-vertex grounding audit",
        "source_ref": ((audit.get("inputs") or {}).get("visual_glb") or {}).get("path"),
        "audit_schema": audit.get("schema"),
        "frame_count": int(sum(int(row["frame_count"]) for row in per_action.values())),
        "actions": per_action,
        "support_plane_y_actor_m": support_y,
        "support_plane_definition": (
            "lowest skinned mesh vertex in the actor root frame across every audited "
            "animation frame"
        ),
        "contact_anchor_sample_count": int(len(points)),
        "contact_footprint_extent_m": _floats(extent),
        "contact_footprint_bounds_xz_m": {
            "min": _floats(horizontal.min(axis=0)),
            "max": _floats(horizontal.max(axis=0)),
        },
        "contact_anchor_min_y_actor_m": float(points[:, 1].min()),
        "measurement_boundary": (
            "anchor joints are anatomical references inside the foot, not soles; this "
            "records the swept horizontal extent and the mesh bottom, not a physics "
            "contact"
        ),
    }


def build_geometry_measurement_catalog(
    registry: Mapping[str, Any] | str | Path,
    *,
    asset_ids: Sequence[str] | None = None,
    prior_catalogs: Sequence[Mapping[str, Any] | str | Path] = (),
    grounding_audits: Mapping[str, Any] | None = None,
    grounding_audit_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Measure every requested asset and return one observation catalog.

    The result is shaped for ``build_qualification_matrix(geometry_measurements=...)``.
    """
    payload = registry if isinstance(registry, Mapping) else _load_json(registry)
    records = {str(item["asset_id"]): item for item in payload.get("assets", [])}
    prior_kind: dict[str, tuple[str, str]] = {}
    prior_refs: list[str] = []
    for source in prior_catalogs:
        ref = str(source) if not isinstance(source, Mapping) else "<mapping>"
        prior_refs.append(ref)
        prior_payload = source if isinstance(source, Mapping) else _load_json(source)
        measured = prior_payload.get("asset_visual_geometry_measurements") or {}
        for asset_id, value in measured.items():
            if isinstance(value, Mapping) and value.get("support_kind"):
                prior_kind[str(asset_id)] = (str(value["support_kind"]), ref)

    selected = list(asset_ids) if asset_ids else sorted(records)
    measurements: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for asset_id in selected:
        record = records.get(asset_id)
        if record is None:
            failures.append(
                {"asset_id": asset_id, "reason": "asset id is not in the registry"}
            )
            continue
        prior = prior_kind.get(asset_id)
        try:
            if str(record.get("entity_class", "")).startswith("articulated"):
                audit = (grounding_audits or {}).get(asset_id)
                if audit is None and grounding_audit_dir is not None:
                    audit = run_grounding_audit(
                        record, Path(grounding_audit_dir) / f"{asset_id}_grounding.json"
                    )
                row = measure_articulated_contact_geometry(
                    record,
                    grounding_audit=audit,
                    prior_support_kind=prior[0] if prior else None,
                )
            else:
                row = measure_rigid_contact_geometry(
                    record,
                    prior_support_kind=prior[0] if prior else None,
                    prior_source_ref=prior[1] if prior else None,
                )
        except (QualificationGeometryError, ValueError, OSError) as error:
            failures.append(
                {"asset_id": asset_id, "reason": f"{type(error).__name__}: {error}"}
            )
            continue
        measurements[asset_id] = row
    return {
        "schema": SCHEMA_GEOMETRY,
        "qualification_claim": False,
        "claim_boundary": (
            "measured asset visual geometry only; this catalog states no dimension "
            "status, no eligibility and no dataset admission"
        ),
        "registry_id": payload.get("registry_id"),
        "registry_revision": payload.get("revision"),
        "requested_asset_count": len(selected),
        "measured_asset_count": len(measurements),
        "prior_catalog_refs": prior_refs,
        "unmeasured": failures,
        "asset_visual_geometry_measurements": measurements,
    }


# --------------------------------------------------------------------------
# room support surfaces from a retained depth/semantic capture
# --------------------------------------------------------------------------

def capture_camera_model(
    capture_root: str | Path,
    capture_request: str | Path,
    *,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Read the pinhole model and world pose of one retained capture frame."""
    root = Path(capture_root)
    request = _load_json(capture_request)
    rig = request["primary_camera_rig"]
    calibration = rig["shared_calibration"]
    height, width = (int(value) for value in calibration["resolution_hw"])
    hfov = float(calibration["hfov_degrees"])
    fx = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    neutral = _load_json(root / "neutral_readback.json")
    basis = neutral["camera"][frame_index]["basis"]
    return {
        "frame_index": int(frame_index),
        "resolution_hw": [height, width],
        "fx_px": fx,
        "fy_px": fx,
        "cx_px": (width - 1.0) / 2.0,
        "cy_px": (height - 1.0) / 2.0,
        "projection": "pinhole",
        "camera_position_m": _floats(rig["world_from_rig"]["translation_m"]),
        "camera_basis_rows": [
            _floats(basis["right"]),
            _floats(basis["up"]),
            _floats(basis["forward"]),
        ],
        "camera_basis_source": str(root / "neutral_readback.json"),
        "camera_pose_source": str(capture_request),
    }


def backproject_frame(
    depth: np.ndarray, camera: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world points plus their pixel rows/cols for one depth frame."""
    rows, cols = np.nonzero((depth > 0.0) & np.isfinite(depth))
    z = depth[rows, cols].astype(float)
    camera_points = np.column_stack(
        (
            (cols.astype(float) - float(camera["cx_px"])) * z / float(camera["fx_px"]),
            (float(camera["cy_px"]) - rows.astype(float)) * z / float(camera["fy_px"]),
            z,
        )
    )
    basis = np.asarray(camera["camera_basis_rows"], dtype=float)
    position = np.asarray(camera["camera_position_m"], dtype=float)
    return camera_points @ basis.T + position, rows, cols


def fit_semantic_support_surface(
    depth: np.ndarray,
    semantic: np.ndarray,
    exclude_mask: np.ndarray,
    camera: Mapping[str, Any],
    *,
    semantic_ids: Sequence[int],
    surface_id: str,
    surface_kind: str,
    orientation: str,
    height_band_m: tuple[float, float] | None = None,
    quantiles: tuple[float, float] = (0.05, 0.95),
    minimum_pixels: int = 200,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one bounded support surface from real depth and semantic pixels.

    ``height_band_m`` restricts the fit to a world height band, which is how a
    lower wall patch is separated from the same semantic wall instance higher
    up.  The returned bounds are the measured extent of the pixels that were
    actually fitted, so the surface is never extended past what was observed.
    """
    if orientation not in ("horizontal", "vertical"):
        raise QualificationGeometryError("orientation must be horizontal or vertical")
    candidate = (
        np.isin(semantic, np.asarray(semantic_ids, dtype=semantic.dtype))
        & (depth > 0.0)
        & np.isfinite(depth)
        & (~exclude_mask)
    )
    rows, cols = np.nonzero(candidate)
    if len(rows) == 0:
        return _not_run(
            f"no pixel of semantic id {list(semantic_ids)} survived exclusion",
            missing=["semantic pixels for this surface"],
            surface_id=surface_id,
        )
    z = depth[rows, cols].astype(float)
    camera_points = np.column_stack(
        (
            (cols.astype(float) - float(camera["cx_px"])) * z / float(camera["fx_px"]),
            (float(camera["cy_px"]) - rows.astype(float)) * z / float(camera["fy_px"]),
            z,
        )
    )
    basis = np.asarray(camera["camera_basis_rows"], dtype=float)
    position = np.asarray(camera["camera_position_m"], dtype=float)
    world = camera_points @ basis.T + position
    raw_count = int(len(world))

    keep = np.ones(len(world), dtype=bool)
    if height_band_m is not None:
        low, high = (float(height_band_m[0]), float(height_band_m[1]))
        keep = (world[:, 1] >= low) & (world[:, 1] <= high)
    fit_points = world[keep]
    fit_rows = rows[keep]
    fit_cols = cols[keep]
    if len(fit_points) < minimum_pixels:
        return _not_run(
            f"only {len(fit_points)} pixels remain after the height band, below the "
            f"{minimum_pixels} pixel minimum",
            missing=["enough observed pixels to fit a plane"],
            surface_id=surface_id,
            raw_pixel_count=raw_count,
            banded_pixel_count=int(len(fit_points)),
        )

    centroid = fit_points.mean(axis=0)
    normal = np.array([0.0, 1.0, 0.0])
    for _ in range(4):
        _values, vectors = np.linalg.eigh(np.cov((fit_points - centroid).T, bias=True))
        normal = _unit(vectors[:, 0])
        if orientation == "horizontal" and normal[1] < 0.0:
            normal = -normal
        if orientation == "vertical":
            toward_camera = position - centroid
            if float(np.dot(normal, toward_camera)) < 0.0:
                normal = -normal
        residual = np.abs((fit_points - centroid) @ normal)
        cutoff = max(0.015, float(np.quantile(residual, 0.95) * 2.0))
        inliers = residual <= cutoff
        if int(inliers.sum()) < 20:
            break
        fit_points = fit_points[inliers]
        fit_rows = fit_rows[inliers]
        fit_cols = fit_cols[inliers]
        centroid = fit_points.mean(axis=0)

    tilt_deg = math.degrees(math.acos(min(1.0, abs(float(normal[1])))))
    if orientation == "horizontal" and tilt_deg > HORIZONTAL_NORMAL_MAX_TILT_DEG:
        return _not_run(
            f"the fitted plane tilts {tilt_deg:.2f} deg from up, beyond the "
            f"{HORIZONTAL_NORMAL_MAX_TILT_DEG} deg horizontal limit",
            missing=["a horizontal plane in these pixels"],
            surface_id=surface_id,
            normal_m=_floats(normal),
        )
    if orientation == "vertical" and tilt_deg < 90.0 - HORIZONTAL_NORMAL_MAX_TILT_DEG:
        return _not_run(
            f"the fitted plane tilts {tilt_deg:.2f} deg from up, which is not vertical",
            missing=["a vertical plane in these pixels"],
            surface_id=surface_id,
            normal_m=_floats(normal),
        )

    seed = np.array([1.0, 0.0, 0.0]) if orientation == "horizontal" else np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(seed, normal))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    basis_u = _unit(seed - normal * float(np.dot(seed, normal)))
    basis_v = _unit(np.cross(normal, basis_u))
    if float(np.dot(np.cross(basis_u, basis_v), normal)) < 0.0:
        basis_v = -basis_v
    uv = np.column_stack(
        ((fit_points - centroid) @ basis_u, (fit_points - centroid) @ basis_v)
    )
    bounds_u = np.quantile(uv[:, 0], list(quantiles))
    bounds_v = np.quantile(uv[:, 1], list(quantiles))
    fit_residual = np.abs((fit_points - centroid) @ normal)
    raw_residual = np.abs((world - centroid) @ normal)

    return {
        "surface_id": str(surface_id),
        "surface_kind": str(surface_kind),
        "origin_m": _floats(centroid),
        "normal_m": _floats(normal),
        "basis_u_m": _floats(basis_u),
        "basis_v_m": _floats(basis_v),
        "bounds_u_m": [float(bounds_u[0]), float(bounds_u[1])],
        "bounds_v_m": [float(bounds_v[0]), float(bounds_v[1])],
        "world_height_range_m": [
            float(fit_points[:, 1].min()),
            float(fit_points[:, 1].max()),
        ],
        "evidence": {
            "semantic_ids": [int(value) for value in semantic_ids],
            "frame_index": int(camera["frame_index"]),
            "source_modality": "rig_depth+rig_semantic",
            "depth_unit": "meter",
            "orientation": orientation,
            "height_band_m": list(height_band_m) if height_band_m else None,
            "raw_pixel_count": raw_count,
            "fit_pixel_count": int(len(fit_points)),
            "fit_pixel_bounds": [
                int(fit_cols.min()),
                int(fit_rows.min()),
                int(fit_cols.max()),
                int(fit_rows.max()),
            ],
            "raw_plane_residual_q50_q95_q99_m": _floats(
                np.quantile(raw_residual, [0.5, 0.95, 0.99])
            ),
            "fit_plane_residual_q50_q95_q99_m": _floats(
                np.quantile(fit_residual, [0.5, 0.95, 0.99])
            ),
            "normal_tilt_from_up_deg": tilt_deg,
            "quantile_bounds": list(quantiles),
            "backprojection": {
                key: camera[key]
                for key in (
                    "projection",
                    "fx_px",
                    "fy_px",
                    "cx_px",
                    "cy_px",
                    "camera_basis_source",
                    "camera_pose_source",
                )
            },
        },
        "source": dict(source or {}),
        "measurement_boundary": (
            "the bounds are the extent of the pixels actually observed in this frame; "
            "the real surface may be larger, so these bounds are a lower bound on it "
            "and never an extension of a single height across the room"
        ),
    }


def build_support_surface_catalog(
    *,
    room: Mapping[str, Any],
    capture_root: str | Path,
    capture_request: str | Path,
    surfaces: Sequence[Mapping[str, Any]],
    frame_index: int = 0,
    actor_semantic_ids: Sequence[int] = (),
    actor_mask_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Fit every requested surface from one retained capture frame."""
    root = Path(capture_root)
    camera = capture_camera_model(root, capture_request, frame_index=frame_index)
    depth = np.asarray(
        np.load(root / "depth.npy", mmap_mode="r", allow_pickle=False)[frame_index]
    )
    semantic = np.asarray(
        np.load(root / "semantic.npy", mmap_mode="r", allow_pickle=False)[frame_index]
    )
    exclude = np.zeros(semantic.shape, dtype=bool)
    mask_path = root / "native_pixel_masks_depth_authority_v1.npz"
    used_fields: list[str] = []
    if actor_mask_fields and mask_path.is_file():
        with np.load(mask_path, allow_pickle=False) as masks:
            for field in actor_mask_fields:
                if field in masks:
                    exclude |= masks[field][frame_index] != 0
                    used_fields.append(field)
    if actor_semantic_ids:
        exclude |= np.isin(
            semantic, np.asarray(actor_semantic_ids, dtype=semantic.dtype)
        )
    source = {
        "capture_root": str(root),
        "frame_index": int(frame_index),
        "geometry_derivation": "native_metric_depth_backprojection_plane_fit",
        "visual_depth_source": str(root / "depth.npy"),
        "visual_semantic_source": str(root / "semantic.npy"),
    }
    fitted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for request in surfaces:
        result = fit_semantic_support_surface(
            depth,
            semantic,
            exclude,
            camera,
            semantic_ids=request["semantic_ids"],
            surface_id=request["surface_id"],
            surface_kind=request["surface_kind"],
            orientation=request["orientation"],
            height_band_m=request.get("height_band_m"),
            quantiles=tuple(request.get("quantiles", (0.05, 0.95))),
            minimum_pixels=int(request.get("minimum_pixels", 200)),
            source=source,
        )
        if result.get("measurement") == "not_run":
            skipped.append(result)
            continue
        result["room_id"] = room.get("room_id")
        fitted.append(result)
    return {
        "schema": SCHEMA_SURFACES,
        "qualification_claim": False,
        "claim_boundary": (
            "frame-level visual support evidence back-projected from retained native "
            "metric depth and semantics; no native placement, collision or admission "
            "claim"
        ),
        "room": dict(room),
        "capture": {
            "capture_root": str(root),
            "frame_index": int(frame_index),
            "depth_source": str(root / "depth.npy"),
            "semantic_source": str(root / "semantic.npy"),
            "camera": camera,
        },
        "actor_exclusion": {
            "mask_source": str(mask_path) if used_fields else None,
            "mask_fields": used_fields,
            "excluded_semantic_ids": [int(value) for value in actor_semantic_ids],
            "excluded_pixel_count": int(exclude.sum()),
        },
        "layout": {"schema": SCHEMA_SURFACES, "support_surfaces": fitted},
        "unfitted_requests": skipped,
    }


def surface_fit_observation(
    measurement: Mapping[str, Any], surface: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure whether an asset footprint fits inside a measured surface patch.

    A negative result is inconclusive, not a refutation: the surface bounds are
    the extent of the pixels observed in one frame, so a real table is at least
    as large as its measured patch and may be much larger.
    """
    footprint = measurement.get("footprint_extent_m")
    bounds_u = surface.get("bounds_u_m")
    bounds_v = surface.get("bounds_v_m")
    if not footprint or not bounds_u or not bounds_v:
        return _not_run(
            "a footprint extent and measured surface bounds are both required",
            missing=["footprint_extent_m", "bounds_u_m", "bounds_v_m"],
            surface_id=surface.get("surface_id"),
        )
    asset = sorted(float(value) for value in footprint[:2])
    patch = sorted(
        (float(bounds_u[1]) - float(bounds_u[0]), float(bounds_v[1]) - float(bounds_v[0]))
    )
    fits = asset[0] <= patch[0] and asset[1] <= patch[1]
    return {
        "surface_id": surface.get("surface_id"),
        "surface_kind": surface.get("surface_kind"),
        "asset_footprint_sorted_m": asset,
        "surface_patch_extent_sorted_m": patch,
        "footprint_fits_measured_patch": bool(fits),
        "interpretation": (
            "fits the observed patch"
            if fits
            else "does not fit the observed patch; the patch is a lower bound on the "
            "real surface, so this does not refute the placement"
        ),
    }


# --------------------------------------------------------------------------
# room clearance against the room's own visual mesh
# --------------------------------------------------------------------------

#: Habitat's simulation frame: +Y up, -Z forward.
HABITAT_WORLD_UP = (0.0, 1.0, 0.0)
HABITAT_WORLD_FRONT = (0.0, 0.0, -1.0)


def stage_world_from_asset(
    up: Sequence[float], front: Sequence[float]
) -> np.ndarray:
    """Rotation taking a stage asset's own frame into the Habitat world frame.

    A scene dataset config declares the asset's ``up`` and ``front`` axes.  An
    HM3D stage is authored +Z up, so its vertices are not in the frame the
    capture's world positions are written in, and comparing the two directly
    silently measures the wrong distance.
    """
    asset_up = _unit(np.asarray(up, dtype=float))
    asset_front = _unit(np.asarray(front, dtype=float))
    if abs(float(np.dot(asset_up, asset_front))) > 1.0e-6:
        raise QualificationGeometryError("declared up and front axes are not orthogonal")
    asset = np.column_stack(
        (asset_up, asset_front, np.cross(asset_up, asset_front))
    )
    world = np.column_stack(
        (
            np.asarray(HABITAT_WORLD_UP, dtype=float),
            np.asarray(HABITAT_WORLD_FRONT, dtype=float),
            np.cross(HABITAT_WORLD_UP, HABITAT_WORLD_FRONT),
        )
    )
    return world @ asset.T


def stage_axes_from_dataset_config(
    config: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Read the declared stage ``up``/``front`` axes from a scene dataset config."""
    payload = config if isinstance(config, Mapping) else _load_json(config)
    attributes = ((payload.get("stages") or {}).get("default_attributes")) or {}
    up = attributes.get("up")
    front = attributes.get("front")
    if not up or not front:
        raise QualificationGeometryError(
            "the scene dataset config declares no stage up/front axes"
        )
    return {
        "up": _floats(up),
        "front": _floats(front),
        "source_ref": None if isinstance(config, Mapping) else str(config),
    }


def load_room_triangles(
    scene_glb: str | Path,
    *,
    scene_dataset_config: Mapping[str, Any] | str | Path | None = None,
    up: Sequence[float] | None = None,
    front: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read the room's visual triangle mesh in the Habitat world frame.

    The rotation comes from the declared stage axes -- given directly, or read
    from the scene dataset config.  With neither supplied the vertices are
    returned in the GLB's own frame and the evidence row says so, because
    guessing a frame here would corrupt every distance measured from it.
    """
    vertices, triangles, evidence = load_asset_triangles(scene_glb)
    axes: dict[str, Any] | None = None
    if up is not None and front is not None:
        axes = {"up": _floats(up), "front": _floats(front), "source_ref": "caller"}
    elif scene_dataset_config is not None:
        axes = stage_axes_from_dataset_config(scene_dataset_config)
    if axes is None:
        evidence["frame"] = "gltf_asset_frame"
        evidence["frame_warning"] = (
            "no stage axes were supplied, so these vertices are in the GLB's own "
            "frame and must not be compared against Habitat world coordinates"
        )
        return vertices, triangles, evidence
    rotation = stage_world_from_asset(axes["up"], axes["front"])
    evidence["frame"] = "habitat_world"
    evidence["stage_axes"] = axes
    evidence["world_from_asset_rotation_row_major"] = _floats(rotation)
    return vertices @ rotation.T, triangles, evidence


def _triangle_aabb_overlap(points: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Exact separating-axis test of triangles against one axis-aligned box."""
    centre = 0.5 * (box_min + box_max)
    half = 0.5 * (box_max - box_min)
    vertices = points - centre
    outside = (vertices.min(axis=1) > half) | (vertices.max(axis=1) < -half)
    alive = ~outside.any(axis=1)
    if not alive.any():
        return alive
    edges = [
        vertices[:, 1] - vertices[:, 0],
        vertices[:, 2] - vertices[:, 1],
        vertices[:, 0] - vertices[:, 2],
    ]
    normal = np.cross(edges[0], -edges[2])
    distance = np.abs(np.einsum("ij,ij->i", normal, vertices[:, 0]))
    radius = np.abs(normal) @ half
    alive &= distance <= radius + 1.0e-12
    basis = np.eye(3)
    for edge in edges:
        for index in range(3):
            axis = np.cross(basis[index], edge)
            projections = np.einsum("ij,ikj->ik", axis, vertices)
            extent = np.abs(axis) @ half
            alive &= ~((projections.min(axis=1) > extent) | (projections.max(axis=1) < -extent))
    return alive


def _nearest_room_geometry(
    tri_min: np.ndarray, tri_max: np.ndarray, box_min: np.ndarray, box_max: np.ndarray
) -> dict[str, Any]:
    """Distance from the placed box to the nearest room triangle.

    Measured between axis-aligned boxes -- the placed box and each triangle's
    own bounding box -- so the result is a lower bound on the true distance to
    the triangle surface.  A wall-mounted device that reports metres of
    clearance here is not touching the wall it is supposed to hang on, which a
    zero-collision result on its own would hide.
    """
    gap = np.maximum(np.maximum(tri_min - box_max, box_min - tri_max), 0.0)
    distance = np.linalg.norm(gap, axis=1)
    index = int(np.argmin(distance))
    return {
        "minimum_distance_lower_bound_m": float(distance[index]),
        "nearest_triangle_index": index,
        "nearest_triangle_aabb_min_m": _floats(tri_min[index]),
        "nearest_triangle_aabb_max_m": _floats(tri_max[index]),
        "measurement_boundary": (
            "distance between axis-aligned boxes, so it is a lower bound on the "
            "distance to the triangle itself"
        ),
    }


def measure_room_collision(
    *,
    world_aabb_min: Sequence[float],
    world_aabb_max: Sequence[float],
    room_vertices: np.ndarray,
    room_triangles: np.ndarray,
    support_plane_y_m: float | None = None,
    contact_tolerance_m: float = DEFAULT_CONTACT_TOLERANCE_M,
    scene_ref: str | None = None,
) -> dict[str, Any]:
    """Test one planned world AABB against the room's own visual triangles.

    The box is shrunk by ``contact_tolerance_m`` before the test.  A triangle
    reaching inside the shrunk box is an interpenetration; a triangle touching
    only the tolerance shell is ordinary resting or mounting contact.  Contact
    lying on the asset's own support plane is reported separately from contact
    elsewhere, so resting on a floor is never counted as a collision with it.
    """
    box_min = np.asarray(world_aabb_min, dtype=float)
    box_max = np.asarray(world_aabb_max, dtype=float)
    if box_min.shape != (3,) or box_max.shape != (3,) or np.any(box_min >= box_max):
        raise QualificationGeometryError("world AABB must be a strictly ordered 3-vector pair")
    requested_tolerance = float(contact_tolerance_m)
    extent = box_max - box_min
    #: The contact shell cannot be thicker than the object it surrounds, or a
    #: small device would have no interior left to test.  Capping it at 40% of
    #: the shortest side keeps a real interior for every box; the effective and
    #: requested values are both recorded.
    tolerance = min(requested_tolerance, 0.4 * float(extent.min()))
    if tolerance <= 0.0:
        return _not_run(
            "the placed bounding box has no positive extent to test",
            missing=["a box with positive extent on every axis"],
            contact_tolerance_m=requested_tolerance,
            world_aabb_extent_m=_floats(extent),
        )
    inner_min = box_min + tolerance
    inner_max = box_max - tolerance

    corners = room_vertices[room_triangles]
    tri_min = corners.min(axis=1)
    tri_max = corners.max(axis=1)
    near = np.all(tri_max >= box_min - tolerance, axis=1) & np.all(
        tri_min <= box_max + tolerance, axis=1
    )
    near_indices = np.nonzero(near)[0]
    facts: dict[str, Any] = {
        "measurement": "measured",
        "method": "axis-aligned box versus room visual triangle separating-axis test",
        "scene_ref": scene_ref,
        "room_triangle_count": int(len(room_triangles)),
        "contact_tolerance_m": tolerance,
        "requested_contact_tolerance_m": requested_tolerance,
        "world_aabb_min_m": _floats(box_min),
        "world_aabb_max_m": _floats(box_max),
        "world_aabb_extent_m": _floats(extent),
        "candidate_triangle_count": int(len(near_indices)),
    }
    facts["nearest_room_geometry"] = _nearest_room_geometry(
        tri_min, tri_max, box_min, box_max
    )
    if len(near_indices) == 0:
        facts.update(
            {
                "penetrating_triangle_count": 0,
                "contact_triangle_count": 0,
                "supporting_contact_triangle_count": 0,
                "interpretation": (
                    "no room triangle reaches the placed box; the nearest room "
                    "geometry distance says whether that is clearance or a gap the "
                    "object should have been resting against"
                ),
                "status": "pass",
                "reason": _CLEAR_BOX_REASON,
            }
        )
        return facts

    candidate_points = corners[near_indices]
    penetrating = _triangle_aabb_overlap(candidate_points, inner_min, inner_max)
    touching = _triangle_aabb_overlap(
        candidate_points, box_min - tolerance, box_max + tolerance
    )
    contact_only = touching & ~penetrating
    penetrating_indices = near_indices[penetrating]

    supporting = np.zeros(len(near_indices), dtype=bool)
    if support_plane_y_m is not None:
        heights = candidate_points[:, :, 1]
        supporting = (
            (heights.min(axis=1) <= float(support_plane_y_m) + tolerance)
            & (heights.max(axis=1) >= float(support_plane_y_m) - tolerance)
        )
    penetration_depth = 0.0
    if penetrating.any():
        inside = candidate_points[penetrating]
        clipped = np.clip(inside, inner_min, inner_max)
        penetration_depth = float(np.abs(inside - clipped).max())
        deepest = np.minimum(
            (inside - inner_min).min(axis=(1, 2)), (inner_max - inside).min(axis=(1, 2))
        )
        facts["deepest_penetration_triangle_index"] = int(
            penetrating_indices[int(np.argmax(deepest))]
        )
    facts.update(
        {
            "penetrating_triangle_count": int(penetrating.sum()),
            "contact_triangle_count": int(contact_only.sum()),
            "supporting_contact_triangle_count": int((contact_only & supporting).sum()),
            "non_supporting_contact_triangle_count": int((contact_only & ~supporting).sum()),
            "support_plane_y_m": support_plane_y_m,
            "maximum_penetration_extent_m": penetration_depth,
            "interpretation": (
                "room geometry reaches inside the placed box beyond the contact "
                "tolerance"
                if penetrating.any()
                else "room geometry only touches the placed box within the contact "
                "tolerance"
            ),
            "measurement_boundary": (
                "the asset is represented by its world axis-aligned bounding box, not "
                "its mesh, so a reported interpenetration is an upper bound on the real "
                "one and a clear box is conclusive only for the box"
            ),
        }
    )
    if not penetrating.any():
        facts.update({"status": "pass", "reason": _CLEAR_BOX_REASON})
    else:
        facts.update(
            {
                "status": "not_run",
                "reason": (
                    "room triangles reach inside the placed bounding box, but the box is "
                    "larger than the asset mesh, so this does not establish that the "
                    "meshes interpenetrate; deciding it needs a mesh-level or native "
                    "collision query"
                ),
            }
        )
    return facts


def annotate_placement_plan_room_collision(
    plan: Mapping[str, Any] | str | Path,
    *,
    room_vertices: np.ndarray,
    room_triangles: np.ndarray,
    scene_ref: str | None = None,
    floor_reference_m: float | None = None,
    contact_tolerance_m: float = DEFAULT_CONTACT_TOLERANCE_M,
) -> dict[str, Any]:
    """Return a copy of a placement plan with measured room collision filled in.

    Only the ``clearance.room_collision`` block of each planned instance is
    written; every transform, support identity and peer check produced by the
    planner is carried through untouched.
    """
    payload = json.loads(json.dumps(plan)) if isinstance(plan, Mapping) else _load_json(plan)
    measured = 0
    for instance in payload.get("instances", []):
        if not isinstance(instance, Mapping) or instance.get("status") != "planned":
            continue
        bounds = instance.get("asset_bounds") or {}
        low = bounds.get("world_aabb_min_m")
        high = bounds.get("world_aabb_max_m")
        clearance = instance.setdefault("clearance", {})
        if not low or not high:
            clearance["room_collision"] = _not_run(
                "the planned row carries no world AABB to test",
                missing=["asset_bounds.world_aabb_min_m", "asset_bounds.world_aabb_max_m"],
            )
            continue
        support_y = None
        if str((instance.get("support_identity") or {}).get("surface_kind")) in (
            "floor",
            "tabletop",
            "shelf",
        ):
            support_y = float(low[1])
        elif floor_reference_m is not None:
            support_y = float(floor_reference_m)
        observation = measure_room_collision(
            world_aabb_min=low,
            world_aabb_max=high,
            room_vertices=room_vertices,
            room_triangles=room_triangles,
            support_plane_y_m=support_y,
            contact_tolerance_m=contact_tolerance_m,
            scene_ref=scene_ref,
        )
        clearance["room_collision"] = observation
        measured += 1
    payload["c05_room_collision_annotation"] = {
        "schema": SCHEMA_CLEARANCE,
        "annotated_instance_count": measured,
        "scene_ref": scene_ref,
        "contact_tolerance_m": float(contact_tolerance_m),
        "annotation_boundary": (
            "only clearance.room_collision was written; every planner field is carried "
            "through unchanged, and this observation is offline visual-mesh geometry, "
            "not a native physics query"
        ),
    }
    return payload


# --------------------------------------------------------------------------
# articulated actors: declared anchors, executed poses, world foot contact
# --------------------------------------------------------------------------

#: Habitat writes one flat ``joint_positions`` array per frame.  The order of
#: that array is Habitat's own link order, which is **not** the authored
#: ``runtime_joint_order``; the package's runtime binding carries the offset of
#: each link into the flat array.  Decoding by position instead of by declared
#: offset produces a scrambled skeleton that still looks plausible in bulk, so
#: the offsets are required, never inferred.
RUNTIME_BINDING_RELATIVE_PATH = "habitat_static_probe/habitat_runtime_binding.json"

#: A native readback carries float drift on each quaternion.  Renormalising is
#: allowed; the measured drift is reported so the allowance stays visible.
MAX_NATIVE_QUATERNION_DRIFT = 1.0e-6


def _resolved_package(value: Any) -> dict[str, Any]:
    """Accept a discovery result, a package root or a visual GLB path."""
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if path.is_file():
        return discover_articulated_package(path)
    for relative in ("package/visual.glb", "visual.glb"):
        probe = path / relative
        if probe.is_file():
            return discover_articulated_package(probe)
    raise QualificationGeometryError(f"no visual GLB found for package {path}")


def load_articulated_mapping(package_root: str | Path) -> tuple[Any, dict[str, Any]]:
    """Rebuild the package's Habitat joint mapping as the real dataclass."""
    from avengine.assets.habitat import HabitatAssetMapping, HabitatJointRest

    package = _resolved_package(package_root)
    mapping_path = package.get("joint_mapping")
    if not mapping_path:
        raise QualificationGeometryError(
            f"no joint mapping under {package.get('package_root')}"
        )
    payload = _load_json(mapping_path)

    def canonical(values: Any) -> tuple[float, ...]:
        # +0.0 turns a serialized -0.0 into the canonical zero the kinematics
        # validator requires; it changes no other value.
        return tuple(float(value) + 0.0 for value in values)

    joints = tuple(
        HabitatJointRest(
            joint_ordinal=int(joint["joint_ordinal"]),
            node_index=int(joint["node_index"]),
            joint_id=str(joint["joint_id"]),
            parent_joint_id=joint["parent_joint_id"],
            local_translation_m=canonical(joint["local_translation_m"]),
            rest_rotation_xyzw=canonical(joint["rest_rotation_xyzw"]),
            local_scale=canonical(joint["local_scale"]),
        )
        for joint in payload["joints"]
    )
    mapping = HabitatAssetMapping(
        source_glb_sha256=str(payload["source_glb_sha256"]),
        root_joint_id=str(payload["root_joint_id"]),
        joint_order=tuple(payload["joint_order"]),
        runtime_joint_order=tuple(payload["runtime_joint_order"]),
        joints=joints,
        actor_from_skin_root=tuple(canonical(row) for row in payload["actor_from_skin_root"]),
        actor_from_skin_root_source=str(payload["actor_from_skin_root_source"]),
    )
    return mapping, payload


def load_anchor_definitions(
    source: str | Path | Mapping[str, Any], *, anchor_ids: Sequence[str] | None = None
) -> tuple[Any, ...]:
    """Build declared anchors from a package contacts or emitter-anchor file."""
    from avengine.assets.kinematics import AnchorDefinition, RigidTransform

    payload = source if isinstance(source, Mapping) else _load_json(source)
    records = payload.get("anchor_definitions") or payload.get("anchors") or []
    wanted = set(anchor_ids) if anchor_ids else None
    definitions = []
    for record in records:
        anchor_id = str(record["anchor_id"])
        if wanted is not None and anchor_id not in wanted:
            continue
        offset = record.get("joint_from_anchor") or {}
        definitions.append(
            AnchorDefinition(
                anchor_id=anchor_id,
                joint_id=str(record["joint_id"]),
                joint_from_anchor=RigidTransform(
                    tuple(float(v) + 0.0 for v in offset.get("translation_m", (0.0, 0.0, 0.0))),
                    tuple(float(v) + 0.0 for v in offset.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0))),
                ),
            )
        )
    return tuple(definitions)


def contact_anchor_ids(package_root: str | Path) -> list[str]:
    """The contact anchors this actor actually declares, biped or quadruped."""
    contacts = _resolved_package(package_root).get("contacts")
    if not contacts:
        return []
    return [str(value) for value in (_load_json(contacts).get("contact_order") or [])]


def decode_native_joint_pose(
    flat_frame: np.ndarray,
    runtime_binding: Mapping[str, Any],
    runtime_joint_order: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode one native ``joint_positions`` frame into authored joint order.

    Uses the declared per-link offsets.  Returns the pose plus the measured
    quaternion drift that renormalisation absorbed.
    """
    offsets = {
        str(link["link_name"]): int(link["joint_position_offset"])
        for link in runtime_binding.get("links", [])
    }
    missing = [name for name in runtime_joint_order if name not in offsets]
    if missing:
        raise QualificationGeometryError(
            f"the runtime binding has no joint_position_offset for {missing[:3]}"
        )
    flat = np.asarray(flat_frame, dtype=float).ravel()
    expected = int(runtime_binding.get("joint_position_count") or 0)
    if expected and flat.size != expected:
        raise QualificationGeometryError(
            f"native frame has {flat.size} values, the binding declares {expected}"
        )
    pose = np.stack([flat[offsets[name]: offsets[name] + 4] for name in runtime_joint_order])
    norms = np.linalg.norm(pose, axis=1)
    drift = float(np.abs(norms - 1.0).max())
    if drift > MAX_NATIVE_QUATERNION_DRIFT:
        raise QualificationGeometryError(
            f"native quaternions deviate from unit by {drift:.3e}, beyond the "
            f"{MAX_NATIVE_QUATERNION_DRIFT:.0e} readback-drift allowance"
        )
    return (pose / norms[:, None]) + 0.0, {
        "decoded_by": "habitat_runtime_binding joint_position_offset per link",
        "joint_count": int(len(pose)),
        "max_quaternion_unit_drift": drift,
        "renormalised": True,
    }


def measure_native_foot_contact(
    *,
    package_root: str | Path,
    capture_root: str | Path,
    actor_index: int,
    actor_id: str,
    asset_id: str,
    floor_points_world: np.ndarray | None = None,
    frame_indices: Sequence[int] | None = None,
    floor_search_radius_m: float = 0.4,
    sole_band_m: float = 0.02,
) -> dict[str, Any]:
    """Measure an actor's world foot contact in the frames that actually ran.

    The pose comes from the retained native ``joint_positions`` readback, the
    root from the retained root readback, and the sole from skinning the real
    mesh with that pose -- not from a rest-pose reconstruction.  The anchor
    joints are the declared contact anchors, which sit inside the foot, so the
    sole is reported separately as the lowest skinned vertex.
    """
    from avengine.assets.kinematics import resolve_actor_anchors

    package = _resolved_package(package_root)
    capture = Path(capture_root)
    binding_path = package.get("runtime_binding")
    joints_path = capture / f"actor_joint_readbacks_{actor_id}.npy"
    roots_path = capture / "actor_root_readbacks.npy"
    missing = [str(path) for path in (joints_path, roots_path) if not path.is_file()]
    if not binding_path:
        missing.append(
            f"{package.get('package_root')}/{RUNTIME_BINDING_RELATIVE_PATH} "
            "(the declared joint_position offsets; without them a native pose "
            "cannot be decoded into authored joint order)"
        )
    if missing:
        return _not_run(
            "this actor has no retained native pose readback in this capture",
            missing=missing,
            asset_id=asset_id,
            actor_id=actor_id,
        )

    mapping, mapping_payload = load_articulated_mapping(package)
    binding = _load_json(binding_path)
    anchors = load_anchor_definitions(
        package["contacts"], anchor_ids=contact_anchor_ids(package)
    ) if package.get("contacts") else ()
    if not anchors:
        return _not_run(
            "this actor declares no contact anchors",
            missing=[f"{package.get('package_root')} contacts contact_order"],
            asset_id=asset_id,
        )

    flat_frames = np.load(joints_path, allow_pickle=False)
    roots = np.load(roots_path, allow_pickle=False)
    if actor_index >= roots.shape[1]:
        return _not_run(
            f"the root readback holds {roots.shape[1]} actors, index {actor_index} is out of range",
            missing=["a root readback row for this actor"],
            asset_id=asset_id,
        )
    frame_count = int(min(len(flat_frames), len(roots)))
    indices = (
        [int(value) for value in frame_indices]
        if frame_indices is not None
        else list(range(frame_count))
    )

    tool = _grounding_module()
    document = tool.load_glb(package["visual_glb"])
    if document.sha256 != mapping_payload["source_glb_sha256"]:
        raise QualificationGeometryError("joint mapping does not bind this visual GLB")
    positions, joint_indices, weights, inverse_bind, mesh_node_global, names, _ = tool._geometry(
        document
    )
    actor_from_skin_root = tool._matrix_from_mapping(mapping_payload["actor_from_skin_root"])

    pose_changes = bool(
        len(flat_frames) > 1
        and float(np.abs(np.diff(np.asarray(flat_frames, dtype=float), axis=0)).max()) > 0.0
    )
    rows: list[dict[str, Any]] = []
    decode_facts: dict[str, Any] = {}
    for index in indices:
        pose, decode_facts = decode_native_joint_pose(
            flat_frames[index], binding, mapping.runtime_joint_order
        )
        frame = resolve_actor_anchors(mapping, pose, anchors)
        root = np.asarray(roots[index, actor_index], dtype=float)
        skinned = tool._skin_actor_vertices(
            positions=positions,
            joints=joint_indices,
            weights=weights,
            inverse_bind=inverse_bind,
            mesh_node_global=mesh_node_global,
            actor_from_skin_root=actor_from_skin_root,
            joint_matrices=tool._pose_joint_matrices(mapping_payload, pose),
            skin_joint_names=names,
        )
        world = skinned @ root[:3, :3].T + root[:3, 3]
        sole_y = float(world[:, 1].min())
        low = world[world[:, 1] <= sole_y + float(sole_band_m)]
        anchor_world = {}
        for anchor in anchors:
            translation = frame.anchor_transform(anchor.anchor_id).translation_m
            point = root @ np.array([*translation, 1.0])
            anchor_world[anchor.anchor_id] = _floats(point[:3])
        rows.append(
            {
                "frame_index": index,
                "root_world_position_m": _floats(root[:3, 3]),
                "sole_world_y_m": sole_y,
                "sole_below_root_m": round(sole_y - float(root[1, 3]), 6),
                "sole_contact_centroid_xz_m": [float(low[:, 0].mean()), float(low[:, 2].mean())],
                "sole_band_vertex_count": int(len(low)),
                "sole_footprint_extent_m": _floats(np.ptp(low[:, [0, 2]], axis=0)),
                "contact_anchor_world_m": anchor_world,
                "mesh_top_world_y_m": float(world[:, 1].max()),
            }
        )

    result: dict[str, Any] = {
        "measurement": "measured",
        "asset_id": asset_id,
        "actor_id": actor_id,
        "method": (
            "retained native joint_positions readback decoded through the declared "
            "runtime binding, real mesh skinned in that pose, root from the retained "
            "root readback"
        ),
        "inputs": {
            "joint_readback": str(joints_path),
            "root_readback": str(roots_path),
            "runtime_binding": str(binding_path),
            "joint_mapping": str(package.get("joint_mapping")),
            "contacts": str(package.get("contacts")),
            "visual_glb": str(package.get("visual_glb")),
            "visual_glb_sha256": document.sha256,
        },
        "declared_contact_anchors": [anchor.anchor_id for anchor in anchors],
        "native_frame_count": frame_count,
        "measured_frame_count": len(rows),
        "pose_varies_across_frames": pose_changes,
        "pose_decode": decode_facts,
        "frames": rows,
        "measurement_boundary": (
            "anchor joints are anatomical references inside the foot, not soles; the "
            "sole figure is the lowest skinned vertex of the real mesh in the pose that "
            "actually ran"
            + (
                ""
                if pose_changes
                else "; every retained frame carries the same joint pose, so this is one "
                "executed static pose and not a dynamic contact trajectory"
            )
        ),
    }

    if floor_points_world is None or len(floor_points_world) == 0:
        result["floor_comparison"] = _not_run(
            "no measured visual floor points were supplied for this world",
            missing=["measured visual floor point cloud for the world this capture ran in"],
        )
        return result

    floor = np.asarray(floor_points_world, dtype=float)
    comparisons = []
    for row in rows:
        centre = row["sole_contact_centroid_xz_m"]
        distance = np.hypot(floor[:, 0] - centre[0], floor[:, 2] - centre[1])
        near = floor[distance <= float(floor_search_radius_m)]
        if len(near) == 0:
            comparisons.append(
                {
                    "frame_index": row["frame_index"],
                    "measurement": "not_run",
                    "reason": (
                        f"no measured floor point within {floor_search_radius_m} m of the "
                        "sole contact centroid"
                    ),
                }
            )
            continue
        floor_y = float(np.median(near[:, 1]))
        comparisons.append(
            {
                "frame_index": row["frame_index"],
                "measurement": "measured",
                "floor_point_count": int(len(near)),
                "floor_search_radius_m": float(floor_search_radius_m),
                "visual_floor_median_y_m": floor_y,
                "sole_above_visual_floor_m": round(row["sole_world_y_m"] - floor_y, 6),
                "root_above_visual_floor_m": round(row["root_world_position_m"][1] - floor_y, 6),
            }
        )
    measured = [row for row in comparisons if row.get("measurement") == "measured"]
    result["floor_comparison"] = {
        "measurement": "measured" if measured else "not_run",
        "method": (
            "median height of the measured visual floor points under the sole contact "
            "centroid, per frame"
        ),
        "frames": comparisons,
        "sole_above_visual_floor_min_m": (
            min(row["sole_above_visual_floor_m"] for row in measured) if measured else None
        ),
        "sole_above_visual_floor_max_m": (
            max(row["sole_above_visual_floor_m"] for row in measured) if measured else None
        ),
        "comparison_boundary": (
            "this is the gap between the sole and the measured visual floor; it is not a "
            "verdict, and which floor a placement should be judged against is a "
            "convention this module does not choose"
        ),
    }
    return result


def measure_anchor_footprint_from_baked_actions(
    package_root: str | Path, *, asset_id: str | None = None
) -> dict[str, Any]:
    """Swept contact-anchor footprint from the package's baked actions.

    This is the asset-level footprint: where the declared contact anchors travel
    across the authored animation, in actor space.  It is a CPU reconstruction
    from baked clips, not an observation of an executed frame, and says so.  It
    works for any declared contact set, so a biped is not excluded.
    """
    from avengine.assets.actions import read_baked_actions_npz
    from avengine.assets.kinematics import resolve_actor_anchors

    package = _resolved_package(package_root)
    actions_path = package.get("actions_npz")
    if not actions_path:
        return _not_run(
            "the package carries no baked actions",
            missing=[f"{package.get('package_root')} actions npz"],
            asset_id=asset_id,
        )
    mapping, _payload = load_articulated_mapping(package)
    anchors = load_anchor_definitions(
        package["contacts"], anchor_ids=contact_anchor_ids(package)
    ) if package.get("contacts") else ()
    if not anchors:
        return _not_run(
            "this actor declares no contact anchors",
            missing=[f"{package.get('package_root')} contacts contact_order"],
            asset_id=asset_id,
        )
    actions = read_baked_actions_npz(actions_path)
    if tuple(actions.runtime_joint_order) != tuple(mapping.runtime_joint_order):
        return _not_run(
            "the baked actions and the joint mapping disagree on runtime joint order",
            missing=["baked actions bound to this joint mapping"],
            asset_id=asset_id,
        )
    points: list[list[float]] = []
    per_action: dict[str, Any] = {}
    for action in actions.actions:
        frames = 0
        for pose in action.rotations_xyzw:
            frame = resolve_actor_anchors(mapping, pose, anchors)
            for anchor in anchors:
                points.append(list(frame.anchor_transform(anchor.anchor_id).translation_m))
            frames += 1
        per_action[str(action.semantic_action_id)] = {"frame_count": frames}
    if not points:
        return _not_run(
            "the baked actions carry no frames",
            missing=["baked action frames"],
            asset_id=asset_id,
        )
    array = np.asarray(points, dtype=float)
    horizontal = array[:, [0, 2]]
    return {
        "measurement": "measured",
        "method": "declared contact anchors resolved through the package's baked actions",
        "evidence_kind": "cpu_reconstruction_from_baked_clips",
        "source_ref": str(actions_path),
        "joint_mapping_ref": str(package.get("joint_mapping")),
        "declared_contact_anchors": [anchor.anchor_id for anchor in anchors],
        "actions": per_action,
        "contact_anchor_sample_count": int(len(array)),
        "contact_footprint_extent_m": _floats(np.ptp(horizontal, axis=0)),
        "contact_footprint_bounds_xz_m": {
            "min": _floats(horizontal.min(axis=0)),
            "max": _floats(horizontal.max(axis=0)),
        },
        "contact_anchor_min_y_actor_m": float(array[:, 1].min()),
        "measurement_boundary": (
            "an authored-animation footprint in actor space; it is not an executed "
            "frame, not a physics contact, and not evidence that any actor stood on any "
            "floor"
        ),
    }


# --------------------------------------------------------------------------
# exact mesh-versus-room narrowphase
# --------------------------------------------------------------------------

#: Edge/triangle crossings closer to a triangle's own plane than this are
#: treated as touching rather than crossing.  It is a numerical guard on the
#: intersection arithmetic, not a physical clearance allowance, and it is the
#: only isotropic epsilon in the narrowphase.
NARROWPHASE_EPSILON_M = 1.0e-9


def _segment_triangle_intersections(
    starts: np.ndarray,
    ends: np.ndarray,
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Möller-Trumbore segment/triangle test, one segment per triangle row.

    Returns a boolean hit mask and the intersection points.  Segments are
    matched row-wise against triangles, so the caller pairs them first.
    """
    edge1 = corners[:, 1] - corners[:, 0]
    edge2 = corners[:, 2] - corners[:, 0]
    direction = ends - starts
    pvec = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    parallel = np.abs(determinant) <= NARROWPHASE_EPSILON_M
    safe = np.where(parallel, 1.0, determinant)
    inverse = 1.0 / safe
    tvec = starts - corners[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", direction, qvec) * inverse
    t = np.einsum("ij,ij->i", edge2, qvec) * inverse
    hit = (
        (~parallel)
        & (u >= 0.0)
        & (v >= 0.0)
        & (u + v <= 1.0)
        & (t >= 0.0)
        & (t <= 1.0)
    )
    points = starts + direction * t[:, None]
    return hit, points


def _triangle_pairs_intersect(
    asset_corners: np.ndarray, room_corners: np.ndarray
) -> tuple[np.ndarray, list[list[float]]]:
    """Exact crossing test for paired triangles, both edge directions."""
    hits = np.zeros(len(asset_corners), dtype=bool)
    points: list[list[float]] = [[] for _ in range(len(asset_corners))]
    for source, target in ((asset_corners, room_corners), (room_corners, asset_corners)):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            hit, point = _segment_triangle_intersections(
                source[:, first], source[:, second], target
            )
            for index in np.nonzero(hit)[0]:
                if not points[index]:
                    points[index] = [float(value) for value in point[index]]
            hits |= hit
    return hits, points


def measure_asset_mesh_room_intersection(
    *,
    asset_glb: str | Path,
    root_transform_matrix_row_major: Sequence[float],
    room_vertices: np.ndarray,
    room_triangles: np.ndarray,
    support_plane: Mapping[str, Any] | None = None,
    broadphase_margin_m: float = 0.01,
    scene_ref: str | None = None,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Intersect one placed asset mesh with the room mesh, triangle by triangle.

    The asset's own GLB triangles are placed by the planner's root transform and
    tested exactly against the room's triangles.  Bounding boxes only choose
    candidates; no verdict rests on them, so an empty corner of a bounding box
    can no longer be reported as a collision.

    A room triangle lying in the asset's declared support plane is separated out
    as resting contact.  That allowance applies **along the support normal
    only**, and its size is the support surface's own measured plane residual --
    never a blanket distance applied in every direction.
    """
    matrix = np.asarray(root_transform_matrix_row_major, dtype=float)
    if matrix.size != 16:
        raise QualificationGeometryError("root transform must be 16 numbers")
    matrix = matrix.reshape(4, 4)
    vertices, triangles, evidence = load_asset_triangles(asset_glb)
    placed = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    asset_corners_all = placed[triangles]
    asset_min = asset_corners_all.min(axis=1)
    asset_max = asset_corners_all.max(axis=1)
    box_min = placed.min(axis=0)
    box_max = placed.max(axis=0)

    room_corners_all = room_vertices[room_triangles]
    room_min = room_corners_all.min(axis=1)
    room_max = room_corners_all.max(axis=1)
    margin = float(broadphase_margin_m)

    # level one: room triangles near the placed asset box
    near_room = np.nonzero(
        np.all(room_max >= box_min - margin, axis=1)
        & np.all(room_min <= box_max + margin, axis=1)
    )[0]
    facts: dict[str, Any] = {
        "measurement": "measured",
        "method": "placed asset triangles versus room triangles, exact edge-crossing test",
        "asset_id": asset_id,
        "asset_mesh": evidence,
        "scene_ref": scene_ref,
        "root_transform_matrix_row_major": _floats(matrix.ravel()),
        "placed_world_aabb_min_m": _floats(box_min),
        "placed_world_aabb_max_m": _floats(box_max),
        "asset_triangle_count": int(len(triangles)),
        "room_triangle_count": int(len(room_triangles)),
        "broadphase_margin_m": margin,
        "broadphase_room_candidates": int(len(near_room)),
    }
    if len(near_room) == 0:
        facts.update(
            {
                "intersecting_pair_count": 0,
                "support_contact_pair_count": 0,
                "penetrating_pair_count": 0,
                "status": "pass",
                "reason": (
                    "no room triangle lies within the broadphase margin of the placed "
                    "asset mesh, so no triangle of the asset can intersect the room"
                ),
            }
        )
        return facts

    # level two: asset triangles near the surviving room triangles
    room_span_min = room_min[near_room].min(axis=0)
    room_span_max = room_max[near_room].max(axis=0)
    near_asset = np.nonzero(
        np.all(asset_max >= room_span_min - margin, axis=1)
        & np.all(asset_min <= room_span_max + margin, axis=1)
    )[0]
    facts["broadphase_asset_candidates"] = int(len(near_asset))
    if len(near_asset) == 0:
        facts.update(
            {
                "intersecting_pair_count": 0,
                "support_contact_pair_count": 0,
                "penetrating_pair_count": 0,
                "status": "pass",
                "reason": "no asset triangle lies within the broadphase margin of the room",
            }
        )
        return facts

    # level three: pairwise box overlap, then the exact test on survivors
    pair_asset: list[int] = []
    pair_room: list[int] = []
    for asset_index in near_asset:
        low = asset_min[asset_index] - margin
        high = asset_max[asset_index] + margin
        overlap = np.all(room_max[near_room] >= low, axis=1) & np.all(
            room_min[near_room] <= high, axis=1
        )
        for room_index in near_room[overlap]:
            pair_asset.append(int(asset_index))
            pair_room.append(int(room_index))
    facts["broadphase_pair_count"] = len(pair_asset)
    if not pair_asset:
        facts.update(
            {
                "intersecting_pair_count": 0,
                "support_contact_pair_count": 0,
                "penetrating_pair_count": 0,
                "status": "pass",
                "reason": "no asset triangle and room triangle share a bounding box",
            }
        )
        return facts

    asset_pairs = asset_corners_all[np.asarray(pair_asset)]
    room_pairs = room_corners_all[np.asarray(pair_room)]
    hits, points = _triangle_pairs_intersect(asset_pairs, room_pairs)
    hit_indices = np.nonzero(hits)[0]
    facts["intersecting_pair_count"] = int(len(hit_indices))
    if len(hit_indices) == 0:
        facts.update(
            {
                "support_contact_pair_count": 0,
                "penetrating_pair_count": 0,
                "status": "pass",
                "reason": (
                    "every candidate triangle pair was tested exactly and none of the "
                    "asset's triangles crosses a room triangle"
                ),
            }
        )
        return facts

    # separate resting contact on the declared support plane from penetration
    support_mask = np.zeros(len(hit_indices), dtype=bool)
    support_facts: dict[str, Any] = {"applied": False}
    if support_plane:
        origin = _vector(support_plane.get("origin_m"))
        normal = _vector(support_plane.get("normal_m"))
        residual = _finite(support_plane.get("plane_residual_q95_m"))
        if origin is not None and normal is not None and residual is not None:
            origin_v = np.asarray(origin, dtype=float)
            normal_v = _unit(np.asarray(normal, dtype=float))
            corners = room_pairs[hit_indices]
            offsets = np.einsum("ijk,k->ij", corners - origin_v, normal_v)
            in_plane = np.all(np.abs(offsets) <= float(residual), axis=1)
            edge_a = corners[:, 1] - corners[:, 0]
            edge_b = corners[:, 2] - corners[:, 0]
            face = np.cross(edge_a, edge_b)
            lengths = np.linalg.norm(face, axis=1)
            aligned = np.zeros(len(corners), dtype=bool)
            valid = lengths > 0.0
            aligned[valid] = (
                np.abs(np.einsum("ij,j->i", face[valid] / lengths[valid, None], normal_v))
                >= math.cos(math.radians(HORIZONTAL_NORMAL_MAX_TILT_DEG))
            )
            support_mask = in_plane & aligned
            support_facts = {
                "applied": True,
                "support_surface_id": support_plane.get("surface_id"),
                "plane_origin_m": _floats(origin_v),
                "plane_normal_m": _floats(normal_v),
                "allowance_m": float(residual),
                "allowance_basis": (
                    "the q95 residual of that surface's own measured plane fit, applied "
                    "along the surface normal only"
                ),
                "allowance_source_ref": support_plane.get("source_ref"),
                "normal_alignment_limit_deg": HORIZONTAL_NORMAL_MAX_TILT_DEG,
            }
        else:
            support_facts = {
                "applied": False,
                "reason": (
                    "the support plane record lacks an origin, a normal or a measured "
                    "plane residual, so no resting-contact allowance was applied"
                ),
                "missing": [
                    name
                    for name, value in (
                        ("origin_m", origin),
                        ("normal_m", normal),
                        ("plane_residual_q95_m", residual),
                    )
                    if value is None
                ],
            }

    penetrating = hit_indices[~support_mask]
    examples = []
    for index in penetrating[:12]:
        examples.append(
            {
                "asset_triangle_index": int(pair_asset[int(index)]),
                "room_triangle_index": int(pair_room[int(index)]),
                "intersection_point_m": points[int(index)],
                "room_triangle_corners_m": [
                    _floats(corner) for corner in room_corners_all[pair_room[int(index)]]
                ],
            }
        )
    facts.update(
        {
            "support_contact_pair_count": int(support_mask.sum()),
            "penetrating_pair_count": int(len(penetrating)),
            "support_contact_exclusion": support_facts,
            "penetrating_examples": examples,
            "distinct_penetrating_room_triangles": int(
                len({pair_room[int(index)] for index in penetrating})
            ),
            "distinct_penetrating_asset_triangles": int(
                len({pair_asset[int(index)] for index in penetrating})
            ),
            "numerical_epsilon_m": NARROWPHASE_EPSILON_M,
        }
    )
    if len(penetrating) == 0:
        facts.update(
            {
                "status": "pass",
                "reason": (
                    "every exact triangle crossing lies in the asset's declared support "
                    "plane within that surface's own measured fit residual, which is "
                    "resting contact and not interpenetration"
                ),
            }
        )
    else:
        facts.update(
            {
                "status": "fail",
                "reason": (
                    f"{len(penetrating)} exact triangle crossings lie outside the "
                    "declared support plane, so the placed asset mesh interpenetrates "
                    "room geometry"
                ),
            }
        )
    return facts


# --------------------------------------------------------------------------
# world identity, support query, and root contact correction
# --------------------------------------------------------------------------

#: A support level groups sample hits whose heights agree to within this.  It is
#: a clustering width for reading one surface out of a mesh, not a placement
#: tolerance and not a penetration allowance.
SUPPORT_LEVEL_CLUSTER_M = 0.01

#: A level is reported as covering the footprint when at least this fraction of
#: the sampled footprint has a hit at that level.  It is a reporting threshold:
#: every level is returned with its own coverage either way, so a consumer can
#: apply its own rule.
DEFAULT_FOOTPRINT_COVERAGE_FRACTION = 0.9


def resolve_world_identity(
    *,
    plan_path: str | Path | None = None,
    capture_root: str | Path | None = None,
    stage_root: str | Path | None = None,
    declared_world_id: str | None = None,
) -> dict[str, Any]:
    """Name a world by what its own plan and capture declare.

    A directory name is not an identity.  This returns the identifiers that are
    actually declared -- the plan's ``episode_id`` and ``scene.room_id`` -- plus
    the directory aliases a consumer may have in hand, and it leaves ``world_id``
    null when nothing declares one.  Inventing ``world_<stage directory>`` would
    manufacture a key that no other producer uses.
    """
    identity: dict[str, Any] = {
        "schema": "avengine_c05_world_identity_v1",
        "world_id": declared_world_id,
        "world_id_source": "caller declared" if declared_world_id else None,
        "episode_id": None,
        "room_id": None,
        "scene_id": None,
        "aliases": {},
        "provenance": {},
    }
    if stage_root is not None:
        stage = Path(stage_root)
        identity["aliases"]["stage_directory_name"] = stage.name
        identity["provenance"]["stage_root"] = str(stage)
    if capture_root is not None:
        capture = Path(capture_root)
        identity["provenance"]["capture_root"] = str(capture)
        identity["aliases"]["capture_attempt_directory"] = capture.parent.name
    if plan_path is not None and Path(plan_path).is_file():
        plan = _load_json(plan_path)
        scene = plan.get("scene") or {}
        identity.update(
            {
                "episode_id": plan.get("episode_id"),
                "room_id": scene.get("room_id"),
                "scene_id": scene.get("scene_id"),
            }
        )
        identity["provenance"]["episode_plan"] = str(plan_path)
        if plan.get("world_id"):
            identity["world_id"] = plan["world_id"]
            identity["world_id_source"] = "episode plan"
    if identity["world_id"] is None:
        identity["world_id_absent_reason"] = (
            "neither the episode plan nor the caller declares a world_id for this "
            "capture; the stage directory name is an alias, not an identity, and is "
            "not promoted to one here"
        )
    identity["join_keys"] = [
        key for key in ("world_id", "episode_id", "room_id") if identity.get(key)
    ]
    return identity


def resolve_room_visual_geometry(plan_path: str | Path) -> dict[str, Any]:
    """Find the room visual mesh an episode plan actually declares.

    A support query needs the room's own visual geometry.  Some rooms are served
    by a UE/SPEAR package and declare no stage GLB at all; for those this returns
    a ``not_run`` naming the absent input, so a caller cannot quietly fall back to
    another room's mesh or to a constant measured somewhere else.
    """
    plan_file = Path(plan_path)
    if not plan_file.is_file():
        return _not_run(
            "the episode plan does not exist", missing=[str(plan_file)]
        )
    plan = _load_json(plan_file)
    resources = plan.get("resources") or {}
    scene_glb = resources.get("scene_glb")
    dataset_config = resources.get("dataset_config")
    backend = resources.get("backend")
    room_id = (plan.get("scene") or {}).get("room_id") or resources.get("room_id")
    if not scene_glb or not Path(str(scene_glb)).is_file():
        return _not_run(
            (
                f"the plan for room {room_id!r} declares no readable stage GLB"
                + (f"; its backend is {backend!r}" if backend else "")
            ),
            missing=[
                f"resources.scene_glb for room {room_id}",
                "a room visual mesh in the Habitat world frame",
            ],
            room_id=room_id,
            backend=backend,
            episode_plan=str(plan_file),
            declared_scene_glb=scene_glb,
            substitution_refused=(
                "another room's mesh and any constant measured in another world are "
                "both refused here; without this room's own geometry the support "
                "query does not run"
            ),
        )
    if not dataset_config or not Path(str(dataset_config)).is_file():
        return _not_run(
            f"the plan for room {room_id!r} declares no readable scene dataset config",
            missing=[f"resources.dataset_config for room {room_id}"],
            room_id=room_id,
            episode_plan=str(plan_file),
            note=(
                "the stage axes come from this config; without it the mesh frame is "
                "unknown and distances measured against it would be wrong"
            ),
        )
    return {
        "measurement": "measured",
        "room_id": room_id,
        "backend": backend,
        "scene_glb": str(scene_glb),
        "scene_dataset_config": str(dataset_config),
        "navmesh": resources.get("navmesh"),
        "episode_plan": str(plan_file),
    }


def _footprint_sample_points(
    centre_world_m: Sequence[float],
    footprint_extent_m: Sequence[float],
    *,
    yaw_deg: float = 0.0,
    grid: int = 9,
) -> np.ndarray:
    """A yaw-rotated grid of XZ sample points over one footprint rectangle."""
    centre = np.asarray(centre_world_m, dtype=float)
    half = 0.5 * np.asarray(footprint_extent_m, dtype=float)[:2]
    steps = np.linspace(-1.0, 1.0, max(2, int(grid)))
    u, v = np.meshgrid(steps * half[0], steps * half[1])
    local = np.column_stack((u.ravel(), v.ravel()))
    angle = math.radians(float(yaw_deg))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rotated = local @ rotation.T
    return np.column_stack(
        (rotated[:, 0] + centre[0], np.full(len(rotated), centre[1]), rotated[:, 1] + centre[2])
    )


def _vertical_hits(
    samples: np.ndarray,
    room_vertices: np.ndarray,
    room_triangles: np.ndarray,
    *,
    low_m: float,
    high_m: float,
    normal_tilt_limit_deg: float,
) -> list[list[tuple[float, np.ndarray]]]:
    """For each XZ sample, the heights of near-horizontal room faces above it."""
    corners = room_vertices[room_triangles]
    tri_min = corners.min(axis=1)
    tri_max = corners.max(axis=1)
    sample_min = samples.min(axis=0)
    sample_max = samples.max(axis=0)
    near = (
        (tri_max[:, 0] >= sample_min[0]) & (tri_min[:, 0] <= sample_max[0])
        & (tri_max[:, 2] >= sample_min[2]) & (tri_min[:, 2] <= sample_max[2])
        & (tri_max[:, 1] >= low_m) & (tri_min[:, 1] <= high_m)
    )
    candidate = corners[near]
    if len(candidate) == 0:
        return [[] for _ in samples]
    edge_a = candidate[:, 1] - candidate[:, 0]
    edge_b = candidate[:, 2] - candidate[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 0.0
    unit = np.zeros_like(normals)
    unit[usable] = normals[usable] / lengths[usable, None]
    horizontal = np.abs(unit[:, 1]) >= math.cos(math.radians(normal_tilt_limit_deg))
    candidate = candidate[horizontal]
    unit = unit[horizontal]
    if len(candidate) == 0:
        return [[] for _ in samples]

    ax, az = candidate[:, 0, 0], candidate[:, 0, 2]
    bx, bz = candidate[:, 1, 0], candidate[:, 1, 2]
    cx, cz = candidate[:, 2, 0], candidate[:, 2, 2]
    denominator = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
    valid = np.abs(denominator) > 1.0e-14
    results: list[list[tuple[float, np.ndarray]]] = []
    for sample in samples:
        px, pz = float(sample[0]), float(sample[2])
        w1 = np.zeros(len(candidate))
        w2 = np.zeros(len(candidate))
        w1[valid] = (
            (bz[valid] - cz[valid]) * (px - cx[valid])
            + (cx[valid] - bx[valid]) * (pz - cz[valid])
        ) / denominator[valid]
        w2[valid] = (
            (cz[valid] - az[valid]) * (px - cx[valid])
            + (ax[valid] - cx[valid]) * (pz - cz[valid])
        ) / denominator[valid]
        w3 = 1.0 - w1 - w2
        inside = valid & (w1 >= -1.0e-9) & (w2 >= -1.0e-9) & (w3 >= -1.0e-9)
        hits: list[tuple[float, np.ndarray]] = []
        for index in np.nonzero(inside)[0]:
            height = float(
                w1[index] * candidate[index, 0, 1]
                + w2[index] * candidate[index, 1, 1]
                + w3[index] * candidate[index, 2, 1]
            )
            if low_m <= height <= high_m:
                hits.append((height, unit[index]))
        results.append(hits)
    return results


def query_support_under_footprint(
    *,
    room_vertices: np.ndarray,
    room_triangles: np.ndarray,
    centre_world_m: Sequence[float],
    footprint_extent_m: Sequence[float],
    navigation_floor_m: float,
    navigation_floor_source: str,
    yaw_deg: float = 0.0,
    search_ceiling_m: float | None = None,
    search_below_floor_m: float | None = None,
    grid: int = 9,
    normal_tilt_limit_deg: float = HORIZONTAL_NORMAL_MAX_TILT_DEG,
    level_cluster_m: float = SUPPORT_LEVEL_CLUSTER_M,
    coverage_fraction: float = DEFAULT_FOOTPRINT_COVERAGE_FRACTION,
    scene_ref: str | None = None,
) -> dict[str, Any]:
    """Measure the horizontal surfaces under one footprint, level by level.

    The storey comes from ``navigation_floor_m``, which the caller must take from
    the room or plan's own navigation component -- never from the height of the
    root being corrected.  Every level is returned with the fraction of the
    footprint it actually covers, so a small object standing on a desk is visible
    as partial coverage instead of being mistaken for a new support surface.
    """
    if navigation_floor_m is None or not navigation_floor_source:
        raise QualificationGeometryError(
            "a navigation floor and its source are required; the storey may not be "
            "inferred from the candidate root height"
        )
    samples = _footprint_sample_points(
        centre_world_m, footprint_extent_m, yaw_deg=yaw_deg, grid=grid
    )
    search_below = (float(level_cluster_m) if search_below_floor_m is None
                    else float(search_below_floor_m))
    if not np.isfinite(search_below) or search_below < 0.0:
        raise QualificationGeometryError("search_below_floor_m must be finite and nonnegative")
    search_below_source = ("level_cluster_m_default" if search_below_floor_m is None
                           else "explicit_search_below_floor_m")
    low = float(navigation_floor_m) - search_below
    high = (
        float(search_ceiling_m)
        if search_ceiling_m is not None
        else float(centre_world_m[1])
    )
    if high <= low:
        return _not_run(
            "the search band is empty: the ceiling is at or below the navigation floor",
            missing=["a search ceiling above the navigation floor"],
            navigation_floor_m=float(navigation_floor_m),
            search_below_floor_m=search_below,
            search_below_floor_source=search_below_source,
            search_ceiling_m=high,
        )
    hits = _vertical_hits(
        samples,
        room_vertices,
        room_triangles,
        low_m=low,
        high_m=high,
        normal_tilt_limit_deg=normal_tilt_limit_deg,
    )
    covered = sum(1 for row in hits if row)
    facts: dict[str, Any] = {
        "measurement": "measured",
        "method": (
            "vertical sampling of the footprint against near-horizontal room triangles, "
            "clustered into levels"
        ),
        "scene_ref": scene_ref,
        "centre_world_m": _floats(centre_world_m),
        "footprint_extent_m": _floats(np.asarray(footprint_extent_m, dtype=float)[:2]),
        "yaw_deg": float(yaw_deg),
        "sample_count": int(len(samples)),
        "samples_with_any_hit": int(covered),
        "search_band_m": [low, high],
        "navigation_floor_m": float(navigation_floor_m),
        "navigation_floor_source": str(navigation_floor_source),
        "search_below_floor_m": float(search_below),
        "search_below_floor_source": search_below_source,
        "level_cluster_m": float(level_cluster_m),
        "coverage_fraction_threshold": float(coverage_fraction),
        "normal_tilt_limit_deg": float(normal_tilt_limit_deg),
        "storey_basis": (
            "the navigation floor supplied by the caller from the room or plan's own "
            "navigation component; no storey was inferred from a root height"
        ),
    }
    if covered == 0:
        facts.update(
            {
                "levels": [],
                "supporting_level": None,
                "reason": (
                    "no near-horizontal room face lies under this footprint between the "
                    "navigation floor and the search ceiling"
                ),
            }
        )
        return facts

    # one entry per sample per level: cluster the heights seen across samples
    heights = sorted(height for row in hits for height, _ in row)
    clusters: list[list[float]] = [[heights[0]]]
    for height in heights[1:]:
        if height - clusters[-1][-1] <= float(level_cluster_m):
            clusters[-1].append(height)
        else:
            clusters.append([height])

    levels = []
    for cluster in clusters:
        low_edge, high_edge = cluster[0], cluster[-1]
        members = []
        normals = []
        sample_tops = []
        for index, row in enumerate(hits):
            best = [
                (height, normal)
                for height, normal in row
                if low_edge - 1.0e-9 <= height <= high_edge + 1.0e-9
            ]
            if best:
                top = max(best, key=lambda item: item[0])
                members.append(index)
                normals.append(top[1])
                sample_tops.append(top[0])
        if not members:
            continue
        stack = np.asarray(normals, dtype=float)
        mean_normal = _unit(stack.mean(axis=0)) if len(stack) else np.array([0.0, 1.0, 0.0])
        level_heights = np.asarray(cluster, dtype=float)
        fraction = len(members) / float(len(samples))
        # Why the rest of the footprint is not on this level matters: a sample with
        # no room geometry at all is a hole in the mesh, while a sample whose only
        # hit is higher up is occupied by something standing on the surface.  The
        # two call for different fixes and must not be summed into one number.
        covered_set = set(members)
        empty_samples = 0
        occupied_above = 0
        only_below = 0
        for index, row in enumerate(hits):
            if index in covered_set:
                continue
            if not row:
                empty_samples += 1
                continue
            if any(height > high_edge + 1.0e-9 for height, _ in row):
                occupied_above += 1
            else:
                only_below += 1
        levels.append(
            {
                "height_median_m": float(np.median(level_heights)),
                "height_min_m": float(level_heights.min()),
                "height_max_m": float(level_heights.max()),
                "height_spread_m": float(level_heights.max() - level_heights.min()),
                # An object rests on the highest face beneath it, not on the middle
                # of a cluster.  A real slab has a top and a bottom face and the
                # clustering can chain them together, so the height a placement must
                # use is the per-sample top, summarised here.
                "support_top_median_m": float(np.median(np.asarray(sample_tops))),
                "support_top_q05_m": float(np.quantile(np.asarray(sample_tops), 0.05)),
                "support_top_q95_m": float(np.quantile(np.asarray(sample_tops), 0.95)),
                "support_top_max_m": float(np.max(np.asarray(sample_tops))),
                "support_top_spread_m": float(
                    np.max(np.asarray(sample_tops)) - np.min(np.asarray(sample_tops))
                ),
                "normal_m": _floats(mean_normal),
                "normal_tilt_from_up_deg": math.degrees(
                    math.acos(min(1.0, abs(float(mean_normal[1]))))
                ),
                "covered_sample_count": len(members),
                "footprint_coverage_fraction": fraction,
                "covers_footprint": bool(fraction >= float(coverage_fraction)),
                "uncovered_samples_with_no_room_geometry": empty_samples,
                "uncovered_samples_occupied_above": occupied_above,
                "uncovered_samples_with_lower_geometry_only": only_below,
                "uncovered_breakdown_note": (
                    "samples with no room geometry are holes in the scene mesh; samples "
                    "occupied above carry something standing on this level"
                ),
                "height_above_navigation_floor_m": round(
                    float(np.median(level_heights)) - float(navigation_floor_m), 6
                ),
            }
        )
    levels.sort(key=lambda row: row["height_median_m"])
    covering = [row for row in levels if row["covers_footprint"]]
    supporting = covering[-1] if covering else None
    ranked = sorted(levels, key=lambda row: row["footprint_coverage_fraction"], reverse=True)
    best = ranked[0] if ranked else None
    facts.update(
        {
            "levels": levels,
            "level_count": len(levels),
            "supporting_level": supporting,
            "supporting_level_basis": (
                "the highest level that covers the footprint at or above the coverage "
                "threshold; levels that cover only part of the footprint are reported "
                "as occupancy and are never promoted to a support surface"
            ),
            "best_covered_level": best,
            "best_covered_level_note": (
                "the level covering the largest share of this footprint, reported "
                "whether or not it reaches the coverage threshold; it is a candidate "
                "for a consumer to judge, not a support this module adopted"
            ),
            "partial_levels_above_supporting": [
                row
                for row in levels
                if not row["covers_footprint"]
                and supporting is not None
                and row["height_median_m"] > supporting["height_median_m"]
            ],
            "measurement_boundary": (
                "horizontal room faces sampled under this footprint; it does not test "
                "the asset's own mesh, run physics, or establish that a placement is "
                "legal"
            ),
        }
    )
    if supporting is None:
        facts["reason"] = (
            "no level covers the footprint at the coverage threshold, so no surface "
            "here supports this asset; the levels found are reported as occupancy"
        )
    return facts


def measure_rigid_contact_offset(
    asset_glb: str | Path, root_transform_matrix_row_major: Sequence[float]
) -> dict[str, Any]:
    """How far the placed root sits above the asset's own lowest point."""
    matrix = np.asarray(root_transform_matrix_row_major, dtype=float)
    if matrix.size != 16:
        raise QualificationGeometryError("root transform must be 16 numbers")
    matrix = matrix.reshape(4, 4)
    vertices, _triangles, evidence = load_asset_triangles(asset_glb)
    placed = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    root_y = float(matrix[1, 3])
    contact_y = float(placed[:, 1].min())
    return {
        "measurement": "measured",
        "evidence_kind": "placed_rigid_mesh",
        "method": "lowest vertex of the asset mesh under the planner's own root transform",
        "source_ref": str(asset_glb),
        "asset_mesh": evidence,
        "root_world_y_m": root_y,
        "contact_world_y_m": contact_y,
        "root_above_contact_m": round(root_y - contact_y, 6),
        "asset_height_m": round(float(placed[:, 1].max()) - contact_y, 6),
    }


def articulated_contact_offset(foot_contact_row: Mapping[str, Any]) -> dict[str, Any]:
    """Root-above-sole offset from a foot contact measurement."""
    frames = _list(foot_contact_row.get("frames"))
    if not frames:
        return _not_run(
            "the foot contact row carries no frames",
            missing=["frames[].sole_world_y_m"],
        )
    offsets = [
        float(frame["root_world_position_m"][1]) - float(frame["sole_world_y_m"])
        for frame in frames
        if frame.get("sole_world_y_m") is not None
    ]
    if not offsets:
        return _not_run(
            "no frame carries a sole height", missing=["frames[].sole_world_y_m"]
        )
    varies = bool(foot_contact_row.get("pose_varies_across_frames"))
    return {
        "measurement": "measured",
        "evidence_kind": (
            "native_executed_pose" if foot_contact_row.get("inputs") else "cpu_reconstruction"
        ),
        "method": "root minus the lowest skinned vertex, per measured frame",
        "frame_count": len(offsets),
        "root_above_contact_m": round(float(min(offsets)), 6),
        "root_above_contact_min_m": round(float(min(offsets)), 6),
        "root_above_contact_max_m": round(float(max(offsets)), 6),
        "pose_varies_across_frames": varies,
        "measurement_boundary": (
            "one executed static pose" if not varies else "measured across differing poses"
        ),
    }


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def plan_root_contact_correction(
    *,
    support_query: Mapping[str, Any],
    contact_offset: Mapping[str, Any],
    current_root_world_m: Sequence[float],
    target_gap_m: float = 0.0,
    identity: Mapping[str, Any] | None = None,
    support_statistic: str = "support_top_q95_m",
) -> dict[str, Any]:
    """Compute the root height a new plan would need for the contact to land.

    The result is a **prediction** for a plan that has not run.  It is not a
    verified contact: the predicted gap is arithmetic on measured inputs, and the
    row says that native verification is still outstanding.  A caller has to
    regenerate the root, skeleton, emitter, trajectory and downstream acoustic
    inputs together; this returns the height and the evidence, not a patched plan.
    """
    root = np.asarray(current_root_world_m, dtype=float)
    if support_query.get("measurement") != "measured":
        return _not_run(
            "no support measurement is available for this footprint",
            missing=["a measured support query"],
            support_query_reason=support_query.get("reason"),
        )
    supporting = support_query.get("supporting_level")
    if not supporting:
        return _not_run(
            "no measured level covers this footprint, so there is no surface to "
            "correct onto; a level that covers only part of the footprint is "
            "occupancy and is not adopted as a support",
            missing=["a room surface covering this footprint"],
            levels_found=support_query.get("level_count"),
            partial_levels=[
                {
                    "height_median_m": row["height_median_m"],
                    "footprint_coverage_fraction": row["footprint_coverage_fraction"],
                }
                for row in _list(support_query.get("levels"))
            ],
        )
    if contact_offset.get("measurement") != "measured":
        return _not_run(
            "no measured contact offset for this asset",
            missing=["root_above_contact_m"],
            contact_offset_reason=contact_offset.get("reason"),
        )
    offset = float(contact_offset["root_above_contact_m"])
    # To rest on a surface an asset has to clear the high points of that surface
    # under its own footprint, so the statistic is a high quantile of the
    # per-sample tops rather than their middle.  The choice is a named parameter
    # and the alternatives travel with the result.
    if support_statistic not in supporting:
        return _not_run(
            f"the measured level carries no {support_statistic!r}",
            missing=[support_statistic],
        )
    support_height = float(supporting[support_statistic])
    corrected_root_y = support_height + offset + float(target_gap_m)
    current_contact_y = float(root[1]) - offset

    # Sitting an asset on a level is only a fix if the space above that level is
    # free.  A level can cover the footprint and still carry objects standing on
    # it, and lowering onto it would bury the asset under them.
    asset_height = _finite(contact_offset.get("asset_height_m"))
    occupied_above = int(supporting.get("uncovered_samples_occupied_above") or 0)
    obstructions = []
    if asset_height is not None:
        ceiling = support_height + float(target_gap_m) + asset_height
        for level in _list(support_query.get("levels")):
            height = float(level["height_median_m"])
            level_top = float(level.get("support_top_median_m", height))
            if support_height + 1.0e-3 < level_top <= ceiling:
                obstructions.append(
                    {
                        "height_median_m": height,
                        "support_top_median_m": level_top,
                        "footprint_coverage_fraction": level["footprint_coverage_fraction"],
                        "height_above_support_m": round(level_top - support_height, 6),
                    }
                )
    obstructed = bool(obstructions) or occupied_above > 0
    return {
        "prediction": "cpu_prediction",
        "native_verification": "not_run",
        "schema": "avengine_c05_root_contact_correction_v1",
        "identity": dict(identity or {}),
        "method": (
            "corrected root height = measured supporting level + the asset's measured "
            "root-above-contact offset + the requested gap"
        ),
        "support_level_height_m": support_height,
        "support_level_statistic": support_statistic,
        "support_level_height_alternatives_m": {
            key: supporting.get(key)
            for key in (
                "height_median_m", "support_top_median_m", "support_top_q95_m",
                "support_top_max_m",
            )
        },
        "support_level_height_basis": (
            "a high quantile of the highest measured face under each covered footprint "
            "sample. A level's median height can fall inside a slab, and resting on a "
            "surface means clearing its high points, not its middle"
        ),
        "support_level_top_spread_m": supporting.get("support_top_spread_m"),
        "support_level_normal_m": supporting.get("normal_m"),
        "support_level_coverage_fraction": supporting.get("footprint_coverage_fraction"),
        "support_level_source": support_query.get("navigation_floor_source"),
        "asset_root_above_contact_m": offset,
        "asset_contact_evidence_kind": contact_offset.get("evidence_kind"),
        "current_root_world_m": _floats(root),
        "current_contact_world_y_m": round(current_contact_y, 6),
        "current_gap_to_support_m": round(current_contact_y - support_height, 6),
        "corrected_root_world_m": [float(root[0]), round(corrected_root_y, 6), float(root[2])],
        "root_delta_m": round(corrected_root_y - float(root[1]), 6),
        "target_gap_m": float(target_gap_m),
        "predicted_gap_after_correction_m": float(target_gap_m),
        "asset_height_m": asset_height,
        "headroom_obstructed": obstructed,
        "headroom_obstructions": obstructions,
        "headroom_obstruction_max_coverage_fraction": (
            max(row["footprint_coverage_fraction"] for row in obstructions)
            if obstructions
            else 0.0
        ),
        "headroom_significance_note": (
            "each obstruction carries the share of the footprint it covers. A level "
            "covering a few per cent may be mesh noise or a small object, while one "
            "covering most of the footprint is a surface the asset would be buried "
            "under. This module reports the coverage and applies no significance "
            "threshold of its own."
        ),
        "support_level_samples_occupied_above": occupied_above,
        "headroom_note": (
            "the space the asset would occupy above this level is not clear, so "
            "lowering onto it would put the asset inside whatever stands there; the "
            "height is reported but this is not a usable spot"
            if obstructed
            else "no measured level falls inside the space the asset would occupy"
        ),
        "consumer_contract": (
            "a new plan must regenerate the root, the skeleton pose, the emitter "
            "anchors, the trajectory and the acoustic inputs together; applying only "
            "this height to an existing plan would leave the rest of that plan stale"
        ),
        "unobstructed": not obstructed,
        "claim_boundary": (
            "the corrected height is arithmetic on measured inputs for a plan that has "
            "not run; the predicted gap is not an executed contact and no native "
            "capture has confirmed it"
        ),
    }


# Small file-backed cache: a bounded body enclosure is reused across candidate cameras.
_BODY_ENVELOPE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def measure_registered_body_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    """Enclose every declared idle/walk pose in the registered actor-root frame.

    This is a CPU geometric enclosure, not an observed silhouette. In particular,
    a box corner is not a point on the actual body and cannot prove visibility.
    """
    from copy import deepcopy
    from itertools import product
    from avengine.assets.actions import read_baked_actions_npz

    backend = (record.get("runtime_backends") or {}).get("habitat") or {}
    visual = backend.get("glb_path")
    if not visual:
        return _not_run("no registered visual mesh for a body enclosure",
                        missing=["runtime_backends.habitat.glb_path"],
                        asset_id=record.get("asset_id"))
    package = discover_articulated_package(visual)
    if not package.get("joint_mapping") or not package.get("actions_npz"):
        return _not_run("body enclosure needs the registered articulated mapping and actions",
                        missing=["joint_mapping/actions_npz"], asset_id=record.get("asset_id"))
    timeline = record.get("timeline") or {}
    wanted = tuple(dict.fromkeys(str(x) for x in (
        timeline.get("idle_action_id"), timeline.get("walking_action_id")) if x))
    if not wanted:
        return _not_run("the asset timeline declares no actions",
                        missing=["timeline action IDs"], asset_id=record.get("asset_id"))
    files = [Path(package["actions_npz"])]
    for action_id in wanted:
        candidate = Path(package["package_root"]) / "actions" / (action_id + ".npz")
        if candidate.is_file() and candidate not in files:
            files.append(candidate)
    inputs = [Path(visual), Path(package["joint_mapping"]), *files]
    key = (record.get("asset_id"), record.get("revision"), wanted,
           tuple((str(x.resolve()), x.stat().st_mtime_ns, x.stat().st_size) for x in inputs))
    if key in _BODY_ENVELOPE_CACHE:
        return deepcopy(_BODY_ENVELOPE_CACHE[key])
    tool = _grounding_module()
    document = tool.load_glb(Path(visual))
    mapping = _load_json(package["joint_mapping"])
    if mapping.get("source_glb_sha256") not in (None, document.sha256):
        raise QualificationGeometryError("body mapping does not bind the registered visual GLB")
    actions = {}
    for file in files:
        baked = read_baked_actions_npz(file)
        if tuple(baked.runtime_joint_order) != tuple(mapping["runtime_joint_order"]):
            raise QualificationGeometryError("body actions differ from the registered joint order")
        for action in baked.actions:
            aid = str(action.semantic_action_id)
            if aid not in wanted:
                continue
            if aid in actions and not np.array_equal(actions[aid], action.rotations_xyzw):
                raise QualificationGeometryError("conflicting declared body action: " + aid)
            actions[aid] = action.rotations_xyzw
    missing = sorted(set(wanted) - set(actions))
    if missing:
        return _not_run("not every allowed action has geometry poses",
                        missing=missing, asset_id=record.get("asset_id"))
    positions, joints, weights, inverse_bind, mesh_global, names, _ = tool._geometry(document)
    actor_matrix = tool._matrix_from_mapping(mapping["actor_from_skin_root"])
    low = np.full(3, np.inf); high = np.full(3, -np.inf)
    counts = {}
    for aid in wanted:
        poses = actions[aid]
        for pose in poses:
            vertices = tool._skin_actor_vertices(
                positions=positions, joints=joints, weights=weights, inverse_bind=inverse_bind,
                mesh_node_global=mesh_global, actor_from_skin_root=actor_matrix,
                joint_matrices=tool._pose_joint_matrices(mapping, pose), skin_joint_names=names)
            low = np.minimum(low, vertices.min(axis=0))
            high = np.maximum(high, vertices.max(axis=0))
        counts[aid] = len(poses)
    result = {
        "measurement": "measured", "asset_id": record.get("asset_id"),
        "frame": "registered_actor_root", "bounds_min_m": low.tolist(),
        "bounds_max_m": high.tolist(),
        "vertices_m": [[float(low[a] if side[a] == 0 else high[a]) for a in range(3)]
                       for side in product((0, 1), repeat=3)],
        "source_ref": str(Path(visual).resolve()),
        "joint_mapping_ref": str(Path(package["joint_mapping"]).resolve()),
        "action_refs": [str(file.resolve()) for file in files],
        "action_frame_counts": counts,
        "pose_coverage": "axis_aligned_enclosure_of_all_declared_idle_walk_poses",
        "claim_boundary": "enclosure only; its corners do not prove any actual body point is visible",
    }
    if len(_BODY_ENVELOPE_CACHE) >= 16:
        _BODY_ENVELOPE_CACHE.clear()
    _BODY_ENVELOPE_CACHE[key] = deepcopy(result)
    return result
