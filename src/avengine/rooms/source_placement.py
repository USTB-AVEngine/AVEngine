from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

SOURCE_PLACEMENT_SCHEMA = "avengine_source_placement_plan_v1"
STATIC_ENTITY_CLASSES = frozenset(("rigid_object", "rigid_static_object"))
SUPPORT_SURFACE_KINDS = frozenset(("floor", "tabletop", "wall", "ceiling"))
VISUAL_GEOMETRY_AUTHORITIES = frozenset(("visual_geometry", "visual_scene", "room_visual_geometry"))
ACOUSTIC_GEOMETRY_MARKERS = frozenset(("acoustic_proxy", "acoustic_package_arrays", "acoustic_geometry"))


class SourcePlacementError(ValueError):
    # A static source placement request cannot be justified by supplied data.
    def __init__(self, code: str, reason: str, **details: Any) -> None:
        self.code = str(code)
        self.reason = str(reason)
        self.details = dict(details)
        super().__init__(f"{self.code}: {self.reason}")

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "reason": self.reason}
        if self.details:
            result["details"] = deepcopy(self.details)
        return result


def _mapping(value: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourcePlacementError("mapping_required", f"{owner} must be an object")
    return value


def _text(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourcePlacementError("field_required", f"{owner} must be non-empty text")
    return value.strip()


def _finite(value: Any, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SourcePlacementError("finite_number_required", f"{owner} must be finite")
    return float(value)


def _vector(value: Any, owner: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise SourcePlacementError("vector3_required", f"{owner} must be a 3-vector")
    return tuple(_finite(item, f"{owner}[{index}]") for index, item in enumerate(value))


def _pair(value: Any, owner: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise SourcePlacementError("pair_required", f"{owner} must be a 2-vector")
    result = tuple(_finite(item, f"{owner}[{index}]") for index, item in enumerate(value))
    if not result[0] < result[1]:
        raise SourcePlacementError("ordered_bounds_required", f"{owner} must be increasing")
    return result

def _extent(value: Any, owner: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise SourcePlacementError("pair_required", f"{owner} must be a 2-vector")
    result = tuple(_finite(item, f"{owner}[{index}]") for index, item in enumerate(value))
    if result[0] <= 0.0 or result[1] <= 0.0:
        raise SourcePlacementError("positive_extent_required", f"{owner} must contain two positive dimensions")
    return result


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(x) + float(y) for x, y in zip(a, b))  # type: ignore[return-value]


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(x) - float(y) for x, y in zip(a, b))  # type: ignore[return-value]


def _scale(a: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(x) * float(scalar) for x in a)  # type: ignore[return-value]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _norm(a: Sequence[float], owner: str) -> float:
    value = math.sqrt(_dot(a, a))
    if value <= 1.0e-12:
        raise SourcePlacementError("nonzero_vector_required", f"{owner} must be non-zero")
    return value


def _unit(a: Sequence[float], owner: str) -> tuple[float, float, float]:
    return _scale(a, 1.0 / _norm(a, owner))


def _mat3_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(_dot(row, vector) for row in matrix)  # type: ignore[return-value]


def _mat4(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> list[float]:
    return [
        float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2]), float(translation[0]),
        float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2]), float(translation[1]),
        float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2]), float(translation[2]),
        0.0, 0.0, 0.0, 1.0,
    ]


def _outer_add(rotation: list[list[float]], world: Sequence[float], local: Sequence[float]) -> None:
    for row in range(3):
        for column in range(3):
            rotation[row][column] += float(world[row]) * float(local[column])


def _first(value: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if value.get(name) is not None:
            return value[name]
    return None


def _asset_record(registry: Mapping[str, Any], asset_id: Any, revision: Any = None) -> Mapping[str, Any]:
    wanted = _text(asset_id, "request.asset_id")
    assets = registry.get("assets")
    if not isinstance(assets, Sequence):
        raise SourcePlacementError("registry_assets_missing", "registry.assets must be a list")
    records = [record for record in assets if isinstance(record, Mapping) and record.get("asset_id") == wanted]
    if len(records) != 1:
        raise SourcePlacementError("asset_not_registered", "asset_id must resolve to one registry record", asset_id=wanted)
    record = records[0]
    if record.get("entity_class") not in STATIC_ENTITY_CLASSES:
        raise SourcePlacementError("static_asset_required", "source placement is only for rigid static entities", asset_id=wanted, entity_class=record.get("entity_class"))
    if revision is not None and str(record.get("revision")) != str(revision):
        raise SourcePlacementError("asset_revision_mismatch", "requested asset revision differs from registry", asset_id=wanted, requested_revision=revision, registry_revision=record.get("revision"))
    return record


def _resting_pose(record: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    habitat = (record.get("runtime_backends") or {}).get("habitat")
    if not isinstance(habitat, Mapping) or not isinstance(habitat.get("resting_pose"), Mapping):
        raise SourcePlacementError("missing_registry_resting_pose", "registry asset has no habitat.resting_pose", asset_id=record.get("asset_id"))
    pose = habitat["resting_pose"]
    geometry = request.get("asset_geometry")
    surface = _text(pose.get("attachment_surface"), "registry.resting_pose.attachment_surface")
    offset_value = pose.get("base_plane_offset_m")
    footprint_value = pose.get("footprint_extent_m")
    bounds_min_value = None
    bounds_max_value = None
    if isinstance(geometry, Mapping):
        if geometry.get("base_plane_offset_m") is not None:
            offset_value = geometry["base_plane_offset_m"]
        if footprint_value is None:
            footprint_value = _first(geometry, ("footprint_extent_m", "support_footprint_extent_m"))
        bounds_min_value = _first(geometry, ("bounds_min_m", "asset_bounds_min_m"))
        bounds_max_value = _first(geometry, ("bounds_max_m", "asset_bounds_max_m"))
    if (bounds_min_value is None) != (bounds_max_value is None):
        raise SourcePlacementError(
            "asset_bounds_incomplete",
            "asset_geometry must provide both bounds_min_m and bounds_max_m",
            asset_id=record.get("asset_id"),
        )
    bounds_min = _vector(bounds_min_value, "asset_geometry.bounds_min_m") if bounds_min_value is not None else None
    bounds_max = _vector(bounds_max_value, "asset_geometry.bounds_max_m") if bounds_max_value is not None else None
    if bounds_min is not None and bounds_max is not None and any(lo >= hi for lo, hi in zip(bounds_min, bounds_max)):
        raise SourcePlacementError(
            "asset_bounds_invalid",
            "asset_geometry bounds_min_m must be strictly below bounds_max_m",
            asset_id=record.get("asset_id"),
        )
    offset = _finite(offset_value, "registry.resting_pose.base_plane_offset_m")
    height = _finite(pose.get("height_m"), "registry.resting_pose.height_m")
    if height <= 0.0:
        raise SourcePlacementError("positive_height_required", "registry.resting_pose.height_m must be positive", asset_id=record.get("asset_id"))
    footprint = _extent(footprint_value, "registry.resting_pose.footprint_extent_m")
    normal = _first(pose, ("plane_normal_m", "base_plane_normal_m", "mounting_plane_normal_m", "support_plane_normal_m"))
    basis_u = _first(pose, ("plane_basis_u_m", "base_plane_basis_u_m", "mounting_plane_basis_u_m"))
    basis_v = _first(pose, ("plane_basis_v_m", "base_plane_basis_v_m", "mounting_plane_basis_v_m"))
    if isinstance(geometry, Mapping):
        if normal is None:
            normal = _first(geometry, ("plane_normal_m", "support_plane_normal_m"))
        if basis_u is None:
            basis_u = _first(geometry, ("plane_basis_u_m", "support_plane_basis_u_m"))
        if basis_v is None:
            basis_v = _first(geometry, ("plane_basis_v_m", "support_plane_basis_v_m"))
    if normal is None:
        raise SourcePlacementError("missing_registry_plane_normal", "resting_pose must carry an explicit plane normal; tilt degrees cannot be converted implicitly", asset_id=record.get("asset_id"), measured_plane=pose.get("measured_plane"))
    if basis_u is None or basis_v is None:
        raise SourcePlacementError("missing_registry_plane_basis", "resting_pose must carry explicit plane basis vectors; yaw/facing cannot be guessed", asset_id=record.get("asset_id"), measured_plane=pose.get("measured_plane"))
    n = _unit(_vector(normal, "registry.resting_pose.plane_normal_m"), "registry.resting_pose.plane_normal_m")
    u = _unit(_vector(basis_u, "registry.resting_pose.plane_basis_u_m"), "registry.resting_pose.plane_basis_u_m")
    v = _unit(_vector(basis_v, "registry.resting_pose.plane_basis_v_m"), "registry.resting_pose.plane_basis_v_m")
    if abs(_dot(u, v)) > 1.0e-6 or abs(_dot(u, n)) > 1.0e-6 or abs(_dot(v, n)) > 1.0e-6:
        raise SourcePlacementError("registry_plane_basis_not_orthogonal", "resting_pose plane basis must be orthogonal", asset_id=record.get("asset_id"))
    if _dot(_cross(u, v), n) <= 0.0:
        raise SourcePlacementError("registry_plane_basis_not_right_handed", "resting_pose plane basis must be right-handed", asset_id=record.get("asset_id"))
    result = {"attachment_surface":surface,"base_plane_offset_m":offset,"height_m":height,"footprint_extent_m":list(footprint),"plane_normal_m":list(n),"plane_basis_u_m":list(u),"plane_basis_v_m":list(v),"measured_plane":pose.get("measured_plane"),"source":"registry.resting_pose"}
    if bounds_min is not None and bounds_max is not None:
        result["bounds_min_m"] = list(bounds_min)
        result["bounds_max_m"] = list(bounds_max)
        result["bounds_source"] = "request.asset_geometry"
    return result


def _emitter(record: Mapping[str, Any]) -> dict[str, Any]:
    anchor_id = _text(record.get("default_emitter_anchor_id"), "registry.default_emitter_anchor_id")
    anchors = record.get("emitter_anchors")
    if not isinstance(anchors, Sequence):
        raise SourcePlacementError("missing_emitter_anchor", "registry.emitter_anchors must be a list", asset_id=record.get("asset_id"))
    matches = [anchor for anchor in anchors if isinstance(anchor, Mapping) and anchor.get("anchor_id") == anchor_id]
    if len(matches) != 1:
        raise SourcePlacementError("emitter_anchor_unresolved", "default emitter anchor must resolve exactly once", asset_id=record.get("asset_id"), anchor_id=anchor_id)
    anchor = matches[0]
    return {"anchor_id":anchor_id,"anchor_type":anchor.get("anchor_type"),"offset_m":list(_vector(anchor.get("offset_m"), "registry.emitter_anchors.offset_m")),"offset_space":anchor.get("offset_space"),"source":"registry.emitter_anchors"}


def _visual_geometry(geometry: Any, *, plane_tolerance_m: float) -> tuple[dict[str, Any], list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    data = _mapping(geometry, "visual_geometry")
    authority = str(data.get("authority") or data.get("geometry_authority") or data.get("kind") or "")
    lowered = authority.lower()
    if lowered in ACOUSTIC_GEOMETRY_MARKERS or "acoustic" in lowered or "proxy" in lowered:
        raise SourcePlacementError("acoustic_proxy_not_visual_geometry", "acoustic proxy geometry cannot justify visual support", geometry_authority=authority)
    if authority not in VISUAL_GEOMETRY_AUTHORITIES:
        raise SourcePlacementError("visual_geometry_authority_required", "visual_geometry.authority must identify visual geometry", geometry_authority=authority)
    geometry_id = _text(data.get("geometry_id") or data.get("id"), "visual_geometry.geometry_id")
    source_ref = _text(data.get("source_ref") or data.get("path") or data.get("source"), "visual_geometry.source_ref")
    if source_ref.startswith("/") and not Path(source_ref).is_file():
        raise SourcePlacementError("visual_geometry_source_missing", "visual geometry source path does not exist", source_ref=source_ref)
    vertices_raw = data.get("vertices_m")
    triangles_raw = data.get("triangles")
    if vertices_raw is None and data.get("vertices_path") is not None:
        try:
            import numpy as np
            vertices_raw = np.load(str(data["vertices_path"]), mmap_mode="r")
        except Exception as exc:
            raise SourcePlacementError("visual_geometry_vertices_unreadable", "visual geometry vertices_path cannot be read", source_ref=str(data.get("vertices_path")), detail=str(exc)) from exc
    if triangles_raw is None and data.get("triangles_path") is not None:
        try:
            import numpy as np
            triangles_raw = np.load(str(data["triangles_path"]), mmap_mode="r")
        except Exception as exc:
            raise SourcePlacementError("visual_geometry_triangles_unreadable", "visual geometry triangles_path cannot be read", source_ref=str(data.get("triangles_path")), detail=str(exc)) from exc
    if vertices_raw is None or triangles_raw is None:
        raise SourcePlacementError("visual_geometry_data_missing", "support planning requires loaded visual vertices and triangles; a scene path alone is not a support surface")
    try:
        vertices = [_vector(row, "visual_geometry.vertices_m[]") for row in vertices_raw]
        triangles = [tuple(int(value) for value in row) for row in triangles_raw]
    except Exception as exc:
        raise SourcePlacementError("visual_geometry_arrays_invalid", "visual geometry arrays must contain finite vertices and integer triangles", detail=str(exc)) from exc
    if not vertices or not triangles:
        raise SourcePlacementError("visual_geometry_arrays_empty", "visual geometry arrays must be non-empty")
    for index, tri in enumerate(triangles):
        if len(tri) != 3 or any(value < 0 or value >= len(vertices) for value in tri):
            raise SourcePlacementError("visual_geometry_triangle_invalid", "visual geometry triangle index is outside vertices", triangle_index=index)
    bounds_min = [min(row[index] for row in vertices) for index in range(3)]
    bounds_max = [max(row[index] for row in vertices) for index in range(3)]
    return ({"geometry_id":geometry_id,"authority":authority,"source_ref":source_ref,"vertex_count":len(vertices),"triangle_count":len(triangles),"bounds_min_m":bounds_min,"bounds_max_m":bounds_max,"plane_tolerance_m":plane_tolerance_m}, vertices, triangles)


def parse_support_surfaces(layout: Any, *, room_id: str | None = None) -> list[dict[str, Any]]:
    if isinstance(layout, Sequence) and not isinstance(layout, (str, bytes)):
        raw = layout
    else:
        data = _mapping(layout, "layout")
        raw = data.get("support_surfaces")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise SourcePlacementError("support_surface_data_missing", "layout must declare a non-empty support_surfaces list; furniture geometry is not inferred")
    result=[]
    for index, item in enumerate(raw):
        row=_mapping(item, f"support_surfaces[{index}]")
        surface_id=_text(row.get("surface_id") or row.get("support_surface_id"), f"support_surfaces[{index}].surface_id")
        kind=_text(row.get("surface_kind") or row.get("kind"), f"support_surfaces[{index}].surface_kind")
        if kind not in SUPPORT_SURFACE_KINDS:
            raise SourcePlacementError("support_surface_kind_invalid", "support surface kind is unsupported", surface_id=surface_id, surface_kind=kind)
        declared_room=row.get("room_id")
        if room_id is not None and declared_room is not None and str(declared_room)!=str(room_id):
            raise SourcePlacementError("support_surface_room_mismatch", "support surface belongs to another room", surface_id=surface_id, room_id=declared_room, requested_room_id=room_id)
        origin=_vector(row.get("origin_m") or row.get("point_m"), f"support_surfaces[{index}].origin_m")
        normal=_unit(_vector(row.get("normal_m"), f"support_surfaces[{index}].normal_m"), f"support_surfaces[{index}].normal_m")
        basis_u=_unit(_vector(row.get("basis_u_m"), f"support_surfaces[{index}].basis_u_m"), f"support_surfaces[{index}].basis_u_m")
        basis_v=_unit(_vector(row.get("basis_v_m"), f"support_surfaces[{index}].basis_v_m"), f"support_surfaces[{index}].basis_v_m")
        if abs(_dot(basis_u,basis_v))>1.0e-6 or abs(_dot(basis_u,normal))>1.0e-6 or abs(_dot(basis_v,normal))>1.0e-6:
            raise SourcePlacementError("support_basis_not_orthogonal", "support surface basis must be orthogonal", surface_id=surface_id)
        if _dot(_cross(basis_u,basis_v),normal)<=0.0:
            raise SourcePlacementError("support_basis_not_right_handed", "support surface basis must be right-handed", surface_id=surface_id)
        extent=row.get("extent_m"); bounds_u=row.get("bounds_u_m"); bounds_v=row.get("bounds_v_m")
        if bounds_u is None or bounds_v is None:
            if extent is None: raise SourcePlacementError("support_bounds_missing", "support surface needs bounds_u_m/bounds_v_m or extent_m", surface_id=surface_id)
            ext=_pair(extent,f"support_surfaces[{index}].extent_m"); bounds_u=(-ext[0]/2.0,ext[0]/2.0); bounds_v=(-ext[1]/2.0,ext[1]/2.0)
        else: bounds_u=_pair(bounds_u,f"support_surfaces[{index}].bounds_u_m"); bounds_v=_pair(bounds_v,f"support_surfaces[{index}].bounds_v_m")
        raw_ref=row.get("geometry_ref")
        measured_source=_mapping(row.get("source"),f"support_surfaces[{index}].source") if raw_ref is None else {}
        if raw_ref is None and measured_source:
            # A catalog that fitted this surface from its own capture has no index
            # into a shared triangle array. Its authority comes from what it was
            # fitted on, and the plane is cross-checked against its own bounds
            # instead of against triangles it never referenced.
            derivation=str(measured_source.get("geometry_derivation") or "").lower()
            visual_depth=measured_source.get("visual_depth_source") or measured_source.get("capture_root")
            if 'acoustic' in derivation or 'proxy' in derivation:
                raise SourcePlacementError("acoustic_proxy_not_visual_geometry", "support surface source identifies acoustic/proxy geometry", surface_id=surface_id, geometry_derivation=derivation)
            if not ('depth' in derivation or 'visual' in derivation) or not visual_depth:
                raise SourcePlacementError("visual_support_geometry_required", "a measured support surface must name the visual capture it was fitted on", surface_id=surface_id, geometry_derivation=derivation or None)
            geometry_ref={"authority":"visual_geometry",
                          "geometry_id":_text(measured_source.get("capture_root") or visual_depth, f"support_surfaces[{index}].source.capture_root"),
                          "triangle_indices":None,
                          "geometry_derivation":derivation,
                          "visual_depth_source":str(visual_depth),
                          "frame_index":measured_source.get("frame_index"),
                          "cross_check":"own_capture_fit_no_shared_triangle_array"}
        else:
            geometry_ref=_mapping(raw_ref,f"support_surfaces[{index}].geometry_ref")
        authority=str(geometry_ref.get("authority") or geometry_ref.get("geometry_authority") or geometry_ref.get("kind") or "").lower()
        if authority in ACOUSTIC_GEOMETRY_MARKERS or 'acoustic' in authority or 'proxy' in authority: raise SourcePlacementError("acoustic_proxy_not_visual_geometry", "support surface geometry_ref identifies acoustic/proxy geometry", surface_id=surface_id, geometry_authority=authority)
        if authority not in VISUAL_GEOMETRY_AUTHORITIES: raise SourcePlacementError("visual_support_geometry_required", "support surface must reference visual geometry", surface_id=surface_id, geometry_authority=authority)
        geometry_id=_text(geometry_ref.get("geometry_id") or geometry_ref.get("id"),f"support_surfaces[{index}].geometry_ref.geometry_id")
        tri_indices=geometry_ref.get("triangle_indices")
        if tri_indices is not None:
            if isinstance(tri_indices,(str,bytes)) or not isinstance(tri_indices,Sequence) or not tri_indices: raise SourcePlacementError("support_surface_triangles_missing", "geometry_ref.triangle_indices must be a non-empty list when supplied", surface_id=surface_id)
            tri_indices=[int(value) for value in tri_indices]
        reference={'authority':authority,'geometry_id':geometry_id,'triangle_indices':tri_indices}
        for key in ('geometry_derivation','visual_depth_source','frame_index','cross_check'):
            if geometry_ref.get(key) is not None:
                reference[key]=geometry_ref[key]
        result.append({'surface_id':surface_id,'surface_kind':kind,'room_id':declared_room,'origin_m':list(origin),'normal_m':list(normal),'basis_u_m':list(basis_u),'basis_v_m':list(basis_v),'bounds_u_m':list(bounds_u),'bounds_v_m':list(bounds_v),'geometry_ref':reference,'source':'layout.support_surfaces'})
    return result


def _surface_geometry_check(surface, geometry, vertices, triangles, plane_tolerance_m):
    ref=surface['geometry_ref'];
    if ref.get('geometry_id')!=geometry['geometry_id']: raise SourcePlacementError('support_geometry_identity_mismatch','support surface and visual geometry IDs differ',surface_id=surface['surface_id'])
    indices=ref.get('triangle_indices')
    if not indices: raise SourcePlacementError('support_surface_triangles_missing','support surface must name visual geometry triangle_indices',surface_id=surface['surface_id'])
    origin=surface['origin_m']; normal=surface['normal_m']; u=surface['basis_u_m']; v=surface['basis_v_m']; bu=surface['bounds_u_m']; bv=surface['bounds_v_m']
    for tri_index in indices:
        if tri_index<0 or tri_index>=len(triangles): raise SourcePlacementError('support_surface_triangle_invalid','support triangle index is outside visual geometry',surface_id=surface['surface_id'],triangle_index=tri_index)
        for vertex_index in triangles[tri_index]:
            point=vertices[vertex_index]
            if abs(_dot(_sub(point,origin),normal))>plane_tolerance_m: raise SourcePlacementError('support_surface_not_planar','named visual triangles do not lie on declared support plane',surface_id=surface['surface_id'],triangle_index=tri_index)
            cu=_dot(_sub(point,origin),u); cv=_dot(_sub(point,origin),v)
            if not (bu[0]-plane_tolerance_m<=cu<=bu[1]+plane_tolerance_m and bv[0]-plane_tolerance_m<=cv<=bv[1]+plane_tolerance_m): raise SourcePlacementError('support_surface_bounds_mismatch','named visual triangles fall outside support bounds',surface_id=surface['surface_id'],triangle_index=tri_index)


def _candidate_uv(surface, footprint, search):
    step=_finite(search.get('grid_step_m'),'candidate_search.grid_step_m');
    if step<=0.0: raise SourcePlacementError('candidate_grid_step_invalid','candidate_search.grid_step_m must be positive')
    max_candidates=search.get('max_candidates')
    if isinstance(max_candidates,bool) or not isinstance(max_candidates,int) or max_candidates<1: raise SourcePlacementError('candidate_bound_invalid','candidate_search.max_candidates must be a positive integer')
    edge=_finite(search.get('edge_margin_m'),'candidate_search.edge_margin_m')
    if edge<0.0: raise SourcePlacementError('candidate_edge_margin_invalid','candidate_search.edge_margin_m must be nonnegative')
    half_u=float(footprint[0])/2.0; half_v=float(footprint[1])/2.0; bu=surface['bounds_u_m']; bv=surface['bounds_v_m']
    lo_u=bu[0]+edge+half_u; hi_u=bu[1]-edge-half_u; lo_v=bv[0]+edge+half_v; hi_v=bv[1]-edge-half_v
    if lo_u>hi_u or lo_v>hi_v: raise SourcePlacementError('no_bounded_support_candidate','support bounds cannot contain asset footprint',surface_id=surface['surface_id'])
    explicit=search.get('candidate_points_uv')
    if explicit is not None:
        if isinstance(explicit,(str,bytes)) or not isinstance(explicit,Sequence): raise SourcePlacementError('candidate_points_invalid','candidate_search.candidate_points_uv must be a list')
        rows=[]
        for index,row in enumerate(explicit):
            if isinstance(row,(str,bytes)) or not isinstance(row,Sequence) or len(row)!=2: raise SourcePlacementError('candidate_points_invalid','candidate point must be [u,v]',candidate_index=index)
            u=_finite(row[0],f'candidate_points_uv[{index}][0]'); v=_finite(row[1],f'candidate_points_uv[{index}][1]')
            if lo_u<=u<=hi_u and lo_v<=v<=hi_v: rows.append((u,v))
            if len(rows)>=max_candidates: break
        if not rows: raise SourcePlacementError('no_bounded_support_candidate','explicit support candidates do not fit footprint',surface_id=surface['surface_id'])
        return rows
    rows=[]; u=lo_u
    while u<=hi_u+1e-12 and len(rows)<max_candidates:
        v=lo_v
        while v<=hi_v+1e-12 and len(rows)<max_candidates: rows.append((u,v)); v+=step
        u+=step
    if not rows: raise SourcePlacementError('no_bounded_support_candidate','candidate grid produced no point',surface_id=surface['surface_id'])
    return rows



def _quaternion_xyzw(rotation: Sequence[Sequence[float]]) -> list[float]:
    trace = float(rotation[0][0] + rotation[1][1] + rotation[2][2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (float(rotation[2][1]) - float(rotation[1][2])) / scale
        y = (float(rotation[0][2]) - float(rotation[2][0])) / scale
        z = (float(rotation[1][0]) - float(rotation[0][1])) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + float(rotation[0][0]) - float(rotation[1][1]) - float(rotation[2][2])) * 2.0
        w = (float(rotation[2][1]) - float(rotation[1][2])) / scale
        x = 0.25 * scale
        y = (float(rotation[0][1]) + float(rotation[1][0])) / scale
        z = (float(rotation[0][2]) + float(rotation[2][0])) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + float(rotation[1][1]) - float(rotation[0][0]) - float(rotation[2][2])) * 2.0
        w = (float(rotation[0][2]) - float(rotation[2][0])) / scale
        x = (float(rotation[0][1]) + float(rotation[1][0])) / scale
        y = 0.25 * scale
        z = (float(rotation[1][2]) + float(rotation[2][1])) / scale
    else:
        scale = math.sqrt(1.0 + float(rotation[2][2]) - float(rotation[0][0]) - float(rotation[1][1])) * 2.0
        w = (float(rotation[1][0]) - float(rotation[0][1])) / scale
        x = (float(rotation[0][2]) + float(rotation[2][0])) / scale
        y = (float(rotation[1][2]) + float(rotation[2][1])) / scale
        z = 0.25 * scale
    return [float(x), float(y), float(z), float(w)]


def _transform_point(
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
    point: Sequence[float],
) -> tuple[float, float, float]:
    return _add(translation, _mat3_vec(rotation, point))


def _asset_world_aabb(
    pose: Mapping[str, Any],
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> dict[str, Any]:
    local_min = pose.get("bounds_min_m")
    local_max = pose.get("bounds_max_m")
    bounds_source = str(pose.get("bounds_source") or "resting_pose_footprint_height_conservative")
    if local_min is None or local_max is None:
        normal = pose["plane_normal_m"]
        basis_u = pose["plane_basis_u_m"]
        basis_v = pose["plane_basis_v_m"]
        offset = float(pose["base_plane_offset_m"])
        height = float(pose["height_m"])
        half_u = float(pose["footprint_extent_m"][0]) / 2.0
        half_v = float(pose["footprint_extent_m"][1]) / 2.0
        base = _scale(normal, offset)
        local_points = []
        for u_sign in (-1.0, 1.0):
            for v_sign in (-1.0, 1.0):
                contact = _add(base, _add(_scale(basis_u, u_sign * half_u), _scale(basis_v, v_sign * half_v)))
                local_points.append(contact)
                local_points.append(_add(contact, _scale(normal, height)))
        local_min = [min(point[index] for point in local_points) for index in range(3)]
        local_max = [max(point[index] for point in local_points) for index in range(3)]
    else:
        local_min = list(_vector(local_min, "asset_bounds.local_min_m"))
        local_max = list(_vector(local_max, "asset_bounds.local_max_m"))
        bounds_source = str(pose.get("bounds_source") or "request.asset_geometry")
    corners = []
    for x in (float(local_min[0]), float(local_max[0])):
        for y in (float(local_min[1]), float(local_max[1])):
            for z in (float(local_min[2]), float(local_max[2])):
                corners.append(_transform_point(rotation, translation, (x, y, z)))
    world_min = [min(point[index] for point in corners) for index in range(3)]
    world_max = [max(point[index] for point in corners) for index in range(3)]
    return {
        "source": bounds_source,
        "local_min_m": list(local_min),
        "local_max_m": list(local_max),
        "world_aabb_min_m": world_min,
        "world_aabb_max_m": world_max,
    }


def _row_world_aabb(row: Mapping[str, Any]) -> dict[str, Any]:
    existing = row.get("asset_bounds")
    if isinstance(existing, Mapping):
        low = existing.get("world_aabb_min_m")
        high = existing.get("world_aabb_max_m")
        if low is not None and high is not None:
            low_v = list(_vector(low, "existing_placement.asset_bounds.world_aabb_min_m"))
            high_v = list(_vector(high, "existing_placement.asset_bounds.world_aabb_max_m"))
            if any(left >= right for left, right in zip(low_v, high_v)):
                raise SourcePlacementError("existing_placement_bounds_invalid", "existing placement AABB must have positive extent", instance_id=row.get("instance_id"))
            return {
                "source": existing.get("source", "existing_placement.asset_bounds"),
                "local_min_m": existing.get("local_min_m"),
                "local_max_m": existing.get("local_max_m"),
                "world_aabb_min_m": low_v,
                "world_aabb_max_m": high_v,
            }
    for owner in ("world_aabb", "aabb"):
        value = row.get(owner)
        if isinstance(value, Mapping) and value.get("min_m") is not None and value.get("max_m") is not None:
            low_v = list(_vector(value["min_m"], f"existing_placement.{owner}.min_m"))
            high_v = list(_vector(value["max_m"], f"existing_placement.{owner}.max_m"))
            if any(left >= right for left, right in zip(low_v, high_v)):
                raise SourcePlacementError("existing_placement_bounds_invalid", "existing placement AABB must have positive extent", instance_id=row.get("instance_id"))
            return {
                "source": f"existing_placement.{owner}",
                "world_aabb_min_m": low_v,
                "world_aabb_max_m": high_v,
            }
    if row.get("aabb_min_m") is not None and row.get("aabb_max_m") is not None:
        low_v = list(_vector(row["aabb_min_m"], "existing_placement.aabb_min_m"))
        high_v = list(_vector(row["aabb_max_m"], "existing_placement.aabb_max_m"))
        if any(left >= right for left, right in zip(low_v, high_v)):
            raise SourcePlacementError("existing_placement_bounds_invalid", "existing placement AABB must have positive extent", instance_id=row.get("instance_id"))
        return {
            "source": "existing_placement.aabb",
            "world_aabb_min_m": low_v,
            "world_aabb_max_m": high_v,
        }
    if isinstance(row.get("root_transform"), Mapping) and isinstance(row.get("asset_resting_pose"), Mapping):
        matrix = row["root_transform"].get("matrix_row_major")
        if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)) or len(matrix) != 16:
            raise SourcePlacementError("existing_placement_transform_missing", "existing placement root_transform must carry a 4x4 row-major matrix", instance_id=row.get("instance_id"))
        rotation = [
            [float(matrix[0]), float(matrix[1]), float(matrix[2])],
            [float(matrix[4]), float(matrix[5]), float(matrix[6])],
            [float(matrix[8]), float(matrix[9]), float(matrix[10])],
        ]
        translation = [float(matrix[3]), float(matrix[7]), float(matrix[11])]
        return _asset_world_aabb(row["asset_resting_pose"], rotation, translation)
    raise SourcePlacementError(
        "existing_placement_bounds_missing",
        "existing placement must carry asset_bounds.world_aabb_min_m/world_aabb_max_m or root_transform plus asset_resting_pose",
        instance_id=row.get("instance_id"),
    )


def _aabb_overlaps(first: Mapping[str, Any], second: Mapping[str, Any], gap_m: float) -> bool:
    first_low = first["world_aabb_min_m"]
    first_high = first["world_aabb_max_m"]
    second_low = second["world_aabb_min_m"]
    second_high = second["world_aabb_max_m"]
    return all(
        float(first_low[index]) < float(second_high[index]) - gap_m
        and float(second_low[index]) < float(first_high[index]) - gap_m
        for index in range(3)
    )


def _batch_clearance(
    row: Mapping[str, Any],
    checked_against: Sequence[Mapping[str, Any]],
    *,
    overlap_conflicts: Sequence[Mapping[str, Any]] = (),
    status: str = "partial",
) -> dict[str, Any]:
    return {
        "status": status,
        "inter_instance_aabb": {
            "status": "fail" if overlap_conflicts else "pass",
            "scope": "batch_planned_instances",
            "checked_against": [item.get("instance_id") for item in checked_against],
            "overlap_conflicts": deepcopy(list(overlap_conflicts)),
            "aabb_min_m": deepcopy(row["asset_bounds"]["world_aabb_min_m"]),
            "aabb_max_m": deepcopy(row["asset_bounds"]["world_aabb_max_m"]),
        },
        "room_collision": {
            "status": "not_run",
            "reason": "room-wide collision query was not supplied to the helper",
        },
        "claim_boundary": "only the bounded inter-instance AABB check ran; room collision remains not_run",
    }


def plan_source_placement(registry,room,layout,visual_geometry,request,*,config):
    room_data=_mapping(room,'room'); room_id=_text(room_data.get('room_id'),'room.room_id'); req=_mapping(request,'request'); asset=_asset_record(registry,req.get('asset_id'),req.get('asset_revision')); support_id=_text(req.get('support_surface_id'),'request.support_surface_id')
    conf=_mapping(config,'config'); normal_tol=_finite(conf.get('normal_tolerance_deg'),'config.normal_tolerance_deg'); plane_tol=_finite(conf.get('plane_tolerance_m'),'config.plane_tolerance_m')
    if normal_tol<0 or normal_tol>=90: raise SourcePlacementError('normal_tolerance_invalid','config.normal_tolerance_deg must be in [0,90)')
    if plane_tol<0: raise SourcePlacementError('plane_tolerance_invalid','config.plane_tolerance_m must be nonnegative')
    surfaces=parse_support_surfaces(layout,room_id=room_id); matches=[s for s in surfaces if s['surface_id']==support_id]
    if len(matches)!=1: raise SourcePlacementError('support_surface_unresolved','request.support_surface_id must resolve to one layout surface',support_surface_id=support_id,available=[s['surface_id'] for s in surfaces])
    surface=matches[0]; gsummary,vertices,triangles=_visual_geometry(visual_geometry,plane_tolerance_m=plane_tol)
    cross_checked=surface['geometry_ref'].get('triangle_indices') is not None and surface['geometry_ref'].get('geometry_id')==gsummary['geometry_id']
    if cross_checked:
        _surface_geometry_check(surface,gsummary,vertices,triangles,plane_tol)
    elif surface['geometry_ref'].get('cross_check')!='own_capture_fit_no_shared_triangle_array':
        _surface_geometry_check(surface,gsummary,vertices,triangles,plane_tol)
    pose=_resting_pose(asset,req); emitter=_emitter(asset); candidates=_candidate_uv(surface,pose['footprint_extent_m'],_mapping(conf.get('candidate_search'),'config.candidate_search')); candidate_index=req.get('candidate_index',0)
    if isinstance(candidate_index,bool) or not isinstance(candidate_index,int) or candidate_index<0 or candidate_index>=len(candidates): raise SourcePlacementError('candidate_index_invalid','request.candidate_index must select a bounded candidate',candidate_index=candidate_index,candidate_count=len(candidates))
    yaw=math.radians(_finite(req.get('yaw_deg',0.0),'request.yaw_deg')); sn=_unit(surface['normal_m'],'support.normal_m'); su=_unit(surface['basis_u_m'],'support.basis_u_m'); sv=_unit(surface['basis_v_m'],'support.basis_v_m'); ru=_unit(_add(_scale(su,math.cos(yaw)),_scale(sv,math.sin(yaw))),'rotated support basis u'); rv=_unit(_cross(sn,ru),'rotated support basis v'); an=pose['plane_normal_m']; au=pose['plane_basis_u_m']; av=pose['plane_basis_v_m']; rotation=[[0.0]*3 for _ in range(3)]; _outer_add(rotation,ru,au); _outer_add(rotation,rv,av); _outer_add(rotation,sn,an)
    if _dot(_mat3_vec(rotation,an),sn)<math.cos(math.radians(normal_tol)): raise SourcePlacementError('asset_support_normal_misaligned','asset plane normal cannot align to support normal within configured tolerance',asset_id=asset['asset_id'],support_surface_id=support_id)
    u_coord,v_coord=candidates[candidate_index]
    # A fitted support plane is a plane through a real slab, not the slab's top.
    # Where the measured surface under this footprint sits above the fit, the
    # request states that distance along the support normal so the asset rests on
    # the surface that is actually there. It moves the asset, never the surface:
    # no plane is invented and no asset geometry is rescaled.
    normal_offset=_finite(req.get('surface_normal_offset_m',0.0),'request.surface_normal_offset_m')
    point=_add(surface['origin_m'],_add(_scale(su,u_coord),_scale(sv,v_coord)))
    if normal_offset:
        point=_add(point,_scale(sn,normal_offset))
    root=_sub(point,_mat3_vec(rotation,_scale(an,pose['base_plane_offset_m']))); emitter_world=_add(root,_mat3_vec(rotation,emitter['offset_m']))
    asset_bounds=_asset_world_aabb(pose,rotation,root)
    explicit_candidate = candidate_index if "candidate_index" in req else None
    return {'schema':SOURCE_PLACEMENT_SCHEMA,'status':'planned','qualification_status':'not_run','native_execution':'not_run','instance_id':_text(req.get('instance_id'),'request.instance_id'),'asset_id':asset['asset_id'],'asset_revision':asset.get('revision'),'room_id':room_id,'support_identity':{'room_id':room_id,'surface_id':surface['surface_id'],'surface_kind':surface['surface_kind'],'geometry_ref':deepcopy(surface['geometry_ref']),'origin_m':list(surface['origin_m']),'normal_m':list(surface['normal_m']),'candidate_uv_m':[u_coord,v_coord],'surface_normal_offset_m':normal_offset,
                              'surface_normal_offset_basis':(
                                  'measured slab top under this footprint, above the fitted plane'
                                  if normal_offset else 'none; the asset rests on the fitted plane')},'candidate':{'index':candidate_index,'count':len(candidates),'requested_index':explicit_candidate,'selected_index':candidate_index,'selection_mode':'explicit' if explicit_candidate is not None else 'single_default','search':deepcopy(dict(conf['candidate_search'])),'footprint_fit':True,'rejected_indices':[],'rejected_for_clearance':[]},'asset_resting_pose':pose,'asset_bounds':asset_bounds,'emitter':emitter,'root_transform':{'matrix_row_major':_mat4(rotation,root),'rotation_xyzw':_quaternion_xyzw(rotation),'translation_m':list(root)},'emitter_transform':{'matrix_row_major':_mat4(rotation,emitter_world),'rotation_xyzw':_quaternion_xyzw(rotation),'position_m':list(emitter_world)},'clearance':{'status':'not_run','inter_instance_aabb':{'status':'not_run','scope':'single placement; no peer list supplied'},'room_collision':{'status':'not_run','reason':'room-wide collision query was not supplied to the helper'},'reason':'no batch peer list supplied; room collision remains not_run'},'planning_evidence':[{'kind':'registry.resting_pose','status':'observed','source':'registry.resting_pose'},{'kind':'registry.emitter_anchor','status':'observed','source':'registry.emitter_anchors','anchor_id':emitter['anchor_id']},{'kind':'asset_bounds','status':'observed' if asset_bounds['source']=='request.asset_geometry' else 'derived_conservative','source':asset_bounds['source']},{'kind':'layout.support_surface','status':'observed','surface_id':surface['surface_id'],'surface_kind':surface['surface_kind']},{'kind':'visual_geometry','status':'observed' if cross_checked else 'observed_on_its_own_capture','geometry_id':gsummary['geometry_id'],'authority':gsummary['authority'],'source_ref':gsummary['source_ref'],'triangle_indices':surface['geometry_ref']['triangle_indices'],'surface_geometry_cross_check':'named_triangles_on_the_supplied_geometry' if cross_checked else surface['geometry_ref'].get('cross_check'),'surface_geometry_id':surface['geometry_ref']['geometry_id'],'surface_visual_depth_source':surface['geometry_ref'].get('visual_depth_source')},{'kind':'candidate_search','status':'planned','max_candidates':conf['candidate_search']['max_candidates'],'candidate_index':candidate_index},{'kind':'support_normal_offset','status':'observed' if normal_offset else 'not_applied','offset_m':normal_offset,'source':req.get('surface_normal_offset_source')}], 'claim_boundary':'planning transform and bounded support fit only; no native visual support, room collision or dataset admission claim'}


def plan_static_source_placements(
    registry,
    room,
    layout,
    visual_geometry,
    requests,
    *,
    config,
    existing_placements: Sequence[Mapping[str, Any]] | None = None,
):
    conf = _mapping(config, "config")
    gap_m = _finite(conf.get("min_inter_instance_gap_m", 0.0), "config.min_inter_instance_gap_m")
    if gap_m < 0.0:
        raise SourcePlacementError("inter_instance_gap_invalid", "config.min_inter_instance_gap_m must be nonnegative")
    occupied: list[dict[str, Any]] = []
    if existing_placements is not None:
        if isinstance(existing_placements, (str, bytes)) or not isinstance(existing_placements, Sequence):
            raise SourcePlacementError("existing_placements_invalid", "existing_placements must be a list of planned placement rows")
        for index, item in enumerate(existing_placements):
            row = _mapping(item, f"existing_placements[{index}]")
            if row.get("status") == "rejected":
                continue
            box = _row_world_aabb(row)
            occupied.append({
                "instance_id": row.get("instance_id"),
                "asset_id": row.get("asset_id"),
                "asset_bounds": box,
            })

    rows = []
    placement_order = []
    for request_index, request in enumerate(requests):
        req = dict(request) if isinstance(request, Mapping) else {}
        explicit = "candidate_index" in req
        candidate_indices = [req.get("candidate_index")] if explicit else [0]
        candidate_count = None
        valid_rows: dict[int, dict[str, Any]] = {}
        conflict_rows: list[dict[str, Any]] = []
        last_error: SourcePlacementError | None = None
        selected = None
        selected_conflicts: list[dict[str, Any]] = []
        for candidate_index in candidate_indices:
            trial = dict(req)
            trial["candidate_index"] = candidate_index
            trial.setdefault("surface_normal_offset_m", req.get("surface_normal_offset_m", 0.0))
            try:
                planned = plan_source_placement(
                    registry, room, layout, visual_geometry, trial, config=conf
                )
            except SourcePlacementError as exc:
                last_error = exc
                break
            valid_rows[int(candidate_index)] = planned
            candidate_count = int(planned["candidate"]["count"])
            conflicts = []
            for peer in occupied:
                if _aabb_overlaps(planned["asset_bounds"], peer["asset_bounds"], gap_m):
                    conflicts.append({
                        "against_instance_id": peer.get("instance_id"),
                        "against_asset_id": peer.get("asset_id"),
                        "candidate_index": int(candidate_index),
                        "candidate_aabb_min_m": deepcopy(planned["asset_bounds"]["world_aabb_min_m"]),
                        "candidate_aabb_max_m": deepcopy(planned["asset_bounds"]["world_aabb_max_m"]),
                        "against_aabb_min_m": deepcopy(peer["asset_bounds"]["world_aabb_min_m"]),
                        "against_aabb_max_m": deepcopy(peer["asset_bounds"]["world_aabb_max_m"]),
                    })
            if conflicts:
                conflict_rows.extend(conflicts)
                selected_conflicts = conflicts
                if explicit:
                    break
                continue
            selected = planned
            break
        if not explicit and selected is None and candidate_count is not None:
            for candidate_index in range(1, candidate_count):
                trial = dict(req)
                trial["candidate_index"] = candidate_index
                try:
                    planned = plan_source_placement(
                        registry, room, layout, visual_geometry, trial, config=conf
                    )
                except SourcePlacementError as exc:
                    last_error = exc
                    continue
                valid_rows[int(candidate_index)] = planned
                conflicts = []
                for peer in occupied:
                    if _aabb_overlaps(planned["asset_bounds"], peer["asset_bounds"], gap_m):
                        conflicts.append({
                            "against_instance_id": peer.get("instance_id"),
                            "against_asset_id": peer.get("asset_id"),
                            "candidate_index": int(candidate_index),
                            "candidate_aabb_min_m": deepcopy(planned["asset_bounds"]["world_aabb_min_m"]),
                            "candidate_aabb_max_m": deepcopy(planned["asset_bounds"]["world_aabb_max_m"]),
                            "against_aabb_min_m": deepcopy(peer["asset_bounds"]["world_aabb_min_m"]),
                            "against_aabb_max_m": deepcopy(peer["asset_bounds"]["world_aabb_max_m"]),
                        })
                if conflicts:
                    conflict_rows.extend(conflicts)
                    selected_conflicts = conflicts
                    continue
                selected = planned
                break

        if selected is not None:
            rejected_indices = sorted({
                int(conflict["candidate_index"]) for conflict in conflict_rows
            })
            selected["candidate"].update({
                "requested_index": int(req["candidate_index"]) if explicit else None,
                "selected_index": int(selected["candidate"]["index"]),
                "selection_mode": "explicit" if explicit else "bounded_joint_nonoverlap",
                "rejected_indices": rejected_indices,
                "rejected_for_clearance": deepcopy(conflict_rows),
            })
            selected["clearance"] = _batch_clearance(
                selected,
                occupied,
                status="partial",
            )
            selected["joint_placement"] = {
                "status": "placed",
                "request_index": request_index,
                "checked_against": [item.get("instance_id") for item in occupied],
                "min_inter_instance_gap_m": gap_m,
            }
            occupied.append({
                "instance_id": selected["instance_id"],
                "asset_id": selected["asset_id"],
                "asset_bounds": deepcopy(selected["asset_bounds"]),
            })
            placement_order.append(selected["instance_id"])
            rows.append(selected)
            continue

        if explicit and conflict_rows:
            reason = SourcePlacementError(
                "explicit_candidate_conflict",
                "explicit candidate_index overlaps a previously planned source; it was not changed",
                instance_id=req.get("instance_id"),
                candidate_index=req.get("candidate_index"),
                conflicts=conflict_rows,
            )
        elif conflict_rows:
            reason = SourcePlacementError(
                "joint_candidate_exhausted",
                "all bounded candidates overlap previously planned sources",
                instance_id=req.get("instance_id"),
                candidate_count=candidate_count,
                conflicts=conflict_rows,
            )
        elif last_error is not None:
            reason = last_error
        else:
            reason = SourcePlacementError(
                "joint_candidate_exhausted",
                "no bounded candidate could be planned",
                instance_id=req.get("instance_id"),
            )
        exemplar = valid_rows.get(0)
        if exemplar is None and valid_rows:
            exemplar = valid_rows[sorted(valid_rows)[0]]
        rejected = {
            "schema": SOURCE_PLACEMENT_SCHEMA,
            "status": "rejected",
            "qualification_status": "not_run",
            "native_execution": "not_run",
            "instance_id": req.get("instance_id"),
            "asset_id": req.get("asset_id"),
            "support_surface_id": req.get("support_surface_id"),
            "reason": reason.to_dict(),
            "claim_boundary": "candidate conflict or invalid placement input rejected; no candidate was silently changed",
        }
        if exemplar is not None:
            rejected.update({
                "room_id": exemplar.get("room_id"),
                "asset_revision": exemplar.get("asset_revision"),
                "support_identity": deepcopy(exemplar.get("support_identity")),
                "asset_resting_pose": deepcopy(exemplar.get("asset_resting_pose")),
                "asset_bounds": deepcopy(exemplar.get("asset_bounds")),
                "emitter": deepcopy(exemplar.get("emitter")),
                "root_transform": deepcopy(exemplar.get("root_transform")),
                "emitter_transform": deepcopy(exemplar.get("emitter_transform")),
            })
        rejected["candidate"] = {
            "requested_index": req.get("candidate_index") if explicit else None,
            "selected_index": None,
            "selection_mode": "explicit_rejected" if explicit else "bounded_joint_nonoverlap_exhausted",
            "count": candidate_count,
            "rejected_indices": sorted({int(item["candidate_index"]) for item in conflict_rows}),
            "rejected_for_clearance": deepcopy(conflict_rows),
            "search": deepcopy(dict(conf.get("candidate_search") or {})),
        }
        rejected["clearance"] = {
            "status": "partial" if conflict_rows else "not_run",
            "inter_instance_aabb": {
                "status": "fail" if conflict_rows else "not_run",
                "scope": "batch_planned_instances",
                "checked_against": [item.get("instance_id") for item in occupied],
                "overlap_conflicts": deepcopy(conflict_rows),
            },
            "room_collision": {
                "status": "not_run",
                "reason": "room-wide collision query was not supplied to the helper",
            },
        }
        rows.append(rejected)

    return {
        "schema": SOURCE_PLACEMENT_SCHEMA,
        "status": "planned" if rows and all(r.get("status") == "planned" for r in rows) else "partial",
        "native_execution": "not_run",
        "joint_candidate_selection": {
            "mode": "bounded_ordered_nonoverlap",
            "explicit_candidate_policy": "honor_exact_index_and_reject_on_inter_instance_overlap",
            "implicit_candidate_policy": "first_bounded_candidate_without_inter_instance_aabb_overlap",
            "min_inter_instance_gap_m": gap_m,
            "existing_placement_count": len(existing_placements or ()),
            "room_collision": "not_run",
            "placement_order": placement_order,
        },
        "instances": rows,
        "claim_boundary": "per-instance static placement planning with bounded inter-instance AABB checks; room collision and native execution remain not_run",
    }



def bind_support_catalog_requests(requests, catalog, *, registry=None, strict: bool = True):
    """Attach measured asset geometry from a support catalog to placement requests.

    ``plan_source_placement`` needs the measured plane normal, plane basis and
    mesh bounds; a registry ``resting_pose`` alone raises
    ``missing_registry_plane_normal``.  The measurements live in the support
    surface catalog next to the surfaces they were measured against, so an
    ordinary consumer binds them here instead of every caller reaching into the
    catalog by hand.  With ``registry`` given, the request also carries the
    registered revision so a stale asset revision is rejected rather than
    silently planned.

    ``strict`` (the default) raises when a request has no measured geometry;
    pass ``False`` to leave that request unbound and let the planner reject it.
    """
    catalog_map = _mapping(catalog, "catalog")
    measurements = catalog_map.get("asset_visual_geometry_measurements")
    if not isinstance(measurements, Mapping):
        measurements = catalog_map if all(
            isinstance(value, Mapping) and value.get("bounds_min_m") is not None
            for value in catalog_map.values()
        ) and catalog_map else {}
    if not isinstance(measurements, Mapping) or not measurements:
        raise SourcePlacementError(
            "support_catalog_measurements_missing",
            "catalog must carry asset_visual_geometry_measurements",
        )
    revisions = {}
    if registry is not None:
        for record in _mapping(registry, "registry").get("assets") or ():
            if isinstance(record, Mapping) and record.get("asset_id"):
                revisions[str(record["asset_id"])] = record.get("revision")
    bound = []
    for index, request in enumerate(requests or ()):
        row = dict(_mapping(request, f"requests[{index}]"))
        asset_id = _text(row.get("asset_id"), f"requests[{index}].asset_id")
        measured = measurements.get(asset_id)
        if not isinstance(measured, Mapping):
            if strict:
                raise SourcePlacementError(
                    "asset_geometry_unmeasured",
                    "the support catalog carries no measured geometry for this asset",
                    asset_id=asset_id,
                    measured_asset_ids=sorted(measurements),
                )
            bound.append(row)
            continue
        row.setdefault("asset_geometry", deepcopy(dict(measured)))
        if asset_id in revisions and revisions[asset_id] is not None:
            row.setdefault("asset_revision", revisions[asset_id])
        bound.append(row)
    return bound


def support_catalog_surface_kinds(catalog) -> dict:
    """Map each measured asset to the support surface kind it may legally use."""
    catalog_map = _mapping(catalog, "catalog")
    measurements = _mapping(catalog_map.get("asset_visual_geometry_measurements"), "catalog.asset_visual_geometry_measurements")
    return {
        str(asset_id): str(_mapping(value, "measurement").get("support_kind") or "")
        for asset_id, value in measurements.items()
    }


def _triangles_overlapping_box(vertices, triangles, box_min, box_max):
    """Exact triangle/AABB overlap by separating axes, after a box broad phase.

    numpy is imported here rather than at module import time: the planning path
    above is pure Python and must keep working where numpy is absent.
    """
    import numpy as np

    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    corners = vertices[triangles]
    lo = corners.min(axis=1)
    hi = corners.max(axis=1)
    near = np.all((lo <= box_max) & (hi >= box_min), axis=1)
    candidates = np.flatnonzero(near)
    if candidates.size == 0:
        return candidates, 0
    centre = (box_min + box_max) / 2.0
    half = (box_max - box_min) / 2.0
    points = corners[candidates] - centre
    edges = np.stack([points[:, 1] - points[:, 0],
                      points[:, 2] - points[:, 1],
                      points[:, 0] - points[:, 2]], axis=1)
    keep = np.ones(len(candidates), dtype=bool)
    for axis_index in range(3):
        projected = points[:, :, axis_index]
        keep &= ~((projected.min(axis=1) > half[axis_index])
                  | (projected.max(axis=1) < -half[axis_index]))
    normals = np.cross(edges[:, 0], edges[:, 1])
    distance = np.einsum("tc,tvc->tv", normals, points)
    radius = np.einsum("tc,c->t", np.abs(normals), half)
    keep &= ~((distance.min(axis=1) > radius) | (distance.max(axis=1) < -radius))
    basis = np.eye(3)
    for edge_index in range(3):
        for axis_index in range(3):
            axis = np.cross(edges[:, edge_index], basis[axis_index])
            projected = np.einsum("tc,tvc->tv", axis, points)
            extent = np.einsum("tc,c->t", np.abs(axis), half)
            degenerate = np.all(np.abs(axis) < 1.0e-12, axis=1)
            separated = ((projected.min(axis=1) > extent)
                         | (projected.max(axis=1) < -extent)) & ~degenerate
            keep &= ~separated
    return candidates[keep], int(candidates.size)


def room_collision_report(rows, *, room_geometry, config, room_id=None):
    """Test each planned placement against the room's own surface mesh.

    A peer AABB check says two sources do not overlap each other; it says
    nothing about the walls and furniture they may be standing inside. This runs
    the real query: every planned world AABB, shrunk by the configured
    penetration tolerance so resting contact with the support is not read as a
    collision, against the room's retained triangles.

    ``room_geometry`` is the room package's ``static_geometry`` block. Its
    ``representation``, ``source`` and ``source_to_common_frame`` are copied into
    the result so the reading can be traced to the mesh it came from. Without a
    usable mesh every row stays ``not_run``; it is never assumed clear.
    """
    conf = _mapping(config, "config")
    tolerance = _finite(conf.get("penetration_tolerance_m"), "config.penetration_tolerance_m")
    if tolerance < 0.0:
        raise SourcePlacementError(
            "penetration_tolerance_invalid",
            "config.penetration_tolerance_m must be nonnegative")
    geometry = _mapping(room_geometry, "room_geometry")
    representation = str(geometry.get("representation") or "")
    provenance = {
        "representation": representation or None,
        "source": geometry.get("source"),
        "source_manifest": geometry.get("source_manifest"),
        "source_to_common_frame": geometry.get("source_to_common_frame"),
        "coordinate_frame": deepcopy(geometry.get("coordinate_frame")),
        "vertices": geometry.get("vertices"),
        "triangles": geometry.get("triangles"),
        "penetration_tolerance_m": tolerance,
        "room_id": room_id,
    }
    vertices = geometry.get("vertices")
    triangles = geometry.get("triangles")
    loaded = None
    reason = None
    if representation and representation != "real_surface_mesh":
        reason = (f"room geometry representation {representation!r} is not a real surface mesh; "
                  "a proxy cannot decide whether a placement sits inside the room")
    else:
        try:
            import numpy as np
            if isinstance(vertices, (str, Path)) and isinstance(triangles, (str, Path)):
                loaded = (np.load(str(vertices), allow_pickle=False),
                          np.load(str(triangles), allow_pickle=False))
            elif vertices is not None and triangles is not None:
                loaded = (np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64))
            else:
                reason = "room geometry supplied no vertices/triangles"
        except Exception as exc:
            loaded = None
            reason = f"room geometry could not be read: {type(exc).__name__}: {exc}"
    results = []
    for index, row in enumerate(rows or ()):
        item = _mapping(row, f"rows[{index}]")
        entry = {
            "instance_id": item.get("instance_id"),
            "asset_id": item.get("asset_id"),
            "support_surface_id": _mapping(item.get("support_identity"), "support_identity").get("surface_id"),
        }
        if str(item.get("status")) != "planned":
            entry.update({"status": "not_run", "reason": "placement row is not planned"})
            results.append(entry)
            continue
        bounds = _mapping(item.get("asset_bounds"), "asset_bounds")
        low = bounds.get("world_aabb_min_m")
        high = bounds.get("world_aabb_max_m")
        if loaded is None:
            entry.update({"status": "not_run", "reason": reason or "no room geometry supplied"})
            results.append(entry)
            continue
        try:
            low = _vector(low, "asset_bounds.world_aabb_min_m")
            high = _vector(high, "asset_bounds.world_aabb_max_m")
        except SourcePlacementError as exc:
            entry.update({"status": "not_run", "reason": str(exc)})
            results.append(entry)
            continue
        shrunk_min = [a + tolerance for a in low]
        shrunk_max = [b - tolerance for b in high]
        if any(a >= b for a, b in zip(shrunk_min, shrunk_max)):
            entry.update({
                "status": "not_run",
                "reason": ("the asset is not larger than twice the configured penetration "
                           "tolerance on every axis, so no interior box remains to test"),
                "world_aabb_min_m": list(low), "world_aabb_max_m": list(high)})
            results.append(entry)
            continue
        hits, considered = _triangles_overlapping_box(loaded[0], loaded[1], shrunk_min, shrunk_max)
        entry.update({
            "status": "fail" if hits.size else "pass",
            "world_aabb_min_m": list(low),
            "world_aabb_max_m": list(high),
            "tested_aabb_min_m": shrunk_min,
            "tested_aabb_max_m": shrunk_max,
            "broad_phase_triangle_count": considered,
            "intersecting_triangle_count": int(hits.size),
            "intersecting_triangle_indices": [int(value) for value in hits[:16]],
            "reason": ("the placement interior intersects room geometry"
                       if hits.size else
                       "no room triangle reaches inside the placement beyond the tolerance"),
        })
        results.append(entry)
    return {
        "schema": "avengine_room_collision_report_v1",
        "status": ("fail" if any(r["status"] == "fail" for r in results)
                   else "pass" if results and all(r["status"] == "pass" for r in results)
                   else "not_run"),
        "geometry": provenance,
        "rows": results,
        "claim_boundary": ("room surface intersection only; it does not test navmesh "
                           "reachability, visibility or acoustic occlusion"),
    }


def apply_room_collision(plan, report):
    """Fold a room collision report into a placement plan's clearance rows."""
    result = deepcopy(dict(_mapping(plan, "plan")))
    by_instance = {str(row.get("instance_id")): row
                   for row in _mapping(report, "report").get("rows") or ()}
    rows = []
    for row in result.get("instances") or ():
        item = dict(row)
        found = by_instance.get(str(item.get("instance_id")))
        if found is not None:
            clearance = dict(_mapping(item.get("clearance"), "clearance"))
            clearance["room_collision"] = {
                "status": found["status"],
                "reason": found.get("reason"),
                "intersecting_triangle_count": found.get("intersecting_triangle_count"),
                "geometry": _mapping(report, "report").get("geometry"),
            }
            inter = _mapping(clearance.get("inter_instance_aabb"), "clearance.inter_instance_aabb")
            if found["status"] == "pass" and str(inter.get("status")) == "pass":
                clearance["status"] = "pass"
            elif found["status"] == "fail":
                clearance["status"] = "fail"
            item["clearance"] = clearance
        rows.append(item)
    result["instances"] = rows
    result["room_collision"] = {
        "status": _mapping(report, "report").get("status"),
        "geometry": _mapping(report, "report").get("geometry"),
    }
    return result

parse_support_surface_config=parse_support_surfaces
plan_static_source_placement=plan_source_placement
__all__=['SOURCE_PLACEMENT_SCHEMA','SourcePlacementError','parse_support_surfaces','parse_support_surface_config','plan_source_placement','plan_static_source_placement','plan_static_source_placements','bind_support_catalog_requests','support_catalog_surface_kinds','room_collision_report','apply_room_collision']
