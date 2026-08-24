"""Declared-certified Apartment viewpoint inputs for QA design-time planning.

This module intentionally validates only the evidence needed to prevent the
current native-pixel-pending 2-D prefilter from entering production design.
It does not define a frozen artifact schema or add content hashes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.m1.contracts import validate_capture_request
from avengine.m5_1.orientation import habitat_yaw_degrees_from_xyzw
from avengine.optional_backends.spear_visual import (
    camera_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)


MIN_VIEWPOINTS = 4
MAX_VIEWPOINTS = 8
CAMERA_MATCH_TOLERANCE = 1.0e-6


class DesignViewpointError(ValueError):
    """A viewpoint bank is incomplete or is only pending prefilter evidence."""


@dataclass(frozen=True)
class DesignCorridor:
    corridor_id: str
    start_ue_cm: tuple[float, float]
    end_ue_cm: tuple[float, float]


@dataclass(frozen=True)
class DesignViewpoint:
    viewpoint_id: str
    camera_position_ue_cm: tuple[float, float, float]
    camera_yaw_ue_deg: float
    m1_request_path: Path
    m1_request_reference: str
    m1_request_id: str
    corridors: tuple[DesignCorridor, ...]
    offscreen_position_ue_cm: tuple[float, float, float]
    authentication_evidence_path: Path | None
    corridor_evidence_path: Path | None
    offscreen_evidence_path: Path | None
    authority: str

    def directed_paths(
        self, *, source_height_ue_cm: float
    ) -> tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ]:
        result = []
        for corridor in self.corridors:
            start = (*corridor.start_ue_cm, source_height_ue_cm)
            end = (*corridor.end_ue_cm, source_height_ue_cm)
            result.extend(((start, end), (end, start)))
        return tuple(result)

    def stand_points(self, *, source_height_ue_cm: float) -> tuple[tuple[float, ...], ...]:
        return tuple(
            sorted(
                {
                    (*corridor.start_ue_cm, source_height_ue_cm)
                    for corridor in self.corridors
                }
                | {
                    (*corridor.end_ue_cm, source_height_ue_cm)
                    for corridor in self.corridors
                }
            )
        )

    def side(self, point_ue_cm: Sequence[float]) -> tuple[str, float]:
        if len(point_ue_cm) < 2:
            raise DesignViewpointError("side point must contain UE X/Y")
        theta = math.radians(self.camera_yaw_ue_deg)
        forward_x, forward_y = math.cos(theta), math.sin(theta)
        delta_x = float(point_ue_cm[0]) - self.camera_position_ue_cm[0]
        delta_y = float(point_ue_cm[1]) - self.camera_position_ue_cm[1]
        cross = (forward_x * delta_y - forward_y * delta_x) / max(
            math.hypot(delta_x, delta_y), 1.0e-9
        )
        return ("right" if cross > 0.0 else "left"), abs(cross)

    def spec_m1_request(self) -> dict[str, str]:
        return {
            "request_id": self.m1_request_id,
            "path": str(self.m1_request_path),
        }


def _load_json(path: Path, *, owner: str) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DesignViewpointError(f"cannot load {owner}: {path}: {exc}") from exc


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DesignViewpointError(f"{owner} must be an object")
    return value


def _nonempty_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DesignViewpointError(f"{owner} must be a non-empty string")
    return value


def _finite_vector(value: Any, size: int, *, owner: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DesignViewpointError(f"{owner} must contain {size} finite numbers")
    if len(value) != size:
        raise DesignViewpointError(f"{owner} must contain {size} finite numbers")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise DesignViewpointError(f"{owner} must contain {size} finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise DesignViewpointError(f"{owner} must contain {size} finite numbers")
        result.append(number)
    return tuple(result)


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignViewpointError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DesignViewpointError(f"{owner} must be a finite number")
    return result


def _resolve_file(base: Path, value: Any, *, owner: str) -> tuple[str, Path]:
    reference = _nonempty_string(value, owner=owner)
    unresolved = Path(reference).expanduser()
    resolved = unresolved if unresolved.is_absolute() else base / unresolved
    try:
        resolved = resolved.resolve(strict=True)
    except OSError as exc:
        raise DesignViewpointError(f"{owner} does not resolve to a file: {reference}") from exc
    if not resolved.is_file():
        raise DesignViewpointError(f"{owner} does not resolve to a file: {reference}")
    return reference, resolved


def _pass_evidence(
    value: Any, *, bank_parent: Path, owner: str
) -> tuple[Mapping[str, Any], Path]:
    record = _mapping(value, owner=owner)
    if record.get("status") != "pass":
        raise DesignViewpointError(f"{owner}.status must be 'pass'")
    _reference, evidence = _resolve_file(
        bank_parent,
        record.get("evidence_path"),
        owner=f"{owner}.evidence_path",
    )
    return record, evidence


def _yaw_difference_degrees(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _camera_from_m1_request(
    request_path: Path,
) -> tuple[Mapping[str, Any], tuple[float, float, float], float]:
    request = _load_json(request_path, owner="M1 request")
    if not isinstance(request, dict):
        raise DesignViewpointError(f"M1 request must be an object: {request_path}")
    errors = validate_capture_request(request)
    if errors:
        raise DesignViewpointError(
            f"M1 request is invalid: {request_path}: " + "; ".join(errors)
        )
    rig = request["primary_camera_rig"]
    transform = rig["world_from_rig"]
    try:
        ue_position = habitat_point_to_apartment_ue_cm(transform["translation_m"])
        habitat_yaw = habitat_yaw_degrees_from_xyzw(transform["rotation_xyzw"])
        ue_yaw = camera_ue_yaw_degrees(habitat_yaw)
    except (KeyError, TypeError, ValueError) as exc:
        raise DesignViewpointError(
            f"M1 request camera cannot be converted to Apartment UE coordinates: {request_path}"
        ) from exc
    return request, tuple(float(item) for item in ue_position), float(ue_yaw)


def _corridor_from_record(
    record: Mapping[str, Any], *, owner: str, default_id: str
) -> DesignCorridor:
    corridor_id = record.get("corridor_id", default_id)
    corridor_id = _nonempty_string(corridor_id, owner=f"{owner}.corridor_id")
    coordinate_space = record.get("coordinate_space", "apartment_ue_cm")
    if "start_ue_cm" in record or "end_ue_cm" in record:
        start = _finite_vector(record.get("start_ue_cm"), 2, owner=f"{owner}.start_ue_cm")
        end = _finite_vector(record.get("end_ue_cm"), 2, owner=f"{owner}.end_ue_cm")
    elif "start_xz_m" in record or "end_xz_m" in record:
        start_m = _finite_vector(record.get("start_xz_m"), 2, owner=f"{owner}.start_xz_m")
        end_m = _finite_vector(record.get("end_xz_m"), 2, owner=f"{owner}.end_xz_m")
        start = (100.0 * start_m[0], 100.0 * start_m[1])
        end = (100.0 * end_m[0], 100.0 * end_m[1])
    else:
        start_raw = _finite_vector(record.get("start"), 2, owner=f"{owner}.start")
        end_raw = _finite_vector(record.get("end"), 2, owner=f"{owner}.end")
        if coordinate_space == "apartment_ue_cm":
            start, end = start_raw, end_raw
        elif coordinate_space == "habitat_xz_m":
            start = (100.0 * start_raw[0], 100.0 * start_raw[1])
            end = (100.0 * end_raw[0], 100.0 * end_raw[1])
        else:
            raise DesignViewpointError(
                f"{owner}.coordinate_space must be apartment_ue_cm or habitat_xz_m"
            )
    if start == end:
        raise DesignViewpointError(f"{owner} endpoints must differ")
    return DesignCorridor(corridor_id, start, end)


def _corridor_catalog(document: Mapping[str, Any]) -> dict[str, DesignCorridor]:
    raw = document.get("corridors", [])
    if not isinstance(raw, list):
        raise DesignViewpointError("viewpoint bank corridors must be a list")
    result: dict[str, DesignCorridor] = {}
    for index, value in enumerate(raw):
        record = _mapping(value, owner=f"corridors[{index}]")
        corridor = _corridor_from_record(
            record,
            owner=f"corridors[{index}]",
            default_id=f"bank_corridor_{index:04d}",
        )
        if corridor.corridor_id in result:
            raise DesignViewpointError(
                f"duplicate corridor_id in bank: {corridor.corridor_id}"
            )
        result[corridor.corridor_id] = corridor
    return result


def _viewpoint_corridors(
    subset: Mapping[str, Any],
    *,
    catalog: Mapping[str, DesignCorridor],
    owner: str,
) -> tuple[DesignCorridor, ...]:
    segments = subset.get("segments")
    corridor_ids = subset.get("corridor_ids")
    if segments is not None and corridor_ids is not None:
        raise DesignViewpointError(
            f"{owner} must use either segments or corridor_ids, not both"
        )
    if segments is not None:
        if not isinstance(segments, list) or not segments:
            raise DesignViewpointError(f"{owner}.segments must be a non-empty list")
        return _deduplicate_corridors(
            _corridor_from_record(
                _mapping(value, owner=f"{owner}.segments[{index}]"),
                owner=f"{owner}.segments[{index}]",
                default_id=f"view_corridor_{index:04d}",
            )
            for index, value in enumerate(segments)
        )
    if not isinstance(corridor_ids, list) or not corridor_ids:
        raise DesignViewpointError(
            f"{owner} must contain non-empty segments or corridor_ids"
        )
    result = []
    for index, value in enumerate(corridor_ids):
        corridor_id = _nonempty_string(
            value, owner=f"{owner}.corridor_ids[{index}]"
        )
        try:
            result.append(catalog[corridor_id])
        except KeyError as exc:
            raise DesignViewpointError(
                f"{owner} references unknown corridor_id: {corridor_id}"
            ) from exc
    return _deduplicate_corridors(result)


def _deduplicate_corridors(
    corridors: Sequence[DesignCorridor],
) -> tuple[DesignCorridor, ...]:
    """Keep the first ID/orientation for each exact undirected geometry."""

    result = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for corridor in corridors:
        key = tuple(sorted((corridor.start_ue_cm, corridor.end_ue_cm)))
        if key in seen:
            continue
        seen.add(key)
        result.append(corridor)
    return tuple(result)


def _reject_pending_prefilter(document: Mapping[str, Any]) -> None:
    if document.get("native_pixel_validation_status") == "pending" or document.get(
        "artifact_status"
    ) == "deterministic_2d_raster_prefilter_only":
        raise DesignViewpointError(
            "native_pixel_validation_status=pending Apartment prefilter cannot be used "
            "for production design"
        )
    viewpoints = document.get("viewpoints")
    if isinstance(viewpoints, list) and any(
        isinstance(item, Mapping)
        and item.get("native_pixel_validation_status") == "pending"
        for item in viewpoints
    ):
        raise DesignViewpointError(
            "native_pixel_validation_status=pending Apartment prefilter cannot be used "
            "for production design"
        )


def load_design_viewpoint_bank(path: str | Path) -> tuple[DesignViewpoint, ...]:
    """Load a declared-pass 4--8-view bank and its design-time references."""

    bank_path = Path(path).expanduser().resolve(strict=True)
    document = _load_json(bank_path, owner="viewpoint bank")
    document = _mapping(document, owner="viewpoint bank")
    _reject_pending_prefilter(document)
    raw_viewpoints = document.get("viewpoints")
    if not isinstance(raw_viewpoints, list):
        raise DesignViewpointError("viewpoint bank must contain a viewpoints list")
    if not MIN_VIEWPOINTS <= len(raw_viewpoints) <= MAX_VIEWPOINTS:
        raise DesignViewpointError("viewpoint bank must contain complete 4-8 viewpoints")
    catalog = _corridor_catalog(document)
    result = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_viewpoints):
        owner = f"viewpoints[{index}]"
        record = _mapping(raw, owner=owner)
        viewpoint_id = _nonempty_string(
            record.get("viewpoint_id"), owner=f"{owner}.viewpoint_id"
        )
        if viewpoint_id in seen_ids:
            raise DesignViewpointError(f"duplicate viewpoint_id in bank: {viewpoint_id}")
        seen_ids.add(viewpoint_id)
        if record.get("native_pixel_validation_status") != "pass":
            raise DesignViewpointError(
                f"viewpoint {viewpoint_id} native_pixel_validation_status must be 'pass'"
            )
        if record.get("native_validation_pending"):
            raise DesignViewpointError(
                f"viewpoint {viewpoint_id} still declares native validation pending"
            )
        _authentication, authentication_evidence = _pass_evidence(
            record.get("authentication"),
            bank_parent=bank_path.parent,
            owner=f"viewpoint {viewpoint_id} authentication",
        )
        m1_record = _mapping(
            record.get("m1_request"), owner=f"viewpoint {viewpoint_id} m1_request"
        )
        if m1_record.get("status") != "pass":
            raise DesignViewpointError(
                f"viewpoint {viewpoint_id} m1_request.status must be 'pass'"
            )
        m1_reference, m1_path = _resolve_file(
            bank_path.parent,
            m1_record.get("path"),
            owner=f"viewpoint {viewpoint_id} m1_request.path",
        )
        request, request_position, request_yaw = _camera_from_m1_request(m1_path)
        camera = _mapping(record.get("camera"), owner=f"viewpoint {viewpoint_id} camera")
        bank_position = _finite_vector(
            camera.get("ue_location_cm"),
            3,
            owner=f"viewpoint {viewpoint_id} camera.ue_location_cm",
        )
        bank_yaw = _finite_number(
            camera.get("ue_yaw_deg", camera.get("yaw")),
            owner=f"viewpoint {viewpoint_id} camera.ue_yaw_deg",
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=CAMERA_MATCH_TOLERANCE)
            for left, right in zip(bank_position, request_position)
        ) or _yaw_difference_degrees(bank_yaw, request_yaw) > CAMERA_MATCH_TOLERANCE:
            raise DesignViewpointError(
                f"viewpoint {viewpoint_id} camera does not match its M1 request"
            )
        subset, corridor_evidence = _pass_evidence(
            record.get("corridor_subset"),
            bank_parent=bank_path.parent,
            owner=f"viewpoint {viewpoint_id} corridor_subset",
        )
        corridors = _viewpoint_corridors(
            subset,
            catalog=catalog,
            owner=f"viewpoint {viewpoint_id} corridor_subset",
        )
        offscreen, offscreen_evidence = _pass_evidence(
            record.get("offscreen_evidence"),
            bank_parent=bank_path.parent,
            owner=f"viewpoint {viewpoint_id} offscreen_evidence",
        )
        offscreen_position = _finite_vector(
            offscreen.get("point_ue_cm"),
            3,
            owner=f"viewpoint {viewpoint_id} offscreen_evidence.point_ue_cm",
        )
        result.append(
            DesignViewpoint(
                viewpoint_id=viewpoint_id,
                camera_position_ue_cm=tuple(request_position),
                camera_yaw_ue_deg=request_yaw,
                m1_request_path=m1_path,
                m1_request_reference=m1_reference,
                m1_request_id=str(request["request_id"]),
                corridors=corridors,
                offscreen_position_ue_cm=tuple(offscreen_position),
                authentication_evidence_path=authentication_evidence,
                corridor_evidence_path=corridor_evidence,
                offscreen_evidence_path=offscreen_evidence,
                authority="declared_certified_viewpoint_input",
            )
        )
    return tuple(result)


def select_design_viewpoints(
    viewpoints: Sequence[DesignViewpoint], viewpoint_id: str | None
) -> tuple[DesignViewpoint, ...]:
    """Select one explicit ID or retain bank order for deterministic rotation."""

    if viewpoint_id is None:
        return tuple(viewpoints)
    matches = tuple(item for item in viewpoints if item.viewpoint_id == viewpoint_id)
    if not matches:
        raise DesignViewpointError(
            f"--viewpoint-id is absent from viewpoint bank: {viewpoint_id}"
        )
    return matches


def load_legacy_view0_report_only(
    *,
    m1_request_path: str | Path,
    corridor_bank_path: str | Path,
    offscreen_position_ue_cm: Sequence[float],
) -> DesignViewpoint:
    """Build the old view0 authority only for an explicit report-only run."""

    request_path = Path(m1_request_path).expanduser().resolve(strict=True)
    request, camera_position, camera_yaw = _camera_from_m1_request(request_path)
    corridor_path = Path(corridor_bank_path).expanduser().resolve(strict=True)
    corridor_document = _load_json(corridor_path, owner="legacy corridor bank")
    corridor_document = _mapping(corridor_document, owner="legacy corridor bank")
    raw_segments = corridor_document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise DesignViewpointError("legacy corridor bank must contain segments")
    corridors = _deduplicate_corridors(
        _corridor_from_record(
            _mapping(value, owner=f"legacy corridors[{index}]"),
            owner=f"legacy corridors[{index}]",
            default_id=f"legacy_corridor_{index:04d}",
        )
        for index, value in enumerate(raw_segments)
    )
    offscreen = _finite_vector(
        offscreen_position_ue_cm,
        3,
        owner="--legacy-offscreen-point-ue-cm",
    )
    return DesignViewpoint(
        viewpoint_id="apartment_view0",
        camera_position_ue_cm=camera_position,
        camera_yaw_ue_deg=camera_yaw,
        m1_request_path=request_path,
        m1_request_reference=str(m1_request_path),
        m1_request_id=str(request["request_id"]),
        corridors=corridors,
        offscreen_position_ue_cm=tuple(offscreen),
        authentication_evidence_path=None,
        corridor_evidence_path=None,
        offscreen_evidence_path=None,
        authority="legacy_view0_report_only",
    )


__all__ = [
    "DesignCorridor",
    "DesignViewpoint",
    "DesignViewpointError",
    "load_design_viewpoint_bank",
    "load_legacy_view0_report_only",
    "select_design_viewpoints",
]
