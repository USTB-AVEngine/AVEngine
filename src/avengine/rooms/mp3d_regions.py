"""Static Matterport3D .house floor polygons and region membership.

The parser is an ordinary room input helper. It only describes the dataset's
declared floor geometry; it does not decide camera legality, actor visibility,
audio or formal dataset admission. This is an AVEngine adaptation of the
retained region slice introduced by 93dc679, moved into the current rooms owner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_GEOMETRIC_EPSILON_M = 1.0e-9

# MP3D source coordinates are x/right, y/front, z/up. Habitat uses x/right,
# y/up, z/back, so [x, y, z] becomes [x, z, -y].
MP3D_Z_UP_Y_FRONT_TO_HABITAT_MATRIX_ROW_MAJOR = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
)

MP3D_REGION_CATEGORY_NAMES: Mapping[str, str] = {
    "a": "bathroom",
    "b": "bedroom",
    "c": "closet",
    "d": "dining room",
    "e": "entryway/foyer/lobby",
    "f": "familyroom/lounge",
    "g": "garage",
    "h": "hallway",
    "i": "library",
    "j": "laundryroom/mudroom",
    "k": "kitchen",
    "l": "living room",
    "m": "meetingroom/conferenceroom",
    "n": "lounge",
    "o": "office",
    "p": "porch/terrace/deck",
    "r": "rec/game",
    "s": "stairs",
    "t": "toilet",
    "u": "utilityroom/toolroom",
    "v": "tv",
    "w": "workout/gym/exercise",
    "x": "outdoor",
    "y": "balcony",
    "z": "other room",
    "B": "bar",
    "C": "classroom",
    "D": "dining booth",
    "S": "spa/sauna",
    "Z": "junk",
}


class MP3DRegionError(ValueError):
    """A house file or floor point cannot support region binding."""


def _finite_float(value: object, *, owner: str) -> float:
    if isinstance(value, bool):
        raise MP3DRegionError(f"{owner} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionError(f"{owner} must be a finite number") from exc
    if not math.isfinite(result):
        raise MP3DRegionError(f"{owner} must be a finite number")
    return result


def _integer(value: object, *, owner: str) -> int:
    if isinstance(value, bool):
        raise MP3DRegionError(f"{owner} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionError(f"{owner} must be an integer") from exc
    if str(value).strip() != str(result):
        raise MP3DRegionError(f"{owner} must be an integer")
    return result


def _point3(value: object, *, owner: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise MP3DRegionError(f"{owner} must contain three finite numbers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise MP3DRegionError(
            f"{owner} must contain three finite numbers"
        ) from exc
    if len(values) != 3:
        raise MP3DRegionError(f"{owner} must contain three finite numbers")
    return tuple(
        _finite_float(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(values)
    )  # type: ignore[return-value]


def mp3d_z_up_y_front_to_habitat(
    point_xyz_m: Sequence[Real],
) -> tuple[float, float, float]:
    """Apply the registered MP3D source-to-Habitat point transform."""

    x, y, z = _point3(point_xyz_m, owner="MP3D source point")
    return (x, z, -y)


@dataclass(frozen=True)
class MP3DFloorPolygon:
    surface_index: int
    region_index: int
    vertices_habitat_xyz_m: tuple[tuple[float, float, float], ...]

    @property
    def horizontal_xz_m(self) -> tuple[tuple[float, float], ...]:
        return tuple((point[0], point[2]) for point in self.vertices_habitat_xyz_m)

    @property
    def floor_elevation_m(self) -> float:
        return sum(point[1] for point in self.vertices_habitat_xyz_m) / len(
            self.vertices_habitat_xyz_m
        )

    @property
    def floor_elevation_span_m(self) -> float:
        elevations = tuple(point[1] for point in self.vertices_habitat_xyz_m)
        return max(elevations) - min(elevations)


@dataclass(frozen=True)
class MP3DRegion:
    region_index: int
    region_instance_id: str
    level_index: int
    category_code: str
    category_name: str | None
    habitat_bbox_min_xyz_m: tuple[float, float, float]
    habitat_bbox_max_xyz_m: tuple[float, float, float]
    floor_polygons: tuple[MP3DFloorPolygon, ...]

    def label_record(self) -> dict[str, object]:
        return {
            "region_index": self.region_index,
            "region_instance_id": self.region_instance_id,
            "level_index": self.level_index,
            "category_code": self.category_code,
            "category_name": self.category_name,
        }

    def contains(
        self,
        point_m: Sequence[Real],
        *,
        y_tolerance_m: float = 0.30,
    ) -> bool:
        point = _point3(point_m, owner="region point")
        tolerance = _finite_float(y_tolerance_m, owner="region y tolerance")
        if tolerance < 0.0:
            raise MP3DRegionError("region y tolerance must be nonnegative")
        if not (
            self.habitat_bbox_min_xyz_m[1] - tolerance
            <= point[1]
            <= self.habitat_bbox_max_xyz_m[1] + tolerance
        ):
            return False
        horizontal = (point[0], point[2])
        return any(
            abs(point[1] - polygon.floor_elevation_m) <= tolerance
            and _polygon_relation(horizontal, polygon.horizontal_xz_m)[0] != "outside"
            for polygon in self.floor_polygons
        )


@dataclass(frozen=True)
class MP3DHouseFloorPlan:
    house_id: str
    house_name: str
    house_label: str
    declared_counts: Mapping[str, int]
    parsed_portal_count: int
    parsed_panorama_count: int
    regions: tuple[MP3DRegion, ...]

    @property
    def by_region_index(self) -> dict[int, MP3DRegion]:
        return {region.region_index: region for region in self.regions}

    @property
    def by_region_instance_id(self) -> dict[str, MP3DRegion]:
        return {region.region_instance_id: region for region in self.regions}


@dataclass(frozen=True)
class FloorRegionMembership:
    floor_sample_index: int
    floor_position_m: tuple[float, float, float]
    membership_status: str
    member_region_indices: tuple[int, ...]
    member_boundary_distances_m: tuple[tuple[int, float], ...]
    nearest_polygon_boundary_distance_m: float
    on_polygon_boundary: bool
    near_boundary: bool

    def to_record(self, plan: MP3DHouseFloorPlan) -> dict[str, object]:
        by_index = plan.by_region_index
        members = []
        distance_by_index = dict(self.member_boundary_distances_m)
        for index in self.member_region_indices:
            region = by_index[index]
            item = region.label_record()
            item["distance_to_region_polygon_boundary_m"] = distance_by_index[index]
            item["floor_elevation_delta_m"] = min(
                abs(self.floor_position_m[1] - polygon.floor_elevation_m)
                for polygon in region.floor_polygons
            )
            members.append(item)
        primary = members[0] if len(members) == 1 else None
        return {
            "floor_sample_index": self.floor_sample_index,
            "floor_position_m": list(self.floor_position_m),
            "membership_status": self.membership_status,
            "member_region_indices": list(self.member_region_indices),
            "member_region_instance_ids": [
                by_index[index].region_instance_id
                for index in self.member_region_indices
            ],
            "member_regions": members,
            "primary_region_index": (
                primary["region_index"] if primary is not None else None
            ),
            "primary_region_instance_id": (
                primary["region_instance_id"] if primary is not None else None
            ),
            "distance_to_nearest_polygon_boundary_m": (
                self.nearest_polygon_boundary_distance_m
            ),
            "on_polygon_boundary": self.on_polygon_boundary,
            "near_boundary_descriptor": self.near_boundary,
        }


def _parse_bbox(
    tokens: Sequence[str],
    *,
    offset: int,
    owner: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if len(tokens) < offset + 6:
        raise MP3DRegionError(f"{owner} is truncated")
    minimum = tuple(
        _finite_float(tokens[offset + index], owner=f"{owner} min[{index}]")
        for index in range(3)
    )
    maximum = tuple(
        _finite_float(tokens[offset + 3 + index], owner=f"{owner} max[{index}]")
        for index in range(3)
    )
    if any(low > high for low, high in zip(minimum, maximum)):
        raise MP3DRegionError(f"{owner} minimum exceeds maximum")
    corners = [
        mp3d_z_up_y_front_to_habitat((x, y, z))
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    return (
        tuple(min(point[axis] for point in corners) for axis in range(3)),
        tuple(max(point[axis] for point in corners) for axis in range(3)),
    )  # type: ignore[return-value]


def _signed_area(polygon: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(polygon, (*polygon[1:], polygon[0]))
    )


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    epsilon: float,
) -> bool:
    return (
        abs(_orientation(first, second, point)) <= epsilon
        and min(first[0], second[0]) - epsilon <= point[0]
        <= max(first[0], second[0]) + epsilon
        and min(first[1], second[1]) - epsilon <= point[1]
        <= max(first[1], second[1]) + epsilon
    )


def _segments_intersect(
    first_a: tuple[float, float],
    first_b: tuple[float, float],
    second_a: tuple[float, float],
    second_b: tuple[float, float],
) -> bool:
    values = (
        _orientation(first_a, first_b, second_a),
        _orientation(first_a, first_b, second_b),
        _orientation(second_a, second_b, first_a),
        _orientation(second_a, second_b, first_b),
    )
    if values[0] * values[1] < 0.0 and values[2] * values[3] < 0.0:
        return True
    return (
        (
            abs(values[0]) <= _GEOMETRIC_EPSILON_M
            and _on_segment(
                second_a,
                first_a,
                first_b,
                epsilon=_GEOMETRIC_EPSILON_M,
            )
        )
        or (
            abs(values[1]) <= _GEOMETRIC_EPSILON_M
            and _on_segment(
                second_b,
                first_a,
                first_b,
                epsilon=_GEOMETRIC_EPSILON_M,
            )
        )
        or (
            abs(values[2]) <= _GEOMETRIC_EPSILON_M
            and _on_segment(
                first_a,
                second_a,
                second_b,
                epsilon=_GEOMETRIC_EPSILON_M,
            )
        )
        or (
            abs(values[3]) <= _GEOMETRIC_EPSILON_M
            and _on_segment(
                first_b,
                second_a,
                second_b,
                epsilon=_GEOMETRIC_EPSILON_M,
            )
        )
    )


def _validate_polygon(polygon: MP3DFloorPolygon) -> None:
    horizontal = polygon.horizontal_xz_m
    if len(horizontal) < 3 or len(set(horizontal)) != len(horizontal):
        raise MP3DRegionError(
            f"floor surface {polygon.surface_index} has a bad polygon vertex set"
        )
    if abs(_signed_area(horizontal)) <= _GEOMETRIC_EPSILON_M:
        raise MP3DRegionError(
            f"floor surface {polygon.surface_index} has zero area"
        )
    for first_index in range(len(horizontal)):
        first_next = (first_index + 1) % len(horizontal)
        for second_index in range(first_index + 1, len(horizontal)):
            second_next = (second_index + 1) % len(horizontal)
            if (
                first_index == second_index
                or first_next == second_index
                or second_next == first_index
            ):
                continue
            if _segments_intersect(
                horizontal[first_index],
                horizontal[first_next],
                horizontal[second_index],
                horizontal[second_next],
            ):
                raise MP3DRegionError(
                    f"floor surface {polygon.surface_index} self-intersects"
                )


def parse_mp3d_house(path: str | Path) -> MP3DHouseFloorPlan:
    """Parse H/L/R/S/V/P records from one ASCII 1.1 MP3D house file."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise MP3DRegionError(f"house input is not a file: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            header = next(handle).strip()
            records = [
                (line_number, line.split())
                for line_number, line in enumerate(handle, 2)
                if line.split()
            ]
    except StopIteration as exc:
        raise MP3DRegionError("house file is empty") from exc
    except UnicodeDecodeError as exc:
        raise MP3DRegionError("house file is not UTF-8") from exc

    if header != "ASCII 1.1":
        raise MP3DRegionError(f"unsupported MP3D house header: {header!r}")

    house_record: tuple[str, ...] | None = None
    levels: set[int] = set()
    raw_regions: dict[int, dict[str, object]] = {}
    surfaces: dict[int, tuple[int, str]] = {}
    vertices: dict[int, list[tuple[int, tuple[float, float, float]]]] = {}
    seen_vertex_ids: set[int] = set()
    panorama_count = 0
    portal_count = 0

    for line_number, tokens in records:
        kind = tokens[0]
        owner = f"house line {line_number} {kind}"
        if kind == "H":
            if house_record is not None:
                raise MP3DRegionError("house file contains multiple H records")
            if len(tokens) < 29:
                raise MP3DRegionError(f"{owner} is truncated")
            house_record = tuple(tokens)
        elif kind == "L":
            if len(tokens) < 18:
                raise MP3DRegionError(f"{owner} is truncated")
            index = _integer(tokens[1], owner=f"{owner} level_index")
            if index in levels:
                raise MP3DRegionError(f"duplicate level index {index}")
            levels.add(index)
            _point3(tokens[4:7], owner=f"{owner} position")
            _parse_bbox(tokens, offset=7, owner=f"{owner} bbox")
        elif kind == "R":
            if len(tokens) < 20:
                raise MP3DRegionError(f"{owner} is truncated")
            index = _integer(tokens[1], owner=f"{owner} region_index")
            if index in raw_regions:
                raise MP3DRegionError(f"duplicate region index {index}")
            level_index = _integer(tokens[2], owner=f"{owner} level_index")
            category_code = tokens[5]
            if len(category_code) != 1:
                raise MP3DRegionError(f"{owner} category code must be one character")
            _point3(tokens[6:9], owner=f"{owner} position")
            bbox_min, bbox_max = _parse_bbox(tokens, offset=9, owner=f"{owner} bbox")
            raw_regions[index] = {
                "level_index": level_index,
                "category_code": category_code,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            }
        elif kind == "S":
            if len(tokens) < 22:
                raise MP3DRegionError(f"{owner} is truncated")
            surface_index = _integer(tokens[1], owner=f"{owner} surface_index")
            if surface_index in surfaces:
                raise MP3DRegionError(f"duplicate surface index {surface_index}")
            region_index = _integer(tokens[2], owner=f"{owner} region_index")
            label = tokens[4]
            _point3(tokens[5:8], owner=f"{owner} position")
            _point3(tokens[8:11], owner=f"{owner} normal")
            _parse_bbox(tokens, offset=11, owner=f"{owner} bbox")
            surfaces[surface_index] = (region_index, label)
        elif kind == "V":
            if len(tokens) < 13:
                raise MP3DRegionError(f"{owner} is truncated")
            vertex_index = _integer(tokens[1], owner=f"{owner} vertex_index")
            if vertex_index in seen_vertex_ids:
                raise MP3DRegionError(f"duplicate vertex index {vertex_index}")
            seen_vertex_ids.add(vertex_index)
            surface_index = _integer(tokens[2], owner=f"{owner} surface_index")
            position = _point3(tokens[4:7], owner=f"{owner} position")
            _point3(tokens[7:10], owner=f"{owner} normal")
            vertices.setdefault(surface_index, []).append((vertex_index, position))
        elif kind == "P":
            if len(tokens) == 13:
                _integer(tokens[2], owner=f"{owner} panorama index")
                _integer(tokens[3], owner=f"{owner} region index")
                _point3(tokens[5:8], owner=f"{owner} position")
                panorama_count += 1
            elif len(tokens) == 15:
                _integer(tokens[1], owner=f"{owner} portal index")
                _integer(tokens[2], owner=f"{owner} region0 index")
                _integer(tokens[3], owner=f"{owner} region1 index")
                _parse_bbox(tokens, offset=5, owner=f"{owner} bbox")
                portal_count += 1
            else:
                raise MP3DRegionError(
                    f"{owner} is neither a panorama nor a portal record"
                )

    if house_record is None:
        raise MP3DRegionError("house file lacks one H record")
    count_names = (
        "images", "panoramas", "vertices", "surfaces", "segments",
        "objects", "categories", "regions", "portals", "levels",
    )
    declared_counts = {
        name: _integer(house_record[index], owner=f"H {name} count")
        for index, name in enumerate(count_names, start=3)
    }
    observed = {
        "panoramas": panorama_count,
        "vertices": len(seen_vertex_ids),
        "surfaces": len(surfaces),
        "regions": len(raw_regions),
        "portals": portal_count,
        "levels": len(levels),
    }
    for name, count in observed.items():
        if declared_counts[name] != count:
            raise MP3DRegionError(
                f"H declares {declared_counts[name]} {name}, parsed {count}"
            )

    for region_index, raw in raw_regions.items():
        level_index = int(raw["level_index"])
        if level_index >= 0 and level_index not in levels:
            raise MP3DRegionError(
                f"region {region_index} references missing level {level_index}"
            )
    for surface_index, (region_index, _label) in surfaces.items():
        if region_index not in raw_regions:
            raise MP3DRegionError(
                f"surface {surface_index} references missing region {region_index}"
            )
    for surface_index in vertices:
        if surface_index not in surfaces:
            raise MP3DRegionError(
                f"vertices reference missing surface {surface_index}"
            )

    polygons_by_region: dict[int, list[MP3DFloorPolygon]] = {
        index: [] for index in raw_regions
    }
    for surface_index, (region_index, label) in sorted(surfaces.items()):
        if label != "F":
            continue
        source_vertices = tuple(
            point for _index, point in sorted(vertices.get(surface_index, ()))
        )
        polygon = MP3DFloorPolygon(
            surface_index=surface_index,
            region_index=region_index,
            vertices_habitat_xyz_m=tuple(
                mp3d_z_up_y_front_to_habitat(point)
                for point in source_vertices
            ),
        )
        _validate_polygon(polygon)
        polygons_by_region[region_index].append(polygon)
    if not any(polygons_by_region.values()):
        raise MP3DRegionError("house file has no valid F floor polygons")

    regions = tuple(
        MP3DRegion(
            region_index=index,
            region_instance_id=f"{resolved.stem}:region:{index:03d}",
            level_index=int(raw["level_index"]),
            category_code=str(raw["category_code"]),
            category_name=MP3D_REGION_CATEGORY_NAMES.get(
                str(raw["category_code"])
            ),
            habitat_bbox_min_xyz_m=raw["bbox_min"],  # type: ignore[arg-type]
            habitat_bbox_max_xyz_m=raw["bbox_max"],  # type: ignore[arg-type]
            floor_polygons=tuple(polygons_by_region[index]),
        )
        for index, raw in sorted(raw_regions.items())
    )
    return MP3DHouseFloorPlan(
        house_id=resolved.stem,
        house_name=house_record[1],
        house_label=house_record[2],
        declared_counts=declared_counts,
        parsed_portal_count=portal_count,
        parsed_panorama_count=panorama_count,
        regions=regions,
    )


def _point_segment_distance(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    delta_x = second[0] - first[0]
    delta_y = second[1] - first[1]
    squared_length = delta_x * delta_x + delta_y * delta_y
    if squared_length <= 0.0:
        return math.dist(point, first)
    fraction = (
        (point[0] - first[0]) * delta_x
        + (point[1] - first[1]) * delta_y
    ) / squared_length
    fraction = max(0.0, min(1.0, fraction))
    closest = (first[0] + fraction * delta_x, first[1] + fraction * delta_y)
    return math.dist(point, closest)


def _polygon_relation(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> tuple[str, float]:
    distances = tuple(
        _point_segment_distance(point, first, second)
        for first, second in zip(polygon, (*polygon[1:], polygon[0]))
    )
    boundary_distance = min(distances)
    if boundary_distance <= _GEOMETRIC_EPSILON_M:
        return "boundary", boundary_distance
    inside = False
    for first, second in zip(polygon, (*polygon[1:], polygon[0])):
        if (first[1] > point[1]) == (second[1] > point[1]):
            continue
        crossing_x = first[0] + (point[1] - first[1]) * (
            second[0] - first[0]
        ) / (second[1] - first[1])
        if point[0] < crossing_x:
            inside = not inside
    return ("inside" if inside else "outside"), boundary_distance


def bind_floor_points_to_regions(
    floor_samples_m: Iterable[Sequence[Real]],
    plan: MP3DHouseFloorPlan,
    *,
    boundary_descriptor_threshold_m: Real = 0.05,
) -> tuple[FloorRegionMembership, ...]:
    """Annotate points without changing their navigation/camera legality."""

    if not isinstance(plan, MP3DHouseFloorPlan):
        raise MP3DRegionError("plan must be an MP3DHouseFloorPlan")
    threshold = _finite_float(
        boundary_descriptor_threshold_m,
        owner="boundary descriptor threshold",
    )
    if threshold < 0.0:
        raise MP3DRegionError("boundary descriptor threshold must be nonnegative")
    try:
        samples = tuple(
            _point3(point, owner=f"floor_samples_m[{index}]")
            for index, point in enumerate(floor_samples_m)
        )
    except TypeError as exc:
        raise MP3DRegionError("floor_samples_m must be iterable") from exc
    if not samples:
        raise MP3DRegionError("floor_samples_m must be nonempty")

    result: list[FloorRegionMembership] = []
    for floor_index, sample in enumerate(samples):
        horizontal = (sample[0], sample[2])
        member_relations: dict[int, str] = {}
        member_distances: dict[int, float] = {}
        all_distances: list[float] = []
        for region in plan.regions:
            vertical_match = (
                region.habitat_bbox_min_xyz_m[1] - _GEOMETRIC_EPSILON_M
                <= sample[1]
                <= region.habitat_bbox_max_xyz_m[1] + _GEOMETRIC_EPSILON_M
            )
            for polygon in region.floor_polygons:
                relation, distance = _polygon_relation(
                    horizontal, polygon.horizontal_xz_m
                )
                all_distances.append(distance)
                if not vertical_match or relation == "outside":
                    continue
                previous = member_relations.get(region.region_index)
                if previous != "boundary":
                    member_relations[region.region_index] = relation
                member_distances[region.region_index] = min(
                    member_distances.get(region.region_index, math.inf),
                    distance,
                )
        member_indices = tuple(sorted(member_relations))
        on_boundary = any(
            value == "boundary" for value in member_relations.values()
        )
        if not member_indices:
            status = "unknown"
        elif len(member_indices) > 1:
            status = "ambiguous"
        elif on_boundary:
            status = "boundary"
        else:
            status = "unique"
        nearest_distance = min(all_distances)
        result.append(
            FloorRegionMembership(
                floor_sample_index=floor_index,
                floor_position_m=sample,
                membership_status=status,
                member_region_indices=member_indices,
                member_boundary_distances_m=tuple(
                    sorted(member_distances.items())
                ),
                nearest_polygon_boundary_distance_m=nearest_distance,
                on_polygon_boundary=on_boundary,
                near_boundary=nearest_distance <= threshold,
            )
        )
    return tuple(result)


__all__ = [
    "FloorRegionMembership",
    "MP3DFloorPolygon",
    "MP3DHouseFloorPlan",
    "MP3DRegion",
    "MP3DRegionError",
    "MP3D_REGION_CATEGORY_NAMES",
    "MP3D_Z_UP_Y_FRONT_TO_HABITAT_MATRIX_ROW_MAJOR",
    "bind_floor_points_to_regions",
    "mp3d_z_up_y_front_to_habitat",
    "parse_mp3d_house",
]