"""Stage 1 decisions: known rejection, qualified candidate, or missing evidence."""

from __future__ import annotations

import shapely

from .geometry import short_side


def decide(metrics, room_type, p):
    failures = []
    unknown = []
    if room_type in p["excluded_types"]:
        failures.append("NON_RESIDENTIAL_OR_TRANSIT")
    for field, threshold, operator, reason in [
        ("floor_area_m2", p["floor_area_min_m2"], "min", "FLOOR_AREA_SMALL"),
        ("short_side_m", p["short_side_min_m"], "min", "SHORT_SIDE_SMALL"),
        (
            "nav_main_area_m2",
            (
                p["bathroom_nav_main_area_min_m2"]
                if room_type == "bathroom"
                else p["nav_main_area_min_m2"]
            ),
            "min",
            "NAV_MAIN_AREA_SMALL",
        ),
        ("black_fraction", p["black_fraction_max"], "max", "SCAN_BLACK_HIGH"),
    ]:
        value = metrics.get(field)
        if value is None:
            unknown.append(field.upper() + "_UNAVAILABLE")
        elif (operator == "min" and value < threshold) or (
            operator == "max" and value > threshold
        ):
            failures.append(reason)
    witness = metrics.get("placement")
    if witness is None:
        unknown.append("PLACEMENT_UNAVAILABLE")
    elif not witness["found"]:
        failures.append("PLACEMENT_NO_WITNESS")
    if failures:
        status = "fail"
    elif unknown:
        status = "review"
    else:
        status = "pass"
    return dict(
        status=status,
        reason_codes=failures + unknown,
        known_failures=failures,
        missing_evidence=unknown,
    )


def scope_metrics(scope, furniture):
    furniture_union = shapely.union_all([f["geometry"] for f in furniture])
    occupied = scope.intersection(furniture_union)
    return dict(
        floor_area_m2=float(scope.area),
        short_side_m=short_side(scope),
        convexity=(
            float(scope.area / scope.convex_hull.area) if scope.convex_hull.area else 0
        ),
        furniture_footprint_m2=float(occupied.area),
        floor_minus_known_furniture_m2=float(scope.difference(occupied).area),
        floor_polygon=shapely.geometry.mapping(scope),
        area_method="horizontal semantic triangle projection union; millimetre precision; not exact architectural net floor area",
    )
