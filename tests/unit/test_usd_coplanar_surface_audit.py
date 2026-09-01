from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/rooms/audit_usd_coplanar_surfaces.py"
SPEC = importlib.util.spec_from_file_location("coplanar_surface_audit", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _face(
    mesh: str,
    points: tuple[tuple[float, float, float], ...],
) -> object:
    normal, area = TOOL.polygon_normal_and_area(points)
    return TOOL.SurfaceFace(
        mesh=mesh,
        face_index=0,
        material=f"/{mesh}_material",
        points=points,
        normal=normal,
        area_m2=area,
    )


def test_detects_same_facing_coplanar_overlap() -> None:
    wall = _face(
        "wall",
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
    )
    panel = _face(
        "panel",
        ((1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 2.0, 0.0), (1.0, 2.0, 0.0)),
    )

    overlaps = TOOL.detect_same_facing_coplanar_overlaps([wall, panel])

    assert len(overlaps) == 1
    assert overlaps[0]["overlap_area_m2"] == 2.0
    assert overlaps[0]["overlap_ratio"] == 0.5


def test_does_not_confuse_opposite_facing_contact_with_z_fighting() -> None:
    upper = _face(
        "upper",
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
    )
    lower = _face(
        "lower",
        ((0.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )

    assert TOOL.detect_same_facing_coplanar_overlaps([upper, lower]) == []


def test_clearance_larger_than_tolerance_is_safe() -> None:
    wall = _face(
        "wall",
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
    )
    panel = _face(
        "panel",
        ((0.0, 0.0, 0.01), (2.0, 0.0, 0.01), (2.0, 2.0, 0.01), (0.0, 2.0, 0.01)),
    )

    assert TOOL.detect_same_facing_coplanar_overlaps([wall, panel]) == []


def test_downward_hidden_interfaces_are_ignored() -> None:
    first = _face(
        "first",
        ((0.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    second = _face(
        "second",
        ((1.0, 2.0, 0.0), (3.0, 2.0, 0.0), (3.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )

    assert TOOL.detect_same_facing_coplanar_overlaps([first, second]) == []
    assert len(
        TOOL.detect_same_facing_coplanar_overlaps(
            [first, second], ignore_downward_faces=False
        )
    ) == 1
