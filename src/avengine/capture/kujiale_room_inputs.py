"""Keep retained UE room admission separate from CPU floor planning.

The caller supplies the original admitted room from its retained runtime.
A CPU cache is optional planning input, never a replacement admission record.
This adapter copies records; it performs no rendering or geometry validation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from numbers import Real
from typing import Any

CPU_PLANNING_FLOOR_REFERENCE_KEY = "cpu_planning_floor_reference"
CPU_PLANNING_FLOOR_STATUS = "pass_cpu_native_LOD_requires_current_UE"
_NATIVE_MEASUREMENT_KEYS = (
    "room_id", "method", "probes", "native_ue_floor_offset_m", "native_floor_surface",
)
_PROBE_ROLES = {"camera", "source_1", "source_2"}


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _measured_probes(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    probes = reference.get("probes")
    if not isinstance(probes, list) or len(probes) != 9:
        raise ValueError("floor_reference requires exactly nine measured witness/role probes")
    identities = []
    for probe in probes:
        if not isinstance(probe, Mapping) or probe.get("status") != "measured":
            raise ValueError("floor_reference requires nine measured probes")
        index, role = probe.get("witness_index"), probe.get("role")
        if (type(index) is not int or index < 0 or not isinstance(role, str)
                or role not in _PROBE_ROLES):
            raise ValueError("floor_reference has an invalid witness/role probe")
        identities.append((index, role))
        point = probe.get("nav_point_m")
        if (not isinstance(point, (list, tuple)) or len(point) != 3
                or not all(_finite(value) for value in point)
                or not _finite(probe.get("native_visual_floor_y_m"))):
            raise ValueError("floor_reference probe coordinates must be finite")
    witnesses = {index for index, _ in identities}
    expected = {(index, role) for index in witnesses for role in _PROBE_ROLES}
    if len(witnesses) != 3 or set(identities) != expected:
        raise ValueError("floor_reference requires nine distinct witness/role probes")
    return probes


def validate_native_floor_reference(reference: Mapping[str, Any], *, room_id: str) -> None:
    """Require retained native admission and its nine actual depth measurements."""
    if not isinstance(reference, Mapping) or reference.get("status") != "pass":
        raise ValueError("admitted floor_reference must retain native UE status 'pass'")
    if reference.get("room_id") != room_id:
        raise ValueError("admitted floor_reference belongs to another room")
    if reference.get("current_UE_readback_status") not in (None, "pass"):
        raise ValueError("CPU or pending UE floor_reference cannot be native admission")
    for probe in _measured_probes(reference):
        pixels = probe.get("native_depth_valid_pixels")
        depth = probe.get("native_depth_m")
        if (type(pixels) is not int or pixels <= 0
                or not _finite(depth) or depth <= 0):
            raise ValueError("native floor admission requires measured UE depth at all nine probes")


def _admission_reference(room: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(room, Mapping) or room.get("native_qa_status") != "pass":
        raise ValueError("room must retain its passed native UE QA admission")
    room_id = room.get("room_label")
    if not isinstance(room_id, str) or not room_id:
        raise ValueError("admitted room requires a room_label")
    reference = room.get("floor_reference")
    validate_native_floor_reference(reference, room_id=room_id)
    return reference


def _same_native_measurements(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in _NATIVE_MEASUREMENT_KEYS)


def _check_planning_reference(reference: Mapping[str, Any], native: Mapping[str, Any]) -> None:
    if not isinstance(reference, Mapping) or reference.get("room_id") != native["room_id"]:
        raise ValueError("CPU planning floor_reference belongs to another room")
    retained = reference.get("original_reference")
    if retained is not None and (
            not isinstance(retained, Mapping) or not _same_native_measurements(retained, native)):
        raise ValueError("CPU planning reference does not retain this original native admission")
    status = reference.get("status")
    if status == "pass":
        # An unrepaired cache may repeat the original UE probes. Changed CPU
        # points cannot acquire UE admission by changing their status to pass.
        if not _same_native_measurements(reference, native):
            raise ValueError("CPU planning points cannot be promoted to native UE pass")
        validate_native_floor_reference(reference, room_id=native["room_id"])
    elif status == CPU_PLANNING_FLOOR_STATUS:
        _measured_probes(reference)
        if reference.get("current_UE_readback_status") not in (None, "not_run"):
            raise ValueError("CPU floor planning cannot assert a current UE readback")
    elif not isinstance(status, str) or not status:
        raise ValueError("CPU planning floor_reference requires its own explicit status")


def prepare_room_inputs(
    original_admitted_room: Mapping[str, Any], *,
    cpu_floor_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a native-admitted room and attach CPU planning under its own key.

    Do not pass a room whose ``floor_reference`` was already overwritten by a
    CPU cache. The original retained runtime remains admission authority.
    Fail/blocked CPU records remain separate facts and are not made ready.
    """
    native = _admission_reference(original_admitted_room)
    planning = cpu_floor_reference
    if planning is None:
        planning = original_admitted_room.get(CPU_PLANNING_FLOOR_REFERENCE_KEY)
    if planning is not None:
        _check_planning_reference(planning, native)
    result = deepcopy(dict(original_admitted_room))
    if planning is not None:
        # Two independent copies even when both inputs initially alias.
        result[CPU_PLANNING_FLOOR_REFERENCE_KEY] = deepcopy(dict(planning))
    return result


def admission_floor_reference(room: Mapping[str, Any]) -> dict[str, Any]:
    """Read the retained native reference; CPU planning never supplies it."""
    return deepcopy(dict(_admission_reference(room)))


def floor_reference_for_planning(room: Mapping[str, Any]) -> dict[str, Any]:
    """Read ready CPU candidates, or the original reference when no cache exists.

    The returned status is preserved. CPU success still requires a new native
    readback; this function never asserts UE admission for those candidate points.
    """
    native = _admission_reference(room)
    planning = room.get(CPU_PLANNING_FLOOR_REFERENCE_KEY)
    if planning is None:
        return deepcopy(dict(native))
    _check_planning_reference(planning, native)
    if planning["status"] not in ("pass", CPU_PLANNING_FLOOR_STATUS):
        raise ValueError("CPU planning floor_reference is not ready: " + planning["status"])
    return deepcopy(dict(planning))
