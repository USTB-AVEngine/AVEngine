"""Prevent CPU cache restoration from replacing prior UE floor admission."""
from copy import deepcopy
import json

import pytest

from avengine.capture.kujiale_room_inputs import (
    CPU_PLANNING_FLOOR_REFERENCE_KEY,
    CPU_PLANNING_FLOOR_STATUS,
    admission_floor_reference,
    floor_reference_for_planning,
    prepare_room_inputs,
)

pytestmark = pytest.mark.fast_unit


def admitted_room():
    probes = [
        {
            "witness_index": witness,
            "role": role,
            "status": "measured",
            "nav_point_m": [float(witness) if role == "camera" else 0.0, 0.01, float(index)],
            "native_visual_floor_y_m": 0.00005,
            "native_depth_m": 0.18995,
            "native_depth_valid_pixels": 81,
            "capsule": {"status": "pass", "bottom_floor_gap_m": 0.0},
        }
        for witness in range(3)
        for index, role in enumerate(("camera", "source_1", "source_2"))
    ]
    return {
        "room_label": "R0",
        "native_qa_status": "pass",
        "native_witnesses": [{"index": i, "placement": {"found": True}} for i in range(3)],
        "placement": {"found": True},
        "floor_reference": {
            "room_id": "R0",
            "status": "pass",
            "method": "UE native depth at nine frozen witness support points",
            "native_ue_floor_offset_m": -0.00995,
            "native_floor_surface": {"native_editor_audit": "retained_native_audit.json"},
            "probes": probes,
        },
    }


def cpu_reference(room):
    original = deepcopy(room["floor_reference"])
    reference = deepcopy(original)
    reference.update(
        status=CPU_PLANNING_FLOOR_STATUS,
        method="CPU Native LOD ray at new planning points; current UE readback pending",
        current_UE_readback_status="not_run",
        original_reference=original,
        original_nine_probe_identity_failures=[{"witness_index": 0, "role": "source_2"}],
    )
    reference["probes"] = [
        {
            "witness_index": probe["witness_index"],
            "role": probe["role"],
            "status": "measured",
            "nav_point_m": [probe["nav_point_m"][0] + 10, 0.01, probe["nav_point_m"][2]],
            "native_visual_floor_y_m": 0.00005,
            "actual_native_LOD_floor_y_m": 0.0,
            "measurement_source": "CPU ray, not a new native UE camera depth",
        }
        for probe in original["probes"]
    ]
    return reference


def test_repaired_cpu_cache_preserves_native_admission_and_old_diagnostics():
    original = admitted_room()
    cpu = cpu_reference(original)
    result = prepare_room_inputs(original, cpu_floor_reference=cpu)
    assert result["floor_reference"] == original["floor_reference"]
    assert result["native_qa_status"] == original["native_qa_status"]
    assert result["native_witnesses"] == original["native_witnesses"]
    assert result[CPU_PLANNING_FLOOR_REFERENCE_KEY] == cpu
    assert admission_floor_reference(result)["status"] == "pass"
    planning = floor_reference_for_planning(result)
    assert planning["status"] == CPU_PLANNING_FLOOR_STATUS
    assert planning["current_UE_readback_status"] == "not_run"
    assert planning["original_nine_probe_identity_failures"] == cpu["original_nine_probe_identity_failures"]
    assert planning["probes"] != result["floor_reference"]["probes"]


def test_inputs_and_both_reference_views_do_not_share_mutable_records():
    original = admitted_room()
    snapshot = deepcopy(original)
    result = prepare_room_inputs(original, cpu_floor_reference=original["floor_reference"])
    result[CPU_PLANNING_FLOOR_REFERENCE_KEY]["probes"][0]["nav_point_m"][0] = 99.0
    assert result["floor_reference"] == snapshot["floor_reference"]
    assert original == snapshot
    native_view = admission_floor_reference(result)
    native_view["probes"][0]["capsule"]["status"] = "fail"
    assert result["floor_reference"] == snapshot["floor_reference"]


def test_fixed_source_coordinates_across_witnesses_are_valid():
    original = admitted_room()
    probes = original["floor_reference"]["probes"]
    assert len({tuple(probe["nav_point_m"]) for probe in probes}) < 9
    result = prepare_room_inputs(original)
    assert admission_floor_reference(result)["probes"] == probes


def test_retained_nonconsecutive_witness_ids_are_not_renumbered():
    original = admitted_room()
    retained_ids = (3, 4, 7)
    for probe in original["floor_reference"]["probes"]:
        probe["witness_index"] = retained_ids[probe["witness_index"]]
    result = prepare_room_inputs(original)
    assert admission_floor_reference(result)["probes"] == original["floor_reference"]["probes"]
    assert {p["witness_index"] for p in result["floor_reference"]["probes"]} == set(retained_ids)


@pytest.mark.parametrize("case", ["four_witnesses", "negative_witness", "invalid_role"])
def test_nine_measured_probes_still_require_three_complete_witnesses(case):
    original = admitted_room()
    probe = original["floor_reference"]["probes"][-1]
    if case == "four_witnesses":
        probe["witness_index"] = 99
    elif case == "negative_witness":
        probe["witness_index"] = -1
    else:
        probe["role"] = []
    with pytest.raises(ValueError):
        prepare_room_inputs(original)


def test_existing_separated_cpu_planning_survives_a_second_input_copy():
    original = admitted_room()
    result = prepare_room_inputs(original, cpu_floor_reference=cpu_reference(original))
    copied = prepare_room_inputs(result)
    assert copied == result and copied is not result
    assert floor_reference_for_planning(copied)["status"] == CPU_PLANNING_FLOOR_STATUS


def test_no_cpu_cache_retains_the_original_native_reference():
    original = admitted_room()
    result = prepare_room_inputs(original)
    assert result == original and result is not original
    assert CPU_PLANNING_FLOOR_REFERENCE_KEY not in result
    assert floor_reference_for_planning(result) == original["floor_reference"]


@pytest.mark.parametrize("status", ["fail", "blocked", "not_run", "qualified", CPU_PLANNING_FLOOR_STATUS])
def test_cpu_success_cannot_upgrade_failed_native_admission(status):
    original = admitted_room()
    cpu = cpu_reference(original)
    original["floor_reference"]["status"] = status
    with pytest.raises(ValueError, match="native UE status"):
        prepare_room_inputs(original, cpu_floor_reference=cpu)


def test_cpu_reference_cannot_replace_native_admission_even_after_status_rewrite():
    original = admitted_room()
    cpu = cpu_reference(original)
    overwritten = deepcopy(original)
    overwritten["floor_reference"] = deepcopy(cpu)
    with pytest.raises(ValueError, match="native UE status"):
        prepare_room_inputs(overwritten)
    overwritten["floor_reference"]["status"] = "pass"
    with pytest.raises(ValueError, match="cannot be native admission"):
        prepare_room_inputs(overwritten)
    cpu["status"] = "pass"
    with pytest.raises(ValueError, match="cannot be promoted"):
        prepare_room_inputs(original, cpu_floor_reference=cpu)


def test_changed_cpu_points_with_retained_depth_fields_still_cannot_claim_ue_pass():
    original = admitted_room()
    cpu = deepcopy(original["floor_reference"])
    cpu["probes"][0]["nav_point_m"][0] += 10
    with pytest.raises(ValueError, match="cannot be promoted"):
        prepare_room_inputs(original, cpu_floor_reference=cpu)


@pytest.mark.parametrize("case", ["missing", "duplicate", "unmeasured", "invalid_depth", "no_pixels", "nonfinite"])
def test_native_admission_requires_nine_distinct_measured_depth_probes(case):
    room = admitted_room()
    probes = room["floor_reference"]["probes"]
    if case == "missing":
        probes.pop()
    elif case == "duplicate":
        probes[-1] = deepcopy(probes[0])
    elif case == "unmeasured":
        probes[0]["status"] = "not_run"
    elif case == "invalid_depth":
        probes[0]["native_depth_m"] = float("nan")
    elif case == "no_pixels":
        probes[0]["native_depth_valid_pixels"] = 0
    else:
        probes[0]["nav_point_m"][0] = float("inf")
    with pytest.raises(ValueError):
        prepare_room_inputs(room)


def test_original_room_qa_must_pass_without_cpu_promotion():
    room = admitted_room()
    cpu = cpu_reference(room)
    room["native_qa_status"] = "fail"
    with pytest.raises(ValueError, match="native UE QA"):
        prepare_room_inputs(room, cpu_floor_reference=cpu)


def test_cpu_reference_from_another_room_or_original_is_rejected():
    room = admitted_room()
    cpu = cpu_reference(room)
    cpu["room_id"] = "R1"
    with pytest.raises(ValueError, match="another room"):
        prepare_room_inputs(room, cpu_floor_reference=cpu)
    cpu = cpu_reference(room)
    cpu["original_reference"]["native_floor_surface"]["native_editor_audit"] = "another_house.json"
    with pytest.raises(ValueError, match="this original native admission"):
        prepare_room_inputs(room, cpu_floor_reference=cpu)


def test_blocked_cpu_planning_is_retained_without_erasing_native_admission():
    room = admitted_room()
    cpu = {"room_id": "R0", "status": "blocked", "reason": "no actual room-floor support"}
    result = prepare_room_inputs(room, cpu_floor_reference=cpu)
    assert admission_floor_reference(result) == room["floor_reference"]
    assert result[CPU_PLANNING_FLOOR_REFERENCE_KEY] == cpu
    with pytest.raises(ValueError, match="not ready: blocked"):
        floor_reference_for_planning(result)


def test_original_training_room_gate_accepts_separated_cpu_cache(tmp_path):
    from avengine.timeline.kujiale_asset_pilot import validate_training_room

    original = admitted_room()
    house = {
        "house": "kujiale_example", "split": "train", "family": "kujiale",
        "acoustic_status": "pass", "checked_room_ids": ["R0"], "checked_rooms": [original],
    }
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "runtime.json").write_text(json.dumps({"houses": [house]}))
    derived = deepcopy(house)
    derived["checked_rooms"][0] = prepare_room_inputs(original, cpu_floor_reference=cpu_reference(original))
    row = {"house": house["house"], "room_id": "R0", "split": "train"}
    selected = validate_training_room(tmp_path, row, [derived])
    assert selected["checked_rooms"][0]["floor_reference"] == original["floor_reference"]
    assert selected["checked_rooms"][0][CPU_PLANNING_FLOOR_REFERENCE_KEY]["status"] == CPU_PLANNING_FLOOR_STATUS
    derived["checked_rooms"][0]["floor_reference"] = cpu_reference(original)
    with pytest.raises(ValueError, match="passed native UE floor/QA admission"):
        validate_training_room(tmp_path, row, [derived])
