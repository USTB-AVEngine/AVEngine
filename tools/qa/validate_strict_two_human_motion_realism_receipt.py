#!/usr/bin/env python3
"""Validate a strict two-human CPU motion-realism receipt, optionally by replay."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BUILDER_PATH = Path(__file__).with_name(
    "build_strict_two_human_motion_realism_receipt.py"
)
SPEC = importlib.util.spec_from_file_location(
    "motion_realism_receipt_builder", BUILDER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import builder: {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return (
        math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance
    )


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    materialization_root: Path | None = None,
    expected_status: str | None = None,
) -> None:
    _require(receipt.get("schema") == BUILDER.RECEIPT_SCHEMA, "receipt schema drift")
    status = receipt.get("status")
    _require(
        status
        in {
            "pass_motion_realism_release_gate",
            "reject_nonrelease_motion_realism_gate",
        },
        "receipt status is invalid",
    )
    if expected_status is not None:
        _require(
            status == expected_status,
            f"expected status {expected_status}, got {status}",
        )
    _require(
        receipt.get("frame_count") == BUILDER.FRAME_COUNT
        and receipt.get("frame_rate_hz") == int(BUILDER.FRAME_RATE_HZ)
        and receipt.get("formal_episode_count") == 0
        and receipt.get("qualification_claim") is False
        and receipt.get("gpu_used") is False
        and receipt.get("other_gate_results_recomputed") is False,
        "CPU/formal/scope boundary drift",
    )
    thresholds = receipt.get("threshold_contract")
    _require(isinstance(thresholds, Mapping), "threshold contract is missing")
    _require(
        thresholds.get("maximum_native_speed_relative_error")
        == BUILDER.MAX_NATIVE_SPEED_RELATIVE_ERROR
        and thresholds.get("maximum_canonical_cadence_relative_error")
        == BUILDER.MAX_CANONICAL_CADENCE_RELATIVE_ERROR
        and thresholds.get("maximum_planted_foot_slip_m_per_frame")
        == BUILDER.MAX_PLANTED_FOOT_SLIP_M_PER_FRAME,
        "hard threshold drift",
    )
    slots = receipt.get("moving_slots")
    _require(
        isinstance(slots, list) and bool(slots), "moving-slot receipts are missing"
    )
    rejected: list[Mapping[str, Any]] = []
    for item in slots:
        _require(isinstance(item, Mapping), "moving-slot receipt is not an object")
        slot_status = item.get("status")
        blockers = item.get("blockers")
        facts = item.get("native_rate_facts")
        _require(slot_status in {"pass", "reject"}, "moving-slot status is invalid")
        _require(isinstance(blockers, list), "moving-slot blocker list is missing")
        _require(isinstance(facts, Mapping), "native-rate facts are missing")
        native_intervals = int(facts["native_interval_count"])
        native_duration = native_intervals / BUILDER.FRAME_RATE_HZ
        path_length = float(facts["path_length_m"])
        output_span = float(facts["output_span_seconds"])
        phase_cycles = float(facts["native_phase_advance_cycles"])
        _require(
            _close(float(facts["native_duration_seconds"]), native_duration)
            and _close(
                float(facts["native_rate_average_root_speed_m_s"]),
                path_length / native_duration,
            )
            and _close(
                float(facts["observed_full75_average_root_speed_m_s"]),
                path_length / output_span,
                1.0e-6,
            )
            and _close(
                float(facts["global_time_stretch_factor"]),
                output_span / native_duration,
            )
            and _close(
                float(facts["native_rate_phase_cycles_per_second"]),
                phase_cycles / native_duration,
            )
            and _close(
                float(facts["observed_full75_phase_cycles_per_second"]),
                phase_cycles / output_span,
            ),
            "derived native-rate arithmetic drift",
        )
        first = item.get("first_blocker")
        if slot_status == "pass":
            _require(not blockers and first is None, "pass slot contains blockers")
        else:
            _require(
                bool(blockers) and first == blockers[0],
                "reject slot first blocker is not deterministic",
            )
            rejected.append(item)

    first_blocker = receipt.get("first_blocker")
    if rejected:
        expected_first = {
            "slot_id": rejected[0]["slot_id"],
            **rejected[0]["first_blocker"],
        }
        _require(
            status == "reject_nonrelease_motion_realism_gate"
            and receipt.get("release_classification")
            == "nonrelease_pipeline_evidence_only"
            and first_blocker == expected_first,
            "top-level rejection closure failed",
        )
    else:
        _require(
            status == "pass_motion_realism_release_gate"
            and first_blocker is None
            and receipt.get("release_classification")
            == "motion_realism_gate_pass_only_other_release_gates_still_required",
            "top-level pass closure failed",
        )

    if materialization_root is not None:
        replayed = BUILDER.build_receipt(materialization_root.resolve())
        _require(
            dict(receipt) == replayed, "receipt does not match deterministic replay"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path)
    parser.add_argument(
        "--expect-status",
        choices=(
            "pass_motion_realism_release_gate",
            "reject_nonrelease_motion_realism_gate",
        ),
    )
    parser.add_argument("--require-release-pass", action="store_true")
    args = parser.parse_args()
    receipt = _load(args.receipt.resolve())
    validate_receipt(
        receipt,
        materialization_root=(
            args.materialization_root.resolve() if args.materialization_root else None
        ),
        expected_status=args.expect_status,
    )
    if args.require_release_pass:
        _require(
            receipt["status"] == "pass_motion_realism_release_gate",
            "motion-realism release pass is required",
        )
    print(
        "STRICT_TWO_HUMAN_MOTION_REALISM_RECEIPT_VALID "
        f"status={receipt['status']} receipt={args.receipt.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
