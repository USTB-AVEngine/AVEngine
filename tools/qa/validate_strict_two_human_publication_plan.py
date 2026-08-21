#!/usr/bin/env python3
"""Fail-closed validation for the strict two-human publication plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
SCHEMA = "avengine_native_strict_two_human_publication_plan_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_capture(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    root = _resolve(str(row["capture_root"]))
    manifest = _load(root / "manifest.json")
    visibility = _load(root / "pixel_visibility_truth.json")
    assets = _load(root / "runtime_asset_readbacks.json")
    _require(manifest.get("status") == "pass", f"{row['row_id']} capture failed")
    _require(
        manifest.get("scenario_id") == row["episode_id"],
        f"{row['row_id']} episode mismatch",
    )
    frame = manifest.get("frame_contract", {})
    _require(
        frame.get("frame_count") == 1
        and frame.get("captured_frame_indices") == [15]
        and frame.get("formal_episode_frame_count") == 75
        and frame.get("resolution_hw") == [720, 1280],
        f"{row['row_id']} sparse frame contract drift",
    )
    _require(
        manifest.get("benchmark_qualification_claim") is False
        and manifest.get("runtime_alignment", {}).get("target_pass_count") == 2
        and manifest.get("runtime_alignment", {}).get("maximum_location_drift_cm")
        == 0.0
        and manifest.get("runtime_alignment", {}).get("maximum_rotation_drift_deg")
        == 0.0,
        f"{row['row_id']} claim/readback drift",
    )
    _require(
        assets.get("status") == "pass"
        and assets.get("per_instance", {}).get("source1", {}).get("status")
        == "pass"
        and assets.get("per_instance", {}).get("source2", {}).get("status")
        == "pass",
        f"{row['row_id']} live6 failed",
    )
    per_instance = visibility.get("per_instance", {})
    target = per_instance.get("source1", {}).get("frames", [{}])[0]
    distractor = per_instance.get("source2", {}).get("frames", [{}])[0]
    _require(
        float(target.get("visible_fraction", -1.0))
        >= float(contract["target_visible_fraction_minimum"]),
        f"{row['row_id']} target visibility failed",
    )
    _require(
        float(distractor.get("visible_fraction", -1.0))
        >= float(contract["distractor_visible_fraction_minimum"]),
        f"{row['row_id']} distractor visibility failed",
    )
    minimum_pixels = int(contract["visible_pixels_minimum_per_actor"])
    _require(
        int(target.get("visible_pixels", -1)) >= minimum_pixels
        and int(distractor.get("visible_pixels", -1)) >= minimum_pixels,
        f"{row['row_id']} visible pixels failed",
    )
    expected_left = row["target_side"] == "left"
    target_x = float(target["target_centroid_xy_px"][0]) / 1280.0
    distractor_x = float(distractor["target_centroid_xy_px"][0]) / 1280.0
    dead_zone = float(contract["screen_side_dead_zone_fraction"])
    _require(
        (target_x < 0.5 - dead_zone and distractor_x > 0.5 + dead_zone)
        if expected_left
        else (target_x > 0.5 + dead_zone and distractor_x < 0.5 - dead_zone),
        f"{row['row_id']} screen side failed",
    )
    for owner, record in (("target", target), ("distractor", distractor)):
        x1, y1, x2, y2 = record["target_bbox_xyxy_px"]
        margin = int(contract["bbox_edge_margin_px_minimum"])
        _require(
            x1 >= margin and y1 >= margin and 1279 - x2 >= margin and 719 - y2 >= margin,
            f"{row['row_id']} {owner} bbox touches edge",
        )
    for role in (
        "metric_depth",
        "pixel_masks",
        "pixel_visibility_truth",
        "runtime_readbacks",
        "runtime_asset_readbacks",
        "native_rgb_visual_only",
        "native_rgb_binaural",
        "rgb_frames",
    ):
        _require(role in manifest.get("artifact_records", {}), f"{row['row_id']} lacks {role}")


def _validate_gate(row: Mapping[str, Any]) -> None:
    gate = _load(_resolve(str(row["cpu_gate"])))
    if row["cpu_gate_kind"] == "final_gate_ledger":
        _require(
            gate.get("status") == "pass"
            and gate.get("episode_id") == row["episode_id"]
            and gate.get("capture_process_exit_code") == 0
            and gate.get("formal_scene_count") == 0
            and gate.get("qualification_claim") is False
            and gate.get("acoustics", {}).get("speech_frame_window_inclusive")
            == row["speech_frame_window_inclusive"],
            f"{row['row_id']} final gate drift",
        )
        return
    _require(row["cpu_gate_kind"] == "ready_request", "unknown CPU gate kind")
    _require(
        gate.get("status") == "ready_for_native_sparse"
        and gate.get("episode_id") == row["episode_id"]
        and gate.get("frame_indices") == [15]
        and gate.get("audio_record", {}).get("channel_count") == 2
        and gate.get("audio_record", {}).get("sample_rate_hz") == 16000
        and gate.get("audio_record", {}).get("sample_count") == 80000,
        f"{row['row_id']} ready request drift",
    )
    for evidence in gate.get("cpu_acoustic_evidence", {}).values():
        _require(_resolve(str(evidence["path"])).is_file(), f"{row['row_id']} CPU evidence missing")
    suite = _load(_resolve(str(gate["suite_plan"])))
    scenario = suite.get("scenarios", [{}])[0]
    recipe_root = _resolve(str(gate["suite_plan"])).parent
    program = _load(recipe_root / "controlled_audio_program/audio_program.json")
    events = program.get("events", [])
    _require(
        scenario.get("scenario_id") == row["episode_id"]
        and len(events) == 1
        and events[0].get("sound_asset_id") == row["target_sound_asset_id"]
        and events[0].get("source_endpoint_id") == "lead_d_source1_mouth",
        f"{row['row_id']} controlled audio drift",
    )


def validate(plan_path: Path) -> dict[str, Any]:
    plan = _load(plan_path)
    _require(plan.get("schema") == SCHEMA, "publication schema mismatch")
    _require(plan.get("status") == "ready_for_cpu_publication", "publication status mismatch")
    _require(
        plan.get("counted_sparse_scene_count") == 8
        and plan.get("formal_scene_count") == 0
        and plan.get("qualification_claim") is False,
        "publication claim boundary drift",
    )
    current = _load(_resolve(str(plan["current_contract"])))
    contract = plan["visibility_contract"]
    current_thresholds = current["projection_and_native_thresholds"]
    _require(
        contract["target_visible_fraction_minimum"]
        == current_thresholds["target_visible_fraction_minimum"]
        == 0.8
        and contract["distractor_visible_fraction_minimum"]
        == current_thresholds["distractor_visible_fraction_minimum"]
        == 0.5,
        "publication visibility contract drift",
    )
    rows = plan.get("rows", [])
    _require(len(rows) == 8, "publication must contain exactly eight counted rows")
    _require([row["row_index"] for row in rows] == list(range(1, 9)), "row indices drift")
    _require(all(row.get("counted") is True for row in rows), "counted row flag drift")
    _require(len({row["row_id"] for row in rows}) == 8, "row IDs are not unique")
    _require(len({row["episode_id"] for row in rows}) == 8, "episode IDs are not unique")
    sides = Counter(row["target_side"] for row in rows)
    _require(sides == {"left": 4, "right": 4}, "target side balance drift")
    targets = Counter(row["target_identity_key"] for row in rows)
    distractors = Counter(row["distractor_identity_key"] for row in rows)
    _require(targets == distractors == {"M": 3, "F": 3, "C": 2}, "identity balance drift")
    for row in rows:
        _validate_capture(row, contract)
        _validate_gate(row)
    excluded = plan.get("excluded_attempts", [])
    _require(len(excluded) == 1 and excluded[0].get("counted") is False, "excluded history drift")
    rejection = _load(_resolve(str(excluded[0]["rejection_record"])))
    _require(
        rejection.get("status") == "rejected"
        and rejection.get("row_id") == rows[6]["row_id"]
        and rejection.get("target_gate", {}).get("observed_visible_fraction", 1.0) < 0.8
        and rejection.get("formal_scene_count") == 0
        and rejection.get("qualification_claim") is False,
        "row7 v1 rejection drift",
    )
    return {
        "schema": "avengine_native_strict_two_human_publication_validation_v1",
        "status": "pass",
        "counted_sparse_scene_count": 8,
        "target_side_counts": dict(sides),
        "target_identity_counts": dict(targets),
        "distractor_identity_counts": dict(distractors),
        "excluded_attempt_count": 1,
        "formal_scene_count": 0,
        "qualification_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.plan.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
