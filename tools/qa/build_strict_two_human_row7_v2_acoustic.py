#!/usr/bin/env python3
"""Prepare and finalize the CPU-only strict two-human row7 v2 acoustics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]


def _module(name: str, relative_path: str) -> Any:
    path = REPOSITORY / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OVERLAY = _module(
    "strict_row7_v2_overlay",
    "tools/qa/build_strict_two_human_row7_v2_preflight.py",
)
BATCH = _module(
    "strict_two_human_expansion_acoustic_batch",
    "tools/qa/build_strict_two_human_expansion_acoustic_batch.py",
)

PREPARED_STATUS = "prepared_cpu_row7_v2_pending_exact_rir_cache_binaural"
DELIVERY_SCHEMA = "avengine_native_strict_two_human_row7_v2_cpu_delivery_v1"
DELIVERY_STATUS = "pass_cpu_row7_v2_ready_for_single_f15_sparse"
DELIVERY_CLAIM = (
    "row7 v2 CPU acoustics and request only; no native capture, formal scene, "
    "or qualification claim"
)
ROW_ID = "strict_07_female_construction_right"
EPISODE_ID = "rocketbox_female_construction__strict_two_human_right_v2"
SOUND_ASSET_ID = "speech_cremad_1002_mti_neu_v1"
OLD_V1_ACOUSTIC_ROOT = (
    REPOSITORY
    / "tmp/lead_d_strict_two_human_expansion_v1/acoustic_batch_v1"
    / ROW_ID
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def prepare(
    *,
    overlay_path: Path,
    overlay_preflight_path: Path,
    registry_path: Path,
    source_suite_path: Path,
    controlled_registry_path: Path,
    output: Path,
) -> Path:
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    overlay = _load(overlay_path)
    overlay_preflight = _load(overlay_preflight_path)
    base_path = OVERLAY._resolve(str(overlay["base_plan"]))
    rejection_path = OVERLAY._resolve(str(overlay["v1_rejection"]))
    base = _load(base_path)
    rejection = _load(rejection_path)
    registry = _load(registry_path)
    errors = OVERLAY.validate_overlay(overlay, base, rejection, registry)
    _require(not errors, "overlay failed: " + "; ".join(errors))
    _require(
        overlay_preflight.get("status")
        == "pass_cpu_geometry_revision_pending_exact_rir_and_single_sparse_gate",
        "row7 v2 overlay preflight status mismatch",
    )
    _require(
        overlay_preflight.get("revision_id") == overlay["revision_id"]
        and overlay_preflight.get("target_row_id") == ROW_ID
        and overlay_preflight.get("replacement_episode_id") == EPISODE_ID,
        "row7 v2 overlay preflight identity mismatch",
    )
    _require(
        overlay_preflight.get("formal_scene_count") == 0
        and overlay_preflight.get("qualification_claim") is False
        and overlay_preflight.get("gpu_or_rir_executed") is False,
        "row7 v2 overlay preflight claim boundary drift",
    )
    effective_plan_path = Path(overlay_preflight["effective_plan"])
    expansion_preflight_path = Path(overlay_preflight["expansion_preflight"])
    effective_plan = _load(effective_plan_path)
    expected_plan, row_index = OVERLAY.apply_overlay(base, overlay)
    _require(effective_plan == expected_plan, "effective plan does not match overlay")
    _require(row_index == 6, "row7 materialized index drift")
    expansion_preflight = _load(expansion_preflight_path)
    _require(
        expansion_preflight.get("status")
        == "pass_cpu_plan_pending_exact_rir_and_seven_sparse_native_gates"
        and expansion_preflight.get("plan_record", {}).get("sha256")
        == BATCH._sha256(effective_plan_path),
        "effective plan expansion preflight drift",
    )
    _require(
        expansion_preflight.get("formal_scene_count") == 0
        and expansion_preflight.get("gpu_or_rir_executed") is False,
        "effective expansion preflight claim boundary drift",
    )
    row = effective_plan["rows"][row_index]
    _require(
        row["row_id"] == ROW_ID
        and row["episode_id"] == EPISODE_ID
        and row["identity_pair"] == "F/C",
        "row7 v2 effective row mismatch",
    )
    _require(
        row["actors"][0]["identity_key"] == "F"
        and row["actors"][0]["source_slot_id"] == "source1"
        and row["actors"][0]["voice_policy"] == "speaking"
        and row["actors"][1]["identity_key"] == "C"
        and row["actors"][1]["source_slot_id"] == "source2"
        and row["actors"][1]["voice_policy"] == "silent",
        "row7 v2 identity/voice contract drift",
    )
    _require(
        not output.resolve().is_relative_to(OLD_V1_ACOUSTIC_ROOT.resolve()),
        "row7 v2 output cannot be placed under the v1 acoustic root",
    )

    source_suite = _load(source_suite_path)
    controlled_registry = _load(controlled_registry_path)
    output.mkdir(parents=True)
    row_record = BATCH._row_recipe(
        row=row,
        plan=effective_plan,
        plan_path=effective_plan_path,
        cpu_preflight_path=overlay_preflight_path,
        registry=registry,
        registry_path=registry_path,
        source_suite=source_suite,
        source_suite_path=source_suite_path,
        controlled_registry=controlled_registry,
        controlled_registry_path=controlled_registry_path,
        output=output / ROW_ID / "recipe_v1",
    )
    _require(
        row_record["episode_id"] == EPISODE_ID
        and row_record["target_sound_asset_id"] == SOUND_ASSET_ID
        and row_record["target_event_frame_window_inclusive"] == [7, 50],
        "row7 v2 recipe speech contract drift",
    )
    manifest = {
        "schema": BATCH.SCHEMA,
        "status": PREPARED_STATUS,
        "claim_boundary": DELIVERY_CLAIM,
        "row_count": 1,
        "overlay_input": BATCH._record("row7_v2_overlay", overlay_path),
        "overlay_preflight_input": BATCH._record(
            "row7_v2_overlay_preflight", overlay_preflight_path
        ),
        "effective_plan_input": BATCH._record(
            "row7_v2_effective_plan", effective_plan_path
        ),
        "expansion_preflight_input": BATCH._record(
            "row7_v2_expansion_preflight", expansion_preflight_path
        ),
        "v1_rejection_input": BATCH._record(
            "row7_v1_rejection", rejection_path
        ),
        "forbidden_v1_acoustic_root": str(OLD_V1_ACOUSTIC_ROOT.resolve()),
        "cross_attempt_rir_cache_audio_reuse_allowed": False,
        "retained_row1_canary": {
            "row_id": effective_plan["rows"][0]["row_id"],
            "episode_id": effective_plan["rows"][0]["episode_id"],
            "status": effective_plan["rows"][0]["status"],
        },
        "rows": [row_record],
        "cross_row_rir_reuse_allowed": False,
        "gpu_executed": False,
        "formal_scene_count": 0,
    }
    manifest_path = output / "manifest.json"
    BATCH.write_json(manifest_path, manifest)
    return manifest_path


def finalize(*, batch_root: Path, output: Path) -> Path:
    manifest_path = batch_root / "manifest.json"
    manifest = _load(manifest_path)
    _require(manifest.get("status") == PREPARED_STATUS, "prepared status drift")
    _require(manifest.get("row_count") == 1, "row7 v2 row count drift")
    _require(
        manifest.get("cross_attempt_rir_cache_audio_reuse_allowed") is False,
        "cross-attempt reuse must remain forbidden",
    )
    _require(
        manifest.get("forbidden_v1_acoustic_root")
        == str(OLD_V1_ACOUSTIC_ROOT.resolve()),
        "forbidden v1 acoustic root drift",
    )
    row = manifest["rows"][0]
    _require(
        row.get("row_id") == ROW_ID and row.get("episode_id") == EPISODE_ID,
        "prepared row7 v2 identity drift",
    )
    for relative in ("exact_rir_plan_v1", "rir_cache_v1", "binaural_v1"):
        artifact_root = (batch_root / ROW_ID / relative).resolve()
        _require(
            artifact_root.is_relative_to(batch_root.resolve())
            and not artifact_root.is_relative_to(OLD_V1_ACOUSTIC_ROOT.resolve()),
            f"{relative} reused the v1 acoustic root",
        )
    delivery_path = BATCH.finalize(
        batch_root=batch_root,
        output=output,
        expected_row_count=1,
        delivery_schema=DELIVERY_SCHEMA,
        delivery_status=DELIVERY_STATUS,
        delivery_claim_boundary=DELIVERY_CLAIM,
    )
    delivery = _load(delivery_path)
    _require(
        delivery.get("row_count") == 1
        and delivery.get("exact_rir_job_count") == 2
        and delivery.get("cross_row_rir_reuse_count") == 0,
        "row7 v2 delivery count drift",
    )
    delivered = delivery["rows"][0]
    _require(
        delivered.get("episode_id") == EPISODE_ID
        and delivered.get("target_sound_asset_id") == SOUND_ASSET_ID
        and delivered.get("target_event_frame_window_inclusive") == [7, 50]
        and delivered.get("exact_rir_job_count") == 2
        and delivered.get("binaural_sample_count") == 80000
        and delivered.get("source1_peak_absolute", 0.0) > 0.0
        and delivered.get("source2_peak_absolute") == 0.0,
        "row7 v2 delivered acoustic contract drift",
    )
    delivery["prepared_manifest_input"] = BATCH._record(
        "row7_v2_prepared_manifest", manifest_path
    )
    delivery["overlay_input"] = manifest["overlay_input"]
    delivery["overlay_preflight_input"] = manifest["overlay_preflight_input"]
    delivery["effective_plan_input"] = manifest["effective_plan_input"]
    delivery["v1_rejection_input"] = manifest["v1_rejection_input"]
    delivery["forbidden_v1_acoustic_root"] = manifest[
        "forbidden_v1_acoustic_root"
    ]
    delivery["cross_attempt_rir_cache_audio_reuse_count"] = 0
    BATCH.write_json(delivery_path, delivery)
    return delivery_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--overlay",
        type=Path,
        default=(
            REPOSITORY
            / "examples/qa/native_strict_two_human_row7_v2_revision_overlay.json"
        ),
    )
    prepare_parser.add_argument("--overlay-preflight", type=Path, required=True)
    prepare_parser.add_argument(
        "--runtime-registry",
        type=Path,
        default=REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
    )
    prepare_parser.add_argument("--source-suite", type=Path, required=True)
    prepare_parser.add_argument("--controlled-registry", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--batch-root", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(
            overlay_path=args.overlay.resolve(),
            overlay_preflight_path=args.overlay_preflight.resolve(),
            registry_path=args.runtime_registry.resolve(),
            source_suite_path=args.source_suite.resolve(),
            controlled_registry_path=args.controlled_registry.resolve(),
            output=args.output.resolve(),
        )
    else:
        result = finalize(
            batch_root=args.batch_root.resolve(), output=args.output.resolve()
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
