#!/usr/bin/env python3
"""Refresh the row8 sparse request against the current split visibility contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
ROW_ID = "strict_08_construction_female_left"
EPISODE_ID = "rocketbox_construction_female__strict_two_human_left_v1"
SOUND_ASSET_ID = "speech_cremad_1005_tie_neu_v1"
SCHEMA = "avengine_native_strict_two_human_row8_ready_refresh_v1"


def _module(name: str, relative_path: str) -> Any:
    path = REPOSITORY / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _module(
    "strict_two_human_expansion_preflight",
    "tools/qa/build_strict_two_human_expansion_preflight.py",
)
BATCH = _module(
    "strict_two_human_expansion_acoustic_batch",
    "tools/qa/build_strict_two_human_expansion_acoustic_batch.py",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(role: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    _require(path.is_file(), f"{label} path missing")
    if "size_bytes" in record:
        _require(path.stat().st_size == record["size_bytes"], f"{label} size drift")
    _require(_sha256(path) == record.get("sha256"), f"{label} digest drift")
    return path


def _validate_static_scene(
    *, plan: Mapping[str, Any], row: Mapping[str, Any], suite: Mapping[str, Any]
) -> None:
    scenarios = suite.get("scenarios", [])
    _require(len(scenarios) == 1, "row8 suite scenario count drift")
    scenario = scenarios[0]
    _require(scenario.get("scenario_id") == EPISODE_ID, "row8 suite episode drift")
    scene_plan = scenario["plan"]
    camera = scene_plan["camera"]
    expected_camera = row["camera_pose"]
    _require(
        camera.get("habitat_position_m") == expected_camera["translation_m"]
        and camera.get("habitat_yaw_deg") == expected_camera["habitat_yaw_deg"],
        "row8 suite camera drift",
    )
    expected_actors = {actor["source_slot_id"]: actor for actor in row["actors"]}
    identity_catalog = plan["approved_identity_catalog"]
    frames = scene_plan.get("frames", [])
    _require(len(frames) == 75, "row8 suite frame count drift")
    for frame_index, frame in enumerate(frames):
        camera_state = frame["camera_state"]
        _require(camera_state.get("frame_index") == frame_index, "camera frame drift")
        _require(
            camera_state.get("habitat_position_m") == expected_camera["translation_m"]
            and camera_state.get("habitat_yaw_deg")
            == expected_camera["habitat_yaw_deg"],
            "row8 per-frame camera drift",
        )
        states = {state["actor_id"]: state for state in frame["actor_states"]}
        _require(set(states) == {"source1_actor", "source2_actor"}, "actor slots drift")
        for slot, actor in expected_actors.items():
            state = states[f"{slot}_actor"]
            identity = identity_catalog[actor["identity_key"]]
            _require(
                state.get("translation_m") == actor["root_translation_m"]
                and abs(
                    float(state.get("actor_yaw_ue_deg", 1e9))
                    - float(actor["actor_yaw_ue_deg"])
                )
                <= 1e-9
                and state.get("asset_id") == identity["runtime_asset_id"]
                and state.get("action_id") == "idle",
                f"row8 {slot} per-frame state drift",
            )


def build(
    *,
    plan_path: Path,
    current_preflight_path: Path,
    registry_path: Path,
    old_ready_path: Path,
    old_delivery_path: Path,
    batch_root: Path,
    output: Path,
) -> Path:
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    plan = _load(plan_path)
    current_preflight = _load(current_preflight_path)
    registry = _load(registry_path)
    errors = PREFLIGHT.validate_plan(plan, registry)
    _require(not errors, "current strict8 plan failed: " + "; ".join(errors))
    thresholds = plan["projection_and_native_thresholds"]
    _require(
        thresholds.get("target_visible_fraction_minimum") == 0.8
        and thresholds.get("distractor_visible_fraction_minimum") == 0.5
        and "visible_fraction_minimum_per_actor" not in thresholds,
        "current split visibility contract mismatch",
    )
    _require(
        current_preflight.get("status")
        == "pass_cpu_plan_pending_exact_rir_and_seven_sparse_native_gates"
        and current_preflight.get("plan_id") == plan["plan_id"]
        and current_preflight.get("plan_record", {}).get("sha256")
        == _sha256(plan_path)
        and current_preflight.get("row_count") == 8
        and current_preflight.get("formal_scene_count") == 0
        and current_preflight.get("qualification_claim") is False
        and current_preflight.get("gpu_or_rir_executed") is False,
        "current strict8 preflight drift",
    )
    _require(len(plan["rows"]) == 8, "strict8 row count drift")
    row = plan["rows"][7]
    _require(
        row.get("row_id") == ROW_ID
        and row.get("episode_id") == EPISODE_ID
        and row.get("identity_pair") == "C/F"
        and row.get("target_expected_screen_side") == "left",
        "row8 identity/side drift",
    )
    actors = row["actors"]
    _require(
        len(actors) == 2
        and actors[0]["role"] == "target"
        and actors[0]["source_slot_id"] == "source1"
        and actors[0]["identity_key"] == "C"
        and actors[0]["voice_policy"] == "speaking"
        and actors[1]["role"] == "distractor"
        and actors[1]["source_slot_id"] == "source2"
        and actors[1]["identity_key"] == "F"
        and actors[1]["voice_policy"] == "silent",
        "row8 actor contract drift",
    )

    old_ready = _load(old_ready_path)
    old_thresholds = old_ready.get("projection_and_native_thresholds", {})
    _require(
        old_ready.get("status") == "ready_for_native_sparse"
        and old_ready.get("row_id") == ROW_ID
        and old_ready.get("episode_id") == EPISODE_ID
        and old_ready.get("frame_indices") == [15]
        and old_thresholds.get("visible_fraction_minimum_per_actor") == 0.5
        and "target_visible_fraction_minimum" not in old_thresholds,
        "old row8 ready is not the superseded symmetric-threshold request",
    )
    _require(
        old_ready.get("physical_gpu_index") == 1
        and old_ready.get("graphics_adapter_argument") == 1
        and old_ready.get("forbidden_physical_gpu_indices") == [0, 3]
        and old_ready.get("required_idle_compute_process_count") == 0,
        "row8 GPU policy drift",
    )
    for role, record in old_ready["cpu_acoustic_evidence"].items():
        _validate_record(record, label=f"old ready {role}")
    audio_path = Path(old_ready["audio_wav"]).resolve()
    _require(audio_path.is_file(), "row8 binaural WAV missing")
    _require(
        audio_path == Path(old_ready["audio_record"]["path"]).resolve()
        and _sha256(audio_path) == old_ready["audio_record"]["sha256"]
        and old_ready["audio_record"]["channel_count"] == 2
        and old_ready["audio_record"]["sample_rate_hz"] == 16000
        and old_ready["audio_record"]["sample_count"] == 80000,
        "row8 ready audio drift",
    )

    old_delivery = _load(old_delivery_path)
    delivered = [item for item in old_delivery.get("rows", []) if item["row_id"] == ROW_ID]
    _require(
        old_delivery.get("status")
        == "pass_cpu_rows2_to8_ready_for_sequential_f15_sparse"
        and old_delivery.get("formal_scene_count") == 0
        and old_delivery.get("gpu_executed") is False
        and len(delivered) == 1,
        "old CPU delivery row8 closure failed",
    )
    delivered_row = delivered[0]
    _require(
        delivered_row.get("episode_id") == EPISODE_ID
        and delivered_row.get("target_sound_asset_id") == SOUND_ASSET_ID
        and delivered_row.get("target_event_frame_window_inclusive") == [7, 50]
        and delivered_row.get("exact_rir_job_count") == 2
        and delivered_row.get("binaural_sample_count") == 80000
        and delivered_row.get("source1_peak_absolute", 0.0) > 0.0
        and delivered_row.get("source2_peak_absolute") == 0.0
        and Path(delivered_row["ready_capture_request"]).resolve()
        == old_ready_path.resolve(),
        "old CPU delivery row8 payload drift",
    )

    row_root = (batch_root / ROW_ID).resolve()
    recipe_root = row_root / "recipe_v1"
    _require(row_root.is_relative_to(batch_root.resolve()), "row8 root escaped batch")
    recipe = _load(recipe_root / "recipe.json")
    recipe_preflight = _load(recipe_root / "preflight.json")
    suite = _load(recipe_root / "suite_execution_plan.pending_fact.json")
    program = _load(recipe_root / "controlled_audio_program/audio_program.json")
    batch_manifest = _load(batch_root / "manifest.json")
    prepared_rows = [item for item in batch_manifest["rows"] if item["row_id"] == ROW_ID]
    _require(len(prepared_rows) == 1, "prepared batch row8 missing")
    prepared = prepared_rows[0]
    _require(
        recipe.get("row_id") == ROW_ID
        and recipe.get("episode_id") == EPISODE_ID
        and recipe.get("target_source_slot_id") == "source1"
        and recipe.get("recipe_identity_sha256")
        == old_ready.get("recipe_identity_sha256")
        == prepared.get("recipe_identity_sha256"),
        "row8 recipe identity drift",
    )
    _require(
        recipe_preflight.get("status")
        == "pass_pending_exact_rir_cache_binaural_and_native_sparse"
        and recipe_preflight.get("target_sound_asset_id") == SOUND_ASSET_ID
        and recipe_preflight.get("target_event_sample_window") == [7467, 53379]
        and recipe_preflight.get("target_event_frame_window_inclusive") == [7, 50]
        and recipe_preflight.get("target_event_count") == 1
        and recipe_preflight.get("distractor_event_count") == 0
        and recipe_preflight.get("f15_target_speaking") is True
        and recipe_preflight.get("formal_scene_count") == 0
        and recipe_preflight.get("qualification_claim") is False,
        "row8 recipe preflight drift",
    )
    _require(
        len(program.get("events", [])) == 1
        and program["events"][0]["source_endpoint_id"] == "lead_d_source1_mouth"
        and program["events"][0]["sound_asset_id"] == SOUND_ASSET_ID
        and program["events"][0]["start_sample"] == 7467
        and program["events"][0]["end_sample_exclusive"] == 53379
        and program["events"][0]["source_start_sample"] == 0
        and program["events"][0]["source_end_sample_exclusive"] == 45912,
        "row8 complete 1005 program drift",
    )
    _validate_static_scene(plan=plan, row=row, suite=suite)
    _require(
        old_ready.get("suite_plan")
        == str((recipe_root / "suite_execution_plan.pending_fact.json").resolve()),
        "old ready suite path drift",
    )

    plan_root = row_root / "exact_rir_plan_v1"
    cache_root = row_root / "rir_cache_v1"
    binaural_root = row_root / "binaural_v1"
    plan_delivery = _load(plan_root / "delivery.json")
    rir_plan = _load(plan_root / "rir_job_plan.json")
    cache_receipt = _load(cache_root / "receipt.json")
    binaural_delivery = _load(binaural_root / "delivery.json")
    samples = _load(binaural_root / "samples.json")
    _require(
        plan_delivery.get("status") == "pass"
        and plan_delivery.get("unique_rir_job_count") == 2
        and len(rir_plan.get("jobs", [])) == 2,
        "row8 exact RIR plan drift",
    )
    _require(
        cache_receipt.get("status") == "pass"
        and cache_receipt.get("selected_job_count") == 2
        and cache_receipt.get("full_plan_complete") is True,
        "row8 RIR cache drift",
    )
    _require(
        binaural_delivery.get("status") == "pass"
        and samples.get("status") == "pass"
        and samples.get("sample_count") == 1,
        "row8 binaural delivery drift",
    )
    sample = samples["samples"][0]
    stems = sample["audio"]["stems"]
    _require(
        sample.get("episode_id") == EPISODE_ID
        and sample["audio"]["sample_count"] == 80000
        and sample["audio"]["sample_rate_hz"] == 16000
        and sample["audio"]["channel_count"] == 2
        and stems["source1"]["peak_absolute"] > 0.0
        and stems["source2"]["peak_absolute"] == 0.0,
        "row8 binaural sample/stem drift",
    )

    ready = deepcopy(old_ready)
    ready["projection_and_native_thresholds"] = deepcopy(thresholds)
    ready["status"] = "ready_for_native_sparse"
    ready["contract_refresh"] = {
        "schema": SCHEMA,
        "status": "pass_current_contract_cpu_refresh_no_acoustic_recompute",
        "current_plan": _record("current_strict8_plan", plan_path),
        "current_preflight": _record(
            "current_strict8_cpu_preflight", current_preflight_path
        ),
        "superseded_ready": _record("superseded_symmetric_ready", old_ready_path),
        "superseded_reason": (
            "old request used symmetric visible_fraction_minimum_per_actor=0.5; "
            "current contract requires target=0.8 and distractor=0.5"
        ),
        "recipe_geometry_identity_unchanged": True,
        "exact_rir_cache_binaural_reused_without_recompute": True,
        "cross_row_or_cross_attempt_acoustic_reuse": False,
        "gpu_executed": False,
        "formal_scene_count": 0,
        "qualification_claim": False,
    }
    ready["formal_scene_count"] = 0
    ready["gpu_executed"] = False
    ready["qualification_claim"] = False
    output.mkdir(parents=True)
    ready_path = output / "sparse_native_gate_request.ready.json"
    receipt_path = output / "refresh_receipt.json"
    BATCH.write_json(ready_path, ready)
    BATCH.write_json(
        receipt_path,
        {
            "schema": SCHEMA,
            "status": "pass_ready_for_single_row8_f15_native_sparse",
            "row_id": ROW_ID,
            "episode_id": EPISODE_ID,
            "ready_request": _record("current_contract_row8_ready", ready_path),
            "superseded_ready": _record("superseded_symmetric_ready", old_ready_path),
            "target_visible_fraction_minimum": 0.8,
            "distractor_visible_fraction_minimum": 0.5,
            "exact_rir_job_count": 2,
            "binaural_sample_count": 80000,
            "source1_peak_absolute": stems["source1"]["peak_absolute"],
            "source2_peak_absolute": stems["source2"]["peak_absolute"],
            "gpu_executed": False,
            "formal_scene_count": 0,
            "qualification_claim": False,
        },
    )
    return ready_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--current-preflight", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--old-ready", type=Path, required=True)
    parser.add_argument("--old-delivery", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build(
        plan_path=args.plan.resolve(),
        current_preflight_path=args.current_preflight.resolve(),
        registry_path=args.runtime_registry.resolve(),
        old_ready_path=args.old_ready.resolve(),
        old_delivery_path=args.old_delivery.resolve(),
        batch_root=args.batch_root.resolve(),
        output=args.output.resolve(),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
