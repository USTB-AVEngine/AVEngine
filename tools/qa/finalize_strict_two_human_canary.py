#!/usr/bin/env python3
"""Finalize and fail-closed validate the strict two-human sparse canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import write_json
from avengine.qa.fact_table import compile_episode_fact_table

EPISODE_ID = "rocketbox_male_female__strict_two_human_canary_v1"
GPU1_UUID = "GPU-6d3e273e-58c6-2a5b-480a-4816fef6c581"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _fact_record(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": str(path.resolve()), "sha256": _sha256(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _one(values: Sequence[Mapping[str, Any]], owner: str) -> Mapping[str, Any]:
    _require(len(values) == 1, f"{owner} must resolve exactly once")
    return values[0]


def _contains_pending(value: Any) -> bool:
    if isinstance(value, str):
        return "PENDING" in value.upper() or "pending_required" in value.lower()
    if isinstance(value, Mapping):
        return any(_contains_pending(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_pending(item) for item in value)
    return False


def _build_facts(
    *,
    recipe_root: Path,
    audio_plan_root: Path,
    audio_batch_root: Path,
    rir_cache_root: Path,
    controlled_registry_path: Path,
    runtime_registry_path: Path,
    source_fact_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    bank_path = recipe_root / "trajectory_bank.json"
    sensor_path = recipe_root / "sensor_rig_trajectory.json"
    pending_suite_path = recipe_root / "suite_execution_plan.pending_fact.json"
    program_root = recipe_root / "controlled_audio_program"
    binding_path = program_root / "controlled_audio_binding.json"
    instance_path = audio_batch_root / "labels/audio_program_instances/v00.json"
    samples_path = audio_batch_root / "samples.json"
    episodes_path = audio_batch_root / "episodes.json"
    dry_library_path = audio_batch_root / "dry_audio_variants.json"
    bank = _load(bank_path)
    sensor = _load(sensor_path)
    suite = _load(pending_suite_path)
    source_fact = _load(source_fact_path)
    registry = _load(runtime_registry_path)
    binding = _load(binding_path)
    instance = _load(instance_path)
    sample = _one(_load(samples_path)["samples"], "audio sample")
    episode = _one(_load(episodes_path)["episodes"], "audio episode")
    episode_id = bank["episodes"][0]["episode_id"]
    _require(episode_id == EPISODE_ID, "strict Episode identity drift")
    _require(sample["episode_id"] == episode_id, "audio Episode identity drift")
    activity = instance["source_activity_summary"]
    _require(
        activity["active_source_slots"] == ["source1"]
        and activity["silent_source_slots"] == ["source2"]
        and activity["active_sample_count_by_source_slot"]
        == {"source1": 25626, "source2": 0},
        "strict source activity drift",
    )
    mixture_path = audio_batch_root / "audio/binaural" / sample["audio"]["mixture"]["path"]
    _require(mixture_path.is_file(), "authoritative binaural mixture is missing")
    _require(
        _sha256(mixture_path) == sample["audio"]["mixture"]["audio_sha256"],
        "authoritative binaural mixture drift",
    )
    sample_entry = deepcopy(sample)
    sample_entry["audio"]["mixture"]["path"] = str(mixture_path.resolve())

    events = instance["mapped_events"]
    _require(len(events) == 1, "strict canary must have exactly one event")
    event = events[0]
    _require(
        event["source_slot_id"] == "source1"
        and event["start_sample"] == 7467
        and event["end_sample_exclusive"] == 33093
        and event["start_tick"] == 22401
        and event["end_tick_exclusive"] == 99279,
        "strict source1 speech interval drift",
    )
    declared_events = {
        "source1": [
            {
                "event_id": event["event_id"],
                "start_sample": event["start_sample"],
                "end_sample_exclusive": event["end_sample_exclusive"],
                "start_tick": event["start_tick"],
                "end_tick_exclusive": event["end_tick_exclusive"],
                "fade_samples": event["fade_samples"],
                "gating": "m6_audio_program_event_window_v1",
            }
        ],
        "source2": [],
    }
    dry_path = Path(binding["sound_audio_paths"][event["sound_asset_id"]]).resolve()
    dry_sha = _sha256(dry_path)
    dry_variants = {
        "source1": {
            "variant_index": 0,
            "record": {
                "input": {"path": str(dry_path), "sha256": dry_sha},
                "linear_gain": float(event["linear_gain"]),
            },
        },
        "source2": {
            "variant_index": 0,
            "record": {
                "input": {"path": str(dry_path), "sha256": dry_sha},
                "linear_gain": 0.0,
            },
        },
    }
    provenance = [
        (recipe_root / "recipe.json", "strict_two_human_recipe"),
        (bank_path, "trajectory_bank"),
        (audio_plan_root / "rir_job_plan.json", "rir_job_plan"),
        (audio_plan_root / "delivery.json", "rir_delivery"),
        (samples_path, "audio_batch_samples"),
        (episodes_path, "audio_batch_episode_cache_index"),
        (dry_library_path, "dry_audio_variant_library"),
        (runtime_registry_path, "source_asset_runtime_registry"),
        (controlled_registry_path, "controlled_sound_content_registry"),
        (program_root / "source_endpoint_registry.json", "source_endpoint_registry"),
        (program_root / "sound_asset_registry.json", "sound_asset_registry"),
        (program_root / "audio_program.json", "audio_program"),
        (instance_path, "materialized_audio_program_instance"),
        (sensor_path, "sensor_rig_trajectory"),
        (rir_cache_root / "receipt.json", "native_rlr_rir_cache_receipt"),
        (source_fact_path, "retained_source_fact"),
        (pending_suite_path, "retained_source_native_suite_recipe"),
    ]
    facts = compile_episode_fact_table(
        bank_header={key: value for key, value in bank.items() if key != "episodes"},
        bank_episode=bank["episodes"][0],
        listener_position_m=sensor["frames"][0]["world_from_rig"]["translation_m"],
        listener_orientation_wxyz=[
            sensor["frames"][0]["world_from_rig"]["rotation_xyzw"][3],
            *sensor["frames"][0]["world_from_rig"]["rotation_xyzw"][:3],
        ],
        sample_entry=sample_entry,
        dry_variants_by_slot=dry_variants,
        registry=registry,
        anchors=source_fact["anchors"],
        room=source_fact["room"],
        camera={
            "hfov_degrees": source_fact["visibility"]["hfov_degrees"],
            "resolution_hw": source_fact["visibility"]["resolution_hw"],
        },
        rir_cache_request_identity_sha256=episode["rir_cache"]["cache_request_identity_sha256"],
        provenance_inputs=[_fact_record(path, role) for path, role in provenance],
        declared_events_by_slot=declared_events,
        sensor_rig_trajectory=sensor,
        controlled_content_by_slot=binding["controlled_content"],
        silent_source_slots={"source2"},
    )
    jsonschema.Draft202012Validator(
        _load(REPOSITORY / "schemas/avengine_qa_fact_table_v1.schema.json")
    ).validate(facts)
    _require(len(facts["sound_events"]) == 1, "Facts event count drift")
    fact_event = facts["sound_events"][0]
    _require(
        fact_event["source_slot_id"] == "source1"
        and fact_event["start_frame"] == 7
        and fact_event["end_frame"] == 32,
        "Facts speaking window drift",
    )
    _require(
        not any(facts["tracks"]["instances"]["source1"]["moving"][7:32]),
        "target moved during its complete speech window",
    )
    return facts, suite, binding, mixture_path


def _validate_capture(
    *,
    capture_root: Path,
    mixture_path: Path,
    capture_exit_code: int,
) -> dict[str, Any]:
    _require(capture_exit_code == 0, "capture process exit code was non-zero")
    manifest = _load(capture_root / "manifest.json")
    readbacks = _load(capture_root / "runtime_asset_readbacks.json")
    pixels = _load(capture_root / "pixel_visibility_truth.json")
    _require(manifest["status"] == "pass", "native capture did not pass")
    _require(
        manifest["scenario_id"] == EPISODE_ID
        and manifest["frame_contract"]["captured_frame_indices"] == [15],
        "native capture Episode/frame drift",
    )
    _require(
        Path(manifest["audio"]["authoritative_wav"]).resolve() == mixture_path.resolve(),
        "native capture audio lineage drift",
    )
    _require(
        manifest["metric_depth"]["shape_nhw"] == [1, 720, 1280]
        and manifest["metric_depth"]["dtype"] == "float16"
        and manifest["metric_depth"]["finite_positive_fraction"] == 1.0,
        "metric-depth gate failed",
    )
    _require(
        manifest["runtime_alignment"]["maximum_location_drift_cm"] == 0.0
        and manifest["runtime_alignment"]["maximum_rotation_drift_deg"] == 0.0,
        "runtime pose/readback drift",
    )
    _require(readbacks["status"] == "pass", "runtime asset readback failed")
    for slot in ("source1", "source2"):
        item = readbacks["per_instance"][slot]
        _require(item["status"] == "pass", f"{slot} runtime readback failed")
        _require(
            item["blueprint"]["spawned_actor_exact_class_match"]
            and item["blueprint"]["expected_class_handle"]
            == item["blueprint"]["observed_class_handle"],
            f"{slot} exact Blueprint gate failed",
        )
        _require(
            item["skeletal_mesh"]["expected_handle"]
            == item["skeletal_mesh"]["observed_handle"],
            f"{slot} mesh gate failed",
        )
        _require(
            item["skeleton"]["expected_handle"]
            == item["skeleton"]["observed_mesh_skeleton_handle"],
            f"{slot} skeleton gate failed",
        )
        _require(
            item["standing_idle"]["expected_handle"]
            == item["standing_idle"]["observed_animation_asset_handle"]
            and item["standing_idle"]["absolute_position_error_seconds"] == 0.0,
            f"{slot} Standing_Idle gate failed",
        )
        _require(
            item["stable_actor_tag"]["status"] == "pass"
            and item["stable_actor_tag"]["descriptor_match_count"] > 0,
            f"{slot} stable tag gate failed",
        )
        emitter = item["emitter_native_readback"]
        _require(
            emitter["status"] == "pass"
            and emitter["authority"] == "native_actor_root_plus_declared_profile_offset"
            and emitter["maximum_absolute_error_m"] <= 1.0e-12,
            f"{slot} emitter binding gate failed",
        )
    height, width = pixels["resolution_hw"]
    center_x = (width - 1) / 2.0
    side_margin = width * 0.02
    summaries: dict[str, Any] = {}
    for slot in ("source1", "source2"):
        frame = pixels["per_instance"][slot]["frames"][0]
        x0, y0, x1, y1 = frame["target_bbox_xyxy_px"]
        _require(x0 > 0 and y0 > 0 and x1 < width - 1 and y1 < height - 1, f"{slot} bbox touches frame edge")
        _require(frame["visible_pixels"] >= 5000, f"{slot} visible-pixel gate failed")
        _require(frame["visible_fraction"] >= 0.50, f"{slot} visible-fraction gate failed")
        centroid_x = float(frame["target_centroid_xy_px"][0])
        if slot == "source1":
            _require(centroid_x > center_x + side_margin, "source1 is not on the right")
        else:
            _require(centroid_x < center_x - side_margin, "source2 is not on the left")
        summaries[slot] = {
            "bbox_xyxy_px": frame["target_bbox_xyxy_px"],
            "centroid_xy_px": frame["target_centroid_xy_px"],
            "visible_pixels": frame["visible_pixels"],
            "visible_fraction": frame["visible_fraction"],
            "state": frame["state"],
        }
    return {"manifest": manifest, "readbacks": readbacks, "pixels": summaries}


def finalize(
    *,
    recipe_root: Path,
    audio_plan_root: Path,
    audio_batch_root: Path,
    rir_cache_root: Path,
    capture_root: Path,
    controlled_registry_path: Path,
    runtime_registry_path: Path,
    source_fact_path: Path,
    capture_exit_code: int,
    output: Path,
) -> Path:
    facts, suite, binding, mixture_path = _build_facts(
        recipe_root=recipe_root,
        audio_plan_root=audio_plan_root,
        audio_batch_root=audio_batch_root,
        rir_cache_root=rir_cache_root,
        controlled_registry_path=controlled_registry_path,
        runtime_registry_path=runtime_registry_path,
        source_fact_path=source_fact_path,
    )
    capture = _validate_capture(
        capture_root=capture_root,
        mixture_path=mixture_path,
        capture_exit_code=capture_exit_code,
    )
    delivery = _load(audio_plan_root / "delivery.json")
    assignments = _load(audio_plan_root / "sound_assignments.json")
    cache_receipt = _load(rir_cache_root / "receipt.json")
    _require(
        delivery["sound_pair_counts"] == {"human_speech|silent_human": 1}
        and assignments["ordered_pair_counts"]
        == {"human_speech|silent_human": 1},
        "strict human/silent acoustic labels drift",
    )
    _require(
        delivery["status"] == "pass"
        and "dog barking" not in json.dumps(delivery).lower()
        and "cat meowing" not in json.dumps(delivery).lower()
        and "dog barking" not in json.dumps(assignments).lower()
        and "cat meowing" not in json.dumps(assignments).lower(),
        "animal template metadata leaked into strict RIR delivery",
    )
    _require(
        cache_receipt["status"] == "pass"
        and cache_receipt["full_plan_complete"] is True
        and cache_receipt["full_plan_job_count"] == 2,
        "exact native RLR cache gate failed",
    )
    audio_sample = _one(
        _load(audio_batch_root / "samples.json")["samples"], "final audio sample"
    )
    audio = audio_sample["audio"]
    _require(
        audio["channel_count"] == 2
        and audio["sample_rate_hz"] == 16000
        and audio["sample_count"] == 80000
        and audio["stems"]["source1"]["peak_absolute"] > 0.0
        and audio["stems"]["source2"]["peak_absolute"] == 0.0,
        "strict binaural audio/stem gate failed",
    )
    output.mkdir(parents=True, exist_ok=False)
    facts_path = output / "facts.json"
    write_json(facts_path, facts)
    request = {
        "request_id": f"{EPISODE_ID}__native_capture",
        "episode_id": EPISODE_ID,
        "scenario_type": "strict_two_human_static_canary",
        "target_source_slot_id": "source1",
        "fact_path": str(facts_path.resolve()),
        "fact_sha256": _sha256(facts_path),
    }
    scenario = suite["scenarios"][0]
    scenario["scenario_directory"] = str(capture_root.resolve())
    scenario["authoritative_capture_request"] = deepcopy(request)
    requests = {
        "schema": "avengine_native_pixel_capture_requests_v1",
        "status": "complete",
        "qualification_claim": False,
        "requests": [deepcopy(request)],
    }
    requests_path = output / "native_capture_requests.json"
    write_json(requests_path, requests)
    suite["camera_upgrade"]["capture_requests"] = str(requests_path.resolve())
    suite["camera_upgrade"]["capture_requests_sha256"] = _sha256(requests_path)
    suite_path = output / "suite_execution_plan.json"
    write_json(suite_path, suite)
    invocation = {
        "schema": "avengine_native_capture_invocation_receipt_v1",
        "status": "pass",
        "process_exit_code": capture_exit_code,
        "selected_physical_gpu_index": 1,
        "selected_physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "gpu_gate_authority": "caller_persisted_from_prelaunch_nvidia_smi_check",
        "captured_frame_indices": [15],
        "capture_manifest": _record(capture_root / "manifest.json", "native_capture_manifest"),
    }
    invocation_path = output / "capture_invocation_receipt.json"
    write_json(invocation_path, invocation)
    capture_binding = {
        "schema": "avengine_strict_two_human_capture_binding_receipt_v1",
        "status": "pass",
        "qualification_claim": False,
        "authoritative_capture_request": deepcopy(request),
        "source_capture_manifest": _record(
            capture_root / "manifest.json", "source_native_capture_manifest"
        ),
        "capture_invocation": _record(
            invocation_path, "capture_invocation_receipt"
        ),
    }
    capture_binding_path = output / "capture_binding_receipt.json"
    write_json(capture_binding_path, capture_binding)
    preflight = _load(recipe_root / "preflight.json")
    preflight.update(
        {
            "status": "pass",
            "exact_rir": "pass",
            "native_sparse": "pass",
            "fact_binding": "pass",
            "capture_process_exit_code": capture_exit_code,
        }
    )
    finalized_preflight_path = output / "finalized_preflight.json"
    write_json(finalized_preflight_path, preflight)
    sparse_gate_receipt = {
        "schema": "avengine_strict_two_human_sparse_gate_receipt_v1",
        "status": "pass",
        "qualification_claim": False,
        "captured_frame_indices": [15],
        "capture_binding": _record(
            capture_binding_path, "capture_binding_receipt"
        ),
        "facts": _record(facts_path, "authoritative_fact"),
        "request": deepcopy(request),
    }
    sparse_gate_receipt_path = output / "finalized_sparse_gate_receipt.json"
    write_json(sparse_gate_receipt_path, sparse_gate_receipt)
    recipe = _load(recipe_root / "recipe.json")
    recipe["status"] = "pass"
    recipe["outputs"]["suite"] = str(suite_path.resolve())
    recipe["outputs"]["preflight"] = str(finalized_preflight_path.resolve())
    recipe["outputs"].pop("sparse_gate_request", None)
    recipe["outputs"]["native_capture_requests"] = str(requests_path.resolve())
    recipe["outputs"]["sparse_gate_receipt"] = str(
        sparse_gate_receipt_path.resolve()
    )
    recipe["finalization"] = {
        "facts": _record(facts_path, "authoritative_fact"),
        "suite": _record(suite_path, "finalized_suite"),
        "rir_delivery": _record(audio_plan_root / "delivery.json", "exact_rir_delivery"),
        "rir_cache": _record(rir_cache_root / "receipt.json", "exact_rir_cache_receipt"),
        "audio": _record(mixture_path, "authoritative_binaural_audio"),
        "capture_binding": _record(
            capture_binding_path, "native_capture_binding_receipt"
        ),
    }
    finalized_recipe_path = output / "finalized_recipe.json"
    write_json(finalized_recipe_path, recipe)
    plan = _load(Path(recipe["inputs"]["plan"]))
    mouth_xy = plan["deterministic_composition"]
    recipe_preflight = _load(recipe_root / "preflight.json")
    target_mouth_xy = recipe_preflight["projection_xy_fraction"]["source1"]
    _require(
        all(0.05 < float(value) < 0.95 for value in target_mouth_xy),
        "target mouth projection leaves 5-95 percent region",
    )
    _require(
        plan["target_only_actor_map"]
        == {"source1": "source1_actor", "source2": "source2_actor"}
        and recipe["target_source_slot_id"] == "source1"
        and binding["controlled_content"]["source1"]["sound_asset_id"]
        == "speech_cremad_1001_ieo_neu_v1"
        and binding["controlled_content"]["source2"] is None,
        "target mouth/audio identity binding drift",
    )
    _require(mouth_xy["post_capture_bbox_edge_margin_px"] >= 1, "bbox margin contract drift")
    _require(not _contains_pending(preflight), "final preflight retains pending state")
    _require(not _contains_pending(recipe), "final recipe retains pending state")
    _require(not _contains_pending(suite), "final suite retains pending state")
    _require(
        not _contains_pending(capture_binding),
        "capture binding retains pending state",
    )
    v1_root = capture_root.parent / "native_sparse_f15_v1"
    v2_root = capture_root.parent / "native_sparse_f15_v2"
    for historical_root in (v1_root, v2_root):
        _require(
            (historical_root / "manifest.json").is_file()
            and (historical_root / "rgb_frames/frame_000000.png").is_file(),
            f"historical canary evidence is missing: {historical_root.name}",
        )
    ledger = {
        "schema": "avengine_strict_two_human_canary_final_gate_v1",
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": "One research sparse f15 canary passed; formal scene count remains zero and no remaining-scene expansion is claimed.",
        "episode_id": EPISODE_ID,
        "formal_scene_count": 0,
        "strict_two_human_qualified_scene_count": 0,
        "expansion_authorized_by_this_ledger": False,
        "capture_process_exit_code": capture_exit_code,
        "gpu": {"physical_index": 1, "uuid": GPU1_UUID, "forbidden_indices_used": []},
        "projection": {
            "target_mouth_source_slot_id": "source1",
            "target_mouth_xy_fraction": target_mouth_xy,
            "safe_fraction_open_interval": [0.05, 0.95],
            "status": "pass",
        },
        "pixels": {"status": "pass", "per_instance": capture["pixels"]},
        "runtime_assets": {
            "status": "pass",
            "required_live_gates": [
                "stable_actor_tag",
                "exact_blueprint_class",
                "skeletal_mesh",
                "skeleton",
                "standing_idle",
                "native_actor_root_plus_declared_profile_offset",
            ],
            "readbacks": _record(capture_root / "runtime_asset_readbacks.json", "live_runtime_asset_readbacks"),
        },
        "acoustics": {
            "status": "pass",
            "pair": "human_speech|silent_human",
            "active_source_slots": ["source1"],
            "silent_source_slots": ["source2"],
            "speech_frame_window_inclusive": [7, 31],
            "channel_count": 2,
            "sample_rate_hz": 16000,
            "sample_count": 80000,
        },
        "fact_and_request_binding": {
            "status": "pass",
            "facts": _record(facts_path, "authoritative_fact"),
            "requests": _record(requests_path, "capture_requests"),
            "suite": _record(suite_path, "finalized_suite"),
            "capture_binding": _record(
                capture_binding_path, "capture_binding_receipt"
            ),
        },
        "fact_protocol_boundary": {
            "status": "sparse_capture_not_promoted_to_full_episode_pixel_truth",
            "fact_pixel_truth": facts["visibility"]["pixel_truth"],
            "reason": "Fact v1 pixel truth requires all 75 frames; this gate captured only f15.",
        },
        "history": {
            "native_sparse_f15_v1": {
                "status": "rejected",
                "reason": "male head and mouth were cropped above the frame",
                "target_bbox_xyxy_px": [453, 0, 1061, 720],
                "manifest": _record(
                    v1_root / "manifest.json", "rejected_v1_manifest"
                ),
                "rgb": _record(
                    v1_root / "rgb_frames/frame_000000.png", "rejected_v1_rgb"
                ),
            },
            "native_sparse_f15_v2": {
                "status": "superseded_visual_pass",
                "reason": "visual geometry passed but lacked exact live runtime asset readbacks",
                "manifest": _record(
                    v2_root / "manifest.json", "superseded_v2_manifest"
                ),
                "rgb": _record(
                    v2_root / "rgb_frames/frame_000000.png", "superseded_v2_rgb"
                ),
            },
            "native_sparse_f15_v3_instrumented": {"status": "pass"},
        },
    }
    ledger_path = output / "final_gate_ledger.json"
    write_json(ledger_path, ledger)
    return ledger_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--audio-plan-root", type=Path, required=True)
    parser.add_argument("--audio-batch-root", type=Path, required=True)
    parser.add_argument("--rir-cache-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--controlled-registry", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--source-fact", type=Path, required=True)
    parser.add_argument("--capture-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ledger = finalize(
        recipe_root=args.recipe_root.resolve(),
        audio_plan_root=args.audio_plan_root.resolve(),
        audio_batch_root=args.audio_batch_root.resolve(),
        rir_cache_root=args.rir_cache_root.resolve(),
        capture_root=args.capture_root.resolve(),
        controlled_registry_path=args.controlled_registry.resolve(),
        runtime_registry_path=args.runtime_registry.resolve(),
        source_fact_path=args.source_fact.resolve(),
        capture_exit_code=args.capture_exit_code,
        output=args.output.resolve(),
    )
    print(f"STRICT_TWO_HUMAN_FINAL_GATE_OK ledger={ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
