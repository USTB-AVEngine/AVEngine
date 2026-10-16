#!/usr/bin/env python3
"""Compile Facts and finalize the native full-occlusion suite after audio render."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import jsonschema


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import write_json  # noqa: E402
from avengine.qa.fact_table import compile_episode_fact_table  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path.resolve()), "sha256": _sha256(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _one(items: list[Mapping[str, Any]], *, owner: str) -> Mapping[str, Any]:
    _require(len(items) == 1, f"{owner} must resolve exactly once")
    return items[0]


def finalize(
    *,
    recipe_root: Path,
    audio_plan_root: Path,
    audio_batch_root: Path,
    rir_cache_root: Path,
    controlled_program_root: Path,
    controlled_registry_path: Path,
    runtime_registry_path: Path,
    source_fact_path: Path,
    output: Path,
) -> dict[str, Path]:
    trajectory_bank_path = recipe_root / "trajectory_bank.json"
    sensor_rig_path = recipe_root / "sensor_rig_trajectory.json"
    pending_suite_path = recipe_root / "suite_execution_plan.pending_fact.json"
    bank = _load(trajectory_bank_path)
    sensor_rig = _load(sensor_rig_path)
    suite = _load(pending_suite_path)
    source_fact = _load(source_fact_path)
    runtime_registry = _load(runtime_registry_path)
    binding = _load(controlled_program_root / "controlled_audio_binding.json")
    program_instance_path = audio_batch_root / "labels/audio_program_instances/v00.json"
    program_instance = _load(program_instance_path)
    samples_path = audio_batch_root / "samples.json"
    episodes_path = audio_batch_root / "episodes.json"
    dry_library_path = audio_batch_root / "dry_audio_variants.json"
    samples = _load(samples_path)
    episodes = _load(episodes_path)
    sample = _one(samples["samples"], owner="audio sample")
    episode_record = _one(episodes["episodes"], owner="audio cache episode")
    episode_id = bank["episodes"][0]["episode_id"]
    _require(sample["episode_id"] == episode_id, "audio/trajectory Episode drift")
    _require(episode_record["episode_id"] == episode_id, "RIR/trajectory Episode drift")

    mixture_name = sample["audio"]["mixture"]["path"]
    mixture_path = audio_batch_root / "audio/binaural" / mixture_name
    _require(mixture_path.is_file(), "authoritative binaural mixture is missing")
    sample_entry = deepcopy(sample)
    sample_entry["audio"]["mixture"]["path"] = str(mixture_path.resolve())
    _require(
        _sha256(mixture_path) == sample_entry["audio"]["mixture"]["audio_sha256"],
        "binaural mixture SHA drift",
    )

    mapped_events = program_instance["mapped_events"]
    declared_events: dict[str, list[dict[str, Any]]] = {"source1": [], "source2": []}
    gains: dict[str, float] = {}
    for event in mapped_events:
        slot = event["source_slot_id"]
        declared_events[slot].append(
            {
                "event_id": event["event_id"],
                "start_sample": event["start_sample"],
                "end_sample_exclusive": event["end_sample_exclusive"],
                "start_tick": event["start_tick"],
                "end_tick_exclusive": event["end_tick_exclusive"],
                "fade_samples": event["fade_samples"],
                "gating": "m6_audio_program_event_window_v1",
            }
        )
        previous = gains.setdefault(slot, float(event["linear_gain"]))
        _require(previous == float(event["linear_gain"]), f"{slot}: mixed event gains")

    controlled = binding["controlled_content"]
    dry_variants: dict[str, dict[str, Any]] = {}
    for slot in ["source1", "source2"]:
        sound_id = controlled[slot]["sound_asset_id"]
        path = Path(binding["sound_audio_paths"][sound_id]).resolve()
        dry_variants[slot] = {
            "variant_index": 0,
            "record": {
                "input": {"path": str(path), "sha256": _sha256(path)},
                "linear_gain": gains[slot],
            },
        }

    provenance_paths = [
        (trajectory_bank_path, "trajectory_bank"),
        (audio_plan_root / "rir_job_plan.json", "rir_job_plan"),
        (samples_path, "audio_batch_samples"),
        (episodes_path, "audio_batch_episode_cache_index"),
        (dry_library_path, "dry_audio_variant_library"),
        (runtime_registry_path, "source_asset_runtime_registry"),
        (controlled_registry_path, "controlled_sound_content_registry"),
        (controlled_program_root / "source_endpoint_registry.json", "source_endpoint_registry"),
        (controlled_program_root / "sound_asset_registry.json", "sound_asset_registry"),
        (controlled_program_root / "audio_program.json", "audio_program"),
        (program_instance_path, "materialized_audio_program_instance"),
        (sensor_rig_path, "sensor_rig_trajectory"),
        (rir_cache_root / "receipt.json", "native_rlr_rir_cache_receipt"),
        (source_fact_path, "retained_source_fact"),
        (pending_suite_path, "retained_source_native_suite_recipe"),
    ]
    facts = compile_episode_fact_table(
        bank_header={key: value for key, value in bank.items() if key != "episodes"},
        bank_episode=bank["episodes"][0],
        listener_position_m=sensor_rig["frames"][0]["world_from_rig"]["translation_m"],
        listener_orientation_wxyz=[
            sensor_rig["frames"][0]["world_from_rig"]["rotation_xyzw"][3],
            *sensor_rig["frames"][0]["world_from_rig"]["rotation_xyzw"][:3],
        ],
        sample_entry=sample_entry,
        dry_variants_by_slot=dry_variants,
        registry=runtime_registry,
        anchors=source_fact["anchors"],
        room=source_fact["room"],
        camera={
            "hfov_degrees": source_fact["visibility"]["hfov_degrees"],
            "resolution_hw": source_fact["visibility"]["resolution_hw"],
        },
        rir_cache_request_identity_sha256=episode_record["rir_cache"][
            "cache_request_identity_sha256"
        ],
        provenance_inputs=[
            _file_record(path, role=role) for path, role in provenance_paths
        ],
        declared_events_by_slot=declared_events,
        sensor_rig_trajectory=sensor_rig,
        controlled_content_by_slot=controlled,
    )
    schema = _load(REPOSITORY / "schemas/avengine_qa_fact_table_v1.schema.json")
    jsonschema.Draft202012Validator(schema).validate(facts)

    output.mkdir(parents=True, exist_ok=False)
    facts_path = output / "facts.json"
    write_json(facts_path, facts)
    scenario = suite["scenarios"][0]
    scenario["scenario_directory"] = str((output / episode_id).resolve())
    scenario["authoritative_capture_request"] = {
        "request_id": f"{episode_id}__native_capture",
        "episode_id": episode_id,
        "scenario_type": "full_occlusion_to_reappearance",
        "target_source_slot_id": "source2",
        "fact_path": str(facts_path.resolve()),
        "fact_sha256": _sha256(facts_path),
    }
    suite["camera_upgrade"]["capture_requests"] = str(
        (output / "native_capture_requests.json").resolve()
    )
    requests = {
        "schema": "avengine_native_pixel_capture_requests_v1",
        "status": "ready",
        "qualification_claim": False,
        "requests": [deepcopy(scenario["authoritative_capture_request"])],
    }
    requests_path = output / "native_capture_requests.json"
    write_json(requests_path, requests)
    suite["camera_upgrade"]["capture_requests_sha256"] = _sha256(requests_path)
    suite_path = output / "suite_execution_plan.json"
    write_json(suite_path, suite)
    receipt = {
        "schema": "avengine_native_full_occlusion_reappearance_finalization_v1",
        "status": "ready_for_native_pixel_capture",
        "qualification_claim": False,
        "episode_id": episode_id,
        "facts": _file_record(facts_path, role="authoritative_fact"),
        "suite": _file_record(suite_path, role="native_suite"),
        "audio": _file_record(mixture_path, role="authoritative_binaural_audio"),
        "sensor_rig_dynamic": facts["listener"]["sensor_rig_trajectory"]["dynamic"],
        "controlled_statement": controlled["source2"],
    }
    write_json(output / "receipt.json", receipt)
    return {"facts": facts_path, "suite": suite_path, "audio": mixture_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--audio-plan-root", type=Path, required=True)
    parser.add_argument("--audio-batch-root", type=Path, required=True)
    parser.add_argument("--rir-cache-root", type=Path, required=True)
    parser.add_argument("--controlled-program-root", type=Path, required=True)
    parser.add_argument("--controlled-registry", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--source-fact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = finalize(
        recipe_root=args.recipe_root.resolve(),
        audio_plan_root=args.audio_plan_root.resolve(),
        audio_batch_root=args.audio_batch_root.resolve(),
        rir_cache_root=args.rir_cache_root.resolve(),
        controlled_program_root=args.controlled_program_root.resolve(),
        controlled_registry_path=args.controlled_registry.resolve(),
        runtime_registry_path=args.runtime_registry.resolve(),
        source_fact_path=args.source_fact.resolve(),
        output=args.output.resolve(),
    )
    print(
        "NATIVE_FULL_OCCLUSION_REAPPEARANCE_FINALIZED "
        f"facts={result['facts']} suite={result['suite']} audio={result['audio']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
