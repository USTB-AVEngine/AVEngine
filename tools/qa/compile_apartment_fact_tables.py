#!/usr/bin/env python3
"""Compile per-episode QA fact tables for the asset-bound Apartment batch.

Joins the frozen unique-1000 trajectory bank, its RIR job plan listener, the
semantic-v2 asset-bound binaural batch, the source-asset runtime registry and
the fixed-apartment anchor library into one hash-bound fact table per episode
(schema ``avengine_qa_fact_table_v1``). Nothing is re-simulated; every
episode output is validated against the repository JSON schema before it is
retained, and the batch refuses to complete unless every RIR plan job
position matches the bank emitter paths it claims to serve.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import (  # noqa: E402
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.qa.fact_table import (  # noqa: E402
    QAFactTableError,
    compile_episode_fact_table,
)

INDEX_SCHEMA = "avengine_qa_fact_table_index_v1"
STATS_SCHEMA = "avengine_qa_fact_table_stats_v1"
DELIVERY_SCHEMA = "avengine_m7_asset_bound_binaural_batch_delivery_v1"
PLAN_POSITION_TOLERANCE_M = 1.0e-12
PLAN_ORIENTATION_TOLERANCE = 1.0e-12
AZIMUTH_SECTOR_DEG = 15.0
DISTANCE_BIN_M = 0.5


class FactTableBatchError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FactTableBatchError(message)


def _provenance_record(role: str, path: Path) -> dict[str, Any]:
    return {"role": role, "path": str(path), "sha256": sha256_file(path)}


def _check_plan_positions_against_bank(
    plan: Mapping[str, Any], episodes_by_id: Mapping[str, Mapping[str, Any]]
) -> int:
    checked = 0
    for job in plan["jobs"]:
        position = np.asarray(job["source_position_m"], dtype=np.float64)
        for use in job["uses"]:
            episode = episodes_by_id.get(use["episode_id"])
            _require(
                episode is not None,
                f"plan job {job['job_id']} references unknown episode "
                f"{use['episode_id']!r}",
            )
            path = episode["source_center_paths_m"][use["source_slot_id"]]
            bank_position = np.asarray(path[use["frame_index"]], dtype=np.float64)
            _require(
                bool(np.all(np.abs(bank_position - position) <= PLAN_POSITION_TOLERANCE_M)),
                f"plan job {job['job_id']} position disagrees with the bank at "
                f"{use['episode_id']}/{use['source_slot_id']}/frame "
                f"{use['frame_index']}",
            )
            checked += 1
    return checked


def _sensor_rig_listener_pose_series(
    sensor_rig_trajectory: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    frames = sensor_rig_trajectory.get("frames")
    _require(isinstance(frames, list) and frames, "sensor rig trajectory has no frames")
    try:
        positions = np.asarray(
            [frame["world_from_rig"]["translation_m"] for frame in frames],
            dtype=np.float64,
        )
        rotations_xyzw = np.asarray(
            [frame["world_from_rig"]["rotation_xyzw"] for frame in frames],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FactTableBatchError(
            "sensor rig trajectory frames do not carry numeric world_from_rig poses"
        ) from error
    _require(
        positions.shape == (len(frames), 3)
        and rotations_xyzw.shape == (len(frames), 4)
        and np.all(np.isfinite(positions))
        and np.all(np.isfinite(rotations_xyzw)),
        "sensor rig trajectory frames have invalid pose shapes or values",
    )
    norms = np.linalg.norm(rotations_xyzw, axis=1)
    _require(
        bool(np.all(np.abs(norms - 1.0) <= PLAN_ORIENTATION_TOLERANCE)),
        "sensor rig trajectory rotations are not unit quaternions",
    )
    orientations_wxyz = rotations_xyzw[:, (3, 0, 1, 2)]
    return positions, orientations_wxyz


def _listener_pose_authority(
    plan: Mapping[str, Any],
    sensor_rig_trajectory: Mapping[str, Any] | None,
) -> tuple[list[float], list[float]]:
    if sensor_rig_trajectory is not None:
        positions, orientations = _sensor_rig_listener_pose_series(
            sensor_rig_trajectory
        )
        return positions[0].tolist(), orientations[0].tolist()
    position = plan.get("listener_position_m")
    orientation = plan.get("listener_orientation_wxyz")
    _require(
        isinstance(position, list) and isinstance(orientation, list),
        "fixed-listener RIR plan lacks its top-level listener pose",
    )
    return list(position), list(orientation)


def _check_plan_listener_poses_against_sensor_rig(
    plan: Mapping[str, Any],
    sensor_rig_trajectory: Mapping[str, Any] | None,
) -> int:
    if sensor_rig_trajectory is None:
        return 0
    positions, orientations = _sensor_rig_listener_pose_series(
        sensor_rig_trajectory
    )
    checked = 0
    for job in plan["jobs"]:
        try:
            job_position = np.asarray(job["listener_position_m"], dtype=np.float64)
            job_orientation = np.asarray(
                job["listener_orientation_wxyz"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FactTableBatchError(
                f"plan job {job.get('job_id')!r} lacks a numeric listener pose"
            ) from error
        _require(
            job_position.shape == (3,) and job_orientation.shape == (4,),
            f"plan job {job.get('job_id')!r} listener pose has an invalid shape",
        )
        for use in job["uses"]:
            frame_index = use["frame_index"]
            _require(
                isinstance(frame_index, int) and 0 <= frame_index < len(positions),
                f"plan job {job['job_id']} references an invalid Listener frame",
            )
            _require(
                bool(
                    np.all(
                        np.abs(job_position - positions[frame_index])
                        <= PLAN_POSITION_TOLERANCE_M
                    )
                ),
                f"plan job {job['job_id']} Listener position disagrees with "
                f"SensorRigTrajectory frame {frame_index}",
            )
            dot = abs(float(np.dot(job_orientation, orientations[frame_index])))
            _require(
                abs(dot - 1.0) <= PLAN_ORIENTATION_TOLERANCE,
                f"plan job {job['job_id']} Listener orientation disagrees with "
                f"SensorRigTrajectory frame {frame_index}",
            )
            checked += 1
    return checked


def _resolve_dry_variants(
    sample: Mapping[str, Any], dry_library: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    refs = sample.get("dry_variant_ids_by_source_slot")
    _require(isinstance(refs, Mapping), "sample lacks dry_variant_ids_by_source_slot")
    resolved: dict[str, Mapping[str, Any]] = {}
    for slot_id, ref in refs.items():
        asset_id = ref.get("asset_id")
        variant_index = ref.get("variant_index")
        _require(
            asset_id == sample["asset_ids_by_source_slot"].get(slot_id),
            f"dry variant asset for {slot_id} disagrees with the sample asset",
        )
        asset_entry = dry_library["assets"].get(asset_id)
        _require(asset_entry is not None, f"dry library lacks asset {asset_id!r}")
        variants = asset_entry.get("variants", [])
        _require(
            isinstance(variant_index, int) and 0 <= variant_index < len(variants),
            f"dry variant index {variant_index!r} is out of range for {asset_id!r}",
        )
        resolved[slot_id] = variants[variant_index]
    return resolved


def _azimuth_sector_histogram(azimuths_deg: np.ndarray) -> dict[str, int]:
    sectors = np.floor((azimuths_deg + 180.0) / AZIMUTH_SECTOR_DEG).astype(int)
    sectors = np.clip(sectors, 0, int(360.0 / AZIMUTH_SECTOR_DEG) - 1)
    histogram: dict[str, int] = {}
    for sector, count in zip(*np.unique(sectors, return_counts=True)):
        low = -180.0 + float(sector) * AZIMUTH_SECTOR_DEG
        histogram[f"[{low:+.0f},{low + AZIMUTH_SECTOR_DEG:+.0f})"] = int(count)
    return histogram


def _distance_histogram(distances_m: np.ndarray) -> dict[str, int]:
    bins = np.floor(distances_m / DISTANCE_BIN_M).astype(int)
    histogram: dict[str, int] = {}
    for bin_index, count in zip(*np.unique(bins, return_counts=True)):
        low = float(bin_index) * DISTANCE_BIN_M
        histogram[f"[{low:.1f},{low + DISTANCE_BIN_M:.1f})"] = int(count)
    return histogram


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        required=True,
        help="Directory holding trajectory_bank.json and rir_job_plan.json",
    )
    parser.add_argument(
        "--audio-batch-dir",
        type=Path,
        required=True,
        help="Asset-bound binaural batch directory (delivery/samples/episodes)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
    )
    parser.add_argument(
        "--anchor-library",
        type=Path,
        default=REPOSITORY / "examples/m6x/fixed_apartment/anchor_library.json",
    )
    parser.add_argument(
        "--room-capsule",
        type=Path,
        default=REPOSITORY / "examples/m6x/fixed_apartment/room_capsule.json",
    )
    parser.add_argument(
        "--m1-request",
        type=Path,
        default=REPOSITORY
        / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json",
        help="Camera calibration authority; must agree with the RIR plan listener",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPOSITORY / "schemas/avengine_qa_fact_table_v1.schema.json",
    )
    parser.add_argument(
        "--sensor-rig-trajectory",
        type=Path,
        help=(
            "Optional SensorRigTrajectory v1 shared by the selected episodes; "
            "frame 0 must agree with the historical fixed RIR Listener pose"
        ),
    )
    parser.add_argument(
        "--pixel-visibility-truth",
        type=Path,
        help=(
            "Optional native pixel-visibility truth for a single selected "
            "Episode; it is schema-, instance-, frame-, resolution- and "
            "SensorRig-pose-bound by the Fact compiler"
        ),
    )
    parser.add_argument("--limit", type=int, help="Compile only the first N episodes")
    parser.add_argument(
        "--intermittent-batch",
        type=Path,
        help=(
            "Intermittent audio batch directory; compiles fact tables for its "
            "episode subset with declared multi-window sound events bound to "
            "the gated mixtures"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import jsonschema  # noqa: PLC0415

    started = time.monotonic()
    plan_dir = args.plan_dir.resolve()
    audio_dir = args.audio_batch_dir.resolve()

    bank_path = plan_dir / "trajectory_bank.json"
    plan_path = plan_dir / "rir_job_plan.json"
    samples_path = audio_dir / "samples.json"
    episodes_path = audio_dir / "episodes.json"
    dry_path = audio_dir / "dry_audio_variants.json"
    delivery_path = audio_dir / "delivery.json"

    bank = load_json(bank_path)
    plan = load_json(plan_path)
    samples = load_json(samples_path)
    episode_cache_index = load_json(episodes_path)
    dry_library = load_json(dry_path)
    delivery = load_json(delivery_path)
    registry = load_json(args.registry)
    anchor_library = load_json(args.anchor_library)
    room_capsule = load_json(args.room_capsule)
    m1_request = load_json(args.m1_request)
    schema_document = load_json(args.schema)
    sensor_rig_trajectory = (
        None
        if args.sensor_rig_trajectory is None
        else load_json(args.sensor_rig_trajectory.resolve())
    )
    pixel_visibility_truth = (
        None
        if args.pixel_visibility_truth is None
        else load_json(args.pixel_visibility_truth.resolve())
    )
    validator = jsonschema.Draft202012Validator(schema_document)

    listener_position_m, listener_orientation_wxyz = _listener_pose_authority(
        plan, sensor_rig_trajectory
    )

    camera_rig = m1_request["primary_camera_rig"]
    calibration = camera_rig["shared_calibration"]
    world_from_rig = camera_rig["world_from_rig"]
    rig_translation = np.asarray(world_from_rig["translation_m"], dtype=np.float64)
    rig_xyzw = np.asarray(world_from_rig["rotation_xyzw"], dtype=np.float64)
    plan_position = np.asarray(listener_position_m, dtype=np.float64)
    plan_wxyz = np.asarray(listener_orientation_wxyz, dtype=np.float64)
    _require(
        bool(np.all(np.abs(rig_translation - plan_position) <= 1.0e-9)),
        "M1 camera rig translation disagrees with the RIR plan listener position",
    )
    rig_wxyz = np.asarray(
        [rig_xyzw[3], rig_xyzw[0], rig_xyzw[1], rig_xyzw[2]], dtype=np.float64
    )
    _require(
        bool(
            np.all(np.abs(rig_wxyz - plan_wxyz) <= 1.0e-9)
            or np.all(np.abs(rig_wxyz + plan_wxyz) <= 1.0e-9)
        ),
        "M1 camera rig rotation disagrees with the RIR plan listener orientation",
    )
    camera = {
        "hfov_degrees": float(calibration["hfov_degrees"]),
        "resolution_hw": [int(value) for value in calibration["resolution_hw"]],
    }

    _require(
        delivery.get("schema") == DELIVERY_SCHEMA and delivery.get("status") == "pass",
        "audio batch delivery is not a passing asset-bound binaural delivery",
    )
    bank_episodes = bank["episodes"]
    _require(
        delivery.get("episode_count") == len(bank_episodes),
        "audio batch episode count disagrees with the trajectory bank",
    )

    episodes_by_id = {episode["episode_id"]: episode for episode in bank_episodes}
    samples_by_episode: dict[str, Mapping[str, Any]] = {}
    for sample in samples["samples"]:
        _require(
            sample["episode_id"] not in samples_by_episode,
            f"duplicate sample for episode {sample['episode_id']!r}",
        )
        samples_by_episode[sample["episode_id"]] = sample
    _require(
        set(samples_by_episode) == set(episodes_by_id),
        "sample episode ids do not match the trajectory bank episode ids",
    )
    cache_identity_by_episode: dict[str, str] = {}
    for entry in episode_cache_index["episodes"]:
        cache_identity_by_episode[entry["episode_id"]] = entry["rir_cache"][
            "cache_request_identity_sha256"
        ]
    _require(
        set(cache_identity_by_episode) == set(episodes_by_id),
        "episode cache index does not match the trajectory bank episode ids",
    )

    plan_position_checks = _check_plan_positions_against_bank(plan, episodes_by_id)
    plan_listener_pose_checks = _check_plan_listener_poses_against_sensor_rig(
        plan, sensor_rig_trajectory
    )

    provenance_inputs = [
        _provenance_record("trajectory_bank", bank_path),
        _provenance_record("rir_job_plan", plan_path),
        _provenance_record("audio_batch_samples", samples_path),
        _provenance_record("audio_batch_episode_cache_index", episodes_path),
        _provenance_record("dry_audio_variant_library", dry_path),
        _provenance_record("source_asset_runtime_registry", args.registry.resolve()),
        _provenance_record("anchor_library", args.anchor_library.resolve()),
        _provenance_record("room_capsule", args.room_capsule.resolve()),
        _provenance_record("m1_capture_request", args.m1_request.resolve()),
    ]
    if args.sensor_rig_trajectory is not None:
        provenance_inputs.append(
            _provenance_record(
                "sensor_rig_trajectory", args.sensor_rig_trajectory.resolve()
            )
        )
    if args.pixel_visibility_truth is not None:
        provenance_inputs.append(
            _provenance_record(
                "native_pixel_visibility_truth",
                args.pixel_visibility_truth.resolve(),
            )
        )
    room = {
        "room_capsule_id": room_capsule["room_capsule_id"],
        "revision": room_capsule["revision"],
    }
    anchors = anchor_library["anchors"]
    bank_header = {
        key: bank[key]
        for key in (
            "frame_count",
            "frame_rate_hz",
            "seconds_per_episode",
            "source_slots",
        )
    }

    intermittent_by_episode: dict[str, Mapping[str, Any]] | None = None
    intermittent_dry_library: Mapping[str, Any] | None = None
    if args.intermittent_batch is not None:
        intermittent_root = args.intermittent_batch.resolve()
        intermittent_manifest = load_json(
            intermittent_root / "intermittent_batch_manifest.json"
        )
        _require(
            intermittent_manifest.get("schema")
            == "avengine_qa_intermittent_audio_batch_v1",
            "intermittent batch manifest has an unexpected schema",
        )
        intermittent_by_episode = {
            sample["episode_id"]: sample
            for sample in intermittent_manifest["samples"]
        }
        unknown = sorted(set(intermittent_by_episode) - set(episodes_by_id))
        _require(
            not unknown,
            f"intermittent batch references unknown episodes: {unknown[:3]}",
        )
        intermittent_dry_library = {
            "assets": intermittent_manifest["dry_audio_variants"]
        }
        provenance_inputs.append(
            _provenance_record(
                "intermittent_batch_manifest",
                intermittent_root / "intermittent_batch_manifest.json",
            )
        )

    output = args.output.resolve()
    facts_dir = output / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)

    if intermittent_by_episode is not None:
        selected = [
            episode
            for episode in bank_episodes
            if episode["episode_id"] in intermittent_by_episode
        ]
    else:
        selected = bank_episodes
    if args.limit is not None:
        selected = selected[: args.limit]
    _require(
        pixel_visibility_truth is None or len(selected) == 1,
        "--pixel-visibility-truth requires exactly one selected Episode",
    )
    entries: list[dict[str, Any]] = []
    azimuth_values: list[np.ndarray] = []
    distance_values: list[np.ndarray] = []
    facing_vs_travel_abs_deg: list[np.ndarray] = []
    moving_frames = 0
    instance_frames = 0
    front_crossing_instances = 0
    species_pair_counts: dict[str, int] = {}
    motion_case_counts: dict[str, int] = {}

    for ordinal, bank_episode in enumerate(selected):
        episode_id = bank_episode["episode_id"]
        declared_events_by_slot = None
        if intermittent_by_episode is not None:
            gated_sample = intermittent_by_episode[episode_id]
            sample = {
                "episode_id": episode_id,
                "asset_ids_by_source_slot": gated_sample["asset_ids_by_source_slot"],
                "dry_variant_ids_by_source_slot": {
                    slot: {"asset_id": asset_id, "variant_index": 0}
                    for slot, asset_id in gated_sample[
                        "asset_ids_by_source_slot"
                    ].items()
                },
                "audio": gated_sample["audio"],
            }
            episode_dry_library = intermittent_dry_library
            declared_events_by_slot = gated_sample["events_by_source_slot"]
        else:
            sample = samples_by_episode[episode_id]
            episode_dry_library = dry_library
        try:
            fact_table = compile_episode_fact_table(
                bank_header=bank_header,
                bank_episode=bank_episode,
                listener_position_m=listener_position_m,
                listener_orientation_wxyz=listener_orientation_wxyz,
                sample_entry=sample,
                dry_variants_by_slot=_resolve_dry_variants(sample, episode_dry_library),
                registry=registry,
                anchors=anchors,
                room=room,
                camera=camera,
                rir_cache_request_identity_sha256=cache_identity_by_episode[episode_id],
                provenance_inputs=provenance_inputs,
                declared_events_by_slot=declared_events_by_slot,
                sensor_rig_trajectory=sensor_rig_trajectory,
                pixel_visibility_truth=pixel_visibility_truth,
            )
        except QAFactTableError as error:
            raise FactTableBatchError(f"{episode_id}: {error}") from error
        schema_errors = sorted(
            validator.iter_errors(fact_table), key=lambda err: list(err.absolute_path)
        )
        _require(
            not schema_errors,
            f"{episode_id}: fact table violates the schema: "
            f"{schema_errors[0].message if schema_errors else ''}",
        )

        fact_path = facts_dir / f"{episode_id}.json"
        write_json(fact_path, fact_table)
        entries.append(
            {
                "episode_id": episode_id,
                "ordinal": ordinal,
                "fact_table": file_record(fact_path, relative_to=output),
            }
        )

        motion_case = fact_table["motion_case"]
        motion_case_counts[motion_case] = motion_case_counts.get(motion_case, 0) + 1
        species_pair = "__".join(
            instance["species_id"] for instance in fact_table["instances"]
        )
        species_pair_counts[species_pair] = species_pair_counts.get(species_pair, 0) + 1
        for track in fact_table["tracks"]["instances"].values():
            azimuth = np.asarray(track["doa"]["azimuth_deg"], dtype=np.float64)
            azimuth_values.append(azimuth)
            distance_values.append(
                np.asarray(track["doa"]["distance_m"], dtype=np.float64)
            )
            moving_frames += sum(1 for value in track["moving"] if value)
            instance_frames += len(track["moving"])
            signs = np.sign(azimuth)
            if np.any(signs > 0) and np.any(signs < 0):
                front_crossing_instances += 1
            facing = track["facing_yaw_deg"]
            if facing is not None:
                roots = np.asarray(track["root_position_m"], dtype=np.float64)
                steps = np.diff(roots, axis=0)
                stepping = np.asarray(track["moving"][:-1], dtype=bool) & (
                    np.linalg.norm(steps, axis=1) > 0.0
                )
                if np.any(stepping):
                    travel = np.degrees(
                        np.arctan2(-steps[stepping, 0], -steps[stepping, 2])
                    )
                    delta = (
                        np.asarray(facing[:-1], dtype=np.float64)[stepping] - travel
                    )
                    delta = np.mod(delta + 180.0, 360.0) - 180.0
                    facing_vs_travel_abs_deg.append(np.abs(delta))

    all_azimuth = np.concatenate(azimuth_values)
    all_distance = np.concatenate(distance_values)
    stats = {
        "schema": STATS_SCHEMA,
        "qualification_claim": False,
        "claim_boundary": (
            "Descriptive statistics over compiled fact tables; no dataset "
            "admission and no acoustic or visual truth claim beyond the "
            "frozen inputs"
        ),
        "episode_count": len(entries),
        "episode_total_available": (
            len(intermittent_by_episode)
            if intermittent_by_episode is not None
            else len(bank_episodes)
        ),
        "realization": (
            "intermittent_declared_windows"
            if intermittent_by_episode is not None
            else "continuous"
        ),
        "complete": len(entries) == (
            len(intermittent_by_episode)
            if intermittent_by_episode is not None
            else len(bank_episodes)
        ),
        "plan_position_checks": plan_position_checks,
        "plan_listener_pose_checks": plan_listener_pose_checks,
        "motion_case_counts": motion_case_counts,
        "species_pair_counts": species_pair_counts,
        "instance_frames": instance_frames,
        "moving_frame_fraction": (
            moving_frames / instance_frames if instance_frames else None
        ),
        "azimuth_sector_histogram_deg": _azimuth_sector_histogram(all_azimuth),
        "azimuth_sign_crossing_instance_count": front_crossing_instances,
        "facing_vs_travel_abs_deg": (
            {
                "mean": float(np.mean(np.concatenate(facing_vs_travel_abs_deg))),
                "p95": float(
                    np.percentile(np.concatenate(facing_vs_travel_abs_deg), 95)
                ),
                "max": float(np.max(np.concatenate(facing_vs_travel_abs_deg))),
                "step_count": int(
                    sum(len(values) for values in facing_vs_travel_abs_deg)
                ),
            }
            if facing_vs_travel_abs_deg
            else None
        ),
        "distance_histogram_m": _distance_histogram(all_distance),
        "distance_min_m": float(np.min(all_distance)),
        "distance_max_m": float(np.max(all_distance)),
        "inputs": provenance_inputs,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    write_json(output / "stats_report.json", stats)

    index = {
        "schema": INDEX_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "Index of compiled per-episode QA fact tables; grants no dataset "
            "admission"
        ),
        "fact_table_schema": "avengine_qa_fact_table_v1",
        "episode_count": len(entries),
        "realization": stats["realization"],
        "complete": stats["complete"],
        "inputs": provenance_inputs,
        "episodes": entries,
    }
    write_json(output / "fact_table_index.json", index)
    print(
        f"QA_FACT_TABLES_OK output={output} episodes={len(entries)} "
        f"plan_checks={plan_position_checks} "
        f"listener_checks={plan_listener_pose_checks} "
        f"moving_frame_fraction={stats['moving_frame_fraction']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
