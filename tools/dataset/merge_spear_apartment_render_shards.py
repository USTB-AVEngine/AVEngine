#!/usr/bin/env python3
"""Merge independently rendered SPEAR Apartment shards without copying media."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from avengine.contracts.json_io import load_json, write_json
from avengine.optional_backends.spear_apartment import contiguous_episode_shard


def _load_runner() -> Any:
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "rooms/run_spear_apartment_canary.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_spear_apartment_canary_for_shard_merge", runner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the SPEAR Apartment runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenario_map(plan: Mapping[str, Any], *, owner: str) -> dict[str, Mapping[str, Any]]:
    scenarios = plan.get("scenarios")
    if not isinstance(scenarios, list):
        raise RuntimeError(f"{owner} has no scenario list")
    result: dict[str, Mapping[str, Any]] = {}
    for value in scenarios:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{owner} scenario is invalid")
        scenario_id = value.get("scenario_id")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in result
        ):
            raise RuntimeError(f"{owner} scenario IDs are invalid")
        result[scenario_id] = value
    return result


def _expected_episode_ids(
    *, visual_bundle_root: Path, full_suite_plan: Mapping[str, Any]
) -> tuple[str, ...]:
    manifest = load_json(visual_bundle_root / "manifest.json")
    values = manifest.get("episode_ids")
    if (
        manifest.get("status") != "pass"
        or not isinstance(values, list)
        or not values
        or manifest.get("episode_count") != len(values)
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise RuntimeError("visual bundle episode declaration is invalid")
    episode_ids = tuple(values)
    full_scenarios = _scenario_map(full_suite_plan, owner="full suite plan")
    if set(full_scenarios) != set(episode_ids):
        raise RuntimeError("full suite plan differs from the visual bundle")
    return episode_ids


def _collect_scenario_sources(
    *,
    expected_episode_ids: Sequence[str],
    full_suite_plan: Mapping[str, Any],
    shard_roots: Sequence[Path],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Select the first passing shard for each expected scenario."""

    expected = set(expected_episode_ids)
    full_scenarios = _scenario_map(full_suite_plan, owner="full suite plan")
    selected: dict[str, Path] = {}
    occurrence_counts = {episode_id: 0 for episode_id in expected_episode_ids}
    shard_records = []
    execution_partitions: list[dict[str, Any]] = []
    for shard_root in shard_roots:
        shard_plan = load_json(shard_root / "suite_execution_plan.json")
        shard_scenarios = _scenario_map(
            shard_plan, owner=f"shard plan {shard_root}"
        )
        raw_partition = shard_plan.get("execution_partition")
        if raw_partition is not None:
            if (
                not isinstance(raw_partition, Mapping)
                or raw_partition.get("kind")
                != "contiguous_manifest_episode_ids"
            ):
                raise RuntimeError(
                    f"shard execution partition is invalid: {shard_root}"
                )
            try:
                partition_ids = contiguous_episode_shard(
                    expected_episode_ids,
                    shard_count=raw_partition.get("shard_count"),
                    shard_index=raw_partition.get("shard_index"),
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"shard execution partition is invalid: {shard_root}"
                ) from exc
            if (
                tuple(shard_scenarios) != partition_ids
                or raw_partition.get("total_episode_count")
                != len(expected_episode_ids)
                or raw_partition.get("selected_episode_count")
                != len(partition_ids)
                or raw_partition.get("first_episode_id") != partition_ids[0]
                or raw_partition.get("last_episode_id") != partition_ids[-1]
            ):
                raise RuntimeError(
                    f"shard plan differs from its exact partition: {shard_root}"
                )
            execution_partitions.append(dict(raw_partition))
        if not set(shard_scenarios) <= expected:
            raise RuntimeError(f"shard contains unexpected scenarios: {shard_root}")
        completed = 0
        for scenario_id, shard_scenario in shard_scenarios.items():
            if shard_scenario != full_scenarios[scenario_id]:
                raise RuntimeError(
                    f"shard scenario differs from full plan: {scenario_id}"
                )
            evidence_path = shard_root / scenario_id / "evidence.json"
            if not evidence_path.is_file():
                continue
            evidence = load_json(evidence_path)
            if (
                evidence.get("status") != "pass"
                or evidence.get("scenario_id") != scenario_id
            ):
                raise RuntimeError(
                    f"shard scenario evidence is invalid: {scenario_id}"
                )
            occurrence_counts[scenario_id] += 1
            selected.setdefault(scenario_id, shard_root / scenario_id)
            completed += 1
        shard_records.append(
            {
                "shard_root": str(shard_root),
                "declared_scenario_count": len(shard_scenarios),
                "completed_scenario_count": completed,
                "has_root_evidence": (
                    shard_root / "evidence.json"
                ).is_file(),
            }
        )
    if execution_partitions:
        shard_counts = {
            value["shard_count"] for value in execution_partitions
        }
        shard_indices = {
            value["shard_index"] for value in execution_partitions
        }
        if (
            len(execution_partitions) != len(shard_roots)
            or len(shard_counts) != 1
            or next(iter(shard_counts)) != len(shard_roots)
            or shard_indices != set(range(len(shard_roots)))
        ):
            raise RuntimeError(
                "exact shard execution partitions are incomplete or overlap"
            )
    missing = sorted(expected - set(selected))
    if missing:
        raise RuntimeError(
            f"render shards are missing {len(missing)} scenarios; first={missing[0]}"
        )
    duplicates = {
        episode_id: count
        for episode_id, count in occurrence_counts.items()
        if count > 1
    }
    return selected, {
        "shards": shard_records,
        "execution_partitions": sorted(
            execution_partitions, key=lambda value: value["shard_index"]
        ),
        "duplicate_scenario_count": len(duplicates),
        "duplicate_occurrence_count": sum(
            count - 1 for count in duplicates.values()
        ),
    }


def _base_root_evidence(shard_roots: Sequence[Path]) -> dict[str, Any]:
    values = [
        load_json(root / "evidence.json")
        for root in shard_roots
        if (root / "evidence.json").is_file()
    ]
    if not values:
        raise RuntimeError(
            "at least one completed shard root evidence is required"
        )
    base = values[0]
    if base.get("status") != "pass":
        raise RuntimeError("completed shard root evidence did not pass")
    for value in values[1:]:
        if (
            value.get("status") != "pass"
            or value.get("native_map") != base.get("native_map")
            or value.get("lighting_profile") != base.get("lighting_profile")
            or value.get("authority") != base.get("authority")
        ):
            raise RuntimeError("completed shard root evidence disagrees")
    return deepcopy(base)


def merge_shards(
    *,
    visual_bundle_root: Path,
    full_suite_plan_path: Path,
    shard_roots: Sequence[Path],
    video_encoder: str,
    verify_workers: int,
    output: Path,
) -> Path:
    started = time.perf_counter()
    visual_bundle_root = visual_bundle_root.resolve()
    full_suite_plan_path = full_suite_plan_path.resolve()
    shard_roots = tuple(path.resolve() for path in shard_roots)
    output = output.resolve()
    if len(shard_roots) < 2:
        raise ValueError("at least two shard roots are required")
    if (
        isinstance(verify_workers, bool)
        or not isinstance(verify_workers, int)
        or not 1 <= verify_workers <= 32
    ):
        raise ValueError("verify_workers must be an integer between 1 and 32")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging: {staging}")

    full_suite_plan = load_json(full_suite_plan_path)
    episode_ids = _expected_episode_ids(
        visual_bundle_root=visual_bundle_root,
        full_suite_plan=full_suite_plan,
    )
    scenario_sources, collection = _collect_scenario_sources(
        expected_episode_ids=episode_ids,
        full_suite_plan=full_suite_plan,
        shard_roots=shard_roots,
    )
    base_evidence = _base_root_evidence(shard_roots)
    runner = _load_runner()

    staging.mkdir(parents=True)
    try:
        hardlinked_file_count = 0
        for episode_id in episode_ids:
            source = scenario_sources[episode_id]
            destination = staging / episode_id
            shutil.copytree(source, destination, copy_function=os.link)
            hardlinked_file_count += sum(
                path.is_file() for path in destination.rglob("*")
            )

        def reopen(episode_id: str) -> dict[str, Any]:
            record = runner._load_resumable_scenario_record(
                output_root=staging,
                scenario={"scenario_id": episode_id},
                video_encoder=video_encoder,
            )
            if record is None:
                raise RuntimeError(
                    f"merged scenario could not be reopened: {episode_id}"
                )
            return record

        with ThreadPoolExecutor(max_workers=verify_workers) as executor:
            scenario_records = list(executor.map(reopen, episode_ids))

        startup_by_shard: dict[Path, Mapping[str, Any]] = {}
        for shard_root in shard_roots:
            path = shard_root / "component_frame_startup_evidence.json"
            if path.is_file():
                value = load_json(path)
                if not isinstance(value, Mapping):
                    raise RuntimeError(
                        f"shard startup evidence is invalid: {shard_root}"
                    )
                startup_by_shard[shard_root] = value
        startup_evidence = {
            episode_id: startup_by_shard[
                scenario_sources[episode_id].parent
            ][episode_id]
            for episode_id in episode_ids
            if (
                scenario_sources[episode_id].parent in startup_by_shard
                and episode_id
                in startup_by_shard[scenario_sources[episode_id].parent]
            )
        }
        if set(startup_evidence) != set(episode_ids):
            raise RuntimeError("merged component-frame startup evidence is incomplete")

        graphics_adapters = sorted(
            {
                int(record["timing"]["encoder_gpu"])
                for record in scenario_records
                if isinstance(record.get("timing", {}).get("encoder_gpu"), int)
            }
        )
        plan_start_times = [
            (root / "suite_execution_plan.json").stat().st_mtime
            for root in shard_roots
        ]
        evidence_end_times = [
            (root / "evidence.json").stat().st_mtime
            for root in shard_roots
            if (root / "evidence.json").is_file()
        ]
        timing = {
            "schema": runner.TIMING_SCHEMA,
            "status": "pass",
            "clock": "time.perf_counter plus filesystem wall-clock envelope",
            "measurement_scope": (
                "parallel independent SPEAR shards followed by hardlink merge "
                "and complete media readback"
            ),
            "execution_mode": "parallel_gpu_shards",
            "video_encoder": video_encoder,
            "encoder_gpus": graphics_adapters,
            "scenario_count": len(scenario_records),
            "media_verify_workers": verify_workers,
            "merge_wall_seconds": time.perf_counter() - started,
            "observed_parallel_wall_seconds_lower_bound": (
                max(evidence_end_times) - min(plan_start_times)
            ),
            "scenario_timings": {
                record["scenario_id"]: record["timing"]
                for record in scenario_records
            },
            "shards": collection["shards"],
        }
        runtime = dict(base_evidence.get("runtime", {}))
        runtime.pop("execution_partition", None)
        runtime["graphics_adapters_used"] = graphics_adapters
        runtime["render_shard_count"] = len(shard_roots)
        runtime["execution_partitions"] = collection["execution_partitions"]
        evidence = base_evidence
        evidence["scenarios"] = scenario_records
        evidence["runtime"] = runtime
        evidence["timing"] = timing
        evidence["merge"] = {
            "status": "pass",
            "media_storage": "hardlinks_to_verified_shard_files",
            "scene_copy_count": 0,
            "hardlinked_file_count": hardlinked_file_count,
            "media_verify_workers": verify_workers,
            **collection,
        }
        os.link(
            full_suite_plan_path,
            staging / "suite_execution_plan.json",
        )
        write_json(
            staging / "component_frame_startup_evidence.json",
            startup_evidence,
        )
        write_json(staging / "timing.json", timing)
        write_json(staging / "evidence.json", evidence)
        write_json(
            staging / "merge_report.json",
            {
                "schema": "avengine_m7_spear_apartment_shard_merge_v1",
                "status": "pass",
                "scenario_count": len(scenario_records),
                "all_media_reopened": True,
                "scene_copy_count": 0,
                "hardlinked_file_count": hardlinked_file_count,
                "encoder_gpus": graphics_adapters,
                "media_verify_workers": verify_workers,
                **collection,
            },
        )
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-bundle-root", type=Path, required=True)
    parser.add_argument("--full-suite-plan", type=Path, required=True)
    parser.add_argument(
        "--shard-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat in preferred duplicate-selection order.",
    )
    parser.add_argument(
        "--video-encoder",
        choices=("libx264", "h264_nvenc"),
        required=True,
    )
    parser.add_argument("--verify-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = merge_shards(
        visual_bundle_root=args.visual_bundle_root,
        full_suite_plan_path=args.full_suite_plan,
        shard_roots=args.shard_root,
        video_encoder=args.video_encoder,
        verify_workers=args.verify_workers,
        output=args.output,
    )
    print(f"SPEAR_APARTMENT_SHARD_MERGE_OK output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
