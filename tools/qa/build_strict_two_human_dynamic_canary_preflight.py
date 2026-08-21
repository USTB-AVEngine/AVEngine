#!/usr/bin/env python3
"""Select four independent true-motion full75 canaries without launching a GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPOSITORY / "tools/qa/build_strict_two_human_full_episode_batch.py"
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "strict2h_full75_builder", BUILDER_PATH
)
if BUILDER_SPEC is None or BUILDER_SPEC.loader is None:
    raise RuntimeError(f"cannot import {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)

SCHEMA = "avengine_native_strict_two_human_dynamic_full75_canary_preflight_v1"
MECHANISMS = (
    "target_moves",
    "distractor_moves",
    "both_move",
    "camera_pan_both_static",
)
SELECTIONS = (
    ("target_moves", "left", "M", "F", "stratum_01"),
    ("distractor_moves", "right", "F", "M", "stratum_02"),
    ("both_move", "left", "M", "F", "stratum_03"),
    ("camera_pan_both_static", "right", "F", "M", "stratum_04"),
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _motion_pattern_inventory(
    trajectories: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for trajectory in trajectories:
        source1, _ = BUILDER._actor_motion_passes(
            trajectory["paths"]["source1_actor"],
            expected_moving=True,
            contract=contract,
        )
        source2, _ = BUILDER._actor_motion_passes(
            trajectory["paths"]["source2_actor"],
            expected_moving=True,
            contract=contract,
        )
        if source1 and source2:
            counts["both_moving"] += 1
        elif source1:
            counts["source1_only_moving"] += 1
        elif source2:
            counts["source2_only_moving"] += 1
        else:
            counts["both_static"] += 1
    return dict(sorted(counts.items()))


def build(request_path: Path, output: Path) -> Path:
    BUILDER._require(not output.exists(), f"refusing to overwrite output: {output}")
    request = BUILDER._load(request_path)
    strict = BUILDER._load(
        BUILDER._resolve(request["inputs"]["strict_sparse_contract"])
    )
    sounds = BUILDER._load(
        BUILDER._resolve(request["inputs"]["controlled_sound_registry"])
    )
    identities = BUILDER._identity_metadata(strict, sounds)
    suite_path = BUILDER._resolve(request["inputs"]["native_floor_point_suite"])
    suite = BUILDER._load(suite_path)
    trajectories, points, _ = BUILDER._source_inventory(suite)
    motion_inventory = _motion_pattern_inventory(
        trajectories, request["motion_contract"]
    )
    BUILDER._require(
        motion_inventory.get("both_moving", 0) > 0,
        "no true both-moving source trajectory exists",
    )

    rng = random.Random(20260812)
    rng.shuffle(trajectories)
    camera_strata = BUILDER._camera_candidates(
        points,
        cell_size=float(request["independence_contract"]["camera_cluster_cell_size_m"]),
    )
    for records in camera_strata.values():
        rng.shuffle(records)

    used_source_scenarios: set[str] = set()
    used_camera_clusters: set[str] = set()
    rows: list[dict[str, Any]] = []
    suite_root = suite_path.parent
    for order, (
        mechanism,
        target_side,
        target_key,
        distractor_key,
        stratum_id,
    ) in enumerate(SELECTIONS, start=1):
        found: dict[str, Any] | None = None
        for camera_record in camera_strata[stratum_id]:
            cell_x, cell_z = camera_record["cell"]
            cluster_id = f"apartment_grid075_x{cell_x:+03d}_z{cell_z:+03d}"
            if cluster_id in used_camera_clusters:
                continue
            camera = [
                float(camera_record["point_m"][0]),
                float(request["geometry_contract"]["camera_height_m"]),
                float(camera_record["point_m"][2]),
            ]
            for trajectory in trajectories:
                scenario_id = str(trajectory["scenario_id"])
                if (
                    scenario_id in used_source_scenarios
                    or scenario_id == camera_record["scenario_id"]
                ):
                    continue
                for (
                    target_actor_id,
                    distractor_actor_id,
                    hold_frame,
                ) in BUILDER._candidate_cases(trajectory, mechanism):
                    target_map, distractor_map = BUILDER._frame_maps(
                        mechanism, hold_frame
                    )
                    target_path = BUILDER._path(trajectory, target_actor_id, target_map)
                    distractor_path = BUILDER._path(
                        trajectory, distractor_actor_id, distractor_map
                    )
                    yaw = BUILDER._camera_yaw_deg(
                        camera, [target_path, distractor_path]
                    )
                    pan = float(
                        request["geometry_contract"]["camera_pan_total_degrees"]
                    )
                    camera_yaw_path = [
                        yaw
                        + (
                            (frame_index / 74.0 - 0.5) * pan
                            if mechanism == "camera_pan_both_static"
                            else 0.0
                        )
                        for frame_index in range(BUILDER.FRAME_COUNT)
                    ]
                    motion = BUILDER._mechanism_motion_preflight(
                        mechanism=mechanism,
                        target_path=target_path,
                        distractor_path=distractor_path,
                        camera_yaw_path=camera_yaw_path,
                        contract=request["motion_contract"],
                    )
                    if motion["status"] != "pass":
                        continue
                    projection = BUILDER._geometry_metrics(
                        request=request,
                        camera=camera,
                        yaw_deg=yaw,
                        target_path=target_path,
                        distractor_path=distractor_path,
                        target_side=target_side,
                        camera_pan=mechanism == "camera_pan_both_static",
                    )
                    if projection is None:
                        continue
                    provenance = BUILDER._validate_provenance(
                        camera_record=camera_record,
                        trajectory=trajectory,
                        target_actor_id=target_actor_id,
                        distractor_actor_id=distractor_actor_id,
                        target_frame_map=target_map,
                        distractor_frame_map=distractor_map,
                        suite_root=suite_root,
                    )
                    found = {
                        "cluster_id": cluster_id,
                        "camera_record": camera_record,
                        "camera": camera,
                        "camera_yaw_path": camera_yaw_path,
                        "trajectory": trajectory,
                        "target_actor_id": target_actor_id,
                        "distractor_actor_id": distractor_actor_id,
                        "target_map": target_map,
                        "distractor_map": distractor_map,
                        "target_path": target_path,
                        "distractor_path": distractor_path,
                        "motion": motion,
                        "projection": projection,
                        "provenance": provenance,
                    }
                    break
                if found is not None:
                    break
            if found is not None:
                break
        BUILDER._require(found is not None, f"no true-motion candidate: {mechanism}")
        episode_id = f"strict2h_dynamic_canary_{order:02d}_{mechanism}_v1"
        target = identities[target_key]
        distractor = identities[distractor_key]
        rows.append(
            {
                "execution_order": order,
                "episode_id": episode_id,
                "mechanism": mechanism,
                "target_side": target_side,
                "target": {
                    **target,
                    "source_slot_id": "source1",
                    "source_actor_id": found["target_actor_id"],
                    "frame_index_map": found["target_map"],
                    "root_path_m": found["target_path"],
                    "voice_policy": "speaking",
                },
                "distractor": {
                    "identity_key": distractor_key,
                    "identity_id": distractor["identity_id"],
                    "runtime_asset_id": distractor["runtime_asset_id"],
                    "runtime_revision": distractor["runtime_revision"],
                    "source_slot_id": "source2",
                    "source_actor_id": found["distractor_actor_id"],
                    "frame_index_map": found["distractor_map"],
                    "root_path_m": found["distractor_path"],
                    "voice_policy": "silent",
                },
                "native_source_scenario_id": found["trajectory"]["scenario_id"],
                "camera_cluster_id": found["cluster_id"],
                "camera": {
                    "translation_m": found["camera"],
                    "yaw_path_deg": found["camera_yaw_path"],
                    "horizontal_fov_deg": request["geometry_contract"][
                        "horizontal_fov_deg"
                    ],
                    "provenance": found["camera_record"],
                },
                "motion_preflight": found["motion"],
                "projection_preflight": found["projection"],
                "native_root_provenance": found["provenance"],
                "source_suite": str(suite_path),
                "suite_plan": "PENDING_DYNAMIC_SUITE_MATERIALIZATION",
                "exact_rir_plan": "PENDING_DYNAMIC_EXACT_RIR_PLAN",
                "binaural_audio": "PENDING_DYNAMIC_BINAURAL_RENDER",
                "gpu_launch_authorized": False,
                "physical_gpu_index": 1,
                "graphics_adapter_argument": 1,
                "rpc_port": 39700 + order,
                "formal": False,
                "qualification_claim": False,
                "status": "pass_cpu_motion_geometry_pending_suite_acoustics_gpu1",
            }
        )
        used_source_scenarios.add(str(found["trajectory"]["scenario_id"]))
        used_camera_clusters.add(str(found["cluster_id"]))

    BUILDER._require(
        [row["mechanism"] for row in rows] == list(MECHANISMS),
        "dynamic mechanism closure failed",
    )
    BUILDER._require(
        len(used_source_scenarios) == len(used_camera_clusters) == 4,
        "dynamic canaries are not independent",
    )
    result = {
        "schema": SCHEMA,
        "status": "pass_cpu_motion_geometry_pending_suite_acoustics_gpu1",
        "dynamic_canary_count": 4,
        "dynamic_canary_gpu_pass_count": 0,
        "single_room_mechanism_pilot_authorized": False,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "source_motion_pattern_counts": motion_inventory,
        "unique_source_scenario_count": len(used_source_scenarios),
        "unique_camera_cluster_count": len(used_camera_clusters),
        "target_side_counts": dict(Counter(row["target_side"] for row in rows)),
        "gpu_policy": request["gpu_policy"],
        "canaries": rows,
    }
    output.mkdir(parents=True)
    path = output / "preflight.json"
    _write(path, result)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = build(args.request.resolve(), args.output.resolve())
    print(f"STRICT_TWO_HUMAN_DYNAMIC_CANARY_PREFLIGHT_OK preflight={path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
