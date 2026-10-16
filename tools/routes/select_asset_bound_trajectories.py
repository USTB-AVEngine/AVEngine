#!/usr/bin/env python3
"""Select asset-bound source-slot routes that pass the real center-point gate.

The reusable room bank deliberately stores only generic ``source1`` and
``source2`` root paths.  A concrete asset can move its emitter sideways from
that root (for example, a muzzle ahead of a cat root), so passing the generic
route gate is necessary but not sufficient.  This tool evaluates every
requested asset pairing across the complete finite bank in one navmesh session,
then emits a balanced, deterministic scenario set containing only routes whose
*bound emitter points* pass.  It never changes a source path to make it pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.routes.asset_emitter import (
    bind_asset_emitters_to_bank,
    validate_asset_emitter_binding_set,
)
from avengine.routes.room_feasibility import MOTION_CASES, TrajectoryBank, TrajectoryEpisode
from avengine.runtime_profiles import (
    build_asset_emitter_binding,
    default_source_asset_runtime_registry_path,
    load_source_asset_runtime_registry,
)
from tools.acoustics.build_asset_bound_rir_plan import (
    SCENARIO_SET_SCHEMA,
    _evaluate_navmesh_center_gate,
    _load_bank,
    _load_listener,
)


OUTPUT_SCHEMA = "avengine_asset_bound_trajectory_selection_delivery_v1"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _binding_set_from_asset_selection(
    selection: Any,
    *,
    source_registry: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(selection, Mapping) or set(selection) != {
        "source1",
        "source2",
    }:
        raise RuntimeError(
            "asset_selection must contain exactly source1 and source2"
        )
    bindings = []
    for slot in ("source1", "source2"):
        value = selection[slot]
        if isinstance(value, str):
            asset_id = value
            revision = None
            anchor_id = None
        elif isinstance(value, Mapping):
            asset_id = value.get("asset_id")
            revision = value.get("revision")
            anchor_id = value.get("anchor_id")
        else:
            raise RuntimeError(f"asset_selection.{slot} is invalid")
        if not isinstance(asset_id, str) or not asset_id:
            raise RuntimeError(f"asset_selection.{slot}.asset_id is invalid")
        bindings.append(
            build_asset_emitter_binding(
                source_registry,
                source_slot_id=slot,
                asset_id=asset_id,
                revision=revision,
                anchor_id=anchor_id,
            )
        )
    return {
        "schema": "avengine_asset_emitter_binding_set_v1",
        "bindings": bindings,
    }


def _pair_templates(
    scenario_set: Mapping[str, Any],
    *,
    source_registry: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    if scenario_set.get("schema") != SCENARIO_SET_SCHEMA:
        raise RuntimeError(f"scenario schema must be {SCENARIO_SET_SCHEMA}")
    raw_scenarios = scenario_set.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise RuntimeError("scenario set must contain scenarios")
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    for index, raw in enumerate(raw_scenarios):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"scenarios[{index}] must be an object")
        binding_set = raw.get("binding_set")
        asset_selection = raw.get("asset_selection")
        if binding_set is not None and asset_selection is not None:
            raise RuntimeError(
                "one template scenario cannot define both binding_set and "
                "asset_selection"
            )
        if binding_set is None and asset_selection is not None:
            if source_registry is None:
                raise RuntimeError(
                    "asset_selection requires a source asset runtime registry"
                )
            binding_set = _binding_set_from_asset_selection(
                asset_selection,
                source_registry=source_registry,
            )
        source_episode_id = raw.get("trajectory_episode_id")
        output_episode_id = raw.get("output_episode_id")
        if (
            not isinstance(binding_set, Mapping)
            or not isinstance(source_episode_id, str)
            or not isinstance(output_episode_id, str)
            or not output_episode_id.endswith(f"_{source_episode_id}")
        ):
            raise RuntimeError(
                "each template scenario needs binding_set or asset_selection "
                "and an output ID ending "
                "with _<trajectory_episode_id>"
            )
        # Validation also refuses an underspecified or slot-ambiguous pairing.
        validate_asset_emitter_binding_set(binding_set)
        pair_id = output_episode_id[: -(len(source_episode_id) + 1)]
        if not pair_id:
            raise RuntimeError("could not derive a nonempty pairing ID")
        key = _canonical_json(binding_set)
        prior = seen.get(key)
        if prior is not None and prior[0] != pair_id:
            raise RuntimeError("one binding_set has inconsistent pairing IDs")
        seen[key] = (pair_id, dict(binding_set))
    templates = tuple(sorted(seen.values(), key=lambda value: value[0]))
    if len({pair_id for pair_id, _ in templates}) != len(templates):
        raise RuntimeError("pairing IDs must be unique")
    return templates


def _quotas(total: int) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int) or total < len(MOTION_CASES):
        raise RuntimeError("episodes-per-pair must be an integer >= four")
    base, remainder = divmod(total, len(MOTION_CASES))
    return {
        motion_case: base + int(index < remainder)
        for index, motion_case in enumerate(MOTION_CASES)
    }


def select_scenarios(
    *,
    bank: TrajectoryBank,
    templates: tuple[tuple[str, dict[str, Any]], ...],
    listener_position_m: np.ndarray,
    navmesh_path: Path,
    floor_height_m: float,
    episodes_per_pair: int,
    seed: int,
    maximum_floor_snap_xz_m: float,
    minimum_navmesh_clearance_m: float,
    minimum_pair_separation_m: float = 0.30,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the selected scenario request and its compact selection report.

    The native gate is called exactly once over all pairing/route candidates.
    Selection shuffles only candidates that have already passed; it does not
    snap, move, or otherwise repair an emitter point.
    """

    quotas = _quotas(episodes_per_pair)
    if (
        not np.isfinite(minimum_pair_separation_m)
        or minimum_pair_separation_m < 0.0
    ):
        raise RuntimeError(
            "minimum-pair-separation-m must be finite and nonnegative"
        )
    candidate_episodes: list[TrajectoryEpisode] = []
    candidate_map: dict[str, tuple[str, TrajectoryEpisode, dict[str, Any]]] = {}
    candidate_pair_separation_m: dict[str, float] = {}
    for pair_id, binding_set in templates:
        bindings = validate_asset_emitter_binding_set(binding_set)
        bound, _report = bind_asset_emitters_to_bank(
            bank, bindings, listener_position_m=listener_position_m
        )
        for source in bound.episodes:
            candidate_id = f"{pair_id}__{source.episode_id}"
            if candidate_id in candidate_map:
                raise RuntimeError("asset-bound candidate IDs must be unique")
            candidate_episodes.append(
                TrajectoryEpisode(
                    episode_id=candidate_id,
                    motion_case=source.motion_case,
                    source_root_paths_m=source.source_root_paths_m,
                    source_center_paths_m=source.source_center_paths_m,
                    statistics=source.statistics,
                )
            )
            candidate_map[candidate_id] = (pair_id, source, binding_set)
            candidate_pair_separation_m[candidate_id] = float(
                np.min(
                    np.linalg.norm(
                        source.source_center_paths_m["source1"][:, (0, 2)]
                        - source.source_center_paths_m["source2"][:, (0, 2)],
                        axis=1,
                    )
                )
            )
    candidate_bank = TrajectoryBank(
        episodes=tuple(candidate_episodes),
        frame_count=bank.frame_count,
        frame_rate_hz=bank.frame_rate_hz,
        seed=bank.seed,
    )
    candidate_gate = _evaluate_navmesh_center_gate(
        candidate_bank,
        navmesh_path=navmesh_path,
        floor_height_m=floor_height_m,
        maximum_floor_snap_xz_m=maximum_floor_snap_xz_m,
        minimum_navmesh_clearance_m=minimum_navmesh_clearance_m,
    )
    passed_ids = {
        candidate_id
        for candidate_id in candidate_map
        if candidate_gate["sources"][f"{candidate_id}::source1"]["status"] == "pass"
        and candidate_gate["sources"][f"{candidate_id}::source2"]["status"] == "pass"
        and candidate_pair_separation_m[candidate_id]
        >= minimum_pair_separation_m
    }
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    per_pair_report: dict[str, Any] = {}
    for pair_id, _binding_template in templates:
        selected_ids: list[str] = []
        availability: dict[str, int] = {}
        for motion_case in MOTION_CASES:
            candidates = sorted(
                candidate_id
                for candidate_id, (candidate_pair, source, _bindings) in candidate_map.items()
                if candidate_pair == pair_id
                and source.motion_case == motion_case
                and candidate_id in passed_ids
            )
            availability[motion_case] = len(candidates)
            required = quotas[motion_case]
            if len(candidates) < required:
                raise RuntimeError(
                    f"{pair_id} has only {len(candidates)} passing {motion_case} "
                    f"candidates, but needs {required}"
                )
            order = rng.permutation(len(candidates))[:required]
            selected_ids.extend(candidates[int(index)] for index in order)
        selected_ids.sort()
        per_pair_report[pair_id] = {
            "requested_episode_count": episodes_per_pair,
            "motion_case_quotas": quotas,
            "passing_candidates_by_motion_case": availability,
            "selected_candidate_ids": selected_ids,
        }
        for candidate_id in selected_ids:
            _candidate_pair, source, binding_set = candidate_map[candidate_id]
            selected.append(
                {
                    "trajectory_episode_id": source.episode_id,
                    "output_episode_id": candidate_id,
                    "binding_set": binding_set,
                }
            )
    selected.sort(key=lambda value: str(value["output_episode_id"]))
    scenario_request = {
        "schema": SCENARIO_SET_SCHEMA,
        "selection": {
            "method": "asset_bound_center_gate_then_seeded_balanced_sampling",
            "source_bank_episode_count": len(bank.episodes),
            "episodes_per_pair": episodes_per_pair,
            "seed": seed,
            "motion_case_quotas": quotas,
        },
        "scenarios": selected,
    }
    report = {
        "schema": OUTPUT_SCHEMA,
        "status": "pass",
        "claim_boundary": (
            "finite route selection after concrete asset emitter offsets; "
            "source centers only, not body-volume collision"
        ),
        "candidate_scenario_count": len(candidate_map),
        "candidate_gate_status": candidate_gate["status"],
        "candidate_passing_scenario_count": len(passed_ids),
        "selected_scenario_count": len(selected),
        "minimum_pair_separation_m": minimum_pair_separation_m,
        "minimum_selected_pair_separation_m": min(
            candidate_pair_separation_m[
                str(value["output_episode_id"])
            ]
            for value in selected
        ),
        "per_pair": per_pair_report,
        "candidate_gate": candidate_gate,
    }
    return scenario_request, report


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if output.exists() or output.is_symlink() or staging.exists() or staging.is_symlink():
        raise RuntimeError(f"refusing to replace output or staging: {output}")
    bank = _load_bank(args.trajectory_bank.resolve())
    scenario_set = load_json(args.scenario_templates.resolve())
    source_registry = load_source_asset_runtime_registry(
        args.source_asset_registry.resolve()
    )
    templates = _pair_templates(
        scenario_set,
        source_registry=source_registry,
    )
    listener, _orientation, _stride = _load_listener(args.template_rir_plan.resolve())
    scenarios, report = select_scenarios(
        bank=bank,
        templates=templates,
        listener_position_m=listener,
        navmesh_path=args.navmesh.resolve(),
        floor_height_m=float(args.floor_height_m),
        episodes_per_pair=args.episodes_per_pair,
        seed=args.seed,
        maximum_floor_snap_xz_m=float(args.maximum_floor_snap_xz_m),
        minimum_navmesh_clearance_m=float(args.minimum_navmesh_clearance_m),
        minimum_pair_separation_m=float(args.minimum_pair_separation_m),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        write_json(staging / "selected_scenarios.json", scenarios)
        write_json(staging / "selection_report.json", report)
        write_json(
            staging / "timing.json",
            {
                "schema": "avengine_asset_bound_trajectory_selection_timing_v1",
                "status": "pass",
                "native_rlr_calls": 0,
                "visual_render_calls": 0,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        write_json(
            staging / "delivery.json",
            {
                "schema": OUTPUT_SCHEMA,
                "status": "pass",
                "selected_scenario_count": report["selected_scenario_count"],
                "source_asset_runtime_registry": {
                    "path": str(args.source_asset_registry.resolve()),
                    "registry_id": source_registry["registry_id"],
                    "revision": source_registry["revision"],
                },
                "outputs": {
                    "selected_scenarios": "selected_scenarios.json",
                    "selection_report": "selection_report.json",
                    "timing": "timing.json",
                },
            },
        )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--scenario-templates", type=Path, required=True)
    parser.add_argument(
        "--source-asset-registry",
        type=Path,
        default=default_source_asset_runtime_registry_path(),
        help=(
            "Resolves asset_selection entries into measured emitter and "
            "anatomical-forward bindings."
        ),
    )
    parser.add_argument("--template-rir-plan", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True)
    parser.add_argument("--floor-height-m", type=float, required=True)
    parser.add_argument("--episodes-per-pair", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--maximum-floor-snap-xz-m", type=float, default=0.03)
    parser.add_argument("--minimum-navmesh-clearance-m", type=float, default=0.02)
    parser.add_argument("--minimum-pair-separation-m", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(run(parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
