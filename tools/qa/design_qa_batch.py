#!/usr/bin/env python3
"""Design one constraint-driven QA v2 batch from declared-pass viewpoints.

Production requires a complete 4--8-view bank. Points rotate deterministically
through bank order unless ``--viewpoint-id`` selects one view. The historical
Apartment view0 path is reachable only through ``--legacy-view0-report-only``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avengine.qa.design_viewpoints import (
    DesignViewpoint,
    DesignViewpointError,
    load_design_viewpoint_bank,
    load_legacy_view0_report_only,
    select_design_viewpoints,
)
from avengine.m5_1.source_contracts import sample_boundary
from avengine.m6.audio_program import (
    bind_audio_program_hash,
    validate_audio_program,
)
from avengine.qa.extended_questions import event_frame_index
from avengine.qa.motion_feasibility import (
    MotionFeasibilityError,
    classify_motion_feasibility,
    evaluate_motion_pair,
    minimum_linear_path_separation,
)
from avengine.security.path_policy import WorkspacePathPolicy, atomic_publish_directory


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
DEFAULT_CORRIDOR_BANK = REPOSITORY / "examples/qa_v2/straight_corridor_bank_v1.json"
DEFAULT_PROGRAMS_DIR = REPOSITORY / "examples/m7/current_apartment/audio_programs/qa_v2"
DEFAULT_LEGACY_M1_REQUEST = (
    REPOSITORY / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json"
)
DEFAULT_SNAPSHOT_CONTENT = Path(
    "/data/avengine_external/ue-assets/actor_content_registry_v9_20260823T033709Z/"
    "cpp/unreal_projects/SpearSim/Content"
)
SOURCE_HEIGHT_UE_CM = 27.1
EVENT_ONE_FRACTION = 4.0 / 75.0
EVENT_TWO_FRACTION = 22.5 / 75.0

HUMANS = {
    "blue": "rocketbox_human_male_adult_01_top_blue_research_v1",
    "green": "rocketbox_human_male_adult_01_top_green_research_v1",
    "burgundy": "rocketbox_human_male_adult_01_top_burgundy_research_v1",
}
DOGS = {
    "collie": "generated_border_collie_black_white_medium_standard_adult_research_v1",
    "labrador": "generated_labrador_yellow_medium_standard_adult_research_v1",
}
HUMAN_COMBOS = (("blue", "green"), ("blue", "burgundy"), ("green", "burgundy"))


class DesignBatchError(ValueError):
    """The requested QA design batch cannot be authored safely."""


def _load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_input_modalities(
    value: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise DesignBatchError(
            "--input-modalities must contain unique audio/video names"
        )
    normalized = tuple(str(item) for item in value)
    if any(item not in {"audio", "video"} for item in normalized):
        raise DesignBatchError("--input-modalities supports only audio and video")
    if "audio" not in normalized:
        raise DesignBatchError(
            "--input-modalities must include audio because every point materializes AudioProgram"
        )
    return normalized


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def mesh_package_for(asset: Mapping[str, Any], snapshot_content: Path) -> str:
    spear_unreal = asset["runtime_backends"]["spear_unreal"]
    mesh_dir_package = spear_unreal["idle_animation"].split(".", 1)[0].rsplit("/", 1)[0]
    gate = mesh_dir_package.rsplit("/", 1)[-1]
    physical_directory = snapshot_content / "MyAssets/Audioset/Meshes" / gate
    names = [path.stem for path in physical_directory.glob("*.uasset")]
    for name in names:
        if f"{name}_Skeleton" in names:
            return f"{mesh_dir_package}/{name}"
    if "runtime" in names:
        return f"{mesh_dir_package}/runtime"
    raise DesignBatchError(
        f"cannot identify skeletal mesh in {physical_directory}: {names}"
    )


def actor_entry(
    slot: str,
    asset_id: str,
    *,
    assets: Mapping[str, Mapping[str, Any]],
    snapshot_content: Path,
) -> dict[str, Any]:
    record = assets[asset_id]
    spear_unreal = record["runtime_backends"]["spear_unreal"]
    blueprint = spear_unreal["blueprint_class_path"]
    blueprint_package = blueprint.split(".", 1)[0]
    mesh_package = mesh_package_for(record, snapshot_content)
    mesh_name = mesh_package.rsplit("/", 1)[-1]

    def physical(package: str) -> str:
        path = snapshot_content / f"{package.split('/Game/', 1)[1]}.uasset"
        if not path.is_file():
            raise DesignBatchError(f"missing physical source: {path}")
        return str(path)

    return {
        "asset_id": asset_id,
        "legacy_timeline_actor_id": f"{record['identity']['species_id']}_{slot[-1]}",
        "physical_authorized_internal_sources": {
            "blueprint": physical(blueprint_package),
            "graph_derived_mesh": physical(mesh_package),
            "idle": physical(spear_unreal["idle_animation"].split(".", 1)[0]),
            "walking": physical(spear_unreal["walking_animation"].split(".", 1)[0]),
        },
        "profile_alias": asset_id,
        "revision": record["revision"],
        "source_slot_id": slot,
        "ue_binding": {
            "blueprint_object_path": blueprint,
            "blueprint_package": blueprint_package,
            "graph_derived_mesh": {
                "derivation": (
                    "direct graph dependency of the selected Blueprint; profile binds "
                    "blueprint_component and declares no standalone mesh path"
                ),
                "object_path": f"{mesh_package}.{mesh_name}",
                "package": mesh_package,
            },
            "idle_object_path": spear_unreal["idle_animation"],
            "idle_package": spear_unreal["idle_animation"].split(".", 1)[0],
            "profile_skeletal_mesh_binding": spear_unreal["skeletal_mesh_binding"],
            "profile_skeletal_mesh_path": spear_unreal["skeletal_mesh_path"],
            "walking_object_path": spear_unreal["walking_animation"],
            "walking_package": spear_unreal["walking_animation"].split(".", 1)[0],
        },
    }


def selection_document(
    source1_asset: str,
    source2_asset: str,
    *,
    assets: Mapping[str, Mapping[str, Any]],
    snapshot_content: Path,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schema": "avengine_apartment_actor_selection_v1",
        "asset_authorization": "verified_internal",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "QA v2 design batch; research only.",
        "generated_at": generated_at,
        "actors": [
            actor_entry(
                "source1",
                source1_asset,
                assets=assets,
                snapshot_content=snapshot_content,
            ),
            actor_entry(
                "source2",
                source2_asset,
                assets=assets,
                snapshot_content=snapshot_content,
            ),
        ],
    }


def _cells() -> tuple[tuple[str, str, str, str], ...]:
    cells = []
    for combo in HUMAN_COMBOS:
        for order in ("afirst", "bfirst"):
            cells.append(("human", HUMANS[combo[0]], HUMANS[combo[1]], order))
    for order in ("afirst", "bfirst"):
        cells.append(("dog", DOGS["collie"], DOGS["labrador"], order))
    return tuple(cells)


minimum_separation_cm = minimum_linear_path_separation


def _gates_ok(
    viewpoint: DesignViewpoint,
    path1: tuple[Sequence[float], Sequence[float]],
    path2: tuple[Sequence[float], Sequence[float]],
    *,
    offscreen: bool = False,
) -> bool:
    """Return the legacy difficulty-profile match, never candidate validity."""

    return evaluate_motion_pair(
        camera_position_xy=viewpoint.camera_position_ue_cm[:2],
        camera_yaw_degrees=viewpoint.camera_yaw_ue_deg,
        source1_path=path1,
        source2_path=path2,
        first_anchor_fraction=EVENT_ONE_FRACTION,
        second_anchor_fraction=EVENT_TWO_FRACTION,
        minimum_first_side_margin=0.12,
        minimum_second_side_margin=0.06,
        minimum_continuous_separation=60.0,
        allow_any_side=offscreen,
    ).passed


def find_combo(
    viewpoint: DesignViewpoint, seed: int, mode: str
) -> tuple[
    tuple[tuple[float, ...], tuple[float, ...]],
    tuple[tuple[float, ...], tuple[float, ...]],
]:
    directed = viewpoint.directed_paths(source_height_ue_cm=SOURCE_HEIGHT_UE_CM)
    stands = viewpoint.stand_points(source_height_ue_cm=SOURCE_HEIGHT_UE_CM)
    if not directed or not stands:
        raise DesignBatchError(f"viewpoint {viewpoint.viewpoint_id} has no corridors")
    if mode == "offscreen":
        offscreen = viewpoint.offscreen_position_ue_cm
        path2 = (offscreen, offscreen)
        path1 = directed[seed % len(directed)]
        return path1, path2
    else:
        try:
            classification = classify_motion_feasibility(
                camera_position_xy=viewpoint.camera_position_ue_cm[:2],
                camera_yaw_degrees=viewpoint.camera_yaw_ue_deg,
                directed_paths=directed,
                stand_points=stands,
                modes=(mode,),
                seed=seed,
                first_anchor_fraction=EVENT_ONE_FRACTION,
                second_anchor_fraction=EVENT_TWO_FRACTION,
                minimum_first_side_margin=0.12,
                minimum_second_side_margin=0.06,
                minimum_continuous_separation=60.0,
            )
        except MotionFeasibilityError as exc:
            raise DesignBatchError(str(exc)) from exc
        representative = classification.modes[0].representative_pair
        if representative is not None:
            return representative.source1_path, representative.source2_path
    raise DesignBatchError(
        f"no combo for viewpoint {viewpoint.viewpoint_id} mode {mode} seed {seed}"
    )


def _program_short(asset_id: str, assets: Mapping[str, Mapping[str, Any]]) -> str:
    asset = assets[asset_id]
    realized = asset.get("realized_attributes", {})
    if "top_color" in realized:
        return str(realized["top_color"])
    return {
        "border_collie": "collie",
        "labrador_retriever": "labrador",
    }[asset["identity"]["breed_id"]]


def motion_case_sequence(cell_size: int) -> tuple[str, ...]:
    """Return one cell composition with both shared-motion cases represented.

    The final point remains the existing off-screen single-mover point.  The
    ordinary points reserve equal counts for both-moving and both-static so
    their all-moving/all-still answers cannot introduce a T7/TA key skew.
    """

    if cell_size < 13:
        raise DesignBatchError(
            "--cell-size must be at least 13 so both single-mover cases, "
            "both-moving, both-static, and off-screen are represented"
        )
    ordinary_count = cell_size - 1
    shared_motion_count = max(1, ordinary_count // 6)
    single_motion_count = ordinary_count - 2 * shared_motion_count
    source1_count = (single_motion_count + 1) // 2
    source2_count = single_motion_count - source1_count
    return (
        ("s1_moving_s2_static",) * source1_count
        + ("s1_static_s2_moving",) * source2_count
        + ("both_moving",) * shared_motion_count
        + ("both_static",) * shared_motion_count
        + ("s1_moving_s2_static",)
    )


def _frame_sample_delta(frame_delta: int) -> int:
    if frame_delta >= 0:
        return sample_boundary(frame_delta)
    return -sample_boundary(-frame_delta)


def materialize_point_audio_program(
    base_program: Mapping[str, Any],
    *,
    point_id: str,
    schedule_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample a point schedule inside the validated AudioProgram timeline.

    Only the first two scheduled output windows move. Event order, durations,
    source slices, endpoint identity, gain/fades, and every later window remain
    those of the registered base AudioProgram. One full video-frame silent gap
    is retained on both sides of the second event. The finite candidate set is
    therefore derived from existing event/timeline authority rather than from
    an unconstrained time distribution.
    """

    errors = validate_audio_program(base_program)
    if errors:
        raise DesignBatchError(f"base AudioProgram is invalid: {'; '.join(errors)}")
    if base_program.get("mode") != "sequential_sources":
        raise DesignBatchError("QA onset sampling requires sequential_sources")
    events = base_program["events"]
    if len(events) < 3:
        raise DesignBatchError(
            "QA onset sampling requires a third authoritative event window "
            "to bound the second onset"
        )
    first, second, third = events[:3]
    first_duration = first["end_sample_exclusive"] - first["start_sample"]
    second_duration = second["end_sample_exclusive"] - second["start_sample"]
    minimum_gap = sample_boundary(1)
    feasible_pairs: list[tuple[int, int]] = []
    for first_delta_frames in range(-2, 3):
        first_start = first["start_sample"] + _frame_sample_delta(first_delta_frames)
        if first_start < minimum_gap:
            continue
        first_end = first_start + first_duration
        for second_delta_frames in range(-4, 5):
            second_start = second["start_sample"] + _frame_sample_delta(
                second_delta_frames
            )
            second_end = second_start + second_duration
            if first_end + minimum_gap > second_start:
                continue
            if second_end + minimum_gap > third["start_sample"]:
                continue
            feasible_pairs.append((first_start, second_start))
    if (
        len({item[0] for item in feasible_pairs}) < 2
        or len({item[1] - item[0] for item in feasible_pairs}) < 2
    ):
        raise DesignBatchError(
            "base AudioProgram does not expose multiple feasible first onsets "
            "and first-to-second intervals"
        )

    first_start, second_start = feasible_pairs[
        random.Random(schedule_seed).randrange(len(feasible_pairs))
    ]
    result = deepcopy(dict(base_program))
    result["program_id"] = f"{base_program['program_id']}.{point_id}.schedule_v1"
    ticks_per_sample = result["timeline"]["ticks_per_sample"]
    for event, start_sample in zip(
        result["events"][:2], (first_start, second_start), strict=True
    ):
        duration = event["end_sample_exclusive"] - event["start_sample"]
        event["start_sample"] = start_sample
        event["end_sample_exclusive"] = start_sample + duration
        event["start_tick"] = start_sample * ticks_per_sample
        event["end_tick_exclusive"] = event["end_sample_exclusive"] * ticks_per_sample
    result = bind_audio_program_hash(result)
    rebound_errors = validate_audio_program(result)
    if rebound_errors:
        raise DesignBatchError(
            f"materialized AudioProgram is invalid: {'; '.join(rebound_errors)}"
        )
    starts = [event["start_sample"] for event in result["events"]]
    ends = [event["end_sample_exclusive"] for event in result["events"]]
    schedule = {
        "policy": "deterministic_point_seed_existing_event_windows_v1",
        "schedule_seed": schedule_seed,
        "base_program_id": base_program["program_id"],
        "base_program_revision": base_program["revision"],
        "materialized_program_id": result["program_id"],
        "event_start_samples": starts,
        "event_end_samples_exclusive": ends,
        "first_event_frame": event_frame_index(starts[0]),
        "second_event_frame": event_frame_index(starts[1]),
        "first_second_onset_interval_samples": starts[1] - starts[0],
        "first_second_onset_interval_frames": (
            event_frame_index(starts[1]) - event_frame_index(starts[0])
        ),
        "feasible_first_second_pair_count": len(feasible_pairs),
        "unchanged_event_fields": [
            "event_id",
            "source_endpoint_id",
            "sound_asset_id",
            "source_start_sample",
            "source_end_sample_exclusive",
            "linear_gain",
            "fade_samples",
            "normalization_policy",
            "render_source_stem",
        ],
        "later_event_windows": "unchanged_from_base_program",
    }
    return result, schedule


def build_specs(
    *,
    viewpoints: Sequence[DesignViewpoint],
    assets: Mapping[str, Mapping[str, Any]],
    cell_size: int,
    twins_per_cell: int,
    input_modalities: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not viewpoints:
        raise DesignBatchError("at least one resolved viewpoint is required")
    motion_cases = motion_case_sequence(cell_size)
    if not 0 <= twins_per_cell <= cell_size:
        raise DesignBatchError("--twins-per-cell must be between zero and cell size")
    normalized_input_modalities = _normalize_input_modalities(input_modalities)
    specs = []
    cells = _cells()
    for cell_index, (kind, asset1, asset2, order) in enumerate(cells):
        for made in range(cell_size):
            point_id = f"b{cell_index * cell_size + made + 1:03d}"
            seed = cell_index * cell_size + made
            viewpoint = viewpoints[(cell_index + made) % len(viewpoints)]
            motion = motion_cases[made]
            if made == cell_size - 1:
                path1, path2 = find_combo(viewpoint, seed, "offscreen")
            else:
                path1, path2 = find_combo(viewpoint, seed, motion)
            short1 = _program_short(asset1, assets)
            short2 = _program_short(asset2, assets)
            pair_kind = "two_human" if kind == "human" else "two_dog"
            suffix = (
                "_turn_taking_v1" if order == "afirst" else "_turn_taking_bfirst_v1"
            )
            spec = {
                "point_id": point_id,
                "pair_kind": kind,
                "source1_asset": asset1,
                "source2_asset": asset2,
                "s1_start": list(path1[0]),
                "s1_end": list(path1[1]),
                "s2_start": list(path2[0]),
                "s2_end": list(path2[1]),
                "motion_case": motion,
                "offscreen_candidate": made == cell_size - 1,
                "twin_of": None,
                "program_id": f"qa_v2_{pair_kind}_{short1}_{short2}{suffix}",
                "schedule_seed": seed,
                "viewpoint_id": viewpoint.viewpoint_id,
                "viewpoint_authority": viewpoint.authority,
                "m1_request": viewpoint.spec_m1_request(),
            }
            if normalized_input_modalities is not None:
                spec["input_modalities"] = list(normalized_input_modalities)
            specs.append(spec)

    by_id = {spec["point_id"]: spec for spec in specs}
    twins = []
    for cell_index in range(len(cells)):
        for offset in range(twins_per_cell):
            source = by_id[f"b{cell_index * cell_size + offset + 1:03d}"]
            kind = source["pair_kind"]
            asset1, asset2 = source["source2_asset"], source["source1_asset"]
            short1 = _program_short(asset1, assets)
            short2 = _program_short(asset2, assets)
            pair_kind = "two_human" if kind == "human" else "two_dog"
            suffix = (
                "_turn_taking_bfirst_v1"
                if "bfirst" in source["program_id"]
                else "_turn_taking_v1"
            )
            twins.append(
                dict(
                    source,
                    point_id=f"bt{len(twins) + 1:03d}",
                    source1_asset=asset1,
                    source2_asset=asset2,
                    twin_of=source["point_id"],
                    program_id=f"qa_v2_{pair_kind}_{short1}_{short2}{suffix}",
                )
            )
    specs.extend(twins)
    return specs, twins


def timeline_author_command(
    *,
    avengine_cli: str,
    point_directory: Path,
    spec: Mapping[str, Any],
    viewpoint: DesignViewpoint,
    registry_path: Path,
) -> list[str]:
    if spec.get("viewpoint_id") != viewpoint.viewpoint_id:
        raise DesignBatchError("spec viewpoint_id does not match timeline viewpoint")
    if spec.get("m1_request") != viewpoint.spec_m1_request():
        raise DesignBatchError("spec M1 request does not match timeline viewpoint")
    command = [
        avengine_cli,
        "m5",
        "author-current-apartment-visual-timeline",
        "--actor-selection",
        str(point_directory / "actor_selection.json"),
        "--source-asset-registry",
        str(registry_path),
        "--camera-position-ue-cm",
        *map(str, viewpoint.camera_position_ue_cm),
        "--camera-yaw-deg",
        str(viewpoint.camera_yaw_ue_deg),
    ]

    def append_route(
        *,
        waypoint_key: str,
        waypoint_flag: str,
        start_key: str,
        start_flag: str,
        end_key: str,
        end_flag: str,
    ) -> None:
        if waypoint_key in spec:
            waypoints = spec[waypoint_key]
            if (
                isinstance(waypoints, (str, bytes))
                or not isinstance(waypoints, Sequence)
                or len(waypoints) < 2
            ):
                raise DesignBatchError(
                    f"spec {waypoint_key} must contain at least two points"
                )
            for waypoint in waypoints:
                if (
                    isinstance(waypoint, (str, bytes))
                    or not isinstance(waypoint, Sequence)
                    or len(waypoint) != 3
                ):
                    raise DesignBatchError(
                        f"spec {waypoint_key} must contain XYZ triplets"
                    )
                command.extend((waypoint_flag, *map(str, waypoint)))
            return
        command.extend((start_flag, *map(str, spec[start_key])))
        command.extend((end_flag, *map(str, spec[end_key])))

    append_route(
        waypoint_key="s1_waypoints_ue_cm",
        waypoint_flag="--human-waypoint-ue-cm",
        start_key="s1_start",
        start_flag="--human-start-ue-cm",
        end_key="s1_end",
        end_flag="--human-end-ue-cm",
    )
    append_route(
        waypoint_key="s2_waypoints_ue_cm",
        waypoint_flag="--beagle-waypoint-ue-cm",
        start_key="s2_start",
        start_flag="--beagle-start-ue-cm",
        end_key="s2_end",
        end_flag="--beagle-end-ue-cm",
    )
    command.extend(("--output", str(point_directory / "timeline.json")))
    return command


def _fresh_staging(output: str | Path) -> tuple[WorkspacePathPolicy, Path, Path]:
    unresolved = Path(output).expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    if os.path.lexists(unresolved):
        raise DesignBatchError(f"output root exists; refusing to clobber: {unresolved}")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    parent = unresolved.parent.resolve(strict=True)
    policy = WorkspacePathPolicy.from_roots((parent,))
    destination = policy.resolve_output(
        parent / unresolved.name, owner="QA design output"
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    return policy, destination, staging


def _resolve_viewpoints(args: argparse.Namespace) -> tuple[DesignViewpoint, ...]:
    if args.legacy_view0_report_only:
        if args.viewpoint_bank is not None or args.viewpoint_id is not None:
            raise DesignBatchError(
                "legacy report-only mode cannot be combined with viewpoint bank selection"
            )
        if args.legacy_offscreen_point_ue_cm is None:
            raise DesignBatchError(
                "--legacy-view0-report-only requires --legacy-offscreen-point-ue-cm"
            )
        return (
            load_legacy_view0_report_only(
                m1_request_path=args.legacy_m1_request,
                corridor_bank_path=args.legacy_corridor_bank,
                offscreen_position_ue_cm=args.legacy_offscreen_point_ue_cm,
            ),
        )
    if args.viewpoint_bank is None:
        raise DesignBatchError(
            "production design requires --viewpoint-bank; legacy view0 requires the "
            "explicit --legacy-view0-report-only mode"
        )
    viewpoints = load_design_viewpoint_bank(args.viewpoint_bank)
    return select_design_viewpoints(viewpoints, args.viewpoint_id)


def run(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    viewpoints = _resolve_viewpoints(args)
    registry_path = Path(args.registry).expanduser().resolve(strict=True)
    registry = _load_json(registry_path)
    assets = {asset["asset_id"]: asset for asset in registry["assets"]}
    snapshot_content = Path(args.snapshot_content).expanduser().resolve(strict=True)
    programs_dir = Path(args.programs_dir).expanduser().resolve(strict=True)
    specs, twins = build_specs(
        viewpoints=viewpoints,
        assets=assets,
        cell_size=args.cell_size,
        twins_per_cell=args.twins_per_cell,
        input_modalities=getattr(args, "input_modalities", None),
    )
    viewpoint_by_id = {item.viewpoint_id: item for item in viewpoints}
    generated_at = datetime.now(timezone.utc).isoformat()
    policy, destination, staging = _fresh_staging(args.output_root)
    selection_cache: dict[tuple[str, str], dict[str, Any]] = {}
    schedule_by_point: dict[str, dict[str, Any]] = {}
    try:
        for spec in specs:
            point_directory = staging / spec["point_id"]
            point_directory.mkdir()
            base_program_path = programs_dir / f"{spec['program_id']}.json"
            if not base_program_path.is_file():
                raise DesignBatchError(
                    f"missing base AudioProgram: {base_program_path}"
                )
            materialized_program, schedule = materialize_point_audio_program(
                _load_json(base_program_path),
                point_id=spec["point_id"],
                schedule_seed=spec["schedule_seed"],
            )
            spec["audio_program_path"] = "audio_program.json"
            spec["audio_schedule"] = schedule
            schedule_by_point[spec["point_id"]] = schedule
            selection_key = (spec["source1_asset"], spec["source2_asset"])
            if selection_key not in selection_cache:
                selection_cache[selection_key] = selection_document(
                    *selection_key,
                    assets=assets,
                    snapshot_content=snapshot_content,
                    generated_at=generated_at,
                )
            _write_json(
                point_directory / "actor_selection.json",
                selection_cache[selection_key],
            )
            _write_json(point_directory / "audio_program.json", materialized_program)
            _write_json(point_directory / "spec.json", spec)
            viewpoint = viewpoint_by_id[spec["viewpoint_id"]]
            command = timeline_author_command(
                avengine_cli=args.avengine_cli,
                point_directory=point_directory,
                spec=spec,
                viewpoint=viewpoint,
                registry_path=registry_path,
            )
            completed = command_runner(
                command,
                capture_output=True,
                text=True,
                cwd=REPOSITORY,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stdout + completed.stderr)[-1000:]
                raise DesignBatchError(
                    f"timeline author failed for {spec['point_id']}: {detail}"
                )

        for twin in twins:
            source_schedule = schedule_by_point[twin["twin_of"]]
            twin_schedule = schedule_by_point[twin["point_id"]]
            if (
                twin_schedule["event_start_samples"]
                != source_schedule["event_start_samples"]
                or twin_schedule["event_end_samples_exclusive"]
                != source_schedule["event_end_samples_exclusive"]
            ):
                raise DesignBatchError(
                    f"twin schedule drift: {twin['point_id']} vs {twin['twin_of']}"
                )

        distribution = Counter(spec["viewpoint_id"] for spec in specs)
        motion_distribution = Counter(spec["motion_case"] for spec in specs)
        first_frames = Counter(
            schedule["first_event_frame"] for schedule in schedule_by_point.values()
        )
        second_frames = Counter(
            schedule["second_event_frame"] for schedule in schedule_by_point.values()
        )
        intervals = Counter(
            schedule["first_second_onset_interval_samples"]
            for schedule in schedule_by_point.values()
        )
        manifest = {
            "schema": "avengine_qa_v2_batch2d_v1",
            "status": (
                "legacy_report_only"
                if args.legacy_view0_report_only
                else "research_only"
            ),
            "qualification_claim": False,
            "generated_at": generated_at,
            "inputs_root": str(destination),
            "viewpoint_selection": {
                "mode": (
                    "legacy_view0_report_only"
                    if args.legacy_view0_report_only
                    else (
                        "explicit_id" if args.viewpoint_id else "deterministic_rotation"
                    )
                ),
                "bank": args.viewpoint_bank,
                "selected_viewpoint_ids": [item.viewpoint_id for item in viewpoints],
                "point_counts": dict(sorted(distribution.items())),
            },
            "input_modalities": (
                None
                if getattr(args, "input_modalities", None) is None
                else list(args.input_modalities)
            ),
            "counts": {
                "total": len(specs),
                "authored": len(specs),
                "twins": len(twins),
            },
            "motion_case_counts": dict(sorted(motion_distribution.items())),
            "audio_onset_schedule": {
                "policy": "deterministic_point_seed_existing_event_windows_v1",
                "first_event_frame_counts": dict(sorted(first_frames.items())),
                "second_event_frame_counts": dict(sorted(second_frames.items())),
                "first_second_onset_interval_sample_counts": dict(
                    sorted(intervals.items())
                ),
                "distinct_first_event_frames": len(first_frames),
                "distinct_second_event_frames": len(second_frames),
                "distinct_first_second_intervals": len(intervals),
                "twins_share_exact_event_schedule": True,
            },
            "author_results": "all_authored",
        }
        _write_json(staging / "batch_manifest.json", manifest)
        return atomic_publish_directory(policy, staging, destination)
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", required=True, help="fresh batch directory; refuses to clobber"
    )
    parser.add_argument(
        "--avengine-cli", default=shutil.which("avengine") or "avengine"
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--snapshot-content", default=str(DEFAULT_SNAPSHOT_CONTENT))
    parser.add_argument("--programs-dir", default=str(DEFAULT_PROGRAMS_DIR))
    parser.add_argument("--cell-size", type=int, default=16)
    parser.add_argument("--twins-per-cell", type=int, default=8)
    parser.add_argument(
        "--input-modalities",
        nargs="+",
        choices=("audio", "video"),
        help=(
            "explicit model-input fact copied to every spec; omit for legacy "
            "design output, which the question producer will reject"
        ),
    )
    parser.add_argument(
        "--viewpoint-bank",
        help="declared-pass 4-8-view bank; required outside legacy report-only mode",
    )
    parser.add_argument(
        "--viewpoint-id",
        help="use one explicit bank viewpoint instead of deterministic rotation",
    )
    parser.add_argument("--legacy-view0-report-only", action="store_true")
    parser.add_argument("--legacy-m1-request", default=str(DEFAULT_LEGACY_M1_REQUEST))
    parser.add_argument("--legacy-corridor-bank", default=str(DEFAULT_CORRIDOR_BANK))
    parser.add_argument(
        "--legacy-offscreen-point-ue-cm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except (
        DesignBatchError,
        DesignViewpointError,
        KeyError,
        OSError,
        TypeError,
    ) as exc:
        print(
            json.dumps({"error": str(exc), "output_written": False}, ensure_ascii=False)
        )
        return 2
    manifest = _load_json(output / "batch_manifest.json")
    print(
        json.dumps(
            {"inputs_root": str(output), "counts": manifest["counts"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
