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
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
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


def build_specs(
    *,
    viewpoints: Sequence[DesignViewpoint],
    assets: Mapping[str, Mapping[str, Any]],
    cell_size: int,
    twins_per_cell: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not viewpoints:
        raise DesignBatchError("at least one resolved viewpoint is required")
    if cell_size < 13:
        raise DesignBatchError(
            "--cell-size must be at least 13 so both six-point single-mover "
            "groups and the off-screen point are distinct"
        )
    if not 0 <= twins_per_cell <= cell_size:
        raise DesignBatchError("--twins-per-cell must be between zero and cell size")
    specs = []
    cells = _cells()
    for cell_index, (kind, asset1, asset2, order) in enumerate(cells):
        for made in range(cell_size):
            point_id = f"b{cell_index * cell_size + made + 1:03d}"
            seed = cell_index * cell_size + made
            viewpoint = viewpoints[(cell_index + made) % len(viewpoints)]
            if made < 6:
                motion = "s1_moving_s2_static"
                path1, path2 = find_combo(viewpoint, seed, motion)
            elif made < 12:
                motion = "s1_static_s2_moving"
                path1, path2 = find_combo(viewpoint, seed, motion)
            elif made < cell_size - 1:
                motion = "both_moving"
                path1, path2 = find_combo(viewpoint, seed, motion)
            else:
                motion = "s1_moving_s2_static"
                path1, path2 = find_combo(viewpoint, seed, "offscreen")
            short1 = _program_short(asset1, assets)
            short2 = _program_short(asset2, assets)
            pair_kind = "two_human" if kind == "human" else "two_dog"
            suffix = (
                "_turn_taking_v1" if order == "afirst" else "_turn_taking_bfirst_v1"
            )
            specs.append(
                {
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
                    "viewpoint_id": viewpoint.viewpoint_id,
                    "viewpoint_authority": viewpoint.authority,
                    "m1_request": viewpoint.spec_m1_request(),
                }
            )

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
    return [
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
        "--human-start-ue-cm",
        *map(str, spec["s1_start"]),
        "--human-end-ue-cm",
        *map(str, spec["s1_end"]),
        "--beagle-start-ue-cm",
        *map(str, spec["s2_start"]),
        "--beagle-end-ue-cm",
        *map(str, spec["s2_end"]),
        "--output",
        str(point_directory / "timeline.json"),
    ]


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
    specs, twins = build_specs(
        viewpoints=viewpoints,
        assets=assets,
        cell_size=args.cell_size,
        twins_per_cell=args.twins_per_cell,
    )
    viewpoint_by_id = {item.viewpoint_id: item for item in viewpoints}
    generated_at = datetime.now(timezone.utc).isoformat()
    policy, destination, staging = _fresh_staging(args.output_root)
    selection_cache: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for spec in specs:
            point_directory = staging / spec["point_id"]
            point_directory.mkdir()
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

        distribution = Counter(spec["viewpoint_id"] for spec in specs)
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
            "counts": {
                "total": len(specs),
                "authored": len(specs),
                "twins": len(twins),
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
    parser.add_argument("--cell-size", type=int, default=16)
    parser.add_argument("--twins-per-cell", type=int, default=8)
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
