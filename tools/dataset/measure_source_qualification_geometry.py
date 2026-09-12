#!/usr/bin/env python3
"""Measure the geometry observations the source-asset qualification matrix reads.

Three catalogs come out of this tool, each one a fresh measurement of real
files:

``assets``
    asset contact geometry for every registered asset, read from the
    registered GLB.  Articulated actors additionally use the retained
    per-frame skinned grounding audit when one is supplied.
``surfaces``
    room support surfaces back-projected from a retained native metric depth
    and semantic frame.
``clearance``
    room collision for each planned placement, tested against the room's own
    visual triangle mesh.

``run`` does all three from one configuration file.  Nothing here starts a
simulator, a render context or a native query, and nothing here writes a
qualification status: the catalogs are observation inputs for
``avengine.dataset.source_asset_qualification.build_qualification_matrix``,
which derives every status itself.

Examples
--------
    python tools/dataset/measure_source_qualification_geometry.py assets \\
        --registry examples/runtime/source_asset_runtime_profiles.json \\
        --output out/geometry_catalog.json

    python tools/dataset/measure_source_qualification_geometry.py run \\
        --config my_room.json --output-dir out/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from avengine.assets.qualification_geometry import (
    build_geometry_measurement_catalog,
    build_support_surface_catalog,
    annotate_placement_plan_room_collision,
    backproject_frame,
    capture_camera_model,
    discover_articulated_package,
    load_room_triangles,
    measure_asset_mesh_room_intersection,
    measure_native_foot_contact,
    measure_rigid_contact_offset,
    plan_root_contact_correction,
    query_support_under_footprint,
    resolve_room_visual_geometry,
    resolve_world_identity,
    DEFAULT_CONTACT_TOLERANCE_M,
    DEFAULT_FOOTPRINT_COVERAGE_FRACTION,
)


def _write_fresh(path: Path, value: Any) -> Path:
    """Write a new file, refusing to replace an existing one."""
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace an existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _grounding_audits(value: Any) -> dict[str, Any]:
    """Resolve ``{asset_id: path}`` into loaded grounding audits."""
    audits: dict[str, Any] = {}
    for asset_id, source in (value or {}).items():
        path = Path(source)
        if not path.is_file():
            continue
        audits[str(asset_id)] = _load(path)
    return audits


def command_assets(args: argparse.Namespace) -> int:
    catalog = build_geometry_measurement_catalog(
        args.registry,
        asset_ids=args.asset_id or None,
        prior_catalogs=args.prior_catalog or (),
        grounding_audits=_grounding_audits(
            _load(args.grounding_audits) if args.grounding_audits else {}
        ),
        grounding_audit_dir=args.grounding_audit_dir,
    )
    _write_fresh(Path(args.output), catalog)
    print(
        json.dumps(
            {
                "command": "assets",
                "output": str(Path(args.output).resolve()),
                "measured": catalog["measured_asset_count"],
                "requested": catalog["requested_asset_count"],
                "unmeasured": len(catalog["unmeasured"]),
            }
        )
    )
    return 0


def command_surfaces(args: argparse.Namespace) -> int:
    request = _load(args.config)
    catalog = build_support_surface_catalog(
        room=request["room"],
        capture_root=request["capture_root"],
        capture_request=request["capture_request"],
        surfaces=request["surfaces"],
        frame_index=int(request.get("frame_index", 0)),
        actor_semantic_ids=request.get("actor_semantic_ids", ()),
        actor_mask_fields=request.get("actor_mask_fields", ()),
    )
    _write_fresh(Path(args.output), catalog)
    print(
        json.dumps(
            {
                "command": "surfaces",
                "output": str(Path(args.output).resolve()),
                "fitted": len(catalog["layout"]["support_surfaces"]),
                "unfitted": len(catalog["unfitted_requests"]),
            }
        )
    )
    return 0


def command_clearance(args: argparse.Namespace) -> int:
    vertices, triangles, evidence = load_room_triangles(
        args.scene_glb, scene_dataset_config=args.scene_dataset_config
    )
    written: list[str] = []
    for source in args.placement_plan:
        annotated = annotate_placement_plan_room_collision(
            source,
            room_vertices=vertices,
            room_triangles=triangles,
            scene_ref=str(args.scene_glb),
            floor_reference_m=args.floor_reference_m,
            contact_tolerance_m=args.contact_tolerance_m,
        )
        annotated["c05_room_collision_annotation"]["room_mesh"] = evidence
        destination = Path(args.output_dir) / f"{Path(source).stem}_room_collision.json"
        _write_fresh(destination, annotated)
        written.append(str(destination.resolve()))
    print(
        json.dumps(
            {
                "command": "clearance",
                "outputs": written,
                "room_triangle_count": evidence["triangle_count"],
            }
        )
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    request = _load(args.config)
    out = Path(args.output_dir)
    results: dict[str, Any] = {"command": "run", "output_dir": str(out.resolve())}

    assets_request = request.get("assets")
    if assets_request:
        catalog = build_geometry_measurement_catalog(
            assets_request["registry"],
            asset_ids=assets_request.get("asset_ids"),
            prior_catalogs=assets_request.get("prior_catalogs", ()),
            grounding_audits=_grounding_audits(assets_request.get("grounding_audits")),
            grounding_audit_dir=assets_request.get("grounding_audit_dir"),
        )
        _write_fresh(out / "geometry_measurement_catalog.json", catalog)
        results["assets"] = {
            "measured": catalog["measured_asset_count"],
            "unmeasured": len(catalog["unmeasured"]),
        }

    surfaces_request = request.get("surfaces")
    if surfaces_request:
        catalog = build_support_surface_catalog(
            room=surfaces_request["room"],
            capture_root=surfaces_request["capture_root"],
            capture_request=surfaces_request["capture_request"],
            surfaces=surfaces_request["surfaces"],
            frame_index=int(surfaces_request.get("frame_index", 0)),
            actor_semantic_ids=surfaces_request.get("actor_semantic_ids", ()),
            actor_mask_fields=surfaces_request.get("actor_mask_fields", ()),
        )
        _write_fresh(out / "support_surface_catalog.json", catalog)
        results["surfaces"] = {
            "fitted": len(catalog["layout"]["support_surfaces"]),
            "unfitted": len(catalog["unfitted_requests"]),
        }

    clearance_request = request.get("clearance")
    if clearance_request:
        vertices, triangles, evidence = load_room_triangles(
            clearance_request["scene_glb"],
            scene_dataset_config=clearance_request.get("scene_dataset_config"),
        )
        annotated_paths: list[str] = []
        for source in clearance_request["placement_plans"]:
            annotated = annotate_placement_plan_room_collision(
                source,
                room_vertices=vertices,
                room_triangles=triangles,
                scene_ref=str(clearance_request["scene_glb"]),
                floor_reference_m=clearance_request.get("floor_reference_m"),
                contact_tolerance_m=float(
                    clearance_request.get("contact_tolerance_m", DEFAULT_CONTACT_TOLERANCE_M)
                ),
            )
            annotated["c05_room_collision_annotation"]["room_mesh"] = evidence
            destination = out / f"{Path(source).stem}_room_collision.json"
            _write_fresh(destination, annotated)
            annotated_paths.append(str(destination.resolve()))
        results["clearance"] = {
            "plans": annotated_paths,
            "room_triangle_count": evidence["triangle_count"],
        }

    _write_fresh(out / "measurement_run_readback.json", results)
    print(json.dumps(results))
    return 0



def command_foot_contact(args: argparse.Namespace) -> int:
    """World foot and sole positions for the actors a capture actually ran."""
    import numpy as np

    request = _load(args.config)
    capture_root = Path(request["capture_root"])
    registry = {
        record["asset_id"]: record
        for record in _load(request["registry"])["assets"]
    }
    identities = _load(capture_root / "neutral_readback.json")["entity_identities"]

    floor_points = None
    floor_semantic_ids = request.get("floor_semantic_ids")
    if floor_semantic_ids and request.get("capture_request"):
        camera = capture_camera_model(
            capture_root, request["capture_request"], frame_index=int(request.get("frame_index", 0))
        )
        depth = np.asarray(
            np.load(capture_root / "depth.npy", mmap_mode="r")[camera["frame_index"]]
        )
        semantic = np.asarray(
            np.load(capture_root / "semantic.npy", mmap_mode="r")[camera["frame_index"]]
        )
        world, rows, cols = backproject_frame(depth, camera)
        floor_points = world[np.isin(semantic[rows, cols], np.asarray(floor_semantic_ids))]

    actors = []
    for index, actor_id in enumerate(sorted(identities)):
        asset_id = identities[actor_id]["asset_id"]
        package = discover_articulated_package(
            registry[asset_id]["runtime_backends"]["habitat"]["glb_path"]
        )
        actors.append(
            measure_native_foot_contact(
                package_root=package,
                capture_root=capture_root,
                actor_index=index,
                actor_id=actor_id,
                asset_id=asset_id,
                floor_points_world=floor_points,
                frame_indices=request.get("frame_indices"),
            )
        )
    payload = {
        "schema": "avengine_c05_native_foot_contact_v1",
        "qualification_claim": False,
        "claim_boundary": (
            "measured world foot and sole positions in the frames that actually ran, "
            "compared against the measured visual floor where one exists; no floor "
            "convention is chosen and no dimension status is written"
        ),
        "capture_root": str(capture_root),
        "world_id": request.get("world_id"),
        "actors": actors,
    }
    _write_fresh(Path(args.output), payload)
    measured = sum(1 for row in actors if row.get("measurement") == "measured")
    print(json.dumps({"command": "foot-contact", "output": str(Path(args.output).resolve()),
                      "actors": len(actors), "measured": measured}))
    return 0


def command_narrowphase(args: argparse.Namespace) -> int:
    """Exact triangle intersection between placed asset meshes and the room."""
    request = _load(args.config)
    registry = {
        record["asset_id"]: record for record in _load(request["registry"])["assets"]
    }
    surfaces = {
        surface["surface_id"]: surface
        for path in request.get("support_catalogs", [])
        for surface in _load(path)["layout"]["support_surfaces"]
    }
    vertices, triangles, mesh_evidence = load_room_triangles(
        request["scene_glb"], scene_dataset_config=request.get("scene_dataset_config")
    )
    wanted = set(request.get("asset_id_contains", []))
    results = []
    for plan_path in request["placement_plans"]:
        plan = _load(plan_path)
        for instance in plan.get("instances", []):
            if instance.get("status") != "planned":
                continue
            asset_id = instance["asset_id"]
            if wanted and not any(token in asset_id for token in wanted):
                continue
            surface_id = (instance.get("support_identity") or {}).get("surface_id")
            surface = surfaces.get(surface_id)
            support_plane = None
            if surface:
                support_plane = {
                    "surface_id": surface_id,
                    "origin_m": surface["origin_m"],
                    "normal_m": surface["normal_m"],
                    "plane_residual_q95_m": surface["evidence"][
                        "fit_plane_residual_q50_q95_q99_m"
                    ][1],
                    "source_ref": surface_id,
                }
            row = measure_asset_mesh_room_intersection(
                asset_glb=registry[asset_id]["runtime_backends"]["habitat"]["glb_path"],
                root_transform_matrix_row_major=instance["root_transform"][
                    "matrix_row_major"
                ],
                room_vertices=vertices,
                room_triangles=triangles,
                support_plane=support_plane,
                scene_ref=request["scene_glb"],
                asset_id=asset_id,
            )
            row["episode_id"] = plan.get("episode_id")
            row["instance_id"] = instance.get("instance_id")
            results.append(row)
    payload = {
        "schema": "avengine_c05_mesh_room_narrowphase_v1",
        "qualification_claim": False,
        "claim_boundary": (
            "an exact triangle-level intersection between one placed asset mesh and the "
            "room's visual mesh; it decides interpenetration for those two meshes and "
            "nothing else, and writes no qualification status"
        ),
        "scene_glb": request["scene_glb"],
        "room_mesh": mesh_evidence,
        "results": results,
    }
    _write_fresh(Path(args.output), payload)
    print(json.dumps({
        "command": "narrowphase", "output": str(Path(args.output).resolve()),
        "placements": len(results),
        "penetrating": sum(1 for r in results if r.get("status") == "fail"),
    }))
    return 0



def command_support_correction(args: argparse.Namespace) -> int:
    """Support levels under each planned footprint, and the root a fix would need.

    The navigation floor must be supplied in the configuration and must come from
    the room or plan's own navigation component; this command never derives a
    storey from the root it is correcting.
    """
    import numpy as np

    request = _load(args.config)
    identity = resolve_world_identity(
        plan_path=request.get("episode_plan"),
        capture_root=request.get("capture_root"),
        stage_root=request.get("stage_root"),
        declared_world_id=request.get("declared_world_id"),
    )
    geometry = resolve_room_visual_geometry(request["episode_plan"])
    if geometry.get("measurement") != "measured":
        payload = {
            "schema": "avengine_c05_contact_correction_v1",
            "qualification_claim": False,
            "claim_boundary": (
                "the room's own visual geometry is absent, so no support was measured "
                "and no correction was produced"
            ),
            "world_identity": identity,
            "room_visual_geometry": geometry,
            "placements": [],
        }
        _write_fresh(Path(args.output), payload)
        print(json.dumps({"command": "support-correction",
                          "output": str(Path(args.output).resolve()),
                          "room_visual_geometry": "not_run",
                          "reason": geometry.get("reason")}))
        return 0

    vertices, triangles, mesh_evidence = load_room_triangles(
        geometry["scene_glb"], scene_dataset_config=geometry["scene_dataset_config"]
    )
    registry = {
        record["asset_id"]: record for record in _load(request["registry"])["assets"]
    }
    coverage = float(request.get("coverage_fraction", DEFAULT_FOOTPRINT_COVERAGE_FRACTION))
    wanted = set(request.get("asset_id_contains", []))
    placements = []
    for plan_path in request["placement_plans"]:
        plan = _load(plan_path)
        for instance in plan.get("instances", []):
            if instance.get("status") != "planned":
                continue
            asset_id = instance["asset_id"]
            if wanted and not any(token in asset_id for token in wanted):
                continue
            matrix = instance["root_transform"]["matrix_row_major"]
            bounds = instance["asset_bounds"]
            low = np.asarray(bounds["world_aabb_min_m"], dtype=float)
            high = np.asarray(bounds["world_aabb_max_m"], dtype=float)
            offset = measure_rigid_contact_offset(
                registry[asset_id]["runtime_backends"]["habitat"]["glb_path"], matrix
            )
            support = query_support_under_footprint(
                room_vertices=vertices,
                room_triangles=triangles,
                centre_world_m=[
                    float((low[0] + high[0]) / 2), float(matrix[7]),
                    float((low[2] + high[2]) / 2),
                ],
                footprint_extent_m=[float(high[0] - low[0]), float(high[2] - low[2])],
                navigation_floor_m=float(request["navigation_floor_m"]),
                navigation_floor_source=str(request["navigation_floor_source"]),
                search_ceiling_m=float(high[1]),
                coverage_fraction=coverage,
                scene_ref=geometry["scene_glb"],
            )
            placements.append({
                "episode_id": plan.get("episode_id"),
                "instance_id": instance.get("instance_id"),
                "asset_id": asset_id,
                "contact_offset": offset,
                "support_query": support,
                "correction": plan_root_contact_correction(
                    support_query=support,
                    contact_offset=offset,
                    current_root_world_m=[matrix[3], matrix[7], matrix[11]],
                    target_gap_m=float(request.get("target_gap_m", 0.0)),
                    identity={"room_id": identity.get("room_id"),
                              "episode_id": plan.get("episode_id")},
                ),
            })
    payload = {
        "schema": "avengine_c05_contact_correction_v1",
        "qualification_claim": False,
        "claim_boundary": (
            "measured support levels and the root heights a new plan would need; every "
            "correction is a CPU prediction for a plan that has not run"
        ),
        "world_identity": identity,
        "room_visual_geometry": geometry,
        "room_mesh": mesh_evidence,
        "coverage_fraction": coverage,
        "navigation_floor_used": {
            "floor_height_m": float(request["navigation_floor_m"]),
            "source": str(request["navigation_floor_source"]),
            "note": "supplied by the caller; never inferred from the root being corrected",
        },
        "placements": placements,
    }
    _write_fresh(Path(args.output), payload)
    corrected = sum(
        1 for row in placements if row["correction"].get("prediction") == "cpu_prediction"
    )
    print(json.dumps({
        "command": "support-correction", "output": str(Path(args.output).resolve()),
        "placements": len(placements), "corrections_predicted": corrected,
        "coverage_fraction": coverage,
    }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    assets = sub.add_parser("assets", help="measure asset contact geometry")
    assets.add_argument("--registry", required=True)
    assets.add_argument("--asset-id", action="append", default=[])
    assets.add_argument("--prior-catalog", action="append", default=[])
    assets.add_argument("--grounding-audits", help="JSON map of asset id to audit path")
    assets.add_argument(
        "--grounding-audit-dir",
        help="run the skinned grounding audit for articulated actors into this directory",
    )
    assets.add_argument("--output", required=True)
    assets.set_defaults(handler=command_assets)

    surfaces = sub.add_parser("surfaces", help="fit room support surfaces from a capture")
    surfaces.add_argument("--config", required=True)
    surfaces.add_argument("--output", required=True)
    surfaces.set_defaults(handler=command_surfaces)

    clearance = sub.add_parser("clearance", help="measure room collision for placements")
    clearance.add_argument("--scene-glb", required=True)
    clearance.add_argument(
        "--scene-dataset-config",
        required=True,
        help="scene dataset config declaring the stage up/front axes",
    )
    clearance.add_argument("--placement-plan", action="append", required=True)
    clearance.add_argument("--floor-reference-m", type=float)
    clearance.add_argument(
        "--contact-tolerance-m", type=float, default=DEFAULT_CONTACT_TOLERANCE_M
    )
    clearance.add_argument("--output-dir", required=True)
    clearance.set_defaults(handler=command_clearance)

    foot = sub.add_parser(
        "foot-contact", help="world foot and sole positions from a retained capture"
    )
    foot.add_argument("--config", required=True)
    foot.add_argument("--output", required=True)
    foot.set_defaults(handler=command_foot_contact)

    narrow = sub.add_parser(
        "narrowphase", help="exact asset mesh versus room mesh intersection"
    )
    narrow.add_argument("--config", required=True)
    narrow.add_argument("--output", required=True)
    narrow.set_defaults(handler=command_narrowphase)

    support = sub.add_parser(
        "support-correction",
        help="support levels under planned footprints and the root a fix would need",
    )
    support.add_argument("--config", required=True)
    support.add_argument("--output", required=True)
    support.set_defaults(handler=command_support_correction)

    run = sub.add_parser("run", help="run every measurement from one configuration")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.set_defaults(handler=command_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
