"""Calibration-only paired audit of shared screening versus retained 82382e2."""

from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from shapely.geometry import shape
from tools.rooms.room_screening import compute_semantic_region_candidates as smy
from tools.rooms.room_screening.geometry import union_projected_polygons
from .geometry import dominant_layer
from .navigation import sample_navigation
from .measurements import (
    load_scene,
    region_geometry,
    layer_furniture,
    navigation_geometry,
)
from .protocol import scope_metrics
from .runtime import habitat_runtime_options
from .run import load_parameters, write_json, now


def relative_difference(old, new):
    if old == 0:
        return None if new else 0.0
    return (new - old) / old


def worker(out, previous, old_checkout, house):
    manifest = json.loads((out / "paired_comparison/sample_manifest.json").read_text())
    if house not in manifest["houses"]:
        raise ValueError("house not in declared calibration-only comparison")
    split = json.loads((previous / "house_analysis_split.json").read_text())
    if house not in split["calibration"]:
        raise ValueError("comparison may not read held-out measurements")
    job = json.loads((out / "jobs" / f"{house}.json").read_text())
    old_result = json.loads((previous / "houses" / f"{house}.json").read_text())
    _, old_p = load_parameters(previous / "thresholds.used.yaml")
    _, p = load_parameters(out / "thresholds.initial.yaml")
    # Read the old implementation without modifying/running its checkout CLI.
    spec = importlib.util.spec_from_file_location(
        "_retained_room_geometry",
        old_checkout / "tools/rooms/room_selection/geometry.py",
    )
    old_geometry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old_geometry
    spec.loader.exec_module(old_geometry)
    scene = Path(job["scene_directory"])
    old_mesh = old_geometry.load_hm3d(scene)
    mesh = load_scene(scene)
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    from tools.rooms.room_screening.build_inventory import scalar_settings

    hs = prepare_installed_habitat_runtime(**habitat_runtime_options()).habitat_sim
    pf, nav_polys, nav_ys = smy.navmesh_triangles(hs, Path(job["navmesh"]))
    settings = scalar_settings(pf.nav_mesh_settings)
    native_levels = smy.cluster_levels(nav_ys, 2 * settings["cell_height"])
    rows = []
    differences = []
    for old in old_result["rows"]:
        rid = old["region_id"]
        label = old["room_label"]
        old_layers, old_furniture, _, _ = old_geometry.region_geometry(
            old_mesh, rid, old_p
        )
        fresh_old_area = sum(g["geometry"].area for g in old_layers)
        stored_old_area = sum(
            f["metrics"]["floor_area_m2"] for f in old.get("floors", [])
        )
        if not np.isclose(fresh_old_area, stored_old_area, atol=1e-6, rtol=1e-6):
            raise ValueError(f"old area replay mismatch {house}/{label}")
        layers, _, _ = region_geometry(mesh, rid, p)
        faces = mesh.ground.get(rid, [])
        provisional = smy.collect_clusters(faces, 2 * settings["cell_height"])
        common = __import__(
            "tools.rooms.room_selection.geometry", fromlist=["floor_windows"]
        ).floor_windows(
            [f for f in faces if f["category"] in old_p["floor_categories"]], p
        )
        measurements = []
        for fi, layer in enumerate(layers):
            furniture = layer_furniture(
                mesh, rid, layer["floor_y_m"], settings["agent_height"]
            )
            m = scope_metrics(layer["geometry"], furniture)
            grid, _, _, _, _ = sample_navigation(
                pf, hs, layer["geometry"], layer["floor_y_m"], p
            )
            m.update(
                navigation_geometry(
                    nav_polys,
                    nav_ys,
                    layer["floor_y_m"],
                    layer["geometry"],
                    p,
                    shape(grid["nav_grid_main_polygon"]),
                )
            )
            m["nav_grid_main_area_m2"] = grid["nav_main_area_m2"]
            m["floor_y_m"] = layer["floor_y_m"]
            measurements.append(dict(floor_id=f"F{fi}", metrics=m))
        selected, fraction = dominant_layer(
            measurements, p["dominant_floor_area_fraction_min"]
        )
        oldsel = (old.get("floor_selection") or {}).get("selected_floor_id")
        rec = dict(
            house=house,
            room_label=label,
            old_floor_count=len(old_layers),
            smy_provisional_floor_count=len(provisional),
            merged_window_floor_count=len(layers),
            old_selected_floor_id=oldsel,
            merged_selected_floor_id=(
                measurements[selected]["floor_id"] if selected is not None else None
            ),
            merged_dominant_fraction=fraction,
            old_floor_area_m2=fresh_old_area,
            smy_provisional_floor_area_m2=sum(
                f["ground_projected_area_m2"] for f in provisional
            ),
            merged_floor_area_m2=sum(
                f["metrics"]["floor_area_m2"] for f in measurements
            ),
            common_categories_merged_floor_area_m2=sum(
                f["geometry"].area for f in common
            ),
            merged_floors=measurements,
            old_floor_replay_matches_cache=True,
        )
        for field, newkey in [("floor_area_m2", "merged_floor_area_m2")]:
            oldv = rec["old_floor_area_m2"]
            newv = rec[newkey]
            diff = relative_difference(oldv, newv)
            if (diff is None and newv > 0) or (diff is not None and abs(diff) > 0.10):
                cat_delta = newv - rec["common_categories_merged_floor_area_m2"]
                mapping_delta = rec["common_categories_merged_floor_area_m2"] - oldv
                differences.append(
                    dict(
                        house=house,
                        room_label=label,
                        metric=field,
                        old=oldv,
                        new=newv,
                        relative_difference=diff,
                        category_expansion_delta_m2=cat_delta,
                        palette_and_precision_delta_m2=mapping_delta,
                        classification="definition_and_shared_loader",
                        explanation="Expanded smy ground labels and ±2-channel unique palette matching; unrounded projection union. Component deltas above separate expanded categories from mapping/precision.",
                    )
                )
        # Height-matched per-layer comparisons, so multi-storey sums never mimic a room.
        for i, of in enumerate(old.get("floors", [])):
            om = of["metrics"]
            oy = om["floor_y_m"]
            match = min(
                measurements,
                key=lambda f: abs(f["metrics"]["floor_y_m"] - oy),
                default=None,
            )
            if (
                match is None
                or abs(match["metrics"]["floor_y_m"] - oy)
                > p["floor_height_separation_m"]
            ):
                differences.append(
                    dict(
                        house=house,
                        room_label=label,
                        old_floor_id=of["floor_id"],
                        metric="floor_alignment",
                        classification="definition",
                        explanation="No same-height merged layer; expanded labels/area-weighted window repartition. Both floor lists retained.",
                        old_height_m=oy,
                    )
                )
                continue
            nm = match["metrics"]
            mi = int(match["floor_id"][1:])
            scope = layers[mi]["geometry"]
            fresh_furniture = layer_furniture(
                mesh, rid, nm["floor_y_m"], settings["agent_height"]
            )
            common_furniture = [
                f
                for f in fresh_furniture
                if f["category"] in old_p["furniture_categories"]
            ]
            common_occ = scope_metrics(scope, common_furniture)[
                "furniture_footprint_m2"
            ]
            for field in ["furniture_footprint_m2", "nav_main_area_m2"]:
                oldv = om.get(field, 0)
                newv = nm.get(field, 0)
                diff = relative_difference(oldv, newv)
                if (diff is None and newv > 0) or (
                    diff is not None and abs(diff) > 0.10
                ):
                    note = (
                        "Smy blocker classifier, body-height clipping and shape-preserving hole-filled footprints versus old whitelist/all-height surface projection; same floor scope matched. Common-category occupancy isolates vocabulary expansion."
                        if field.startswith("furniture")
                        else "Continuous triangulated navmesh intersection on native main-component support versus old whole snapped 0.25 m cells. Cell snapping can overcount narrow boundary strips; main-component support excludes other connected components and unsampled cells. This is an area definition change, not new navmesh generation."
                    )
                    differences.append(
                        dict(
                            house=house,
                            room_label=label,
                            old_floor_id=of["floor_id"],
                            merged_floor_id=match["floor_id"],
                            metric=field,
                            old=oldv,
                            new=newv,
                            relative_difference=diff,
                            classification="definition",
                            explanation=note,
                            common_category_furniture_area_m2=(
                                common_occ if field.startswith("furniture") else None
                            ),
                            continuous_total_nav_area_m2=(
                                nm["nav_triangle_intersection_area_m2"]
                                if field.startswith("nav")
                                else None
                            ),
                            new_grid_area_m2=(
                                nm["nav_grid_main_area_m2"]
                                if field.startswith("nav")
                                else None
                            ),
                            grid_to_continuous_delta_m2=(
                                (nm["nav_main_area_m2"] - nm["nav_grid_main_area_m2"])
                                if field.startswith("nav")
                                else None
                            ),
                        )
                    )
        if len(provisional) != len(layers) or oldsel != rec["merged_selected_floor_id"]:
            differences.append(
                dict(
                    house=house,
                    room_label=label,
                    metric="floor_decision",
                    classification="definition",
                    old_floor_count=len(old_layers),
                    smy_provisional_floor_count=len(provisional),
                    merged_floor_count=len(layers),
                    explanation="Smy provisional split uses chained adjacent gaps 2*serialized cell_height and face-count medians; eligibility retains bounded 0.3 m area-weighted windows plus dominant-area fraction. Provisional clusters are diagnostics, not final floor decisions.",
                )
            )
        rows.append(rec)
    result = dict(
        house=house,
        status="complete",
        finished_at_sgt=now(),
        native_navmesh_floor_count=len(native_levels),
        native_navmesh_floor_heights_m=native_levels,
        navmesh_settings=settings,
        old_geometry_replayed=True,
        old_measurement_source=str(previous / "houses" / f"{house}.json"),
        old_diagnostics=old_mesh.diagnostics,
        shared_diagnostics=mesh.diagnostics,
        regions=rows,
        differences_over_10_percent_or_floor_changes=differences,
    )
    write_json(out / "paired_comparison/houses" / f"{house}.json", result)
    print("PAIRED_DONE", house, len(rows), len(differences), flush=True)


def aggregate(out):
    manifest = json.loads((out / "paired_comparison/sample_manifest.json").read_text())
    results = [
        json.loads((out / "paired_comparison/houses" / f"{h}.json").read_text())
        for h in manifest["houses"]
    ]
    diffs = [
        d for r in results for d in r["differences_over_10_percent_or_floor_changes"]
    ]
    summary = dict(
        status="complete",
        house_count=len(results),
        region_count=sum(len(r["regions"]) for r in results),
        calibration_only=True,
        nav_main_definition="native geodesic grid support clips shared continuous intersection",
        old_geometry_replays_passed=all(r["old_geometry_replayed"] for r in results),
        difference_counts=dict(Counter(d["metric"] for d in diffs)),
        unexplained_count=sum(not d.get("explanation") for d in diffs),
        bug_fixes=[
            "duplicate palette rows no longer overwrite instance/region",
            "furniture audit retains all region/floor rows",
            "ground and furniture use one semantic loader",
        ],
        source="paired_comparison/houses/*.json",
        differences=diffs,
        houses=[
            dict(
                house=r["house"],
                native_navmesh_floor_count=r["native_navmesh_floor_count"],
                regions=len(r["regions"]),
                old_total_floor_area_m2=sum(
                    x["old_floor_area_m2"] for x in r["regions"]
                ),
                merged_total_floor_area_m2=sum(
                    x["merged_floor_area_m2"] for x in r["regions"]
                ),
            )
            for r in results
        ],
    )
    write_json(out / "paired_comparison/summary.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous-output", type=Path)
    p.add_argument("--old-checkout", type=Path)
    p.add_argument("--house")
    a = p.parse_args()
    if a.house:
        worker(a.output, a.previous_output, a.old_checkout, a.house)
    else:
        aggregate(a.output)


if __name__ == "__main__":
    main()
