#!/usr/bin/env python3
"""Select, measure and report a CPU-only house acoustic acceptance batch.

The manifest supplies every external path. Native RLR is reused from v6.
Generated packages are research assets; an acoustic pass does not count an Episode.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY))
import numpy as np
from avengine.acoustics.house_acceptance_metrics import rir_metrics, foa_direction, distance_attenuation


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def choose_rooms(available, count=3):
    choices = [r for r in available if (r.get("metrics") or {}).get("placement", {}).get("found")]
    selected, seen = [], set()
    choices.sort(key=lambda r: r["metrics"].get("floor_area_m2", 0))
    if not choices:
        return []
    # Small, large and a new room type, all chosen before acoustics are observed.
    selected.append(choices[0]); seen.add(choices[0].get("room_type", "unknown"))
    if len(choices) > 1:
        selected.append(choices[-1]); seen.add(choices[-1].get("room_type", "unknown"))
    for r in sorted(choices[1:-1], key=lambda r: (r.get("room_type") in seen, r["room_label"])):
        if len(selected) == count:
            break
        selected.append(r)
    return selected[:count]


def compact_room(r):
    m = r["metrics"]
    return {"room_label": r["room_label"], "region_id": r["region_id"],
            "room_type": r.get("room_type", "unknown"),
            "floor_area_m2": m.get("floor_area_m2"), "placement": m["placement"],
            "floor_polygon": m.get("floor_polygon"), "human": r.get("human"),
            "source_registry": r.get("legacy_rooms_source")}


def prepare(args):
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    candidate = read(args.candidates)
    clean = set(read(args.clean_houses))
    registry = rows(args.registry)
    exposure = read(args.exposure_inventory)
    by_key = {(r["house"], r["room_label"]): r for r in registry}
    candidate_by_house = defaultdict(list)
    for r in candidate["rooms"]:
        if r["house"] not in clean:
            continue
        full = by_key[(r["house"], r["room_label"])]
        if full.get("human", {}).get("verdict") != "use":
            raise ValueError("Candidate lost its use verdict")
        candidate_by_house[r["house"]].append(full)
    houses = []
    for house, rs in candidate_by_house.items():
        h = rs[0]
        sid = house.rsplit("_", 1)[-1]
        if sid in exposure["so_overlap"]["houses"]["hm3d"]:
            raise ValueError("Clean list conflicts with known SO exposure")
        houses.append({"house": house, "family": "hm3d",
                       "original_split": house.split("_")[1], "scan_id": sid,
                       "scene_directory": h["scene_directory"],
                       "semantic_source": h["semantic_source"],
                       "annotation_source": h["annotation_source"],
                       "navmesh_source": h["navmesh_source"],
                       "candidate_room_count": len(rs),
                       "candidate_area_sum_m2": sum(r["metrics"]["floor_area_m2"] for r in rs),
                       "room_types": sorted({r.get("room_type", "unknown") for r in rs}),
                       "floor_count_proxy": len({round(r["metrics"]["placement"]["camera_m"][1] / .3)
                                                 for r in rs}),
                       "known_exposure": False, "clean_list_source": str(args.clean_houses),
                       "acoustic_rooms": [compact_room(r) for r in choose_rooms(rs)],
                       "registry_source": str(args.registry)})
    selected = []
    for split, count in (("train", args.hm3d_train), ("val", args.hm3d_val)):
        group = sorted((h for h in houses if h["original_split"] == split),
                       key=lambda h: (h["candidate_area_sum_m2"], h["house"]))
        indices = np.linspace(0, len(group) - 1, min(count, len(group))).round().astype(int)
        selected.extend(group[i] for i in sorted(set(indices)))
    demo = read(args.mp3d_demo_inputs)
    mp = rows(args.mp3d_registry)
    for h in demo["mp3d"]:
        rs = [r for r in mp if r["house"] == h["house"]]
        if not rs:
            raise ValueError("Demo measurement missing: " + h["house"])
        first = rs[0]
        selected.append({"house": h["house"], "family": "mp3d", "scan_id": h["id"],
                         "original_split": None, "scene_directory": first["scene_directory"],
                         "semantic_source": first["semantic_source"],
                         "annotation_source": first["annotation_source"],
                         "navmesh_source": first["navmesh_source"],
                         "candidate_room_count": sum((r.get("metrics") or {}).get("placement", {}).get("found", False) for r in rs),
                         "candidate_area_sum_m2": sum((r.get("metrics") or {}).get("floor_area_m2", 0) for r in rs),
                         "room_types": sorted({r.get("room_type", "unknown") for r in rs}),
                         "floor_count_proxy": len(h["native_levels"]),
                         "known_exposure": h["id"] in exposure["so_overlap"]["houses"]["mp3d"],
                         "known_exposure_source": exposure["so_overlap"]["source"],
                         "acoustic_rooms": [compact_room(r) for r in choose_rooms(rs)],
                         "registry_source": str(args.mp3d_registry)})
    sample_ep = read(args.v6_episode_request)
    sim = read(args.v6_simulation_request)["simulation"]
    receipt = read(args.v6_audio_receipt)
    actual_sim = receipt["qa"]["propagation"]["simulation"]
    if sim != actual_sim:
        changed = {k: [sim.get(k), actual_sim.get(k)] for k in actual_sim if sim.get(k) != actual_sim[k]}
    else:
        changed = {}
    # The receipt records the final CLI overrides, including disabled diffraction.
    manifest = {"created_at": datetime.now().astimezone().isoformat(),
                "code_base": args.code_base, "houses": selected,
                "selection": {"hm3d_all_clean_candidate_houses": len(houses),
                              "method": "area quantile selection within original train/val, then small/large/type-diverse rooms",
                              "house_features_are_candidate_room_coverage_not_entire_house_area": True,
                              "selected_before_acoustic_measurement": True},
                "runtime": {k: sample_ep["runtime"][k] for k in ("runtime_prefix", "magnum_python_site", "rlr_sdk_root")},
                "simulation": actual_sim, "v6_cli_overrides": changed,
                "v6_evidence": {k: str(getattr(args, k)) for k in ("v6_episode_request", "v6_simulation_request", "v6_audio_receipt")},
                "material_rules": {"hm3d": str(args.hm3d_material_rules), "mp3d": str(args.mp3d_material_rules)},
                "material_seeds": {"hm3d": 20260905, "mp3d": 917},
                "source_radius_m": 0.0, "listener_radius_m": 0.0,
                "criteria": {"primary_rt60_mid_s": [.2, 1.5], "long_tail_rt60_mid_s": [1.5, 2.5],
                             "minimum_rirs": 6, "minimum_listeners": 2, "minimum_valid_rt_fraction": .7,
                             "leak_pass_max": .05, "leak_fail_above": .15,
                             "ray_count_per_origin": 512,
                             "foa_angle_pass_deg": 15, "foa_angle_fail_deg": 30,
                             "distance_exponent_pass": [.7, 1.3], "tail_terminal_max_db": -40,
                             "criteria_origin": "Astra RT60 range; remaining thresholds are premeasurement engineering starting points"}}
    write(out / "acceptance_manifest.json", manifest)
    write(out / "candidate_selection.json", {"selected_houses": [h["house"] for h in selected],
                                            "all_hm3d_candidate_features": houses})
    print(json.dumps({"selected": len(selected), "families": dict(Counter(h["family"] for h in selected)),
                      "known_exposure": [h["house"] for h in selected if h["known_exposure"]]}))


def room_manifest(h, dataset_config):
    # Only change the concrete external assets in v6's room manifest schema.
    sid = h["scan_id"]
    scene = str(Path(h["scene_directory"]) / (sid + ".glb"))
    return {"schema": "avengine_room_package_v1", "room_id": h["house"], "room_kind": "habitat_native",
            "geometry_representation": "real_surface_mesh",
            "coordinate_system": {"handedness": "right", "linear_unit": "meter", "up_axis": "+Y",
                                  "forward_axis": "-Z", "quaternion_order": "xyzw"},
            "assets": [{"role": role, "path": path, "license": h["family"].upper() + " dataset terms",
                        "redistribution": "external_test_asset_not_committed"}
                       for role, path in (("render_surface_mesh", scene), ("semantic_surface_mesh", h["semantic_source"]),
                                          ("semantic_descriptor", h["annotation_source"]), ("navmesh", h["navmesh_source"]), ("scene_dataset_config", str(dataset_config)))],
            "scene": {"scene_id": scene, "scene_id_kind": "path", "enable_physics": False,
                      "dataset_config_path": str(dataset_config),
                      "load_semantic_mesh": True, "navmesh_path": h["navmesh_source"], "navmesh_policy": "load_declared"},
            "navigation": {"agent_height_m": 1.5, "agent_radius_m": .1, "include_static_objects": False},
            "connectivity_pairs": [{"pair_id": "declared_source_witness_pair_not_recomputed",
                                    "start_m": [h["acoustic_rooms"][0]["placement"]["source_1_m"][0],
                                                h["acoustic_rooms"][0]["placement"]["source_1_m"][1]-1.2,
                                                h["acoustic_rooms"][0]["placement"]["source_1_m"][2]],
                                    "end_m": [h["acoustic_rooms"][0]["placement"]["source_2_m"][0],
                                              h["acoustic_rooms"][0]["placement"]["source_2_m"][1]-1.2,
                                              h["acoustic_rooms"][0]["placement"]["source_2_m"][2]]}],
            "ray_checks": [], "openings": [],
            "acoustics": {"status": "deferred_to_m3", "reason": "RLR research package is compiled below"},
            "provenance": {"source": h["registry_source"], "source_revision": "existing dataset source; CPU audit"},
            "semantics": {"interpretation": "same v6 semantic mesh parser and canonicalization"}}


def spherical_directions(n):
    k = np.arange(n)
    y = 1 - 2 * (k + .5) / n
    phi = k * np.pi * (3 - np.sqrt(5))
    return np.stack((np.sqrt(1 - y*y) * np.cos(phi), y, np.sqrt(1 - y*y) * np.sin(phi)), axis=1)


def ray_checks(context, origins, bounds, n=512):
    directions = spherical_directions(n)
    maximum = max(30., float(np.linalg.norm(bounds[1] - bounds[0])) * 2)
    result = []
    for room, role, origin in origins:
        escaped, near = [], 0
        for i, direction in enumerate(directions):
            hit = context.trace_ray_first_hit(list(origin), direction.tolist(), .01, maximum)
            # A raw geometric escape is never explained away as a semantic window.
            if not hit.hit:
                endpoint = np.asarray(origin) + maximum * direction
                positive = [(bounds[side, ax] - origin[ax]) / direction[ax]
                            for ax in range(3) if abs(direction[ax]) > 1e-10
                            for side in (0, 1)
                            if (bounds[side, ax] - origin[ax]) / direction[ax] > 0]
                shell = np.asarray(origin) + (min(positive) if positive else maximum) * direction
                escaped.append({"direction_index": i, "direction": direction.tolist(),
                                "aabb_exit_m": shell.tolist(), "end_m": endpoint.tolist()})
            elif hit.distance < .05:
                near += 1
        result.append({"room_label": room, "role": role, "origin_m": list(origin),
                       "ray_count": n, "escape_count": len(escaped), "escape_fraction": len(escaped) / n,
                       "near_surface_hits": near, "escaped_rays": escaped,
                       "hole_localization": "intersection with scan AABB is a locator, not a recovered hole boundary"})
    return result


def alternate_listener(placement):
    listener = np.asarray(placement["camera_m"],float)
    delta = np.asarray(placement["source_1_m"],float)-listener
    delta[1]=0.
    delta /= np.linalg.norm(delta)
    return (listener+.4*delta).tolist()



def measure(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' for this CPU-only tool")
    manifest = read(args.manifest)
    h = next(h for h in manifest["houses"] if h["house"] == args.house)
    out = args.output / "houses" / h["house"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"house": h["house"], "family": h["family"], "status": "review",
              "known_exposure": h["known_exposure"], "reasons": [], "rirs": [],
              "input": h, "compute_device": "CPU", "gpu_renderer_created": False}
    try:
        if not h["acoustic_rooms"]:
            raise ValueError("NO_PLACEMENT_WITNESS")
        from avengine.acoustics.compiler import compile_hm3d_semantic_research_scene, compile_mp3d_semantic_research_scene
        from avengine.acoustics.runtime import load_compiled_acoustic_scene, RLRSimulationConfig
        from avengine.acoustics.rir_cache import _NativeRIRBatchRenderer
        rm = out / "room_manifest.json"
        dataset_config = out / "scene_dataset_config.json"
        write(dataset_config, {"stages": {"paths": {".glb": [str(Path(h["scene_directory"]) / (h["scan_id"] + ".glb"))]}},
                               "note": "asset index for manifest completeness; no Habitat visual Simulator is constructed"})
        write(rm, room_manifest(h, dataset_config))
        origin = [r["placement"]["camera_m"] for r in h["acoustic_rooms"]]
        compile_func = compile_hm3d_semantic_research_scene if h["family"] == "hm3d" else compile_mp3d_semantic_research_scene
        stamp = time.monotonic()
        if args.reuse_package:
            package = args.reuse_package
            coverage = package.parent / "semantic_material_coverage.json"
            result["compile_mode"] = "reused_task_package"
            result["initial_compilation_evidence"] = str(package.parent.parent / "result.json")
        else:
            package, coverage = compile_func(room_manifest=rm, material_rules=manifest["material_rules"][h["family"]],
                                            output=out / "package", seed=manifest["material_seeds"][h["family"]],
                                            probe_origins=origin, probe_direction_count=16)
            result["compile_mode"] = "compiled_now"
        result["acoustic_package_manifest"] = str(package)
        result["compile_seconds"] = time.monotonic() - stamp
        result["material_coverage_path"] = str(coverage)
        scene = load_compiled_acoustic_scene(package, allow_nonpassing_research_qa=True)
        if scene.manifest["source_room"]["room_id"] != h["house"]:
            raise ValueError("Reused package belongs to a different house")
        if args.omit_degenerate_for_diagnostics:
            from avengine.acoustics.house_acceptance_geometry import diagnostic_scene_without_degenerate_faces
            scene, result["geometry_diagnostic"] = diagnostic_scene_without_degenerate_faces(scene, out)
        sim = RLRSimulationConfig.from_mapping(manifest["simulation"])
        first = h["acoustic_rooms"][0]["placement"]
        renderer = _NativeRIRBatchRenderer(scene, sim, batch_size=1,
                                          initial_positions_m=[first["source_1_m"]],
                                          listener_position_m=first["camera_m"],
                                          listener_orientation_wxyz=[1., 0., 0., 0.],
                                          layout_type="ambisonics", channel_count=4, hrtf_file_path="",
                                          source_radius_m=manifest["source_radius_m"],
                                          listener_radius_m=manifest["listener_radius_m"],
                                          **manifest["runtime"])
        result["native_setup"] = renderer.setup_report
        # Native BVH ray queries are defined only after the first completed simulation.
        warmup = renderer.render([first["source_1_m"]])
        result["native_ray_warmup_seconds"] = warmup.wall_seconds
        vertices = np.concatenate([np.asarray(o["vertices"]) for o in scene.objects])
        bounds = np.stack((vertices.min(0), vertices.max(0)))
        # Include actual listener and both source origins, never a bbox center outside a room.
        origins = [(r["room_label"], key, r["placement"][key]) for r in h["acoustic_rooms"]
                   for key in ("camera_m", "source_1_m", "source_2_m")]
        if len(h["acoustic_rooms"])==1:
            origins.append((h["acoustic_rooms"][0]["room_label"],"camera_alt_m",alternate_listener(first)))
        result["leakage"] = ray_checks(renderer.context, origins, bounds, manifest["criteria"]["ray_count_per_origin"])
        write(out / "leakage.json", result["leakage"])
        arrays, edcs, directions = {}, {}, []
        # Three independent clear positions for the FOA check, distributed over rooms where available.
        room_distance = []
        for r in h["acoustic_rooms"]:
            p = r["placement"]
            listener = np.asarray(p["camera_m"], float)
            far = np.asarray(p["source_1_m"], float)
            axis = far - listener
            d_far = float(np.linalg.norm(axis)); axis /= d_far
            forward = ((far - listener) / d_far +
                       (np.asarray(p["source_2_m"]) - listener) /
                       np.linalg.norm(np.asarray(p["source_2_m"]) - listener))
            # Same clear line supplies 1 m, halfway and endpoint, isolating range from angle.
            probes = [("range_near", listener + axis * min(1., d_far / 3)),
                      ("range_middle", listener + axis * d_far * .65),
                      ("source_1", far), ("source_2", np.asarray(p["source_2_m"], float))]
            if len(h["acoustic_rooms"]) == 1:
                second_axis = np.asarray(p["source_2_m"], float) - listener
                probes += [("second_range_near", listener + second_axis * .35),
                           ("second_range_middle", listener + second_axis * .7)]
            distances, energies = [], []
            for label, source in probes:
                blocked = renderer.context.trace_ray_any_hit(listener.tolist(), ((source-listener)/np.linalg.norm(source-listener)).tolist(), .01, max(.02, float(np.linalg.norm(source - listener)) - .03)).hit
                if blocked:
                    result.setdefault("occluded_probe_lines", []).append({"room_label": r["room_label"], "source_label": label,
                                                                          "listener_m": listener.tolist(), "source_m": source.tolist()})
                stamp = time.monotonic()
                batch = renderer.render([source.tolist()], listener_position_m=listener.tolist(),
                                        listener_orientation_wxyz=[1., 0., 0., 0.])
                ir = batch.samples[0]
                key = r["room_label"] + "_" + label
                arrays[key] = ir
                metrics, curves = rir_metrics(ir, int(sim.sample_rate_hz), float(np.linalg.norm(source - listener)), sim.max_ir_seconds)
                metrics.update({"probe": key, "room_label": r["room_label"], "room_type": r["room_type"],
                                "listener_m": listener.tolist(), "source_m": source.tolist(),
                                "distance_m": float(np.linalg.norm(source - listener)),
                                "render_wall_seconds": time.monotonic() - stamp,
                                "native_simulation_wall_seconds": batch.wall_seconds,
                                "native_simulation_cpu_seconds": batch.process_cpu_seconds,
                                "rir_path": str(out / "rirs.npz") + "#/" + key})
                result["rirs"].append(metrics)
                edcs[key] = curves
                if label in ("range_near", "range_middle", "source_1") and not blocked:
                    distances.append(metrics["distance_m"]); energies.append(metrics.get("direct_energy", 0))
                if label in ("source_1", "source_2"):
                    probe_forward = source - listener
                    probe_forward[1] = 0.; probe_forward /= np.linalg.norm(probe_forward)
                    yaw = np.radians(20 if label == "source_1" else -20)
                    probe_forward = np.array([probe_forward[0]*np.cos(yaw)+probe_forward[2]*np.sin(yaw),
                                              0., -probe_forward[0]*np.sin(yaw)+probe_forward[2]*np.cos(yaw)])
                    direction = foa_direction(ir, listener, source, probe_forward)
                    direction.update({"direct_line_clear": not blocked, "probe": key, "room_label": r["room_label"],
                                      "listener_m": listener.tolist(), "source_m": source.tolist()})
                    directions.append(direction)
            room_distance.append({"room_label": r["room_label"],
                                  **distance_attenuation(distances, energies)})

        if len(h["acoustic_rooms"]) == 1:
            r = h["acoustic_rooms"][0]; p = r["placement"]
            listener_alt = alternate_listener(p)
            result["additional_listener_positions_m"] = [listener_alt]
            for source_label in ("source_1_m","source_2_m"):
                source = np.asarray(p[source_label],float)
                relative = source-np.asarray(listener_alt)
                distance = float(np.linalg.norm(relative))
                if renderer.context.trace_ray_any_hit(listener_alt,(relative/distance).tolist(),.01,max(.02,distance-.03)).hit:
                    raise ValueError("ALTERNATE_LISTENER_SOURCE_LINE_OCCLUDED")
                batch = renderer.render([source.tolist()],listener_position_m=listener_alt,
                                        listener_orientation_wxyz=[1.,0.,0.,0.])
                key = r["room_label"]+"_alternate_listener_"+source_label
                ir = batch.samples[0]; arrays[key]=ir
                metrics,curves = rir_metrics(ir,int(sim.sample_rate_hz),distance,sim.max_ir_seconds)
                metrics.update({"probe":key,"room_label":r["room_label"],"room_type":r["room_type"],
                                "listener_m":listener_alt,"source_m":source.tolist(),"distance_m":distance,
                                "render_wall_seconds":batch.wall_seconds,
                                "native_simulation_wall_seconds":batch.wall_seconds,
                                "native_simulation_cpu_seconds":batch.process_cpu_seconds,
                                "rir_path":str(out/"rirs.npz")+"#/"+key})
                result["rirs"].append(metrics);edcs[key]=curves


        np.savez_compressed(out / "rirs.npz", **arrays)
        result["distance_attenuation"] = room_distance
        clear_directions = [d for d in directions if d.get("direct_line_clear", True)]
        primary_directions = [d for d in clear_directions if d["probe"].endswith("source_1")]
        result["foa_checks"] = (primary_directions + [d for d in clear_directions if d not in primary_directions])[:3]
        # Houses with one room still get a third distinct location check.
        if len(h["acoustic_rooms"]) == 1 and len(result["foa_checks"]) < 3 and directions:
            r = h["acoustic_rooms"][0]
            key = r["room_label"] + "_range_middle"
            metrics = next(m for m in result["rirs"] if m["probe"] == key)
            forward = directions[0]["camera_forward"]
            d = foa_direction(arrays[key], metrics["listener_m"], metrics["source_m"], forward)
            d.update({"probe": key, "room_label": r["room_label"],
                      "listener_m": metrics["listener_m"], "source_m": metrics["source_m"]})
            result["foa_checks"].append(d)
        result["foa_all_positions"] = directions
        classify(result, manifest["criteria"])
        plots(result, edcs, out)
    except Exception as exc:
        result["reasons"].append(type(exc).__name__ + ": " + str(exc))
        result["exception_traceback"] = traceback.format_exc()
        print(result["exception_traceback"], file=sys.stderr, flush=True)
    result["wall_seconds"] = time.monotonic() - started
    result["finished_at"] = datetime.now().astimezone().isoformat()
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("house", "status", "reasons", "wall_seconds")}, ensure_ascii=False), flush=True)


def classify(r, c):
    fail, review = [], []
    worst_leak = max(x["escape_fraction"] for x in r["leakage"])
    r["max_escape_fraction"] = worst_leak
    if worst_leak > c["leak_fail_above"]:
        fail.append("GEOMETRIC_ESCAPE_ABOVE_15_PERCENT")
    elif worst_leak > c["leak_pass_max"]:
        review.append("GEOMETRIC_ESCAPE_5_TO_15_PERCENT")
    if r.get("geometry_diagnostic"):
        review.append("DEGENERATE_INPUT_GEOMETRY_DIAGNOSTIC_ONLY")
    if r.get("occluded_probe_lines"):
        review.append("RAW_WITNESS_VS_ACOUSTIC_DIRECT_PATH_DISAGREEMENT")
    rs = r["rirs"]
    listener_positions = {tuple(round(float(v),6) for v in item["listener_m"]) for item in rs}
    r["distinct_listener_positions"] = len(listener_positions)
    if len(listener_positions)<c.get("minimum_listeners",2):
        review.append("MULTIPLE_LISTENER_POSITIONS_MISSING")
    valid = [x["rt60_mid_s"] for x in rs if x.get("status") == "valid" and x["rt60_mid_s"] is not None]
    fraction = len(valid) / max(1, len(rs))
    r["rt60_valid_fraction"] = fraction
    r["rt60_mid_median_s"] = float(np.median(valid)) if valid else None
    r["rt60_mid_range_s"] = [min(valid), max(valid)] if valid else None
    if len(rs) < c["minimum_rirs"] or fraction < c["minimum_valid_rt_fraction"]:
        review.append("INSUFFICIENT_VALID_RIRS")
    if valid:
        median = r["rt60_mid_median_s"]
        r["rt60_coverage_bin"] = "primary" if .2 <= median <= 1.5 else "long_tail" if 1.5 < median <= 2.5 else "out_of_starting_range"
        if median < .2 or median > 2.5:
            review.append("RT60_OUTSIDE_STARTING_RANGE")
        if sum(.15 <= x <= 2.5 for x in valid) / len(valid) < .8:
            review.append("RT60_POSITION_OUTLIERS")
    if any(x.get("tail_truncated") for x in rs):
        review.append("IR_TAIL_UNRESOLVED")
    errors = [x["check"]["angle_deg"] for x in r["foa_checks"] if x["check"]["angle_deg"] is not None]
    r["max_foa_angle_deg"] = max(errors) if errors else None
    if len(errors) != 3:
        review.append("FOA_THREE_POSITIONS_MISSING")
    elif max(errors) > c["foa_angle_fail_deg"]:
        fail.append("FOA_ANGLE_ABOVE_30_DEG")
    elif max(errors) > c["foa_angle_pass_deg"]:
        review.append("FOA_ANGLE_15_TO_30_DEG")
    exponents = [x["pressure_exponent"] for x in r["distance_attenuation"] if x["status"] == "valid"]
    if not exponents:
        review.append("DISTANCE_ATTENUATION_UNVERIFIED")
    elif any(not c["distance_exponent_pass"][0] <= x <= c["distance_exponent_pass"][1] for x in exponents):
        review.append("DIRECT_DISTANCE_ATTENUATION_OUTLIER")
    if any(abs(x.get("direct_delay_error_ms", 0)) > 3 for x in rs):
        review.append("DIRECT_PATH_DELAY_OUTLIER")
    r["acoustic_status"] = "fail" if fail else "review" if review else "pass"
    r["status"] = r["acoustic_status"]
    r["reasons"] = fail + review
    r["batch_eligible"] = r["status"] == "pass" and not r["known_exposure"]
    if r["known_exposure"]:
        r["batch_exclusion"] = "KNOWN_SO_HOUSE_EXPOSURE"


def plots(r, edcs, out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for l in r["leakage"]:
        origin = np.asarray(l["origin_m"])
        axs[0, 0].scatter(origin[0], origin[2], s=15)
        for ray in l["escaped_rays"][::max(1, len(l["escaped_rays"]) // 25)]:
            endpoint = np.asarray(ray["aabb_exit_m"])
            axs[0, 0].plot([origin[0], endpoint[0]], [origin[2], endpoint[2]], c="red", alpha=.15)
    axs[0, 0].set(title="Escape rays: origins to scan AABB exits (locator)", xlabel="world X (m)", ylabel="world Z (m)")
    axs[0, 0].axis("equal")
    for key, curves in edcs.items():
        for fc in (500, 1000, 2000):
            if fc in curves:
                values = curves[fc][::16]
                axs[0, 1].plot(np.arange(len(values)) * .001, values, alpha=.25)
    axs[0, 1].set(title="Schroeder decay, 500/1000/2000 Hz", xlabel="seconds from onset", ylabel="dB", ylim=(-60, 0))
    centers = (125, 250, 500, 1000, 2000, 4000)
    for x in r["rirs"]:
        axs[1, 0].plot(centers, [next((b["t20"]["rt60_s"] for b in x["octaves"] if b["center_hz"]==fc), np.nan) for fc in centers], alpha=.4, marker=".")
    axs[1, 0].set_xscale("log")
    axs[1, 0].set(title="T20 extrapolated RT60 (see validity in JSON)", xlabel="octave center Hz", ylabel="seconds")
    for label in sorted({x["room_label"] for x in r["rirs"]}):
        xs = [x for x in r["rirs"] if x["room_label"] == label and "range_" in x["probe"] or x["room_label"] == label and x["probe"].endswith("source_1")]
        axs[1, 1].plot([x["distance_m"] for x in xs],
                       [10 * np.log10(max(x.get("direct_energy",0), 1e-30)) for x in xs], marker="o", label=label)
    axs[1, 1].set(title="Direct-path distance attenuation", xlabel="distance m", ylabel="energy dB")
    axs[1, 1].legend()
    fig.suptitle(r["house"] + " / " + r["status"])
    fig.tight_layout(); fig.savefig(out / "acoustics.png", dpi=140); plt.close(fig)
    # CPU pinhole projection: cyan = geometric direction, orange = measured FOA.
    # This is explicit geometry evidence, not a fabricated Habitat RGB render.
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, d in zip(axs, r["foa_checks"]):
        if d["check"]["angle_deg"] is None:
            ax.text(.2,.5,"FOA unavailable"); continue
        g = np.radians(d["check"]["geometry_azimuth_left_positive_deg"])
        f = np.radians(d["check"]["foa_azimuth_left_positive_deg"])
        ax.scatter([-np.tan(g)], [0], marker="o", label="projected source", c="cyan", edgecolors="black")
        ax.axvline(-np.tan(f), color="orange", label="FOA intensity")
        ax.set(title=d["probe"], xlim=(-1.5, 1.5), ylim=(-1, 1), xlabel="pinhole x/f")
        ax.legend()
    fig.suptitle("v6 FOA / camera projection check; CPU geometry diagnostic")
    fig.tight_layout(); fig.savefig(out / "foa_projection.png", dpi=140); plt.close(fig)



def visual(args):
    from avengine.acoustics.house_acceptance_visual import inspect_visual
    output = args.output / "houses" / args.house
    result = read(output / "result.json")
    if any((output / name).exists() for name in ("visual_checks.json", "cpu_view_1.png", "cpu_view_2.png", "cpu_view_3.png")):
        raise FileExistsError("Visual evidence already exists; use a fresh output directory")
    if not result.get("foa_checks"):
        raise ValueError("Native RIR measurement has no FOA checks")
    try:
        measured = inspect_visual(result, output, read(args.manifest))
    except Exception as exc:
        measured = {"status": "review", "reasons": [type(exc).__name__ + ": " + str(exc)],
                    "exception_traceback": traceback.format_exc(), "compute_device": "CPU"}
        print(measured["exception_traceback"], file=sys.stderr)
    write(output / "visual_checks.json", measured)
    print(json.dumps({"house": args.house, "visual_status": measured["status"]}), flush=True)

def run(args):
    (args.output / "logs").mkdir(parents=True,exist_ok=True)
    m = read(args.manifest)
    selected = [h for h in m["houses"] if not args.house or h["house"] in args.house]
    if args.resume:
        selected = [h for h in selected if not (args.output / "houses" / h["house"] / "result.json").exists()]
    def worker(h):
        logfile = args.output / "logs" / (h["house"] + ".log")
        command = [sys.executable, str(Path(__file__).resolve()), "measure", "--manifest", str(args.manifest),
                   "--output", str(args.output), "--house", h["house"]]
        if args.omit_degenerate_for_diagnostics:
            command += ["--omit-degenerate-for-diagnostics"]
        if args.reuse_root:
            command += ["--reuse-package", str(args.reuse_root / "houses" / h["house"] / "package" / "manifest.json")]
        with logfile.open("x") as log:
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with (args.output / "logs" / (h["house"] + ".pid")).open("x") as f:
                f.write(str(p.pid) + "\n")
            try:
                rc = p.wait(timeout=args.house_timeout)
            except subprocess.TimeoutExpired:
                # Popen owns this exact child; never signal unrelated shared processes.
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
                raise RuntimeError("House timeout; stopped owned PID " + str(p.pid))
        return {"house": h["house"], "pid": p.pid, "returncode": rc, "log": str(logfile)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, h) for h in selected]
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                print(json.dumps({"error": repr(exc)}), flush=True)




def boundary(args):
    from avengine.acoustics.house_acceptance_visual import inspect_boundary_geometry
    output = args.output / "houses" / args.house
    result = read(output / "result.json")
    report = inspect_boundary_geometry(result, output)
    write(output / "boundary_review.json", report)
    print(json.dumps({"house": args.house, "boundary_edge_count": report["boundary_edge_count"],
                      "image_path": report["image_path"], "native_rir_or_rgb_claim": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("output", "candidates", "clean_houses", "registry", "exposure_inventory",
                 "mp3d_demo_inputs", "mp3d_registry", "v6_episode_request", "v6_simulation_request",
                 "v6_audio_receipt", "hm3d_material_rules", "mp3d_material_rules"):
        prep.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    prep.add_argument("--hm3d-train", type=int, default=14)
    prep.add_argument("--hm3d-val", type=int, default=8)
    prep.add_argument("--code-base", required=True)
    for cmd in ("measure", "run", "visual", "boundary"):
        p = sub.add_parser(cmd)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--house", required=cmd in ("measure", "visual", "boundary"), action="append" if cmd == "run" else "store")
        if cmd == "measure":
            p.add_argument("--reuse-package",type=Path)
            p.add_argument("--omit-degenerate-for-diagnostics",action="store_true")
        if cmd == "run":
            p.add_argument("--reuse-root",type=Path)
            p.add_argument("--omit-degenerate-for-diagnostics",action="store_true")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--house-timeout", type=int, default=5400)
    args = parser.parse_args()
    {"prepare": prepare, "measure": measure, "run": run, "visual": visual, "boundary": boundary}[args.command](args)


if __name__ == "__main__":
    main()
