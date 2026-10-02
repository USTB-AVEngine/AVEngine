#!/usr/bin/env python3
"""Register every semantic region, measure stages 1-2 on CPU, join house gates.

Run from the AVEngine worktree with python -m tools.rooms.room_selection.run.
Only --output receives derived files. GLBs, navmeshes, room inventories and
review stamps are never opened for writing. GPU simulation is not initialized.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np
import shapely
import yaml

from . import PROTOCOL_VERSION, REPO_ROOT
from .geometry import (
    collision_mesh,
    hm3d_annotations,
    mp3d_house,
    load_hm3d,
    load_mp3d,
    region_geometry,
    infer_type,
)
from .media import load_overhead, black_metric, overlay
from .navigation import sample_navigation, placement, cells_polygon, components
from .protocol import decide, scope_metrics
from .splitting import furniture_clusters, split_triggers, propose
from tools.rooms.runtime_config import (
    TASKS_ROOT,
    MEDIA_ROOT,
    VERDICT_ROOT,
    habitat_runtime_options,
)


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def load_parameters(path):
    spec = yaml.safe_load(Path(path).read_text())
    return spec, {k: v["value"] for k, v in spec["parameters"].items()}


def house_key(scene_dir):
    s = Path(scene_dir)
    index, sid = s.name.split("-", 1)
    return f"hm3d_{s.parent.name}_{index}_{sid}"


def latest_tasks(tasks_root):
    latest = {}
    all_tasks = {}
    for path in sorted(Path(tasks_root).glob("*hm3d_end_to_end/task.json")):
        d = json.loads(path.read_text())
        argv = d.get("argv") or []
        if "--scene-dir" not in argv:
            continue
        scene = argv[argv.index("--scene-dir") + 1]
        house = house_key(scene)
        record = dict(
            task_id=d.get("task_id", path.parent.name),
            status=d.get("status", "unknown"),
            created_at=d.get("created_at", path.parent.name),
            scene_directory=scene,
            source=str(path),
        )
        all_tasks[path.parent.name] = record
        if house not in latest or (record["created_at"], record["task_id"]) > (
            latest[house]["created_at"],
            latest[house]["task_id"],
        ):
            latest[house] = record
    inventories = {}
    for path in sorted(
        Path(tasks_root).glob("*hm3d_end_to_end/output/render/rooms/*/rooms.json")
    ):
        house = path.parent.name
        tid = path.relative_to(tasks_root).parts[0]
        timestamp = all_tasks.get(tid, {}).get("created_at", tid)
        if house not in inventories or timestamp > inventories[house]["created_at"]:
            inventories[house] = dict(
                created_at=timestamp,
                source=str(path),
                rooms=json.loads(path.read_text()).get("rooms", []),
            )
    return latest, inventories


def register(args):
    spec, p = load_parameters(args.thresholds)
    inventory = json.loads(args.inventory.read_text())
    tasks, old_rooms = latest_tasks(args.tasks_root)
    reviews = {}
    for path in sorted(args.verdict_root.glob("*.json")):
        d = json.loads(path.read_text())
        reviews[(d["house"], d["room_label"])] = dict(d, source=str(path))
    jobs = []
    for h in inventory["houses"]:
        if h["family"] != args.family:
            continue
        scene = Path(h["scene_directory"])
        if args.family == "hm3d":
            sid = scene.name.split("-", 1)[1]
            ann = scene / f"{sid}.semantic.txt"
            sem = scene / f"{sid}.semantic.glb"
            nav = scene / f"{sid}.basis.navmesh"
            if not (ann.is_file() and sem.is_file() and nav.is_file()):
                continue
            instances, _, regions = hm3d_annotations(ann)
            house = house_key(scene)
        else:
            sid = scene.name
            ann = scene / f"{sid}.house"
            sem = scene / f"{sid}_semantic.ply"
            nav = scene / f"{sid}.navmesh"
            if not (ann.is_file() and sem.is_file() and nav.is_file()):
                continue
            instances, regions = mp3d_house(ann)
            house = "mp3d_" + sid
        known = {
            int(r["region_id"]): r for r in old_rooms.get(house, {}).get("rooms", [])
        }
        # The union retains any legacy review/room record whose semantic mapping is missing.
        ids = (
            set(regions)
            | set(known)
            | {
                int(label[1:])
                for (rh, label) in reviews
                if rh == house and label[1:].lstrip("-").isdigit()
            }
        )
        if not ids:
            ids = {-1}
        rows = []
        for rid in sorted(ids):
            label = f"R{rid}"
            members = [v for v in instances.values() if v["region_id"] == rid]
            row = dict(
                schema=PROTOCOL_VERSION,
                family=args.family,
                house=house,
                room_label=label,
                region_id=rid,
                registered_at_sgt=now(),
                semantic_region_present=rid in regions,
                semantic_instance_count=len(members),
                semantic_categories=sorted({m["category"] for m in members}),
                native_region=regions.get(rid),
                legacy_room=known.get(rid),
                legacy_rooms_source=old_rooms.get(house, {}).get("source"),
                human=reviews.get((house, label)),
                stage0=dict(
                    status="registered",
                    reason_codes=[] if rid >= 0 else ["UNASSIGNED_SEMANTIC_BUCKET"],
                ),
                stage4=tasks.get(
                    house,
                    dict(status="not_run", reason_codes=["HOUSE_GATE_UNAVAILABLE"]),
                ),
                scene_directory=str(scene),
                semantic_source=str(sem),
                annotation_source=str(ann),
                navmesh_source=str(nav),
                stage1=dict(status="not_run", reason_codes=[]),
                stage2=dict(status="not_run", reason_codes=[]),
                metrics=None,
            )
            rows.append(row)
        jobs.append(
            dict(
                house=house,
                family=args.family,
                scene_directory=str(scene),
                navmesh=str(nav),
                rows=rows,
            )
        )
    jobs.sort(key=lambda j: j["house"])
    if args.house:
        jobs = [j for j in jobs if args.house in j["house"]]
    if args.limit:
        jobs = jobs[: args.limit]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if not jobs:
        raise ValueError("no houses in declared scope")
    houses = [j["house"] for j in jobs]
    # Stratify by source train/val and shuffle whole houses before reading labels for analysis.
    rng = random.Random(p["random_seed"])
    calibration = []
    holdout = []
    strata = {}
    for h in houses:
        strata.setdefault(h.split("_")[1], []).append(h)
    for names in strata.values():
        names = sorted(names)
        rng.shuffle(names)
        n = len(names) // 2
        calibration.extend(names[:n])
        holdout.extend(names[n:])
    split = dict(
        seed=p["random_seed"],
        unit="whole house",
        calibration=sorted(calibration),
        holdout=sorted(holdout),
        label_use="only calibration labels may inform tuning; holdout evaluated after fixed thresholds",
        threshold_selection="task R initial thresholds; no optimization on holdout",
    )
    write_json(out / "house_analysis_split.json", split)
    write_json(
        out / "inputs.json",
        dict(
            created_at_sgt=now(),
            inventory_source=str(args.inventory),
            family=args.family,
            house_count=len(jobs),
            houses=houses,
            parameters=p,
            threshold_spec=spec,
            latest_gate_counts=dict(
                Counter(j["rows"][0]["stage4"]["status"] for j in jobs)
            ),
            historical_baseline_task_R=dict(
                agreement_approx=0.72,
                precision_approx=0.74,
                recall_approx=0.87,
                status="task-provided historical approximate values; exact original rule/unsure denominator unverified",
            ),
            missing_legacy_houses=[
                j["house"]
                for j in jobs
                if not any(r["legacy_room"] is not None for r in j["rows"])
            ],
        ),
    )
    (out / "thresholds.initial.yaml").write_text(Path(args.thresholds).read_text())
    (out / "thresholds.used.yaml").write_text(Path(args.thresholds).read_text())
    for job in jobs:
        write_json(out / "jobs" / f"{job['house']}.json", job)
    with (out / "rooms_registry.stage0.jsonl").open("w") as stream:
        for j in jobs:
            for r in j["rows"]:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        "REGISTER",
        json.dumps(
            dict(
                houses=len(jobs),
                regions=sum(len(j["rows"]) for j in jobs),
                human_reviews=sum(
                    r["human"] is not None for j in jobs for r in j["rows"]
                ),
                gate_counts=dict(
                    Counter(j["rows"][0]["stage4"]["status"] for j in jobs)
                ),
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    return jobs


def qualify_region_evidence(row, skip_splitting=False):
    """Keep incomplete source evidence visible at every derived unit.

    This only qualifies status labels. It never changes measured geometry,
    numeric thresholds, human verdicts, or the original region's decision.
    """
    if not row.get("floors"):
        row["stage2"] = dict(
            status="not_run",
            reason_codes=["NO_SEMANTIC_FLOOR_GEOMETRY"],
            proposal_floor_ids=[],
        )
    if (row.get("native_region") or {}).get("ambiguous_colours"):
        for floor in row.get("floors", []):
            for unit in [floor] + floor.get("split_parts", []):
                previous = unit.get("stage1", {})
                unit["stage1"] = dict(
                    status="review",
                    reason_codes=sorted(
                        set(previous.get("reason_codes", []))
                        | {"SEMANTIC_COLOUR_CONFLICT"}
                    ),
                )
    if skip_splitting:
        row["stage2"] = dict(
            status="not_run",
            reason_codes=["STAGE2_NOT_REQUESTED"],
            proposal_floor_ids=[],
        )
        for floor in row.get("floors", []):
            floor["stage2"] = dict(
                status="not_run", reason_codes=["STAGE2_NOT_REQUESTED"]
            )
    return row


def process_house(args, job):
    _, p = load_parameters(args.thresholds)
    started = time.time()
    scene = Path(job["scene_directory"])
    mesh = load_hm3d(scene) if job["family"] == "hm3d" else load_mp3d(scene)
    collision, collision_source = collision_mesh(scene)
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    rt = prepare_installed_habitat_runtime(**habitat_runtime_options())
    hs = rt.habitat_sim
    pf = hs.PathFinder()
    if not pf.load_nav_mesh(job["navmesh"]):
        raise ValueError("NAVMESH_LOAD_FAILED: " + job["navmesh"])
    provenance = Path(job["navmesh"] + ".provenance.json")
    nav_identity = dict(
        path=job["navmesh"],
        size_bytes=Path(job["navmesh"]).stat().st_size,
        provenance=json.loads(provenance.read_text()) if provenance.exists() else None,
        settings_status=(
            "recorded" if provenance.exists() else "unknown_existing_navmesh_parameters"
        ),
        native_area_m2=float(pf.navigable_area),
        island_count=int(pf.num_islands),
    )
    for row in job["rows"]:
        rid = row["region_id"]
        label = row["room_label"]
        print("REGION", job["house"], label, flush=True)
        layers, furniture, members, bbox = region_geometry(mesh, rid, p)
        native_label = (row.get("native_region") or {}).get("region_label")
        room_type, type_evidence = infer_type(members, native_label)
        row.update(
            room_type=room_type,
            room_type_evidence=type_evidence,
            semantic_bbox_xz_m=bbox,
            navmesh_identity=nav_identity,
            mesh_diagnostics=mesh.diagnostics,
            collision_source=collision_source,
            furniture_instances=[
                {k: v for k, v in f.items() if k != "geometry"} for f in furniture
            ],
        )
        overhead = None
        media_error = None
        if job["family"] == "hm3d" and row["legacy_room"] is not None:
            try:
                overhead = load_overhead(
                    job["house"],
                    label,
                    args.output / "media_reference",
                    args.overhead_base,
                )
            except Exception as e:
                media_error = type(e).__name__ + ": " + str(e)
        row["overhead"] = overhead[3] if overhead else None
        row["overhead_error"] = media_error
        row["video_path"] = str(args.media_root / job["house"] / (label + ".mp4"))
        row["video_exists"] = Path(row["video_path"]).is_file()
        row["camera_metadata_path"] = str(
            Path(row["video_path"]).with_suffix(".camera.json")
        )
        row["floors"] = []
        for fi, layer in enumerate(layers):
            scope = layer["geometry"]
            floor_y = layer["floor_y_m"]
            same = [
                f
                for f in furniture
                if f["height_range_m"][0] <= floor_y + p["camera_height_m"]
                and f["height_range_m"][1] >= floor_y - p["floor_height_separation_m"]
            ]
            m = scope_metrics(scope, same)
            nav, points, clearance, adj, comps = sample_navigation(
                pf, hs, scope, floor_y, p
            )
            m.update(nav)
            m["placement"] = placement(collision, points, clearance, adj, p)
            if overhead:
                m.update(black_metric(scope, floor_y, *overhead[:3], p))
            else:
                m.update(
                    black_fraction=None, scan_quality_status="OVERHEAD_UNAVAILABLE"
                )
            m["floor_y_m"] = floor_y
            m["height_range_m"] = layer["height_range_m"]
            m["floor_face_count"] = layer["face_count"]
            s1 = decide(m, room_type, p)
            clusters = furniture_clusters(same, floor_y, p)
            trigger, convexity = split_triggers(scope, clusters, p)
            floor = dict(
                floor_id=f"F{fi}",
                metrics=m,
                stage1=s1,
                furniture_clusters=clusters,
                stage2=dict(status="not_triggered", reason_codes=[]),
                split_parts=[],
            )
            if trigger and not args.skip_splitting:
                proposal, parts = propose(scope, points, clearance, adj, clusters, p)
                groups = proposal.pop("part_sample_indices", [])
                proposal["trigger_codes"] = trigger
                floor["stage2"] = proposal
                if parts:
                    rel = f"split_proposals/{job['house']}__{label}__F{fi}.png"
                    drawn = overlay(args.output / rel, scope, floor_y, parts, overhead)
                    floor["stage2"]["overlay_path"] = rel if drawn else None
                    floor["stage2"]["overlay_status"] = (
                        "saved" if drawn else "missing_original_overhead"
                    )
                for pi, (part, indices) in enumerate(zip(parts, groups)):
                    pm = scope_metrics(part, same)
                    # Each child independently re-enters the native stage 1 sampler.
                    pn, pp, pc, pa, _ = sample_navigation(pf, hs, part, floor_y, p)
                    pm.update(pn)
                    pm["placement"] = placement(collision, pp, pc, pa, p)
                    if overhead:
                        pm.update(black_metric(part, floor_y, *overhead[:3], p))
                    else:
                        pm["black_fraction"] = None
                    floor["split_parts"].append(
                        dict(
                            part_id=f"F{fi}.S{pi}",
                            metrics=pm,
                            stage1=decide(pm, room_type, p),
                            human_review_status="not_reviewed",
                        )
                    )
            row["floors"].append(floor)
        if not layers:
            row["stage1"] = dict(
                status="review", reason_codes=["NO_SEMANTIC_FLOOR_GEOMETRY"]
            )
            row["metrics"] = None
        elif len(layers) == 1:
            row["stage1"] = row["floors"][0]["stage1"]
            row["metrics"] = row["floors"][0]["metrics"]
        else:
            row["stage1"] = dict(
                status="review", reason_codes=["MULTILEVEL_REGION_REQUIRES_REVIEW"]
            )
            row["metrics"] = dict(
                floor_count=len(layers),
                floor_area_m2=sum(f["metrics"]["floor_area_m2"] for f in row["floors"]),
            )
        if (row.get("native_region") or {}).get("ambiguous_colours"):
            row["stage1"] = dict(
                status="review", reason_codes=["SEMANTIC_COLOUR_CONFLICT"]
            )
        if rid < 0:
            row["stage1"] = dict(
                status="review", reason_codes=["UNASSIGNED_SEMANTIC_BUCKET"]
            )
        if not row["semantic_region_present"]:
            row["stage1"] = dict(
                status="review", reason_codes=["SEMANTIC_REGION_MAPPING_MISSING"]
            )
        triggers = [
            f["stage2"]
            for f in row["floors"]
            if f["stage2"]["status"] != "not_triggered"
        ]
        row["stage2"] = dict(
            status=(
                "proposed"
                if any(f["status"] == "proposed" for f in triggers)
                else ("review" if triggers else "not_triggered")
            ),
            reason_codes=sorted(
                {
                    r
                    for f in triggers
                    for r in f.get("trigger_codes", f.get("reason_codes", []))
                }
            ),
            proposal_floor_ids=[
                f["floor_id"]
                for f in row["floors"]
                if f["stage2"]["status"] == "proposed"
            ],
        )
        qualify_region_evidence(row, skip_splitting=args.skip_splitting)
    result = dict(
        house=job["house"],
        family=job["family"],
        finished_at_sgt=now(),
        seconds=time.time() - started,
        rows=job["rows"],
        status="measured",
    )
    write_json(args.output / "houses" / f"{job['house']}.json", result)
    print("HOUSE_DONE", job["house"], round(result["seconds"], 1), flush=True)
    return result


def measure(args, jobs):
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def one(job):
        house = job["house"]
        dest = args.output / "houses" / f"{house}.json"
        if args.resume and dest.exists():
            return house, "cached"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "tools.rooms.room_selection.run",
            "worker",
            "--job",
            str(args.output / "jobs" / f"{house}.json"),
            "--output",
            str(args.output),
            "--thresholds",
            str(args.thresholds),
            "--overhead-base",
            args.overhead_base,
            "--media-root",
            str(args.media_root),
        ]
        if args.skip_splitting:
            cmd.append("--skip-splitting")
        try:
            with (logs / f"{house}.log").open("w") as stream:
                cp = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.house_timeout,
                )
            if cp.returncode:
                raise RuntimeError(
                    f"worker returncode={cp.returncode}; see logs/{house}.log"
                )
            return house, "measured"
        except Exception as e:
            for r in job["rows"]:
                r["stage1"] = dict(
                    status="review", reason_codes=["HOUSE_MEASUREMENT_FAILED"]
                )
                r["stage2"] = dict(
                    status="not_run", reason_codes=["HOUSE_MEASUREMENT_FAILED"]
                )
                r["measurement_error"] = str(e)
            write_json(
                dest, dict(house=house, rows=job["rows"], status="failed", error=str(e))
            )
            return house, "failed"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            house, status = future.result()
            print(f"PROGRESS {i}/{len(jobs)} {house} {status} {now()}", flush=True)
    assemble(args)


def assemble(args):
    inputs = json.loads((args.output / "inputs.json").read_text())
    rows = []
    houses = []
    for house in inputs["houses"]:
        path = args.output / "houses" / f"{house}.json"
        d = (
            json.loads(path.read_text())
            if path.exists()
            else json.loads((args.output / "jobs" / f"{house}.json").read_text())
        )
        rows.extend(d["rows"])
        houses.append(
            dict(
                house=house,
                status=d.get("status", "not_run"),
                seconds=d.get("seconds"),
                source=str(path),
            )
        )
    with (args.output / "rooms_registry.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(args.output / "house_execution.json", houses)
    from .analysis import summarize

    summarize(args, rows, inputs)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["register", "measure", "summarize", "all", "worker"]
    )
    p.add_argument("--inventory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--thresholds", type=Path, default=Path(__file__).with_name("thresholds.yaml")
    )
    p.add_argument("--family", choices=["hm3d", "mp3d"], default="hm3d")
    p.add_argument("--tasks-root", type=Path, default=TASKS_ROOT)
    p.add_argument("--verdict-root", type=Path, default=VERDICT_ROOT)
    p.add_argument("--media-root", type=Path, default=MEDIA_ROOT)
    p.add_argument("--overhead-base", default="http://127.0.0.1:8766")
    p.add_argument("--house")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--house-timeout", type=int, default=1800)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--skip-splitting",
        action="store_true",
        help="Measure stage 1 only; stage 2 is explicitly not_run (MP3D pilot)",
    )
    p.add_argument("--job", type=Path)
    p.add_argument("--manual-splits", type=Path)
    p.add_argument("--clean-houses", type=Path)
    return p


def main():
    args = parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.stage == "worker":
        process_house(args, json.loads(args.job.read_text()))
        return
    if args.stage in ["register", "all"]:
        if args.inventory is None:
            raise SystemExit("--inventory is required for registration")
        jobs = register(args)
    else:
        inputs = json.loads((args.output / "inputs.json").read_text())
        jobs = [
            json.loads((args.output / "jobs" / f"{h}.json").read_text())
            for h in inputs["houses"]
        ]
    if args.stage in ["measure", "all"]:
        measure(args, jobs)
    elif args.stage == "summarize":
        assemble(args)


if __name__ == "__main__":
    main()
