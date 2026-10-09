"""MP3D scope, existing-room enclave cleanup, and shared cap-35 construction."""
from __future__ import annotations
import argparse
import copy
import datetime
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np
import shapely
from shapely.geometry import shape, mapping
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_selection.measurements import navmesh_triangles
from tools.rooms.room_split_auto import mp3d_adapter as a
from tools.rooms.room_split_auto.seam_connectivity import Connectivity, exterior
from tools.rooms.room_split_auto.pipeline import nav_scope_at, finalize_interfaces, adjacency


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bounded():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (10 * 1024**3, 10 * 1024**3))


def prepare(args):
    root = Path(args.root)
    rooms = [json.loads(p.read_text()) for p in sorted((Path(args.prep) / "rooms").glob("*.json"))]
    if len(rooms) != 235 or len({r["house"] for r in rooms}) != 47:
        raise RuntimeError("input scope differs from authorised 235 rooms / 47 houses")
    auto = Path(args.reference_artifacts)
    p = json.loads((auto / "processing_plan_v1.json").read_text())["parameters"]
    parameter = json.loads((auto / "selected_parameter_v1.json").read_text())["selected_parameter"]
    large = [r for r in rooms if r["floor_area_m2"] > 35 + 1e-8]
    if len(large) != 22:
        raise RuntimeError("large-room scope differs from 22")
    jobs = {}
    for r in rooms:
        jobs.setdefault(r["house"], dict(house=r["house"], scene_directory=r["scene_directory"],
                                        navmesh=r["navmesh_source"], rooms=[]))["rooms"].append(r)
    dump(root / "plan.json", dict(created_at_utc=now(), rooms=235, houses=47, large_rooms=22,
         jobs=list(jobs.values()), parameters=p, selected_parameter=parameter,
         prep=str(args.prep), reference_artifacts=str(auto), adapter_root=str(args.adapter_root),
         base_commit="47aca4f17b64565b4fda071410101692e0b72cb2",
         cpu_only=True, workers=args.workers, per_process_address_space_gib=10,
         max_compute_processes=args.workers + 1, total_address_space_gib=10 * (args.workers + 1),
         original_subcap_policy="raw direct-connected groups only; no cuts, no new disk/shape/visibility/black gate",
         production_placement="raw filled exteriors only; 0.25 m margin; original native navmesh; three CPU raw-mesh rays"))
    print("MP3D_PREPARED", len(rooms), len(jobs), "LARGE", len(large), flush=True)


def historical_witness(room):
    path = Path(room["source_list_record"]["result_path"])
    d = json.loads(path.read_text())
    old = next(r for r in d["input"]["acoustic_rooms"] if r["room_label"] == room["room_label"])
    return dict(old["placement"], validation_provenance=str(path))


def existing_block(room, g, index, ctx, decision, witness, reasons, pending):
    main = index == 0
    return dict(schema="mp3d_native_room_connectivity_cap35_v1",
                id=room["house"] + "__" + room["room_label"] + "__" + room["selected_floor_id"] + f"__E{index:03d}",
                house=room["house"], source_region=room["room_label"], source_region_id=room["region_id"],
                floor_id=room["selected_floor_id"], floor_y_m=room["floor_y_m"],
                floor_polygon_xz_m=mapping(g), floor_area_m2=float(g.area), short_side_m=short_side(g),
                placement_witness=witness, decision=decision, discard_reasons=reasons,
                unresolved_reasons=pending, room_type=room["room_type"], new_room=not main, native_main=main,
                source="original" if main else "native_detached_candidate",
                source_selection=room["source_list_name"],
                origin_list="strict" if "strict" in room["source_list_name"] else "review band",
                original_room_id=room["room_id"], connectivity_certificate=ctx.certificate(g),
                black_fraction=float(room["source_list_record"]["black_fraction"]),
                black_measurement=dict(method="original admission inherited; not a new rejection",
                                       source=room["source_list_record"]["visual_path"]),
                area_method=room["floor_area_method"], measurement_source=room["semantic_source"],
                inherited_original_circle_gate=False, inherited_original_visibility_gate=False,
                shape_quality_admission_gate=False, acoustics="not_run", leakage=None,
                original_source_list_record=room["source_list_record"])


def unresolved_existing(room, reason, detail=None):
    g = shape(room["floor_polygon_xz_m"])
    b = dict(id=room["house"] + "__" + room["room_label"] + "__" + room["selected_floor_id"] + "__E000",
             house=room["house"], source_region=room["room_label"], source_region_id=room["region_id"],
             floor_id=room["selected_floor_id"], floor_y_m=room["floor_y_m"],
             floor_polygon_xz_m=mapping(g), floor_area_m2=float(g.area), short_side_m=short_side(g),
             decision="unresolved", discard_reasons=[], unresolved_reasons=[reason],
             placement_witness=dict(found=False, not_run_reason=reason), native_main=True, new_room=False,
             source="original", room_type=room["room_type"], leakage=None)
    return dict(house=room["house"], source_region=room["room_label"], source_floor_area_m2=float(g.area),
                requires_split=False, status="unresolved", blocks=[b], cut_lines=[], error_detail=detail,
                area_partition_error_m2=0, metrics=dict(room_id=room["room_id"], status="unresolved", affected=False))


def existing_worker(job, out, p, adapter_root=None):
    bounded()
    started = time.time()
    out = Path(out)
    mesh = None
    receipt = None
    regions = []
    try:
        from tools.rooms.room_split_auto.shape_quality_geometry import PartConnectivity
        objects = a.selection_adapter(adapter_root).load_mp3d_scene(job["scene_directory"])
        hs = a.load_native()
        pf, nav_polys, nav_ys = navmesh_triangles(hs, Path(job["navmesh"]))
        for room in job["rooms"]:
            try:
                g = shape(room["floor_polygon_xz_m"])
                nav = nav_scope_at(nav_polys, nav_ys, room["floor_y_m"], exterior(g).buffer(.3), p)
                foreign = a.foreign_floor_geometry(objects, room["region_id"], room["floor_y_m"], exterior(g).buffer(.3), p)
                ctx = PartConnectivity(nav, foreign)
                groups = sorted(ctx.groups(g), key=lambda q: (-q.area, q.bounds))
                metrics = dict(room_id=room["room_id"], house=room["house"], room_label=room["room_label"],
                               floor_area_m2=float(g.area), listed_area_m2=room["listed_floor_area_m2"],
                               raw_parts=len(list(shapely.get_parts(g))), direct_groups=len(groups),
                               opened_components=ctx.count(g), affected=len(groups) > 1,
                               new_direct_component_areas_m2=[float(q.area) for q in groups],
                               original_room_uncut=True, original_circle_gate=False)
                reg = a.source_region(room)
                reg.update(requires_split=False, cut_lines=[], metrics=metrics)
                if g.area > 35 + 1e-8:
                    reg.update(status="delegated_to_cap35_split", delegated_to="mp3d_delivery_v1/final_v1",
                               blocks=[existing_block(room, g, 0, ctx, "delegated_split",
                                       dict(found=False, not_run_reason="large room delegated"), [], [])])
                    metrics["affected"] = False
                else:
                    outputs = []
                    for i, part in enumerate(groups):
                        reasons, pending = [], []
                        # Original gates are inherited. Detached candidates must meet owner size rules.
                        if i:
                            if part.area < 6 - 1e-8:
                                reasons.append("DETACHED_FRAGMENT")
                            elif short_side(part) < 2.4 - 1e-8:
                                reasons.append("DETACHED_SHORT_SIDE_BELOW_2_4")
                        if reasons:
                            witness = dict(found=False, not_run_reason="discarded detached fragment")
                        else:
                            if mesh is None:
                                mesh, receipt = a.raw_collision(job["scene_directory"])
                            witness = a.placement_in_outline(mesh, pf, hs, part, room["floor_y_m"], p,
                                                             historical_witness(room) if i == 0 else None)
                            if not witness.get("found"):
                                pending.append("PLACEMENT_NO_VALID_MARGIN_WITNESS_WITHIN_FROZEN_BUDGET")
                            if i == 0 and part.area < 6 - 1e-8:
                                pending.append("HISTORICAL_ADMISSION_BELOW_6" if g.area < 6 - 1e-8 else "ORIGINAL_MAIN_BELOW_6_AFTER_ENCLAVE_REMOVAL")
                            if i == 0 and short_side(part) < 2.4 - 1e-8:
                                pending.append("ORIGINAL_MAIN_SHORT_SIDE_AFTER_ENCLAVE_REMOVAL")
                        decision = "discard" if reasons else "unresolved" if pending else "retain"
                        outputs.append(existing_block(room, part, i, ctx, decision, witness, reasons, pending))
                    reg.update(blocks=outputs, status="partially_unresolved" if any(b["decision"] == "unresolved" for b in outputs)
                               else "connectivity_cleaned" if len(groups) > 1 else "unchanged")
                reg["metrics"]["status"] = reg["status"]
                reg["area_partition_error_m2"] = abs(sum(b["floor_area_m2"] for b in reg["blocks"]) - g.area)
                if reg["area_partition_error_m2"] > 1e-6:
                    raise RuntimeError("existing ground partition area mismatch")
            except Exception as exc:
                reg = unresolved_existing(room, "EXISTING_ROOM_JOB_FAILURE", traceback.format_exc())
            dump(out / "regions" / (room["house"] + "__" + room["room_label"] + ".json"), reg)
            for b in reg["blocks"]:
                dump(out / "rooms" / (b["id"] + ".json"), b)
            regions.append(reg["metrics"])
            print("MP3D_EXISTING", room["room_id"], reg["status"],
                  "RETAIN", sum(b["decision"] == "retain" for b in reg["blocks"]), flush=True)
    except Exception:
        error = traceback.format_exc()
        for room in job["rooms"]:
            path = out / "regions" / (room["house"] + "__" + room["room_label"] + ".json")
            if path.exists():
                continue
            reg = unresolved_existing(room, "EXISTING_HOUSE_JOB_FAILURE", error)
            dump(path, reg)
            dump(out / "rooms" / (reg["blocks"][0]["id"] + ".json"), reg["blocks"][0])
            regions.append(reg["metrics"])
    return dict(house=job["house"], rows=regions, seconds=time.time() - started,
                pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0),
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, ray_receipt=receipt)


def run_existing(args):
    bounded()
    root = Path(args.root)
    out = root / "mp3d_existing_rooms_connectivity_v1"
    out.mkdir(exist_ok=False)
    (out / "regions").mkdir()
    (out / "rooms").mkdir()
    plan = json.loads((root / "plan.json").read_text())
    receipts = []
    # fork has no extra resource-tracker process; all native runtimes are loaded in children.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        fs = {pool.submit(existing_worker, j, out, plan["parameters"], plan["adapter_root"]): j for j in plan["jobs"]}
        for f in as_completed(fs):
            job = fs[f]
            try:
                receipts.append(f.result())
            except Exception:
                for room in job["rooms"]:
                    reg = unresolved_existing(room, "EXISTING_WORKER_FAILURE", traceback.format_exc())
                    path = out / "regions" / (room["house"] + "__" + room["room_label"] + ".json")
                    if not path.exists():
                        dump(path, reg)
                        dump(out / "rooms" / (reg["blocks"][0]["id"] + ".json"), reg["blocks"][0])
                receipts.append(dict(house=job["house"], error=traceback.format_exc(), rows=[]))
    dump(out / "runtime_receipt.json", receipts)
    dump(out / "metrics.json", [r for x in receipts for r in x["rows"]])
    regs = [json.loads(p.read_text()) for p in (out / "regions").glob("*.json")]
    blocks = [b for r in regs for b in r["blocks"]]
    counts = Counter(b["decision"] for b in blocks)
    dump(out / "stage1_summary.json", dict(original_rooms=len(regs), decision_counts=dict(counts),
         retained_native_main=sum(b["decision"] == "retain" and b.get("native_main", False) for b in blocks),
         new_detached_candidates_retained=sum(b["decision"] == "retain" and not b.get("native_main", False) for b in blocks),
         affected_subcap_originals=sum(r.get("metrics", {}).get("affected", False) for r in regs),
         area_partition_max_error_m2=max(r["area_partition_error_m2"] for r in regs),
         retained_area_m2=sum(b["floor_area_m2"] for b in blocks if b["decision"] == "retain"),
         discarded_area_m2=sum(b["floor_area_m2"] for b in blocks if b["decision"] == "discard"),
         pending_area_m2=sum(b["floor_area_m2"] for b in blocks if b["decision"] == "unresolved"),
         delegated_large_rooms=22, pid=os.getpid(), finished_at_utc=now()))
    print("MP3D_EXISTING_DONE", len(regs), dict(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "existing", "split"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--prep")
    parser.add_argument("--reference-artifacts")
    parser.add_argument("--adapter-root")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    if args.action == "prepare":
        prepare(args)
    elif args.action == "existing":
        run_existing(args)
    else:
        from tools.rooms.room_split_auto.mp3d_split import run
        run(args)


if __name__ == "__main__":
    main()
