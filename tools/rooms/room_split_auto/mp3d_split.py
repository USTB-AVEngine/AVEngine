"""Build the 22 MP3D cap-35 deliveries with the committed HM3D v5/v6 algorithms."""
from __future__ import annotations
import argparse
import copy
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np
import shapely
from shapely.geometry import shape, mapping, LineString
from PIL import Image
from tools.rooms.room_split_auto import mp3d_adapter as a
from tools.rooms.room_split_auto.mp3d_run import dump, bounded, now
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_selection.measurements import navmesh_triangles
from tools.rooms.room_split_auto.pipeline import nav_scope_at


def frame_for(row, floor, cache, root, old):
    root = Path(root)
    path = root / "region_renders" / (row["house"] + "__" + row["room_label"] + "__" + floor["floor_id"] + ".json")
    meta = json.loads(path.read_text())
    sensor = meta["images"][0]
    raw_path = meta.get("black_source_path", sensor["path"])
    ref = dict(metadata_path=str(path), image_path=raw_path, display_image_path=sensor["path"],
               source="authorised MP3D whole-house raw textured CPU render")
    return meta, sensor, Image.open(raw_path).convert("RGB"), ref


def unresolved_region(room, reason, details=None):
    reg = a.source_region(room)
    reg.update(status="unresolved", revision="mp3d_cap35_v6", unresolved_reasons=[reason], error_detail=details)
    reg["blocks"] = [dict(schema="mp3d_auto_room_split_v6", id=room["house"] + "__" + room["room_label"] +
                         "__" + room["selected_floor_id"] + "__M000",
                         house=room["house"], source_region=room["room_label"], source_region_id=room["region_id"],
                         floor_id=room["selected_floor_id"], floor_y_m=room["floor_y_m"],
                         floor_polygon_xz_m=room["floor_polygon_xz_m"], floor_area_m2=room["floor_area_m2"],
                         short_side_m=room["short_side_m"], decision="unresolved", discard_reasons=[],
                         unresolved_reasons=[reason], placement_witness=dict(found=False, not_run_reason=reason),
                         new_room=True, room_type=room["room_type"], leakage=None, acoustics="not_run")]
    reg.update(retained_new_rooms=0, area_partition_error_m2=0)
    return reg


def timeout(signum, frame):
    raise TimeoutError("MP3D region finite CPU time budget exhausted")


def audit_final(reg, room, mesh, pf, hs, nav_polys, nav_ys, objects, p, root):
    from tools.rooms.room_split_auto import shape_quality_geometry as sg
    from tools.rooms.room_split_auto import shape_quality_repair as q
    from tools.rooms.room_split_auto import connected_split as cs
    floor = reg["source_geometry"]["floors"][0]
    fy = floor["floor_y_m"]
    fid = floor["floor_id"]
    source = shape(floor["floor_polygon"])
    nav = nav_scope_at(nav_polys, nav_ys, fy, sg.own_outline(source).buffer(.3), p)
    furniture = cs.furniture_for(objects, room["region_id"], fy, p, source)
    foreign_floor = a.foreign_floor_geometry(objects, room["region_id"], fy, source.buffer(.3), p)
    nav = nav.difference(foreign_floor)
    # Any retained room must have an independently rechecked, real-margin witness.
    for b in reg["blocks"]:
        eligible = b["decision"] == "retain" or (
            b["decision"] == "discard" and b.get("discard_reasons") == ["PLACEMENT_NO_WITNESS_WITHIN_FROZEN_BUDGET"])
        if not eligible:
            continue
        g = shape(b["floor_polygon_xz_m"])
        reasons = []
        defects = sg.defects(g)
        if not 6 - 1e-7 <= g.area <= 35 + 1e-7 or short_side(g) < 2.4 - 1e-7:
            reasons.append("FINAL_SIZE_CONSTRUCTION_UNRESOLVED")
        if not sg.disk(g)["fits"]:
            reasons.append("FINAL_OWN_OUTLINE_2_4_M_DISK_UNRESOLVED")
        if defects["neck_count"]:
            reasons.append("FINAL_NECK_SHAPE_REPAIR_UNRESOLVED")
        if defects["corridor_count"]:
            reasons.append("FINAL_CORRIDOR_SHAPE_REPAIR_UNRESOLVED")
        if reasons:
            b.update(decision="unresolved", discard_reasons=[], unresolved_reasons=reasons)
            continue
        witness = a.placement_in_outline(mesh, pf, hs, g, fy, p, b.get("placement_witness"))
        b["placement_witness"] = witness
        if not witness.get("found"):
            b.update(decision="unresolved", discard_reasons=[],
                     unresolved_reasons=["FINAL_PLACEMENT_NO_VALID_MARGIN_WITNESS_WITHIN_FROZEN_BUDGET"])
        else:
            b.update(decision="retain", discard_reasons=[], unresolved_reasons=[])
    # Navigation certificates are only pair passage evidence; other source rooms are forbidden.
    for b in reg["blocks"]:
        if b["decision"] != "retain":
            continue
        g = shape(b["floor_polygon_xz_m"])
        other = shapely.union_all([shape(x["floor_polygon_xz_m"]) for x in reg["blocks"]
                                  if x is not b and x["decision"] == "retain" and x["floor_id"] == fid])
        ctx = sg.PartConnectivity(nav, shapely.union_all([other, foreign_floor]))
        b["connectivity_certificate"] = ctx.certificate(g)
        b["connectivity_rule"] = "distinct_part_seam_nav_v6"
        if len(ctx.groups(g)) != 1:
            b.update(decision="unresolved", unresolved_reasons=["FINAL_DIRECT_PART_CONNECTIVITY_UNRESOLVED"])
    # A failed v6 repair does not make a deformed room a final retained candidate.
    live = [b for b in reg["blocks"] if b["decision"] == "retain"]
    for b in live:
        g = sg.own_outline(shape(b["floor_polygon_xz_m"]))
        wraps = [x["id"] for x in live if x is not b and
                 g.convex_hull.intersection(sg.own_outline(shape(x["floor_polygon_xz_m"]))).area /
                 sg.own_outline(shape(x["floor_polygon_xz_m"])).area >= .3]
        if wraps:
            b.update(decision="unresolved", unresolved_reasons=["FINAL_WRAP_REPAIR_UNRESOLVED"],
                     shape_quality_remaining_wrap_targets=wraps)
    q.evaluate_interfaces(reg, {fid: furniture})
    bad_lines = [c for c in reg["cut_lines"] if c.get("active_in_final_partition", True) and (
                 (c.get("furniture_intersection_length_m") or 0) > .5 + 1e-8 or
                 c.get("segment_count", 0) > 3 or max(c.get("segment_angle_errors_deg", []), default=0) > 15 + 1e-6)]
    if bad_lines:
        raise RuntimeError("FINAL_CUT_DESIGN_UNRESOLVED: " + json.dumps([
            dict(id=x["id"], furniture_m=x.get("furniture_intersection_length_m"),
                 segments=x.get("segment_count")) for x in bad_lines]))
    for i, b in enumerate(sorted(reg["blocks"], key=lambda x: (-x["floor_area_m2"], x["id"]))):
        b["construction_block_id"] = b["id"]
        b["id"] = room["house"] + "__" + room["room_label"] + "__" + fid + f"__M{i:03d}"
        b.update(schema="mp3d_auto_room_split_v6", original_room_id=room["room_id"],
                 whole_house_visual_source=room["whole_house_visual_source"],
                 whole_house_acoustic_manifest=room["whole_house_acoustic_manifest"],
                 navmesh_source=room["navmesh_source"], source="cap_cut_new",
                 origin_list="strict" if "strict" in room["source_list_name"] else "review band",
                 source_selection=room["source_list_name"], leakage=None, acoustics="not_run_per_task",
                 production_placement="real filled exterior; 0.25 m margin; native navmesh feet; three CPU raw-scan rays")
    # Rename adjacency targets without reactivating point-only, superseded cuts.
    q.evaluate_interfaces(reg, {fid: furniture})
    measured = shapely.union_all([shape(b["floor_polygon_xz_m"]) for b in reg["blocks"]])
    difference = float(measured.symmetric_difference(source).area)
    overlap = sum(b["floor_area_m2"] for b in reg["blocks"]) - measured.area
    if difference > 1e-6 or abs(overlap) > 1e-6:
        raise RuntimeError("FINAL_RAW_PARTITION_CONSERVATION_FAILURE")
    reg.update(retained_new_rooms=sum(b["decision"] == "retain" for b in reg["blocks"]),
               status="partially_unresolved" if any(b["decision"] == "unresolved" for b in reg["blocks"]) else "processed",
               raw_partition_symmetric_difference_m2=difference, raw_partition_overlap_m2=float(overlap),
               area_partition_error_m2=abs(sum(b["floor_area_m2"] for b in reg["blocks"]) - source.area),
               revision="mp3d_shared_v5_v6", source_original_room_id=room["room_id"])
    return reg


def worker(job, root, out, plan):
    bounded()
    root, out = Path(root), Path(out)
    started = time.time()
    statuses = []
    source_receipt = {}
    try:
        from tools.rooms.room_split_auto import walkable_split as w, capped_split as cap, connected_split as cs
        from tools.rooms.room_split_auto import shape_quality_repair as q, shape_quality_geometry as sg
        hs = a.load_native()
        pf, nav_polys, nav_ys = navmesh_triangles(hs, Path(job["navmesh"]))
        mesh, ray_receipt = a.raw_collision(job["scene_directory"])
        adapter = a.selection_adapter(plan["adapter_root"])
        objects = adapter.load_mp3d_scene(job["scene_directory"])
        markers, door_info = a.structural_instances(job["scene_directory"], adapter)
        source_receipt = dict(ray_receipt=ray_receipt, adapter_source=plan["adapter_root"],
                              door_info=door_info, semantic_diagnostics=objects.diagnostics)
        p = plan["parameters"]
        # These are construction controls explicitly requested by this task, not edits to frozen gates.
        parameter = dict(plan["selected_parameter"], width_min=0., width_max=.6)
        cap.rgb_for = frame_for
        w.CORRIDOR_JUNCTION_SEARCH_M = 2.4
        w.CORRIDOR_ELBOW_VERTEX_REFINEMENT = True
        for room in job["rooms"]:
            t = time.time()
            name = room["house"] + "__" + room["room_label"] + ".json"
            signal.signal(signal.SIGALRM, timeout)
            signal.alarm(600)
            reg = None
            try:
                old = a.source_region(room)
                foreign = a.foreign_floor_geometry(objects, room["region_id"], room["floor_y_m"],
                                                     shape(room["floor_polygon_xz_m"]).buffer(.3), p)
                def local_nav(np_, ny, fy, scope, params):
                    return nav_scope_at(np_, ny, fy, scope, params).difference(foreign)
                w.nav_scope_at = local_nav
                q.nav_scope_at = local_nav
                axis = a.wall_axis(markers, shape(room["floor_polygon_xz_m"]), room["floor_y_m"],
                                   cap.main_axis(shape(room["floor_polygon_xz_m"])))
                saved_axis = cap.main_axis
                cap.main_axis = lambda g: axis["primary_deg"]
                try:
                    initial = w.process(old, mesh, pf, hs, nav_polys, nav_ys, objects, markers, door_info,
                                        p, parameter, 35, {}, root)
                finally:
                    cap.main_axis = saved_axis
                initial["wall_axes"][room["selected_floor_id"]] = axis
                dump(out / "attempt_v1/regions" / (Path(name).stem + "__v5.json"), initial)
                # The v6 repair measures its own outline. Restore the v5-installed global contour hook.
                cs.filled_footprint = sg.own_outline
                cap.outline = sg.own_outline
                reg = q.process(initial, mesh, pf, hs, nav_polys, nav_ys, objects, p, root)
                dump(out / "attempt_v1/regions" / (Path(name).stem + "__v6.json"), reg)
                reg = audit_final(reg, room, mesh, pf, hs, nav_polys, nav_ys, objects, p, root)
            except Exception:
                detail = traceback.format_exc()
                reg = unresolved_region(room, "MP3D_CAP35_CONSTRUCTION_UNRESOLVED", detail)
            finally:
                signal.alarm(0)
            dump(out / "final_v1/regions" / name, reg)
            for b in reg["blocks"]:
                dump(out / "final_v1/rooms" / (b["id"] + ".json"), b)
            statuses.append(dict(room_id=room["room_id"], status=reg["status"],
                                 retained=reg["retained_new_rooms"], seconds=time.time() - t,
                                 error_detail=reg.get("error_detail")))
            print("MP3D_SPLIT", room["room_id"], reg["status"], "KEPT", reg["retained_new_rooms"],
                  "SECONDS", round(time.time() - t, 2), flush=True)
    except Exception:
        detail = traceback.format_exc()
        for room in job["rooms"]:
            name = room["house"] + "__" + room["room_label"] + ".json"
            if (out / "final_v1/regions" / name).exists():
                continue
            reg = unresolved_region(room, "MP3D_SPLIT_HOUSE_JOB_FAILURE", detail)
            dump(out / "final_v1/regions" / name, reg)
            dump(out / "final_v1/rooms" / (reg["blocks"][0]["id"] + ".json"), reg["blocks"][0])
            statuses.append(dict(room_id=room["room_id"], status="unresolved", retained=0, error_detail=detail))
    return dict(house=job["house"], statuses=statuses, seconds=time.time() - started, pid=os.getpid(),
                nice=os.getpriority(os.PRIO_PROCESS, 0), peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                **source_receipt)


def run(args):
    bounded()
    root = Path(args.root)
    # This dependency must be committed and merged, not copied from the other task's working files.
    tracked = subprocess.check_output(["git", "ls-files", "tools/rooms/room_split_auto/shape_quality_geometry.py",
                                      "tools/rooms/room_split_auto/shape_quality_repair.py"], text=True).splitlines()
    if len(tracked) != 2:
        raise RuntimeError("committed and merged HM3D v6 is required before MP3D final cutting")
    from tools.rooms.room_split_auto import shape_quality_geometry
    plan = json.loads((root / "plan.json").read_text())
    jobs = [dict(j, rooms=[r for r in j["rooms"] if r["floor_area_m2"] > 35 + 1e-8]) for j in plan["jobs"]]
    jobs = [j for j in jobs if j["rooms"]]
    out = root / "mp3d_delivery_v1"
    out.mkdir(exist_ok=False)
    for folder in ("final_v1/regions", "final_v1/rooms", "attempt_v1/regions"):
        (out / folder).mkdir(parents=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dump(out / "revision_plan.json", dict(jobs=jobs, parameters=plan["parameters"],
         selected_parameter=plan["selected_parameter"], construction_narrow_priority_max_m=.6,
         source_commit=source_commit, committed_v6=True, workers=args.workers, cpu_only=True,
         region_cpu_budget_seconds=600, address_space_gib_total=10 * (args.workers + 1),
         existing_subcap_no_new_shape_gate=True))
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = {pool.submit(worker, j, root, out, plan): j for j in jobs}
        for f in as_completed(futures):
            job = futures[f]
            try:
                receipts.append(f.result())
            except Exception:
                detail = traceback.format_exc()
                for room in job["rooms"]:
                    name = room["house"] + "__" + room["room_label"] + ".json"
                    if (out / "final_v1/regions" / name).exists():
                        continue
                    reg = unresolved_region(room, "MP3D_SPLIT_WORKER_FAILURE", detail)
                    dump(out / "final_v1/regions" / name, reg)
                    dump(out / "final_v1/rooms" / (reg["blocks"][0]["id"] + ".json"), reg["blocks"][0])
                receipts.append(dict(house=job["house"], error=detail))
    dump(out / "final_v1/native_receipt.json", receipts)
    dump(out / "final_v1/completed.json", dict(finished_at_utc=now(), source_commit=source_commit,
         original_large_rooms=22, workers=args.workers, pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0)))
    print("MP3D_SPLIT_ALL_DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error("workers must be 1..8")
    run(args)
