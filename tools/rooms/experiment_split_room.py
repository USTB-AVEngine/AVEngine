#!/usr/bin/env python3
"""Read-only pilot experiment for splitting one HM3D room.

This tool deliberately does not edit rooms.json, a navmesh, curation verdicts,
or any Studio configuration.  It samples the declared room bbox against the
*loaded* Habitat navmesh, builds geodesic Voronoi partitions, and writes only
temporary evidence files under --output-dir.

The acoustic section is intentionally conservative: this experiment reports
``not_run`` unless the caller supplies an actual acoustic probe command.  A
distance threshold is not an acoustic measurement and must not be reported as
one.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import subprocess
import sys
import shutil
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--navmesh", required=True, type=Path)
    p.add_argument("--room-json", required=True, type=Path)
    p.add_argument("--label", default="R11")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--original-topdown", type=Path, help="copy one original connectivity topdown into the temporary output")
    p.add_argument("--grid-step-m", type=float, default=0.25)
    p.add_argument("--camera-height-m", type=float, default=1.5)
    p.add_argument("--source-height-m", type=float, default=1.2)
    p.add_argument("--hfov-deg", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--runtime-prefix", default="/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z")
    p.add_argument("--magnum-site", default="/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages")
    p.add_argument("--rlr-sdk-root", default="/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg")
    p.add_argument("--acoustic-probe", type=Path, help="optional executable probe; it must accept the JSON evidence path")
    return p


def room_record(path: Path, label: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    wanted = [r for r in data.get("rooms", []) if f"R{r.get('region_id')}" == label]
    if len(wanted) != 1:
        raise SystemExit(f"expected exactly one {label} in {path}, found {len(wanted)}")
    room = wanted[0]
    if room.get("floor_y_m") is None:
        raise SystemExit(f"{label} has no floor_y_m")
    return room


def vec3(mn: Any, value: np.ndarray) -> Any:
    return mn.Vector3(float(value[0]), float(value[1]), float(value[2]))


def load_runtime(scene: Path, navmesh: Path, a: argparse.Namespace):
    if scene.name.endswith(".basis.glb"):
        raise SystemExit("refusing .basis.glb")
    dataset_root = next((p for p in scene.resolve().parents if (p / "scene_datasets").is_dir()), None)
    if dataset_root is None:
        # HM3D installations commonly put scene_datasets above the dataset
        # root; the runtime loader can still use the explicit scene path.
        dataset_root = scene.parent
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    rt = prepare_installed_habitat_runtime(
        runtime_prefix=a.runtime_prefix,
        magnum_python_site=a.magnum_site,
        rlr_sdk_root=a.rlr_sdk_root,
        mp3d_root=str(dataset_root),
        allow_mp3d_environment=False,
    )
    hs, mn = rt.habitat_sim, rt.magnum
    backend = hs.SimulatorConfiguration()
    backend.scene_id = str(scene)
    backend.load_semantic_mesh = False
    backend.enable_physics = True
    if rt.physics_config_path:
        backend.physics_config_file = str(rt.physics_config_path)
    sim = hs.Simulator(hs.Configuration(backend, [hs.agent.AgentConfiguration()]))
    if not sim.pathfinder.is_loaded:
        if not sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise SystemExit(f"Habitat could not load navmesh: {navmesh}")
    if not sim.pathfinder.is_loaded:
        raise SystemExit(f"navmesh is not loaded: {navmesh}")
    return rt, sim, hs, mn


def snap_candidates(pathfinder: Any, bbox: tuple[float, float, float, float], floor_y: float, step: float) -> list[dict[str, Any]]:
    min_x, min_z, max_x, max_z = bbox
    xs = np.arange(min_x, max_x + step * 0.5, step)
    zs = np.arange(min_z, max_z + step * 0.5, step)
    points: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for x in xs:
        for z in zs:
            probe = np.array([x, floor_y + 0.5, z], dtype=np.float64)
            if not pathfinder.is_navigable(probe):
                continue
            q = np.asarray(pathfinder.snap_point(probe), dtype=np.float64)
            if not np.all(np.isfinite(q)) or abs(float(q[1]) - floor_y) > 0.45:
                continue
            if not (min_x - 1e-5 <= q[0] <= max_x + 1e-5 and min_z - 1e-5 <= q[2] <= max_z + 1e-5):
                continue
            key = (round(float(q[0]) / step), round(float(q[2]) / step))
            if key in seen:
                continue
            seen.add(key)
            points.append({"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "gx": int(round((float(q[0]) - min_x) / step)), "gz": int(round((float(q[2]) - min_z) / step))})
    return points


def grid_components(points: list[dict[str, Any]]) -> list[list[int]]:
    by_grid = {(p["gx"], p["gz"]): i for i, p in enumerate(points)}
    adj: list[list[int]] = [[] for _ in points]
    for i, p in enumerate(points):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dz == 0:
                    continue
                j = by_grid.get((p["gx"] + dx, p["gz"] + dz))
                if j is not None:
                    adj[i].append(j)
    seen: set[int] = set()
    comps = []
    for start in range(len(points)):
        if start in seen:
            continue
        q = [start]; seen.add(start); comp = []
        while q:
            i = q.pop(); comp.append(i)
            for j in adj[i]:
                if j not in seen:
                    seen.add(j); q.append(j)
        comps.append(comp)
    return comps, adj


def dijkstra(adj: list[list[int]], points: list[dict[str, Any]], source: int) -> list[float]:
    d = [math.inf] * len(points); d[source] = 0.0; heap = [(0.0, source)]
    while heap:
        du, u = heapq.heappop(heap)
        if du != d[u]:
            continue
        pu = points[u]
        for v in adj[u]:
            pv = points[v]
            w = math.hypot(pu["x"] - pv["x"], pu["z"] - pv["z"])
            nd = du + w
            if nd < d[v]:
                d[v] = nd; heapq.heappush(heap, (nd, v))
    return d


def farthest_seeds(points: list[dict[str, Any]], comp: list[int], k: int) -> list[int]:
    if not comp:
        return []
    centre = min(comp, key=lambda i: (points[i]["x"] - np.mean([points[j]["x"] for j in comp])) ** 2 + (points[i]["z"] - np.mean([points[j]["z"] for j in comp])) ** 2)
    seeds = [centre]
    while len(seeds) < min(k, len(comp)):
        nxt = max(comp, key=lambda i: min((points[i]["x"] - points[s]["x"]) ** 2 + (points[i]["z"] - points[s]["z"]) ** 2 for s in seeds))
        if nxt in seeds:
            break
        seeds.append(nxt)
    return seeds


def assign_geodesic(points: list[dict[str, Any]], adj: list[list[int]], comp: list[int], k: int) -> list[list[int]]:
    seeds = farthest_seeds(points, comp, k)
    distances = [dijkstra(adj, points, s) for s in seeds]
    groups = [[] for _ in seeds]
    for i in comp:
        owner = min(range(len(seeds)), key=lambda j: distances[j][i])
        groups[owner].append(i)
    return [g for g in groups if g]


def ray_clear(sim: Any, hs: Any, mn: Any, start: np.ndarray, end: np.ndarray) -> bool:
    delta = end - start; distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return False
    ray = hs.geo.Ray(vec3(mn, start), vec3(mn, delta / distance))
    hits = sim.cast_ray(ray, max_distance=distance, buffer_distance=0.0)
    return not hits.has_hits() or float(hits.hits[0].ray_distance) >= distance - 0.08


def evaluate_group(group: list[int], points: list[dict[str, Any]], sim: Any, hs: Any, mn: Any, floor_y: float, a: argparse.Namespace) -> dict[str, Any]:
    # Candidates are deterministic interior samples, not arbitrary bbox points.
    stride = max(1, len(group) // 20)
    chosen = group[::stride][:20]
    candidates = []
    visible_pairs = []
    for ci in chosen:
        p = points[ci]
        camera = np.array([p["x"], floor_y + a.camera_height_m, p["z"]], dtype=float)
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if i == j:
                    continue
                s1f, s2f = points[chosen[i]], points[chosen[j]]
                s1 = np.array([s1f["x"], floor_y + a.source_height_m, s1f["z"]], dtype=float)
                s2 = np.array([s2f["x"], floor_y + a.source_height_m, s2f["z"]], dtype=float)
                d1 = float(np.linalg.norm(s1 - camera)); d2 = float(np.linalg.norm(s2 - camera))
                if not (1.0 <= d1 <= 6.0 and 1.0 <= d2 <= 6.0 and float(np.linalg.norm(s1 - s2)) >= 0.8):
                    continue
                v1, v2 = ray_clear(sim, hs, mn, camera, s1), ray_clear(sim, hs, mn, camera, s2)
                # Camera faces the midpoint. The angular separation is the
                # invariant quantity; rotation can always point at midpoint.
                a1 = math.atan2(s1[0] - camera[0], -(s1[2] - camera[2]))
                a2 = math.atan2(s2[0] - camera[0], -(s2[2] - camera[2]))
                sep = abs(math.degrees(math.atan2(math.sin(a1 - a2), math.cos(a1 - a2))))
                rec = {"camera_m": camera.tolist(), "source_1_m": s1.tolist(), "source_2_m": s2.tolist(), "source_distance_m": float(np.linalg.norm(s1 - s2)), "angular_separation_deg": sep, "ray_source_1": v1, "ray_source_2": v2, "visible": bool(v1 and v2 and sep <= min(70.0, a.hfov_deg - 10.0))}
                candidates.append(rec)
                if rec["visible"]:
                    visible_pairs.append(rec)
    return {"camera_candidate_count": len(chosen), "tested_two_source_combinations": len(candidates), "visible_two_source_combinations": len(visible_pairs), "visibility_rate": (len(visible_pairs) / len(candidates) if candidates else None), "best_visible_combination": visible_pairs[0] if visible_pairs else None, "hearing": {"status": "not_run", "reason": "本试验没有把距离或导航连通性冒充声学结果；需要真实声源、RLR/AudioSensor 和可执行的听音探针"}, "candidate_evidence": candidates[:100]}


def draw_topdown(path: Path, bbox: tuple[float, float, float, float], points: list[dict[str, Any]], schemes: dict[str, list[list[int]]], original: dict[str, Any], step: float, scheme_name: str) -> None:
    w = h = 1000; min_x, min_z, max_x, max_z = bbox
    def xy(p): return (int((p[0] - min_x) / (max_x - min_x) * (w - 40) + 20), int((p[1] - min_z) / (max_z - min_z) * (h - 40) + 20))
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im)
    colors = [(52,152,219),(231,76,60),(46,204,113),(155,89,182),(241,196,15),(230,126,34)]
    for idx, (name, groups) in enumerate(schemes.items()):
        if name != scheme_name: continue
        for gi, g in enumerate(groups):
            for pi in g:
                q = points[pi]; x, y = xy((q["x"], q["z"])); r = max(3, int(step * (w - 40) / (max_x - min_x) * 0.42)); d.ellipse((x-r,y-r,x+r,y+r), fill=colors[gi % len(colors)])
        d.text((20, 10), f"R11 临时切分试验：{scheme_name}，点=可行域采样点；颜色=子区域", fill="black")
    for r in original.get("rooms", []):
        if f"R{r.get('region_id')}" == "R11":
            continue
    im.save(path)


def main() -> int:
    a = args_parser().parse_args()
    if a.grid_step_m <= 0 or a.output_dir.resolve() == a.room_json.resolve() or a.output_dir.resolve() == a.navmesh.resolve():
        raise SystemExit("invalid temporary output configuration")
    room = room_record(a.room_json, a.label)
    bbox0, bbox1 = room["bbox_xz_m"]
    bbox = (float(bbox0[0]), float(bbox0[1]), float(bbox1[0]), float(bbox1[1]))
    scene_hash = __import__("hashlib").sha256(a.scene.read_bytes()).hexdigest()
    nav_hash = __import__("hashlib").sha256(a.navmesh.read_bytes()).hexdigest()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rt, sim, hs, mn = load_runtime(a.scene, a.navmesh, a)
    try:
        floor_y = float(room["floor_y_m"])
        points = snap_candidates(sim.pathfinder, bbox, floor_y, a.grid_step_m)
        comps, adj = grid_components(points)
        comps.sort(key=len, reverse=True)
        main_comp = comps[0] if comps else []
        schemes: dict[str, list[list[int]]] = {}
        metrics: dict[str, Any] = {}
        for k in (1, 2, 3, 4):
            groups = assign_geodesic(points, adj, main_comp, k) if main_comp else []
            schemes[f"K={k}"] = groups
            rows = []
            for n, g in enumerate(groups, 1):
                xs = [points[i]["x"] for i in g]; zs = [points[i]["z"] for i in g]
                ev = evaluate_group(g, points, sim, hs, mn, floor_y, a)
                rows.append({"id": f"apartment_{n:02d}", "area_m2_estimate": len(g) * a.grid_step_m * a.grid_step_m, "floor_y_m": floor_y, "connected": True, "point_count": len(g), "bbox_xz_m": [[min(xs), min(zs)], [max(xs), max(zs)]], "feasible_mask": [{"x": points[i]["x"], "z": points[i]["z"]} for i in g], **ev})
            metrics[f"K={k}"] = rows
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = {"schema": "avengine_room_split_experiment_v1", "status": "temporary_only", "created_at": now, "input": {"room_json": str(a.room_json), "scene": str(a.scene), "scene_sha256": scene_hash, "navmesh": str(a.navmesh), "navmesh_sha256": nav_hash, "label": a.label, "original_floor_y_m": floor_y, "original_bbox_xz_m": [list(bbox0), list(bbox1)], "original_area_m2": room.get("floor_area_m2")}, "sampling": {"grid_step_m": a.grid_step_m, "floor_band_tolerance_m": 0.45, "main_component_points": len(main_comp), "components": [len(c) for c in comps], "stair_connector_policy": "not assigned to ordinary subrooms; points outside the selected floor band are excluded"}, "floor_analysis": {"declared_floor_y_m": floor_y, "distinct_floor_bands_in_room_bbox": [floor_y], "note": "R11 试验使用声明的上层 floor_y；不把其他高度的点混入分区"}, "schemes": metrics, "hearing_experiment": {"status": "not_run", "reason": "未提供与该房间绑定的声源/声学包探针；禁止用经验或距离阈值伪造听音通过率", "acoustic_probe_command": None if a.acoustic_probe is None else str(a.acoustic_probe)}, "production_safety": {"original_rooms_json_modified": False, "original_navmesh_modified": False, "formal_room_registry_modified": False, "formal_verdict_modified": False, "production_config_modified": False}}
        (a.output_dir / "subroom_split_experiment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (a.output_dir / "subroom_split_experiment.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["scheme", "id", "area_m2_estimate", "connected", "camera_candidate_count", "tested_two_source_combinations", "visible_two_source_combinations", "visibility_rate", "hearing_status"]
            out = csv.DictWriter(f, fieldnames=fields); out.writeheader()
            for scheme, rows in metrics.items():
                for row in rows:
                    out.writerow({"scheme": scheme, **{k: row.get(k) for k in fields[1:-1]}, "hearing_status": row["hearing"]["status"]})
        original_topdown = a.original_topdown
        if original_topdown is not None:
            if not original_topdown.is_file():
                raise SystemExit(f"original topdown does not exist: {original_topdown}")
            shutil.copy2(original_topdown, a.output_dir / "original_connectivity_topdown.png")
        draw_topdown(a.output_dir / "subroom_split_topdown_K2.png", bbox, points, schemes, json.loads(a.room_json.read_text(encoding="utf-8")), a.grid_step_m, "K=2")
        draw_topdown(a.output_dir / "subroom_split_topdown_K3.png", bbox, points, schemes, json.loads(a.room_json.read_text(encoding="utf-8")), a.grid_step_m, "K=3")
        (a.output_dir / "README.txt").write_text("这是 R11 的临时试验结果，不是正式房间清单。\nJSON 中的 feasible_mask 是导航采样点，不是矩形内部全部可行域；由于 R11 的 bbox 导航面积大于 rooms.json 面积，不能直接据此写正式数据。\nK=2/K=3 图片中的颜色是导航采样分区，不代表语义房间边界。\nhearing.status=not_run 表示没有伪造声学结论。\n", encoding="utf-8")
        print(json.dumps({"output_dir": str(a.output_dir), "points": len(points), "components": [len(c) for c in comps], "schemes": {k: [{"area": round(r["area_m2_estimate"], 2), "visible": r["visible_two_source_combinations"], "tested": r["tested_two_source_combinations"]} for r in v] for k, v in metrics.items()}}, ensure_ascii=False, indent=2))
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
