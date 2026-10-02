#!/usr/bin/env python3
"""Temporary, non-production split experiment for train scene 00006."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.rooms.runtime_config import (
    RUNTIME_PREFIX,
    MAGNUM_SITE,
    RLR_SDK_ROOT,
    MP3D_ROOT,
    TASKS_ROOT,
    MEDIA_ROOT,
    ROOM_PYTHON,
)

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tools.rooms.experiment_split_room import (  # noqa: E402
    dijkstra,
    evaluate_group,
    grid_components,
    snap_candidates,
)
from avengine.rooms.habitat_capture import (
    prepare_installed_habitat_runtime,
)  # noqa: E402

SCENE = Path(
    "/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.glb"
)
NAV = SCENE.with_suffix(".basis.navmesh")
ROOMS = Path(
    "/data/avengine_external/studio/tasks/20260901T085133Z-hm3d_end_to_end/output/render/rooms/hm3d_train_00006_HkseAnWCgqk/rooms.json"
)
OUT = Path("/data/smy/room_split_experiments/00006_20260907_v9")


def mask_rect(points, box):
    x0, z0, x1, z1 = box
    return [
        i for i, p in enumerate(points) if x0 <= p["x"] <= x1 and z0 <= p["z"] <= z1
    ]


def max_path(points, adj, group):
    anchors = group[:: max(1, len(group) // 8)][:8]
    values = []
    for source in anchors:
        distances = dijkstra(adj, points, source)
        values.extend(distances[i] for i in anchors if math.isfinite(distances[i]))
    return max(values) if values else None


def make_record(parent, name, kind, points, group, room, sim, hs, mn, test_visibility):
    _components, adjacency = grid_components(points)
    group_set = set(group)
    connected = any(group_set.issubset(set(component)) for component in _components)
    xs = [points[i]["x"] for i in group]
    zs = [points[i]["z"] for i in group]
    if test_visibility:
        opts = type(
            "Options",
            (),
            {"camera_height_m": 1.5, "source_height_m": 1.2, "hfov_deg": 90.0},
        )()
        feasibility = evaluate_group(
            group, points, sim, hs, mn, float(room["floor_y_m"]), opts
        )
    else:
        feasibility = {
            "camera_candidate_count": 0,
            "tested_two_source_combinations": 0,
            "visible_two_source_combinations": 0,
            "visibility_rate": None,
        }
    return {
        "id": name,
        "parent_room": parent,
        "class": kind,
        "floor_y_m": room["floor_y_m"],
        "bbox_xz_m": [[min(xs), min(zs)], [max(xs), max(zs)]],
        "area_m2_estimate": len(group) * 0.25 * 0.25,
        "point_count": len(group),
        "connected": connected,
        "max_sampled_path_m": max_path(points, adjacency, group),
        "feasibility": feasibility,
        "hearing": {
            "status": "not_run",
            "reason": "未调用真实声学探针；仅记录几何可见性",
        },
    }


def draw_panel(draw, title, points, bbox, offset_x, offset_y, scale, colored):
    min_x, min_z, max_x, max_z = bbox

    def project(x, z):
        return (
            int(offset_x + (x - min_x) * scale),
            int(offset_y + (max_z - z) * scale),
        )

    draw.text((offset_x, offset_y - 30), title, fill="black")
    for point in points:
        x, y = project(point["x"], point["z"])
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(185, 185, 185))
    draw.rectangle(
        (
            project(min_x, min_z)[0],
            project(min_x, max_z)[1],
            project(max_x, max_z)[0],
            project(max_x, min_z)[1],
        ),
        outline="black",
        width=2,
    )
    for box, colour in colored:
        x0, z0, x1, z1 = box
        draw.rectangle(
            (
                project(x0, z0)[0],
                project(x0, z1)[1],
                project(x1, z1)[0],
                project(x1, z0)[1],
            ),
            outline=colour,
            width=5,
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rooms = {f"R{r['region_id']}": r for r in json.loads(ROOMS.read_text())["rooms"]}
    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=RUNTIME_PREFIX,
        magnum_python_site=MAGNUM_SITE,
        rlr_sdk_root=RLR_SDK_ROOT,
        mp3d_root=MP3D_ROOT,
        allow_mp3d_environment=False,
    )
    hs, mn = runtime.habitat_sim, runtime.magnum
    config = hs.SimulatorConfiguration()
    config.scene_id = str(SCENE)
    config.load_semantic_mesh = False
    config.enable_physics = True
    if runtime.physics_config_path:
        config.physics_config_file = str(runtime.physics_config_path)
    sim = hs.Simulator(hs.Configuration(config, [hs.agent.AgentConfiguration()]))
    if not sim.pathfinder.load_nav_mesh(str(NAV)):
        raise SystemExit("cannot load original navmesh")

    points = {
        label: snap_candidates(
            sim.pathfinder,
            tuple(sum(room["bbox_xz_m"], [])),
            float(room["floor_y_m"]),
            0.25,
        )
        for label, room in rooms.items()
        if label in {"R8", "R9", "R6"}
    }
    # 两个约 2.75m × 2.75m 的核心区；门口和其余边缘不分配给小房间。
    # R8 按实际家具分区：原先标作沙发区的范围实际为餐桌侧；
    # 另一范围作为沙发侧。这里保持区域坐标不变，只纠正语义对应。
    # 中间过渡带不分配给任一小房间。
    r8_table = (-4.5, -0.8, -1.0, 2.2)
    r8_sofa = (-4.5, -4.5, -1.0, -0.8)
    r9_camera = (-6.99, -4.76, -5.0, 4.99)
    records = []
    records.append(
        make_record(
            "R8",
            "apartment_R8_01",
            "placeable",
            points["R8"],
            mask_rect(points["R8"], r8_table),
            rooms["R8"],
            sim,
            hs,
            mn,
            True,
        )
    )
    records.append(
        make_record(
            "R8",
            "apartment_R8_02",
            "placeable",
            points["R8"],
            mask_rect(points["R8"], r8_sofa),
            rooms["R8"],
            sim,
            hs,
            mn,
            True,
        )
    )
    records.append(
        make_record(
            "R9",
            "apartment_R9_01",
            "placeable",
            points["R9"],
            mask_rect(points["R9"], r9_camera),
            rooms["R9"],
            sim,
            hs,
            mn,
            True,
        )
    )
    records += [
        {
            "id": "connector_R8_remaining",
            "parent_room": "R8",
            "class": "connector_or_excluded",
            "note": "不属于左右两个核心区，不放相机和声源",
        },
        {
            "id": "connector_R9_door",
            "parent_room": "R9",
            "class": "connector",
            "bbox_xz_m": [[-5.0, -4.76], [-4.49, 4.99]],
            "note": "门侧窄带，不放相机和声源",
        },
        make_record(
            "R6",
            "connector_R6",
            "connector",
            points["R6"],
            list(range(len(points["R6"]))),
            rooms["R6"],
            sim,
            hs,
            mn,
            False,
        ),
    ]
    result = {
        "schema": "avengine_train_00006_split_experiment_v2",
        "status": "temporary_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"scene": str(SCENE), "navmesh": str(NAV), "rooms_json": str(ROOMS)},
        "rule": {
            "R1": "keep_single_placeable_room",
            "R8": "corrected_table_sofa_mapping",
            "R9": "door_side_boundary",
            "R6": "connector_only",
        },
        "regions": records,
        "production_safety": {
            "original_data_modified": False,
            "formal_verdict_modified": False,
            "formal_registry_modified": False,
            "official_media_overwritten": False,
            "hearing_verified": False,
        },
    }
    (OUT / "split_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    with (OUT / "split_result.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "parent",
                "class",
                "area_m2",
                "connected",
                "max_sampled_path_m",
                "visible_pairs",
                "tested_pairs",
                "hearing",
            ]
        )
        for record in records:
            f = record.get("feasibility", {})
            writer.writerow(
                [
                    record["id"],
                    record["parent_room"],
                    record["class"],
                    record.get("area_m2_estimate"),
                    record.get("connected"),
                    record.get("max_sampled_path_m"),
                    f.get("visible_two_source_combinations"),
                    f.get("tested_two_source_combinations"),
                    record.get("hearing", {}).get("status"),
                ]
            )
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    draw_panel(
        draw,
        "R8：绿=餐桌区，蓝=沙发区，灰=过渡带",
        points["R8"],
        tuple(sum(rooms["R8"]["bbox_xz_m"], [])),
        60,
        730,
        45,
        [(r8_sofa, (46, 160, 67)), (r8_table, (44, 100, 210))],
    )
    draw_panel(
        draw,
        "R9：绿=门外可放置区，门侧为连接带",
        points["R9"],
        tuple(sum(rooms["R9"]["bbox_xz_m"], [])),
        700,
        730,
        45,
        [(r9_camera, (46, 160, 67))],
    )
    image.save(OUT / "temporary_boundaries.png")

    jobs = [
        ("R8", "apartment_R8_01", r8_table, (-2.65, 0.0)),
        ("R8", "apartment_R8_02", r8_sofa, (-2.4, -2.0)),
        ("R9", "apartment_R9_01", r9_camera, None),
    ]
    sim.close()
    common = [
        sys.executable,
        str(ROOT / "tools/rooms/render_room_tour.py"),
        "--glb",
        str(SCENE),
        "--house",
        "hm3d_train_00006_split_temp",
        "--frames",
        "36",
        "--hfov",
        "90",
        "--cam-height",
        "1.5",
        "--pitch-deg",
        "-8",
        "--width",
        "960",
        "--height",
        "540",
        "--frame-rate",
        "12",
        "--runtime-prefix",
        RUNTIME_PREFIX,
        "--magnum-site",
        MAGNUM_SITE,
        "--rlr-sdk-root",
        RLR_SDK_ROOT,
        "--output-dir",
        str(OUT / "videos"),
    ]
    for parent, label, box, preferred_xz in jobs:
        room = rooms[parent]
        target_x = preferred_xz[0] if preferred_xz else (box[0] + box[2]) / 2
        target_z = preferred_xz[1] if preferred_xz else (box[1] + box[3]) / 2
        p = min(
            mask_rect(points[parent], box),
            key=lambda i: (points[parent][i]["x"] - target_x) ** 2
            + (points[parent][i]["z"] - target_z) ** 2,
        )
        point = points[parent][p]
        command = common + [
            "--label",
            label,
            f"--center-xz={point['x']:.6f},{point['z']:.6f}",
            "--floor-y",
            str(room["floor_y_m"]),
            "--bbox-xz=" + ",".join(map(str, box)),
        ]
        print("render", label, point, flush=True)
        subprocess.run(command, check=True)
    (OUT / "README.txt").write_text(
        "00006 临时切分试验：R1 不切；R8_01 为沙发侧，R8_02 为餐桌侧（原先标记的沙发区实际是餐桌区）；R9 门外区；R6 连接区。未修改原始数据、审核章或正式媒体。hearing.status=not_run。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
