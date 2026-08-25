#!/usr/bin/env python3
"""Render the apartment route bank as a top-down map: engine navigation vs the hand-mined corridors.

Draws what UE's own navmesh considers walkable, the polylines its pathfinder
returns, and the straight corridor bank we mined by hand, so the coverage gap
between the two route sources is visible at a glance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK"):
    if any(f.name == candidate for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED = "#0F1619", "#77868D"
NAV, PATH, CORRIDOR = "#1baf7a", "#2a78d6", "#eb6834"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", required=True, type=Path)
    p.add_argument("--corridors", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--min-length-m", type=float, default=0.5)
    p.add_argument("--max-paths", type=int, default=24,
                   help="how many routes to draw, spread across the length range")
    p.add_argument("--scene-bundle", type=Path,
                   help="Studio scene bundle whose obstacle map draws the floor plan")
    return p.parse_args()


def draw_floor_plan(ax, bundle_path: Path) -> dict | None:
    """Paint the draft occupancy grid (walkable vs blocked) as the backdrop."""
    import base64

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    obstacle = bundle.get("obstacle_map") or {}
    packed = obstacle.get("navmesh_grid_packbits_b64")
    shape = obstacle.get("grid_shape")
    bounds = obstacle.get("bounds_m")
    if not (packed and shape and bounds):
        return None
    rows, cols = int(shape[0]), int(shape[1])
    flat = np.unpackbits(np.frombuffer(base64.b64decode(packed), dtype=np.uint8))
    grid = flat[: rows * cols].reshape(rows, cols)          # row=z, col=x
    (x0, _, z0), (x1, _, z1) = bounds[0], bounds[1]
    ax.imshow(
        grid,
        origin="lower",
        extent=[x0 * 100.0, x1 * 100.0, z0 * 100.0, z1 * 100.0],
        cmap=matplotlib.colors.ListedColormap(["#2B3540", "#E8EDF1"]),
        interpolation="nearest",
        zorder=0,
        alpha=0.95,
    )
    cell_area = float(obstacle.get("meters_per_pixel", 0.05)) ** 2
    return {"walkable_m2": float(grid.sum()) * cell_area,
            "blocked_m2": float((1 - grid).sum()) * cell_area}


def main() -> int:
    args = parse_args()
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    routes_in = bank.get("routes", [])
    probe = {
        "paths": [r["waypoints_ue_cm"] for r in routes_in],
        "random_points": [pt for r in routes_in for pt in r["waypoints_ue_cm"][:1]],
    }
    fig, ax = plt.subplots(figsize=(13.0, 9.0), dpi=150)
    floor = None
    if args.scene_bundle and args.scene_bundle.is_file():
        floor = draw_floor_plan(ax, args.scene_bundle)

    corridor_pts = []
    if args.corridors and args.corridors.is_file():
        bank = json.loads(args.corridors.read_text(encoding="utf-8"))
        segments = bank.get("segments", [])
        for index, seg in enumerate(segments):
            ax.plot([seg["start"][0], seg["end"][0]],
                    [seg["start"][1], seg["end"][1]],
                    color=CORRIDOR, alpha=0.95, linewidth=2.6, zorder=9,
                    label=f"手挖直线走廊库（{len(segments)} 段）" if index == 0 else None)
            corridor_pts += [seg["start"], seg["end"]]

    points = np.asarray(probe.get("random_points") or [], dtype=float)
    points = points[(points != 0).any(axis=1)] if points.size else points

    routes = [np.asarray(r, dtype=float) for r in (probe.get("paths") or [])]
    routes = [r for r in routes if r.ndim == 2 and r.shape[0] >= 2
              and np.linalg.norm(np.diff(r[:, :2], axis=0), axis=1).sum()
              >= args.min_length_m * 100.0]

    total_routes = len(routes)
    if routes:
        all_lengths = np.array([np.linalg.norm(np.diff(r[:, :2], axis=0), axis=1).sum()
                                for r in routes])
        # spread the drawn sample across the length range instead of the first N
        order = np.argsort(all_lengths)
        picks = np.unique(np.linspace(0, len(order) - 1,
                                      min(args.max_paths, len(order))).astype(int))
        routes = [routes[int(order[i])] for i in picks]

    for index, route in enumerate(routes):
        ax.plot(route[:, 0], route[:, 1], color=PATH, linewidth=1.7, alpha=0.8,
                zorder=5,
                label=f"引擎自动规划路径（抽样 {len(routes)}/{total_routes}）"
                if index == 0 else None)
        if route.shape[0] > 2:
            ax.scatter(route[1:-1, 0], route[1:-1, 1], s=16, color="white",
                       edgecolor=PATH, linewidth=1.1, zorder=6)

    if routes:
        lengths = np.array([np.linalg.norm(np.diff(r[:, :2], axis=0), axis=1).sum()
                            for r in routes])
        best = routes[int(lengths.argmax())]
        ax.plot(best[:, 0], best[:, 1], color=PATH, linewidth=3.6, zorder=7)
        ax.scatter(best[[0], 0], best[[0], 1], s=150, marker="^", color=PATH, zorder=8)
        ax.scatter(best[[-1], 0], best[[-1], 1], s=150, marker="s", color=PATH, zorder=8)
        mid = best[best.shape[0] // 2]
        ax.annotate(f"最长 {lengths.max()/100:.1f} m · {best.shape[0]} 个路点",
                    (mid[0], mid[1]), fontsize=11, color=PATH, weight="bold",
                    xytext=(10, 10), textcoords="offset points")

    if points.size:
        step = max(1, points.shape[0] // 400)
        shown = points[::step]
        ax.scatter(shown[:, 0], shown[:, 1], s=7, color=NAV, zorder=4, alpha=0.55,
                   linewidth=0,
                   label=f"引擎判定可行点（采样 {points.shape[0]}，显示 {shown.shape[0]}）")

    if corridor_pts:
        cp = np.asarray(corridor_pts, dtype=float)
        rect = Rectangle((cp[:, 0].min(), cp[:, 1].min()),
                         np.ptp(cp[:, 0]), np.ptp(cp[:, 1]),
                         fill=False, edgecolor=CORRIDOR, linestyle="--",
                         linewidth=1.4, alpha=0.8, zorder=2,
                         label="走廊库覆盖范围")
        ax.add_patch(rect)
        if points.size:
            area_bank = float(np.ptp(cp[:, 0]) * np.ptp(cp[:, 1]))
            area_nav = float(np.ptp(points[:, 0]) * np.ptp(points[:, 1]))
            note = (f"走廊库覆盖包围盒 {area_bank/1e4:.1f} m²")
            if floor:
                note += (f" / 草稿可行域实测 {floor['walkable_m2']:.1f} m²"
                         f"  →  约 {100*area_bank/1e4/floor['walkable_m2']:.0f}%")
            else:
                note += f" / 引擎点包围盒 {area_nav/1e4:.1f} m²"
            ax.text(0.5, -0.075, note, transform=ax.transAxes, fontsize=10.5,
                    color=CORRIDOR, va="top", ha="center")

    ax.set_aspect("equal")
    ax.set_xlabel("UE X (cm)", color=MUTED, fontsize=10)
    ax.set_ylabel("UE Y (cm)", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#D9E1E4")
    ax.grid(color="#EEF2F3", linewidth=0.8)
    ax.set_title("公寓路线图：UE 引擎导航 vs 手挖走廊库", color=INK, fontsize=15, pad=14)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.92,
              edgecolor="#D9E1E4", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white")
    print(json.dumps({"output": str(args.output),
                      "points": int(points.shape[0]) if points.size else 0,
                      "paths_drawn": len(routes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
