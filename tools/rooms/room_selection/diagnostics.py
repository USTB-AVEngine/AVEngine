"""Read-only calibration diagnostics: candidates, actual CPU ray hits and heights.

The manifest gate prevents accidental use of held-out houses for diagnosis.
Original triangle IDs are raw GLB IDs, not guessed semantic categories.
"""

from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape
from .geometry import load_hm3d, collision_mesh
from .navigation import sample_navigation, components, farthest_sample, ray_clear_batch
from .media import load_overhead, project_xz
from .run import load_parameters, write_json, now
from tools.rooms.runtime_config import habitat_runtime_options


def search(
    mesh,
    points,
    clearance,
    adj,
    p,
    camera_clearance,
    source_clearance,
    camera_budget,
    source_budget,
    source_ray=True,
    trace=False,
):
    main = max(components(adj), key=len, default=[])
    cams = farthest_sample(
        points, [i for i in main if clearance[i] >= camera_clearance], camera_budget
    )
    sources = farthest_sample(
        points, [i for i in main if clearance[i] >= source_clearance], source_budget
    )
    records = []
    counts = Counter()
    witness = None
    for ci in cams:
        camera = points[ci] + [0, p["camera_height_m"], 0]
        ids = [i for i in sources if i != ci]
        ends = points[ids] + [0, p["source_height_m"], 0]
        lengths = np.linalg.norm(ends - camera, axis=1)
        keep = (lengths >= p["distance_min_m"]) & (lengths <= p["distance_max_m"])
        ends = ends[keep]
        ids = np.asarray(ids)[keep]
        counts["distance_valid_camera_source_rays"] += len(ends)
        if trace and len(ends):
            directions = ends - camera
            lengths = np.linalg.norm(directions, axis=1)
            origins = np.repeat(camera[None, :], len(ends), axis=0)
            locations, rays, triangles = mesh.ray.intersects_location(
                origins, directions / lengths[:, None], multiple_hits=False
            )
            hits = {
                int(i): (location, int(t))
                for location, i, t in zip(locations, rays, triangles)
            }
            clear = []
            for i, (end, sid) in enumerate(zip(ends, ids)):
                hit, triangle = hits.get(i, (None, None))
                ok = (
                    hit is None
                    or np.linalg.norm(hit - camera)
                    >= lengths[i] - p["ray_endpoint_tolerance_m"]
                )
                clear.append(ok)
                records.append(
                    dict(
                        camera_index=int(ci),
                        source_index=int(sid),
                        camera_m=camera.tolist(),
                        source_m=end.tolist(),
                        clear=bool(ok),
                        hit_m=hit.tolist() if hit is not None else None,
                        raw_triangle_id=triangle,
                    )
                )
            clear = np.asarray(clear, bool)
        else:
            clear = ray_clear_batch(mesh, camera, ends, p["ray_endpoint_tolerance_m"])
        counts["camera_source_clear_rays"] += int(clear.sum())
        ends = ends[clear]
        ids = ids[clear]
        # Vectorized distance and angle filtering avoids a Python loop over
        # thousands of impossible pairs, without changing geometric constraints.
        ii, jj = np.triu_indices(len(ends), 1)
        if len(ii):
            dist = np.linalg.norm(ends[ii] - ends[jj], axis=1)
            a = ends[ii][:, [0, 2]] - camera[[0, 2]]
            b = ends[jj][:, [0, 2]] - camera[[0, 2]]
            cos = np.sum(a * b, axis=1) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            )
            angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            keep = (
                (dist >= p["distance_min_m"])
                & (dist <= p["distance_max_m"])
                & (angle <= p["camera_hfov_deg"])
            )
            counts["distance_and_fov_valid_pairs"] += int(keep.sum())
            for i, j, d, ang in zip(ii[keep], jj[keep], dist[keep], angle[keep]):
                if (
                    source_ray
                    and not ray_clear_batch(
                        mesh, ends[i], [ends[j]], p["ray_endpoint_tolerance_m"]
                    )[0]
                ):
                    counts["source_source_blocked_pairs"] += 1
                    continue
                witness = dict(
                    camera_m=camera.tolist(),
                    source_1_m=ends[i].tolist(),
                    source_2_m=ends[j].tolist(),
                    horizontal_angle_deg=float(ang),
                    pairwise_distances_m=[
                        float(np.linalg.norm(ends[i] - camera)),
                        float(np.linalg.norm(ends[j] - camera)),
                        float(d),
                    ],
                )
                break
        if witness is not None:
            break
    return dict(
        found=witness is not None,
        witness=witness,
        camera_candidates=len(cams),
        source_candidates=len(sources),
        counts=dict(counts),
        camera_points_m=(points[cams] + [0, p["camera_height_m"], 0]).tolist(),
        source_points_m=(points[sources] + [0, p["source_height_m"], 0]).tolist(),
        rays=records,
    )


def floor_histogram(mesh, rid):
    ids = [
        iid
        for iid, d in mesh.instances.items()
        if d["region_id"] == rid
        and d["category"] in {"floor", "rug", "carpet", "flooring"}
    ]
    corners = mesh.corners[np.isin(mesh.face_instances, ids)]
    heights = corners[:, :, 1].mean(axis=1)
    xz = corners[:, :, [0, 2]]
    u = xz[:, 1] - xz[:, 0]
    v = xz[:, 2] - xz[:, 0]
    area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    bins = (
        np.arange(math.floor(heights.min() / 0.05) * 0.05, heights.max() + 0.1, 0.05)
        if len(heights)
        else np.array([0, 0.05])
    )
    counts, _ = np.histogram(heights, bins)
    weighted, _ = np.histogram(heights, bins, weights=area)
    return dict(
        bin_edges_m=bins.tolist(),
        face_counts=counts.tolist(),
        projected_triangle_area_sums_m2=weighted.tolist(),
        area_definition="diagnostic face projected-area sum; overlapping surfaces may double count, decision uses union area",
    )


def plot(path, overhead, floor_y, result, hist, title):
    entry, im, image, _ = overhead
    canvas = Image.new("RGB", (1500, 1150), "white")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    d = ImageDraw.Draw(canvas)
    d.text((8, 5), title, fill="black", font=font)
    for panel, name in enumerate(["old", "dense_old_clearance", "dense_025_015"]):
        r = result[name]
        view = image.copy().convert("RGB")
        draw = ImageDraw.Draw(view)

        def coords(v):
            return (
                project_xz([(q[0], q[2]) for q in v], floor_y, entry, im, image.size)
                if v
                else []
            )

        for q in coords(r["source_points_m"]):
            draw.ellipse((q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3), fill="cyan")
        for q in coords(r["camera_points_m"]):
            draw.ellipse((q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4), fill="yellow")
        for ray in r["rays"]:
            a, b = coords([ray["camera_m"], ray["source_m"]])
            draw.line(
                [tuple(a), tuple(b)],
                fill=(0, 180, 0) if ray["clear"] else (255, 60, 60),
                width=1,
            )
            if not ray["clear"] and ray["hit_m"]:
                q = coords([ray["hit_m"]])[0]
                draw.ellipse((q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4), fill="magenta")
        if r["witness"]:
            w = r["witness"]
            q = coords([w["camera_m"], w["source_1_m"], w["source_2_m"]])
            draw.line([tuple(q[1]), tuple(q[0]), tuple(q[2])], fill="lime", width=7)
        canvas.paste(view.resize((495, 495)), (panel * 500, 70))
        d.text(
            (panel * 500 + 5, 38),
            f'{name}: found={r["found"]} C={r["camera_candidates"]} S={r["source_candidates"]}',
            fill="black",
            font=font,
        )
        d.text((panel * 500 + 5, 580), json.dumps(r["counts"]), fill="black", font=font)
    d.text(
        (10, 650),
        "Yellow camera / Cyan source / Magenta first blocking hit / Green witness; original overhead matrices",
        fill="black",
        font=font,
    )
    edges = np.asarray(hist["bin_edges_m"])
    vals = np.asarray(hist["projected_triangle_area_sums_m2"])
    counts = np.asarray(hist["face_counts"])
    for col, (values, label) in enumerate(
        [
            (vals, "Projected triangle area (overlap may double count)"),
            (counts, "Face count (old unweighted clustering)"),
        ]
    ):
        x0 = 20 + col * 750
        y0 = 720
        width = 700
        height = 340
        d.text((x0, y0 - 30), label, fill="black", font=font)
        d.line(
            [(x0, y0), (x0, y0 + height), (x0 + width, y0 + height)],
            fill="black",
            width=2,
        )
        for i, value in enumerate(values):
            x = x0 + i * width / max(1, len(values))
            right = x0 + (i + 1) * width / max(1, len(values))
            h = height * float(value) / max(1e-9, float(values.max(initial=0)))
            d.rectangle(
                (x, y0 + height - h, max(x + 1, right - 1), y0 + height),
                fill=(70, 110, 180),
            )
        d.text(
            (x0, y0 + height + 5),
            f"Y {edges[0]:.3f} to {edges[-1]:.3f} m; bin 0.05 m; max {values.max(initial=0):.3f}",
            fill="black",
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-cache", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads((args.baseline / "house_analysis_split.json").read_text())
    cal = set(split["calibration"])
    held = set(split["holdout"])
    _, p = load_parameters(args.baseline / "thresholds.used.yaml")
    rows = [
        json.loads(x)
        for x in (args.baseline / "rooms_registry.jsonl").read_text().splitlines()
    ]
    rng = random.Random(20261003)
    samples = []
    population = {}
    for code in ["PLACEMENT_NO_WITNESS", "MULTILEVEL_REGION_REQUIRES_REVIEW"]:
        pool = sorted(
            [
                r
                for r in rows
                if r["house"] in cal
                and r.get("human")
                and r["human"]["verdict"] == "use"
                and code in r["stage1"]["reason_codes"]
            ],
            key=lambda r: (r["house"], r["region_id"]),
        )
        population[code] = len(pool)
        for r in sorted(
            rng.sample(pool, 15), key=lambda r: (r["house"], r["region_id"])
        ):
            samples.append(
                dict(
                    rule=code,
                    house=r["house"],
                    room_label=r["room_label"],
                    region_id=r["region_id"],
                )
            )
    assert not any(s["house"] in held for s in samples)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "sample_manifest.json",
        dict(
            seed=20261003,
            calibration_only=True,
            population=population,
            samples=samples,
            created_at_sgt=now(),
        ),
    )
    by_key = {(r["house"], r["room_label"]): r for r in rows if r["house"] in cal}
    groups = defaultdict(list)
    for s in samples:
        groups[s["house"]].append(s)
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    hs = prepare_installed_habitat_runtime(**habitat_runtime_options()).habitat_sim
    summary = []
    for house, sample in groups.items():
        row = by_key[house, sample[0]["room_label"]]
        mesh = load_hm3d(row["scene_directory"])
        collision, _ = collision_mesh(row["scene_directory"])
        pf = hs.PathFinder()
        assert pf.load_nav_mesh(row["navmesh_source"])
        for s in sample:
            row = by_key[house, s["room_label"]]
            floor = max(row["floors"], key=lambda f: f["metrics"]["floor_area_m2"])
            m = floor["metrics"]
            g = shape(m["floor_polygon"])
            y = m["floor_y_m"]
            nav, points, clearance, adj, _ = sample_navigation(pf, hs, g, y, p)
            variants = {}
            for name, cc, sc, cb, sb, sr, trace in [
                ("old", 0.5, 0.5, 24, 48, True, True),
                ("dense_old_clearance", 0.5, 0.5, 128, 256, True, False),
                ("dense_025_015", 0.25, 0.15, 128, 256, True, True),
                ("dense_025_015_camera_rays", 0.25, 0.15, 128, 256, False, False),
                ("dense_015_010", 0.15, 0.1, 128, 256, True, False),
            ]:
                variants[name] = search(
                    collision, points, clearance, adj, p, cc, sc, cb, sb, sr, trace
                )
            hist = floor_histogram(mesh, row["region_id"])
            layers = [
                dict(
                    floor_y_m=f["metrics"]["floor_y_m"],
                    area_m2=f["metrics"]["floor_area_m2"],
                    height_range_m=f["metrics"]["height_range_m"],
                    face_count=f["metrics"]["floor_face_count"],
                )
                for f in row["floors"]
            ]
            dominance = m["floor_area_m2"] / sum(v["area_m2"] for v in layers)
            data = dict(
                **s,
                baseline_stage1=row["stage1"],
                navigation=nav,
                floor_layers=layers,
                main_layer_fraction=dominance,
                height_histogram=hist,
                variants=variants,
                raw_collision_source=row["collision_source"],
                category_attribution="not inferred from raw triangle index; no obstacle category claim",
                created_at_sgt=now(),
            )
            stem = f'{house}__{s["room_label"]}'
            write_json(args.output / (stem + ".json"), data)
            overhead = load_overhead(
                house, s["room_label"], args.media_cache, "http://127.0.0.1:8766"
            )
            plot(
                args.output / (stem + ".png"),
                overhead,
                y,
                variants,
                hist,
                stem + " / " + s["rule"],
            )
            summary.append(
                dict(
                    **s,
                    main_layer_fraction=dominance,
                    layers=layers,
                    variants={
                        k: dict(
                            found=v["found"],
                            camera_candidates=v["camera_candidates"],
                            source_candidates=v["source_candidates"],
                            counts=v["counts"],
                        )
                        for k, v in variants.items()
                    },
                    diagnostic_json=stem + ".json",
                    diagnostic_png=stem + ".png",
                )
            )
            write_json(
                args.output / "summary.json",
                dict(calibration_only=True, examples=summary),
            )
            print(
                "DIAG_DONE",
                stem,
                json.dumps({k: v["found"] for k, v in variants.items()}),
                flush=True,
            )
        del mesh, collision, pf


if __name__ == "__main__":
    main()
