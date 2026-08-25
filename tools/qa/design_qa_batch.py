#!/usr/bin/env python3
"""Design and author one constraint-driven QA v2 batch (reverse fitting).

Movers draw from the proven straight-corridor bank (natural speeds from the
per-species motion profiles); side separation is enforced at the question
anchor frames; min inter-actor separation 60 cm; per-cell composition
6 s1-move / 6 s2-move / 3 both / 1 off-screen; attribute twins swap slot
assets on the first half of each cell. Usage:

  design_qa_batch.py --output-root DIR [--cell-size 16] [--twins-per-cell 8]

Fixes over pilot48 + owner feedback 20260823: (1) speaking-order balance via
*_bfirst variants; (2) NATURAL SPEEDS - mover paths 3.0-3.8 m over the 5 s
clip (0.60-0.76 m/s, matching the v1-validated 0.757 m/s) instead of the
gliding 0.14-0.18 m/s of pilot48; (3) side separation enforced at the
question anchor frames (events 1/2), allowing long diagonal paths;
(4) off-screen candidates use the near-camera corner spot only.
Composition: 128 primaries (96 human / 32 dog) + 64 attribute twins.
Per 16-point cell: 6 s1-move, 6 s2-move, 3 both-move, 1 off-screen.
research_only; no dataset admission."""
import argparse
import json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
_ap = argparse.ArgumentParser()
_ap.add_argument("--output-root", required=True,
                 help="fresh batch inputs directory (refuses to clobber)")
_ap.add_argument("--avengine-cli", default=shutil.which("avengine") or "avengine")
_ap.add_argument("--registry", default=REPO + "/examples/runtime/source_asset_runtime_profiles.json")
_ap.add_argument("--snapshot-content",
                 default="/data/avengine_external/ue-assets/actor_content_registry_v9_20260823T033709Z/cpp/unreal_projects/SpearSim/Content")
_ap.add_argument("--cell-size", type=int, default=16)
_ap.add_argument("--twins-per-cell", type=int, default=8)
_args = _ap.parse_args()
PY_AVENGINE = _args.avengine_cli
REG = _args.registry
SNAP = _args.snapshot_content
ROOT = _args.output_root
if os.path.exists(ROOT):
    print(json.dumps({"error": "output root exists; refusing to clobber", "root": ROOT}))
    sys.exit(2)

CAMERA_POS = [-70.0, 65.0, 147.1]
CAMERA_YAW = -145.0
Z = 27.1


_reg_doc = json.load(open(REG))
_BY_ID = {a["asset_id"]: a for a in _reg_doc["assets"]}


def mesh_package_for(asset):
    su = asset["runtime_backends"]["spear_unreal"]
    mesh_dir_pkg = su["idle_animation"].split(".", 1)[0].rsplit("/", 1)[0]
    gate = mesh_dir_pkg.rsplit("/", 1)[-1]
    phys_dir = os.path.join(SNAP, "MyAssets/Audioset/Meshes", gate)
    names = [f[:-7] for f in os.listdir(phys_dir) if f.endswith(".uasset")]
    for n in names:
        if n + "_Skeleton" in names:
            return mesh_dir_pkg + "/" + n
    if "runtime" in names:
        return mesh_dir_pkg + "/runtime"
    raise RuntimeError(f"cannot identify skeletal mesh in {phys_dir}: {names}")


def actor_entry(slot, asset_id):
    rec = _BY_ID[asset_id]
    su = rec["runtime_backends"]["spear_unreal"]
    bp = su["blueprint_class_path"]
    bp_pkg = bp.split(".", 1)[0]
    mesh_pkg = mesh_package_for(rec)
    mesh_name = mesh_pkg.rsplit("/", 1)[-1]

    def phys(package):
        p = os.path.join(SNAP, package.split("/Game/", 1)[1] + ".uasset")
        if not os.path.isfile(p):
            raise RuntimeError(f"missing physical source: {p}")
        return p

    return {
        "asset_id": asset_id,
        "legacy_timeline_actor_id": f"{rec['identity']['species_id']}_{slot[-1]}",
        "physical_authorized_internal_sources": {
            "blueprint": phys(bp_pkg),
            "graph_derived_mesh": phys(mesh_pkg),
            "idle": phys(su["idle_animation"].split(".", 1)[0]),
            "walking": phys(su["walking_animation"].split(".", 1)[0]),
        },
        "profile_alias": asset_id,
        "revision": rec["revision"],
        "source_slot_id": slot,
        "ue_binding": {
            "blueprint_object_path": bp,
            "blueprint_package": bp_pkg,
            "graph_derived_mesh": {
                "derivation": "direct graph dependency of the selected Blueprint; profile binds blueprint_component and declares no standalone mesh path",
                "object_path": f"{mesh_pkg}.{mesh_name}",
                "package": mesh_pkg,
            },
            "idle_object_path": su["idle_animation"],
            "idle_package": su["idle_animation"].split(".", 1)[0],
            "profile_skeletal_mesh_binding": su["skeletal_mesh_binding"],
            "profile_skeletal_mesh_path": su["skeletal_mesh_path"],
            "walking_object_path": su["walking_animation"],
            "walking_package": su["walking_animation"].split(".", 1)[0],
        },
    }


def selection_doc(a1, a2):
    return {
        "schema": "avengine_apartment_actor_selection_v1",
        "asset_authorization": "verified_internal",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "QA v2 pilot48 batch; research only.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actors": [actor_entry("source1", a1), actor_entry("source2", a2)],
    }




HUMANS = {"blue": "rocketbox_human_male_adult_01_top_blue_research_v1",
          "green": "rocketbox_human_male_adult_01_top_green_research_v1",
          "burgundy": "rocketbox_human_male_adult_01_top_burgundy_research_v1"}
DOGS = {"collie": "generated_border_collie_black_white_medium_standard_adult_research_v1",
        "labrador": "generated_labrador_yellow_medium_standard_adult_research_v1"}
HUMAN_COMBOS = [("blue", "green"), ("blue", "burgundy"), ("green", "burgundy")]

# Movers draw from near-straight chords actually navigated by unique1000 v1
# episodes (proven furniture-free at natural speed); statics stand on corridor
# endpoints. Bank: examples/qa_v2/straight_corridor_bank_v1.json.
OFF_CORNER = (-152.0, 38.0)
CAM2 = (-70.0, 65.0)
from avengine.route_sampling import (  # shared with the renderer and bank
    arc_length_cm,
    planar_cumulative,
    sample_polyline,
)

_BANK = json.load(open(REPO + "/examples/qa_v2/straight_corridor_bank_v1.json"))["segments"]
_DIRECTED = []
for _seg in _BANK:
    _DIRECTED.append((tuple(_seg["start"]), tuple(_seg["end"])))
    _DIRECTED.append((tuple(_seg["end"]), tuple(_seg["start"])))
_STANDS = sorted({tuple(_seg["start"]) for _seg in _BANK} | {tuple(_seg["end"]) for _seg in _BANK})


def _side(p):
    import math as _m
    th = _m.radians(CAMERA_YAW)
    fx, fy = _m.cos(th), _m.sin(th)
    dx, dy = p[0] - CAM2[0], p[1] - CAM2[1]
    c = (fx * dy - fy * dx) / max(_m.hypot(dx, dy), 1e-9)
    return ("right" if c > 0 else "left"), abs(c)


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _route_points(route):
    """A route is (start, end) for a chord or a waypoint list for a polyline."""
    if len(route) == 2 and all(
        isinstance(value, (int, float)) for value in route[0]
    ):
        return [list(route[0]), list(route[1])]
    return [list(point) for point in route]


def _at(route, t):
    """Planar position a fraction t along a route, by arc length.

    Chords keep the old straight-line interpolation exactly; polylines are
    resampled so the actor holds one speed across segments of unequal length.
    """
    points = _route_points(route)
    if len(points) == 2:
        first, second = points
        if first == second:
            return tuple(first[:2])
        return _lerp(tuple(first[:2]), tuple(second[:2]), t)
    padded = [list(point) + [0.0] * (3 - len(point)) for point in points]
    cumulative = planar_cumulative(padded)
    position, _ = sample_polyline(padded, cumulative, cumulative[-1] * t)
    return (position[0], position[1])


def route_speed_mps(route, clip_seconds=5.0):
    """The speed the route implies: arc length over the clip duration."""
    padded = [list(point) + [0.0] * (3 - len(point))
              for point in _route_points(route)]
    return arc_length_cm(padded) / 100.0 / clip_seconds


def walk_from_bank(i):
    return _DIRECTED[i % len(_DIRECTED)]


def stand_from_bank(i):
    q = _STANDS[i % len(_STANDS)]
    return q, q


def min_separation_cm(p1, p2):
    import math as _m
    worst = 1e9
    for t in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        a, b = _at(p1, t), _at(p2, t)
        worst = min(worst, _m.hypot(a[0] - b[0], a[1] - b[1]))
    return worst


def prog_short(asset_id, reg_assets):
    a = reg_assets[asset_id]
    ra = a.get("realized_attributes", {})
    if "top_color" in ra:
        return ra["top_color"]
    return {"border_collie": "collie", "labrador_retriever": "labrador"}[a["identity"]["breed_id"]]


sys.path.insert(0, "/data/jzy/tmp")
reg = json.load(open(REG))
REG_ASSETS = {a["asset_id"]: a for a in reg["assets"]}

cells = []
for combo in HUMAN_COMBOS:
    for order in ("afirst", "bfirst"):
        cells.append(("human", HUMANS[combo[0]], HUMANS[combo[1]], order))
for order in ("afirst", "bfirst"):
    cells.append(("dog", DOGS["collie"], DOGS["labrador"], order))

specs = []
E1_T, E2_T = 4.0 / 75.0, 22.5 / 75.0


def _gates_ok(p1, p2, offscreen=False):
    q1, q2 = _at(p1, E1_T), _at(p2, E1_T)
    (sa, ma), (sb, mb) = _side(q1), _side(q2)
    r1, r2 = _at(p1, E2_T), _at(p2, E2_T)
    (_, m2a), (_, m2b) = _side(r1), _side(r2)
    side_ok = (sa != sb and ma >= 0.12 and mb >= 0.12 and m2a >= 0.06 and m2b >= 0.06)
    return (side_ok or offscreen) and min_separation_cm(p1, p2) >= 60.0


def find_combo(seed, mode):
    n_walk, n_stand = len(_DIRECTED), len(_STANDS)
    if mode == "s1_moving_s2_static":
        for a in range(n_walk):
            for b in range(n_stand):
                p1 = _DIRECTED[(seed + a) % n_walk]
                p2 = stand_from_bank(seed * 3 + b)
                if _gates_ok(p1, p2):
                    return p1, p2
    elif mode == "s1_static_s2_moving":
        for a in range(n_stand):
            for b in range(n_walk):
                p1 = stand_from_bank(seed * 3 + a)
                p2 = _DIRECTED[(seed + b) % n_walk]
                if _gates_ok(p1, p2):
                    return p1, p2
    elif mode == "both_moving":
        for a in range(n_walk):
            for b in range(n_walk):
                p1 = _DIRECTED[(seed + a) % n_walk]
                p2 = _DIRECTED[(seed * 5 + b) % n_walk]
                if p1 != p2 and _gates_ok(p1, p2):
                    return p1, p2
    else:  # offscreen
        for a in range(n_walk):
            p1 = _DIRECTED[(seed + a) % n_walk]
            p2 = (OFF_CORNER, OFF_CORNER)
            if _gates_ok(p1, p2, offscreen=True):
                return p1, p2
    raise RuntimeError(f"no combo for mode {mode} seed {seed}")


for ci, (kind, a1, a2, order) in enumerate(cells):
    for made in range(_args.cell_size):
        pid = f"b{ci * _args.cell_size + made + 1:03d}"
        seed = ci * _args.cell_size + made
        if made < 6:
            motion = "s1_moving_s2_static"
            p1, p2 = find_combo(seed, motion)
        elif made < 12:
            motion = "s1_static_s2_moving"
            p1, p2 = find_combo(seed, motion)
        elif made < _args.cell_size - 1:
            motion = "both_moving"
            p1, p2 = find_combo(seed, motion)
        else:
            motion = "s1_moving_s2_static"
            p1, p2 = find_combo(seed, "offscreen")
        s1s, s2s = prog_short(a1, REG_ASSETS), prog_short(a2, REG_ASSETS)
        pk = "two_human" if kind == "human" else "two_dog"
        suffix = "_turn_taking_v1" if order == "afirst" else "_turn_taking_bfirst_v1"
        specs.append({
            "point_id": pid, "pair_kind": kind,
            "source1_asset": a1, "source2_asset": a2,
            "s1_start": list(p1[0]) + [Z], "s1_end": list(p1[1]) + [Z],
            "s2_start": list(p2[0]) + [Z], "s2_end": list(p2[1]) + [Z],
            "motion_case": motion, "offscreen_candidate": made == _args.cell_size - 1,
            "twin_of": None,
            "program_id": f"qa_v2_{pk}_{s1s}_{s2s}{suffix}",
        })

# 64 twins: first 8 points of each cell, assets swapped (program follows assets)
by_id = {s["point_id"]: s for s in specs}
twins = []
for ci in range(len(cells)):
    for j in range(_args.twins_per_cell):
        src = by_id[f"b{ci * _args.cell_size + j + 1:03d}"]
        kind = src["pair_kind"]
        a1, a2 = src["source2_asset"], src["source1_asset"]
        s1s, s2s = prog_short(a1, REG_ASSETS), prog_short(a2, REG_ASSETS)
        pk = "two_human" if kind == "human" else "two_dog"
        suffix = "_turn_taking_v1" if "bfirst" not in src["program_id"] else "_turn_taking_bfirst_v1"
        twins.append(dict(src, point_id=f"bt{len(twins) + 1:03d}",
                          source1_asset=a1, source2_asset=a2,
                          twin_of=src["point_id"],
                          program_id=f"qa_v2_{pk}_{s1s}_{s2s}{suffix}"))
specs.extend(twins)


# ---- selection docs + author (reuse pilot48 generator semantics) ----

os.makedirs(ROOT)
results, sel_cache = [], {}
for s in specs:
    pdir = os.path.join(ROOT, s["point_id"])
    os.makedirs(pdir)
    key = (s["source1_asset"], s["source2_asset"])
    if key not in sel_cache:
        sel_cache[key] = selection_doc(*key)  # noqa: F821
    with open(os.path.join(pdir, "actor_selection.json"), "w") as f:
        json.dump(sel_cache[key], f, ensure_ascii=False, indent=2)
    with open(os.path.join(pdir, "spec.json"), "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    cmd = [PY_AVENGINE, "m5", "author-current-apartment-visual-timeline",
           "--actor-selection", os.path.join(pdir, "actor_selection.json"),
           "--source-asset-registry", REG,
           "--camera-position-ue-cm", *map(str, CAMERA_POS),
           "--camera-yaw-deg", str(CAMERA_YAW),
           "--human-start-ue-cm", *map(str, s["s1_start"]),
           "--human-end-ue-cm", *map(str, s["s1_end"]),
           "--beagle-start-ue-cm", *map(str, s["s2_start"]),
           "--beagle-end-ue-cm", *map(str, s["s2_end"]),
           "--output", os.path.join(pdir, "timeline.json")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    ok = proc.returncode == 0
    results.append({"point_id": s["point_id"], "authored": ok,
                    "error": None if ok else (proc.stdout + proc.stderr)[-300:]})

manifest = {
    "schema": "avengine_qa_v2_batch2d_v1", "status": "research_only",
    "qualification_claim": False,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "inputs_root": ROOT,
    "counts": {"total": len(specs),
               "authored": sum(1 for r in results if r["authored"]),
               "twins": len(twins)},
    "author_results": [r for r in results if not r["authored"]] or "all_authored",
}
with open(os.path.join(ROOT, "batch_manifest.json"), "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(json.dumps({"inputs_root": ROOT, "counts": manifest["counts"],
                  "failed": [r["point_id"] for r in results if not r["authored"]][:8]},
                 ensure_ascii=False))
