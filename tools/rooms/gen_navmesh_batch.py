#!/usr/bin/env python3
"""Batch-generate .basis.navmesh for the failed train houses.

Each house's navigation mesh is rebuilt from its .glb with the habitat
recompute_navmesh path (the same one the repo's `m1 build-navmesh` uses) and
saved as <scene-dir>/<scene_id>.basis.navmesh - the exact file the
hm3d_end_to_end tasks check for. Verified against the official val navmeshes
(area diff 0.8% / 3.4% on the test houses).

Strictly serial (shared GPU), idempotent (skips houses whose navmesh already
exists and loads), and never touches anything but the train house dirs.

Usage:
  nohup nice -n 10 python3 tools/rooms/gen_navmesh_batch.py > logs/navmesh_gen.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime  # noqa: E402

TASKS_ROOT = Path("/data/avengine_external/studio/tasks")
TEMPLATE = "hm3d_end_to_end"

RUNTIME = dict(
    runtime_prefix="/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z",
    magnum_python_site="/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages",
    rlr_sdk_root="/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg",
    mp3d_root="/data/datasets/habitat_data",
    allow_mp3d_environment=False,
)


def failed_train_scene_dirs() -> list[Path]:
    dirs: set[Path] = set()
    for task_json in TASKS_ROOT.glob(f"*{TEMPLATE}/task.json"):
        record = json.loads(task_json.read_text(encoding="utf-8"))
        if record.get("status") != "fail":
            continue
        argv = record.get("argv") or []
        for i in range(len(argv) - 1):
            if argv[i] == "--scene-dir":
                scene = Path(argv[i + 1])
                if "/train/" in str(scene):
                    dirs.add(scene)
    return sorted(dirs)


def generate(scene_dir: Path) -> dict:
    glbs = sorted(scene_dir.glob("*.glb"))
    glb = next((g for g in glbs if not g.name.endswith(".semantic.glb")), None)
    if glb is None:
        return {"ok": False, "reason": "no .glb"}
    output = scene_dir / f"{glb.stem}.basis.navmesh"
    if output.is_file() and output.stat().st_size > 0:
        # idempotent: already generated; verify it loads and move on.
        runtime = prepare_installed_habitat_runtime(**RUNTIME)
        pf = runtime.habitat_sim.PathFinder()
        if pf.load_nav_mesh(str(output)):
            return {"ok": True, "skipped_existing": True, "path": str(output)}
        return {"ok": False, "reason": "existing file does not load"}

    runtime = prepare_installed_habitat_runtime(**RUNTIME)
    hs = runtime.habitat_sim
    backend = hs.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    settings = hs.NavMeshSettings()
    with hs.Simulator(hs.Configuration(backend, [hs.agent.AgentConfiguration()])) as sim:
        success = bool(sim.recompute_navmesh(sim.pathfinder, settings))
        if not success or not sim.pathfinder.is_loaded:
            return {"ok": False, "reason": "recompute_navmesh failed"}
        sim.pathfinder.save_nav_mesh(str(output))

    pf = hs.PathFinder()
    loaded = bool(pf.load_nav_mesh(str(output)))
    if not loaded:
        output.unlink(missing_ok=True)
        return {"ok": False, "reason": "generated file fails to load"}
    return {
        "ok": True,
        "path": str(output),
        "byte_size": output.stat().st_size,
        "navigable_area_m2": float(pf.navigable_area),
        "num_islands": int(pf.num_islands),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", help="only this scene dir basename")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log", type=Path, default=Path("logs/navmesh_gen.log"))
    args = parser.parse_args()

    scene_dirs = failed_train_scene_dirs()
    if args.house:
        scene_dirs = [d for d in scene_dirs if d.name == args.house]
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]
    print(f"plan: {len(scene_dirs)} houses", flush=True)

    results: list[dict] = []
    for position, scene_dir in enumerate(scene_dirs, 1):
        started = time.time()
        result = generate(scene_dir)
        result["house"] = scene_dir.name
        result["seconds"] = round(time.time() - started, 1)
        results.append(result)
        line = (
            f"[{position}/{len(scene_dirs)}] {scene_dir.name} "
            f"ok={result['ok']} {result.get('reason', '')}"
            f"({result.get('navigable_area_m2', '')}m2, {result['seconds']}s)"
        )
        print(line, flush=True)

    ok = sum(1 for r in results if r["ok"])
    print(f"SUMMARY: {ok}/{len(results)} ok", flush=True)
    for r in results:
        if not r["ok"]:
            print(f"  FAILED {r['house']}: {r.get('reason')}", flush=True)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
