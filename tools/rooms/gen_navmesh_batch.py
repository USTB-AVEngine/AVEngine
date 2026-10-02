#!/usr/bin/env python3
"""Serial navmesh generation with a read-only plan and per-file provenance.

Existing navmeshes are only loaded, never rebuilt or attributed retroactively.
--dry-run enumerates inputs and proposed outputs without importing Habitat or
writing files. Generation retains smy's default NavMeshSettings constructor.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
from tools.rooms.runtime_config import TASKS_ROOT, habitat_runtime_options

TEMPLATE = "hm3d_end_to_end"
RUNTIME = habitat_runtime_options()
TOOL_VERSION = "room-navmesh-provenance-v1"


def failed_train_scene_dirs() -> list[Path]:
    dirs = set()
    for path in TASKS_ROOT.glob(f"*{TEMPLATE}/task.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "fail":
            continue
        argv = record.get("argv") or []
        for i in range(len(argv) - 1):
            if argv[i] == "--scene-dir" and "/train/" in argv[i + 1]:
                dirs.add(Path(argv[i + 1]))
    return sorted(dirs)


def settings_parameters(settings) -> dict:
    """Capture every public data property, including runtime-added settings."""
    result = {}
    for name in dir(settings):
        if name.startswith("_"):
            continue
        value = getattr(settings, name)
        if callable(value):
            continue
        # Fail visibly if a future binding introduces an unserializable field.
        json.dumps(value)
        result[name] = value
    return result


def generate(scene_dir: Path, *, dry_run: bool = False) -> dict:
    glb = next(
        (
            p
            for p in sorted(scene_dir.glob("*.glb"))
            if not p.name.endswith(".semantic.glb")
        ),
        None,
    )
    if glb is None:
        return {"ok": False, "reason": "no .glb"}
    output = scene_dir / f"{glb.stem}.basis.navmesh"
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "source_glb": str(glb),
            "path": str(output),
            "action": "load_existing" if output.is_file() else "generate",
            "provenance_path": str(output) + ".provenance.json",
        }
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    runtime = prepare_installed_habitat_runtime(**RUNTIME)
    hs = runtime.habitat_sim
    if output.is_file() and output.stat().st_size > 0:
        pf = hs.PathFinder()
        if pf.load_nav_mesh(str(output)):
            return {"ok": True, "skipped_existing": True, "path": str(output)}
        return {"ok": False, "reason": "existing file does not load"}
    backend = hs.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.load_semantic_mesh = False
    backend.enable_physics = False
    settings = hs.NavMeshSettings()
    parameters = settings_parameters(settings)
    with hs.Simulator(
        hs.Configuration(backend, [hs.agent.AgentConfiguration()])
    ) as sim:
        if (
            not sim.recompute_navmesh(sim.pathfinder, settings)
            or not sim.pathfinder.is_loaded
        ):
            return {"ok": False, "reason": "recompute_navmesh failed"}
        sim.pathfinder.save_nav_mesh(str(output))
    pf = hs.PathFinder()
    if not pf.load_nav_mesh(str(output)):
        # Only the file just created by this call may be removed.
        output.unlink(missing_ok=True)
        return {"ok": False, "reason": "generated file fails to load"}
    revision = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    provenance = {
        "schema": TOOL_VERSION,
        "tool_version": TOOL_VERSION,
        "git_revision": revision,
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
            )
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": socket.gethostname(),
        "source_glb": str(glb.resolve()),
        "navmesh": str(output.resolve()),
        "NavMeshSettings": parameters,
        "settings_initialization": "NavMeshSettings() (original behavior)",
        "runtime": RUNTIME,
        "habitat_version": getattr(hs, "__version__", "unknown"),
    }
    with Path(str(output) + ".provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read-only plan; no Habitat import or writes",
    )
    args = parser.parse_args()
    scene_dirs = failed_train_scene_dirs()
    if args.house:
        scene_dirs = [d for d in scene_dirs if d.name == args.house]
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]
    print(f"plan: {len(scene_dirs)} houses", flush=True)
    results = []
    for position, scene_dir in enumerate(scene_dirs, 1):
        started = time.time()
        result = generate(scene_dir, dry_run=args.dry_run)
        result.update(house=scene_dir.name, seconds=round(time.time() - started, 1))
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    ok = sum(r["ok"] for r in results)
    print(f"SUMMARY: {ok}/{len(results)} ok", flush=True)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
