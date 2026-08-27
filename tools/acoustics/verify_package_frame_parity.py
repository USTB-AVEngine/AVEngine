#!/usr/bin/env python3
"""Cross-system frame parity: the same rays in Habitat and in the package.

The one check that would have caught the sideways-HM3D package on day one,
made permanent. Rays are cast from caller-supplied anchors in six axis
directions twice over: once in Habitat against the render mesh with physics
enabled, once inside the compiled package through the native runtime's own
ray tracer. A frame-correct package reproduces Habitat's hit distances to
centimetres; a mis-framed one misses wholesale - measured on the incident
that motivated this tool, the sideways package failed 12 of 12 while the
corrected one agreed 12 of 12 with every delta at 0.000 m.

Anchors default to the room manifest's first connectivity pair, lifted to
listener and source heights - the exact points whose impulse responses get
published, which is what makes the parity meaningful.

Exit code 0 only when every ray agrees. This is a frame check, not a mesh
QA: open scan seams that both systems miss identically still pass, and
should.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

DIRECTIONS = {
    "down": (0.0, -1.0, 0.0),
    "up": (0.0, 1.0, 0.0),
    "east": (1.0, 0.0, 0.0),
    "west": (-1.0, 0.0, 0.0),
    "north": (0.0, 0.0, -1.0),
    "south": (0.0, 0.0, 1.0),
}


def habitat_truths(args, anchors) -> list[dict]:
    import numpy as np
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_site,
        rlr_sdk_root=args.rlr_sdk_root,
        mp3d_root=args.mp3d_root,
        allow_mp3d_environment=False,
    )
    hs = runtime.habitat_sim
    backend = hs.SimulatorConfiguration()
    backend.scene_id = str(args.scene)
    backend.load_semantic_mesh = False
    # Physics on is load-bearing: with it off the static stage has no
    # collision mesh and every ray sails through the house unhit.
    backend.enable_physics = True
    simulator = hs.Simulator(hs.Configuration(backend, [hs.agent.AgentConfiguration()]))
    try:
        records = []
        for anchor_name, origin in anchors.items():
            for direction_name, vector in DIRECTIONS.items():
                ray = hs.geo.Ray(
                    np.asarray(origin, dtype=np.float32),
                    np.asarray(vector, dtype=np.float32),
                )
                result = simulator.cast_ray(ray, float(args.maximum_distance_m))
                distance = (
                    float(result.hits[0].ray_distance) if result.has_hits() else None
                )
                records.append(
                    {
                        "check_id": f"{anchor_name}_{direction_name}",
                        "origin_m": [float(value) for value in origin],
                        "direction": list(vector),
                        "habitat_distance_m": distance,
                    }
                )
        return records
    finally:
        simulator.close()


def package_replay(args, truths: list[dict]):
    from avengine.acoustics.runtime import (
        RLRSimulationConfig,
        RuntimeAnchor,
        RUNTIME_MODE_CURRENT_INSTALLED,
        load_compiled_acoustic_scene,
        simulate_compiled_acoustic_scene,
    )

    scene = load_compiled_acoustic_scene(
        args.package_manifest, allow_nonpassing_research_qa=True
    )
    checks = tuple(
        {
            "check_id": record["check_id"],
            "expectation": "hit_within_m",
            "distance_m": float(args.maximum_distance_m),
            "origin_m": record["origin_m"],
            "direction": record["direction"],
        }
        for record in truths
    )
    simulation = RLRSimulationConfig.from_mapping(
        {
            "sample_rate_hz": 16000, "max_ir_seconds": 0.1, "frequency_bands": 4,
            "direct_sh_order": 1, "indirect_sh_order": 1, "direct_ray_count": 8,
            "indirect_ray_count": 8, "indirect_ray_depth": 2,
            "source_ray_count": 8, "source_ray_depth": 2,
            "max_diffraction_order": 1, "thread_count": 4, "unit_scale": 1.0,
            "global_volume": 1.0, "speed_of_sound_m_s": 343.0, "direct": True,
            "indirect": True, "diffraction": False, "transmission": False,
            "mesh_simplification": False, "temporal_coherence": False,
            "channel_layout": {"type": "ambisonics", "channel_count": 4},
        }
    )
    result = simulate_compiled_acoustic_scene(
        scene,
        simulation,
        source=RuntimeAnchor(anchor_id="s", position_m=tuple(truths[-1]["origin_m"])),
        listener=RuntimeAnchor(anchor_id="l", position_m=tuple(truths[0]["origin_m"])),
        runtime_mode=RUNTIME_MODE_CURRENT_INSTALLED,
        runtime_prefix=args.runtime_prefix,
        rlr_sdk_root=args.rlr_sdk_root,
        magnum_python_site=args.magnum_site,
        ray_checks=checks,
    )
    return result.ray_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--scene", required=True, type=Path,
                        help="render mesh Habitat loads (never *.basis.glb)")
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--room-manifest", type=Path,
                        help="anchors from its first connectivity pair")
    parser.add_argument("--anchor", nargs=3, type=float, action="append",
                        metavar=("X", "Y", "Z"),
                        help="explicit anchor(s); overrides --room-manifest")
    parser.add_argument("--mp3d-root", default="/data/datasets/habitat_data")
    parser.add_argument("--maximum-distance-m", type=float, default=50.0)
    parser.add_argument("--tolerance-m", type=float, default=0.05)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.scene.name.endswith(".basis.glb"):
        raise SystemExit("refusing *.basis.glb: no BasisImporter, renders segfault")

    anchors: dict[str, tuple[float, float, float]] = {}
    if args.anchor:
        for index, point in enumerate(args.anchor):
            anchors[f"anchor{index}"] = tuple(float(v) for v in point)
    elif args.room_manifest:
        pair = json.loads(args.room_manifest.read_text(encoding="utf-8"))[
            "connectivity_pairs"
        ][0]
        start = pair["start_m"]
        anchors["listener"] = (start[0], start[1] + 1.5, start[2])
        anchors["source"] = (pair["end_m"][0], pair["end_m"][1] + 1.2, pair["end_m"][2])
    else:
        raise SystemExit("give --room-manifest or at least one --anchor")

    truths = habitat_truths(args, anchors)
    reports = package_replay(args, truths)

    rows = []
    agree = 0
    for record, report in zip(truths, reports):
        habitat = record["habitat_distance_m"]
        package = report["cpu_first_hit_distance_m"]
        if habitat is None and package is None:
            verdict, ok = "both-miss", True
        elif habitat is None or package is None:
            verdict, ok = "hit/miss mismatch", False
        else:
            delta = abs(habitat - package)
            ok = delta <= args.tolerance_m
            verdict = f"delta {delta:.3f} m"
        agree += ok
        rows.append(
            {
                "check_id": record["check_id"],
                "habitat_distance_m": habitat,
                "package_distance_m": package,
                "verdict": verdict,
                "ok": ok,
            }
        )
        habitat_text = "  none" if habitat is None else f"{habitat:6.3f}"
        package_text = "  none" if package is None else f"{package:6.3f}"
        print(
            f"{'OK ' if ok else 'BAD'} {record['check_id']:20s} "
            f"habitat {habitat_text}  package {package_text}  {verdict}"
        )
    total = len(rows)
    print(f"frame parity: {agree}/{total} rays agree "
          f"(tolerance {args.tolerance_m} m)")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "schema": "avengine_package_frame_parity_v1",
                    "scene": str(args.scene),
                    "package_manifest": str(args.package_manifest),
                    "tolerance_m": args.tolerance_m,
                    "agree": agree,
                    "total": total,
                    "rays": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    return 0 if agree == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
