#!/usr/bin/env python3
"""Render a resumable native-RLR RIR cache from an M6.x job plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.m3.runtime import load_compiled_acoustic_scene
from avengine.m4.runtime import M4SimulationConfig
from avengine.m6x.rir_cache import render_rir_cache


REPOSITORY = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir-job-plan", type=Path, required=True)
    parser.add_argument(
        "--acoustic-package-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--simulation-request",
        type=Path,
        default=REPOSITORY
        / "examples/m4/blender_custom/multi_source_canary_request.json",
    )
    parser.add_argument(
        "--hrtf",
        type=Path,
        default=Path("/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=("binaural", "ambisonics"), default="binaural"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thread-count", type=int)
    parser.add_argument("--job-offset", type=int, default=0)
    parser.add_argument("--job-limit", type=int)
    parser.add_argument(
        "--coordinate-translation-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--source-radius-m", type=float, default=0.0)
    parser.add_argument("--listener-radius-m", type=float, default=0.0)
    parser.add_argument("--uncompressed", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    simulation_request = args.simulation_request.resolve()
    simulation_value = load_json(simulation_request)["simulation"]
    if args.thread_count is not None:
        simulation_value = dict(simulation_value)
        simulation_value["thread_count"] = args.thread_count
    simulation = M4SimulationConfig.from_mapping(simulation_value)
    scene = load_compiled_acoustic_scene(
        args.acoustic_package_manifest.resolve(),
        allow_nonpassing_research_qa=True,
    )
    result = render_rir_cache(
        plan_path=args.rir_job_plan,
        scene=scene,
        simulation_request_path=simulation_request,
        simulation=simulation,
        output=args.output,
        layout_type=args.layout,
        hrtf_file_path=args.hrtf if args.layout == "binaural" else None,
        batch_size=args.batch_size,
        job_offset=args.job_offset,
        job_limit=args.job_limit,
        coordinate_translation_m=args.coordinate_translation_m,
        source_radius_m=args.source_radius_m,
        listener_radius_m=args.listener_radius_m,
        compressed=not args.uncompressed,
    )
    print(
        "RIR_CACHE_OK "
        f"output={result.output} "
        f"jobs={result.receipt['selected_job_count']} "
        f"full_plan_complete={result.receipt['full_plan_complete']}",
        flush=True,
    )
    return result.output


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
