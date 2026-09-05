#!/usr/bin/env python3
"""Run the modern RLR TraceRay checks for a compiled acoustic package.

The compiler-side enclosure probe is deliberately retained as a CPU diagnostic.
When the external Habitat/RLR runtime is available this tool replays those same
origins and directions through the native context, so a package cannot silently
report only a CPU preflight. Missing runtime inputs are returned as an explicit
"unavailable" result.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.acoustics.runtime import (  # noqa: E402
    RLRSimulationConfig,
    RuntimeAnchor,
    RuntimeContractError,
    RuntimeExecutionError,
    RuntimeUnavailableError,
    RUNTIME_MODE_CURRENT_INSTALLED,
    load_compiled_acoustic_scene,
    simulate_compiled_acoustic_scene,
)


def _dependency_errors(
    runtime_prefix: str | Path | None,
    magnum_site: str | Path | None,
    rlr_sdk_root: str | Path | None,
) -> list[str]:
    errors = []
    for name, raw in (
        ("runtime_prefix", runtime_prefix),
        ("magnum_site", magnum_site),
        ("rlr_sdk_root", rlr_sdk_root),
    ):
        if raw is None or not str(raw).strip():
            errors.append(f"{name} argument is missing")
            continue
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            errors.append(f"{name} directory is unavailable: {path}")
    if rlr_sdk_root is not None and str(rlr_sdk_root).strip():
        root = Path(rlr_sdk_root).expanduser().resolve()
        for relative in (
            Path("headers") / "RLRAudioPropagation.h",
            Path("libs") / "linux" / "x64" / "libRLRAudioPropagation.so",
        ):
            path = root / relative
            if not path.is_file():
                errors.append(f"rlr_sdk_root missing required file: {path}")
    return errors


def _automatic_declarations(report: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    automatic = report.get("automatic_enclosure_probe")
    if not isinstance(automatic, Mapping):
        return ()
    directions = automatic.get("directions")
    origins = automatic.get("origins")
    if not isinstance(directions, list) or not directions:
        return ()
    if not isinstance(origins, list) or not origins:
        return ()
    try:
        maximum_distance = float(automatic["maximum_distance_m"])
    except (KeyError, TypeError, ValueError):
        return ()
    declarations: list[dict[str, Any]] = []
    for origin_record in origins:
        if not isinstance(origin_record, Mapping):
            continue
        origin_index = origin_record.get("origin_index")
        origin = origin_record.get("origin_m")
        escaped = origin_record.get("escaped_direction_indices", [])
        if (
            not isinstance(origin_index, int)
            or not isinstance(origin, list)
            or len(origin) != 3
            or not isinstance(escaped, list)
        ):
            continue
        escaped_set = {int(index) for index in escaped}
        for direction_index, direction in enumerate(directions):
            if not isinstance(direction, list) or len(direction) != 3:
                continue
            hit = direction_index not in escaped_set
            declarations.append(
                {
                    "check_id": f"automatic_origin{origin_index}_direction{direction_index}",
                    "expectation": "hit_within_m" if hit else "clear_until_m",
                    "distance_m": maximum_distance,
                    "origin_m": origin,
                    "direction": direction,
                }
            )
    return tuple(declarations)


def _declared_declarations(
    room_manifest: Path | None,
) -> tuple[dict[str, Any], ...]:
    if room_manifest is None:
        return ()
    try:
        value = json.loads(room_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    declarations = value.get("ray_checks") if isinstance(value, Mapping) else None
    if not isinstance(declarations, list):
        return ()
    return tuple(dict(item) for item in declarations if isinstance(item, Mapping))


def _simulation_mapping(path: Path | None) -> dict[str, Any]:
    source = path or (
        REPOSITORY / "examples/runtime/rir_cache_simulation_request_v2.json"
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    simulation = dict(value["simulation"])
    simulation["channel_layout"] = {"type": "ambisonics", "channel_count": 4}
    return simulation


def verify_package_ray_leakage(
    *,
    package_manifest: Path,
    room_manifest: Path | None,
    runtime_prefix: str | Path | None,
    magnum_site: str | Path | None,
    rlr_sdk_root: str | Path | None,
    simulation_request: Path | None = None,
) -> dict[str, Any]:
    scene = load_compiled_acoustic_scene(
        package_manifest,
        allow_nonpassing_research_qa=True,
    )
    base = deepcopy(scene.qa_reports.get("ray_leakage", {}))
    declarations = _declared_declarations(room_manifest) + _automatic_declarations(base)
    if not declarations:
        base.update(
            {
                "status": "unavailable",
                "rlr_runtime_ray_check_status": "unavailable",
                "rlr_runtime_ray_check_count": 0,
                "rlr_runtime_unavailable_reason": (
                    "package has neither declared ray_checks nor automatic "
                    "interior probe directions"
                ),
            }
        )
        return base
    missing = _dependency_errors(runtime_prefix, magnum_site, rlr_sdk_root)
    if missing:
        base.update(
            {
                "status": "unavailable",
                "rlr_runtime_ray_check_status": "unavailable",
                "rlr_runtime_ray_check_count": len(declarations),
                "rlr_runtime_unavailable_reason": "; ".join(missing),
            }
        )
        return base

    try:
        simulation = RLRSimulationConfig.from_mapping(
            _simulation_mapping(simulation_request)
        )
        first_origin = tuple(float(value) for value in declarations[0]["origin_m"])
        last_origin = tuple(float(value) for value in declarations[-1]["origin_m"])
        result = simulate_compiled_acoustic_scene(
            scene,
            simulation,
            source=RuntimeAnchor(anchor_id="ray_source", position_m=first_origin),
            listener=RuntimeAnchor(anchor_id="ray_listener", position_m=last_origin),
            runtime_mode=RUNTIME_MODE_CURRENT_INSTALLED,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_site,
            ray_checks=declarations,
        )
    except RuntimeUnavailableError as error:
        base.update(
            {
                "status": "unavailable",
                "rlr_runtime_ray_check_status": "unavailable",
                "rlr_runtime_ray_check_count": len(declarations),
                "rlr_runtime_unavailable_reason": (
                    f"{type(error).__name__}: {error}"
                ),
            }
        )
        return base
    except (RuntimeExecutionError, RuntimeContractError, OSError, ValueError) as error:
        base.update(
            {
                "status": "error",
                "rlr_runtime_ray_check_status": "error",
                "rlr_runtime_ray_check_count": len(declarations),
                "rlr_runtime_unavailable_reason": None,
                "rlr_runtime_error": f"{type(error).__name__}: {error}",
            }
        )
        return base

    reports = list(result.ray_checks)
    status = (
        "pass"
        if reports and all(item.get("passed") is True for item in reports)
        else "fail"
    )
    base.update(
        {
            "status": status,
            "rlr_runtime_ray_check_status": status,
            "rlr_runtime_ray_check_count": len(reports),
            "rlr_runtime_unavailable_reason": None,
            "rlr_runtime_ray_checks": reports,
            "rlr_runtime_backend": "avengine_modern_rlr_context_trace_ray",
        }
    )
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--room-manifest", type=Path)
    parser.add_argument("--runtime-prefix")
    parser.add_argument("--magnum-site")
    parser.add_argument("--rlr-sdk-root")
    parser.add_argument("--simulation-request", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    report = verify_package_ray_leakage(
        package_manifest=args.package_manifest.resolve(),
        room_manifest=args.room_manifest.resolve() if args.room_manifest else None,
        runtime_prefix=args.runtime_prefix,
        magnum_site=args.magnum_site,
        rlr_sdk_root=args.rlr_sdk_root,
        simulation_request=(
            args.simulation_request.resolve()
            if args.simulation_request
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "rlr_runtime_ray_check_status": report.get(
                    "rlr_runtime_ray_check_status"
                ),
                "rlr_runtime_ray_check_count": report.get(
                    "rlr_runtime_ray_check_count"
                ),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    status = report.get("status")
    if status == "pass":
        return 0
    if status == "unavailable":
        return 3
    if status in {"fail", "error"}:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
