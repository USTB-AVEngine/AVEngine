#!/usr/bin/env python3
"""Measure one GLB mesh with bounded memory and optional explicit policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from avengine.assets.mesh_quality import (  # noqa: E402
    MeshQualityError,
    load_quality_policy,
    measure_glb,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input triangle GLB")
    parser.add_argument("--output", type=Path, help="fresh JSON measurement report")
    parser.add_argument(
        "--quality-policy",
        type=Path,
        help="optional policy JSON; without it the result is measured_unclassified",
    )
    parser.add_argument(
        "--support-plane-manifest",
        type=Path,
        help="explicit support-plane metadata file to record as present",
    )
    parser.add_argument(
        "--tiny-area-threshold",
        type=float,
        default=None,
        help="descriptive tiny-face area threshold; policy measurement may set it",
    )
    parser.add_argument(
        "--small-component-max-faces",
        type=int,
        default=None,
        help="descriptive cutoff for small-component metrics; policy may set it",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="maximum faces processed by one geometry temporary buffer",
    )
    return parser.parse_args(argv)


def _load_support_plane(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise MeshQualityError(f"support-plane manifest is missing or unsafe: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeshQualityError(f"invalid support-plane manifest {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise MeshQualityError("support-plane manifest must be a JSON object")
    return resolved


def _peak_rss_kib() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; the CLI is executed on the AVEngine server Linux host.
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise MeshQualityError(f"refusing to overwrite report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy = load_quality_policy(args.quality_policy) if args.quality_policy else None
    measurement = policy.get("measurement", {}) if policy else {}
    tiny_area_threshold = (
        args.tiny_area_threshold
        if args.tiny_area_threshold is not None
        else measurement.get("tiny_face_area_threshold", 1.0e-12)
    )
    small_component_max_faces = (
        args.small_component_max_faces
        if args.small_component_max_faces is not None
        else measurement.get("small_component_max_faces", 10)
    )
    support_plane = _load_support_plane(args.support_plane_manifest)
    report = measure_glb(
        args.input,
        tiny_area_threshold=tiny_area_threshold,
        small_component_max_faces=small_component_max_faces,
        chunk_size=args.chunk_size,
        support_plane_path=support_plane,
        quality_policy=policy,
    )
    report["runtime"] = {
        "python": sys.executable,
        "peak_rss_kib": _peak_rss_kib(),
        "chunk_size": args.chunk_size,
        "mutation": {"input_modified": False, "components_deleted": False},
    }
    if args.output:
        _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MeshQualityError, OSError, ValueError) as error:
        print(f"mesh quality inspection refused: {error}", file=sys.stderr)
        raise SystemExit(2)
