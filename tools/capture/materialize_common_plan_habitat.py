#!/usr/bin/env python3
"""Materialize one common renderer-neutral plan into a Habitat case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.assets.mp3d_region_actor_tracks import (
    MP3DRegionActorTrackError,
    materialize_common_plan_habitat,
)


def _read_plan(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise MP3DRegionActorTrackError(f"common plan must be a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MP3DRegionActorTrackError(f"cannot read common plan: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MP3DRegionActorTrackError("common plan must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--runtime-registry", required=True, type=Path)
    parser.add_argument("--habitat-binding-delta", type=Path)
    parser.add_argument("--base-m1-request", type=Path)
    parser.add_argument(
        "--allow-research-candidate",
        action="store_true",
        help="Allow research_candidate P12 packages for non-counted native evidence only",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = materialize_common_plan_habitat(
            plan=_read_plan(args.plan),
            room_manifest=args.room_manifest,
            runtime_registry=args.runtime_registry,
            output=args.output,
            habitat_binding_delta=args.habitat_binding_delta,
            base_m1_request=args.base_m1_request,
            allow_research_candidate=args.allow_research_candidate,
        )
    except (MP3DRegionActorTrackError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(args.output.expanduser().resolve()),
                "frame_count": receipt["clock"]["frame_count"],
                "actor_count": len(receipt["actors"]),
                "capture_input_validation": receipt["checks"][
                    "capture_input_validation"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
