#!/usr/bin/env python3
"""Author one current residential visual-only research episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.optional_backends.residential_episode import (  # noqa: E402
    FRAME_COUNT,
    FPS,
    TICKS_PER_FRAME,
    build_residential_source_episode,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _visual_episode_projection(episode: dict[str, Any]) -> dict[str, Any]:
    """Keep only the authorities consumed by the current UE visual replay."""

    visual_plan = episode["visual_plan"]
    return {
        "status": "research_only",
        "scene": episode["scene"],
        "review_lights": episode["review_lights"],
        "visual_plan": {
            "backend_role": visual_plan["backend_role"],
            "camera": visual_plan["camera"],
            "actors": visual_plan["actors"],
            "frames": visual_plan["frames"],
        },
    }


def author(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)

    episode = build_residential_source_episode(
        scene_metadata=_load(args.scene_metadata), profile=_load(args.profile)
    )
    visual_episode = _visual_episode_projection(episode)
    records = {
        "episode_plan": ("episode_plan.json", visual_episode),
        "visual_plan": ("visual_plan.json", visual_episode["visual_plan"]),
    }
    artifacts: dict[str, str] = {}
    for role, (name, value) in records.items():
        path = output / name
        _write(path, value)
        artifacts[role] = str(path)

    receipt = {
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification": False,
        "qualification_claim": False,
        "clock": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "ticks_per_frame": TICKS_PER_FRAME,
        },
        "audio": {"status": "not_requested"},
        "rlr": {"status": "not_requested"},
        "artifacts": artifacts,
    }
    _write(output / "research_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result = author(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
