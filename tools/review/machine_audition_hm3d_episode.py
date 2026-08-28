#!/usr/bin/env python3
"""Machine-audit one rendered HM3D episode and write the verdict beside it.

Reads the episode's own receipt, judges the artifacts it names (direction
records, cardinal probes, the shipped wavs, the shipped frames, the mux,
and the pose identity through all of them), and writes
machine_audition.json with the verdict and its reasons. Exit code follows
the verdict, so a chain that runs this last fails when the deliverable is
bad even though every render step exited zero.

The reasons are the review surface: pages show them so nobody has to put
on headphones to know whether an episode is sound. A human verdict remains
possible as an override, never as a requirement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.review.episode_audition import write_audition  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-dir",
        required=True,
        type=Path,
        help="directory holding the episode's receipt.json",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-audit even if machine_audition.json already exists",
    )
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    if not (episode_dir / "receipt.json").is_file():
        raise SystemExit(f"no receipt.json in {episode_dir}")
    document = write_audition(episode_dir, refresh=args.refresh)
    print(json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True))
    return 0 if document["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
