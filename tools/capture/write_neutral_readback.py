#!/usr/bin/env python3
"""Convert retained native UE or Habitat capture into a fresh neutral readback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.capture.neutral_readback import validate_neutral_readback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("ue_spear", "habitat"), required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.renderer == "ue_spear":
        from avengine.capture.ue_neutral_readback import write_ue_neutral_readback as write
    else:
        from avengine.capture.habitat_neutral_readback import write_habitat_neutral_readback as write
    data = write(args.capture, plan, args.output)
    print(json.dumps({"output": str(args.output), **validate_neutral_readback(data, plan=plan)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
