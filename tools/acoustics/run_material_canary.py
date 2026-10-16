#!/usr/bin/env python3
"""Run the hash-bound repeated M3 RLR material activation canary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from avengine.acoustics.canary import (
    load_and_verify_canary_evidence,
    run_material_activation_canary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--compile-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    evidence_path = run_material_activation_canary(
        args.request,
        args.compile_evidence,
        args.output,
    )
    print(evidence_path)
    result = load_and_verify_canary_evidence(evidence_path)
    status = result.evidence.get("overall_status")
    if not result.errors and status == "pass":
        return 0
    for error in result.errors:
        print(error, file=sys.stderr)
    print(f"material canary overall_status={status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
