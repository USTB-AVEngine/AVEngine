#!/usr/bin/env python3
"""Verify M3 canary schema, lineage, raw IRs and recomputed gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from avengine.m3.canary import load_and_verify_canary_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    result = load_and_verify_canary_evidence(args.evidence)
    if result.errors:
        for error in result.errors:
            print(error)
        return 1
    print("pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
