#!/usr/bin/env python3
"""Select one UE ground-height route domain from a multi-level route bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_routes(bank: dict, *, ground_z_ue_cm: float,
                  tolerance_ue_cm: float) -> tuple[list[dict], dict]:
    if bank.get("schema") != "avengine_apartment_route_bank_v1":
        raise ValueError("unsupported route-bank schema")
    if tolerance_ue_cm < 0:
        raise ValueError("ground tolerance must be non-negative")
    kept = []
    rejected_missing_z = 0
    rejected_other_ground = 0
    for route in bank.get("routes") or []:
        waypoints = route.get("waypoints_ue_cm") or []
        if not waypoints or any(len(point) < 3 for point in waypoints):
            rejected_missing_z += 1
            continue
        z_values = [float(point[2]) for point in waypoints]
        if max(abs(value - ground_z_ue_cm) for value in z_values) \
                > tolerance_ue_cm:
            rejected_other_ground += 1
            continue
        kept.append(route)
    return kept, {
        "input_routes": len(bank.get("routes") or []),
        "kept_routes": len(kept),
        "rejected_missing_z": rejected_missing_z,
        "rejected_other_ground": rejected_other_ground,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ground-z-ue-cm", required=True, type=float)
    parser.add_argument("--tolerance-ue-cm", required=True, type=float)
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    bank = json.loads(source.read_text())
    routes, counts = filter_routes(
        bank, ground_z_ue_cm=args.ground_z_ue_cm,
        tolerance_ue_cm=args.tolerance_ue_cm)
    if not routes:
        raise SystemExit("ground filter retained zero routes")
    result = dict(bank)
    result["routes"] = routes
    result["source"] = {
        **dict(bank.get("source") or {}),
        "parent_route_bank": str(source),
        "parent_route_bank_sha256": sha256_file(source),
        "route_domain_ground_z_ue_cm": args.ground_z_ue_cm,
        "route_domain_ground_tolerance_ue_cm": args.tolerance_ue_cm,
    }
    result["counts"] = {**dict(bank.get("counts") or {}), **counts}
    result["claim_boundary"] = (
        "research-only route-domain filter; ground height comes from runtime "
        "Floor actor evidence and does not establish pixel visibility or "
        "question admission")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({"output": str(output), **counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
