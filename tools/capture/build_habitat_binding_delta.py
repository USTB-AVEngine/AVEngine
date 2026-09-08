#!/usr/bin/env python3
"""Build a reviewable Habitat binding delta from the two source registries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.assets.habitat_static_assets import (
    HabitatStaticAssetError,
    make_binding_delta,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-registry", required=True, type=Path)
    parser.add_argument("--external-index", required=True, type=Path)
    parser.add_argument("--beagle-asset-manifest", type=Path)
    parser.add_argument("--beagle-m2-request", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        delta = make_binding_delta(
            runtime_registry_path=args.runtime_registry,
            external_index_path=args.external_index,
            beagle_asset_manifest_path=args.beagle_asset_manifest,
            beagle_m2_request_path=args.beagle_m2_request,
        )
        output = args.output.expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise HabitatStaticAssetError(f"refusing to replace binding delta: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(delta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (HabitatStaticAssetError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "status": "pass",
        "output": str(output),
        "binding_count": len(delta["bindings"]),
        "excluded_count": len(delta["excluded"]),
        "inventory": delta["inventory"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
