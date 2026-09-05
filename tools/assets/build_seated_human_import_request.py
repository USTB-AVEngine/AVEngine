"""Create a no-clobber SPEAR/UE seated-human skeletal import request."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.assets.seated_humans import build_ue_import_request, load_seated_human_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--content-root", default="/Game/AVEngine/SeatedHumans")
    args = parser.parse_args()
    specs = load_seated_human_batch(args.spec, require_sources=False)
    request = build_ue_import_request(
        specs, output_root=args.output_root, content_root=args.content_root
    )
    destination = args.request.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"refusing to replace import request: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(request, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({"status": "pass", "request": str(destination), "assets": len(specs)}, indent=2))


if __name__ == "__main__":
    main()
