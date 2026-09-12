#!/usr/bin/env python3
"""Export a self-contained delivery, build its dataset index, attach a layout.

Two delivery routes are supported:

* ``--core-bundle`` with ``--catalog-index`` delivers accepted four-member core
  groups together with their full catalog.
* ``--catalog-index`` alone delivers an ordinary Episode catalog. An accepted
  core group is not a precondition for having the QA-01..25 catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.binding_delivery import (  # noqa: E402
    BindingDeliveryError,
    attach_audio_layout,
    build_dataset_index,
    export_binding_delivery,
    write_static_index,
)


def _read(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-bundle",
        "--core",
        type=Path,
        default=None,
        help="completed core binding_groups.json; omit for an Episode-only delivery",
    )
    parser.add_argument(
        "--catalog-index",
        "--catalog",
        required=True,
        type=Path,
        help="catalog_index.json from derive_binding_catalog or derive_episode_catalog",
    )
    parser.add_argument(
        "--output",
        "--out",
        required=True,
        type=Path,
        help="fresh portable delivery directory",
    )
    parser.add_argument(
        "--skip-media-check",
        action="store_true",
        help="skip the ffprobe media readback during validation",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="write public/dataset_index.json and private/gold_index.json",
    )
    parser.add_argument(
        "--index-config",
        type=Path,
        default=None,
        help="JSON overrides for the dataset index configuration (layouts, splits, ...)",
    )
    parser.add_argument(
        "--attach-layout",
        action="append",
        default=[],
        metavar="SAMPLE_ID:LAYOUT:RECEIPT[:MIXTURE]",
        help="attach one extra audio layout view, verified against that member's own render",
    )
    parser.add_argument(
        "--static-index",
        action="store_true",
        help="rewrite the small static HTML index for an existing delivery",
    )
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        result: dict = {}
        if not args.static_index or not output.exists():
            if output.exists() or output.is_symlink():
                print(f"refusing to overwrite delivery output: {output}", file=sys.stderr)
                return 2
            result = export_binding_delivery(
                None if args.core_bundle is None else args.core_bundle.expanduser().resolve(),
                args.catalog_index.expanduser().resolve(),
                output,
                check_media=not args.skip_media_check,
            )
        attached = []
        for spec in args.attach_layout:
            parts = spec.split(":")
            if len(parts) not in (3, 4):
                print(
                    f"--attach-layout needs SAMPLE_ID:LAYOUT:RECEIPT[:MIXTURE], got {spec!r}",
                    file=sys.stderr,
                )
                return 2
            attached.append(
                attach_audio_layout(
                    output,
                    sample_id=parts[0],
                    layout=parts[1],
                    receipt=Path(parts[2]),
                    mixture=Path(parts[3]) if len(parts) == 4 else None,
                )
            )
        index = None
        if args.build_index or attached:
            index = build_dataset_index(
                output,
                config=_read(args.index_config) if args.index_config else None,
            )
        elif args.static_index:
            write_static_index(output)
    except (BindingDeliveryError, FileExistsError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result.get("status", "index_only"),
                "output": str(output),
                "delivery_route": "episode_catalog" if args.core_bundle is None else "core_group",
                "joined_members": result.get("joined_members"),
                "copied_files": result.get("copied_files"),
                "external_dependencies": result.get("external_dependencies"),
                "validation_status": (
                    result["validation"].get("status")
                    if isinstance(result.get("validation"), dict)
                    else None
                ),
                "attached_layouts": [
                    {"sample_id": row["sample_id"], "layout": row["layout"]} for row in attached
                ],
                "dataset_index": index,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
