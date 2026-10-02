#!/usr/bin/env python3
"""Convert an external review-item index into a safe relative-path manifest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "tmp/room_screening/review_manifest.json"
PATH_FIELDS = {"image_path": "image_file", "geometry_path": "geometry_file", "video_path": "video_file"}


def asset_reference(value: str, asset_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = asset_root / path
    resolved = path.resolve()
    try:
        return resolved.relative_to(asset_root).as_posix()
    except ValueError as error:
        raise ValueError(f"asset path is outside --asset-root: {value}") from error


def build_manifest(source: dict, asset_root: Path) -> dict:
    items = source.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("input must contain a non-empty items array")
    converted = []
    seen = set()
    for index, source_item in enumerate(items):
        if not isinstance(source_item, dict):
            raise ValueError(f"items[{index}] must be an object")
        item = {key: value for key, value in source_item.items() if key not in PATH_FIELDS}
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"items[{index}].id must be a non-empty string")
        if item_id in seen:
            raise ValueError(f"duplicate review item id: {item_id}")
        seen.add(item_id)
        for source_key, output_key in PATH_FIELDS.items():
            value = source_item.get(source_key)
            if source_key == "image_path" and not value:
                raise ValueError(f"items[{index}].image_path is required")
            if value is None:
                continue
            relative = asset_reference(str(value), asset_root)
            if not (asset_root / relative).is_file():
                raise FileNotFoundError(f"items[{index}].{source_key} not found: {relative}")
            item[output_key] = relative
        converted.append(item)
    return {
        "schema_version": "room_screening_review_manifest_v1",
        **{key: source[key] for key in ("title", "purpose", "source_note", "choices") if key in source},
        "items": converted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON object with items using image_path and optional geometry_path/video_path")
    parser.add_argument("--asset-root", type=Path, required=True,
                        help="directory containing all referenced external review assets")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="manifest output; default is under repository tmp/")
    args = parser.parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    if not asset_root.is_dir():
        parser.error(f"asset root not found: {asset_root}")
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        result = build_manifest(source, asset_root)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Wrote {len(result['items'])} review items to {args.output}")


if __name__ == "__main__":
    main()
