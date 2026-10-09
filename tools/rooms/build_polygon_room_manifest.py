#!/usr/bin/env python3
"""Write a new polygon placement manifest, optionally cloning registered room templates."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from avengine.rooms.room_polygon import RoomPolygon, room_polygon_record
from avengine.rooms.room_providers import V1_CATALOG_SCHEMA
from avengine.rooms.room_package import resolve_catalog_room_package_path, resolve_room_package_paths


def read_records(path):
    path = Path(path)
    if path.is_dir():
        return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))], {}
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()], {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data, {}
    if "rooms" in data:
        return data["rooms"], data
    if "records" in data:
        return data["records"], data
    return [data], {}


def identity(record):
    value = record.get("room_id") or record.get("id")
    if not value:
        raise ValueError("every original/split room needs room_id or id")
    return str(value)


def build_manifest(*, original_rooms=(), split_rooms, output_dir, template_map=None):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists; refusing overwrite: {output}")
    originals, header, catalog_source = [], {}, None
    for path in original_rooms:
        records, source_header = read_records(path)
        originals.extend(records)
        if source_header:
            header = source_header
            catalog_source = Path(path)
    by_id = {}
    for row in originals:
        key = identity(row)
        if key in by_id:
            raise ValueError(f"duplicate original room: {key}")
        by_id[key] = deepcopy(row)
    template_map = template_map or {}
    added = []
    sources = []
    for path in sorted(Path(split_rooms).glob("*.json")):
        split = json.loads(path.read_text())
        if split.get("decision") != "retain":
            continue
        key = identity(split)
        if key in by_id:
            # Explicitly preserve existing identity/resources, enriching only geometry.
            if room_polygon_record(by_id[key]) is None:
                by_id[key]["floor_polygon_xz_m"] = deepcopy(split["floor_polygon_xz_m"])
            continue
        template_id = template_map.get(key) or template_map.get(split.get("house"))
        if template_map and not template_id:
            raise ValueError(f"no explicit registered template for split room: {key}")
        if template_id:
            if template_id not in by_id:
                raise ValueError(f"unknown registered template: {template_id}")
            record = deepcopy(by_id[template_id])
            record["room_id"] = key
            if "room_package" in record:
                package = record["room_package"]
                if isinstance(package, str):
                    runtime = {"path_bindings": header.get("path_bindings") or {}}
                    source = resolve_catalog_room_package_path(package, catalog_path=catalog_source, runtime=runtime)
                    package = resolve_room_package_paths(json.loads(source.read_text()), runtime=runtime,
                                                         relative_roots=[source.parent])
                    record["room_package"] = package
                record["room_package"]["room_id"] = key
                legacy = record["room_package"].get("legacy_catalog_entry")
                if isinstance(legacy, dict):
                    legacy["room_id"] = key
        else:
            record = deepcopy(split)
            record.setdefault("room_id", key)
        record["floor_polygon_xz_m"] = deepcopy(split["floor_polygon_xz_m"])
        record["floor_y_m"] = split["floor_y_m"]
        polygon = RoomPolygon(record["floor_polygon_xz_m"])
        x0, z0, x1, z1 = polygon.geometry.bounds
        floor = float(split["floor_y_m"])
        record["planning_region_m"] = [[x0, floor - .3, z0], [x1, floor + .3, z1]]
        record.setdefault("room_polygon_placement", {"boundary_margin_m": .25,
            "path_sample_step_m": .25, "path_tolerance_m": .1,
            "max_point_attempts": 512, "max_route_attempts": 32})
        record["polygon_split_source"] = str(path.resolve())
        if template_id:
            record["polygon_template_room_id"] = template_id
        added.append(record)
        sources.append(str(path.resolve()))
    rows = list(by_id.values()) + added
    if not rows:
        raise ValueError("no rooms to write")
    ids = [identity(row) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate room identities in new manifest")
    # Validate all opt-in geometry before creating the destination.
    for row in rows:
        RoomPolygon.from_room(row)
    output.mkdir(parents=True, exist_ok=False)
    (output / "rooms").mkdir()
    for index, row in enumerate(rows):
        with (output / "rooms" / f"room_{index:05d}.json").open("x") as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    registered = bool(template_map and header.get("schema") == V1_CATALOG_SCHEMA)
    manifest = {k: deepcopy(v) for k, v in header.items() if k != "rooms"} if registered else {}
    manifest.update(schema=(V1_CATALOG_SCHEMA if registered else "avengine_polygon_room_manifest_v1"),
                    rooms=rows)
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {"original_rooms": len(by_id), "added_retained_rooms": len(added),
               "rooms_with_polygon": sum(room_polygon_record(row) is not None for row in rows),
               "registered_catalog_templates_supplied": registered,
               "runtime_and_acoustic_resources_reused_without_clipping": bool(template_map),
               "split_sources": sources, "existing_inputs_modified": False,
               "boundary_definition": "union of every Polygon exterior; interior holes filled"}
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-rooms", type=Path, nargs="*", default=[])
    parser.add_argument("--split-rooms", type=Path, required=True)
    parser.add_argument("--template-map", type=Path,
                        help="explicit {split house or room ID: existing catalog room ID}")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    templates = json.loads(args.template_map.read_text()) if args.template_map else None
    print(json.dumps(build_manifest(original_rooms=args.original_rooms,
          split_rooms=args.split_rooms, output_dir=args.output_dir, template_map=templates),
          ensure_ascii=False))


if __name__ == "__main__":
    main()
