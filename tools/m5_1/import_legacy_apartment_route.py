#!/usr/bin/env python3
"""Import the legacy 18 s apartment route into the M5.1 route manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from avengine.contracts.json_io import load_json, sha256_file, write_json
from avengine.m5_1.legacy_route import assert_valid_route_manifest, build_route_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_RELATIVE = Path(
    "external/SPEAR/tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/"
    "specs/rocketbox_adults_male_adult_01_original_ue_v1/"
    "camera_pass_table_loop_walking.json"
)
DEFAULT_FURNITURE_RELATIVE = Path("external/SPEAR/data/apartment_furniture_map.json")
DEFAULT_SHELL_RELATIVE = Path("external/SPEAR/data/apartment_shell_map.json")
DEFAULT_CATEGORIES_RELATIVE = Path(
    "external/SPEAR/tools/spike_rlr/apartment_furniture_categories.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "examples/m5_1/legacy_apartment/route_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate the authoritative legacy 270-frame apartment route and "
            "recompute its zero-radius center-point AABB gates."
        )
    )
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--legacy-spec", type=Path)
    parser.add_argument("--furniture-map", type=Path)
    parser.add_argument("--shell-map", type=Path)
    parser.add_argument("--furniture-categories", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(explicit: Path | None, root: Path, relative: Path) -> Path:
    return (explicit if explicit is not None else root / relative).resolve()


def _record(path: Path, *, legacy_root: Path) -> dict[str, Any]:
    try:
        label = path.relative_to(legacy_root).as_posix()
    except ValueError:
        label = str(path)
    return {
        "path": label,
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    legacy_root = args.legacy_root.resolve()
    spec_path = _resolve(args.legacy_spec, legacy_root, DEFAULT_SPEC_RELATIVE)
    furniture_path = _resolve(
        args.furniture_map, legacy_root, DEFAULT_FURNITURE_RELATIVE
    )
    shell_path = _resolve(args.shell_map, legacy_root, DEFAULT_SHELL_RELATIVE)
    categories_path = _resolve(
        args.furniture_categories, legacy_root, DEFAULT_CATEGORIES_RELATIVE
    )
    inputs = {
        "legacy_spec": spec_path,
        "furniture_map": furniture_path,
        "shell_map": shell_path,
        "furniture_categories": categories_path,
    }
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing M5.1 legacy input(s): " + "; ".join(missing))

    manifest = build_route_manifest(
        load_json(spec_path),
        load_json(furniture_path),
        load_json(shell_path),
        load_json(categories_path),
        source_records={
            name: _record(path, legacy_root=legacy_root)
            for name, path in inputs.items()
        },
    )
    assert_valid_route_manifest(manifest)
    destination = args.output.resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite route manifest: {destination}")
    write_json(destination, manifest)
    readback = load_json(destination)
    assert_valid_route_manifest(readback)
    print(
        "M5_1_LEGACY_ROUTE_OK "
        f"status={readback['status']} frames={readback['timebase']['frame_count']} "
        f"human_clearance_m={readback['gates']['human_center_point_aabb']['minimum_clearance_m']:.9f} "
        f"dog_clearance_m={readback['gates']['dog_center_point_aabb']['minimum_clearance_m']:.9f} "
        f"output={destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
