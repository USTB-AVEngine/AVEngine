#!/usr/bin/env python3
"""Prepare independent QA batch requests, collect outcomes, group splits, or dry-run a scattered scale-up manifest."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.batch_manifest import (
    collect_batch_outcomes,
    format_class_pair_condition_group_crosstab,
    grouped_splits,
    prepare_batch_manifest,
    prepare_scaleup_dry_run,
)
from avengine.qa.batch_sound_pool import build_batch_sound_pool
from avengine.rooms.conditioned_sampler import load_conditioned_sound_pool
from avengine.runtime_profiles import load_source_asset_runtime_registry


def read(path):
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _existing(path, fallback):
    path = Path(path).expanduser()
    if path.exists():
        return path
    fallback = Path(fallback)
    if fallback.exists():
        return fallback
    return path


def _load_sounds(config, output):
    base = config["base_request"]
    registry = load_source_asset_runtime_registry(
        _existing(base["source_registry"], REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"))
    catalog_path = _existing(base["room_catalog"], REPOSITORY / "examples/rooms/packages/catalog.json")
    catalog = read(catalog_path)
    if config.get("sound_sources") is not None:
        pool = build_batch_sound_pool(config["sound_sources"], registry)
        pool_path = output / "batch_sounds.json"
        base["sound_pool"] = str(pool_path)
        base.setdefault("sound_selection", {}).pop("prepared_set", None)
        sounds = pool["sounds"]
    else:
        pool = None
        pool_path = base.get("sound_selection", {}).get("prepared_set") or base["sound_pool"]
        sounds = load_conditioned_sound_pool(read(pool_path), source_path=pool_path)
    return registry, catalog, sounds, pool, pool_path


def _annotate_prepare(result, output, config_path):
    for row in result["episodes"]:
        name = row["episode_id"] + ".json"
        if Path(name).name != name:
            raise ValueError("episode_id cannot contain path separators")
        request_path = output / "requests" / name
        write(request_path, row["request"])
        row["request_path"] = str(request_path)
        row["controller_entrypoint"] = str(REPOSITORY / "tools/studio/run_qa_episode.py")
    result["source_config_path"] = str(Path(config_path).resolve())
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPOSITORY, text=True).splitlines()
    result["producer"] = {"repository": str(REPOSITORY),
                          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                cwd=REPOSITORY, text=True).strip(),
                          "working_tree_changes": dirty,
                          "code_state": "working_tree" if dirty else "committed",
                          "python": sys.executable}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    outcomes = commands.add_parser("outcomes")
    outcomes.add_argument("--manifest", type=Path, required=True)
    outcomes.add_argument("--records", type=Path, required=True)
    outcomes.add_argument("--output", type=Path, required=True)
    split = commands.add_parser("split")
    split.add_argument("--records", type=Path, required=True)
    split.add_argument("--ratios", default="{\"train\":0.8,\"eval\":0.2}")
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--output", type=Path, required=True)
    scaleup = commands.add_parser("scaleup-dry-run",
                                  help="Scatter 7-room scale-up slots and preallocate without GPU execution")
    scaleup.add_argument("--config", type=Path, required=True)
    scaleup.add_argument("--output", type=Path, required=True)
    scaleup.add_argument("--seed", type=int, default=20260907)
    scaleup.add_argument("--episodes-per-room", type=int, default=50)
    scaleup.add_argument("--batch-id", default=None)
    scaleup.add_argument("--catalog", type=Path, default=None)
    scaleup.add_argument("--registry", type=Path, default=None)
    scaleup.add_argument("--sounds", type=Path, default=None,
                         help="Reuse an existing joined sound pool instead of rebuilding P7/event PCM")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if args.command == "prepare":
        config = read(args.config)
        registry, catalog, sounds, pool, pool_path = _load_sounds(config, output)
        result = prepare_batch_manifest(config, registry, catalog, sounds)
        if pool is not None:
            write(Path(pool_path), pool)
        result = _annotate_prepare(result, output, args.config)
        write(output / "batch_manifest.json", result)
    elif args.command == "outcomes":
        result = collect_batch_outcomes(read(args.manifest), read(args.records))
        write(output / "batch_outcomes.json", result)
    elif args.command == "scaleup-dry-run":
        config = read(args.config)
        base = config.setdefault("base_request", {})
        if args.catalog is not None:
            base["room_catalog"] = str(args.catalog)
        if args.registry is not None:
            base["source_registry"] = str(args.registry)
        base["room_catalog"] = str(_existing(
            base.get("room_catalog", REPOSITORY / "examples/rooms/packages/catalog.json"),
            REPOSITORY / "examples/rooms/packages/catalog.json"))
        base["source_registry"] = str(_existing(
            base.get("source_registry", REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"),
            REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"))
        registry = load_source_asset_runtime_registry(base["source_registry"])
        catalog = read(base["room_catalog"])
        if args.sounds is not None:
            payload = read(args.sounds)
            sounds = payload.get("sounds", payload) if isinstance(payload, dict) else payload
            pool = None
        else:
            registry, catalog, sounds, pool, pool_path = _load_sounds(config, output)
        packed = prepare_scaleup_dry_run(
            config, registry, catalog, sounds, seed=args.seed,
            episodes_per_room=args.episodes_per_room, batch_id=args.batch_id)
        result = packed["manifest"]
        result = _annotate_prepare(result, output, args.config)
        if pool is not None:
            write(output / "batch_sounds.json", pool)
        write(output / "scaleup_config.json", packed["config"])
        write(output / "batch_manifest.json", result)
        off_screen = sum(
            1 for row in result["episodes"]
            if (row.get("requested_profile") or {}).get("anchor_visibility") == "off_screen"
            or (row.get("requested_profile") or {}).get("competitor_visibility") == "off_screen")
        summary = {
            "batch_id": result["batch_id"],
            "requested_episode_count": result["requested_episode_count"],
            "repeat_deficit_count": packed["repeat_deficit_count"],
            "preallocation_gap_counts": result.get("preallocation_gap_counts", {}),
            "class_pair_condition_group_crosstab": packed["class_pair_condition_group_crosstab"],
            "off_screen_portrait_count": off_screen,
            "gpu_execution": False,
            "crosstab_text": format_class_pair_condition_group_crosstab(
                packed["class_pair_condition_group_crosstab"]),
        }
        write(output / "scaleup_dry_run_summary.json", summary)
        print(summary["crosstab_text"])
        print(json.dumps({
            "repeat_deficit_count": packed["repeat_deficit_count"],
            "off_screen_portrait_count": off_screen,
            "meets_crosstab_acceptance": packed["class_pair_condition_group_crosstab"].get("meets_acceptance"),
            "requested_episode_count": result["requested_episode_count"],
        }, ensure_ascii=False))
        result = summary
    else:
        result = grouped_splits(read(args.records), ratios=json.loads(args.ratios), seed=args.seed)
        write(output / "grouped_splits.json", result)
    print(json.dumps({"output": str(output), "command": args.command,
                      "records": result.get("requested_episode_count", result.get("episode_denominator",
                                  result.get("record_denominator", result.get("repeat_deficit_count"))))},
                     ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
