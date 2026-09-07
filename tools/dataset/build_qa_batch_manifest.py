#!/usr/bin/env python3
"""Prepare independent QA batch requests, collect outcomes, or group splits."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.batch_manifest import collect_batch_outcomes, grouped_splits, prepare_batch_manifest
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
    split.add_argument("--ratios", default='{"train":0.8,"eval":0.2}')
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if args.command == "prepare":
        config = read(args.config)
        base = config["base_request"]
        registry = load_source_asset_runtime_registry(base["source_registry"])
        catalog = read(base["room_catalog"])
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
        result = prepare_batch_manifest(config, registry, catalog, sounds)
        if pool is not None:
            write(Path(pool_path), pool)
        # Save ordinary controller requests, not a second execution system.
        for row in result["episodes"]:
            name = row["episode_id"] + ".json"
            if Path(name).name != name:
                raise ValueError("episode_id cannot contain path separators")
            request_path = output / "requests" / name
            write(request_path, row["request"])
            row["request_path"] = str(request_path)
            row["controller_entrypoint"] = str(REPOSITORY / "tools/studio/run_qa_episode.py")
        result["source_config_path"] = str(args.config.resolve())
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPOSITORY, text=True).splitlines()
        result["producer"] = {"repository": str(REPOSITORY),
                              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                    cwd=REPOSITORY, text=True).strip(),
                              "working_tree_changes": dirty,
                              "code_state": "working_tree" if dirty else "committed",
                              "python": sys.executable}
        write(output / "batch_manifest.json", result)
    elif args.command == "outcomes":
        result = collect_batch_outcomes(read(args.manifest), read(args.records))
        write(output / "batch_outcomes.json", result)
    else:
        result = grouped_splits(read(args.records), ratios=json.loads(args.ratios), seed=args.seed)
        write(output / "grouped_splits.json", result)
    print(json.dumps({"output": str(output), "command": args.command,
                      "records": result.get("requested_episode_count", result.get("episode_denominator",
                                  result.get("record_denominator")))}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
