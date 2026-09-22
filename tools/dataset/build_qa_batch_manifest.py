#!/usr/bin/env python3
"""Prepare independent QA batch requests, collect outcomes, group splits, or dry-run a scattered scale-up manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

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

PRODUCTION_ROOM_CATALOG = REPOSITORY / "examples/rooms/packages/catalog.json"


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


def _resolve_declared_file(
    configured: Any,
    fallback: Path | None = None,
    *,
    label: str,
) -> Path:
    candidate = fallback if configured is None or str(configured).strip() == "" else Path(str(configured)).expanduser()
    if candidate is None:
        raise FileNotFoundError(f"{label} is required")
    if not candidate.is_absolute():
        candidate = REPOSITORY / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def _input_metadata(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _resolve_declared_file(path, label=label)
    record = {"path": str(resolved), "size_bytes": resolved.stat().st_size}
    try:
        relative = resolved.relative_to(REPOSITORY.resolve())
    except ValueError:
        return record
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=REPOSITORY, capture_output=True, text=True,
    )
    if tracked.returncode == 0:
        record["git"] = {
            "repository": str(REPOSITORY),
            "path": str(relative),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
            ).strip(),
            "working_tree_changes": subprocess.check_output(
                ["git", "status", "--porcelain", "--", str(relative)],
                cwd=REPOSITORY, text=True,
            ).splitlines(),
        }
    return record


def _effective_runtime_path_inputs(rows: list[Mapping[str, Any]], bindings: Mapping[str, Any]) -> dict[str, Any]:
    runtimes = [
        row.get("request", {}).get("runtime", {})
        for row in rows
        if isinstance(row.get("request"), Mapping)
    ]
    normalized = [json.dumps(value, ensure_ascii=False, sort_keys=True) for value in runtimes]
    runtime = runtimes[0] if normalized and all(value == normalized[0] for value in normalized) else {
        row.get("episode_id"): row.get("request", {}).get("runtime", {})
        for row in rows
    }
    return {
        "path_bindings": {str(key): str(value) for key, value in dict(bindings).items()},
        "runtime": runtime,
    }


def production_room_catalog_path() -> Path:
    return PRODUCTION_ROOM_CATALOG.resolve()


def resolve_request_room_catalog(configured: Any = None, *, explicit: Path | None = None) -> Path:
    """Resolve the declared catalog against the repository or an explicit input."""
    production = production_room_catalog_path()
    if explicit is not None:
        return _resolve_declared_file(explicit, label="room catalog")
    if configured is None or str(configured).strip() == "":
        return _resolve_declared_file(production, label="room catalog")
    return _resolve_declared_file(configured, label="room catalog")


def catalog_path_bindings(catalog: Any) -> dict[str, str]:
    if not isinstance(catalog, Mapping):
        return {}
    raw = catalog.get("path_bindings") or {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def stamp_request_catalog(
    request: dict[str, Any],
    *,
    catalog_path: Path,
    path_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an absolute catalog path and full path_bindings onto a request."""
    request["room_catalog"] = str(Path(catalog_path).resolve())
    runtime = request.get("runtime")
    runtime = dict(runtime) if isinstance(runtime, dict) else {}
    existing = runtime.get("path_bindings")
    existing = dict(existing) if isinstance(existing, dict) else {}
    runtime["path_bindings"] = {
        **{str(key): str(value) for key, value in dict(path_bindings).items()},
        **existing,
    }
    request["runtime"] = runtime
    return request


def _load_sounds(config, output):
    base = config["base_request"]
    registry_path = _resolve_declared_file(
        base.get("source_registry"),
        REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
        label="source registry",
    )
    base["source_registry"] = str(registry_path)
    registry = load_source_asset_runtime_registry(registry_path)
    catalog_path = resolve_request_room_catalog(base.get("room_catalog"))
    catalog = read(catalog_path)
    stamp_request_catalog(
        base, catalog_path=catalog_path, path_bindings=catalog_path_bindings(catalog)
    )
    if config.get("sound_sources") is not None:
        pool = build_batch_sound_pool(config["sound_sources"], registry)
        pool_path = (Path(output) / "batch_sounds.json").resolve()
        base["sound_pool"] = str(pool_path)
        base.setdefault("sound_selection", {}).pop("prepared_set", None)
        sounds = pool["sounds"]
    else:
        pool = None
        selection = base.get("sound_selection") if isinstance(base.get("sound_selection"), Mapping) else {}
        configured_pool = selection.get("prepared_set") or base.get("sound_pool")
        pool_path = _resolve_declared_file(configured_pool, label="sound pool")
        sounds = load_conditioned_sound_pool(read(pool_path), source_path=pool_path)
        base["sound_pool"] = str(pool_path)
        base.setdefault("sound_selection", {}).pop("prepared_set", None)
    return registry, catalog, sounds, pool, pool_path, catalog_path, registry_path


def _annotate_prepare(
    result,
    output,
    config_path,
    *,
    catalog_path,
    path_bindings,
    source_registry_path,
    sound_pool_path,
):
    catalog_path = Path(catalog_path).expanduser().resolve()
    source_registry_path = Path(source_registry_path).expanduser().resolve()
    sound_pool_path = Path(sound_pool_path).expanduser().resolve()
    bindings = {str(key): str(value) for key, value in dict(path_bindings).items()}
    for row in result["episodes"]:
        name = row["episode_id"] + ".json"
        if Path(name).name != name:
            raise ValueError("episode_id cannot contain path separators")
        stamp_request_catalog(row["request"], catalog_path=catalog_path, path_bindings=bindings)
        row["request"]["source_registry"] = str(source_registry_path)
        row["request"]["sound_pool"] = str(sound_pool_path)
        request_path = output / "requests" / name
        write(request_path, row["request"])
        row["request_path"] = str(request_path)
        row["controller_entrypoint"] = str(REPOSITORY / "tools/studio/run_qa_episode.py")
    result["source_config_path"] = str(Path(config_path).expanduser().resolve())
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY, text=True
    ).splitlines()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPOSITORY, text=True
    ).strip()
    inputs = {
        "room_catalog": _input_metadata(catalog_path, label="room catalog"),
        "source_registry": _input_metadata(source_registry_path, label="source registry"),
        "sound_pool": _input_metadata(sound_pool_path, label="sound pool"),
    }
    result["producer"] = {
        "repository": str(REPOSITORY),
        "git_commit": commit,
        "git_branch": branch,
        "working_tree_changes": dirty,
        "code_state": "working_tree" if dirty else "committed",
        "python": sys.executable,
        "room_catalog": str(catalog_path),
        "source_registry": str(source_registry_path),
        "sound_pool": str(sound_pool_path),
        "inputs": inputs,
        "effective_runtime_path_inputs": _effective_runtime_path_inputs(
            result["episodes"], bindings
        ),
    }
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
    scaleup.add_argument(
        "--off-screen-fraction", type=float, default=None,
        help="share of each room's episodes whose question subject the camera never shows; "
             "omitted keeps the two slots per room a scale-up has always marked")
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
        registry, catalog, sounds, pool, pool_path, catalog_path, registry_path = _load_sounds(config, output)
        result = prepare_batch_manifest(config, registry, catalog, sounds)
        if pool is not None:
            write(Path(pool_path), pool)
        result = _annotate_prepare(
            result, output, args.config, catalog_path=catalog_path,
            path_bindings=catalog_path_bindings(catalog),
            source_registry_path=registry_path, sound_pool_path=pool_path)
        write(output / "batch_manifest.json", result)
    elif args.command == "outcomes":
        result = collect_batch_outcomes(read(args.manifest), read(args.records))
        write(output / "batch_outcomes.json", result)
    elif args.command == "scaleup-dry-run":
        config = read(args.config)
        base = config.setdefault("base_request", {})
        if args.registry is not None:
            base["source_registry"] = str(args.registry)
        catalog_path = resolve_request_room_catalog(base.get("room_catalog"), explicit=args.catalog)
        catalog = read(catalog_path)
        stamp_request_catalog(base, catalog_path=catalog_path, path_bindings=catalog_path_bindings(catalog))
        registry_path = _resolve_declared_file(
            base.get("source_registry"),
            REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
            label="source registry",
        )
        base["source_registry"] = str(registry_path)
        registry = load_source_asset_runtime_registry(registry_path)
        if args.sounds is not None:
            sounds_path = _resolve_declared_file(args.sounds, label="sound pool")
            payload = read(sounds_path)
            sounds = payload.get("sounds", payload) if isinstance(payload, dict) else payload
            pool = None
            base["sound_pool"] = str(sounds_path)
            pool_path = sounds_path
            base.setdefault("sound_selection", {}).pop("prepared_set", None)
        else:
            registry, catalog, sounds, pool, pool_path, catalog_path, registry_path = _load_sounds(config, output)
            stamp_request_catalog(base, catalog_path=catalog_path, path_bindings=catalog_path_bindings(catalog))
        packed = prepare_scaleup_dry_run(
            config, registry, catalog, sounds, seed=args.seed,
            episodes_per_room=args.episodes_per_room, batch_id=args.batch_id,
            off_screen_fraction=args.off_screen_fraction)
        result = packed["manifest"]
        bindings = catalog_path_bindings(catalog)
        sound_pool = base.get("sound_pool")
        for row in result["episodes"]:
            req = row["request"]
            stamp_request_catalog(req, catalog_path=catalog_path, path_bindings=bindings)
            req["source_registry"] = base["source_registry"]
            if sound_pool:
                req["sound_pool"] = sound_pool
        if pool is not None:
            write(output / "batch_sounds.json", pool)
        write(output / "scaleup_config.json", packed["config"])
        result = _annotate_prepare(
            result, output, args.config, catalog_path=catalog_path,
            path_bindings=bindings,
            source_registry_path=registry_path, sound_pool_path=pool_path)
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
