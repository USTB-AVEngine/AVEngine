#!/usr/bin/env python3
"""Build an evidence-derived source-asset qualification matrix.

The command reads measured geometry/support/placement/native artifacts and lets
the qualification module derive every status. It never accepts a caller-written
qualification status and never starts native, UE, Studio, or RLR work.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from avengine.dataset import source_asset_qualification as saq


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_value(root: Path, value: Any) -> Any:
    if isinstance(value, (str, Path)):
        return _load_json(_resolve(root, value))
    return value


def _write_fresh(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace an existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Any]:
    return (
        list(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else []
    )


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _provider_from_foot_contact(
    foot_contact: Mapping[str, Any], *, source_ref: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Convert measured C05 sole/floor rows into observations only."""
    provider: dict[str, dict[str, Any]] = {}
    world_floor_planes: dict[str, dict[str, Any]] = {}
    default_world = foot_contact.get("world_id")
    for actor_value in _items(foot_contact.get("actors")):
        actor = _mapping(actor_value)
        comparison = _mapping(actor.get("floor_comparison"))
        measured = [
            _mapping(frame)
            for frame in _items(comparison.get("frames"))
            if _mapping(frame).get("measurement") == "measured"
        ]
        frames = [_mapping(frame) for frame in _items(actor.get("frames"))]
        if not actor.get("asset_id") or not measured or not frames:
            continue
        floor_values = [
            number
            for frame in measured
            for number in [_finite(frame.get("visual_floor_median_y_m"))]
            if number is not None
        ]
        world_id = str(default_world or actor.get("world_id") or "")
        if not floor_values or not world_id:
            continue
        floor_height = float(median(floor_values))
        world_floor_planes[world_id] = {
            "visual_floor_plane_y_m": floor_height,
            "source": source_ref,
            "method": comparison.get("method"),
            "measured_frame_count": len(floor_values),
        }
        root_position = _items(frames[0].get("root_world_position_m"))
        provider[str(actor["asset_id"])] = {
            "retained_observations": [
                {
                    "world_id": world_id,
                    "facts_path": str(actor.get("capture_root") or source_ref),
                    "native_frames": actor.get("native_frame_count"),
                    "native_emitter_frames": 0,
                    "first_observed_root_y_m": (
                        root_position[1] if len(root_position) >= 2 else None
                    ),
                    "sole_world_y_m": frames[0].get("sole_world_y_m"),
                    "visual_floor_median_y_m": floor_height,
                    "visual_floor_source": source_ref,
                    "measured_frame_count": len(measured),
                    "foot_contact_source": (
                        "avengine.assets.qualification_geometry.measure_native_foot_contact"
                    ),
                    "max_visible_pixels": 0,
                    "visibility_state_counts": {},
                    "audio_events": [],
                }
            ]
        }
    return provider, world_floor_planes


def _navigation_association(
    association: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    floor_references: dict[str, dict[str, Any]] = {}
    world_room_map: dict[str, str] = {}
    for world_id, row_value in _mapping(
        association.get("floor_references_by_world")
    ).items():
        row = _mapping(row_value)
        floor_references[str(world_id)] = {
            "floor_height_m": row.get("floor_height_m"),
            "source": row.get("source"),
            "basis": row.get("basis"),
            "layer": "navigation",
        }
        if row.get("room_id"):
            world_room_map[str(world_id)] = str(row["room_id"])
    return floor_references, world_room_map, {
        "method": association.get("method"),
        "boundary": association.get("boundary"),
    }


def _placement_intents(
    config: Mapping[str, Any], *, config_ref: str
) -> dict[str, dict[str, Any]]:
    intents: dict[str, dict[str, Any]] = {}
    episodes = config.get("episodes")
    rows = episodes.values() if isinstance(episodes, Mapping) else _items(episodes)
    for episode_value in rows:
        episode = _mapping(episode_value)
        extras = _mapping(episode.get("request_extras"))
        placement = _mapping(extras.get("static_source_placement"))
        for request_value in _items(placement.get("requests")):
            request = _mapping(request_value)
            intent = request.get("declared_placement_intent")
            if request.get("asset_id") and isinstance(intent, Mapping):
                intents[str(request["asset_id"])] = {
                    **dict(intent),
                    "episode_id": episode.get("episode_id"),
                    "source": f"{config_ref}#episodes[{episode.get('episode_id')}]",
                }
    return intents


def _media_probe(root: Path):
    def probe(path: str) -> dict[str, Any]:
        resolved = _resolve(root, path)
        result: dict[str, Any] = {"path": str(path), "exists": resolved.exists()}
        if not resolved.exists():
            return result
        result["size_bytes"] = resolved.stat().st_size
        try:
            import numpy as np
            import soundfile as sf

            peak = 0.0
            finite = True
            frame_count = 0
            with sf.SoundFile(str(resolved)) as handle:
                while True:
                    data = handle.read(frames=65536, dtype="float64", always_2d=True)
                    if data.size == 0:
                        break
                    frame_count += int(data.shape[0])
                    finite = finite and bool(np.all(np.isfinite(data)))
                    peak = max(peak, float(np.max(np.abs(data))))
                result.update(
                    {
                        "frames": frame_count,
                        "channels": handle.channels,
                        "sample_rate_hz": handle.samplerate,
                        "finite": finite,
                        "peak_abs": peak,
                        "reader": "soundfile",
                    }
                )
        except Exception as exc:
            result["reader_error"] = f"{type(exc).__name__}: {exc}"
        return result

    return probe


def _run(
    root: Path,
    manifest_path: Path,
    output: Path,
    csv_output: Path | None,
    receipt_output: Path | None,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if _mapping(manifest).get("schema") != "avengine_source_asset_qualification_run_v1":
        raise SystemExit("qualification manifest schema mismatch")
    os.chdir(root)

    def source(key: str) -> Any:
        value = manifest.get(key)
        if isinstance(value, list):
            return [
                str(_resolve(root, item)) if Path(item).is_absolute() else str(item)
                for item in value
            ]
        return value

    association = _load_value(root, manifest["world_floor_association"])
    foot_contact = _load_value(root, manifest["native_foot_contact"])
    floor_references, world_room_map, association_meta = _navigation_association(
        association
    )
    provider, world_floor_planes = _provider_from_foot_contact(
        foot_contact, source_ref=str(manifest["native_foot_contact"])
    )
    config = _load_value(root, manifest["config"])
    matrix = saq.build_qualification_matrix(
        registry=source("registry"),
        source_type_inventory=source("source_type_inventory"),
        asset_worklist=source("asset_worklist"),
        config=source("config"),
        geometry_measurements=source("geometry_measurements"),
        support_catalogs=source("support_catalogs"),
        placement_plans=source("placement_plans"),
        retained_readback=source("retained_readback"),
        provider_evidence=provider,
        floor_reference_m=manifest.get("floor_reference_m"),
        floor_references=floor_references,
        world_room_map=world_room_map,
        world_floor_planes=world_floor_planes,
        placement_intents=_placement_intents(
            config, config_ref=str(manifest["config"])
        ),
        floor_contact_tolerance_m=float(
            manifest.get("floor_contact_tolerance_m", 0.03)
        ),
        media_probe=_media_probe(root),
    )
    matrix["world_association"] = {
        "navigation_floor_by_world": floor_references,
        "navigation_floor_method": association_meta.get("method"),
        "navigation_layer_boundary": association_meta.get("boundary"),
        "visual_floor_by_world": world_floor_planes,
        "world_id_normalisation": (
            "world ids come from retained observations and the C05 capture; "
            "a group_id is never used as a world id"
        ),
        "worlds_without_a_measured_visual_floor": sorted(
            set(floor_references) - set(world_floor_planes)
        ),
    }
    matrix["run_receipt"] = {
        "schema": "avengine_source_asset_qualification_run_receipt_v1",
        "tool": "tools/dataset/run_source_asset_qualification.py",
        "manifest": str(manifest_path),
        "native_started": False,
        "native_start_boundary": (
            "qualification only; native/RLR must be launched by the "
            "allocator-owned production runner after this evidence is reviewed"
        ),
        "provider_assets": sorted(provider),
        "counts": matrix["counts"],
        "status": matrix["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    saq.validate_qualification_matrix(matrix)
    _write_fresh(output, matrix)
    if csv_output is not None:
        if csv_output.exists() or csv_output.is_symlink():
            raise SystemExit(f"refusing to replace an existing output: {csv_output}")
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(saq.qualification_matrix_csv_rows(matrix))
    receipt = {
        "schema": "avengine_source_asset_qualification_run_receipt_v1",
        "tool": "tools/dataset/run_source_asset_qualification.py",
        "matrix_output": str(output),
        "csv_output": str(csv_output) if csv_output else None,
        "manifest": str(manifest_path),
        "native_started": False,
        "status": matrix["status"],
        "counts": matrix["counts"],
        "dimension_totals": matrix["dimension_totals"],
        "work_buckets": matrix["work_buckets"]["counts"],
        "provider_assets": sorted(provider),
        "worlds_without_a_measured_visual_floor": matrix[
            "world_association"
        ]["worlds_without_a_measured_visual_floor"],
    }
    if receipt_output is not None:
        _write_fresh(receipt_output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an evidence-derived 31/59 source qualification matrix; "
            "does not launch native/RLR."
        )
    )
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--manifest", required=True, help="run manifest JSON")
    parser.add_argument("--output", required=True, help="fresh matrix JSON")
    parser.add_argument("--csv-output", help="optional fresh asset-detail CSV")
    parser.add_argument("--receipt-output", help="optional fresh run receipt JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    _run(
        root,
        _resolve(root, args.manifest),
        _resolve(root, args.output),
        _resolve(root, args.csv_output) if args.csv_output else None,
        _resolve(root, args.receipt_output) if args.receipt_output else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
