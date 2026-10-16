#!/usr/bin/env python3
"""Derive static occluder identity from native modal/target-only pixels."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "avengine_native_static_occluder_evidence_v1"
AUTHORITY = (
    "same_renderer_same_camera_occluded_target_footprint_"
    "normal_static_object_ids_v1"
)
MIN_OCCLUDED_PIXELS = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _occluder_id(stable_name: str) -> str:
    return "native_static_object::" + stable_name


def _admit_unique_occluder(
    *, occluded_pixels: int, grouped: Mapping[str, int]
) -> list[str]:
    """Admit only one known static object covering every hidden target pixel."""
    if (
        occluded_pixels < MIN_OCCLUDED_PIXELS
        or len(grouped) != 1
        or sum(grouped.values()) != occluded_pixels
    ):
        return []
    return [next(iter(grouped))]


def derive(manifest_path: Path, output_path: Path) -> Mapping[str, Any]:
    manifest = _load(manifest_path)
    _require(
        manifest.get("status") == "pass"
        and manifest.get("native_pixel_fact_binding_claim") is True,
        "native pixel capture did not pass",
    )
    paths = {
        name: Path(path).resolve() for name, path in manifest["artifacts"].items()
    }
    for name in (
        "pixel_masks",
        "pixel_visibility_truth",
        "normal_object_ids",
        "object_id_descriptors",
    ):
        _require(
            manifest["sha256"].get(name) == _sha256(paths[name]),
            f"native artifact hash drift: {name}",
        )
    truth = _load(paths["pixel_visibility_truth"])
    _require(
        truth.get("authority")
        == "same_renderer_same_camera_normal_vs_target_only_metric_depth_v1",
        "occluder identity requires native metric-depth pixel truth",
    )
    descriptors = _load(paths["object_id_descriptors"])["descriptors"]
    descriptor_by_raw_id = {
        int(item["rawId"]): item
        for item in descriptors
        if isinstance(item.get("rawId"), int)
        and isinstance(item.get("actorStableName"), str)
    }
    registry: dict[str, dict[str, Any]] = {}
    frame_records: list[dict[str, Any]] = []
    with np.load(paths["pixel_masks"]) as masks, np.load(
        paths["normal_object_ids"]
    ) as raw_payload:
        raw_ids = raw_payload["normal_object_ids"]
        for instance_id, entry in sorted(truth["per_instance"].items()):
            semantic_id = int(entry["semantic_id"])
            target = masks[f"target_only_{instance_id}"] == semantic_id
            visible = masks[f"modal_visible_{instance_id}"].astype(bool)
            for capture_index, frame in enumerate(entry["frames"]):
                if frame["state"] not in {"visible_occluded", "fully_occluded"}:
                    continue
                occluded = target[capture_index] & ~visible[capture_index]
                occluded_pixels = int(np.count_nonzero(occluded))
                if occluded_pixels == 0:
                    continue
                raw_counts = Counter(
                    int(value) for value in raw_ids[capture_index][occluded]
                )
                grouped: Counter[str] = Counter()
                raw_ids_by_occluder: dict[str, list[int]] = {}
                for raw_id, count in raw_counts.items():
                    descriptor = descriptor_by_raw_id.get(raw_id)
                    if descriptor is None:
                        continue
                    stable_name = descriptor["actorStableName"]
                    occluder_id = _occluder_id(stable_name)
                    grouped[occluder_id] += count
                    raw_ids_by_occluder.setdefault(occluder_id, []).append(raw_id)
                    registry.setdefault(
                        occluder_id,
                        {
                            "occluder_id": occluder_id,
                            "display_label": stable_name.rsplit("/", 1)[-1],
                            "actor_stable_name": stable_name,
                            "actor_names": [],
                            "raw_object_ids": [],
                        },
                    )
                    actor_name = descriptor.get("actorName")
                    if isinstance(actor_name, str):
                        registry[occluder_id]["actor_names"].append(actor_name)
                candidates = [
                    {
                        "occluder_id": occluder_id,
                        "pixel_count": count,
                        "fraction_of_occluded_target": count / occluded_pixels,
                        "raw_object_ids": sorted(raw_ids_by_occluder[occluder_id]),
                    }
                    for occluder_id, count in grouped.most_common()
                ]
                admitted = _admit_unique_occluder(
                    occluded_pixels=occluded_pixels,
                    grouped=grouped,
                )
                frame_records.append(
                    {
                        "target_instance_id": instance_id,
                        "frame_index": int(frame["frame_index"]),
                        "pixel_state": frame["state"],
                        "occluded_pixels": occluded_pixels,
                        "known_static_object_pixels": sum(grouped.values()),
                        "candidates": candidates,
                        "occluder_instance_ids": admitted,
                        "decision": (
                            "unique_static_occluder"
                            if admitted
                            else "rejected_ambiguous_or_insufficient"
                        ),
                    }
                )
    for record in registry.values():
        record["actor_names"] = sorted(set(record["actor_names"]))
        record["raw_object_ids"] = sorted(
            {
                raw_id
                for frame in frame_records
                for candidate in frame["candidates"]
                if candidate["occluder_id"] == record["occluder_id"]
                for raw_id in candidate["raw_object_ids"]
            }
        )
    result = {
        "schema": SCHEMA,
        "status": "computed_native_static_object_ids_v1",
        "authority": AUTHORITY,
        "camera_pose_ids": manifest["frame_contract"]["camera_pose_ids"],
        "decision_policy": {
            "minimum_occluded_pixels": MIN_OCCLUDED_PIXELS,
            "unique_static_object_required": True,
            "all_occluded_pixels_require_known_static_object_id": True,
            "unknown_raw_ids_are_never_admitted": True,
        },
        "source_artifacts": {
            name: {"path": str(paths[name]), "sha256": _sha256(paths[name])}
            for name in (
                "pixel_masks",
                "pixel_visibility_truth",
                "normal_object_ids",
                "object_id_descriptors",
            )
        },
        "occluder_registry": registry,
        "frame_records": frame_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = derive(args.capture_manifest.resolve(), args.output.resolve())
    admitted = sum(
        bool(frame["occluder_instance_ids"]) for frame in result["frame_records"]
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "frame_record_count": len(result["frame_records"]),
                "unique_occluder_frame_count": admitted,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
