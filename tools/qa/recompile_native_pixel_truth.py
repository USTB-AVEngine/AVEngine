#!/usr/bin/env python3
"""Recompile retained metric-depth truth with the current lossless fields."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import write_json  # noqa: E402
from avengine.qa.pixel_visibility import (  # noqa: E402
    compile_depth_pixel_visibility_truth,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompile(
    *,
    manifest_path: Path,
    output: Path,
    fact_path: Path | None = None,
    asset_registry_path: Path | None = None,
) -> tuple[Path, Path]:
    manifest = _load(manifest_path)
    source_truth = _load(Path(manifest["artifacts"]["pixel_visibility_truth"]))
    comparison = source_truth["depth_comparison"]
    frame_indices = source_truth["frame_indices"]
    camera_pose_ids = source_truth["camera_pose_ids"]
    common = {
        "renderer_backend": source_truth["renderer_backend"],
        "rgb_renderer_backend": source_truth["rgb_renderer_backend"],
        "camera_contract_id": source_truth["camera_contract_id"],
        "semantic_id_namespace": source_truth["semantic_id_namespace"],
        "resolution_hw": source_truth["resolution_hw"],
        "frame_indices": frame_indices,
        "camera_pose_ids": camera_pose_ids,
    }
    depth_path = Path(manifest["artifacts"]["metric_depth"])
    with np.load(depth_path) as payload:
        normal = payload["normal_depth_m"]
        targets = {
            "source1": payload["target_only_source1_depth_m"],
            "source2": payload["target_only_source2_depth_m"],
        }
        truth = compile_depth_pixel_visibility_truth(
            normal_depth_m_frames=[frame for frame in normal],
            target_only_depth_m_frames_by_instance={
                key: [frame for frame in values] for key, values in targets.items()
            },
            semantic_ids_by_instance={"source1": 1, "source2": 2},
            normal_context={"pass_kind": "modal_scene", **common},
            target_only_contexts_by_instance={
                slot: {
                    "pass_kind": "target_only",
                    "target_instance_id": slot,
                    **common,
                }
                for slot in targets
            },
            target_only_background_depth_m=comparison[
                "target_only_background_depth_m"
            ],
            absolute_tolerance_m=comparison["absolute_tolerance_m"],
            relative_tolerance=comparison["relative_tolerance"],
        )
    for slot in ["source1", "source2"]:
        old = source_truth["per_instance"][slot]
        new = truth["per_instance"][slot]
        if old["state_counts"] != new["state_counts"]:
            raise RuntimeError(f"{slot}: state counts changed during lossless recompile")
        for old_frame, new_frame in zip(old["frames"], new["frames"]):
            for field in [
                "frame_index",
                "state",
                "visible_pixels",
                "target_pixels",
                "visible_fraction",
            ]:
                if old_frame[field] != new_frame[field]:
                    raise RuntimeError(
                        f"{slot} frame {old_frame['frame_index']}: {field} changed"
                    )

    output.mkdir(parents=True, exist_ok=False)
    truth_path = output / "pixel_visibility_truth.json"
    write_json(truth_path, truth)
    derived_manifest = deepcopy(manifest)
    derived_manifest["artifacts"]["pixel_visibility_truth"] = str(truth_path.resolve())
    record = {
        "kind": "file",
        "path": str(truth_path.resolve()),
        "size_bytes": truth_path.stat().st_size,
        "sha256": _sha256(truth_path),
    }
    derived_manifest["artifact_records"]["pixel_visibility_truth"] = record
    derived_manifest["sha256"]["pixel_visibility_truth"] = record["sha256"]
    derived_manifest["lossless_truth_recompile"] = {
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "added_fields": ["target_bbox_xyxy_px", "target_centroid_xy_px"],
        "state_counts_unchanged": True,
        "qualification_claim": False,
    }
    if fact_path is not None or asset_registry_path is not None:
        if fact_path is None or asset_registry_path is None:
            raise RuntimeError("fact and asset registry must be supplied together")
        facts = deepcopy(_load(fact_path))
        registry = _load(asset_registry_path)
        assets = {item["asset_id"]: item for item in registry["assets"]}
        for instance in facts["instances"]:
            asset = assets[instance["asset_id"]]
            # This is the only field added after the retained 0323 Fact was
            # compiled.  Populate it from the exact registry already declared
            # by that Fact; do not infer appearance from pixels or labels.
            instance["attributes"]["sex_or_gender_label"] = asset[
                "realized_attributes"
            ].get("sex_or_gender_label")
        facts["provenance"]["inputs"].append(
            {
                "role": "lossless_source_fact_attribute_upgrade",
                "path": str(fact_path.resolve()),
                "sha256": _sha256(fact_path),
            }
        )
        upgraded_fact_path = output / "facts.json"
        write_json(upgraded_fact_path, facts)
        request = derived_manifest["authoritative_capture_request"]
        request["fact_path"] = str(upgraded_fact_path.resolve())
        request["fact_sha256"] = _sha256(upgraded_fact_path)
        derived_manifest["lossless_truth_recompile"]["fact_attribute_upgrade"] = {
            "field": "instances[*].attributes.sex_or_gender_label",
            "authority": "exact_declared_source_asset_runtime_registry",
            "source_fact": str(fact_path.resolve()),
            "source_fact_sha256": _sha256(fact_path),
            "asset_registry": str(asset_registry_path.resolve()),
            "asset_registry_sha256": _sha256(asset_registry_path),
        }
    output_manifest = output / "manifest.json"
    write_json(output_manifest, derived_manifest)
    return truth_path, output_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--fact", type=Path)
    parser.add_argument("--asset-registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth, manifest = recompile(
        manifest_path=args.capture_manifest.resolve(),
        output=args.output.resolve(),
        fact_path=None if args.fact is None else args.fact.resolve(),
        asset_registry_path=(
            None if args.asset_registry is None else args.asset_registry.resolve()
        ),
    )
    print(f"NATIVE_PIXEL_TRUTH_RECOMPILED truth={truth} manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
