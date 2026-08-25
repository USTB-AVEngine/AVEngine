#!/usr/bin/env python3
"""Derive synchronized RGB + navmesh/descriptor top-down QA review media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, sha256_file
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import (
    _resolved_assets,
    _resolved_scene,
    discover_runtime_root,
)
from avengine.assets.habitat_capture import (
    EVIDENCE_SCHEMA as HABITAT_CAPTURE_EVIDENCE_SCHEMA,
    verify_saved_capture_arrays,
)
from avengine.assets.local_tr_review import (
    EVIDENCE_SCHEMA as LOCAL_TR_REVIEW_EVIDENCE_SCHEMA,
    verify_local_tr_review_evidence,
)
from avengine.assets.review_topdown import (
    TopdownReviewError,
    compose_topdown_review,
    semantic_object_footprint_from_obb,
)
from avengine.assets.variant_review import verify_variant_review_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Existing review-only capture directory containing evidence.json",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="Explicit non-checkout Habitat runtime data root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh directory for the derived QA video and evidence",
    )
    return parser


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TopdownReviewError(f"{owner} must be a JSON object")
    return value


def _verified_input_path(
    record: Any,
    *,
    owner: str,
    expected_sha256: str | None = None,
) -> Path:
    value = _mapping(record, owner=owner)
    raw_path = value.get("path")
    expected = expected_sha256 or value.get("sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise TopdownReviewError(f"{owner} must bind an absolute path")
    if not isinstance(expected, str) or len(expected) != 64:
        raise TopdownReviewError(f"{owner} must bind a SHA-256")
    path = Path(raw_path).resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise TopdownReviewError(f"{owner} bytes differ from capture evidence")
    return path


def _verified_capture(capture: Path) -> tuple[Mapping[str, Any], Path, Path | None]:
    evidence_path = capture / "evidence.json"
    if not evidence_path.is_file():
        raise TopdownReviewError("capture/evidence.json is missing")
    try:
        evidence = load_json(evidence_path)
    except (OSError, ValueError) as exc:
        raise TopdownReviewError(f"capture evidence is invalid: {exc}") from exc
    if (
        evidence.get("review_only") is not True
        or evidence.get("qualification_claim") is not False
        or evidence.get("formal_view_ids") != []
        or evidence.get("review_view_ids") != ["view0"]
        or evidence.get("sensor_contract", {}).get("view_id") != "view0"
    ):
        raise TopdownReviewError(
            "top-down derivation requires one non-qualifying review-only view0"
        )
    array_errors = verify_saved_capture_arrays(evidence, capture)
    if array_errors:
        raise TopdownReviewError(
            "capture array verification failed: " + "; ".join(array_errors)
        )
    wrapper_path = capture / "variant_review_evidence.json"
    schema = evidence.get("schema")
    if schema == LOCAL_TR_REVIEW_EVIDENCE_SCHEMA:
        if wrapper_path.exists() or wrapper_path.is_symlink():
            raise TopdownReviewError(
                "local-TR core capture must not use a rotation-only variant wrapper"
            )
        local_tr_errors = verify_local_tr_review_evidence(evidence_path)
        if local_tr_errors:
            raise TopdownReviewError(
                "local-TR capture verification failed: " + "; ".join(local_tr_errors)
            )
        wrapper: Path | None = None
    elif schema == HABITAT_CAPTURE_EVIDENCE_SCHEMA and wrapper_path.is_file():
        wrapper_errors = verify_variant_review_evidence(wrapper_path)
        if wrapper_errors:
            raise TopdownReviewError(
                "variant wrapper verification failed: " + "; ".join(wrapper_errors)
            )
        wrapper: Path | None = wrapper_path
    else:
        raise TopdownReviewError(
            "capture schema requires a supported strict core/wrapper verifier"
        )
    return evidence, evidence_path, wrapper


def _rgb_array_path(evidence: Mapping[str, Any], capture: Path) -> Path:
    arrays = _mapping(evidence.get("array_artifacts"), owner="array_artifacts")
    rgb = _mapping(arrays.get("rgb"), owner="array_artifacts.rgb")
    artifact = _mapping(rgb.get("artifact"), owner="array_artifacts.rgb.artifact")
    relative = artifact.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise TopdownReviewError("RGB array artifact must use a relative path")
    path = (capture / relative).resolve()
    try:
        path.relative_to(capture)
    except ValueError as exc:
        raise TopdownReviewError("RGB array artifact escapes capture") from exc
    if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
        raise TopdownReviewError("RGB array artifact bytes differ")
    return path


def _topdown_declaration(room_request: Mapping[str, Any]) -> Mapping[str, Any]:
    declarations = [
        value
        for value in room_request.get("qa_views", [])
        if isinstance(value, Mapping) and value.get("kind") == "topdown"
    ]
    if len(declarations) != 1:
        raise TopdownReviewError(
            "room request must declare exactly one topdown QA view"
        )
    return declarations[0]


def _semantic_descriptor_path(room_inputs: Any, runtime: Path) -> Path | None:
    descriptors = [
        record
        for record in _resolved_assets(room_inputs, runtime)
        if record.get("role") == "semantic_descriptor"
    ]
    if not descriptors:
        return None
    if len(descriptors) != 1:
        raise TopdownReviewError(
            "room must declare at most one semantic_descriptor asset"
        )
    record = descriptors[0]
    path = Path(str(record.get("resolved_path"))).resolve()
    if record.get("exists") is not True or not path.is_file():
        raise TopdownReviewError("declared semantic descriptor is missing")
    return path


def _descriptor_object_footprints(
    *,
    room_inputs: Any,
    runtime: Path,
    semantic_descriptor: Path | None,
) -> list[dict[str, Any]]:
    """Read Habitat descriptor OBBs without creating or rendering a sensor."""

    if semantic_descriptor is None:
        return []
    resolved = _resolved_scene(room_inputs, runtime)

    # The pinned audio-enabled build requires numpy-quaternion before Habitat.
    import quaternion  # noqa: F401

    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(resolved["scene_id"])
    sim_cfg.scene_dataset_config_file = str(resolved["dataset_config"])
    sim_cfg.load_semantic_mesh = bool(resolved["load_semantic_mesh"])
    sim_cfg.enable_physics = False
    sim_cfg.random_seed = int(room_inputs.request["seed"])
    if hasattr(sim_cfg, "create_renderer"):
        sim_cfg.create_renderer = False
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    agent_cfg.action_space = {}

    footprints: list[dict[str, Any]] = []
    configuration = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    with habitat_sim.Simulator(configuration) as simulator:
        for semantic_object in simulator.semantic_scene.objects:
            if semantic_object is None or semantic_object.category is None:
                continue
            category = str(semantic_object.category.name())
            footprint = semantic_object_footprint_from_obb(
                object_id=str(semantic_object.id),
                category=category,
                local_to_world=np.asarray(
                    semantic_object.obb.local_to_world, dtype=np.float64
                ),
            )
            if footprint is not None:
                footprints.append(footprint)
    footprints.sort(key=lambda item: (item["category"], item["object_id"]))
    return footprints


def compose_capture_topdown_review(
    *, capture_dir: str | Path, runtime_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Verify a retained capture and derive one independent QA-only video."""

    capture = Path(capture_dir).resolve()
    output = Path(output_dir).resolve()
    if not capture.is_dir():
        raise TopdownReviewError("capture directory is missing")
    if output.exists() or output.is_symlink():
        raise TopdownReviewError("refusing to replace top-down review output")
    evidence, evidence_path, wrapper_path = _verified_capture(capture)

    inputs = _mapping(evidence.get("inputs"), owner="capture inputs")
    room_manifest_path = _verified_input_path(
        inputs.get("m1_room_manifest"), owner="M1 room manifest"
    )
    room_request_path = _verified_input_path(
        inputs.get("m1_camera_request"), owner="M1 camera request"
    )
    room_inputs = load_m1_inputs(room_manifest_path, room_request_path)
    qa_view = _topdown_declaration(room_inputs.request)

    runtime = discover_runtime_root(runtime_root)
    semantic_descriptor = _semantic_descriptor_path(room_inputs, runtime)
    semantic_footprints = _descriptor_object_footprints(
        room_inputs=room_inputs,
        runtime=runtime,
        semantic_descriptor=semantic_descriptor,
    )
    scene = _resolved_scene(room_inputs, runtime)
    navmesh_raw = scene.get("navmesh")
    if navmesh_raw is None:
        raise TopdownReviewError("room does not declare a navmesh")
    navmesh_path = Path(navmesh_raw).resolve()
    if not navmesh_path.is_file():
        raise TopdownReviewError("declared room navmesh is missing")

    # The pinned audio-enabled build requires numpy-quaternion before Habitat.
    import quaternion  # noqa: F401

    import habitat_sim

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path)) or not pathfinder.is_loaded:
        raise TopdownReviewError("Habitat failed to load the declared navmesh")
    meters_per_pixel = float(qa_view.get("meters_per_pixel", 0.05))
    height_m = float(qa_view.get("height_m", 0.1))
    navmesh_map = np.asarray(
        pathfinder.get_topdown_view(meters_per_pixel, height_m), dtype=np.uint8
    )
    if navmesh_map.ndim != 2 or not np.any(navmesh_map):
        raise TopdownReviewError("top-down navmesh QA map is empty")
    lower, upper = pathfinder.get_bounds()
    bounds = (
        tuple(float(value) for value in lower),
        tuple(float(value) for value in upper),
    )

    rgb_path = _rgb_array_path(evidence, capture)
    rgb_frames = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    frame_records = evidence.get("frames")
    if not isinstance(frame_records, list):
        raise TopdownReviewError("capture frame records are missing")
    artifacts: dict[str, Path] = {
        "core_capture_evidence": evidence_path,
        "rgb_array": rgb_path,
        "room_manifest": room_manifest_path,
        "room_request": room_request_path,
        "navmesh": navmesh_path,
    }
    if wrapper_path is not None:
        artifacts["variant_review_evidence"] = wrapper_path
    if semantic_descriptor is not None:
        artifacts["semantic_descriptor"] = semantic_descriptor

    result = compose_topdown_review(
        rgb_frames=rgb_frames,
        frame_records=frame_records,
        room_camera_request=room_inputs.request,
        navmesh_binary_map=navmesh_map,
        navmesh_bounds=bounds,
        output_dir=output,
        source_anchors=room_inputs.request.get("sources", []),
        semantic_object_footprints=semantic_footprints,
        panel_size_wh=(240, 240),
        fps=15,
        output_name="view0_rgb_topdown_review.mp4",
        navmesh_metadata={
            "qa_id": qa_view.get("qa_id"),
            "meters_per_pixel": meters_per_pixel,
            "height_m": height_m,
            "navigable_pixel_count": int(np.count_nonzero(navmesh_map)),
            "room_id": room_inputs.room.get("room_id"),
        },
        input_artifacts=artifacts,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compose_capture_topdown_review(
        capture_dir=args.capture,
        runtime_root=args.runtime_root,
        output_dir=args.output,
    )
    output = args.output.resolve()
    print(
        json.dumps(
            {
                "status": result["status"],
                "review_only": result["review_only"],
                "qa_only": result["qa_only"],
                "formal_view": result["formal_view"],
                "qualification_claim": result["qualification_claim"],
                "formal_view_ids": result["formal_view_ids"],
                "video": {
                    **result["output"]["video"],
                    "absolute_path": str(output / result["output"]["video"]["path"]),
                },
                "layout": result["layout"],
                "semantic_object_footprints": {
                    "count": result["inputs"]["semantic_object_footprints"]["count"],
                    "descriptor_semantics_not_object_detection": result["inputs"][
                        "semantic_object_footprints"
                    ]["descriptor_semantics_not_object_detection"],
                    "canonical_content_sha256": result["inputs"][
                        "semantic_object_footprints"
                    ]["canonical_content_sha256"],
                },
                "evidence": str(output / "topdown_review_evidence.json"),
                "evidence_content_sha256": result["evidence_content_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
