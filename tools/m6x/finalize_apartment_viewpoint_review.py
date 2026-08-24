#!/usr/bin/env python3
"""Finalize retained Apartment proxy review or publish a truthful blocker.

This tool never turns proxy review into UE/SPEAR certification.  Four or more
exact retained poses produce a native-pixel-pending review bank that the
production loader must reject; fewer poses produce only ``BLOCKED.json``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


MINIMUM_RETAINED_PROXY_COUNT = 4


class ApartmentViewpointReviewFinalizerError(ValueError):
    """The review inputs cannot produce a truthful retained-proxy artifact."""


def _required_file(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ApartmentViewpointReviewFinalizerError(
            f"{owner} does not resolve: {path}"
        ) from exc
    if not resolved.is_file():
        raise ApartmentViewpointReviewFinalizerError(
            f"{owner} must resolve to a file: {path}"
        )
    return resolved


def _fresh_staging(
    output: Path,
) -> tuple[WorkspacePathPolicy, Path, Path]:
    unresolved = output.expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    unresolved_staging = unresolved.with_name(f".{unresolved.name}.staging")
    if os.path.lexists(unresolved) or os.path.lexists(unresolved_staging):
        raise ApartmentViewpointReviewFinalizerError(
            f"refusing to replace output or staging path: {unresolved}"
        )
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    parent = unresolved.parent.resolve(strict=True)
    policy = WorkspacePathPolicy.from_roots((parent,))
    destination = policy.resolve_output(
        parent / unresolved.name,
        owner="Apartment retained review output",
    )
    staging = policy.resolve_output(
        parent / unresolved_staging.name,
        owner="Apartment retained review staging",
    )
    staging.mkdir()
    return policy, destination, staging


def _camera_key(value: Any, *, owner: str) -> tuple[tuple[float, float, float], float]:
    if not isinstance(value, Mapping):
        raise ApartmentViewpointReviewFinalizerError(f"{owner} must be an object")
    try:
        position = np.asarray(value["habitat_translation_m"], dtype=np.float64)
        yaw = float(value["habitat_yaw_deg"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ApartmentViewpointReviewFinalizerError(
            f"{owner} camera pose is invalid"
        ) from exc
    if (
        position.shape != (3,)
        or not np.all(np.isfinite(position))
        or not math.isfinite(yaw)
    ):
        raise ApartmentViewpointReviewFinalizerError(f"{owner} camera pose is invalid")
    return tuple(float(item) for item in position), yaw


def _review_decisions(
    path: Path,
) -> dict[tuple[tuple[float, float, float], float], Mapping[str, Any]]:
    document = load_json(path)
    if not isinstance(document, Mapping) or not isinstance(
        document.get("decisions"), list
    ):
        raise ApartmentViewpointReviewFinalizerError(
            "review decision must contain a decisions list"
        )
    result = {}
    for index, value in enumerate(document["decisions"]):
        if not isinstance(value, Mapping):
            raise ApartmentViewpointReviewFinalizerError(
                f"review decisions[{index}] must be an object"
            )
        decision = value.get("decision")
        reason = value.get("reason")
        if (
            decision not in ("retain", "reject")
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ApartmentViewpointReviewFinalizerError(
                f"review decisions[{index}] is incomplete"
            )
        key = _camera_key(value.get("camera"), owner=f"review decisions[{index}]")
        if key in result:
            raise ApartmentViewpointReviewFinalizerError(
                "review decision contains duplicate exact camera poses"
            )
        result[key] = value
    return result


def _prefilter_viewpoints(
    path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    document = load_json(path)
    if not isinstance(document, Mapping):
        raise ApartmentViewpointReviewFinalizerError("prefilter must be an object")
    if (
        document.get("artifact_status") != "deterministic_2d_raster_prefilter_only"
        or document.get("native_pixel_validation_status") != "pending"
        or document.get("studio_used") is not False
    ):
        raise ApartmentViewpointReviewFinalizerError(
            "prefilter must be a Studio-free native-pixel-pending 2-D prefilter"
        )
    viewpoints = document.get("viewpoints")
    if not isinstance(viewpoints, list) or not viewpoints:
        raise ApartmentViewpointReviewFinalizerError(
            "prefilter must contain a nonempty viewpoints list"
        )
    return document, viewpoints


def run(args: argparse.Namespace) -> tuple[Path, str]:
    prefilter_path = _required_file(args.prefilter, owner="--prefilter")
    review_path = _required_file(args.review_decision, owner="--review-decision")
    policy, output, staging = _fresh_staging(args.output)
    prefilter, viewpoints = _prefilter_viewpoints(prefilter_path)
    decisions = _review_decisions(review_path)
    by_pose = {
        _camera_key(item.get("camera"), owner=f"prefilter viewpoint {index}"): item
        for index, item in enumerate(viewpoints)
        if isinstance(item, Mapping)
    }
    retained = []
    for pose, decision in decisions.items():
        if decision["decision"] != "retain":
            continue
        viewpoint = by_pose.get(pose)
        if viewpoint is None:
            raise ApartmentViewpointReviewFinalizerError(
                "retained review pose is absent from the selected source prefilter"
            )
        retained.append((viewpoint, decision))
    retained.sort(key=lambda item: str(item[0].get("viewpoint_id")).encode("utf-8"))

    common = {
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "studio_used": False,
        "source_prefilter": str(prefilter_path),
        "review_decision": str(review_path),
        "retained_proxy_count": len(retained),
        "required_retained_proxy_count": MINIMUM_RETAINED_PROXY_COUNT,
        "ue_spear_pixel_certification_status": "pending",
    }
    if len(retained) < MINIMUM_RETAINED_PROXY_COUNT:
        write_json(
            staging / "BLOCKED.json",
            {
                **common,
                "status": "blocked_insufficient_retained_proxy_viewpoints",
                "retained_viewpoint_ids": [
                    item[0].get("viewpoint_id") for item in retained
                ],
                "claim_boundary": (
                    "proxy review only; no viewpoint bank, native certification, "
                    "Episode or dataset item was produced"
                ),
            },
        )
        return atomic_publish_directory(policy, staging, output), "blocked"

    request_paths = prefilter.get("request_paths")
    if not isinstance(request_paths, Mapping):
        raise ApartmentViewpointReviewFinalizerError(
            "prefilter request_paths must be an object"
        )
    requests_directory = staging / "requests"
    requests_directory.mkdir()
    finalized_viewpoints = []
    for viewpoint, decision in retained:
        viewpoint_id = viewpoint.get("viewpoint_id")
        relative = request_paths.get(viewpoint_id)
        if not isinstance(viewpoint_id, str) or not isinstance(relative, str):
            raise ApartmentViewpointReviewFinalizerError(
                "retained viewpoint lacks its M1 request reference"
            )
        request_source = _required_file(
            prefilter_path.parent / relative,
            owner=f"{viewpoint_id} M1 request",
        )
        request_relative = Path("requests") / f"{viewpoint_id}.json"
        shutil.copy2(request_source, staging / request_relative)
        record = deepcopy(dict(viewpoint))
        record.update(
            {
                "native_pixel_validation_status": "pending",
                "native_validation_pending": sorted(
                    set(record.get("native_validation_pending", []))
                    | {"ue_spear_first_person_pixels", "same_renderer_visibility"}
                ),
                "m1_request": {
                    "status": "proxy_authored_native_certification_pending",
                    "path": request_relative.as_posix(),
                },
                "review_decision": {
                    "decision": "retain",
                    "reason": decision["reason"],
                },
            }
        )
        finalized_viewpoints.append(record)
    write_json(
        staging / "viewpoint_bank_apartment_v1.json",
        {
            **common,
            "artifact_status": "retained_proxy_viewpoint_bank_native_pixel_pending",
            "native_pixel_validation_status": "pending",
            "claim_boundary": (
                "retained proxy camera poses only; production loader must reject "
                "until UE/SPEAR pixels, same-renderer visibility, side calibration, "
                "RIR identity and per-view mini-pilots pass"
            ),
            "corridors": deepcopy(prefilter.get("corridors", [])),
            "viewpoints": finalized_viewpoints,
        },
    )
    return atomic_publish_directory(policy, staging, output), "pending_bank"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefilter", required=True, type=Path)
    parser.add_argument("--review-decision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output, status = run(parse_args(argv))
    print(f"{status}: {output}")
    return 2 if status == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
