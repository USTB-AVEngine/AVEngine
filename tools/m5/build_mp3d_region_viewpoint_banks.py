#!/usr/bin/env python3
"""Build four target-agnostic CPU camera requests per MP3D region instance."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from avengine.camera_pose import CameraPoseError, apply_camera_listener_pose
from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import validate_capture_request
from avengine.m5.mp3d_region_viewpoints import (
    MP3DRegionViewpointAnalysis,
    MP3DRegionViewpointError,
    analyze_mp3d_region_viewpoints,
)
from avengine.security import (
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)


class MP3DRegionViewpointBuildError(ValueError):
    """The declared inputs cannot produce one fresh per-region bank."""


def _required_file(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{owner} is not a file: {resolved}")
    return resolved


def _fresh_output(path: Path) -> tuple[Path, WorkspacePathPolicy]:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise MP3DRegionViewpointBuildError("--output must be an absolute path")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise MP3DRegionViewpointBuildError(
            f"--output parent does not exist: {raw.parent}"
        ) from exc
    output = parent / raw.name
    if os.path.lexists(output):
        raise MP3DRegionViewpointBuildError(f"refusing to replace output: {output}")
    try:
        policy = WorkspacePathPolicy.from_roots((parent,))
    except PathPolicyError as exc:
        raise MP3DRegionViewpointBuildError(str(exc)) from exc
    return output, policy


def _safe_region_token(region_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", region_id).strip("._-")
    if not token:
        raise MP3DRegionViewpointBuildError("region instance ID has no safe token")
    return token


def _materialize_requests(
    staging: Path,
    *,
    analysis: MP3DRegionViewpointAnalysis,
    base_request: Mapping[str, Any],
) -> dict[str, str]:
    base_request_id = base_request.get("request_id")
    if not isinstance(base_request_id, str) or not base_request_id:
        raise MP3DRegionViewpointBuildError("base M1 request_id must be nonempty")
    result: dict[str, str] = {}
    request_ids: set[str] = set()
    for bank in analysis.region_banks:
        region_id = str(bank["region_instance_id"])
        request_root = staging / "requests" / _safe_region_token(region_id)
        request_root.mkdir(parents=True)
        for selected in bank["selected_candidates"]:
            placement_id = str(selected["placement_id"])
            camera = selected["camera"]
            request_id = (
                f"{base_request_id}_{_safe_region_token(region_id)}_"
                f"{placement_id}_cpu_viewpoint_candidate"
            )
            if request_id in request_ids:
                raise MP3DRegionViewpointBuildError(
                    "materialized M1 request IDs must be unique"
                )
            request_ids.add(request_id)
            try:
                request = apply_camera_listener_pose(
                    base_request,
                    request_id=request_id,
                    position_m=camera["position_m"],
                    yaw_deg=camera["yaw_deg"],
                )
                topdown_views = [
                    view
                    for view in request.get("qa_views", [])
                    if view.get("kind") == "topdown"
                ]
                if len(topdown_views) != 1:
                    raise MP3DRegionViewpointBuildError(
                        "region viewpoint M1 must contain exactly one Topdown QA view"
                    )
                topdown_views[0]["height_m"] = float(camera["floor_position_m"][1])
                errors = validate_capture_request(request, room_id=analysis.room_id)
                if errors:
                    raise MP3DRegionViewpointBuildError(
                        "region viewpoint M1 is invalid after floor binding: "
                        + "; ".join(errors)
                    )
            except (CameraPoseError, KeyError, TypeError, ValueError) as exc:
                raise MP3DRegionViewpointBuildError(
                    f"could not author M1 request for {placement_id}: {exc}"
                ) from exc
            relative = (
                Path("requests")
                / _safe_region_token(region_id)
                / f"{placement_id}.json"
            )
            write_json(staging / relative, request)
            result[placement_id] = relative.as_posix()
    return result


def _record_with_request_paths(
    analysis: MP3DRegionViewpointAnalysis, request_paths: Mapping[str, str]
) -> dict[str, Any]:
    record = analysis.to_record()
    seen: set[str] = set()
    for bank in record["region_banks"]:
        for selected in bank["selected_candidates"]:
            placement_id = str(selected["placement_id"])
            if placement_id in seen or placement_id not in request_paths:
                raise MP3DRegionViewpointBuildError(
                    "selected placement/request path mapping drift"
                )
            seen.add(placement_id)
            selected["m1_capture_request_path"] = request_paths[placement_id]
    if seen != set(request_paths):
        raise MP3DRegionViewpointBuildError(
            "materialized request paths differ from selected placements"
        )
    return record


def run(args: argparse.Namespace) -> Path:
    sidecar_path = _required_file(
        args.camera_region_sidecar, owner="--camera-region-sidecar"
    )
    catalog_path = _required_file(args.camera_catalog, owner="--camera-catalog")
    request_path = _required_file(args.base_m1_request, owner="--base-m1-request")
    output, policy = _fresh_output(args.output)
    sidecar = load_json(sidecar_path)
    catalog = load_json(catalog_path)
    base_request = load_json(request_path)
    room_id = sidecar.get("room_id")
    if not isinstance(room_id, str) or not room_id:
        raise MP3DRegionViewpointBuildError("sidecar room_id must be nonempty")
    errors = validate_capture_request(base_request, room_id=room_id)
    if errors:
        raise MP3DRegionViewpointBuildError(
            "base M1 request is invalid: " + "; ".join(errors)
        )
    try:
        analysis = analyze_mp3d_region_viewpoints(
            sidecar=sidecar,
            catalog=catalog,
            base_request=base_request,
            views_per_region=args.views_per_region,
        )
    except MP3DRegionViewpointError as exc:
        raise MP3DRegionViewpointBuildError(str(exc)) from exc

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        request_paths = _materialize_requests(
            staging,
            analysis=analysis,
            base_request=base_request,
        )
        record = _record_with_request_paths(analysis, request_paths)
        record["inputs"] = {
            "camera_region_sidecar": str(sidecar_path),
            "camera_catalog": str(catalog_path),
            "base_m1_request": str(request_path),
        }
        record["base_m1_source_declarations_preserved_but_not_ranked"] = True
        record["m1_capture_request_paths"] = dict(sorted(request_paths.items()))
        write_json(staging / "region_viewpoint_banks.json", record)
        (staging / "README.md").write_text(
            "# MP3D per-region CPU viewpoint banks\n\n"
            "This bundle retains the complete sidecar statistics and selects four "
            "distinct uniquely-bound floor positions per static `.house` region. "
            "Position, yaw, height and boundary distance are continuous preferences; "
            "there are no numeric selection thresholds. Region categories, source "
            "routes and visibility are not ranking inputs. The 40 M1 files are CPU "
            "probe requests, not native-certified viewpoints. Unknown, ambiguous and "
            "boundary input memberships remain retained in the global statistics and "
            "are not called illegal.\n",
            encoding="utf-8",
        )
        try:
            atomic_publish_directory(policy, staging, output)
        except (FileExistsError, PathPolicyError) as exc:
            raise MP3DRegionViewpointBuildError(str(exc)) from exc
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-region-sidecar", required=True, type=Path)
    parser.add_argument("--camera-catalog", required=True, type=Path)
    parser.add_argument("--base-m1-request", required=True, type=Path)
    parser.add_argument("--views-per-region", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
