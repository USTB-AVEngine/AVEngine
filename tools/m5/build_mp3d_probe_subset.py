#!/usr/bin/env python3
"""Join a full legal MP3D camera catalog and rank native-probe candidates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.camera_pose import CameraPoseError, apply_camera_listener_pose
from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import (
    load_and_validate_inputs,
    validate_capture_request,
)
from avengine.m5.current_visual import _current_mp3d_room_error
from avengine.m5.mp3d_probe_subset import (
    MP3DProbeSubsetAnalysis,
    MP3DProbeSubsetError,
    analyze_mp3d_probe_candidates,
    build_placement_descriptor,
)
from avengine.security import (
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)


class MP3DProbeSubsetBuildError(ValueError):
    """The retained inputs cannot produce one fresh probe subset."""


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
        raise MP3DProbeSubsetBuildError("--output must be an absolute path")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise MP3DProbeSubsetBuildError(
            f"--output parent does not exist: {raw.parent}"
        ) from exc
    output = parent / raw.name
    if os.path.lexists(output):
        raise MP3DProbeSubsetBuildError(f"refusing to replace output: {output}")
    try:
        policy = WorkspacePathPolicy.from_roots((parent,))
    except PathPolicyError as exc:
        raise MP3DProbeSubsetBuildError(str(exc)) from exc
    return output, policy


def _source_ids(request: Mapping[str, Any]) -> tuple[str, str]:
    sources = request.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise MP3DProbeSubsetBuildError("base M1 request must contain two sources")
    result = tuple(
        item.get("source_id") if isinstance(item, Mapping) else None for item in sources
    )
    if (
        len(result) != 2
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != 2
    ):
        raise MP3DProbeSubsetBuildError(
            "base M1 request must contain two unique source IDs"
        )
    return result  # type: ignore[return-value]


def _validate_source_receipt(
    value: Any,
    *,
    receipt_path: Path,
    source_positions_path: Path,
    source_m2_request_path: Path,
    base_request: Mapping[str, Any],
    source_positions: np.ndarray,
    expected_room_id: str,
    expected_source_ids: tuple[str, str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DProbeSubsetBuildError("source receipt must be an object")
    if (
        value.get("status") != "research_only"
        or value.get("research_only") is not True
        or value.get("episode_counted") is not False
    ):
        raise MP3DProbeSubsetBuildError(
            "source receipt is not a current research-only native capture"
        )

    actors = value.get("actors")
    artifacts = value.get("artifacts")
    capture = value.get("capture")
    inputs = value.get("inputs")
    selected_room = value.get("selected_room")
    if (
        not isinstance(selected_room, Mapping)
        or selected_room.get("room_id") != expected_room_id
        or selected_room.get("room_kind") != "habitat_native"
        or not isinstance(capture, Mapping)
        or capture.get("native_habitat_started") is not True
        or capture.get("frame_count") != 75
    ):
        raise MP3DProbeSubsetBuildError(
            "source receipt does not identify the current native Habitat room and 75 frames"
        )
    if not isinstance(actors, Mapping):
        raise MP3DProbeSubsetBuildError("source receipt lacks native actor bindings")
    actor_ids = _nonempty_string_list(
        actors.get("actor_ids"), owner="source receipt actor_ids", expected_count=2
    )
    source_ids = _nonempty_string_list(
        actors.get("source_ids"), owner="source receipt source_ids", expected_count=2
    )
    emitter_link_names = _nonempty_string_list(
        actors.get("emitter_link_names"),
        owner="source receipt emitter_link_names",
        expected_count=2,
    )
    if len(set(actor_ids)) != 2 or source_ids != expected_source_ids:
        raise MP3DProbeSubsetBuildError(
            "source receipt actor/source order does not match the expected two-source capture"
        )
    if not isinstance(inputs, Mapping):
        raise MP3DProbeSubsetBuildError("source receipt lacks declared M1/M2 inputs")
    receipt_m1_path = _declared_input_file(
        receipt_path,
        inputs.get("m1_request"),
        owner="source receipt inputs.m1_request",
    )
    receipt_m2_path = _declared_input_file(
        receipt_path,
        inputs.get("m2_request"),
        owner="source receipt inputs.m2_request",
    )
    if receipt_m2_path != source_m2_request_path:
        raise MP3DProbeSubsetBuildError(
            "source receipt M2 request differs from --source-m2-request"
        )
    receipt_m1 = load_json(receipt_m1_path)
    m1_errors = validate_capture_request(receipt_m1, room_id=expected_room_id)
    if m1_errors:
        raise MP3DProbeSubsetBuildError(
            "source receipt M1 request is invalid: " + "; ".join(m1_errors)
        )
    receipt_independent, receipt_request_id = _camera_independent_m1(receipt_m1)
    base_independent, base_request_id = _camera_independent_m1(base_request)
    if receipt_independent != base_independent:
        raise MP3DProbeSubsetBuildError(
            "source receipt M1 camera-independent content differs from base M1"
        )

    if not isinstance(artifacts, Mapping):
        raise MP3DProbeSubsetBuildError("source receipt lacks artifact declarations")
    receipt_source_positions_path = _contained_artifact_file(
        receipt_path,
        artifacts.get("source_positions_m"),
        owner="source receipt source_positions_m artifact",
    )
    frame_records_path = _contained_artifact_file(
        receipt_path,
        artifacts.get("frame_records"),
        owner="source receipt frame_records artifact",
    )
    if receipt_source_positions_path != source_positions_path:
        raise MP3DProbeSubsetBuildError(
            "source receipt does not bind the supplied source-position array"
        )
    supplied = np.asarray(source_positions, dtype=np.float64)
    if supplied.shape != (75, 2, 3) or not np.all(np.isfinite(supplied)):
        raise MP3DProbeSubsetBuildError(
            "supplied source positions must be finite [75,2,3]"
        )
    frame_positions = _frame_record_source_positions(load_json(frame_records_path))
    if not np.array_equal(frame_positions, supplied):
        raise MP3DProbeSubsetBuildError(
            "frame_records source positions differ from the supplied NPY array"
        )

    return {
        "status": "verified_current_native_habitat_emitter_link_readback",
        "receipt_status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "room_id": expected_room_id,
        "room_kind": "habitat_native",
        "native_habitat_started": True,
        "frame_count": 75,
        "actor_ids_in_array_axis_1_order": list(actor_ids),
        "source_ids_in_array_axis_1_order": list(source_ids),
        "emitter_link_names_in_array_axis_1_order": list(emitter_link_names),
        "receipt_declared_m1_request": str(receipt_m1_path),
        "receipt_m1_request_id": receipt_request_id,
        "base_m1_request_id": base_request_id,
        "m1_camera_independent_content_exact_match": True,
        "receipt_declared_m2_request": str(receipt_m2_path),
        "explicit_source_m2_request_exact_match": True,
        "source_positions_artifact": str(receipt_source_positions_path),
        "frame_records_artifact": str(frame_records_path),
        "artifacts_contained_by_receipt_root": True,
        "frame_records_source_positions_exact_match": True,
        "same_renderer_emitter_link_provenance_verified": True,
    }


def _nonempty_string_list(
    value: Any, *, owner: str, expected_count: int
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != expected_count
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise MP3DProbeSubsetBuildError(
            f"{owner} must contain {expected_count} nonempty strings"
        )
    return tuple(str(item) for item in value)


def _declared_input_file(receipt_path: Path, value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MP3DProbeSubsetBuildError(f"{owner} must name an ordinary file")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else receipt_path.parent / raw
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MP3DProbeSubsetBuildError(f"{owner} cannot be resolved") from exc
    if not resolved.is_file():
        raise MP3DProbeSubsetBuildError(f"{owner} is not an ordinary file")
    return resolved


def _contained_artifact_file(receipt_path: Path, value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise MP3DProbeSubsetBuildError(
            f"{owner} must be a relative path contained by the receipt root"
        )
    root = receipt_path.parent.resolve(strict=True)
    try:
        resolved = (root / value).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise MP3DProbeSubsetBuildError(
            f"{owner} escapes or is missing from the receipt root"
        ) from exc
    if not resolved.is_file():
        raise MP3DProbeSubsetBuildError(f"{owner} is not an ordinary file")
    return resolved


def _camera_independent_m1(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    result = deepcopy(dict(value))
    request_id = result.pop("request_id", None)
    try:
        result["primary_camera_rig"].pop("world_from_rig")
    except (KeyError, TypeError) as exc:
        raise MP3DProbeSubsetBuildError(
            "M1 request lacks the camera pose excluded from provenance comparison"
        ) from exc
    if not isinstance(request_id, str) or not request_id:
        raise MP3DProbeSubsetBuildError("M1 request_id must be nonempty")
    return result, request_id


def _frame_record_source_positions(value: Any) -> np.ndarray:
    frames = value.get("frames") if isinstance(value, Mapping) else None
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise MP3DProbeSubsetBuildError("frame_records must contain 75 frames")
    if len(frames) != 75:
        raise MP3DProbeSubsetBuildError("frame_records must contain 75 frames")
    positions = []
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != expected_index:
            raise MP3DProbeSubsetBuildError(
                "frame_records frame indices must be contiguous 0--74"
            )
        try:
            frame_positions = np.asarray(frame["source_positions_m"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MP3DProbeSubsetBuildError(
                "frame_records source_positions_m must be finite [2,3]"
            ) from exc
        if frame_positions.shape != (2, 3) or not np.all(np.isfinite(frame_positions)):
            raise MP3DProbeSubsetBuildError(
                "frame_records source_positions_m must be finite [2,3]"
            )
        positions.append(frame_positions)
    return np.asarray(positions, dtype=np.float64)


def _materialize_requests(
    staging: Path,
    *,
    analysis: MP3DProbeSubsetAnalysis,
    base_request: Mapping[str, Any],
) -> dict[str, str]:
    request_root = staging / "requests"
    request_root.mkdir()
    base_request_id = base_request.get("request_id")
    if not isinstance(base_request_id, str) or not base_request_id:
        raise MP3DProbeSubsetBuildError("base M1 request_id must be nonempty")
    result: dict[str, str] = {}
    for selected in analysis.selected:
        placement_id = str(selected["placement_id"])
        camera = selected["camera"]
        try:
            request = apply_camera_listener_pose(
                base_request,
                request_id=f"{base_request_id}_{placement_id}_native_probe_candidate",
                position_m=camera["position_m"],
                yaw_deg=camera["yaw_deg"],
            )
            topdown_views = [
                view
                for view in request.get("qa_views", [])
                if view.get("kind") == "topdown"
            ]
            if len(topdown_views) != 1:
                raise MP3DProbeSubsetBuildError(
                    "selected MP3D M1 request must contain one Topdown QA view"
                )
            topdown_views[0]["height_m"] = float(camera["floor_position_m"][1])
            errors = validate_capture_request(request, room_id=request.get("room_id"))
            if errors:
                raise MP3DProbeSubsetBuildError(
                    "selected MP3D M1 request is invalid after floor binding: "
                    + "; ".join(errors)
                )
        except (CameraPoseError, KeyError, TypeError, ValueError) as exc:
            raise MP3DProbeSubsetBuildError(
                f"could not author M1 request for {placement_id}: {exc}"
            ) from exc
        relative = Path("requests") / f"{placement_id}.json"
        write_json(staging / relative, request)
        result[placement_id] = relative.as_posix()
    return result


def _write_descriptors(
    path: Path,
    *,
    analysis: MP3DProbeSubsetAnalysis,
    base_request: Mapping[str, Any],
    source_positions: np.ndarray,
) -> None:
    selected = {str(item["placement_id"]): dict(item) for item in analysis.selected}
    with path.open("x", encoding="utf-8") as stream:
        for candidate in analysis.candidates:
            record = build_placement_descriptor(
                candidate,
                base_request=base_request,
                source_positions_m=source_positions,
            )
            if candidate.placement_id in selected:
                record["probe_selection"] = {
                    "selected": True,
                    **selected[candidate.placement_id],
                }
            else:
                record["probe_selection"] = {
                    "selected": False,
                    "reason": (
                        "retained in the full descriptor catalog but not among the "
                        "requested continuous top-ranked probe subset"
                    ),
                }
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def run(args: argparse.Namespace) -> Path:
    room_path = _required_file(args.room_manifest, owner="--room-manifest")
    request_path = _required_file(args.base_m1_request, owner="--base-m1-request")
    catalog_path = _required_file(args.camera_catalog, owner="--camera-catalog")
    positions_path = _required_file(args.source_positions, owner="--source-positions")
    receipt_path = _required_file(args.source_receipt, owner="--source-receipt")
    source_m2_path = _required_file(args.source_m2_request, owner="--source-m2-request")
    output, policy = _fresh_output(args.output)
    try:
        inputs = load_and_validate_inputs(room_path, request_path)
    except (OSError, ValueError) as exc:
        raise MP3DProbeSubsetBuildError(f"MP3D M1 inputs are invalid: {exc}") from exc
    room_error = _current_mp3d_room_error(inputs)
    if room_error is not None:
        raise MP3DProbeSubsetBuildError(room_error)
    room_id = str(inputs.room["room_id"])
    source_ids = _source_ids(inputs.request)
    try:
        source_positions = np.load(positions_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise MP3DProbeSubsetBuildError(
            f"could not read source-position array: {exc}"
        ) from exc
    source_provenance_verification = _validate_source_receipt(
        load_json(receipt_path),
        receipt_path=receipt_path,
        source_positions_path=positions_path,
        source_m2_request_path=source_m2_path,
        base_request=inputs.request,
        source_positions=source_positions,
        expected_room_id=room_id,
        expected_source_ids=source_ids,
    )
    try:
        analysis = analyze_mp3d_probe_candidates(
            catalog=load_json(catalog_path),
            base_request=inputs.request,
            source_positions_m=source_positions,
            selection_count=args.selection_count,
        )
    except MP3DProbeSubsetError as exc:
        raise MP3DProbeSubsetBuildError(str(exc)) from exc

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        request_paths = _materialize_requests(
            staging,
            analysis=analysis,
            base_request=inputs.request,
        )
        descriptor_path = staging / "placement_descriptors.jsonl"
        _write_descriptors(
            descriptor_path,
            analysis=analysis,
            base_request=inputs.request,
            source_positions=source_positions,
        )
        write_json(
            staging / "source_tracks.json",
            {
                "coordinate_space": "habitat_world_xyz_m",
                "frame_count": analysis.frame_count,
                "source_ids_in_array_axis_1_order": list(analysis.source_ids),
                "positions_m": np.asarray(source_positions, dtype=np.float64).tolist(),
                "track_provenance": (
                    "verified_same_habitat_renderer_emitter_link_readback"
                ),
                "provenance_verification": source_provenance_verification,
            },
        )
        summary = analysis.to_record()
        summary.update(
            {
                "inputs": {
                    "room_manifest": str(room_path),
                    "base_m1_request": str(request_path),
                    "camera_catalog": str(catalog_path),
                    "source_positions": str(positions_path),
                    "source_receipt": str(receipt_path),
                    "source_m2_request": str(source_m2_path),
                },
                "source_track_provenance_verification": (
                    source_provenance_verification
                ),
                "placement_descriptors_path": "placement_descriptors.jsonl",
                "source_tracks_path": "source_tracks.json",
                "m1_capture_request_paths": request_paths,
                "artifact_file_sizes_bytes": {
                    "placement_descriptors.jsonl": descriptor_path.stat().st_size,
                    "source_tracks.json": (staging / "source_tracks.json")
                    .stat()
                    .st_size,
                },
            }
        )
        for selected in summary["selected_candidates"]:
            selected["m1_capture_request_path"] = request_paths[
                selected["placement_id"]
            ]
        write_json(staging / "probe_subset.json", summary)
        (staging / "README.md").write_text(
            "# MP3D full-catalog probe subset\n\n"
            "`placement_descriptors.jsonl` retains every input legal placement and "
            "adds post-hoc 75-frame analytic source geometry. Descriptors do not "
            "change legality. `probe_subset.json` records a continuous ranking and "
            "the independent M1 requests for the selected native probes. Native "
            "Habitat pixels, occlusion, audio/RIR and certification remain pending. "
            "Same-renderer emitter-link provenance is written only after the native "
            "receipt, camera-independent M1 relation, explicit M2 path and all 75 "
            "frame records agree.\n",
            encoding="utf-8",
        )
        try:
            atomic_publish_directory(policy, staging, output)
        except (FileExistsError, PathPolicyError) as exc:
            raise MP3DProbeSubsetBuildError(str(exc)) from exc
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--base-m1-request", required=True, type=Path)
    parser.add_argument("--camera-catalog", required=True, type=Path)
    parser.add_argument("--source-positions", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-m2-request", required=True, type=Path)
    parser.add_argument("--selection-count", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
