#!/usr/bin/env python3
"""Render and retain variable-duration M5.1 binaural RIR evidence."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.acoustics.runtime import load_compiled_acoustic_scene
from avengine.spatial_audio.runtime import M4SimulationConfig
from avengine.capture.acoustics import (
    build_strided_review_keyframes,
    render_research_review_binaural_rir_sequence,
    research_review_trajectory_record,
)
from avengine.capture.delivery import (
    M51DeliveryError,
    source_actor_binding_record,
    source_binding_entries,
)
from avengine.capture.legacy_route import (
    ROUTE_SCHEMA as LEGACY_APARTMENT_ROUTE_SCHEMA,
    validate_route_manifest as validate_legacy_route_manifest,
)
from avengine.capture.source_contracts import load_source_manifest


SOURCE_IDS = (
    "source0",
    "source1",
)
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = (
    REPOSITORY / "examples/m5_1/legacy_apartment/source_manifest.json"
)


def _yaw_orientation_wxyz(yaw_degrees: float) -> tuple[float, float, float, float]:
    """Return a stable canonical +Y yaw quaternion for trajectory hashing."""

    if not math.isfinite(yaw_degrees):
        raise RuntimeError("listener yaw must be finite")
    half_yaw = math.radians(yaw_degrees) / 2.0
    values = [math.cos(half_yaw), 0.0, math.sin(half_yaw), 0.0]
    values = [0.0 if abs(value) < 1.0e-15 else float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise RuntimeError("listener yaw produced a zero quaternion")
    normalized = [value / norm for value in values]
    return (normalized[0], normalized[1], normalized[2], normalized[3])


def _root_relative_file_record(
    path: Path, *, root: Path, root_id: str
) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{resolved} escapes portable root {root_id}") from exc
    return {
        "root_id": root_id,
        "relative_path": relative.as_posix(),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _portable_file_record(path: Path, *, hrtf_root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        return _root_relative_file_record(
            resolved,
            root=REPOSITORY,
            root_id="AVENGINE_REPOSITORY_ROOT",
        )
    except RuntimeError:
        evidence_root = os.environ.get("AVENGINE_EVIDENCE_ROOT")
        if evidence_root:
            try:
                return _root_relative_file_record(
                    resolved,
                    root=Path(evidence_root),
                    root_id="AVENGINE_EVIDENCE_ROOT",
                )
            except RuntimeError:
                pass
        if hrtf_root is not None:
            return _root_relative_file_record(
                resolved,
                root=hrtf_root,
                root_id="AVENGINE_HRTF_ROOT",
            )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one persistent-context dynamic binaural RIR sequence from "
            "an M5.1 mixed-capture anchor array."
        )
    )
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
        help="Canonical source/event/audio manifest (defaults to M5.1 legacy source program)",
    )
    parser.add_argument("--acoustic-package-manifest", required=True, type=Path)
    parser.add_argument("--m4-request", required=True, type=Path)
    parser.add_argument("--hrtf", required=True, type=Path)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help=(
            "explicit Habitat runtime root; falls back to "
            "AVENGINE_HABITAT_RUNTIME_ROOT (sibling-checkout discovery is retired)"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--rir-stride-frames", type=int, default=3)
    parser.add_argument("--listener-position-m", nargs=3, type=float, required=True)
    parser.add_argument("--listener-yaw-deg", type=float, required=True)
    return parser.parse_args(argv)


def _save_npy(path: Path, value: np.ndarray, *, root: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value)
    np.save(path, array, allow_pickle=False)
    readback = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        readback.shape != array.shape
        or readback.dtype != array.dtype
        or not np.array_equal(readback, array)
    ):
        raise RuntimeError(f"retained array differs on readback: {path}")
    return {
        **file_record(path, relative_to=root),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "readback_verified": True,
    }


def _resolve_capture_input_record(record: Any, *, owner: str) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"capture {owner} record is missing")
    raw_path = record.get("path")
    if isinstance(raw_path, str):
        repository_token = "${AVENGINE_REPOSITORY_ROOT}"
        if raw_path == repository_token:
            path = REPOSITORY
        elif raw_path.startswith(f"{repository_token}/"):
            path = (REPOSITORY / raw_path[len(repository_token) + 1 :]).resolve()
            try:
                path.relative_to(REPOSITORY)
            except ValueError as exc:
                raise RuntimeError(
                    f"capture {owner} token escapes repository"
                ) from exc
        else:
            path = Path(raw_path).resolve()
    else:
        root_id = record.get("root_id")
        relative = record.get("relative_path")
        if root_id != "AVENGINE_REPOSITORY_ROOT" or not isinstance(relative, str):
            raise RuntimeError(f"capture {owner} locator is unsupported")
        path = (REPOSITORY / relative).resolve()
        try:
            path.relative_to(REPOSITORY)
        except ValueError as exc:
            raise RuntimeError(f"capture {owner} locator escapes repository") from exc
    if (
        not path.is_file()
        or path.stat().st_size != record.get("byte_size", path.stat().st_size)
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"capture {owner} bytes differ from evidence")
    return path


def _legacy_source_actor_binding_record(
    source_manifest: Mapping[str, Any],
    route: Mapping[str, Any],
    request: Mapping[str, Any],
    capture_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the frozen legacy capture's source IDs to retained mouth anchors.

    The first Apartment capture predates route-level semantic IDs and retained
    per-actor emitter-anchor indices.  Its evidence does, however, freeze the
    exact actor set, semantic IDs and three-entry anchor order.  This adapter
    accepts only that historical shape; modern MP3D and ReplicaCAD continue to
    use :func:`source_actor_binding_record` without this compatibility path.
    """

    sources_raw = source_manifest.get("sources")
    actors_raw = capture_evidence.get("actors")
    anchor_order = capture_evidence.get("anchor_order")
    if not isinstance(sources_raw, list) or not all(
        isinstance(item, Mapping) for item in sources_raw
    ):
        raise RuntimeError("legacy source manifest sources must be an object array")
    if not isinstance(actors_raw, list) or not all(
        isinstance(item, Mapping) for item in actors_raw
    ):
        raise RuntimeError("legacy capture actors must be an object array")
    expected_anchor_order = [
        "human0.head",
        "human0.mouth_emitter",
        "dog0.mouth_emitter",
    ]
    if anchor_order != expected_anchor_order:
        raise RuntimeError("legacy capture anchor_order differs from the frozen layout")

    sources = {str(item.get("source_id")): item for item in sources_raw}
    actors = {str(item.get("actor_id")): item for item in actors_raw}
    if len(sources) != len(sources_raw) or set(sources) != {"source0", "source1"}:
        raise RuntimeError("legacy source manifest requires exactly source0/source1")
    if len(actors) != len(actors_raw) or set(actors) != {"human0", "dog0"}:
        raise RuntimeError("legacy capture requires exactly human0/dog0")

    expected = {
        "source0": {
            "actor_id": "human0",
            "asset_class": "human",
            "actor_class": "human",
            "semantic_id": 220,
            "anchor_index": 1,
        },
        "source1": {
            "actor_id": "dog0",
            "asset_class": "animal",
            "actor_class": "dog",
            "semantic_id": 221,
            "anchor_index": 2,
        },
    }
    bindings: dict[str, dict[str, Any]] = {}
    for source_id in ("source0", "source1"):
        source = sources[source_id]
        contract = expected[source_id]
        actor_id = str(contract["actor_id"])
        actor = actors[actor_id]
        emitter = source.get("emitter")
        if not isinstance(emitter, Mapping):
            raise RuntimeError(f"legacy {source_id} lacks an emitter declaration")
        if (
            source.get("asset_class") != contract["asset_class"]
            or actor.get("actor_class") != contract["actor_class"]
            or actor.get("semantic_id") != contract["semantic_id"]
        ):
            raise RuntimeError(f"legacy {source_id}/{actor_id} identity differs")
        link_name = emitter.get("link_name")
        if not isinstance(link_name, str) or link_name != actor.get("emitter_link"):
            raise RuntimeError(f"legacy {source_id} emitter link differs from {actor_id}")
        semantic_anchor_id = emitter.get("semantic_anchor_id")
        emitter_id = emitter.get("emitter_id")
        if (
            not isinstance(semantic_anchor_id, str)
            or not semantic_anchor_id
            or not isinstance(emitter_id, str)
            or not emitter_id
        ):
            raise RuntimeError(f"legacy {source_id} emitter identity is invalid")
        anchor_index = int(contract["anchor_index"])
        capture_anchor_id = expected_anchor_order[anchor_index]
        binding: dict[str, Any] = {
            "source_id": source_id,
            "actor_id": actor_id,
            "asset_class": source["asset_class"],
            "actor_class": actor["actor_class"],
            "emitter_id": emitter_id,
            "emitter_anchor_id": semantic_anchor_id,
            "capture_anchor_id": capture_anchor_id,
            "emitter_link_name": link_name,
            "emitter_anchor_index": anchor_index,
            "anchor_index_authority": (
                "legacy_apartment_capture_v1_frozen_anchor_order_adapter"
            ),
            "semantic_id": int(contract["semantic_id"]),
            "position_authority": (
                "captured_articulated_emitter_link_world_transform"
            ),
        }
        binding["binding_content_sha256"] = canonical_json_sha256(binding)
        bindings[source_id] = binding

    record: dict[str, Any] = {
        "schema": "avengine_m5_1_source_actor_binding_v1",
        "room_family": "legacy_apartment",
        "route_id": route.get("route_id"),
        "room_id": request.get("room_id"),
        "source_ids": ["source0", "source1"],
        "bindings": bindings,
    }
    record["record_content_sha256"] = canonical_json_sha256(record)
    # Exercise the shared strict reader before the record reaches RIR code.
    source_binding_entries(record)
    return record


def _capture_contract(
    capture_evidence: Mapping[str, Any],
    *,
    capture_evidence_path: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    if (
        capture_evidence.get("status") != "pass"
        or capture_evidence.get("qualification_claim") is not False
    ):
        raise RuntimeError("capture evidence is not a bounded pass")
    content = dict(capture_evidence)
    capture_content_sha256 = content.pop("evidence_content_sha256", None)
    if (
        not isinstance(capture_content_sha256, str)
        or canonical_json_sha256(content) != capture_content_sha256
    ):
        raise RuntimeError("capture evidence content identity differs")
    inputs = capture_evidence.get("inputs")
    route_provenance = (
        inputs.get("route_provenance") if isinstance(inputs, Mapping) else None
    )
    route_record = None
    legacy_route_locator_adapter = False
    if isinstance(route_provenance, Mapping):
        modern_route_record = route_provenance.get("route_manifest")
        if modern_route_record is not None:
            route_record = modern_route_record
        elif route_provenance.get("route_manifest_path") is not None:
            # Narrow adapter for the retained first-generation Apartment
            # capture.  Do not fall back to these fields when a modern record
            # is present but malformed.
            route_record = {
                "path": route_provenance.get("route_manifest_path"),
                "sha256": route_provenance.get("route_manifest_sha256"),
            }
            legacy_route_locator_adapter = True
    request_record = inputs.get("m1_request") if isinstance(inputs, Mapping) else None
    room_record = inputs.get("room_manifest") if isinstance(inputs, Mapping) else None
    route_path = _resolve_capture_input_record(route_record, owner="route manifest")
    request_path = _resolve_capture_input_record(request_record, owner="M1 request")
    room_path = _resolve_capture_input_record(room_record, owner="room manifest")
    route = load_json(route_path)
    request = load_json(request_path)
    schema = route.get("schema")
    if schema == "avengine_m5_1_mp3d_center_route_v1":
        if legacy_route_locator_adapter:
            raise RuntimeError("modern MP3D capture lacks a structured route record")
        room_family = "mp3d"
        capture_geometry_sha256 = None
    elif schema == "avengine_m5_1_replicacad_center_route_v1":
        if legacy_route_locator_adapter:
            raise RuntimeError(
                "modern ReplicaCAD capture lacks a structured route record"
            )
        room_family = "replicacad"
        preflight = route_provenance.get("real_replicacad_preflight")
        selected_scene = (
            preflight.get("selected_scene") if isinstance(preflight, Mapping) else None
        )
        stage_surface = (
            selected_scene.get("stage_surface")
            if isinstance(selected_scene, Mapping)
            else None
        )
        capture_geometry_sha256 = (
            stage_surface.get("sha256")
            if isinstance(stage_surface, Mapping)
            else None
        )
        if not isinstance(capture_geometry_sha256, str):
            raise RuntimeError("ReplicaCAD capture lacks stage-surface identity")
    elif schema == LEGACY_APARTMENT_ROUTE_SCHEMA:
        if not legacy_route_locator_adapter:
            raise RuntimeError(
                "legacy Apartment capture lacks its frozen route locator"
            )
        room_family = "legacy_apartment"
        capture_geometry_sha256 = None
        route_errors = validate_legacy_route_manifest(route)
        if route_errors:
            raise RuntimeError(
                "legacy Apartment route validation failed: " + "; ".join(route_errors)
            )
    else:
        raise RuntimeError("capture route has an unsupported room-family schema")
    if not isinstance(route_provenance, Mapping) or route_provenance.get(
        "route_id"
    ) != route.get("route_id"):
        raise RuntimeError("capture route logical identity differs")

    room = load_json(room_path)
    if room_family == "legacy_apartment":
        if (
            route_provenance.get("path_consumption")
            != "verbatim_manifest_routes_habitat_trajectory_m"
            or route_provenance.get("human_habitat_trajectory_sha256")
            != route.get("routes", {}).get("human_path", {}).get(
                "habitat_trajectory_sha256"
            )
            or route_provenance.get("dog_habitat_trajectory_sha256")
            != route.get("routes", {}).get("dog_path", {}).get(
                "habitat_trajectory_sha256"
            )
        ):
            raise RuntimeError("legacy Apartment route provenance differs")
        if (
            request.get("schema") != "avengine_m1_capture_request_v1"
            or not isinstance(request.get("request_id"), str)
            or not request.get("request_id")
            or request.get("room_id") != room.get("room_id")
        ):
            raise RuntimeError("legacy Apartment request/room identity differs")
        route_camera = route.get("camera")
        request_rig = request.get("primary_camera_rig")
        request_transform = (
            request_rig.get("world_from_rig")
            if isinstance(request_rig, Mapping)
            else None
        )
        request_calibration = (
            request_rig.get("shared_calibration")
            if isinstance(request_rig, Mapping)
            else None
        )
        if (
            not isinstance(route_camera, Mapping)
            or not isinstance(request_transform, Mapping)
            or not isinstance(request_calibration, Mapping)
            or not np.allclose(
                np.asarray(route_camera.get("habitat_position_m"), dtype=np.float64),
                np.asarray(request_transform.get("translation_m"), dtype=np.float64),
                atol=1.0e-12,
                rtol=0.0,
            )
            or route_camera.get("horizontal_fov_deg")
            != request_calibration.get("hfov_degrees")
        ):
            raise RuntimeError("legacy Apartment route camera differs from M1 request")
        expected_wxyz = np.asarray(
            _yaw_orientation_wxyz(float(route_camera["habitat_yaw_deg"])),
            dtype=np.float64,
        )
        request_xyzw = np.asarray(
            request_transform.get("rotation_xyzw"), dtype=np.float64
        )
        if (
            request_xyzw.shape != (4,)
            or not np.all(np.isfinite(request_xyzw))
            or float(np.linalg.norm(request_xyzw)) <= 0.0
        ):
            raise RuntimeError("legacy Apartment M1 request rotation is invalid")
        request_wxyz = request_xyzw[[3, 0, 1, 2]]
        request_wxyz /= np.linalg.norm(request_wxyz)
        if not (
            np.allclose(request_wxyz, expected_wxyz, atol=1.0e-12, rtol=0.0)
            or np.allclose(request_wxyz, -expected_wxyz, atol=1.0e-12, rtol=0.0)
        ):
            raise RuntimeError("legacy Apartment route yaw differs from M1 request")
        captured_camera = capture_evidence.get("camera")
        if (
            not isinstance(captured_camera, Mapping)
            or not np.allclose(
                np.asarray(captured_camera.get("position_m"), dtype=np.float64),
                np.asarray(request_transform.get("translation_m"), dtype=np.float64),
                atol=1.0e-12,
                rtol=0.0,
            )
            or not np.allclose(
                np.asarray(captured_camera.get("rotation_xyzw"), dtype=np.float64),
                request_xyzw,
                atol=1.0e-12,
                rtol=0.0,
            )
            or captured_camera.get("horizontal_fov_deg")
            != request_calibration.get("hfov_degrees")
        ):
            raise RuntimeError("legacy Apartment capture camera differs from M1 request")
    else:
        if (
            request.get("request_id") != route.get("request_id")
            or request.get("room_id") != route.get("room_id")
        ):
            raise RuntimeError("capture route/request logical identity differs")
        for field in ("route_id", "request_id", "room_id"):
            captured = capture_evidence.get(field)
            if captured is None and room_family == "mp3d":
                continue
            if captured != route.get(field):
                raise RuntimeError(f"capture {field} differs from retained route")
    source = load_source_manifest(source_manifest_path)
    if room_family == "legacy_apartment":
        bindings = _legacy_source_actor_binding_record(
            source, route, request, capture_evidence
        )
    else:
        try:
            bindings = source_actor_binding_record(
                source,
                route,
                capture_evidence,
                room_family=room_family,
            )
        except M51DeliveryError as exc:
            raise RuntimeError(f"capture source/actor binding failed: {exc}") from exc
    rig = request.get("primary_camera_rig")
    transform = rig.get("world_from_rig") if isinstance(rig, Mapping) else None
    listener_position = (
        transform.get("translation_m") if isinstance(transform, Mapping) else None
    )
    listener_rotation = (
        transform.get("rotation_xyzw") if isinstance(transform, Mapping) else None
    )
    listener_position_array = np.asarray(listener_position, dtype=np.float64)
    listener_rotation_array = np.asarray(listener_rotation, dtype=np.float64)
    if (
        listener_position_array.shape != (3,)
        or listener_rotation_array.shape != (4,)
        or not np.all(np.isfinite(listener_position_array))
        or not np.all(np.isfinite(listener_rotation_array))
        or float(np.linalg.norm(listener_rotation_array)) <= 0.0
    ):
        raise RuntimeError("capture M1 request lacks a camera/listener transform")
    room_id = request["room_id"] if room_family == "legacy_apartment" else route["room_id"]
    request_id = (
        request["request_id"]
        if room_family == "legacy_apartment"
        else route["request_id"]
    )
    return {
        "room_family": room_family,
        "room_id": room_id,
        "route_id": route["route_id"],
        "request_id": request_id,
        "capture_evidence_sha256": sha256_file(capture_evidence_path),
        "capture_content_sha256": capture_content_sha256,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_actor_binding_content_sha256": bindings[
            "record_content_sha256"
        ],
        "source_actor_bindings": bindings,
        "route_manifest": _portable_file_record(route_path),
        "m1_request": _portable_file_record(request_path),
        "room_manifest": _portable_file_record(room_path),
        "room_manifest_sha256": sha256_file(room_path),
        "capture_geometry_sha256": capture_geometry_sha256,
        "listener_position_m": list(listener_position),
        "listener_rotation_xyzw": list(listener_rotation),
    }


def _capture_array(
    capture: Path, capture_evidence: Mapping[str, Any], *, role: str
) -> np.ndarray:
    artifacts = capture_evidence.get("array_artifacts")
    record = artifacts.get(role) if isinstance(artifacts, Mapping) else None
    relative = record.get("path") if isinstance(record, Mapping) else None
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise RuntimeError(f"capture array record is invalid: {role}")
    path = (capture / relative).resolve()
    try:
        path.relative_to(capture)
    except ValueError as exc:
        raise RuntimeError(f"capture array escapes bundle: {role}") from exc
    if (
        not path.is_file()
        or path.stat().st_size != record.get("byte_size")
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"capture array bytes differ from evidence: {role}")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != record.get("shape") or array.dtype.str != record.get(
        "dtype"
    ):
        raise RuntimeError(f"capture array metadata differs from evidence: {role}")
    return np.ascontiguousarray(array)


def _portableize_paths(value: Any, roots: Sequence[tuple[str, Path]]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _portableize_paths(item, roots) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portableize_paths(item, roots) for item in value]
    if isinstance(value, tuple):
        return [_portableize_paths(item, roots) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value).resolve()
        for root_id, root in roots:
            try:
                relative = path.relative_to(root.resolve())
            except ValueError:
                continue
            suffix = relative.as_posix()
            return f"${{{root_id}}}/{suffix}" if suffix != "." else f"${{{root_id}}}"
        raise RuntimeError(f"RIR metadata contains an undeclared absolute path: {path}")
    return value


def _resolve_runtime_root(
    explicit: Path | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the runtime root without requiring one workspace layout."""

    environment = os.environ if environ is None else environ
    raw_environment = environment.get("AVENGINE_HABITAT_RUNTIME_ROOT")
    if explicit is not None:
        candidate = explicit.resolve()
        authority = "--runtime-root"
    elif isinstance(raw_environment, str) and raw_environment:
        candidate = Path(raw_environment).expanduser().resolve()
        authority = "AVENGINE_HABITAT_RUNTIME_ROOT"
    else:
        raise RuntimeError(
            "runtime root is required: pass --runtime-root or set "
            "AVENGINE_HABITAT_RUNTIME_ROOT; implicit sibling checkout "
            "discovery is retired"
        )
    if not candidate.is_dir():
        raise RuntimeError(f"{authority} is not a readable runtime directory: {candidate}")
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    capture = args.capture_dir.resolve()
    source_manifest_path = args.source_manifest.resolve()
    acoustic_manifest = args.acoustic_package_manifest.resolve()
    m4_request_path = args.m4_request.resolve()
    hrtf = args.hrtf.resolve()
    runtime_root = _resolve_runtime_root(args.runtime_root)
    destination = args.output_dir.resolve()
    staging = destination.with_name(f".{destination.name}.staging")
    if os.path.lexists(destination) or os.path.lexists(staging):
        raise RuntimeError(
            f"refusing to replace existing output or staging path: {destination}"
        )
    required = (
        capture / "arrays" / "anchor_positions_m.npy",
        capture / "evidence.json",
        source_manifest_path,
        acoustic_manifest,
        m4_request_path,
        hrtf,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing M5.1 acoustic input(s): {missing}")

    capture_evidence = load_json(capture / "evidence.json")
    capture_contract = _capture_contract(
        capture_evidence,
        capture_evidence_path=capture / "evidence.json",
        source_manifest_path=source_manifest_path,
    )

    anchors = _capture_array(capture, capture_evidence, role="anchor_positions_m")
    if (
        anchors.ndim != 3
        or anchors.shape[1:] != (3, 3)
        or anchors.shape[0] < 1
        or not np.all(np.isfinite(anchors))
    ):
        raise RuntimeError("capture anchors must be finite [frame,3,3]")
    if args.fps <= 0 or args.rir_stride_frames <= 0:
        raise RuntimeError("fps and RIR stride must be positive")
    listener = np.asarray(args.listener_position_m, dtype=np.float64)
    if listener.shape != (3,) or not np.all(np.isfinite(listener)):
        raise RuntimeError("listener position must be a finite length-3 vector")
    orientation_wxyz = _yaw_orientation_wxyz(args.listener_yaw_deg)
    declared_listener = np.asarray(
        capture_contract["listener_position_m"], dtype=np.float64
    )
    declared_xyzw = np.asarray(
        capture_contract["listener_rotation_xyzw"], dtype=np.float64
    )
    declared_xyzw /= np.linalg.norm(declared_xyzw)
    declared_wxyz = np.asarray(
        (declared_xyzw[3], declared_xyzw[0], declared_xyzw[1], declared_xyzw[2]),
        dtype=np.float64,
    )
    observed_wxyz = np.asarray(orientation_wxyz, dtype=np.float64)
    orientation_matches = np.allclose(
        observed_wxyz, declared_wxyz, atol=1.0e-10, rtol=0.0
    ) or np.allclose(observed_wxyz, -declared_wxyz, atol=1.0e-10, rtol=0.0)
    if not np.allclose(
        listener, declared_listener, atol=1.0e-10, rtol=0.0
    ) or not orientation_matches:
        raise RuntimeError("RIR listener differs from the M1 camera/listener rig")
    binding_entries = source_binding_entries(
        capture_contract["source_actor_bindings"]
    )
    trajectories = {}
    for source_id in SOURCE_IDS:
        binding = binding_entries[source_id]
        anchor_index = int(binding["emitter_anchor_index"])
        if not 0 <= anchor_index < anchors.shape[1]:
            raise RuntimeError(f"{source_id} emitter anchor index escapes capture")
        trajectories[source_id] = np.ascontiguousarray(anchors[:, anchor_index, :])
    grid = build_strided_review_keyframes(
        trajectories,
        visual_frame_rate_hz=args.fps,
        rir_stride_frames=args.rir_stride_frames,
        listener_position_m=listener,
        listener_orientation_wxyz=orientation_wxyz,
    )
    scene = load_compiled_acoustic_scene(
        acoustic_manifest,
        allow_nonpassing_research_qa=True,
    )
    source_room = scene.manifest.get("source_room")
    if (
        not isinstance(source_room, Mapping)
        or source_room.get("room_id") != capture_contract["room_id"]
        or source_room.get("manifest_sha256")
        != capture_contract["room_manifest_sha256"]
        or (
            capture_contract["capture_geometry_sha256"] is not None
            and source_room.get("geometry_asset_sha256")
            != capture_contract["capture_geometry_sha256"]
        )
    ):
        raise RuntimeError(
            "acoustic package source_room differs from capture room manifest"
        )
    request = load_json(m4_request_path)
    simulation = M4SimulationConfig.from_mapping(request["simulation"])

    staging.mkdir(parents=True)
    try:
        sequence = render_research_review_binaural_rir_sequence(
            scene,
            simulation,
            grid=grid,
            hrtf_file_path=str(hrtf),
        )
        trajectory = research_review_trajectory_record(grid)
        trajectory["trajectory_content_sha256"] = canonical_json_sha256(
            trajectory
        )
        trajectory_path = staging / "trajectory.json"
        metadata_path = staging / "rir" / "metadata.json"
        portability_roots: list[tuple[str, Path]] = [
            ("AVENGINE_REPOSITORY_ROOT", REPOSITORY),
            ("AVENGINE_HABITAT_RUNTIME_ROOT", runtime_root),
            ("AVENGINE_PYTHON_ENV_ROOT", Path(sys.prefix)),
            ("AVENGINE_HRTF_ROOT", hrtf.parent),
        ]
        portable_metadata = _portableize_paths(
            dict(sequence.metadata), portability_roots
        )
        portable_metadata["source_room"] = dict(source_room)
        portable_metadata["capture_binding"] = {
            key: capture_contract[key]
            for key in (
                "room_id",
                "route_id",
                "request_id",
                "capture_evidence_sha256",
                "capture_content_sha256",
                "source_manifest_sha256",
                "source_actor_binding_content_sha256",
            )
        }
        portable_metadata["path_roots"] = {
            root_id: {
                "locator_kind": "environment_or_workspace_root",
                "absolute_path_recorded": False,
            }
            for root_id, _root in portability_roots
        }
        write_json(trajectory_path, trajectory)
        write_json(metadata_path, portable_metadata)
        artifacts = {
            "rir_samples": _save_npy(
                staging / "rir" / "samples.npy", sequence.samples, root=staging
            ),
            "rir_lengths": _save_npy(
                staging / "rir" / "lengths.npy", sequence.lengths, root=staging
            ),
            "trajectory": file_record(trajectory_path, relative_to=staging),
            "rir_metadata": file_record(metadata_path, relative_to=staging),
        }
        evidence: dict[str, Any] = {
            "schema": "avengine_m5_1_dynamic_binaural_rir_evidence_v1",
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "claim_boundary": (
                "M5.1 dynamic research-review binaural RIR sequence for "
                f"room {scene.manifest.get('source_room', {}).get('room_id')}; no "
                "room, material, asset, episode, or dataset admission claim"
            ),
            "source_room": dict(source_room),
            "capture_binding": portable_metadata["capture_binding"],
            "source_actor_bindings": capture_contract["source_actor_bindings"],
            "capture_evidence": _portable_file_record(capture / "evidence.json"),
            "source_manifest": _portable_file_record(source_manifest_path),
            "route_manifest": capture_contract["route_manifest"],
            "m1_request": capture_contract["m1_request"],
            "room_manifest": capture_contract["room_manifest"],
            "acoustic_package_manifest": _portable_file_record(acoustic_manifest),
            "acoustic_package_identity": {
                "package_id": scene.manifest.get("package_id"),
                "package_content_sha256": scene.manifest.get(
                    "package_content_sha256"
                ),
                "manifest_sha256": sha256_file(acoustic_manifest),
                "source_room": dict(source_room),
            },
            "acoustic_package_gate": {
                "load_policy": "explicit_nonpassing_research_qa_review_only",
                "package_mode": scene.manifest.get("package_mode"),
                "material_semantics": scene.manifest.get("materials", {}).get(
                    "material_semantics"
                ),
                "qualification_claim": scene.manifest.get("materials", {}).get(
                    "qualification_claim"
                ),
                "qa_status_by_report": {
                    name: report.get("status")
                    for name, report in sorted(scene.qa_reports.items())
                },
            },
            "m4_request": _portable_file_record(m4_request_path),
            "hrtf": _portable_file_record(hrtf, hrtf_root=hrtf.parent),
            "listener": {
                "position_m": listener.tolist(),
                "yaw_deg": args.listener_yaw_deg,
                "orientation_wxyz": list(orientation_wxyz),
                "motion": "fixed",
            },
            "source_ids": list(sequence.source_ids),
            "capture_anchor_indices": {
                source_id: int(binding_entries[source_id]["emitter_anchor_index"])
                for source_id in SOURCE_IDS
            },
            "path_roots": portable_metadata["path_roots"],
            "visual_frame_count": int(anchors.shape[0]),
            "visual_frame_rate_hz": args.fps,
            "rir_stride_frames": args.rir_stride_frames,
            "rir_keyframe_count": len(grid.keyframes),
            "sample_rate_hz": sequence.sample_rate_hz,
            "episode_sample_count": grid.episode_sample_count,
            "trajectory_sha256": sequence.trajectory_sha256,
            "layout": {
                "type": sequence.layout_type,
                "layout_id": sequence.layout_id,
                "channel_labels": list(sequence.channel_labels),
            },
            "artifacts": artifacts,
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        write_json(staging / "evidence.json", evidence)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(destination / "evidence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
