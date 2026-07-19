"""Capture-provider boundary for the fixed-Apartment M6.x canary.

The S0--S5 runner owns room/source/audio logic.  This module owns the exact
visual-capture layout used by the current human + Beagle provider: actor array
slots, retained record paths, semantic IDs, and source-endpoint-to-capture-
anchor routing.  A different articulated pair can therefore supply another
adapter without adding species branches to the room or scenario code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, load_json, sha256_file
from avengine.m5_1.mixed_capture import capture_human_beagle_paths
from avengine.m6x.trajectory import materialize_template_route


MASTER_FRAME_COUNT = 270
FRAME_RATE_HZ = 15
TIME_BASE_HZ = 48_000


class CaptureAdapterError(RuntimeError):
    """A provider result does not satisfy its declared capture adapter."""


@dataclass(frozen=True)
class CaptureData:
    """Arrays and per-frame readback shared by M6.x capture adapters."""

    root: Path
    rgb: np.ndarray
    semantic: np.ndarray
    actor_world_matrices: np.ndarray
    anchor_positions_m: np.ndarray
    records: tuple[Mapping[str, Any], ...]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class ActorCaptureBinding:
    """Timeline and retained-array binding for one visible actor."""

    actor_id: str
    asset_id: str
    template_id: str
    body_plan_id: str
    semantic_id: int | None
    capture_matrix_index: int | None
    pose_hash_record_path: tuple[str, ...] | None
    action_id: str
    action_phase_period_frames: int
    trajectory_template_id: str | None = None
    trajectory_route_id: str | None = None
    always_include_in_timeline: bool = False

    def timeline_record(self) -> dict[str, str]:
        return {
            "actor_id": self.actor_id,
            "asset_id": self.asset_id,
            "template_id": self.template_id,
            "body_plan_id": self.body_plan_id,
        }


@dataclass(frozen=True)
class SourceCaptureBinding:
    """Data-only routing from a stable endpoint to capture/room evidence."""

    source_endpoint_id: str
    actor_id: str
    emitter_anchor_id: str
    capture_anchor_id: str | None
    room_anchor_id: str | None
    semantic_id: int | None
    position_authority: str
    label: str
    sound_class: str
    color_rgb: tuple[int, int, int]
    asset_class: str


@dataclass(frozen=True)
class CaptureActorState:
    """Adapter-normalized actor state consumed by Timeline v2 assembly."""

    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    action_id: str
    action_phase: float
    pose_hash: str


class FixedApartmentCaptureAdapter(Protocol):
    """Minimal provider interface required by the fixed-room scenario runner."""

    adapter_id: str
    capture_schema: str
    capture_anchor_order: tuple[str, ...]

    def actor_binding(self, actor_id: str) -> ActorCaptureBinding: ...

    def source_binding(self, source_endpoint_id: str) -> SourceCaptureBinding: ...

    def validate_registry_bindings(
        self, endpoint_registry: Mapping[str, Any]
    ) -> list[str]: ...

    def materialize_actor_root_paths(
        self,
        trajectory_templates: Mapping[str, Any],
        anchor_library: Mapping[str, Any],
    ) -> dict[str, np.ndarray]: ...

    def provisional_source_paths(
        self,
        anchor_library: Mapping[str, Any],
        actor_root_paths: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]: ...

    def actual_source_paths(
        self,
        anchor_library: Mapping[str, Any],
        capture: CaptureData,
    ) -> dict[str, np.ndarray]: ...

    def timeline_actor_ids(
        self, candidate_actor_ids: Sequence[str]
    ) -> tuple[str, ...]: ...

    def timeline_actor_state(
        self,
        capture: CaptureData,
        *,
        actor_id: str,
        master_frame: int,
        local_frame: int,
        source_endpoint_id: str | None,
        trajectories: Mapping[str, np.ndarray],
    ) -> CaptureActorState: ...

    def validate_capture_reuse_contract(
        self,
        evidence: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
    ) -> None: ...

    def load_capture(
        self,
        path: str | Path,
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
    ) -> CaptureData: ...

    def capture(
        self,
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
        provider_assets: Mapping[str, str | Path],
        actor_root_paths: Mapping[str, np.ndarray],
        output_dir: str | Path,
        runtime_root: str | Path | None,
        route_provenance: Mapping[str, Any],
    ) -> CaptureData: ...


def _nested_record_value(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise CaptureAdapterError(
                f"capture record is missing provider field {'.'.join(path)!r}"
            )
        value = value[key]
    return value


def _matrix_quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise CaptureAdapterError("actor rotation matrix is invalid")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = (
                math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            )
            values = np.asarray(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = (
                math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            )
            values = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = (
                math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            )
            values = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12 or not math.isfinite(norm):
        raise CaptureAdapterError("actor rotation quaternion is invalid")
    values /= norm
    if values[3] < 0.0:
        values *= -1.0
    normalized = tuple(
        0.0 if abs(float(value)) < 1.0e-15 else float(value) for value in values
    )
    return normalized  # type: ignore[return-value]


def _anchor_positions(anchor_library: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        item["anchor_id"]: np.asarray(item["position_m"], dtype=np.float64)
        for item in anchor_library["anchors"]
    }


def _static_path(point: Sequence[float]) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise CaptureAdapterError("static source position is invalid")
    return np.ascontiguousarray(np.repeat(value[None, :], MASTER_FRAME_COUNT, axis=0))


@dataclass(frozen=True)
class HumanBeagleCaptureAdapter:
    """Adapter for the retained M5.1 Rocketbox-human + Beagle provider."""

    adapter_id: str = "m5_1_human_beagle_capture_adapter_v1"
    capture_schema: str = "avengine_m5_1_human_beagle_capture_v1"
    capture_anchor_order: tuple[str, ...] = (
        "human0.head",
        "human0.mouth_emitter",
        "dog0.mouth_emitter",
    )
    actor_bindings: tuple[ActorCaptureBinding, ...] = (
        ActorCaptureBinding(
            actor_id="dog0",
            asset_id="rocketbox_dog_beagle_01_m2_v7_world_contact_candidate",
            template_id="rocketbox_dog_beagle_01",
            body_plan_id="quadruped_canine",
            semantic_id=221,
            capture_matrix_index=1,
            pose_hash_record_path=("beagle", "readback", "state_sha256"),
            action_id="walk",
            action_phase_period_frames=45,
            trajectory_template_id="dog_master_motion_270",
            trajectory_route_id="dog_master_route",
            always_include_in_timeline=True,
        ),
        ActorCaptureBinding(
            actor_id="human0",
            asset_id="rocketbox_human_male_adult_01_m5_1_candidate",
            template_id="rocketbox_human_male_adult_01",
            body_plan_id="biped_human",
            semantic_id=220,
            capture_matrix_index=0,
            pose_hash_record_path=("human", "pose_sha256"),
            action_id="walk",
            action_phase_period_frames=16,
            trajectory_template_id="human_master_motion_270",
            trajectory_route_id="human_master_route",
            always_include_in_timeline=True,
        ),
        ActorCaptureBinding(
            actor_id="marker_front",
            asset_id="legacy_apartment_source_marker_front_v1",
            template_id="rigid_source_marker",
            body_plan_id="rigid_object",
            semantic_id=101,
            capture_matrix_index=None,
            pose_hash_record_path=None,
            action_id="static",
            action_phase_period_frames=1,
        ),
        ActorCaptureBinding(
            actor_id="marker_rear",
            asset_id="legacy_apartment_source_marker_rear_v1",
            template_id="rigid_source_marker",
            body_plan_id="rigid_object",
            semantic_id=102,
            capture_matrix_index=None,
            pose_hash_record_path=None,
            action_id="static",
            action_phase_period_frames=1,
        ),
        ActorCaptureBinding(
            actor_id="spear_apartment_los_point",
            asset_id="world_source_point",
            template_id="world_source_point",
            body_plan_id="environmental_source",
            semantic_id=None,
            capture_matrix_index=None,
            pose_hash_record_path=None,
            action_id="static",
            action_phase_period_frames=1,
        ),
        ActorCaptureBinding(
            actor_id="spear_apartment_nlos_point",
            asset_id="world_source_point",
            template_id="world_source_point",
            body_plan_id="environmental_source",
            semantic_id=None,
            capture_matrix_index=None,
            pose_hash_record_path=None,
            action_id="static",
            action_phase_period_frames=1,
        ),
    )
    source_bindings: tuple[SourceCaptureBinding, ...] = (
        SourceCaptureBinding(
            source_endpoint_id="m6x_dog0_muzzle",
            actor_id="dog0",
            emitter_anchor_id="muzzle",
            capture_anchor_id="dog0.mouth_emitter",
            room_anchor_id=None,
            semantic_id=221,
            position_authority="captured_articulated_emitter_link_world_transform",
            label="Beagle",
            sound_class="animal_vocalization",
            color_rgb=(250, 120, 70),
            asset_class="dog",
        ),
        SourceCaptureBinding(
            source_endpoint_id="m6x_human0_mouth",
            actor_id="human0",
            emitter_anchor_id="mouth",
            capture_anchor_id="human0.mouth_emitter",
            room_anchor_id=None,
            semantic_id=220,
            position_authority="captured_articulated_emitter_link_world_transform",
            label="Human",
            sound_class="human_speech",
            color_rgb=(42, 210, 220),
            asset_class="human",
        ),
        SourceCaptureBinding(
            source_endpoint_id="m6x_marker_front_speaker",
            actor_id="marker_front",
            emitter_anchor_id="speaker",
            capture_anchor_id=None,
            room_anchor_id="marker_front",
            semantic_id=101,
            position_authority="fixed_anchor_world_position",
            label="Front marker",
            sound_class="test_signal",
            color_rgb=(255, 90, 70),
            asset_class="rigid_object",
        ),
        SourceCaptureBinding(
            source_endpoint_id="m6x_marker_rear_speaker",
            actor_id="marker_rear",
            emitter_anchor_id="speaker",
            capture_anchor_id=None,
            room_anchor_id="marker_rear",
            semantic_id=102,
            position_authority="fixed_anchor_world_position",
            label="Rear marker",
            sound_class="test_signal",
            color_rgb=(89, 156, 255),
            asset_class="rigid_object",
        ),
        SourceCaptureBinding(
            source_endpoint_id="m6x_world_los_speaker",
            actor_id="spear_apartment_los_point",
            emitter_anchor_id="speaker",
            capture_anchor_id=None,
            room_anchor_id="world_los",
            semantic_id=None,
            position_authority="fixed_anchor_world_position",
            label="LOS source",
            sound_class="test_signal",
            color_rgb=(120, 220, 112),
            asset_class="world_point",
        ),
        SourceCaptureBinding(
            source_endpoint_id="m6x_world_nlos_speaker",
            actor_id="spear_apartment_nlos_point",
            emitter_anchor_id="speaker",
            capture_anchor_id=None,
            room_anchor_id="world_nlos",
            semantic_id=None,
            position_authority="fixed_anchor_world_position",
            label="NLOS source",
            sound_class="test_signal",
            color_rgb=(167, 121, 255),
            asset_class="world_point",
        ),
    )

    @property
    def actors_by_id(self) -> dict[str, ActorCaptureBinding]:
        return {item.actor_id: item for item in self.actor_bindings}

    @property
    def sources_by_id(self) -> dict[str, SourceCaptureBinding]:
        return {item.source_endpoint_id: item for item in self.source_bindings}

    def actor_binding(self, actor_id: str) -> ActorCaptureBinding:
        try:
            return self.actors_by_id[actor_id]
        except KeyError as exc:
            raise CaptureAdapterError(
                f"capture adapter {self.adapter_id!r} has no actor {actor_id!r}"
            ) from exc

    def source_binding(self, source_endpoint_id: str) -> SourceCaptureBinding:
        try:
            return self.sources_by_id[source_endpoint_id]
        except KeyError as exc:
            raise CaptureAdapterError(
                f"capture adapter {self.adapter_id!r} has no source endpoint "
                f"{source_endpoint_id!r}"
            ) from exc

    def validate_registry_bindings(
        self, endpoint_registry: Mapping[str, Any]
    ) -> list[str]:
        records = {
            item.get("source_endpoint_id"): item
            for item in endpoint_registry.get("source_endpoints", [])
            if isinstance(item, Mapping)
        }
        errors: list[str] = []
        for declared in self.source_bindings:
            endpoint = records.get(declared.source_endpoint_id)
            if not isinstance(endpoint, Mapping):
                errors.append(
                    f"capture adapter endpoint {declared.source_endpoint_id!r} is missing"
                )
                continue
            binding = endpoint.get("binding")
            if not isinstance(binding, Mapping):
                errors.append(
                    f"capture adapter endpoint {declared.source_endpoint_id!r} has no binding"
                )
                continue
            observed_actor = (
                binding.get("entity_instance_id")
                if binding.get("kind") == "entity_anchor"
                else binding.get("world_point_id")
            )
            if observed_actor != declared.actor_id:
                errors.append(
                    f"capture adapter endpoint {declared.source_endpoint_id!r} actor "
                    f"must be {declared.actor_id!r}"
                )
            if binding.get("emitter_anchor_id") != declared.emitter_anchor_id:
                errors.append(
                    f"capture adapter endpoint {declared.source_endpoint_id!r} emitter "
                    f"must be {declared.emitter_anchor_id!r}"
                )
            actor = self.actor_binding(declared.actor_id)
            if (
                binding.get("kind") == "entity_anchor"
                and binding.get("entity_asset_id") != actor.asset_id
            ):
                errors.append(
                    f"capture adapter endpoint {declared.source_endpoint_id!r} asset "
                    f"must be {actor.asset_id!r}"
                )
        return errors

    def materialize_actor_root_paths(
        self,
        trajectory_templates: Mapping[str, Any],
        anchor_library: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for actor in self.actor_bindings:
            if actor.trajectory_template_id is None:
                continue
            if actor.trajectory_route_id is None:
                raise CaptureAdapterError(
                    f"actor {actor.actor_id!r} has a template without a route"
                )
            route = materialize_template_route(
                trajectory_templates,
                template_id=actor.trajectory_template_id,
                route_id=actor.trajectory_route_id,
                anchor_library=anchor_library,
            )
            if route.shape != (MASTER_FRAME_COUNT, 3):
                raise CaptureAdapterError(
                    f"actor {actor.actor_id!r} route must contain exactly "
                    f"{MASTER_FRAME_COUNT} points"
                )
            result[actor.actor_id] = np.ascontiguousarray(route)
        return dict(sorted(result.items()))

    def provisional_source_paths(
        self,
        anchor_library: Mapping[str, Any],
        actor_root_paths: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        anchors = _anchor_positions(anchor_library)
        result: dict[str, np.ndarray] = {}
        for source in self.source_bindings:
            if source.capture_anchor_id is not None:
                try:
                    result[source.source_endpoint_id] = np.ascontiguousarray(
                        actor_root_paths[source.actor_id]
                    )
                except KeyError as exc:
                    raise CaptureAdapterError(
                        f"missing root path for captured actor {source.actor_id!r}"
                    ) from exc
            elif source.room_anchor_id is not None:
                try:
                    result[source.source_endpoint_id] = _static_path(
                        anchors[source.room_anchor_id]
                    )
                except KeyError as exc:
                    raise CaptureAdapterError(
                        f"missing room anchor {source.room_anchor_id!r}"
                    ) from exc
            else:
                raise CaptureAdapterError(
                    f"source {source.source_endpoint_id!r} has no position binding"
                )
        return dict(sorted(result.items()))

    def actual_source_paths(
        self,
        anchor_library: Mapping[str, Any],
        capture: CaptureData,
    ) -> dict[str, np.ndarray]:
        anchors = _anchor_positions(anchor_library)
        anchor_order = capture.evidence.get("anchor_order")
        if (
            not isinstance(anchor_order, list)
            or len(anchor_order) != len(set(anchor_order))
            or not all(isinstance(item, str) for item in anchor_order)
        ):
            raise CaptureAdapterError(
                "capture anchor_order must contain unique string names"
            )
        anchor_indices = {name: index for index, name in enumerate(anchor_order)}
        result: dict[str, np.ndarray] = {}
        for source in self.source_bindings:
            if source.capture_anchor_id is not None:
                try:
                    index = anchor_indices[source.capture_anchor_id]
                except KeyError as exc:
                    raise CaptureAdapterError(
                        f"capture is missing emitter anchor "
                        f"{source.capture_anchor_id!r} for "
                        f"{source.source_endpoint_id!r}"
                    ) from exc
                result[source.source_endpoint_id] = np.ascontiguousarray(
                    capture.anchor_positions_m[:, index, :]
                )
            elif source.room_anchor_id is not None:
                try:
                    result[source.source_endpoint_id] = _static_path(
                        anchors[source.room_anchor_id]
                    )
                except KeyError as exc:
                    raise CaptureAdapterError(
                        f"missing room anchor {source.room_anchor_id!r}"
                    ) from exc
            else:
                raise CaptureAdapterError(
                    f"source {source.source_endpoint_id!r} has no position binding"
                )
        return dict(sorted(result.items()))

    def timeline_actor_ids(self, candidate_actor_ids: Sequence[str]) -> tuple[str, ...]:
        actor_ids = {
            actor.actor_id
            for actor in self.actor_bindings
            if actor.always_include_in_timeline
        }
        actor_ids.update(candidate_actor_ids)
        missing = sorted(actor_ids.difference(self.actors_by_id))
        if missing:
            raise CaptureAdapterError(
                f"capture adapter has no Timeline actor bindings: {missing}"
            )
        return tuple(sorted(actor_ids))

    def timeline_actor_state(
        self,
        capture: CaptureData,
        *,
        actor_id: str,
        master_frame: int,
        local_frame: int,
        source_endpoint_id: str | None,
        trajectories: Mapping[str, np.ndarray],
    ) -> CaptureActorState:
        actor = self.actor_binding(actor_id)
        if actor.capture_matrix_index is not None:
            matrix = capture.actor_world_matrices[
                master_frame, actor.capture_matrix_index
            ]
            position = matrix[:3, 3]
            rotation = _matrix_quaternion_xyzw(matrix[:3, :3])
            if actor.pose_hash_record_path is None:
                raise CaptureAdapterError(
                    f"captured actor {actor_id!r} has no pose-hash binding"
                )
            pose_hash = str(
                _nested_record_value(
                    capture.records[master_frame], actor.pose_hash_record_path
                )
            )
        else:
            if source_endpoint_id is None:
                raise CaptureAdapterError(
                    f"static actor {actor_id!r} has no routed source endpoint"
                )
            try:
                position = trajectories[source_endpoint_id][local_frame]
            except KeyError as exc:
                raise CaptureAdapterError(
                    f"missing trajectory for source {source_endpoint_id!r}"
                ) from exc
            rotation = (0.0, 0.0, 0.0, 1.0)
            pose_hash = canonical_json_sha256(
                {"actor_id": actor_id, "position_m": position.tolist()}
            )
        phase = (
            0.0
            if actor.action_phase_period_frames == 1
            else (master_frame % actor.action_phase_period_frames)
            / actor.action_phase_period_frames
        )
        return CaptureActorState(
            translation_m=tuple(float(value) for value in position),
            rotation_xyzw=rotation,
            action_id=actor.action_id,
            action_phase=phase,
            pose_hash=pose_hash,
        )

    def validate_capture_reuse_contract(
        self,
        evidence: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
    ) -> None:
        room_manifest_path = Path(room_manifest_path).resolve()
        m1_request_path = Path(m1_request_path).resolve()
        room_manifest = load_json(room_manifest_path)
        request = load_json(m1_request_path)
        errors: list[str] = []

        if evidence.get("schema") != self.capture_schema:
            errors.append(f"schema must be {self.capture_schema!r}")
        if evidence.get("status") != "pass":
            errors.append("status must be 'pass'")
        if evidence.get("frame_count") != MASTER_FRAME_COUNT:
            errors.append(f"frame_count must be {MASTER_FRAME_COUNT}")
        if evidence.get("frame_rate_hz") != FRAME_RATE_HZ:
            errors.append(f"frame_rate_hz must be {FRAME_RATE_HZ}")
        if evidence.get("time_base_hz") != TIME_BASE_HZ:
            errors.append(f"time_base_hz must be {TIME_BASE_HZ}")
        if evidence.get("anchor_order") != list(self.capture_anchor_order):
            errors.append(f"anchor_order must be {list(self.capture_anchor_order)!r}")

        request_room_id = request.get("room_id")
        manifest_room_id = room_manifest.get("room_id")
        if not isinstance(request_room_id, str) or request_room_id != manifest_room_id:
            errors.append("current request room_id differs from its room manifest")

        inputs = evidence.get("inputs")
        if not isinstance(inputs, Mapping):
            errors.append("inputs must identify the retained capture inputs")
        else:
            for key, expected_path in (
                ("room_manifest", room_manifest_path),
                ("m1_request", m1_request_path),
            ):
                record = inputs.get(key)
                if not isinstance(record, Mapping):
                    errors.append(f"inputs.{key} is missing")
                elif record.get("sha256") != sha256_file(expected_path):
                    errors.append(
                        f"inputs.{key} differs from the current Apartment input"
                    )

        rig = request.get("primary_camera_rig")
        if not isinstance(rig, Mapping):
            errors.append("current Apartment request has no primary_camera_rig")
        else:
            world_from_rig = rig.get("world_from_rig")
            calibration = rig.get("shared_calibration")
            if not isinstance(world_from_rig, Mapping) or not isinstance(
                calibration, Mapping
            ):
                errors.append("current Apartment camera contract is incomplete")
            else:
                expected_camera = {
                    "position_m": world_from_rig.get("translation_m"),
                    "rotation_xyzw": world_from_rig.get("rotation_xyzw"),
                    "horizontal_fov_deg": calibration.get("hfov_degrees"),
                    "legacy_camera_contract_required": True,
                }
                camera = evidence.get("camera")
                if not isinstance(camera, Mapping) or any(
                    camera.get(key) != value for key, value in expected_camera.items()
                ):
                    errors.append(
                        "camera differs from the current fixed-Apartment request"
                    )

        ticks_per_frame = TIME_BASE_HZ // FRAME_RATE_HZ
        if len(records) != MASTER_FRAME_COUNT:
            errors.append(f"frame_readback must contain {MASTER_FRAME_COUNT} records")
        elif any(
            not isinstance(record, Mapping)
            or record.get("frame_index") != index
            or record.get("pts_ticks") != index * ticks_per_frame
            for index, record in enumerate(records)
        ):
            errors.append(
                "frame_readback frame_index/pts_ticks do not match the time base"
            )

        if errors:
            raise CaptureAdapterError(
                "capture reuse contract failed: " + "; ".join(errors)
            )

    def load_capture(
        self,
        path: str | Path,
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
    ) -> CaptureData:
        root = Path(path).resolve()
        required = {
            "rgb": root / "arrays/rgb.npy",
            "semantic": root / "arrays/semantic.npy",
            "actor": root / "arrays/actor_world_matrices.npy",
            "anchors": root / "arrays/anchor_positions_m.npy",
            "records": root / "frame_readback.json",
            "evidence": root / "evidence.json",
        }
        missing = [str(value) for value in required.values() if not value.is_file()]
        if missing:
            raise CaptureAdapterError(f"capture is incomplete: {missing}")
        records = json.loads(required["records"].read_text(encoding="utf-8"))
        evidence = load_json(required["evidence"])
        if not isinstance(records, list):
            raise CaptureAdapterError("capture frame_readback must be a JSON array")
        self.validate_capture_reuse_contract(
            evidence,
            records,
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
        )
        rgb = np.load(required["rgb"], allow_pickle=False)
        semantic = np.load(required["semantic"], allow_pickle=False)
        actor = np.load(required["actor"], allow_pickle=False)
        anchors = np.load(required["anchors"], allow_pickle=False)
        captured_actor_count = sum(
            item.capture_matrix_index is not None for item in self.actor_bindings
        )
        if (
            rgb.shape != (MASTER_FRAME_COUNT, 240, 320, 3)
            or rgb.dtype != np.uint8
            or semantic.shape != (MASTER_FRAME_COUNT, 240, 320)
            or actor.shape != (MASTER_FRAME_COUNT, captured_actor_count, 4, 4)
            or anchors.shape != (MASTER_FRAME_COUNT, len(self.capture_anchor_order), 3)
        ):
            raise CaptureAdapterError(
                "capture arrays/evidence differ from the adapter contract"
            )
        return CaptureData(
            root=root,
            rgb=np.ascontiguousarray(rgb),
            semantic=np.ascontiguousarray(semantic),
            actor_world_matrices=np.ascontiguousarray(actor),
            anchor_positions_m=np.ascontiguousarray(anchors),
            records=tuple(records),
            evidence=evidence,
        )

    def capture_data(self, result: Any) -> CaptureData:
        return CaptureData(
            root=result.output_dir,
            rgb=result.rgb,
            semantic=result.semantic,
            actor_world_matrices=result.actor_world_matrices,
            anchor_positions_m=result.anchor_positions_m,
            records=result.records,
            evidence=result.evidence,
        )

    def capture(
        self,
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
        provider_assets: Mapping[str, str | Path],
        actor_root_paths: Mapping[str, np.ndarray],
        output_dir: str | Path,
        runtime_root: str | Path | None,
        route_provenance: Mapping[str, Any],
    ) -> CaptureData:
        required_assets = {
            "human_runtime_glb_path",
            "animal_manifest_path",
            "animal_request_path",
        }
        missing_assets = sorted(required_assets.difference(provider_assets))
        if missing_assets:
            raise CaptureAdapterError(
                f"capture provider assets are missing: {missing_assets}"
            )
        required_actors = {"human0", "dog0"}
        missing_actors = sorted(required_actors.difference(actor_root_paths))
        if missing_actors:
            raise CaptureAdapterError(
                f"capture provider root paths are missing: {missing_actors}"
            )
        result = capture_human_beagle_paths(
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
            human_runtime_glb_path=provider_assets["human_runtime_glb_path"],
            beagle_animal_manifest_path=provider_assets["animal_manifest_path"],
            beagle_m2_request_path=provider_assets["animal_request_path"],
            human_root_path_m=actor_root_paths["human0"],
            beagle_root_path_m=actor_root_paths["dog0"],
            output_dir=output_dir,
            runtime_root=runtime_root,
            route_provenance=route_provenance,
            require_legacy_camera=True,
        )
        capture = self.capture_data(result)
        self.validate_capture_reuse_contract(
            capture.evidence,
            capture.records,
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
        )
        return capture


HUMAN_BEAGLE_CAPTURE_ADAPTER = HumanBeagleCaptureAdapter()


__all__ = [
    "ActorCaptureBinding",
    "CaptureActorState",
    "CaptureAdapterError",
    "CaptureData",
    "FixedApartmentCaptureAdapter",
    "HUMAN_BEAGLE_CAPTURE_ADAPTER",
    "HumanBeagleCaptureAdapter",
    "SourceCaptureBinding",
]
