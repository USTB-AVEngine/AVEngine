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
from avengine.m5_1.mixed_capture import (
    LOCOMOTION_POLICY_ID,
    MixedCaptureError,
    capture_human_beagle_paths,
    locomotion_schedule_from_root_trajectory,
    trajectory_world_matrices,
)
from avengine.m5_1.orientation import (
    M51OrientationError,
    habitat_basis_from_yaw_degrees,
)
from avengine.m6x.trajectory import (
    M6XTrajectoryError,
    materialize_template_route,
    template_route_first_anchor_yaw_degrees,
)
from avengine.m6x.visual_profile import (
    M6XVisualProfileError,
    apply_runtime_review_profile,
    configure_runtime_review_profile,
    load_review_visual_profile,
    validate_profile_capture_request,
    validate_runtime_review_light_readback,
)


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
    action_id_record_path: tuple[str, ...] | None = None
    action_time_ticks_record_path: tuple[str, ...] | None = None
    action_sample_index_record_path: tuple[str, ...] | None = None
    action_phase_record_path: tuple[str, ...] | None = None
    actor_root_position_record_path: tuple[str, ...] | None = None
    action_sample_counts_evidence_path: tuple[str, ...] | None = None
    local_anatomical_forward_axis: tuple[float, float, float] | None = None

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
    action_time_ticks: int
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

    def materialize_actor_fallback_forwards_xz(
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

    def validate_capture_locomotion(self, capture: CaptureData) -> None: ...

    def validate_capture_orientation(
        self,
        capture: CaptureData,
        *,
        actor_root_paths: Mapping[str, np.ndarray],
        actor_fallback_forwards_xz: Mapping[str, np.ndarray],
    ) -> None: ...

    def load_capture(
        self,
        path: str | Path,
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
        actor_root_paths: Mapping[str, np.ndarray],
        actor_fallback_forwards_xz: Mapping[str, np.ndarray],
    ) -> CaptureData: ...

    def capture(
        self,
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
        provider_assets: Mapping[str, str | Path],
        actor_root_paths: Mapping[str, np.ndarray],
        actor_fallback_forwards_xz: Mapping[str, np.ndarray],
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
            action_id_record_path=("beagle", "action_id"),
            action_time_ticks_record_path=("beagle", "action_time_ticks"),
            action_sample_index_record_path=("beagle", "action_sample_index"),
            action_phase_record_path=("beagle", "action_phase"),
            actor_root_position_record_path=("beagle", "actor_root_position_m"),
            action_sample_counts_evidence_path=(
                "runtime",
                "beagle_action_sample_counts",
            ),
            local_anatomical_forward_axis=(1.0, 0.0, 0.0),
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
            action_id_record_path=("human", "action_id"),
            action_time_ticks_record_path=("human", "action_time_ticks"),
            action_sample_index_record_path=("human", "action_sample_index"),
            action_phase_record_path=("human", "action_phase"),
            actor_root_position_record_path=("human", "actor_root_position_m"),
            action_sample_counts_evidence_path=(
                "runtime",
                "human_action_sample_counts",
            ),
            local_anatomical_forward_axis=(0.0, 0.0, 1.0),
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

    def materialize_actor_fallback_forwards_xz(
        self,
        trajectory_templates: Mapping[str, Any],
        anchor_library: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        """Resolve each captured route's authored first-anchor Habitat yaw."""

        result: dict[str, np.ndarray] = {}
        for actor in self.actor_bindings:
            if actor.trajectory_template_id is None:
                continue
            if actor.trajectory_route_id is None:
                raise CaptureAdapterError(
                    f"actor {actor.actor_id!r} has a template without a route"
                )
            try:
                yaw_degrees = template_route_first_anchor_yaw_degrees(
                    trajectory_templates,
                    template_id=actor.trajectory_template_id,
                    route_id=actor.trajectory_route_id,
                    anchor_library=anchor_library,
                )
                forward_xz = habitat_basis_from_yaw_degrees(
                    yaw_degrees
                ).forward_xz
            except (M6XTrajectoryError, M51OrientationError) as exc:
                raise CaptureAdapterError(
                    f"actor {actor.actor_id!r} has no valid authored first-anchor yaw"
                ) from exc
            result[actor.actor_id] = np.ascontiguousarray(
                np.asarray(forward_xz, dtype=np.float64)
            )
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
        action_record_paths = (
            actor.action_id_record_path,
            actor.action_time_ticks_record_path,
            actor.action_phase_record_path,
        )
        if any(path is not None for path in action_record_paths):
            if not all(path is not None for path in action_record_paths):
                raise CaptureAdapterError(
                    f"actor {actor_id!r} has an incomplete action-record binding"
                )
            record = capture.records[master_frame]
            action_id = _nested_record_value(
                record, actor.action_id_record_path or ()
            )
            action_time_ticks = _nested_record_value(
                record, actor.action_time_ticks_record_path or ()
            )
            phase = _nested_record_value(
                record, actor.action_phase_record_path or ()
            )
            if (
                not isinstance(action_id, str)
                or not action_id
                or isinstance(action_time_ticks, bool)
                or not isinstance(action_time_ticks, int)
                or action_time_ticks < 0
                or isinstance(phase, bool)
                or not isinstance(phase, (int, float))
                or not math.isfinite(float(phase))
                or not 0.0 <= float(phase) < 1.0
            ):
                raise CaptureAdapterError(
                    f"actor {actor_id!r} captured action state is invalid"
                )
            phase = float(phase)
        else:
            action_id = actor.action_id
            action_time_ticks = (
                0
                if actor.action_phase_period_frames == 1
                else master_frame * (TIME_BASE_HZ // FRAME_RATE_HZ)
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
            action_id=action_id,
            action_time_ticks=action_time_ticks,
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

        locomotion = evidence.get("locomotion")
        if (
            not isinstance(locomotion, Mapping)
            or locomotion.get("policy_id") != LOCOMOTION_POLICY_ID
        ):
            errors.append(
                f"locomotion.policy_id must be {LOCOMOTION_POLICY_ID!r}"
            )
        if len(records) == MASTER_FRAME_COUNT:
            for actor in self.actor_bindings:
                paths = (
                    actor.action_id_record_path,
                    actor.action_time_ticks_record_path,
                    actor.action_phase_record_path,
                )
                if not any(path is not None for path in paths):
                    continue
                if not all(path is not None for path in paths):
                    errors.append(
                        f"actor {actor.actor_id!r} action-record binding is incomplete"
                    )
                    continue
                previous_action: str | None = None
                previous_time = 0
                for frame_index, record in enumerate(records):
                    try:
                        action_id = _nested_record_value(
                            record, actor.action_id_record_path or ()
                        )
                        action_time = _nested_record_value(
                            record, actor.action_time_ticks_record_path or ()
                        )
                        phase = _nested_record_value(
                            record, actor.action_phase_record_path or ()
                        )
                    except CaptureAdapterError as exc:
                        errors.append(str(exc))
                        break
                    if (
                        action_id not in {"idle", "walk"}
                        or isinstance(action_time, bool)
                        or not isinstance(action_time, int)
                        or action_time < 0
                        or isinstance(phase, bool)
                        or not isinstance(phase, (int, float))
                        or not math.isfinite(float(phase))
                        or not 0.0 <= float(phase) < 1.0
                    ):
                        errors.append(
                            f"actor {actor.actor_id!r} frame {frame_index} has an "
                            "invalid captured locomotion state"
                        )
                        break
                    expected_time = (
                        0
                        if frame_index == 0 or action_id != previous_action
                        else previous_time + ticks_per_frame
                    )
                    if action_time != expected_time:
                        errors.append(
                            f"actor {actor.actor_id!r} frame {frame_index} action "
                            "clock did not reset/increment deterministically"
                        )
                        break
                    previous_action = action_id
                    previous_time = action_time

        if errors:
            raise CaptureAdapterError(
                "capture reuse contract failed: " + "; ".join(errors)
            )

    def validate_capture_locomotion(self, capture: CaptureData) -> None:
        """Close retained action records against retained actor-root matrices."""

        matrices = np.asarray(capture.actor_world_matrices, dtype=np.float64)
        if (
            matrices.ndim != 4
            or matrices.shape[0] != MASTER_FRAME_COUNT
            or matrices.shape[2:] != (4, 4)
            or not np.all(np.isfinite(matrices))
        ):
            raise CaptureAdapterError(
                "capture locomotion closure requires finite actor_world_matrices"
            )
        if len(capture.records) != MASTER_FRAME_COUNT:
            raise CaptureAdapterError(
                f"capture locomotion closure requires {MASTER_FRAME_COUNT} records"
            )

        ticks_per_frame = TIME_BASE_HZ // FRAME_RATE_HZ
        errors: list[str] = []
        for actor in self.actor_bindings:
            paths = {
                "action_id": actor.action_id_record_path,
                "action_time_ticks": actor.action_time_ticks_record_path,
                "action_sample_index": actor.action_sample_index_record_path,
                "action_phase": actor.action_phase_record_path,
                "actor_root_position_m": actor.actor_root_position_record_path,
                "action_sample_counts": actor.action_sample_counts_evidence_path,
            }
            if not any(path is not None for path in paths.values()):
                continue
            missing_paths = sorted(
                name for name, path in paths.items() if path is None
            )
            if actor.capture_matrix_index is None or missing_paths:
                errors.append(
                    f"actor {actor.actor_id!r} locomotion binding is incomplete: "
                    f"{missing_paths}"
                )
                continue
            matrix_index = actor.capture_matrix_index
            if matrix_index < 0 or matrix_index >= matrices.shape[1]:
                errors.append(
                    f"actor {actor.actor_id!r} capture matrix index is out of range"
                )
                continue
            try:
                raw_counts = _nested_record_value(
                    capture.evidence,
                    actor.action_sample_counts_evidence_path or (),
                )
            except CaptureAdapterError as exc:
                errors.append(str(exc))
                continue
            if not isinstance(raw_counts, Mapping) or any(
                action_id not in raw_counts
                or isinstance(raw_counts[action_id], bool)
                or not isinstance(raw_counts[action_id], int)
                or raw_counts[action_id] < 1
                for action_id in ("idle", "walk")
            ):
                errors.append(
                    f"actor {actor.actor_id!r} action sample counts are invalid"
                )
                continue
            roots = np.ascontiguousarray(matrices[:, matrix_index, :3, 3])
            try:
                expected_schedule = locomotion_schedule_from_root_trajectory(
                    roots,
                    action_sample_counts={
                        action_id: int(raw_counts[action_id])
                        for action_id in ("idle", "walk")
                    },
                )
            except MixedCaptureError as exc:
                errors.append(
                    f"actor {actor.actor_id!r} retained root trajectory is invalid: {exc}"
                )
                continue

            for frame_index, (record, expected) in enumerate(
                zip(capture.records, expected_schedule, strict=True)
            ):
                try:
                    action_id = _nested_record_value(
                        record, actor.action_id_record_path or ()
                    )
                    action_time_ticks = _nested_record_value(
                        record, actor.action_time_ticks_record_path or ()
                    )
                    sample_index = _nested_record_value(
                        record, actor.action_sample_index_record_path or ()
                    )
                    phase = _nested_record_value(
                        record, actor.action_phase_record_path or ()
                    )
                    root_record = _nested_record_value(
                        record, actor.actor_root_position_record_path or ()
                    )
                except CaptureAdapterError as exc:
                    errors.append(str(exc))
                    break
                try:
                    root_position = np.asarray(root_record, dtype=np.float64)
                except (TypeError, ValueError, OverflowError):
                    root_position = np.empty(0, dtype=np.float64)
                if (
                    root_position.shape != (3,)
                    or not np.all(np.isfinite(root_position))
                    or not np.allclose(
                        root_position,
                        roots[frame_index],
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    errors.append(
                        f"actor {actor.actor_id!r} frame {frame_index} root position "
                        "differs from actor_world_matrices"
                    )
                    break
                expected_time_ticks = expected.action_frame_index * ticks_per_frame
                if action_id != expected.action_id:
                    errors.append(
                        f"actor {actor.actor_id!r} frame {frame_index} action_id "
                        "differs from recomputed root-speed locomotion"
                    )
                    break
                if (
                    isinstance(action_time_ticks, bool)
                    or not isinstance(action_time_ticks, int)
                    or action_time_ticks != expected_time_ticks
                ):
                    errors.append(
                        f"actor {actor.actor_id!r} frame {frame_index} "
                        "action_time_ticks differs from recomputed locomotion"
                    )
                    break
                if (
                    isinstance(sample_index, bool)
                    or not isinstance(sample_index, int)
                    or sample_index != expected.action_sample_index
                ):
                    errors.append(
                        f"actor {actor.actor_id!r} frame {frame_index} "
                        "action_sample_index differs from recomputed locomotion"
                    )
                    break
                if (
                    isinstance(phase, bool)
                    or not isinstance(phase, (int, float))
                    or not math.isfinite(float(phase))
                    or not math.isclose(
                        float(phase),
                        expected.action_phase,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ):
                    errors.append(
                        f"actor {actor.actor_id!r} frame {frame_index} action_phase "
                        "differs from recomputed locomotion"
                    )
                    break

        if errors:
            raise CaptureAdapterError(
                "capture locomotion closure failed: " + "; ".join(errors)
            )

    def validate_capture_orientation(
        self,
        capture: CaptureData,
        *,
        actor_root_paths: Mapping[str, np.ndarray],
        actor_fallback_forwards_xz: Mapping[str, np.ndarray],
    ) -> None:
        """Close retained rigid transforms against the current authored routes.

        Retained evidence is deliberately not an authority here.  The expected
        heading is recomputed from the current materialized route and its
        current first-anchor yaw fallback, using the same path-tangent rule as
        a fresh capture.
        """

        matrices = np.asarray(capture.actor_world_matrices, dtype=np.float64)
        if (
            matrices.ndim != 4
            or matrices.shape[0] != MASTER_FRAME_COUNT
            or matrices.shape[2:] != (4, 4)
        ):
            raise CaptureAdapterError(
                "capture orientation closure requires actor_world_matrices "
                f"with shape ({MASTER_FRAME_COUNT}, actor_count, 4, 4)"
            )

        errors: list[str] = []
        identity = np.eye(3, dtype=np.float64)
        expected_bottom_row = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        for actor in self.actor_bindings:
            matrix_index = actor.capture_matrix_index
            if matrix_index is None:
                continue
            if matrix_index < 0 or matrix_index >= matrices.shape[1]:
                errors.append(
                    f"actor {actor.actor_id!r} capture matrix index is out of range"
                )
                continue
            if actor.local_anatomical_forward_axis is None:
                errors.append(
                    f"actor {actor.actor_id!r} has no anatomical forward binding"
                )
                continue
            try:
                route = np.asarray(
                    actor_root_paths[actor.actor_id], dtype=np.float64
                )
                fallback = np.asarray(
                    actor_fallback_forwards_xz[actor.actor_id], dtype=np.float64
                )
            except KeyError:
                errors.append(
                    f"actor {actor.actor_id!r} lacks a current authoritative "
                    "route or first-anchor yaw fallback"
                )
                continue
            if (
                route.shape != (MASTER_FRAME_COUNT, 3)
                or not np.all(np.isfinite(route))
            ):
                errors.append(
                    f"actor {actor.actor_id!r} current authoritative route is invalid"
                )
                continue

            actor_matrices = matrices[:, matrix_index]
            if not np.all(np.isfinite(actor_matrices)):
                errors.append(
                    f"actor {actor.actor_id!r} retained 4x4 transforms are not finite"
                )
                continue
            bottom_error = np.max(
                np.abs(actor_matrices[:, 3, :] - expected_bottom_row), axis=1
            )
            bad_bottom = np.flatnonzero(bottom_error > 1.0e-12)
            if len(bad_bottom):
                errors.append(
                    f"actor {actor.actor_id!r} frame {int(bad_bottom[0])} retained "
                    "4x4 transform is not affine/rigid"
                )
                continue

            rotations = actor_matrices[:, :3, :3]
            gram = np.swapaxes(rotations, 1, 2) @ rotations
            orthogonality_error = np.max(np.abs(gram - identity), axis=(1, 2))
            bad_rigid = np.flatnonzero(orthogonality_error > 1.0e-9)
            if len(bad_rigid):
                errors.append(
                    f"actor {actor.actor_id!r} frame {int(bad_rigid[0])} retained "
                    "rotation is not rigid/orthonormal"
                )
                continue
            determinants = np.linalg.det(rotations)
            bad_handedness = np.flatnonzero(
                np.abs(determinants - 1.0) > 1.0e-9
            )
            if len(bad_handedness):
                errors.append(
                    f"actor {actor.actor_id!r} frame "
                    f"{int(bad_handedness[0])} retained rotation is not right-handed"
                )
                continue

            root_error = np.max(
                np.abs(actor_matrices[:, :3, 3] - route), axis=1
            )
            bad_root = np.flatnonzero(root_error > 1.0e-12)
            if len(bad_root):
                errors.append(
                    f"actor {actor.actor_id!r} frame {int(bad_root[0])} retained "
                    "root differs from the current authoritative route"
                )
                continue

            local_forward = np.asarray(
                actor.local_anatomical_forward_axis, dtype=np.float64
            )
            try:
                expected_matrices = trajectory_world_matrices(
                    route,
                    local_forward_axis=local_forward,
                    fallback_forward_xz=fallback,
                )
            except MixedCaptureError as exc:
                errors.append(
                    f"actor {actor.actor_id!r} current heading authority is invalid: "
                    f"{exc}"
                )
                continue
            actual_forward = rotations @ local_forward
            expected_forward = expected_matrices[:, :3, :3] @ local_forward
            forward_error = np.linalg.norm(
                actual_forward - expected_forward, axis=1
            )
            bad_forward = np.flatnonzero(forward_error > 1.0e-9)
            if len(bad_forward):
                errors.append(
                    f"actor {actor.actor_id!r} frame {int(bad_forward[0])} retained "
                    "world forward differs from the current authoritative route "
                    "tangent or first-anchor yaw fallback"
                )
                continue
            canonical_rotation_error = np.max(
                np.abs(rotations - expected_matrices[:, :3, :3]), axis=(1, 2)
            )
            bad_canonical_rotation = np.flatnonzero(
                canonical_rotation_error > 1.0e-9
            )
            if len(bad_canonical_rotation):
                errors.append(
                    f"actor {actor.actor_id!r} frame "
                    f"{int(bad_canonical_rotation[0])} retained rotation differs "
                    "from the canonical current route orientation"
                )

        if errors:
            raise CaptureAdapterError(
                "capture orientation closure failed: " + "; ".join(errors)
            )

    def load_capture(
        self,
        path: str | Path,
        *,
        room_manifest_path: str | Path,
        m1_request_path: str | Path,
        actor_root_paths: Mapping[str, np.ndarray],
        actor_fallback_forwards_xz: Mapping[str, np.ndarray],
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
        request = load_json(Path(m1_request_path).resolve())
        captured_actor_count = sum(
            item.capture_matrix_index is not None for item in self.actor_bindings
        )
        calibration = request["primary_camera_rig"]["shared_calibration"]
        capture_height, capture_width = calibration["resolution_hw"]
        if (
            rgb.shape
            != (MASTER_FRAME_COUNT, capture_height, capture_width, 3)
            or rgb.dtype != np.uint8
            or semantic.shape
            != (MASTER_FRAME_COUNT, capture_height, capture_width)
            or actor.shape != (MASTER_FRAME_COUNT, captured_actor_count, 4, 4)
            or anchors.shape != (MASTER_FRAME_COUNT, len(self.capture_anchor_order), 3)
        ):
            raise CaptureAdapterError(
                "capture arrays/evidence differ from the adapter contract"
            )
        capture = CaptureData(
            root=root,
            rgb=np.ascontiguousarray(rgb),
            semantic=np.ascontiguousarray(semantic),
            actor_world_matrices=np.ascontiguousarray(actor),
            anchor_positions_m=np.ascontiguousarray(anchors),
            records=tuple(records),
            evidence=evidence,
        )
        self.validate_capture_locomotion(capture)
        self.validate_capture_orientation(
            capture,
            actor_root_paths=actor_root_paths,
            actor_fallback_forwards_xz=actor_fallback_forwards_xz,
        )
        return capture

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
        actor_fallback_forwards_xz: Mapping[str, np.ndarray],
        output_dir: str | Path,
        runtime_root: str | Path | None,
        route_provenance: Mapping[str, Any],
    ) -> CaptureData:
        required_assets = {
            "human_runtime_glb_path",
            "animal_manifest_path",
            "animal_request_path",
            "review_visual_profile_path",
            "exterior_proxy_glb_path",
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
        missing_forwards = sorted(
            required_actors.difference(actor_fallback_forwards_xz)
        )
        if missing_forwards:
            raise CaptureAdapterError(
                "capture provider authored fallback forwards are missing: "
                f"{missing_forwards}"
            )
        try:
            visual_profile = load_review_visual_profile(
                provider_assets["review_visual_profile_path"]
            )
            validate_profile_capture_request(
                visual_profile, load_json(Path(m1_request_path).resolve())
            )

            def review_configuration_hook(**kwargs: Any) -> Mapping[str, Any]:
                return configure_runtime_review_profile(
                    configuration=kwargs["configuration"],
                    profile=visual_profile,
                    habitat_sim=kwargs["habitat_sim"],
                )

            def review_scene_hook(**kwargs: Any) -> Mapping[str, Any]:
                return apply_runtime_review_profile(
                    simulator=kwargs["simulator"],
                    profile=visual_profile,
                    exterior_proxy_glb_path=provider_assets[
                        "exterior_proxy_glb_path"
                    ],
                    camera_listener_position_m=kwargs[
                        "camera_listener_position_m"
                    ],
                    habitat_sim=kwargs["habitat_sim"],
                    mn=kwargs["mn"],
                )

            def review_scene_readback_hook(**kwargs: Any) -> Mapping[str, Any]:
                return validate_runtime_review_light_readback(
                    simulator=kwargs["simulator"],
                    profile=visual_profile,
                    habitat_sim=kwargs["habitat_sim"],
                    actor_light_setup_key=kwargs["actor_light_setup_key"],
                )

            result = capture_human_beagle_paths(
                room_manifest_path=room_manifest_path,
                m1_request_path=m1_request_path,
                human_runtime_glb_path=provider_assets["human_runtime_glb_path"],
                beagle_animal_manifest_path=provider_assets["animal_manifest_path"],
                beagle_m2_request_path=provider_assets["animal_request_path"],
                human_root_path_m=actor_root_paths["human0"],
                beagle_root_path_m=actor_root_paths["dog0"],
                human_fallback_forward_xz=actor_fallback_forwards_xz["human0"],
                beagle_fallback_forward_xz=actor_fallback_forwards_xz["dog0"],
                output_dir=output_dir,
                runtime_root=runtime_root,
                route_provenance=route_provenance,
                require_legacy_camera=True,
                review_configuration_hook=review_configuration_hook,
                review_scene_hook=review_scene_hook,
                review_scene_readback_hook=review_scene_readback_hook,
            )
        except M6XVisualProfileError as exc:
            raise CaptureAdapterError(str(exc)) from exc
        capture = self.capture_data(result)
        self.validate_capture_reuse_contract(
            capture.evidence,
            capture.records,
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
        )
        self.validate_capture_locomotion(capture)
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
