"""Per-episode QA fact tables aggregated from frozen research artifacts.

The fact table is the aggregation layer of the QA benchmark: it joins the
retained trajectory bank, the RIR job plan listener pose, the asset-bound
binaural batch records, the source-asset runtime registry and the room anchor
library into one hash-bound document per episode. It never re-simulates or
re-renders anything; the only new values are analytic derivations from those
frozen inputs (listener-local direction of arrival, speeds, headings and
anchor distances).

Azimuth follows the AVEngine public listener convention also used by the
v4_3 label pipeline: front is 0 degrees, right is +90, left is -90 and rear
is +/-180. Elevation is positive upwards. The implementation is independent
of any private model branch.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

FACT_TABLE_SCHEMA = "avengine_qa_fact_table_v1"
TIME_BASE_HZ = 48000
MOVING_SPEED_THRESHOLD_MPS = 0.05
AZIMUTH_CONVENTION = "avengine_native_full_circle_front0_right_plus90"
EMITTER_OFFSET_TOLERANCE_M = 1.0e-6
HEADING_MIN_HORIZONTAL_OFFSET_M = 1.0e-6
_UNIT_QUATERNION_TOLERANCE = 1.0e-9
FRUSTUM_AUTHORITY = "center_point_pinhole_frustum_no_occlusion_v1"

CLAIM_BOUNDARY = (
    "Aggregated research facts for QA mining; joins frozen trajectory, "
    "listener, audio-batch, registry and anchor artifacts without granting "
    "any dataset, room or asset admission"
)


class QAFactTableError(ValueError):
    """A fact-table input violates the aggregation contract."""


def _as_float_array(value: Any, *, name: str, shape_tail: int = 3) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QAFactTableError(f"{name} must be numeric") from exc
    if array.ndim != 2 or array.shape[1] != shape_tail:
        raise QAFactTableError(f"{name} must be a sequence of {shape_tail}-vectors")
    if not np.all(np.isfinite(array)):
        raise QAFactTableError(f"{name} must be finite")
    return array


def _unit_quaternion_wxyz(value: Any) -> np.ndarray:
    try:
        quaternion = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QAFactTableError("listener orientation must be numeric") from exc
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise QAFactTableError("listener orientation must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_QUATERNION_TOLERANCE):
        raise QAFactTableError("listener orientation must be unit normalized")
    return quaternion


def _rotate_by_inverse_quaternion(
    world_vectors: np.ndarray, quaternion_wxyz: np.ndarray
) -> np.ndarray:
    """Rotate world vectors into the listener-local frame."""

    w = float(quaternion_wxyz[0])
    inverse_vector = -quaternion_wxyz[1:]
    uv = np.cross(inverse_vector, world_vectors)
    uuv = np.cross(inverse_vector, uv)
    return world_vectors + 2.0 * (w * uv + uuv)


def _rotate_by_quaternion(
    local_vector: np.ndarray, quaternion_wxyz: np.ndarray
) -> np.ndarray:
    w = float(quaternion_wxyz[0])
    vector = quaternion_wxyz[1:]
    uv = np.cross(vector, local_vector)
    uuv = np.cross(vector, uv)
    return local_vector + 2.0 * (w * uv + uuv)


def _yaw_deg_of_world_forward(forward_xz: np.ndarray) -> float:
    """Yaw in the anchor-library convention: 0 faces -Z, +yaw turns toward -X."""

    return float(math.degrees(math.atan2(-float(forward_xz[0]), -float(forward_xz[1]))))


def listener_local_spherical_track(
    source_positions_m: Any,
    listener_position_m: Any,
    listener_orientation_wxyz: Any,
) -> dict[str, list[float]]:
    """Per-frame azimuth/elevation/distance of world points around a listener."""

    sources = _as_float_array(source_positions_m, name="source positions")
    try:
        listener = np.asarray(listener_position_m, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QAFactTableError("listener position must be numeric") from exc
    if listener.shape != (3,) or not np.all(np.isfinite(listener)):
        raise QAFactTableError("listener position must be a finite xyz point")
    quaternion = _unit_quaternion_wxyz(listener_orientation_wxyz)

    world_direction = sources - listener[None, :]
    distance = np.linalg.norm(world_direction, axis=1)
    if not np.all(distance > 0.0):
        raise QAFactTableError("source and listener positions must differ")
    local = _rotate_by_inverse_quaternion(world_direction, quaternion)
    azimuth = np.degrees(np.arctan2(local[:, 0], -local[:, 2]))
    azimuth = np.where(azimuth > 180.0, azimuth - 360.0, azimuth)
    azimuth = np.where(np.isclose(azimuth, 0.0, rtol=0.0, atol=1.0e-15), 0.0, azimuth)
    elevation = np.degrees(np.arcsin(np.clip(local[:, 1] / distance, -1.0, 1.0)))
    return {
        "azimuth_deg": [float(value) for value in azimuth],
        "elevation_deg": [float(value) for value in elevation],
        "distance_m": [float(value) for value in distance],
    }


def center_frustum_track(
    source_positions_m: Any,
    listener_position_m: Any,
    listener_orientation_wxyz: Any,
    *,
    hfov_degrees: float,
    resolution_hw: tuple[int, int],
) -> dict[str, Any]:
    """Exact pinhole frustum test for instance centre points.

    This is the geometric judgment behind off-screen certificates: a centre
    outside the frustum is definitely not visible. It says nothing about
    occlusion, so it never certifies that an instance IS visible; pixel
    truth arrives with the P1 amodal/modal semantic passes.
    """

    if not isinstance(hfov_degrees, (int, float)) or not 0.0 < hfov_degrees < 180.0:
        raise QAFactTableError("hfov_degrees must be in (0, 180)")
    height, width = resolution_hw
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise QAFactTableError("resolution_hw must be positive integers")
    sources = _as_float_array(source_positions_m, name="source positions")
    listener = np.asarray(listener_position_m, dtype=np.float64)
    quaternion = _unit_quaternion_wxyz(listener_orientation_wxyz)
    local = _rotate_by_inverse_quaternion(sources - listener[None, :], quaternion)

    tan_half_h = math.tan(math.radians(float(hfov_degrees)) / 2.0)
    tan_half_v = tan_half_h * (height / width)
    depth = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        horizontal = np.abs(local[:, 0]) <= tan_half_h * depth
        vertical = np.abs(local[:, 1]) <= tan_half_v * depth
    in_frustum = (depth > 0.0) & horizontal & vertical

    events: list[dict[str, Any]] = []
    flags = [bool(value) for value in in_frustum]
    azimuth = np.degrees(np.arctan2(local[:, 0], -local[:, 2]))
    for index in range(1, len(flags)):
        if flags[index] == flags[index - 1]:
            continue
        kind = "entry" if flags[index] else "exit"
        # Judge the side on the outside frame adjacent to the transition:
        # "entered from the left" means it was outside on the left just before.
        outside_index = index - 1 if flags[index] else index
        events.append(
            {
                "kind": kind,
                "frame": index,
                "side": "right" if float(local[outside_index, 0]) > 0.0 else "left",
                "azimuth_deg_at_event": float(azimuth[index]),
            }
        )
    return {
        "in_frustum": flags,
        "in_frustum_frame_count": int(np.count_nonzero(in_frustum)),
        "always_outside_frustum": not any(flags),
        "always_inside_frustum": all(flags),
        "events": events,
    }


def _speed_track_mps(root_positions: np.ndarray, frame_rate_hz: float) -> list[float]:
    if root_positions.shape[0] < 2:
        raise QAFactTableError("speed derivation needs at least two frames")
    steps = np.linalg.norm(np.diff(root_positions, axis=0), axis=1) * frame_rate_hz
    return [float(value) for value in steps] + [float(steps[-1])]


def _registry_asset(registry: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    assets = registry.get("assets")
    if not isinstance(assets, Sequence):
        raise QAFactTableError("registry must contain an assets sequence")
    for asset in assets:
        if isinstance(asset, Mapping) and asset.get("asset_id") == asset_id:
            return asset
    raise QAFactTableError(f"asset {asset_id!r} is not in the runtime registry")


def _default_emitter_anchor(asset: Mapping[str, Any]) -> Mapping[str, Any]:
    anchor_id = asset.get("default_emitter_anchor_id")
    for anchor in asset.get("emitter_anchors", []):
        if isinstance(anchor, Mapping) and anchor.get("anchor_id") == anchor_id:
            offset = np.asarray(anchor.get("offset_m"), dtype=np.float64)
            if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                raise QAFactTableError(
                    f"emitter anchor {anchor_id!r} must declare a finite offset"
                )
            return anchor
    raise QAFactTableError(
        f"asset {asset.get('asset_id')!r} does not resolve its default emitter anchor"
    )


def _check_emitter_offset_consistency(
    *,
    slot_id: str,
    root: np.ndarray,
    emitter: np.ndarray,
    offset_m: np.ndarray,
) -> None:
    delta = emitter - root
    vertical_error = np.max(np.abs(delta[:, 1] - float(offset_m[1])))
    horizontal = np.linalg.norm(delta[:, [0, 2]], axis=1)
    offset_horizontal = float(math.hypot(float(offset_m[0]), float(offset_m[2])))
    horizontal_error = np.max(np.abs(horizontal - offset_horizontal))
    if vertical_error > EMITTER_OFFSET_TOLERANCE_M:
        raise QAFactTableError(
            f"{slot_id}: emitter path is not the root path plus the registry "
            f"emitter offset (vertical error {float(vertical_error):.3e} m)"
        )
    if horizontal_error > EMITTER_OFFSET_TOLERANCE_M:
        raise QAFactTableError(
            f"{slot_id}: emitter path is not a yaw rotation of the registry "
            f"emitter offset (horizontal error {float(horizontal_error):.3e} m)"
        )


def _anchor_convention_angle_deg(x: float, z: float) -> float:
    """Planar angle in the anchor-library convention (0 faces -Z, +yaw to -X)."""

    return math.degrees(math.atan2(-x, -z))


def _facing_track_yaw_deg(
    *,
    root: np.ndarray,
    emitter: np.ndarray,
    offset_m: np.ndarray,
    forward_axis_local: Any,
) -> list[float] | None:
    """World yaw of the anatomical forward axis, from the rotated emitter offset.

    The trajectory bank applies the registry emitter offset as a pure yaw
    rotation about +Y on top of the root path, so the horizontal component of
    ``emitter - root`` reveals the body yaw whenever the offset and the
    declared anatomical forward axis both have a horizontal component.
    Returns ``None`` when the geometry cannot disambiguate facing (for
    example the human mouth anchor, which sits directly above the root).
    """

    offset_horizontal = math.hypot(float(offset_m[0]), float(offset_m[2]))
    if offset_horizontal <= HEADING_MIN_HORIZONTAL_OFFSET_M:
        return None
    if forward_axis_local is None:
        return None
    try:
        forward = np.asarray(forward_axis_local, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QAFactTableError("anatomical forward axis must be numeric") from exc
    if forward.shape != (3,) or not np.all(np.isfinite(forward)):
        raise QAFactTableError("anatomical forward axis must be a finite 3-vector")
    forward_horizontal = math.hypot(float(forward[0]), float(forward[2]))
    if forward_horizontal <= HEADING_MIN_HORIZONTAL_OFFSET_M:
        return None
    delta = emitter - root
    world_angle = np.degrees(np.arctan2(-delta[:, 0], -delta[:, 2]))
    offset_angle = _anchor_convention_angle_deg(float(offset_m[0]), float(offset_m[2]))
    forward_angle = _anchor_convention_angle_deg(float(forward[0]), float(forward[2]))
    yaw = world_angle - offset_angle + forward_angle
    yaw = np.mod(yaw + 180.0, 360.0) - 180.0
    return [float(value) for value in yaw]


def _track_summary(track: Mapping[str, list[float]], moving: list[bool]) -> dict[str, Any]:
    azimuth = np.asarray(track["azimuth_deg"], dtype=np.float64)
    distance = np.asarray(track["distance_m"], dtype=np.float64)
    return {
        "moving_frame_count": int(sum(1 for value in moving if value)),
        "distance_min_m": float(np.min(distance)),
        "distance_max_m": float(np.max(distance)),
        "distance_mean_m": float(np.mean(distance)),
        "azimuth_min_deg": float(np.min(azimuth)),
        "azimuth_max_deg": float(np.max(azimuth)),
        "azimuth_at_first_frame_deg": float(azimuth[0]),
        "azimuth_at_last_frame_deg": float(azimuth[-1]),
    }


def _sound_event(
    *,
    slot_id: str,
    asset: Mapping[str, Any],
    dry_variant: Mapping[str, Any],
    frame_count: int,
    ticks_per_frame: int,
    audio_sample_count: int,
) -> dict[str, Any]:
    record = dry_variant.get("record")
    if not isinstance(record, Mapping):
        raise QAFactTableError(f"{slot_id}: dry variant is missing its record")
    source_input = record.get("input")
    if not isinstance(source_input, Mapping) or not source_input.get("sha256"):
        raise QAFactTableError(f"{slot_id}: dry variant record lacks input provenance")
    identity = asset.get("identity")
    if not isinstance(identity, Mapping) or not identity.get("species_id"):
        raise QAFactTableError(f"{slot_id}: registry asset lacks identity.species_id")
    return {
        "event_id": f"{slot_id}_event_000",
        "source_slot_id": slot_id,
        "asset_id": asset.get("asset_id"),
        "sound_class": {
            "species_id": identity["species_id"],
            "display_label": asset.get("display_label"),
        },
        "start_frame": 0,
        "end_frame": frame_count,
        "start_tick": 0,
        "end_tick": frame_count * ticks_per_frame,
        "start_sample": 0,
        "end_sample": audio_sample_count,
        "active_full_window": True,
        "window_authority": "asset_bound_batch_continuous_v1",
        "dry_variant": {
            "variant_index": dry_variant.get("variant_index"),
            "input_path": source_input.get("path"),
            "input_sha256": source_input["sha256"],
            "linear_gain": record.get("linear_gain"),
        },
    }


def _instance_entry(
    *,
    slot_id: str,
    asset: Mapping[str, Any],
    emitter_anchor: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    identity = asset.get("identity")
    realized = asset.get("realized_attributes")
    if not isinstance(identity, Mapping) or not isinstance(realized, Mapping):
        raise QAFactTableError(
            f"asset {asset.get('asset_id')!r} lacks identity or realized attributes"
        )
    coat = realized.get("coat_profile")
    return {
        "instance_id": slot_id,
        "source_slot_id": slot_id,
        "asset_id": asset.get("asset_id"),
        "display_label": asset.get("display_label"),
        "entity_class": asset.get("entity_class"),
        "species_id": identity.get("species_id"),
        "breed_id": identity.get("breed_id"),
        "attributes": {
            "size": realized.get("size"),
            "body_build": realized.get("body_build"),
            "life_stage": realized.get("life_stage"),
            "coat_profile_id": coat.get("profile_id") if isinstance(coat, Mapping) else None,
            "coat_value": coat.get("value") if isinstance(coat, Mapping) else None,
        },
        "emitter": {
            "anchor_id": emitter_anchor.get("anchor_id"),
            "anchor_type": emitter_anchor.get("anchor_type"),
            "offset_m": [float(v) for v in emitter_anchor["offset_m"]],
            "offset_space": emitter_anchor.get("offset_space"),
        },
        "registry": {
            "registry_id": registry.get("registry_id"),
            "revision": registry.get("revision"),
            "asset_revision": asset.get("revision"),
            "admission_state": asset.get("admission_state"),
        },
    }


def _anchor_entries(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in anchors:
        anchor_id = anchor.get("anchor_id")
        position = anchor.get("position_m")
        if not isinstance(anchor_id, str) or not anchor_id:
            raise QAFactTableError("anchor entries must declare anchor_id")
        if anchor_id in seen:
            raise QAFactTableError(f"duplicate anchor id {anchor_id!r}")
        seen.add(anchor_id)
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise QAFactTableError(f"anchor {anchor_id!r} must have a finite position")
        yaw = anchor.get("yaw_deg")
        entries.append(
            {
                "anchor_id": anchor_id,
                "kind": anchor.get("kind"),
                "position_m": [float(v) for v in point],
                "yaw_deg": float(yaw) if isinstance(yaw, (int, float)) else None,
            }
        )
    return entries


def compile_episode_fact_table(
    *,
    bank_header: Mapping[str, Any],
    bank_episode: Mapping[str, Any],
    listener_position_m: Any,
    listener_orientation_wxyz: Any,
    sample_entry: Mapping[str, Any],
    dry_variants_by_slot: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    room: Mapping[str, Any],
    camera: Mapping[str, Any],
    rir_cache_request_identity_sha256: str,
    provenance_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile one episode's fact table from already-frozen artifacts."""

    if not isinstance(camera, Mapping):
        raise QAFactTableError("camera must declare hfov_degrees and resolution_hw")
    camera_hfov = camera.get("hfov_degrees")
    camera_resolution = camera.get("resolution_hw")
    if (
        not isinstance(camera_resolution, Sequence)
        or len(camera_resolution) != 2
        or not all(isinstance(value, int) for value in camera_resolution)
    ):
        raise QAFactTableError("camera resolution_hw must be two integers")

    episode_id = bank_episode.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise QAFactTableError("bank episode must declare episode_id")

    frame_count = bank_header.get("frame_count")
    frame_rate_hz = bank_header.get("frame_rate_hz")
    if not isinstance(frame_count, int) or frame_count < 2:
        raise QAFactTableError("bank header frame_count must be an integer >= 2")
    if not isinstance(frame_rate_hz, (int, float)) or frame_rate_hz <= 0:
        raise QAFactTableError("bank header frame_rate_hz must be positive")
    if TIME_BASE_HZ % int(frame_rate_hz) != 0 or frame_rate_hz != int(frame_rate_hz):
        raise QAFactTableError("frame rate must divide the 48 kHz timeline base")
    ticks_per_frame = TIME_BASE_HZ // int(frame_rate_hz)
    duration_seconds = frame_count / float(frame_rate_hz)

    declared_duration = bank_header.get("seconds_per_episode")
    if isinstance(declared_duration, (int, float)) and not math.isclose(
        float(declared_duration), duration_seconds, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise QAFactTableError("bank header duration disagrees with frame contract")

    slots = bank_header.get("source_slots")
    if not isinstance(slots, Sequence) or not slots:
        raise QAFactTableError("bank header must declare source_slots")

    audio_record = sample_entry.get("audio")
    if not isinstance(audio_record, Mapping):
        raise QAFactTableError("sample entry must carry an audio record")
    audio_sample_rate = audio_record.get("sample_rate_hz")
    audio_sample_count = audio_record.get("sample_count")
    if not isinstance(audio_sample_rate, int) or not isinstance(audio_sample_count, int):
        raise QAFactTableError("sample entry audio must declare integer rate and count")
    if not math.isclose(
        audio_sample_count / audio_sample_rate,
        duration_seconds,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise QAFactTableError("audio duration disagrees with the frame contract")
    mixture = audio_record.get("mixture")
    if not isinstance(mixture, Mapping) or not mixture.get("audio_sha256"):
        raise QAFactTableError("sample entry must carry mixture provenance")

    asset_ids = sample_entry.get("asset_ids_by_source_slot")
    if not isinstance(asset_ids, Mapping):
        raise QAFactTableError("sample entry must map source slots to asset ids")

    quaternion = _unit_quaternion_wxyz(listener_orientation_wxyz)
    listener_point = np.asarray(listener_position_m, dtype=np.float64)
    if listener_point.shape != (3,) or not np.all(np.isfinite(listener_point)):
        raise QAFactTableError("listener position must be a finite xyz point")
    listener_forward = _rotate_by_quaternion(
        np.asarray([0.0, 0.0, -1.0], dtype=np.float64), quaternion
    )
    listener_yaw_deg = _yaw_deg_of_world_forward(listener_forward[[0, 2]])

    center_paths = bank_episode.get("source_center_paths_m")
    root_paths = bank_episode.get("source_root_paths_m")
    if not isinstance(center_paths, Mapping) or not isinstance(root_paths, Mapping):
        raise QAFactTableError("bank episode must carry center and root path families")

    instances: list[dict[str, Any]] = []
    sound_events: list[dict[str, Any]] = []
    instance_tracks: dict[str, dict[str, Any]] = {}
    emitters_by_slot: dict[str, np.ndarray] = {}
    frustum_by_slot: dict[str, dict[str, Any]] = {}
    for slot_id in slots:
        if slot_id not in center_paths or slot_id not in root_paths:
            raise QAFactTableError(f"bank episode lacks paths for slot {slot_id!r}")
        emitter = _as_float_array(center_paths[slot_id], name=f"{slot_id} emitter path")
        root = _as_float_array(root_paths[slot_id], name=f"{slot_id} root path")
        if emitter.shape[0] != frame_count or root.shape[0] != frame_count:
            raise QAFactTableError(
                f"{slot_id}: path lengths must equal the declared frame count"
            )
        asset_id = asset_ids.get(slot_id)
        if not isinstance(asset_id, str) or not asset_id:
            raise QAFactTableError(f"sample entry lacks an asset id for {slot_id!r}")
        asset = _registry_asset(registry, asset_id)
        emitter_anchor = _default_emitter_anchor(asset)
        offset_m = np.asarray(emitter_anchor["offset_m"], dtype=np.float64)
        _check_emitter_offset_consistency(
            slot_id=slot_id, root=root, emitter=emitter, offset_m=offset_m
        )
        dry_variant = dry_variants_by_slot.get(slot_id)
        if not isinstance(dry_variant, Mapping):
            raise QAFactTableError(f"dry variant for slot {slot_id!r} is missing")

        instances.append(
            _instance_entry(
                slot_id=slot_id,
                asset=asset,
                emitter_anchor=emitter_anchor,
                registry=registry,
            )
        )
        sound_events.append(
            _sound_event(
                slot_id=slot_id,
                asset=asset,
                dry_variant=dry_variant,
                frame_count=frame_count,
                ticks_per_frame=ticks_per_frame,
                audio_sample_count=audio_sample_count,
            )
        )

        speed = _speed_track_mps(root, float(frame_rate_hz))
        moving = [value > MOVING_SPEED_THRESHOLD_MPS for value in speed]
        doa = listener_local_spherical_track(emitter, listener_point, quaternion)
        timeline_block = asset.get("timeline")
        facing = _facing_track_yaw_deg(
            root=root,
            emitter=emitter,
            offset_m=offset_m,
            forward_axis_local=(
                timeline_block.get("local_anatomical_forward_axis")
                if isinstance(timeline_block, Mapping)
                else None
            ),
        )
        frustum = center_frustum_track(
            emitter,
            listener_point,
            quaternion,
            hfov_degrees=camera_hfov,
            resolution_hw=(camera_resolution[0], camera_resolution[1]),
        )
        emitters_by_slot[slot_id] = emitter
        frustum_by_slot[slot_id] = frustum
        instance_tracks[slot_id] = {
            "root_position_m": [[float(v) for v in row] for row in root],
            "emitter_position_m": [[float(v) for v in row] for row in emitter],
            "speed_mps": speed,
            "moving": moving,
            "moving_threshold_mps": MOVING_SPEED_THRESHOLD_MPS,
            "facing_yaw_deg": facing,
            "doa": doa,
            "summary": _track_summary(doa, moving),
        }

    pairwise: dict[str, dict[str, Any]] = {}
    slot_list = list(slots)
    for index, slot_a in enumerate(slot_list):
        for slot_b in slot_list[index + 1 :]:
            distances = np.linalg.norm(
                emitters_by_slot[slot_a] - emitters_by_slot[slot_b], axis=1
            )
            pairwise[f"{slot_a}__{slot_b}"] = {
                "emitter_distance_m": [float(value) for value in distances],
                "min_m": float(np.min(distances)),
                "max_m": float(np.max(distances)),
                "mean_m": float(np.mean(distances)),
            }

    anchor_entries = _anchor_entries(anchors)
    anchor_relations: dict[str, dict[str, Any]] = {}
    for slot_id in slot_list:
        emitter = emitters_by_slot[slot_id]
        per_anchor: dict[str, Any] = {}
        for anchor in anchor_entries:
            if anchor["kind"] == "camera_listener_pose":
                continue
            distances = np.linalg.norm(
                emitter - np.asarray(anchor["position_m"], dtype=np.float64)[None, :],
                axis=1,
            )
            per_anchor[anchor["anchor_id"]] = {
                "min_m": float(np.min(distances)),
                "max_m": float(np.max(distances)),
                "mean_m": float(np.mean(distances)),
                "argmin_frame": int(np.argmin(distances)),
            }
        anchor_relations[slot_id] = per_anchor

    if not isinstance(rir_cache_request_identity_sha256, str) or len(
        rir_cache_request_identity_sha256
    ) != 64:
        raise QAFactTableError("rir cache identity must be a sha256 hex digest")

    return {
        "schema": FACT_TABLE_SCHEMA,
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "episode_id": episode_id,
        "motion_case": bank_episode.get("motion_case"),
        "room": {
            "room_capsule_id": room.get("room_capsule_id"),
            "revision": room.get("revision"),
        },
        "time": {
            "frame_count": frame_count,
            "frame_rate_hz": int(frame_rate_hz),
            "duration_seconds": duration_seconds,
            "time_base_hz": TIME_BASE_HZ,
            "ticks_per_frame": ticks_per_frame,
            "audio_sample_rate_hz": audio_sample_rate,
            "audio_sample_count": audio_sample_count,
        },
        "listener": {
            "position_m": [float(v) for v in listener_point],
            "orientation_wxyz": [float(v) for v in quaternion],
            "forward_world": [float(v) for v in listener_forward],
            "yaw_deg": listener_yaw_deg,
            "static": True,
            "azimuth_convention": AZIMUTH_CONVENTION,
        },
        "instances": instances,
        "sound_events": sound_events,
        "tracks": {
            "per_frame_count": frame_count,
            "instances": instance_tracks,
            "pairwise": pairwise,
        },
        "anchors": anchor_entries,
        "relations": {
            "allocentric_status": "deferred_pending_furniture_obb_snapshot",
            "anchor_distances": anchor_relations,
        },
        "visibility": {
            "status": "computed_center_point_v0",
            "authority": FRUSTUM_AUTHORITY,
            "hfov_degrees": float(camera_hfov),
            "resolution_hw": [int(camera_resolution[0]), int(camera_resolution[1])],
            "per_instance": {
                slot_id: {
                    "in_frustum": frustum["in_frustum"],
                    "in_frustum_frame_count": frustum["in_frustum_frame_count"],
                    "always_outside_frustum": frustum["always_outside_frustum"],
                    "always_inside_frustum": frustum["always_inside_frustum"],
                }
                for slot_id, frustum in frustum_by_slot.items()
            },
            "pixel_truth": "pending_P1_amodal_modal_pass",
        },
        "frame_events": {
            "status": "computed_center_point_v0",
            "authority": FRUSTUM_AUTHORITY,
            "events": sorted(
                (
                    {"instance_id": slot_id, **event}
                    for slot_id, frustum in frustum_by_slot.items()
                    for event in frustum["events"]
                ),
                key=lambda event: (event["frame"], event["instance_id"]),
            ),
        },
        "flags": {
            "status": "not_evaluated",
            "reason": (
                "the asset-bound batch line retains m5_1 semantic flags as "
                "not_evaluated; re-evaluation is deferred"
            ),
        },
        "audio": {
            "mixture_path": mixture.get("path"),
            "mixture_sha256": mixture["audio_sha256"],
            "sample_rate_hz": audio_sample_rate,
            "sample_count": audio_sample_count,
            "channel_count": audio_record.get("channel_count"),
            "peak_absolute": audio_record.get("peak_absolute"),
            "rir_cache_request_identity_sha256": rir_cache_request_identity_sha256,
        },
        "provenance": {"inputs": [dict(record) for record in provenance_inputs]},
    }
