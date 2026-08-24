"""Native-evidence helpers for QA v2 DoA, off-screen, and refusal questions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EPISODE_FRAME_COUNT = 75
EPISODE_FRAME_RATE_HZ = 15
EPISODE_TICKS_PER_FRAME = 3_200
EPISODE_AUDIO_SAMPLE_COUNT = 80_000
T5_FULL_CREDIT_DEG = 15.0
T5_PARTIAL_CREDIT_DEG = 30.0
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


class ExtendedQuestionEvidenceError(ValueError):
    """Native evidence cannot support an extended question."""


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtendedQuestionEvidenceError(f"cannot load JSON evidence: {path}") from exc


def _finite_numbers(value: Any, count: int, *, owner: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != count:
        raise ExtendedQuestionEvidenceError(f"{owner} must contain {count} numbers")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ExtendedQuestionEvidenceError(f"{owner} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ExtendedQuestionEvidenceError(f"{owner} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtendedQuestionEvidenceError(f"{owner} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ExtendedQuestionEvidenceError(f"{owner} must be finite")
    return result


def _yaw_error_degrees(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def event_frame_index(start_sample: Any) -> int:
    if isinstance(start_sample, bool) or not isinstance(start_sample, int):
        raise ExtendedQuestionEvidenceError("event start_sample must be an integer")
    if not 0 <= start_sample < EPISODE_AUDIO_SAMPLE_COUNT:
        raise ExtendedQuestionEvidenceError("event start_sample is outside the 80,000-sample episode")
    return min(
        EPISODE_FRAME_COUNT - 1,
        round(start_sample * EPISODE_FRAME_COUNT / EPISODE_AUDIO_SAMPLE_COUNT),
    )


def event_frame_window(start_sample: Any, end_sample_exclusive: Any) -> tuple[int, ...]:
    """Return every authored frame touched by a non-empty audio event."""

    start_frame = event_frame_index(start_sample)
    if (
        isinstance(end_sample_exclusive, bool)
        or not isinstance(end_sample_exclusive, int)
        or not int(start_sample) < end_sample_exclusive <= EPISODE_AUDIO_SAMPLE_COUNT
    ):
        raise ExtendedQuestionEvidenceError("event end_sample_exclusive is invalid")
    end_frame = event_frame_index(end_sample_exclusive - 1)
    return tuple(range(start_frame, end_frame + 1))


def listener_local_azimuth_deg_ue(
    source_position_ue_cm: Sequence[float],
    camera_position_ue_cm: Sequence[float],
    camera_yaw_deg: float,
) -> float:
    """UE Z-up listener-local azimuth in [-180, 180], positive to the right."""

    source = _finite_numbers(source_position_ue_cm, 3, owner="source position")
    camera = _finite_numbers(camera_position_ue_cm, 3, owner="camera position")
    yaw = _finite_number(camera_yaw_deg, owner="camera yaw")
    theta = math.radians(yaw)
    forward_x, forward_y = math.cos(theta), math.sin(theta)
    right_x, right_y = -forward_y, forward_x
    dx, dy = source[0] - camera[0], source[1] - camera[1]
    if math.hypot(dx, dy) <= 1.0e-9:
        raise ExtendedQuestionEvidenceError("source and listener have the same horizontal position")
    local_forward = dx * forward_x + dy * forward_y
    local_right = dx * right_x + dy * right_y
    azimuth = math.degrees(math.atan2(local_right, local_forward))
    if math.isclose(azimuth, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return 0.0
    return float(azimuth)


def parse_numeric_degrees(value: Any) -> float:
    """Parse one finite numeric answer, rejecting ambiguous multi-number text."""

    if isinstance(value, bool):
        raise ExtendedQuestionEvidenceError("numeric degree answer must not be boolean")
    if isinstance(value, (int, float)):
        return _finite_number(value, owner="numeric degree answer")
    if not isinstance(value, str):
        raise ExtendedQuestionEvidenceError("numeric degree answer must be text or a number")
    matches = _NUMBER_PATTERN.findall(value.strip())
    if len(matches) != 1:
        raise ExtendedQuestionEvidenceError(
            "numeric degree answer must contain exactly one number"
        )
    return _finite_number(float(matches[0]), owner="numeric degree answer")


def score_t5_doa_prediction(prediction: Any, gold_degrees: Any) -> dict[str, Any]:
    """Execute the approved circular +/-15 degree and +/-30 degree bands."""

    predicted = parse_numeric_degrees(prediction)
    gold = parse_numeric_degrees(gold_degrees)
    if not -180.0 <= gold <= 180.0:
        raise ExtendedQuestionEvidenceError("T5 gold must be in [-180, 180] degrees")
    error = abs((predicted - gold + 180.0) % 360.0 - 180.0)
    if error <= T5_FULL_CREDIT_DEG:
        score, band = 1.0, "full_credit"
    elif error <= T5_PARTIAL_CREDIT_DEG:
        score, band = 0.5, "partial_credit"
    else:
        score, band = 0.0, "outside_band"
    return {
        "score": score,
        "band": band,
        "prediction_deg": predicted,
        "gold_deg": gold,
        "absolute_circular_error_deg": error,
        "full_credit_within_deg": T5_FULL_CREDIT_DEG,
        "partial_credit_within_deg": T5_PARTIAL_CREDIT_DEG,
    }


@dataclass(frozen=True)
class NativePointEvidence:
    """Validated UE camera and source-anchor readbacks for one 75-frame point."""

    point_id: str
    frame_records_path: Path
    receipt_path: Path
    camera_positions_ue_cm: tuple[tuple[float, float, float], ...]
    camera_yaws_deg: tuple[float, ...]
    source_positions_ue_cm: Mapping[str, tuple[tuple[float, float, float], ...]]

    def azimuth_deg(self, source_slot: str, frame_index: int) -> float:
        if source_slot not in self.source_positions_ue_cm:
            raise ExtendedQuestionEvidenceError(f"unknown source slot: {source_slot}")
        if not 0 <= frame_index < EPISODE_FRAME_COUNT:
            raise ExtendedQuestionEvidenceError(f"frame index is out of range: {frame_index}")
        return listener_local_azimuth_deg_ue(
            self.source_positions_ue_cm[source_slot][frame_index],
            self.camera_positions_ue_cm[frame_index],
            self.camera_yaws_deg[frame_index],
        )


def _validate_receipt(receipt: Any, *, point_id: str) -> None:
    if not isinstance(receipt, Mapping):
        raise ExtendedQuestionEvidenceError(f"{point_id}: research receipt must be an object")
    if receipt.get("research_only") is not True or receipt.get("episode_counted") is not False:
        raise ExtendedQuestionEvidenceError(
            f"{point_id}: capture receipt must remain research_only and episode_counted=false"
        )
    capture = receipt.get("capture")
    if not isinstance(capture, Mapping):
        raise ExtendedQuestionEvidenceError(f"{point_id}: capture receipt is missing capture")
    expected = {
        "completed_frame_count": EPISODE_FRAME_COUNT,
        "frame_count": EPISODE_FRAME_COUNT,
        "frame_rate_hz": EPISODE_FRAME_RATE_HZ,
        "ticks_per_frame": EPISODE_TICKS_PER_FRAME,
    }
    for key, value in expected.items():
        if capture.get(key) != value:
            raise ExtendedQuestionEvidenceError(
                f"{point_id}: capture receipt {key} must equal {value}"
            )
    summary = capture.get("root_readback_summary")
    camera_summary = summary.get("camera") if isinstance(summary, Mapping) else None
    if not isinstance(camera_summary, Mapping) or camera_summary.get("status") != "pass":
        raise ExtendedQuestionEvidenceError(f"{point_id}: camera readback summary did not pass")
    for slot in ("source1", "source2"):
        matches = [
            row
            for key, row in summary.items()
            if isinstance(key, str) and key.startswith(slot) and isinstance(row, Mapping)
        ]
        if len(matches) != 1 or matches[0].get("status") != "pass":
            raise ExtendedQuestionEvidenceError(f"{point_id}: {slot} root readback did not pass")


def load_native_point_evidence(
    capture_root: str | Path,
    *,
    point_id: str,
    expected_camera_position_ue_cm: Sequence[float],
    expected_camera_yaw_deg: float,
    tolerance: float = 1.0e-6,
) -> NativePointEvidence:
    """Load real UE readbacks and cross-check them against viewpoint authority."""

    point_dir = Path(capture_root) / point_id
    records_path = point_dir / "frame_records.json"
    receipt_path = point_dir / "research_receipt.json"
    records = _load_json(records_path)
    receipt = _load_json(receipt_path)
    _validate_receipt(receipt, point_id=point_id)
    frames = records.get("frames") if isinstance(records, Mapping) else None
    if not isinstance(frames, list) or len(frames) != EPISODE_FRAME_COUNT:
        raise ExtendedQuestionEvidenceError(f"{point_id}: frame_records must contain 75 frames")

    expected_position = _finite_numbers(
        expected_camera_position_ue_cm, 3, owner="expected camera position"
    )
    expected_yaw = _finite_number(expected_camera_yaw_deg, owner="expected camera yaw")
    camera_positions: list[tuple[float, float, float]] = []
    camera_yaws: list[float] = []
    source_positions: dict[str, list[tuple[float, float, float]]] = {
        "source1": [],
        "source2": [],
    }
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != index:
            raise ExtendedQuestionEvidenceError(f"{point_id}: frame indices must be 0..74")
        if frame.get("pts_ticks") != index * EPISODE_TICKS_PER_FRAME:
            raise ExtendedQuestionEvidenceError(f"{point_id}: frame {index} has wrong pts_ticks")
        pose = frame.get("camera_pose")
        camera = frame.get("camera")
        if not isinstance(pose, Mapping) or not isinstance(camera, Mapping):
            raise ExtendedQuestionEvidenceError(f"{point_id}: frame {index} lacks camera readback")
        position = _finite_numbers(pose.get("location_cm"), 3, owner="camera_pose.location_cm")
        rotation = _finite_numbers(pose.get("rotation_deg"), 3, owner="camera_pose.rotation_deg")
        authored_position = _finite_numbers(
            camera.get("translation_ue_cm"), 3, owner="camera.translation_ue_cm"
        )
        authored_yaw = _finite_number(camera.get("yaw_ue_deg"), owner="camera.yaw_ue_deg")
        yaw = rotation[2]
        if max(abs(a - b) for a, b in zip(position, authored_position, strict=True)) > tolerance:
            raise ExtendedQuestionEvidenceError(f"{point_id}: camera pose/readback mismatch at frame {index}")
        if _yaw_error_degrees(yaw, authored_yaw) > tolerance:
            raise ExtendedQuestionEvidenceError(f"{point_id}: camera yaw/readback mismatch at frame {index}")
        if max(abs(a - b) for a, b in zip(position, expected_position, strict=True)) > tolerance:
            raise ExtendedQuestionEvidenceError(f"{point_id}: camera position disagrees with viewpoint bank")
        if _yaw_error_degrees(yaw, expected_yaw) > tolerance:
            raise ExtendedQuestionEvidenceError(f"{point_id}: camera yaw disagrees with viewpoint bank")
        camera_positions.append(position)
        camera_yaws.append(yaw)

        anchors = frame.get("actor_anchor_poses")
        states = frame.get("actor_states")
        if not isinstance(anchors, Mapping) or not isinstance(states, list):
            raise ExtendedQuestionEvidenceError(f"{point_id}: frame {index} lacks actor readbacks")
        state_by_slot = {
            state.get("source_slot_id"): state
            for state in states
            if isinstance(state, Mapping)
        }
        for slot in ("source1", "source2"):
            anchor = anchors.get(slot)
            state = state_by_slot.get(slot)
            if not isinstance(anchor, Mapping) or not isinstance(state, Mapping):
                raise ExtendedQuestionEvidenceError(f"{point_id}: frame {index} lacks {slot}")
            anchor_position = _finite_numbers(
                anchor.get("location_cm"), 3, owner=f"{slot}.anchor.location_cm"
            )
            state_position = _finite_numbers(
                state.get("translation_ue_cm"), 3, owner=f"{slot}.translation_ue_cm"
            )
            if max(
                abs(a - b)
                for a, b in zip(anchor_position, state_position, strict=True)
            ) > tolerance:
                raise ExtendedQuestionEvidenceError(
                    f"{point_id}: {slot} anchor/state mismatch at frame {index}"
                )
            source_positions[slot].append(anchor_position)

    return NativePointEvidence(
        point_id=point_id,
        frame_records_path=records_path.resolve(),
        receipt_path=receipt_path.resolve(),
        camera_positions_ue_cm=tuple(camera_positions),
        camera_yaws_deg=tuple(camera_yaws),
        source_positions_ue_cm={
            slot: tuple(values) for slot, values in source_positions.items()
        },
    )


def load_native_pixel_visibility(
    visibility_root: str | Path,
    *,
    point_id: str,
) -> Mapping[str, Any]:
    path = Path(visibility_root) / point_id / "pixel_visibility_truth.json"
    document = _load_json(path)
    if not isinstance(document, Mapping):
        raise ExtendedQuestionEvidenceError(f"{point_id}: pixel truth must be an object")
    if (
        document.get("schema") != "avengine_qa_pixel_visibility_truth_v1"
        or document.get("status") != "computed_modal_target_only_v1"
        or document.get("frame_indices") != list(range(EPISODE_FRAME_COUNT))
    ):
        raise ExtendedQuestionEvidenceError(f"{point_id}: native pixel truth is not complete")
    per_instance = document.get("per_instance")
    if not isinstance(per_instance, Mapping) or set(per_instance) != {"source1", "source2"}:
        raise ExtendedQuestionEvidenceError(f"{point_id}: pixel truth must cover source1/source2")
    for slot in ("source1", "source2"):
        frames = per_instance[slot].get("frames") if isinstance(per_instance[slot], Mapping) else None
        if not isinstance(frames, list) or len(frames) != EPISODE_FRAME_COUNT:
            raise ExtendedQuestionEvidenceError(f"{point_id}: {slot} pixel truth has wrong length")
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping) or frame.get("frame_index") != index:
                raise ExtendedQuestionEvidenceError(
                    f"{point_id}: {slot} pixel truth frame indices must be 0..74"
                )
            visible = frame.get("visible_pixels")
            target = frame.get("target_pixels")
            state = frame.get("state")
            if (
                isinstance(visible, bool)
                or not isinstance(visible, int)
                or visible < 0
                or isinstance(target, bool)
                or not isinstance(target, int)
                or target < visible
            ):
                raise ExtendedQuestionEvidenceError(
                    f"{point_id}: {slot} pixel counts are inconsistent at frame {index}"
                )
            expected_state = (
                "out_of_view"
                if target == 0
                else "visible_clear"
                if visible == target
                else "fully_occluded"
                if visible == 0
                else "visible_occluded"
            )
            if state != expected_state:
                raise ExtendedQuestionEvidenceError(
                    f"{point_id}: {slot} pixel state is inconsistent at frame {index}"
                )
    return document


def require_native_out_of_view(
    pixel_truth: Mapping[str, Any],
    *,
    source_slot: str,
    frame_index: int,
) -> Mapping[str, Any]:
    try:
        frame = pixel_truth["per_instance"][source_slot]["frames"][frame_index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExtendedQuestionEvidenceError("pixel truth does not cover the event frame") from exc
    if (
        not isinstance(frame, Mapping)
        or frame.get("frame_index") != frame_index
        or frame.get("state") != "out_of_view"
        or frame.get("visible_pixels") != 0
        or frame.get("target_pixels") != 0
    ):
        raise ExtendedQuestionEvidenceError(
            f"{source_slot} is not native-pixel out_of_view at frame {frame_index}"
        )
    return frame


def choose_refusal_quota(
    base_question_count: int,
    candidate_count: int,
    *,
    minimum_fraction: float = 0.10,
    maximum_fraction: float = 0.15,
    target_fraction: float = 0.125,
) -> tuple[int, tuple[int, ...], float]:
    """Choose a deterministic in-range refusal count without deleting base rows."""

    if base_question_count <= 0 or candidate_count <= 0:
        raise ExtendedQuestionEvidenceError("refusal quota needs base questions and candidates")
    if not 0.0 < minimum_fraction <= target_fraction <= maximum_fraction < 1.0:
        raise ExtendedQuestionEvidenceError("refusal quota fractions are invalid")
    options = []
    for count in range(1, candidate_count + 1):
        fraction = count / (base_question_count + count)
        if minimum_fraction <= fraction <= maximum_fraction:
            options.append((abs(fraction - target_fraction), count, fraction))
    if not options:
        raise ExtendedQuestionEvidenceError(
            "available refusal candidates cannot satisfy the requested 10-15% quota"
        )
    _, selected_count, fraction = min(options)
    indices = tuple(
        ((2 * ordinal + 1) * candidate_count) // (2 * selected_count)
        for ordinal in range(selected_count)
    )
    if len(set(indices)) != selected_count:
        raise ExtendedQuestionEvidenceError("refusal candidate selection was not unique")
    return selected_count, indices, fraction


__all__ = [
    "EPISODE_AUDIO_SAMPLE_COUNT",
    "EPISODE_FRAME_COUNT",
    "ExtendedQuestionEvidenceError",
    "NativePointEvidence",
    "T5_FULL_CREDIT_DEG",
    "T5_PARTIAL_CREDIT_DEG",
    "choose_refusal_quota",
    "event_frame_index",
    "event_frame_window",
    "listener_local_azimuth_deg_ue",
    "load_native_pixel_visibility",
    "load_native_point_evidence",
    "parse_numeric_degrees",
    "require_native_out_of_view",
    "score_t5_doa_prediction",
]
