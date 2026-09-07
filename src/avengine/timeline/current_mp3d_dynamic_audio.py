"""Motion-following research audio for authored current routes.

The static ``render_current_m1_research_audio`` verb stays the frozen
baseline: its pair IRs carry one state per source, so mixes hold no motion.
This module renders the audio the visual capture actually shows instead:
captured per-frame emitter positions drive a strided keyframe grid through
the persistent-context M5.1 review renderer, and the dry buses come from an
M6 AudioProgram routing variant, so the probe sources take turns instead of
sharing one schedule. The core ``render_dynamic_research_audio`` is
room-agnostic; ``render_current_mp3d_dynamic_audio`` binds it to the current
MP3D capture record layout.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math
import os
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.contracts.transforms import compose_transforms
from avengine.capture.dry_audio import DryAudioClipSpec, assemble_dry_audio_buses
from avengine.capture.neutral_readback import validate_neutral_readback
from avengine.rooms.contracts import validate_capture_request
from avengine.acoustics.runtime import load_compiled_acoustic_scene
from avengine.spatial_audio.audio import write_float32_wav
from avengine.spatial_audio.current_request_pair_ir import _load_simulation_request
from avengine.spatial_audio.runtime import M4SimulationConfig
from avengine.capture.acoustics import (
    build_strided_review_keyframes,
    render_research_review_audio,
    render_research_review_rir_sequence,
    research_review_trajectory_record,
)
from avengine.timeline.acoustics import DynamicRIRSequence
from avengine.timeline.audio import time_varying_convolve
from avengine.timeline.audio_program import bind_audio_program_hash, validate_audio_program
from avengine.timeline.audio_render import (
    AudioProgramDryAssembly,
    assemble_audio_program_dry_buses,
)
from avengine.registry.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
    sound_index,
)

CURRENT_MP3D_DYNAMIC_AUDIO_SCHEMA = "avengine_m5_current_mp3d_dynamic_audio_v1"
UNIFIED_AUDIO_RECEIPT_SCHEMA = "avengine_unified_audio_receipt_v1"
DEFAULT_VISUAL_FRAME_RATE_HZ = 15
DEFAULT_EPISODE_FRAME_COUNT = 75
DEFAULT_TIMELINE_TICK_RATE_HZ = 48_000
DEFAULT_TICKS_PER_FRAME = 3_200
AUDIO_SAMPLE_RATE_HZ = 16_000
DEFAULT_EPISODE_SAMPLE_COUNT = 80_000
SUPPORTED_LAYOUTS = ("binaural", "ambisonics")
_LAYOUT_CHANNEL_COUNTS = {"binaural": 2, "ambisonics": 4}
_LAYOUT_CHANNEL_LABELS = {
    "binaural": ("left", "right"),
    "ambisonics": ("W", "Y", "Z", "X"),
}
_LAYOUT_OUTPUT_DIRS = {"binaural": "binaural", "ambisonics": "foa"}

# Backward-compatible names retained for the original current-MP3D route.
VISUAL_FRAME_RATE_HZ = DEFAULT_VISUAL_FRAME_RATE_HZ
EPISODE_FRAME_COUNT = DEFAULT_EPISODE_FRAME_COUNT
EPISODE_SAMPLE_COUNT = DEFAULT_EPISODE_SAMPLE_COUNT
CLAIM_BOUNDARY = (
    "Research review audio only. Captured per-frame source positions and one "
    "AudioProgram routing variant drive strided per-state RIRs for the "
    "requested binaural or ambisonics layouts; no dataset admission, "
    "qualification, or new gate is claimed."
)


class CurrentMP3DDynamicAudioError(RuntimeError):
    """Raised when the dynamic research-audio contract is violated."""


def _nonzero_interval(
    values: Any, *, threshold: float, offset: int = 0
) -> list[int] | None:
    array = np.asarray(values)
    if array.ndim == 1:
        mask = np.abs(array) > threshold
    elif array.ndim == 2:
        mask = np.any(np.abs(array) > threshold, axis=0)
    else:
        raise CurrentMP3DDynamicAudioError(
            "audio interval inspection expects one or two dimensions"
        )
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    return [int(offset + indices[0]), int(offset + indices[-1] + 1)]



def _clamp_wet_tail_interval(interval: Sequence[int], sample_count: int) -> tuple[list[int], bool, int]:
    """Clip a measured wet-tail interval onto the episode sample clock."""
    start = int(interval[0])
    end = int(interval[1])
    original_end = end
    clamped = end > sample_count or start > sample_count or start < 0
    if start < 0:
        start = 0
    if start > sample_count:
        start = sample_count
    if end > sample_count:
        end = sample_count
    if end <= start and sample_count > 0:
        start = max(0, sample_count - 1)
        end = sample_count
        clamped = True
    return [start, end], clamped, original_end


def _peak_dbfs(value: Any) -> float | None:
    array = np.asarray(value)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak <= 0.0:
        return None
    return float(20.0 * math.log10(peak))


def _matrix_to_orientation_wxyz(matrix: Any) -> list[float]:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise CurrentMP3DDynamicAudioError(
            "neutral camera basis cannot be converted to an orientation"
        )
    trace = float(np.trace(rotation))
    if trace > 0.0:
        w = math.sqrt(trace + 1.0) / 2.0
        x = (rotation[2, 1] - rotation[1, 2]) / max(1.0e-12, 4.0 * w)
        y = (rotation[0, 2] - rotation[2, 0]) / max(1.0e-12, 4.0 * w)
        z = (rotation[1, 0] - rotation[0, 1]) / max(1.0e-12, 4.0 * w)
    else:
        diagonal = int(np.argmax(np.diag(rotation)))
        if diagonal == 0:
            x = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) / 2.0
            y = (rotation[0, 1] + rotation[1, 0]) / max(1.0e-12, 4.0 * x)
            z = (rotation[0, 2] + rotation[2, 0]) / max(1.0e-12, 4.0 * x)
            w = (rotation[2, 1] - rotation[1, 2]) / max(1.0e-12, 4.0 * x)
        elif diagonal == 1:
            y = math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) / 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / max(1.0e-12, 4.0 * y)
            z = (rotation[1, 2] + rotation[2, 1]) / max(1.0e-12, 4.0 * y)
            w = (rotation[0, 2] - rotation[2, 0]) / max(1.0e-12, 4.0 * y)
        else:
            z = math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) / 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / max(1.0e-12, 4.0 * z)
            y = (rotation[1, 2] + rotation[2, 1]) / max(1.0e-12, 4.0 * z)
            w = (rotation[1, 0] - rotation[0, 1]) / max(1.0e-12, 4.0 * z)
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CurrentMP3DDynamicAudioError(
            "neutral camera basis produced an invalid orientation"
        )
    return (quaternion / norm).tolist()


def _neutral_camera_pose(
    neutral: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    camera = neutral.get("camera")
    if not isinstance(camera, list) or not camera:
        raise CurrentMP3DDynamicAudioError(
            "neutral readback must contain a nonempty camera series"
        )
    first = camera[0]
    if not isinstance(first, Mapping):
        raise CurrentMP3DDynamicAudioError("neutral camera rows must be objects")
    basis = first.get("basis")
    if not isinstance(basis, Mapping):
        raise CurrentMP3DDynamicAudioError("neutral camera basis is missing")
    try:
        position = [float(value) for value in first["position_m"]]
        forward = np.asarray(basis["forward"], dtype=np.float64)
        right = np.asarray(basis["right"], dtype=np.float64)
        up = np.asarray(basis["up"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise CurrentMP3DDynamicAudioError(
            "neutral camera pose is malformed"
        ) from error
    if len(position) != 3 or any(not math.isfinite(value) for value in position):
        raise CurrentMP3DDynamicAudioError("neutral camera position is malformed")
    for label, vector in (("forward", forward), ("right", right), ("up", up)):
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise CurrentMP3DDynamicAudioError(
                f"neutral camera {label} basis is malformed"
            )
    matrix = np.column_stack((right, up, -forward))
    orientation = _matrix_to_orientation_wxyz(matrix)
    for row in camera[1:]:
        if not isinstance(row, Mapping):
            raise CurrentMP3DDynamicAudioError("neutral camera rows must be objects")
        if not np.allclose(row.get("position_m"), position, atol=1.0e-6, rtol=0.0):
            raise CurrentMP3DDynamicAudioError(
                "shared dynamic renderer requires a stationary listener readback"
            )
        other_basis = row.get("basis")
        if not isinstance(other_basis, Mapping) or any(
            not np.allclose(other_basis.get(key), basis[key], atol=1.0e-6, rtol=0.0)
            for key in ("forward", "right", "up")
        ):
            raise CurrentMP3DDynamicAudioError(
                "shared dynamic renderer requires a stationary listener orientation"
            )
    return position, orientation


def _load_neutral_input(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        neutral = deepcopy(dict(value))
    else:
        path = Path(value).expanduser().resolve()
        try:
            neutral = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"cannot read neutral readback: {error}"
            ) from error
    if not isinstance(neutral, Mapping):
        raise CurrentMP3DDynamicAudioError("neutral readback must be a JSON object")
    try:
        validate_neutral_readback(neutral)
    except (TypeError, ValueError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"neutral readback validation failed: {error}"
        ) from error
    return deepcopy(dict(neutral))


def _neutral_source_trajectories(
    neutral: Mapping[str, Any],
    source_ids: Sequence[str],
    *,
    source_endpoint_by_entity: Mapping[str, str] | None = None,
) -> dict[str, list[list[float]]]:
    entities = neutral.get("entities")
    if not isinstance(entities, Mapping):
        raise CurrentMP3DDynamicAudioError("neutral readback has no entity series")
    explicit = {
        str(key): str(value)
        for key, value in (source_endpoint_by_entity or {}).items()
    }
    identities = neutral.get("entity_identities")
    if isinstance(identities, Mapping):
        for entity_id, record in identities.items():
            if entity_id not in explicit and isinstance(record, Mapping):
                endpoint = record.get("source_endpoint_id")
                if isinstance(endpoint, str) and endpoint:
                    explicit[str(entity_id)] = endpoint
    source_set = {str(value) for value in source_ids}
    result: dict[str, list[list[float]]] = {}
    for entity_id, rows in entities.items():
        endpoint = explicit.get(str(entity_id))
        if endpoint is None and str(entity_id) in source_set:
            endpoint = str(entity_id)
        if endpoint is None:
            raise CurrentMP3DDynamicAudioError(
                f"neutral entity {entity_id!r} lacks an explicit source endpoint binding"
            )
        if endpoint in result:
            raise CurrentMP3DDynamicAudioError(
                f"neutral source endpoint is bound more than once: {endpoint}"
            )
        if not isinstance(rows, list) or not rows:
            raise CurrentMP3DDynamicAudioError(
                f"neutral entity {entity_id!r} has no frame series"
            )
        positions: list[list[float]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("frame_index") != index:
                raise CurrentMP3DDynamicAudioError(
                    f"neutral entity {entity_id!r} has a missing or reordered frame"
                )
            try:
                position = [float(value) for value in row["emitter"]]
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise CurrentMP3DDynamicAudioError(
                    f"neutral entity {entity_id!r} emitter is malformed"
                ) from error
            if len(position) != 3 or any(not math.isfinite(value) for value in position):
                raise CurrentMP3DDynamicAudioError(
                    f"neutral entity {entity_id!r} emitter is malformed"
                )
            positions.append(position)
        result[endpoint] = positions
    if set(result) != source_set:
        raise CurrentMP3DDynamicAudioError(
            "neutral source endpoint IDs differ from the AudioProgram candidates: "
            f"{sorted(result)} != {sorted(source_set)}"
        )
    return {source_id: result[source_id] for source_id in source_ids}


def _direct_program_dry_assembly(
    program: Mapping[str, Any],
    *,
    variant_id: str,
    event_asset_bindings: Mapping[str, str | Path] | None,
) -> AudioProgramDryAssembly:
    """Assemble a plan-local program without inventing a registry override."""
    if variant_id != "A":
        raise CurrentMP3DDynamicAudioError(
            "direct AudioProgram materialization currently supports variant 'A'"
        )
    timeline = program.get("timeline")
    candidates = program.get("candidate_source_endpoint_ids")
    events = program.get("events")
    if not isinstance(timeline, Mapping) or not isinstance(candidates, list) or not isinstance(events, list):
        raise CurrentMP3DDynamicAudioError(
            "direct AudioProgram must declare timeline, candidates and events"
        )
    if candidates != sorted(set(candidates)) or any(
        not isinstance(value, str) or not value for value in candidates
    ):
        raise CurrentMP3DDynamicAudioError(
            "AudioProgram candidate source IDs must be unique and canonical"
        )
    try:
        clip = DryAudioClipSpec.from_values(
            frame_count=int(timeline["frame_count"]),
            fps_numerator=int(timeline["video_fps"]),
            sample_rate_hz=int(timeline["sample_rate_hz"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"direct AudioProgram timeline is invalid: {error}"
        ) from error
    if clip.sample_count != int(timeline.get("sample_count", -1)):
        raise CurrentMP3DDynamicAudioError(
            "direct AudioProgram timeline sample_count differs from its exact clock"
        )
    bindings = {
        str(key): Path(value).expanduser().resolve()
        for key, value in (event_asset_bindings or {}).items()
    }
    event_mappings: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise CurrentMP3DDynamicAudioError(
                f"AudioProgram events[{index}] must be an object"
            )
        sound_id = event.get("sound_asset_id")
        path_value = event.get("path") or bindings.get(str(sound_id))
        if not isinstance(sound_id, str) or not sound_id:
            raise CurrentMP3DDynamicAudioError(
                f"AudioProgram events[{index}] lacks sound_asset_id"
            )
        if path_value is None:
            raise CurrentMP3DDynamicAudioError(
                f"no direct dry binding was supplied for {sound_id!r}"
            )
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise CurrentMP3DDynamicAudioError(f"direct dry asset is missing: {path}")
        if str(sound_id) in bindings and bindings[str(sound_id)] != path:
            raise CurrentMP3DDynamicAudioError(
                f"direct dry binding conflicts for {sound_id!r}"
            )
        bindings.setdefault(str(sound_id), path)
        try:
            event_mappings.append(
                {
                    "event_id": str(event["event_id"]),
                    "source_id": str(event["source_endpoint_id"]),
                    "start_sample": int(event["start_sample"]),
                    "end_sample_exclusive": int(event["end_sample_exclusive"]),
                    "dry_asset_id": str(sound_id),
                    "dry_asset_sha256": sha256_file(path),
                    "dry_clip_start_sample": int(event.get("source_start_sample", 0)),
                    "dry_clip_end_sample_exclusive": (
                        int(event["source_end_sample_exclusive"])
                        if event.get("source_end_sample_exclusive") is not None
                        else None
                    ),
                    "linear_gain": float(event.get("linear_gain", 1.0)),
                    "fade_samples": int(event.get("fade_samples", 0)),
                }
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"AudioProgram events[{index}] has invalid sample fields"
            ) from error
    try:
        dry_audio = assemble_dry_audio_buses(
            event_mappings,
            source_ids=tuple(candidates),
            clip=clip,
            asset_bindings={key: str(value) for key, value in bindings.items()},
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"direct dry-audio assembly failed: {error}"
        ) from error
    return AudioProgramDryAssembly(
        materialized_program=deepcopy(dict(program)),
        compiled_program=None,
        dry_audio=dry_audio,
    )


def _validate_execution_variant(value: Any) -> str | None:
    """Validate an external execution label without constraining its vocabulary.

    variant_id belongs to AudioProgram materialization and remains A for
    pre-materialized QA-v3 programs. execution_variant labels the surrounding
    batch artifact, such as main or gateA.
    """

    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CurrentMP3DDynamicAudioError(
            "execution_variant must be a nonempty string without surrounding "
            "whitespace or control characters"
        )
    return value


def _round_fraction(value: Fraction) -> int:
    if value < 0:
        raise CurrentMP3DDynamicAudioError("timeline duration cannot be negative")
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(remainder * 2 >= value.denominator)


def _resolve_visual_clock(
    *,
    frame_count: object,
    frame_rate_hz: object,
    ticks_per_frame: object | None,
    time_base_hz: object = DEFAULT_TIMELINE_TICK_RATE_HZ,
) -> dict[str, int | float]:
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise CurrentMP3DDynamicAudioError(
            "visual frame_count must be a positive integer"
        )
    if (
        isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, (int, float))
        or not math.isfinite(float(frame_rate_hz))
        or float(frame_rate_hz) <= 0.0
    ):
        raise CurrentMP3DDynamicAudioError(
            "visual frame_rate_hz must be positive and finite"
        )
    if (
        isinstance(time_base_hz, bool)
        or not isinstance(time_base_hz, int)
        or time_base_hz < 1
    ):
        raise CurrentMP3DDynamicAudioError(
            "timeline time_base_hz must be a positive integer"
        )
    rate = float(frame_rate_hz)
    if ticks_per_frame is None:
        implied = float(time_base_hz) / rate
        rounded = int(round(implied))
        if not math.isclose(implied, rounded, rel_tol=0.0, abs_tol=1.0e-9):
            raise CurrentMP3DDynamicAudioError(
                "visual clock needs an explicit integer ticks_per_frame"
            )
        ticks = rounded
    elif (
        isinstance(ticks_per_frame, bool)
        or not isinstance(ticks_per_frame, (int, float))
        or not math.isfinite(float(ticks_per_frame))
        or float(ticks_per_frame) < 1.0
        or not float(ticks_per_frame).is_integer()
    ):
        raise CurrentMP3DDynamicAudioError(
            "visual ticks_per_frame must be a positive integer"
        )
    else:
        ticks = int(ticks_per_frame)
    if not math.isclose(
        rate * float(ticks),
        float(time_base_hz),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise CurrentMP3DDynamicAudioError(
            "visual frame_rate_hz and ticks_per_frame disagree with time_base_hz"
        )
    rate_fraction = Fraction(str(rate))
    sample_count = _round_fraction(
        Fraction(
            int(frame_count) * AUDIO_SAMPLE_RATE_HZ * rate_fraction.denominator,
            rate_fraction.numerator,
        )
    )
    if sample_count < 1:
        raise CurrentMP3DDynamicAudioError(
            "visual duration rounds to zero audio samples"
        )
    normalized_rate: int | float = int(rate) if rate.is_integer() else rate
    return {
        "frame_count": int(frame_count),
        "frame_rate_hz": normalized_rate,
        "ticks_per_frame": int(ticks),
        "time_base_hz": int(time_base_hz),
        "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
        "sample_count": sample_count,
    }


def _read_frame_records(
    visual_capture_dir: str | Path,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    records_path = Path(visual_capture_dir).resolve() / "frame_records.json"
    if not records_path.is_file():
        raise CurrentMP3DDynamicAudioError(
            f"visual capture is missing frame_records.json: {records_path}"
        )
    try:
        payload = json.loads(records_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"cannot read visual frame_records.json: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CurrentMP3DDynamicAudioError(
            "frame_records.json must contain an object"
        )
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise CurrentMP3DDynamicAudioError(
            "frame_records must contain a non-empty frames list"
        )
    if not all(isinstance(frame, Mapping) for frame in frames):
        raise CurrentMP3DDynamicAudioError(
            "frame_records entries must be objects"
        )
    return payload, frames


def load_captured_render_clock(
    visual_capture_dir: str | Path,
    *,
    frame_count: int | None = None,
    frame_rate_hz: int | float | None = None,
    ticks_per_frame: int | None = None,
) -> dict[str, int | float]:
    """Resolve one capture clock from receipt/frame records or explicit values."""
    payload, frames = _read_frame_records(visual_capture_dir)
    declared: Mapping[str, Any] = {}
    render = payload.get("render")
    if isinstance(render, Mapping):
        declared = render
    clock = payload.get("clock")
    if isinstance(clock, Mapping):
        declared = {**dict(declared), **dict(clock)}
    receipt_path = Path(visual_capture_dir).resolve() / "research_receipt.json"
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"cannot read visual research receipt: {error}"
            ) from error
        if isinstance(receipt, Mapping):
            receipt_capture = receipt.get("capture")
            if isinstance(receipt_capture, Mapping):
                declared = {**dict(declared), **dict(receipt_capture)}
    resolved = _resolve_visual_clock(
        frame_count=(
            frame_count
            if frame_count is not None
            else declared.get("frame_count", len(frames))
        ),
        frame_rate_hz=(
            frame_rate_hz
            if frame_rate_hz is not None
            else declared.get("frame_rate_hz", DEFAULT_VISUAL_FRAME_RATE_HZ)
        ),
        ticks_per_frame=(
            ticks_per_frame
            if ticks_per_frame is not None
            else declared.get("ticks_per_frame", DEFAULT_TICKS_PER_FRAME)
        ),
        time_base_hz=declared.get(
            "time_base_hz", DEFAULT_TIMELINE_TICK_RATE_HZ
        ),
    )
    if len(frames) != resolved["frame_count"]:
        raise CurrentMP3DDynamicAudioError(
            f"frame_records must carry exactly {resolved['frame_count']} frames"
        )
    for index, frame in enumerate(frames):
        if frame.get("frame_index") != index:
            raise CurrentMP3DDynamicAudioError(
                "frame_records indices must be contiguous from zero"
            )
        if "pts_ticks" in frame and frame.get("pts_ticks") != (
            index * resolved["ticks_per_frame"]
        ):
            raise CurrentMP3DDynamicAudioError(
                f"frame {index} PTS differs from the declared capture clock"
            )
    return resolved


def _fresh_output(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise CurrentMP3DDynamicAudioError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    return output


def load_captured_source_paths(
    visual_capture_dir: str | Path,
    source_ids: tuple[str, ...],
    *,
    frame_count: int | None = None,
    frame_rate_hz: int | float | None = None,
    ticks_per_frame: int | None = None,
) -> dict[str, list[list[float]]]:
    """Read per-frame source positions using the capture's declared clock.

    Frame-record slot i maps to source_ids[i]. The default clock keeps the
    legacy 75-frame behavior; a current visual receipt or explicit clock may
    declare a different duration such as 150 frames at 15 Hz.
    """
    clock = load_captured_render_clock(
        visual_capture_dir,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        ticks_per_frame=ticks_per_frame,
    )
    payload, frames = _read_frame_records(visual_capture_dir)
    if len(set(source_ids)) != len(source_ids) or any(not source_id for source_id in source_ids):
        raise CurrentMP3DDynamicAudioError("program candidate source IDs must be unique and nonempty")
    recorded_ids = payload.get("source_endpoint_ids")
    if recorded_ids is None:
        recorded_ids = list(source_ids)  # explicit legacy capture ordering
    if (not isinstance(recorded_ids, list)
            or any(not isinstance(value, str) or not value for value in recorded_ids)
            or len(recorded_ids) != len(set(recorded_ids))
            or set(recorded_ids) != set(source_ids)):
        raise CurrentMP3DDynamicAudioError(
            "captured source endpoint IDs must uniquely match the program candidates")
    index_by_id = {source_id: index for index, source_id in enumerate(recorded_ids)}
    trajectories: dict[str, list[list[float]]] = {
        source_id: [] for source_id in source_ids
    }
    for index, frame in enumerate(frames):
        positions = frame.get("source_positions_m")
        if not isinstance(positions, list) or len(positions) != len(source_ids):
            raise CurrentMP3DDynamicAudioError(
                "each frame must record one source position per program candidate"
            )
        for source_id in source_ids:
            slot = index_by_id[source_id]
            try:
                point = [float(value) for value in positions[slot]]
            except (TypeError, ValueError, OverflowError) as error:
                raise CurrentMP3DDynamicAudioError(
                    "source positions must be finite 3-vectors"
                ) from error
            if len(point) != 3 or not all(np.isfinite(point)):
                raise CurrentMP3DDynamicAudioError(
                    "source positions must be finite 3-vectors"
                )
            trajectories[source_id].append(point)
    if any(len(points) != clock["frame_count"] for points in trajectories.values()):
        raise CurrentMP3DDynamicAudioError(
            "source trajectory length differs from the capture clock"
        )
    return trajectories




def listener_pose_from_m1_request(
    m1_request: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    """Compose the static camera-colocated listener pose (wxyz orientation)."""

    errors = validate_capture_request(m1_request)
    if errors:
        raise CurrentMP3DDynamicAudioError("; ".join(errors))
    rig = m1_request["primary_camera_rig"]
    listener = m1_request["listener"]
    world_from_listener = compose_transforms(
        rig["world_from_rig"], listener["rig_from_listener"]
    )
    x, y, z, w = world_from_listener["rotation_xyzw"]
    return list(world_from_listener["translation_m"]), [w, x, y, z]


def _resolve_registry_dry_audio_path(
    uri: Any,
    *,
    sound_id: str,
    repository_root: Path,
    external_sound_asset_paths: Mapping[str, Path],
) -> Path:
    """Resolve one registry URI without treating URI text as a local path.

    ``repo://`` keeps its existing repository-relative spelling.  ``file://``
    is an absolute local URI and is decoded with the standard URI parser.
    Other legacy schemes (currently ``artifact://``) remain usable only through
    the caller's explicit per-asset deployment mapping.
    """

    if not isinstance(uri, str) or not uri:
        raise CurrentMP3DDynamicAudioError(
            f"registry dry audio URI is invalid for {sound_id}"
        )
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    if scheme == "repo":
        if not uri.startswith("repo://"):
            raise CurrentMP3DDynamicAudioError(
                f"registry dry audio URI must use repo:// for {sound_id}"
            )
        return (repository_root / uri.removeprefix("repo://")).resolve()
    if scheme == "file":
        if parsed.netloc:
            raise CurrentMP3DDynamicAudioError(
                f"file URI for {sound_id} must not contain a host"
            )
        if parsed.query or parsed.fragment:
            raise CurrentMP3DDynamicAudioError(
                f"file URI for {sound_id} must not contain query or fragment"
            )
        decoded_path = unquote(parsed.path)
        if not decoded_path.startswith("/"):
            raise CurrentMP3DDynamicAudioError(
                f"file URI for {sound_id} must contain an absolute path"
            )
        return Path(decoded_path).resolve()
    path = external_sound_asset_paths.get(sound_id)
    if path is None:
        raise CurrentMP3DDynamicAudioError(
            f"program sound {sound_id} requires an explicit external dry path "
            f"for URI scheme {scheme or '<none>'!r}"
        )
    return Path(path).resolve()


def _asset_bindings(
    sounds: Mapping[str, Any],
    *,
    repository_root: Path,
    external_sound_asset_paths: Mapping[str, Path],
    required_sound_ids: set[str],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    records = sound_index(sounds)
    for sound_id in sorted(required_sound_ids):
        record = records.get(sound_id)
        if record is None:
            raise CurrentMP3DDynamicAudioError(
                f"program sound is not registered: {sound_id}"
            )
        dry_audio = record.get("dry_audio")
        if not isinstance(dry_audio, Mapping):
            raise CurrentMP3DDynamicAudioError(
                f"registry sound has no dry_audio record: {sound_id}"
            )
        resolved = _resolve_registry_dry_audio_path(
            dry_audio.get("uri"),
            sound_id=sound_id,
            repository_root=repository_root,
            external_sound_asset_paths=external_sound_asset_paths,
        )
        if not resolved.is_file():
            raise CurrentMP3DDynamicAudioError(f"dry audio is missing: {resolved}")
        expected = str(dry_audio["sha256"])
        if sha256_file(resolved) != expected:
            raise CurrentMP3DDynamicAudioError(
                f"dry audio differs from the registry digest for {sound_id}"
            )
        result[sound_id] = {"path": str(resolved), "sha256": expected}
    return result


def _input_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _expected_program_timeline_fields(
    clock: Mapping[str, int | float],
) -> dict[str, int | float]:
    return {
        "time_base_hz": clock["time_base_hz"],
        "ticks_per_frame": clock["ticks_per_frame"],
        "video_fps": int(clock["frame_rate_hz"]),
        "frame_count": clock["frame_count"],
        "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
        "ticks_per_sample": 3,
        "sample_count": clock["sample_count"],
    }


def _program_clock_binding(
    program: Mapping[str, Any],
    clock: Mapping[str, int | float],
) -> dict[str, Any]:
    """Validate a declared AudioProgram clock without retiming it."""
    timeline = program.get("timeline")
    if not isinstance(timeline, Mapping):
        raise CurrentMP3DDynamicAudioError(
            "the AudioProgram must carry a timeline before clock binding"
        )
    expected = _expected_program_timeline_fields(clock)
    missing = [field for field in expected if field not in timeline]
    mismatches = [
        (field, timeline[field], value)
        for field, value in expected.items()
        if field in timeline and timeline[field] != value
    ]
    if mismatches:
        field, declared, visual = mismatches[0]
        raise CurrentMP3DDynamicAudioError(
            "AudioProgram timeline clock differs from the visual clock: "
            f"{field} declares {declared!r}, visual requires {visual!r}"
        )
    return {
        "mode": "legacy_default_fill" if missing else "validated_declared",
        "filled_fields": missing,
    }


def _program_for_visual_clock(
    program: Mapping[str, Any],
    clock: Mapping[str, int | float],
) -> dict[str, Any]:
    """Bind only missing historical metadata; never retime declared values."""
    binding = _program_clock_binding(program, clock)
    if not binding["filled_fields"]:
        return deepcopy(dict(program))
    result = deepcopy(dict(program))
    timeline = dict(result["timeline"])
    timeline.update(_expected_program_timeline_fields(clock))
    result["timeline"] = timeline
    return bind_audio_program_hash(result)


def _assert_no_cropped_dry_audio(assembly: Any) -> None:
    """Preserve the explicitly selected source window during placement.

    Source start/end selection is applied before these fit receipts, so an
    intentional excerpt from a longer ordinary sound remains supported. This
    rejects an additional implicit crop of that selected window, not the wet
    reverberation tail. The selected source clip is preserved exactly; this
    check does not establish transcript or sentence-boundary completeness.
    """

    dry_audio = getattr(assembly, "dry_audio", None)
    receipts = getattr(dry_audio, "placement_receipts", ())
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise CurrentMP3DDynamicAudioError(
                "dry-audio placement receipt must be an object"
            )
        fit = receipt.get("fit")
        if fit is None:
            continue
        if not isinstance(fit, Mapping):
            raise CurrentMP3DDynamicAudioError(
                "dry-audio placement fit must be an object"
            )
        cropped = fit.get("cropped_tail_sample_count", 0)
        if isinstance(cropped, bool) or not isinstance(cropped, (int, np.integer)):
            raise CurrentMP3DDynamicAudioError(
                "dry-audio cropped_tail_sample_count must be an integer"
            )
        if int(cropped) < 0:
            raise CurrentMP3DDynamicAudioError(
                "dry-audio cropped_tail_sample_count cannot be negative"
            )
        if int(cropped) > 0:
            event_id = receipt.get("event_id", "<unknown>")
            raise CurrentMP3DDynamicAudioError(
                "selected source window for event "
                f"{event_id!r} exceeds its AudioProgram event window by "
                f"{int(cropped)} samples; refusing to crop the utterance"
            )


def _require_exact_episode_samples(
    samples: Any,
    *,
    expected: int,
    owner: str,
    channel_major: bool,
) -> np.ndarray:
    """Require an audio array to retain the visual clock's sample boundary."""

    array = np.asarray(samples)
    expected_ndim = 2 if channel_major else 1
    if array.ndim != expected_ndim:
        layout = "[channels, samples]" if channel_major else "[samples]"
        raise CurrentMP3DDynamicAudioError(
            f"{owner} must have channel-major {layout} audio samples"
        )
    actual = int(array.shape[1] if channel_major else array.shape[0])
    if actual != int(expected):
        raise CurrentMP3DDynamicAudioError(
            f"{owner} has {actual} samples; the visual AudioProgram clock "
            f"requires exactly {int(expected)}; refusing to retime or truncate"
        )
    return array


def _normalize_layouts(layouts: Sequence[str] | None) -> tuple[str, ...]:
    """Validate the requested output layouts while retaining request order."""
    if layouts is None:
        values = ("binaural",)
    elif isinstance(layouts, (str, bytes)):
        raise CurrentMP3DDynamicAudioError(
            "layouts must be a sequence of canonical layout names"
        )
    else:
        try:
            values = tuple(layouts)
        except TypeError as error:
            raise CurrentMP3DDynamicAudioError(
                "layouts must be a sequence of canonical layout names"
            ) from error
    if not values:
        raise CurrentMP3DDynamicAudioError("layouts must contain at least one layout")
    if any(value not in SUPPORTED_LAYOUTS for value in values):
        raise CurrentMP3DDynamicAudioError(
            "layouts must contain only " + ", ".join(SUPPORTED_LAYOUTS)
        )
    if len(set(values)) != len(values):
        raise CurrentMP3DDynamicAudioError("layouts must not contain duplicates")
    return values


def _render_layout_rir_sequence(
    scene: Any,
    simulation: Any,
    *,
    grid: Any,
    layout: str,
    hrtf_file_path: Path | None,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
) -> Any:
    """Render one layout through the capture-acoustics API with explicit runtime inputs."""
    updates = {
        "AVENGINE_HABITAT_RUNTIME_PREFIX": runtime_prefix,
        "AVENGINE_RLR_SDK_ROOT": rlr_sdk_root,
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE": magnum_python_site,
    }
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is not None:
                os.environ[key] = str(Path(value).expanduser().resolve())
        return render_research_review_rir_sequence(
            scene,
            simulation,
            grid=grid,
            layout_type=layout,
            hrtf_file_path=(
                str(hrtf_file_path) if layout == "binaural" else None
            ),
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _render_layout_audio(
    dry_buses: Mapping[str, Any],
    sequence: Any,
    *,
    grid: Any,
    layout: str,
) -> tuple[Mapping[str, Any], Any]:
    """Render one layout from the shared dry buses and RIR sequence."""
    return render_research_review_audio(dry_buses, sequence, grid=grid)


def _layout_sequence_record(sequence: Any, layout: str) -> dict[str, Any]:
    expected_labels = _LAYOUT_CHANNEL_LABELS[layout]
    if getattr(sequence, "layout_type", None) != layout:
        raise CurrentMP3DDynamicAudioError(
            f"RIR sequence layout differs from requested {layout!r}"
        )
    labels = tuple(getattr(sequence, "channel_labels", ()))
    if labels != expected_labels:
        raise CurrentMP3DDynamicAudioError(
            f"{layout} RIR sequence channel labels differ from its layout contract"
        )
    layout_id = getattr(sequence, "layout_id", None)
    if not isinstance(layout_id, str) or not layout_id:
        raise CurrentMP3DDynamicAudioError(
            f"{layout} RIR sequence has no layout_id"
        )
    keyframe_samples = getattr(sequence, "keyframe_samples", None)
    if keyframe_samples is None:
        raise CurrentMP3DDynamicAudioError(
            f"{layout} RIR sequence has no keyframe samples"
        )
    trajectory_sha256 = getattr(sequence, "trajectory_sha256", None)
    if not isinstance(trajectory_sha256, str) or not trajectory_sha256:
        raise CurrentMP3DDynamicAudioError(
            f"{layout} RIR sequence has no trajectory identity"
        )
    return {
        "layout_type": layout,
        "layout_id": layout_id,
        "channel_count": _LAYOUT_CHANNEL_COUNTS[layout],
        "channel_labels": list(labels),
        "keyframe_count": len(keyframe_samples),
        "keyframe_samples": list(keyframe_samples),
        "trajectory_sha256": trajectory_sha256,
        "output_directory": _LAYOUT_OUTPUT_DIRS[layout],
    }


def _require_layout_episode_samples(
    samples: Any,
    *,
    expected: int,
    expected_channels: int,
    owner: str,
) -> np.ndarray:
    array = _require_exact_episode_samples(
        samples,
        expected=expected,
        owner=owner,
        channel_major=True,
    )
    if int(array.shape[0]) != expected_channels:
        raise CurrentMP3DDynamicAudioError(
            f"{owner} has {array.shape[0]} channels; "
            f"{expected_channels} are required for this layout"
        )
    return array


def _simulation_from_inputs(
    *,
    simulation_request_path: str | Path | None,
    simulation_mapping: Mapping[str, Any] | None,
    diffraction: bool | None,
    max_diffraction_order: int | None,
) -> tuple[M4SimulationConfig, dict[str, Any]]:
    if simulation_mapping is not None and simulation_request_path is not None:
        raise CurrentMP3DDynamicAudioError(
            "provide either simulation_request_path or simulation_mapping, not both"
        )
    if simulation_mapping is not None:
        try:
            simulation = M4SimulationConfig.from_mapping(simulation_mapping)
        except (RuntimeError, TypeError, ValueError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"inline simulation mapping is invalid: {error}"
            ) from error
        source_record: dict[str, Any] = {
            "kind": "inline_mapping",
            "value": simulation.to_dict(),
        }
    else:
        if simulation_request_path is None:
            raise CurrentMP3DDynamicAudioError(
                "a simulation request or inline simulation mapping is required"
            )
        try:
            request, simulation = _load_simulation_request(
                Path(simulation_request_path).resolve()
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"simulation request is invalid: {error}"
            ) from error
        source_record = {
            "path": str(Path(simulation_request_path).resolve()),
            "sha256": sha256_file(simulation_request_path),
            "request_id": request.get("request_id")
            if isinstance(request, Mapping)
            else None,
        }
    if diffraction is not None and not isinstance(diffraction, bool):
        raise CurrentMP3DDynamicAudioError("diffraction must be boolean when supplied")
    if max_diffraction_order is not None and (
        isinstance(max_diffraction_order, bool)
        or not isinstance(max_diffraction_order, int)
        or max_diffraction_order < 0
    ):
        raise CurrentMP3DDynamicAudioError(
            "max_diffraction_order must be a non-negative integer"
        )
    updates: dict[str, Any] = {}
    if diffraction is not None:
        updates["diffraction"] = diffraction
    if max_diffraction_order is not None:
        updates["max_diffraction_order"] = max_diffraction_order
    if updates:
        try:
            simulation = M4SimulationConfig.from_mapping(
                {**simulation.to_dict(), **updates}
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"diffraction configuration is invalid: {error}"
            ) from error
    return simulation, source_record


def _prepared_activity_index(
    prepared_manifest_path: str | Path | None,
) -> dict[str, Mapping[str, Any]]:
    if prepared_manifest_path is None:
        return {}
    path = Path(prepared_manifest_path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"cannot read prepared audio manifest: {error}"
        ) from error
    clips = value.get("clips") if isinstance(value, Mapping) else None
    if not isinstance(clips, list):
        raise CurrentMP3DDynamicAudioError(
            "prepared audio manifest must contain a clips list"
        )
    index: dict[str, Mapping[str, Any]] = {}
    for clip in clips:
        if not isinstance(clip, Mapping):
            continue
        keys = (
            clip.get("prepared_audio_id"),
            clip.get("source_asset_id"),
            clip.get("sound_asset_id"),
            clip.get("prepared"),
            clip.get("source_pcm_path"),
            clip.get("source_metadata_path"),
        )
        for key in keys:
            if key is None:
                continue
            text = str(key)
            if text.startswith("/"):
                text = str(Path(text).expanduser().resolve())
            if text in index and index[text] != clip:
                raise CurrentMP3DDynamicAudioError(
                    f"prepared audio manifest has conflicting key {text!r}"
                )
            index[text] = clip
    return index


def _active_intervals_from_pcm(
    path: Path,
    *,
    start_sample: int,
    end_sample: int,
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getsampwidth() != 2
                or handle.getcomptype() != "NONE"
                or handle.getframerate() != AUDIO_SAMPLE_RATE_HZ
            ):
                raise ValueError("dry audio must be mono 16-bit PCM at 16 kHz")
            count = handle.getnframes()
            payload = handle.readframes(count)
    except (OSError, EOFError, wave.Error, ValueError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"cannot inspect source activity in {path}: {error}"
        ) from error
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    if not 0 <= start_sample < end_sample <= len(samples):
        raise CurrentMP3DDynamicAudioError(
            f"source activity slice [{start_sample},{end_sample}) escapes {path}"
        )
    selected = samples[start_sample:end_sample]
    peak = float(np.max(np.abs(selected))) if selected.size else 0.0
    threshold = max(1.0e-6, peak * 1.0e-3)
    active = np.flatnonzero(np.abs(selected) > threshold)
    intervals: list[dict[str, int]] = []
    if active.size:
        run_start = int(active[0])
        previous = run_start
        for value in active[1:]:
            current = int(value)
            if current != previous + 1:
                intervals.append(
                    {
                        "start_sample": int(start_sample + run_start),
                        "end_sample_exclusive": int(start_sample + previous + 1),
                    }
                )
                run_start = current
            previous = current
        intervals.append(
            {
                "start_sample": int(start_sample + run_start),
                "end_sample_exclusive": int(start_sample + previous + 1),
            }
        )
    return intervals, {
        "detector": "full_band_abs_threshold",
        "threshold_relative_peak": 1.0e-3,
        "threshold_abs": threshold,
        "coordinate_space": "source_asset_samples",
    }


def _event_source_activity(
    event: Mapping[str, Any],
    *,
    event_metadata: Mapping[str, Any] | None,
    prepared_index: Mapping[str, Mapping[str, Any]],
    asset_path: Path,
    episode_sample_count: int,
) -> dict[str, Any]:
    event_id = str(event.get("event_id", "<unknown>"))
    event_start = int(event["start_sample"])
    source_start = int(event.get("source_start_sample", 0))
    source_end_raw = event.get("source_end_sample_exclusive")
    source_end = int(source_end_raw) if source_end_raw is not None else None
    metadata = event_metadata if isinstance(event_metadata, Mapping) else {}
    manifest = None
    keys = (
        event.get("prepared_audio_id"),
        event.get("sound_asset_id"),
        str(asset_path),
        event.get("path"),
    )
    for key in keys:
        if key is None:
            continue
        lookup = str(key)
        if lookup.startswith("/"):
            lookup = str(Path(lookup).expanduser().resolve())
        manifest = prepared_index.get(lookup)
        if manifest is not None:
            break
    source_intervals: list[dict[str, int]] = []
    original_intervals: list[dict[str, int]] = []
    provenance = "pcm_source_activity_measurement"
    crop_start = 0
    facts: Mapping[str, Any] = {}
    if isinstance(manifest, Mapping):
        candidate_facts = manifest.get("facts")
        facts = candidate_facts if isinstance(candidate_facts, Mapping) else {}
        activity = manifest.get("source_activity")
        if not isinstance(activity, Mapping):
            activity = facts.get("source_activity")
        rows = activity.get("intervals") if isinstance(activity, Mapping) else None
        if isinstance(rows, list) and rows:
            crop_value = facts.get(
                "source_crop_start_sample",
                manifest.get("source_crop_start_sample", 0),
            )
            try:
                crop_start = int(crop_value)
            except (TypeError, ValueError) as error:
                raise CurrentMP3DDynamicAudioError(
                    f"prepared activity crop_start is invalid for {event_id}"
                ) from error
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                try:
                    original_start = int(row["start_sample"])
                    original_end = int(row["end_sample_exclusive"])
                except (KeyError, TypeError, ValueError) as error:
                    raise CurrentMP3DDynamicAudioError(
                        f"prepared activity interval is invalid for {event_id}"
                    ) from error
                original_intervals.append(
                    {
                        "start_sample": original_start,
                        "end_sample_exclusive": original_end,
                    }
                )
                local_start = original_start - crop_start
                local_end = original_end - crop_start
                if local_start < 0 or local_end <= local_start:
                    continue
                source_intervals.append(
                    {
                        "start_sample": local_start,
                        "end_sample_exclusive": local_end,
                    }
                )
            provenance = (
                "prepared_manifest_original_source_samples_minus_crop_start"
            )
    if not source_intervals and isinstance(metadata.get("source_activity_intervals_samples"), list):
        coordinate = str(metadata.get("activity_coordinate", "prepared_clip_samples"))
        for row in metadata["source_activity_intervals_samples"]:
            if isinstance(row, Mapping):
                start_value = row.get("start_sample")
                end_value = row.get("end_sample_exclusive")
            elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) == 2:
                start_value, end_value = row
            else:
                continue
            try:
                local_start = int(start_value)
                local_end = int(end_value)
            except (TypeError, ValueError) as error:
                raise CurrentMP3DDynamicAudioError(
                    f"event source activity interval is invalid for {event_id}"
                ) from error
            if coordinate in {"prepared_clip_samples", "event_samples", "source_slice_samples"}:
                source_intervals.append(
                    {
                        "start_sample": source_start + local_start,
                        "end_sample_exclusive": source_start + local_end,
                    }
                )
            else:
                source_intervals.append(
                    {
                        "start_sample": local_start,
                        "end_sample_exclusive": local_end,
                    }
                )
        provenance = "event_binding_source_activity_intervals"
    if not source_intervals:
        interval = metadata.get("source_activity_interval")
        if isinstance(interval, Mapping):
            local_start = int(interval.get("start_sample", 0))
            local_end = int(interval["end_sample_exclusive"])
            source_intervals = [
                {
                    "start_sample": source_start + local_start,
                    "end_sample_exclusive": source_start + local_end,
                }
            ]
            provenance = "event_binding_detected_source_activity_interval"
    if not source_intervals:
        if source_end is None:
            with wave.open(str(asset_path), "rb") as handle:
                source_end = handle.getnframes()
        measured, detector = _active_intervals_from_pcm(
            asset_path,
            start_sample=source_start,
            end_sample=source_end,
        )
        source_intervals = measured
        detector_record = detector
    else:
        detector_record = {
            "detector": "prepared_manifest_or_event_binding",
            "coordinate_space": (
                "original_source_samples" if original_intervals else "source_asset_samples"
            ),
        }
    episode_intervals: list[dict[str, int]] = []
    for row in source_intervals:
        local_start = int(row["start_sample"])
        local_end = int(row["end_sample_exclusive"])
        # Prepared manifests store source coordinates. Their crop offset was
        # removed above; ordinary bindings already store selected-slice coords.
        if original_intervals and provenance.startswith("prepared_manifest"):
            pass
        else:
            local_start -= source_start
            local_end -= source_start
        absolute_start = event_start + local_start
        absolute_end = event_start + local_end
        if absolute_end <= absolute_start:
            continue
        episode_intervals.append(
            {
                "start_sample": max(0, absolute_start),
                "end_sample_exclusive": min(episode_sample_count, absolute_end),
            }
        )
    episode_intervals = [
        row
        for row in episode_intervals
        if row["end_sample_exclusive"] > row["start_sample"]
    ]
    return {
        "source_activity_intervals_samples": episode_intervals,
        "source_activity_coordinate_space": "episode_sample_clock",
        "source_activity_provenance": provenance,
        "source_activity_detector": detector_record,
        "source_activity_original_intervals_samples": original_intervals,
        "source_crop_start_sample": crop_start,
        "source_slice": {
            "start_sample": source_start,
            "end_sample_exclusive": source_end,
        },
    }


def _sequence_from_override(
    override: Mapping[str, Any],
    *,
    grid: Any,
    layout: str,
) -> DynamicRIRSequence:
    try:
        samples = np.ascontiguousarray(override["samples"], dtype="<f4")
        lengths = np.ascontiguousarray(override["lengths"], dtype="<u4")
    except (KeyError, TypeError, ValueError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"{layout} RIR override is malformed"
        ) from error
    expected_channels = _LAYOUT_CHANNEL_COUNTS[layout]
    if (
        samples.ndim != 4
        or samples.shape[:3] != (len(grid.keyframes), len(grid.source_ids), expected_channels)
        or lengths.shape != (len(grid.keyframes), len(grid.source_ids))
    ):
        raise CurrentMP3DDynamicAudioError(
            f"{layout} RIR override shape differs from the neutral clock"
        )
    trajectory_hash = canonical_json_sha256(research_review_trajectory_record(grid))
    metadata = dict(override.get("metadata") or {})
    metadata.setdefault("override_source", "legacy_dynamic_rir_cache_adapter")
    metadata.setdefault("qualification_claim", False)
    return DynamicRIRSequence(
        samples=samples,
        lengths=lengths,
        source_ids=tuple(grid.source_ids),
        keyframe_ticks=tuple(frame.tick for frame in grid.keyframes),
        keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
        sample_rate_hz=int(grid.sample_rate_hz),
        layout_type=layout,
        layout_id=str(override.get("layout_id", _LAYOUT_OUTPUT_DIRS[layout])),
        channel_labels=tuple(_LAYOUT_CHANNEL_LABELS[layout]),
        trajectory_sha256=trajectory_hash,
        metadata=metadata,
    )


def render_dynamic_research_audio(
    *,
    source_trajectories_m: Mapping[str, Sequence[Sequence[float]]],
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
    simulation_request_path: str | Path | None = None,
    simulation_mapping: Mapping[str, Any] | None = None,
    package_manifest_path: str | Path,
    audio_program_path: str | Path | None = None,
    audio_program: Mapping[str, Any] | None = None,
    source_endpoint_registry_path: str | Path | None = None,
    sound_asset_registry_path: str | Path | None = None,
    external_sound_asset_paths: Mapping[str, str | Path] | None = None,
    event_asset_bindings: Mapping[str, str | Path] | None = None,
    event_metadata: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    prepared_manifest_path: str | Path | None = None,
    hrtf_file_path: str | Path | None = None,
    output_path: str | Path,
    position_authority: str,
    listener_authority: str,
    rir_stride_frames: int = 3,
    variant_id: str = "A",
    execution_variant: str | None = None,
    hrtf_license_path: str | Path | None = None,
    extra_inputs: Mapping[str, Any] | None = None,
    visual_frame_count: int | None = None,
    visual_frame_rate_hz: int | float | None = None,
    timeline_tick_rate_hz: int | None = None,
    ticks_per_frame: int | None = None,
    layouts: Sequence[str] | None = None,
    neutral_readback_path: str | Path | None = None,
    neutral_readback_data: Mapping[str, Any] | None = None,
    rir_sequence_override: Mapping[str, Mapping[str, Any]] | None = None,
    scene_override: Any | None = None,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    diffraction: bool | None = None,
    max_diffraction_order: int | None = None,
) -> dict[str, Any]:
    """Render one room-agnostic episode from explicit positions and clock.

    The function is the shared acoustic renderer used by both the UE
    frame-readback adapter and the current MP3D CLI adapter. It consumes no
    renderer-specific coordinates and applies event gain only in the dry bus
    assembler; convolution is always called with the already-gained buses.
    """

    execution_label = _validate_execution_variant(execution_variant)
    selected_layouts = _normalize_layouts(layouts)
    hrtf = None
    if "binaural" in selected_layouts:
        if hrtf_file_path is None:
            raise CurrentMP3DDynamicAudioError(
                "binaural output requires an explicit HRTF file"
            )
        hrtf = Path(hrtf_file_path).resolve()
        if not hrtf.is_file():
            raise CurrentMP3DDynamicAudioError(
                f"binaural output requires a readable HRTF: {hrtf}"
            )
    if (audio_program is None) == (audio_program_path is None):
        raise CurrentMP3DDynamicAudioError(
            "provide exactly one of audio_program_path or audio_program"
        )
    if (source_endpoint_registry_path is None) != (sound_asset_registry_path is None):
        raise CurrentMP3DDynamicAudioError(
            "source and sound registries must be supplied together"
        )
    output = _fresh_output(output_path)
    if audio_program is None:
        program_path = Path(audio_program_path).resolve()
        try:
            program = json.loads(program_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"cannot read AudioProgram: {error}"
            ) from error
        program_input = {
            "path": str(program_path),
            "sha256": sha256_file(program_path),
        }
    else:
        program_path = None
        program = deepcopy(dict(audio_program))
        program_input = {"kind": "inline_mapping"}
    if not isinstance(program, Mapping):
        raise CurrentMP3DDynamicAudioError("audio program must be a JSON object")
    program_errors = validate_audio_program(program)
    if program_errors:
        raise CurrentMP3DDynamicAudioError(
            "AudioProgram validation failed: " + "; ".join(program_errors)
        )
    source_ids = tuple(
        str(value) for value in program.get("candidate_source_endpoint_ids") or ()
    )
    if len(source_ids) < 2 or source_ids != tuple(sorted(set(source_ids))):
        raise CurrentMP3DDynamicAudioError(
            "the program must carry at least two canonical candidate source endpoints"
        )
    if set(source_ids) != set(source_trajectories_m):
        raise CurrentMP3DDynamicAudioError(
            "trajectory source IDs must equal the program candidates: "
            f"{sorted(source_trajectories_m)} != {sorted(source_ids)}"
        )
    timeline = program.get("timeline")
    if not isinstance(timeline, Mapping):
        raise CurrentMP3DDynamicAudioError(
            "the AudioProgram must carry an explicit timeline"
        )
    trajectory_lengths = set()
    ordered_trajectories: dict[str, list[list[float]]] = {}
    for source_id in source_ids:
        try:
            points = [
                [float(value) for value in point]
                for point in source_trajectories_m[source_id]
            ]
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"source trajectory {source_id!r} is invalid"
            ) from error
        if any(len(point) != 3 or not np.all(np.isfinite(point)) for point in points):
            raise CurrentMP3DDynamicAudioError(
                f"source trajectory {source_id!r} is invalid"
            )
        ordered_trajectories[source_id] = points
        trajectory_lengths.add(len(points))
    if len(trajectory_lengths) != 1 or not trajectory_lengths:
        raise CurrentMP3DDynamicAudioError(
            "all source trajectories must have one equal frame count"
        )
    trajectory_frame_count = next(iter(trajectory_lengths))
    clock = _resolve_visual_clock(
        frame_count=(
            visual_frame_count
            if visual_frame_count is not None
            else trajectory_frame_count
        ),
        frame_rate_hz=(
            visual_frame_rate_hz
            if visual_frame_rate_hz is not None
            else timeline.get("video_fps", DEFAULT_VISUAL_FRAME_RATE_HZ)
        ),
        ticks_per_frame=(
            ticks_per_frame
            if ticks_per_frame is not None
            else timeline.get("ticks_per_frame", DEFAULT_TICKS_PER_FRAME)
        ),
        time_base_hz=(
            timeline_tick_rate_hz
            if timeline_tick_rate_hz is not None
            else timeline.get("time_base_hz", DEFAULT_TIMELINE_TICK_RATE_HZ)
        ),
    )
    if trajectory_frame_count != clock["frame_count"]:
        raise CurrentMP3DDynamicAudioError(
            "source trajectory length differs from the visual clock"
        )
    program_clock_binding = _program_clock_binding(program, clock)
    program = _program_for_visual_clock(program, clock)
    timeline = program["timeline"]
    external_sound_asset_paths = external_sound_asset_paths or {}
    if source_endpoint_registry_path is not None:
        endpoint_registry_path = Path(source_endpoint_registry_path).resolve()
        sound_registry_path = Path(sound_asset_registry_path).resolve()
        endpoints = load_source_endpoint_registry(endpoint_registry_path)
        sounds = load_sound_asset_registry(sound_registry_path)
        required_sounds = {
            str(event["sound_asset_id"]) for event in program.get("events") or ()
        }
        repository_root = Path(__file__).resolve().parents[3]
        bindings = _asset_bindings(
            sounds,
            repository_root=repository_root,
            external_sound_asset_paths={
                key: Path(value) for key, value in external_sound_asset_paths.items()
            },
            required_sound_ids=required_sounds,
        )
        assembly = assemble_audio_program_dry_buses(
            program,
            variant_id,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
            asset_bindings=bindings,
        )
        dry_asset_records = bindings
    else:
        endpoint_registry_path = None
        sound_registry_path = None
        assembly = _direct_program_dry_assembly(
            program,
            variant_id=variant_id,
            event_asset_bindings=event_asset_bindings,
        )
        direct_bindings = dict(event_asset_bindings or {})
        for event in program.get("events") or ():
            if isinstance(event, Mapping) and event.get("sound_asset_id"):
                path_value = event.get("path") or direct_bindings.get(
                    str(event["sound_asset_id"])
                )
                if path_value is not None:
                    path = Path(path_value).expanduser().resolve()
                    direct_bindings[str(event["sound_asset_id"])] = path
        dry_asset_records = {
            key: {"path": str(Path(value).resolve()), "sha256": sha256_file(value)}
            for key, value in direct_bindings.items()
        }
    _assert_no_cropped_dry_audio(assembly)
    dry_buses = assembly.dry_audio.buses
    expected_sample_count = int(clock["sample_count"])
    for source_id in source_ids:
        _require_exact_episode_samples(
            dry_buses[source_id],
            expected=expected_sample_count,
            owner=f"dry bus {source_id!r}",
            channel_major=False,
        )
    grid = build_strided_review_keyframes(
        ordered_trajectories,
        visual_frame_rate_hz=clock["frame_rate_hz"],
        rir_stride_frames=rir_stride_frames,
        listener_position_m=list(listener_position_m),
        listener_orientation_wxyz=list(listener_orientation_wxyz),
        timeline_tick_rate_hz=clock["time_base_hz"],
        sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
    )
    simulation, simulation_input = _simulation_from_inputs(
        simulation_request_path=simulation_request_path,
        simulation_mapping=simulation_mapping,
        diffraction=diffraction,
        max_diffraction_order=max_diffraction_order,
    )
    simulation_dict = simulation.to_dict() if simulation is not None else {}
    simulation_diffraction = bool(getattr(simulation, "diffraction", False))
    simulation_max_diffraction_order = int(
        getattr(simulation, "max_diffraction_order", 0)
    )
    scene = (
        scene_override
        if scene_override is not None
        else load_compiled_acoustic_scene(
            package_manifest_path, allow_nonpassing_research_qa=True
        )
    )

    rendered: dict[str, dict[str, Any]] = {}
    for layout in selected_layouts:
        override = (rir_sequence_override or {}).get(layout)
        if override is not None:
            sequence = _sequence_from_override(override, grid=grid, layout=layout)
        else:
            sequence = _render_layout_rir_sequence(
                scene,
                simulation,
                grid=grid,
                layout=layout,
                hrtf_file_path=hrtf,
                runtime_prefix=runtime_prefix,
                rlr_sdk_root=rlr_sdk_root,
                magnum_python_site=magnum_python_site,
            )
        layout_record = _layout_sequence_record(sequence, layout)
        stems, mixture = _render_layout_audio(
            dry_buses,
            sequence,
            grid=grid,
            layout=layout,
        )
        if int(grid.episode_sample_count) != expected_sample_count:
            raise CurrentMP3DDynamicAudioError(
                "dynamic acoustic grid sample boundary differs from the visual "
                "AudioProgram clock"
            )
        expected_channels = _LAYOUT_CHANNEL_COUNTS[layout]
        if not isinstance(stems, Mapping):
            raise CurrentMP3DDynamicAudioError(
                f"{layout} renderer must return a mapping of source stems"
            )
        for source_id in source_ids:
            stem = stems.get(source_id)
            if stem is None:
                raise CurrentMP3DDynamicAudioError(
                    f"{layout} renderer omitted source stem {source_id!r}"
                )
            _require_layout_episode_samples(
                stem.episode,
                expected=expected_sample_count,
                expected_channels=expected_channels,
                owner=f"{layout} stem {source_id!r}",
            )
        mixture = _require_layout_episode_samples(
            mixture,
            expected=expected_sample_count,
            expected_channels=expected_channels,
            owner=f"{layout} mixture",
        )
        rendered[layout] = {
            "sequence": sequence,
            "record": layout_record,
            "stems": stems,
            "mixture": mixture,
        }

    audio_root = output / "audio"
    outputs: dict[str, str] = {}
    float32_precision: dict[str, dict[str, float | str]] = {}

    def _write(path: Path, samples: np.ndarray) -> None:
        array = np.asarray(samples, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise CurrentMP3DDynamicAudioError(
                f"audio output contains non-finite samples: {path}"
            )
        peak = float(np.max(np.abs(array))) if array.size else 0.0
        if peak > 1.0 + 1.0e-12:
            raise CurrentMP3DDynamicAudioError(
                f"audio output would clip without normalization/limiting: peak={peak:.9g}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_float32_wav(path, array, AUDIO_SAMPLE_RATE_HZ)
        outputs[str(path.relative_to(output))] = sha256_file(path)
        cast = np.asarray(array, dtype=np.float32).astype(np.float64)
        error = float(np.max(np.abs(cast - array))) if array.size else 0.0
        float32_precision[str(path.relative_to(output))] = {
            "encoding": "IEEE_FLOAT32",
            "max_abs_error_vs_float64": error,
            "dtype": "float32",
        }

    program_output = output / "audio_program.json"
    _write_json = lambda path, value: path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_json(program_output, program)
    for source_id in source_ids:
        dry = np.asarray(dry_buses[source_id], dtype=np.float64)[None, :]
        _write(audio_root / "dry" / f"{source_id}.wav", dry)
    for layout in selected_layouts:
        layout_output_dir = _LAYOUT_OUTPUT_DIRS[layout]
        stems = rendered[layout]["stems"]
        for source_id in source_ids:
            _write(
                audio_root / layout_output_dir / f"{source_id}_stem.wav",
                stems[source_id].episode,
            )
        _write(
            audio_root / layout_output_dir / "mixture.wav",
            rendered[layout]["mixture"],
        )

    # Materialize a neutral copy only when the adapter supplied an in-memory
    # conversion. A supplied P1 path remains the input authority in the receipt.
    neutral_record = None
    if neutral_readback_path is not None:
        neutral_record = _input_record(neutral_readback_path)
    elif neutral_readback_data is not None:
        neutral_copy = output / "neutral_readback.json"
        _write_json(neutral_copy, neutral_readback_data)
        neutral_record = _input_record(neutral_copy)

    compatibility_layout = (
        "binaural" if "binaural" in selected_layouts else selected_layouts[0]
    )
    compatibility_record = rendered[compatibility_layout]["record"]
    primary_stems = {
        source_id: str(
            (audio_root / _LAYOUT_OUTPUT_DIRS[compatibility_layout] / f"{source_id}_stem.wav").resolve()
        )
        for source_id in source_ids
    }
    primary_mixture_path = (
        audio_root / _LAYOUT_OUTPUT_DIRS[compatibility_layout] / "mixture.wav"
    ).resolve()
    metadata_by_event: dict[str, Mapping[str, Any]] = {}
    if isinstance(event_metadata, Mapping):
        metadata_by_event = {
            str(key): value
            for key, value in event_metadata.items()
            if isinstance(value, Mapping)
        }
    elif isinstance(event_metadata, Sequence) and not isinstance(event_metadata, (str, bytes)):
        metadata_by_event = {
            str(row["event_id"]): row
            for row in event_metadata
            if isinstance(row, Mapping) and row.get("event_id") is not None
        }
    prepared_index = _prepared_activity_index(prepared_manifest_path)
    primary_sequence = rendered[compatibility_layout]["sequence"]
    source_index = {source_id: index for index, source_id in enumerate(source_ids)}
    event_records: list[dict[str, Any]] = []
    wet_tail_intervals: list[dict[str, Any]] = []
    events = program.get("events") or ()
    for event in events:
        if not isinstance(event, Mapping):
            raise CurrentMP3DDynamicAudioError("AudioProgram event must be an object")
        event_id = str(event["event_id"])
        source_id = str(event["source_endpoint_id"])
        event_start = int(event["start_sample"])
        event_end = int(event["end_sample_exclusive"])
        event_bus = np.zeros(expected_sample_count, dtype=np.float64)
        event_bus[event_start:event_end] = dry_buses[source_id][event_start:event_end]
        sequence_samples = getattr(primary_sequence, "samples", None)
        sequence_lengths = getattr(primary_sequence, "lengths", None)
        if sequence_samples is not None and sequence_lengths is not None:
            isolated = time_varying_convolve(
                event_bus,
                sequence_samples[:, source_index[source_id]],
                primary_sequence.keyframe_samples,
                rir_lengths=sequence_lengths[:, source_index[source_id]],
                output_sample_count=expected_sample_count,
            )
            wet_tail = _nonzero_interval(isolated.full_tail, threshold=1.0e-12)
            if wet_tail is None:
                raise CurrentMP3DDynamicAudioError(
                    f"event {event_id!r} produced no measurable wet tail"
                )
            event_stem = isolated.episode
            wet_peak_abs = float(np.max(np.abs(isolated.full_tail))) if isolated.full_tail.size else 0.0
            wet_peak_dbfs = _peak_dbfs(isolated.full_tail)
            event_activity_signal = True
        else:
            isolated = None
            wet_tail = [event_start, event_end]
            event_stem = np.zeros((2, max(0, event_end - event_start)), dtype=np.float64)
            wet_peak_abs = 0.0
            wet_peak_dbfs = None
            event_activity_signal = False
        asset = dry_asset_records.get(str(event["sound_asset_id"]))
        if isinstance(asset, Mapping):
            asset_path = Path(str(asset["path"])).resolve()
        elif asset is not None:
            asset_path = Path(str(asset)).resolve()
        else:
            asset_path = None
        if asset_path is not None and asset_path.is_file():
            activity = _event_source_activity(
                event,
                event_metadata=metadata_by_event.get(event_id),
                prepared_index=prepared_index,
                asset_path=asset_path,
                episode_sample_count=expected_sample_count,
            )
        else:
            activity = {
                "source_activity_intervals_samples": [],
                "source_activity_coordinate_space": "episode_sample_clock",
                "source_activity_provenance": "not_available_in_renderer_stub",
                "source_activity_detector": {"status": "not_available"},
                "source_activity_original_intervals_samples": [],
                "source_crop_start_sample": 0,
                "source_slice": {
                    "start_sample": int(event.get("source_start_sample", 0)),
                    "end_sample_exclusive": event.get("source_end_sample_exclusive"),
                },
            }
        event_peak = float(np.max(np.abs(event_stem))) if event_stem.size else 0.0
        wet_tail, wet_tail_clamped, wet_tail_end_original = _clamp_wet_tail_interval(
            wet_tail, expected_sample_count
        )
        event_record = {
            **dict(event),
            "source_endpoint_id": source_id,
            "linear_gain": float(event.get("linear_gain", 1.0)),
            "output_stem": primary_stems[source_id],
            "wet_tail_interval": wet_tail,
            "wet_tail_intervals": [wet_tail],
            "wet_tail_end_sample": wet_tail[1],
            "wet_tail_clamped": wet_tail_clamped,
            "wet_tail_end_sample_original": wet_tail_end_original,
            "wet_tail_coordinate_space": "episode_sample_clock",
            "wet_tail_peak_abs": wet_peak_abs,
            "wet_tail_peak_dbfs": wet_peak_dbfs,
            "wet_tail_measurement": "event_isolated_convolution" if event_activity_signal else "scheduled_interval_stub",
            "event_output_peak_abs": event_peak,
            "event_output_peak_dbfs": _peak_dbfs(event_stem),
            "peak_abs": event_peak,
            "peak_dbfs": _peak_dbfs(event_stem),
            "gain_application": {
                "applied_at": "dry_audio_assembly",
                "application_count": 1,
                "linear_gain": float(event.get("linear_gain", 1.0)),
                "post_assembly_convolution_gain": 1.0,
                "normalization": False,
                "proof": "convolution_consumes_already_gained_named_dry_bus",
            },
            **activity,
        }
        binding_meta = metadata_by_event.get(event_id)
        if isinstance(binding_meta, Mapping):
            if binding_meta.get("actor_id") is not None:
                event_record["voice_binding_actor_id"] = binding_meta["actor_id"]
            if binding_meta.get("prepared_audio_id") is not None:
                event_record["prepared_audio_id"] = binding_meta["prepared_audio_id"]
        event_records.append(event_record)
        wet_tail_intervals.append(
            {
                "event_id": event_id,
                "source_endpoint_id": source_id,
                "start_sample": wet_tail[0],
                "end_sample_exclusive": wet_tail[1],
                "wet_tail_clamped": wet_tail_clamped,
                "wet_tail_end_sample_original": wet_tail_end_original,
            }
        )

    if neutral_record is None:
        neutral_record = {
            "kind": "legacy_explicit_listener_and_source_poses",
            "path": None,
        }
    peak_abs = {
        "mixture": float(np.max(np.abs(rendered[compatibility_layout]["mixture"])))
        if rendered[compatibility_layout]["mixture"].size
        else 0.0,
        "stems": {
            source_id: float(np.max(np.abs(rendered[compatibility_layout]["stems"][source_id].episode)))
            if rendered[compatibility_layout]["stems"][source_id].episode.size
            else 0.0
            for source_id in source_ids
        },
    }
    peak_dbfs = {
        "mixture": _peak_dbfs(rendered[compatibility_layout]["mixture"]),
        "stems": {
            source_id: _peak_dbfs(
                rendered[compatibility_layout]["stems"][source_id].episode
            )
            for source_id in source_ids
        },
    }
    outputs_by_layout = {
        layout: {
            "mixture": str(
                (audio_root / _LAYOUT_OUTPUT_DIRS[layout] / "mixture.wav").resolve()
            ),
            "stems": {
                source_id: str(
                    (audio_root / _LAYOUT_OUTPUT_DIRS[layout] / f"{source_id}_stem.wav").resolve()
                )
                for source_id in source_ids
            },
        }
        for layout in selected_layouts
    }
    inputs: dict[str, Any] = {
        "simulation_request": simulation_input,
        "package_manifest": _input_record(package_manifest_path),
        "audio_program": program_input,
        "source_endpoint_registry": (
            _input_record(source_endpoint_registry_path)
            if source_endpoint_registry_path is not None
            else None
        ),
        "sound_asset_registry": (
            _input_record(sound_asset_registry_path)
            if sound_asset_registry_path is not None
            else None
        ),
        "hrtf": (
            {
                **_input_record(hrtf),
                "id": hrtf.name,
                "license_path": (
                    str(Path(hrtf_license_path).resolve())
                    if hrtf_license_path is not None
                    else None
                ),
            }
            if hrtf is not None
            else None
        ),
        "dry_assets": dry_asset_records,
        "neutral_readback": neutral_record,
    }
    if prepared_manifest_path is not None:
        inputs["prepared_manifest"] = _input_record(prepared_manifest_path)
    if extra_inputs:
        inputs.update({key: value for key, value in extra_inputs.items()})
    materialized = assembly.materialized_program
    receipt_clock = {**dict(clock), "ticks_per_sample": 3}
    receipt: dict[str, Any] = {
        "schema": UNIFIED_AUDIO_RECEIPT_SCHEMA,
        "legacy_schema": CURRENT_MP3D_DYNAMIC_AUDIO_SCHEMA,
        "status": "research",
        "claim_boundary": CLAIM_BOUNDARY,
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification_claim": False,
        "clock": receipt_clock,
        "input_neutral_readback": neutral_record,
        "input_neutral_readback_path": neutral_record.get("path"),
        "audio_program": {
            "path": str(program_output.resolve()),
            "program_id": program.get("program_id"),
            "revision": program.get("revision"),
            "mode": program.get("mode"),
            "variant_id": variant_id,
            "program_content_sha256": materialized.get("program_content_sha256"),
            "event_count": len(program.get("events") or ()),
            "timeline": dict(timeline),
        },
        "audio_program_path": str(program_output.resolve()),
        "mixture_path": str(primary_mixture_path),
        "stems": primary_stems,
        "outputs_by_layout": outputs_by_layout,
        "wet_tail_intervals": wet_tail_intervals,
        "peak_abs": peak_abs,
        "peak_dbfs": peak_dbfs,
        "mixture": {"path": str(primary_mixture_path), "peak_dbfs": peak_dbfs["mixture"]},
        "stem_records": {
            source_id: {"path": primary_stems[source_id], "peak_dbfs": peak_dbfs["stems"][source_id]}
            for source_id in source_ids
        },
        "pcm_encoding": {
            "format": "IEEE_FLOAT32_WAV",
            "byte_order": "little",
            "precision_by_output": float32_precision,
        },
        "gain_application": {
            "authority": "assemble_dry_audio_buses",
            "applied_once_per_event": True,
            "post_assembly_convolution_gain": 1.0,
            "normalization": False,
            "limiting": False,
        },
        "propagation": {
            "diffraction": simulation_diffraction,
            "max_diffraction_order": simulation_max_diffraction_order,
        },
        "diffraction": simulation_diffraction,
        "max_diffraction_order": simulation_max_diffraction_order,
        "audio": {
            "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
            "sample_count": clock["sample_count"],
            "layouts": list(selected_layouts),
            "layout_type": compatibility_layout,
            "channel_labels": list(_LAYOUT_CHANNEL_LABELS[compatibility_layout]),
            "mixture_path": str(primary_mixture_path),
            "mixture": {"path": str(primary_mixture_path), "peak_dbfs": peak_dbfs["mixture"]},
            "stems": primary_stems,
            "stem_records": {
                source_id: {"path": primary_stems[source_id], "peak_dbfs": peak_dbfs["stems"][source_id]}
                for source_id in source_ids
            },
            "peak_dbfs": peak_dbfs,
            "by_layout": {
                layout: {
                    "layout_type": layout,
                    "output_directory": rendered[layout]["record"]["output_directory"],
                    "channel_count": _LAYOUT_CHANNEL_COUNTS[layout],
                    "channel_labels": list(_LAYOUT_CHANNEL_LABELS[layout]),
                    "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
                    "sample_count": clock["sample_count"],
                }
                for layout in selected_layouts
            },
        },
        "audio_program_record": {
            "path": str(program_output.resolve()),
            "input": program_input,
            "program_id": program.get("program_id"),
            "revision": program.get("revision"),
            "mode": program.get("mode"),
            "variant_id": variant_id,
            "program_content_sha256": materialized.get("program_content_sha256"),
            "event_count": len(program.get("events") or ()),
            "timeline": dict(timeline),
            "clock_binding": program_clock_binding,
        },
        "audio_program_metadata": {
            "path": str(program_output.resolve()),
            "program_id": program.get("program_id"),
            "revision": program.get("revision"),
            "mode": program.get("mode"),
            "variant_id": variant_id,
            "program_content_sha256": materialized.get("program_content_sha256"),
            "event_count": len(program.get("events") or ()),
            "timeline": dict(timeline),
        },
        "sources": {
            "source_ids": list(source_ids),
            "frame_count": clock["frame_count"],
            "frame_rate_hz": clock["frame_rate_hz"],
            "ticks_per_frame": clock["ticks_per_frame"],
            "time_base_hz": clock["time_base_hz"],
            "position_authority": position_authority,
        },
        "listener": {
            "position_m": list(listener_position_m),
            "orientation_wxyz": list(listener_orientation_wxyz),
            "authority": listener_authority,
        },
        "hrtf_id": hrtf.name if hrtf is not None else None,
        "hrtf": (
            {
                "id": hrtf.name,
                "path": str(hrtf),
                "sha256": sha256_file(hrtf),
                "channel_order": ["left", "right"],
            }
            if hrtf is not None
            else None
        ),
        "rir": {
            "stride_frames": rir_stride_frames,
            "cache": (
                dict((rir_sequence_override or {}).get(compatibility_layout, {}).get("cache", {}))
                if (rir_sequence_override or {}).get(compatibility_layout) is not None
                else {"status": "not_used"}
            ),
            "layout_type": compatibility_record["layout_type"],
            "layout_id": compatibility_record["layout_id"],
            "channel_labels": compatibility_record["channel_labels"],
            "keyframe_count": compatibility_record["keyframe_count"],
            "keyframe_samples": compatibility_record["keyframe_samples"],
            "trajectory_sha256": compatibility_record["trajectory_sha256"],
            "diffraction": simulation_diffraction,
            "max_diffraction_order": simulation_max_diffraction_order,
            "by_layout": {
                layout: rendered[layout]["record"] for layout in selected_layouts
            },
        },
        "inputs": inputs,
        "outputs": outputs,
        "events": event_records,
        "qa": {
            "event_clock_and_gain": {
                "status": "pass",
                "source": "AudioProgram.events + assemble_dry_audio_buses",
                "gain_applied_once": True,
                "normalization": False,
                "peak_abs_by_stream": peak_abs,
                "peak_dbfs_by_stream": peak_dbfs,
            },
            "wet_tail": {
                "status": "pass",
                "event_count": len(event_records),
                "intervals": wet_tail_intervals,
                "source": "event-isolated time-varying convolution full_tail",
            },
            "binaural_channels": {
                "status": "pass" if "binaural" in selected_layouts else "not_requested",
                "layout": "binaural",
                "channel_count": 2,
                "channel_labels": ["left", "right"],
                "hrtf": str(hrtf) if hrtf is not None else None,
            },
            "propagation": {
                "status": "pass",
                "diffraction": simulation_diffraction,
                "max_diffraction_order": simulation_max_diffraction_order,
                "simulation": simulation_dict,
            },
            "neutral_readback": {
                "status": "pass",
                "path": neutral_record.get("path"),
                "coordinate_frame": (
                    neutral_readback_data.get("coordinate_frame")
                    if isinstance(neutral_readback_data, Mapping)
                    else None
                ),
            },
        },
    }
    # The historical UE report called this section ``dynamic_rir`` while the
    # M5 entry called it ``rir``. Keep both aliases during the transition;
    # they describe the same shared sequence and cache record.
    receipt["dynamic_rir"] = deepcopy(receipt["rir"])
    if execution_label is not None:
        receipt["execution_variant"] = execution_label
    if prepared_manifest_path is not None:
        receipt["prepared_audio_manifest"] = _input_record(prepared_manifest_path)
    # The neutral/P7 shared path preserves the selected AudioProgram source
    # window, but it has no transcript or sentence-boundary oracle. Keep those
    # claims separate so prepared/cropped speech is never promoted to a
    # complete-sentence assertion merely because its PCM rendered successfully.
    # The historical direct dynamic entry keeps its established report field.
    shared_neutral_output = (
        neutral_readback_path is not None
        or neutral_readback_data is not None
        or prepared_manifest_path is not None
    )
    if shared_neutral_output:
        receipt["source_clip_preserved"] = True
        receipt["source_clip_preservation"] = {
            "status": "pass",
            "scope": "selected_audio_program_source_windows",
            "basis": "dry_audio_assembly_rejected_additional_source_window_crop",
        }
        receipt["sentence_preservation"] = {
            "status": "not_assessed",
            "reason": (
                "shared audio rendering has no transcript or sentence-boundary "
                "oracle; source activity and PCM continuity do not prove sentence completeness"
            ),
        }
        receipt["complete_sentences_preserved"] = None
    else:
        receipt["complete_sentences_preserved"] = True
    receipt["native_rlr_dynamic"] = True
    receipt["formal_admission"] = False
    (output / "research_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return receipt


def render_neutral_readback_audio(
    neutral_readback: Mapping[str, Any] | str | Path,
    *,
    audio_program_path: str | Path | None = None,
    audio_program: Mapping[str, Any] | None = None,
    source_endpoint_by_entity: Mapping[str, str] | None = None,
    simulation_request_path: str | Path | None = None,
    simulation_mapping: Mapping[str, Any] | None = None,
    package_manifest_path: str | Path,
    source_endpoint_registry_path: str | Path | None = None,
    sound_asset_registry_path: str | Path | None = None,
    external_sound_asset_paths: Mapping[str, str | Path] | None = None,
    event_asset_bindings: Mapping[str, str | Path] | None = None,
    event_metadata: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    prepared_manifest_path: str | Path | None = None,
    hrtf_file_path: str | Path | None = None,
    output_path: str | Path,
    position_authority: str = "neutral_readback.entities[].emitter",
    listener_authority: str = "neutral_readback.camera[0]",
    rir_stride_frames: int = 3,
    variant_id: str = "A",
    execution_variant: str | None = None,
    hrtf_license_path: str | Path | None = None,
    extra_inputs: Mapping[str, Any] | None = None,
    layouts: Sequence[str] | None = None,
    rir_sequence_override: Mapping[str, Mapping[str, Any]] | None = None,
    scene_override: Any | None = None,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    diffraction: bool | None = None,
    max_diffraction_order: int | None = None,
) -> dict[str, Any]:
    """Shared renderer entry that consumes a validated P1 NeutralReadback."""
    neutral = _load_neutral_input(neutral_readback)
    if isinstance(audio_program, Mapping):
        program = audio_program
    elif audio_program_path is not None:
        try:
            program = json.loads(
                Path(audio_program_path).expanduser().resolve().read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CurrentMP3DDynamicAudioError(
                f"cannot read AudioProgram for neutral rendering: {error}"
            ) from error
    else:
        raise CurrentMP3DDynamicAudioError(
            "neutral rendering requires an AudioProgram path or mapping"
        )
    if not isinstance(program, Mapping):
        raise CurrentMP3DDynamicAudioError("AudioProgram must be an object")
    source_ids = tuple(
        str(value) for value in program.get("candidate_source_endpoint_ids") or ()
    )
    if len(source_ids) < 2:
        raise CurrentMP3DDynamicAudioError(
            "neutral rendering requires at least two program source endpoints"
        )
    trajectories = _neutral_source_trajectories(
        neutral,
        source_ids,
        source_endpoint_by_entity=source_endpoint_by_entity,
    )
    listener_position, listener_orientation = _neutral_camera_pose(neutral)
    clock = neutral["clock"]
    source_path = (
        Path(neutral_readback).expanduser().resolve()
        if not isinstance(neutral_readback, Mapping)
        else None
    )
    return render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=listener_position,
        listener_orientation_wxyz=listener_orientation,
        simulation_request_path=simulation_request_path,
        simulation_mapping=simulation_mapping,
        package_manifest_path=package_manifest_path,
        audio_program_path=audio_program_path,
        audio_program=audio_program,
        source_endpoint_registry_path=source_endpoint_registry_path,
        sound_asset_registry_path=sound_asset_registry_path,
        external_sound_asset_paths=external_sound_asset_paths,
        event_asset_bindings=event_asset_bindings,
        event_metadata=event_metadata,
        prepared_manifest_path=prepared_manifest_path,
        hrtf_file_path=hrtf_file_path,
        output_path=output_path,
        position_authority=position_authority,
        listener_authority=listener_authority,
        rir_stride_frames=rir_stride_frames,
        variant_id=variant_id,
        execution_variant=execution_variant,
        hrtf_license_path=hrtf_license_path,
        extra_inputs=extra_inputs,
        visual_frame_count=int(clock["frame_count"]),
        visual_frame_rate_hz=clock["frame_rate_hz"],
        timeline_tick_rate_hz=int(clock["time_base_hz"]),
        ticks_per_frame=int(clock["ticks_per_frame"]),
        layouts=layouts,
        neutral_readback_path=source_path,
        neutral_readback_data=neutral if source_path is None else None,
        rir_sequence_override=rir_sequence_override,
        scene_override=scene_override,
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
        diffraction=diffraction,
        max_diffraction_order=max_diffraction_order,
    )


def render_current_mp3d_dynamic_audio(
    *,
    visual_capture_dir: str | Path,
    m1_request_path: str | Path,
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    audio_program_path: str | Path,
    source_endpoint_registry_path: str | Path | None,
    sound_asset_registry_path: str | Path | None,
    external_sound_asset_paths: Mapping[str, str | Path] | None,
    hrtf_file_path: str | Path | None = None,
    output_path: str | Path,
    rir_stride_frames: int = 3,
    variant_id: str = "A",
    execution_variant: str | None = None,
    hrtf_license_path: str | Path | None = None,
    frame_count: int | None = None,
    frame_rate_hz: int | float | None = None,
    ticks_per_frame: int | None = None,
    layouts: Sequence[str] | None = None,
    neutral_readback_path: str | Path | None = None,
    event_asset_bindings: Mapping[str, str | Path] | None = None,
    prepared_manifest_path: str | Path | None = None,
    diffraction: bool | None = None,
    max_diffraction_order: int | None = None,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    source_endpoint_by_entity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Render the current MP3D route through the shared neutral renderer."""
    program_path = Path(audio_program_path).resolve()
    try:
        program = json.loads(program_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CurrentMP3DDynamicAudioError(
            f"cannot read MP3D AudioProgram: {error}"
        ) from error
    if not isinstance(program, Mapping):
        raise CurrentMP3DDynamicAudioError("audio program must be a JSON object")
    source_ids = tuple(
        str(value) for value in program.get("candidate_source_endpoint_ids") or ()
    )
    if neutral_readback_path is not None:
        return render_neutral_readback_audio(
            neutral_readback_path,
            audio_program_path=program_path,
            simulation_request_path=simulation_request_path,
            package_manifest_path=package_manifest_path,
            source_endpoint_registry_path=source_endpoint_registry_path,
            sound_asset_registry_path=sound_asset_registry_path,
            external_sound_asset_paths=external_sound_asset_paths,
            source_endpoint_by_entity=source_endpoint_by_entity,
            event_asset_bindings=event_asset_bindings,
            prepared_manifest_path=prepared_manifest_path,
            hrtf_file_path=hrtf_file_path,
            output_path=output_path,
            layouts=layouts,
            rir_stride_frames=rir_stride_frames,
            variant_id=variant_id,
            execution_variant=execution_variant,
            hrtf_license_path=hrtf_license_path,
            position_authority="P1 NeutralReadback entities[].emitter",
            listener_authority="P1 NeutralReadback camera[0]",
            diffraction=diffraction,
            max_diffraction_order=max_diffraction_order,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
            extra_inputs={
                "m1_request": _input_record(m1_request_path),
                **(
                    {
                        "visual_capture_frame_records": _input_record(
                            Path(visual_capture_dir).resolve() / "frame_records.json"
                        )
                    }
                    if (Path(visual_capture_dir).resolve() / "frame_records.json").is_file()
                    else {}
                ),
            },
        )
    clock = load_captured_render_clock(
        visual_capture_dir,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        ticks_per_frame=ticks_per_frame,
    )
    trajectories = load_captured_source_paths(
        visual_capture_dir,
        source_ids,
        frame_count=int(clock["frame_count"]),
        frame_rate_hz=clock["frame_rate_hz"],
        ticks_per_frame=int(clock["ticks_per_frame"]),
    )
    m1_request = json.loads(
        Path(m1_request_path).resolve().read_text(encoding="utf-8")
    )
    listener_position, listener_wxyz = listener_pose_from_m1_request(m1_request)
    return render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=listener_position,
        listener_orientation_wxyz=listener_wxyz,
        simulation_request_path=simulation_request_path,
        package_manifest_path=package_manifest_path,
        audio_program_path=program_path,
        source_endpoint_registry_path=source_endpoint_registry_path,
        sound_asset_registry_path=sound_asset_registry_path,
        external_sound_asset_paths=external_sound_asset_paths,
        event_asset_bindings=event_asset_bindings,
        prepared_manifest_path=prepared_manifest_path,
        hrtf_file_path=hrtf_file_path,
        output_path=output_path,
        layouts=layouts,
        position_authority="current-visual frame_records per-frame source_positions_m",
        listener_authority="research M1 request primary_camera_rig composed with rig_from_listener",
        rir_stride_frames=rir_stride_frames,
        variant_id=variant_id,
        execution_variant=execution_variant,
        hrtf_license_path=hrtf_license_path,
        visual_frame_count=int(clock["frame_count"]),
        visual_frame_rate_hz=clock["frame_rate_hz"],
        timeline_tick_rate_hz=int(clock["time_base_hz"]),
        ticks_per_frame=int(clock["ticks_per_frame"]),
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
        extra_inputs={
            "visual_capture_frame_records": _input_record(
                Path(visual_capture_dir).resolve() / "frame_records.json"
            ),
            "m1_request": _input_record(m1_request_path),
        },
        diffraction=diffraction,
        max_diffraction_order=max_diffraction_order,
    )
