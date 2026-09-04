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
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np

from avengine.contracts.json_io import sha256_file
from avengine.contracts.transforms import compose_transforms
from avengine.rooms.contracts import validate_capture_request
from avengine.acoustics.runtime import load_compiled_acoustic_scene
from avengine.spatial_audio.audio import write_float32_wav
from avengine.spatial_audio.current_request_pair_ir import _load_simulation_request
from avengine.capture.acoustics import (
    build_strided_review_keyframes,
    render_research_review_binaural_audio,
    render_research_review_binaural_rir_sequence,
)
from avengine.timeline.audio_program import bind_audio_program_hash
from avengine.timeline.audio_render import assemble_audio_program_dry_buses
from avengine.registry.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
    sound_index,
)

CURRENT_MP3D_DYNAMIC_AUDIO_SCHEMA = "avengine_m5_current_mp3d_dynamic_audio_v1"
DEFAULT_VISUAL_FRAME_RATE_HZ = 15
DEFAULT_EPISODE_FRAME_COUNT = 75
DEFAULT_TIMELINE_TICK_RATE_HZ = 48_000
DEFAULT_TICKS_PER_FRAME = 3_200
AUDIO_SAMPLE_RATE_HZ = 16_000
DEFAULT_EPISODE_SAMPLE_COUNT = 80_000

# Backward-compatible names retained for the original current-MP3D route.
VISUAL_FRAME_RATE_HZ = DEFAULT_VISUAL_FRAME_RATE_HZ
EPISODE_FRAME_COUNT = DEFAULT_EPISODE_FRAME_COUNT
EPISODE_SAMPLE_COUNT = DEFAULT_EPISODE_SAMPLE_COUNT
CLAIM_BOUNDARY = (
    "Research review audio only. Captured per-frame source positions and one "
    "AudioProgram routing variant drive strided per-state binaural RIRs; no "
    "dataset admission, qualification, or new gate is claimed."
)


class CurrentMP3DDynamicAudioError(RuntimeError):
    """Raised when the dynamic research-audio contract is violated."""


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
    reverberation tail. Speech producers select the whole utterance as their
    source window; the same rule then preserves its complete content.
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


def render_dynamic_research_audio(
    *,
    source_trajectories_m: Mapping[str, Sequence[Sequence[float]]],
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    audio_program_path: str | Path,
    source_endpoint_registry_path: str | Path,
    sound_asset_registry_path: str | Path,
    external_sound_asset_paths: Mapping[str, str | Path],
    hrtf_file_path: str | Path,
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
) -> dict[str, Any]:
    """Room-agnostic core: trajectories + program variant -> binaural episode."""

    execution_label = _validate_execution_variant(execution_variant)
    output = _fresh_output(output_path)
    program_path = Path(audio_program_path).resolve()
    program = json.loads(program_path.read_text(encoding="utf-8"))
    if not isinstance(program, Mapping):
        raise CurrentMP3DDynamicAudioError("audio program must be a JSON object")
    endpoint_registry_path = Path(source_endpoint_registry_path).resolve()
    sound_registry_path = Path(sound_asset_registry_path).resolve()
    endpoints = load_source_endpoint_registry(endpoint_registry_path)
    sounds = load_sound_asset_registry(sound_registry_path)

    source_ids = tuple(
        str(value) for value in program.get("candidate_source_endpoint_ids") or ()
    )
    if len(source_ids) < 2:
        raise CurrentMP3DDynamicAudioError(
            "the program must carry at least two candidate source endpoints"
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
    ordered_trajectories = {}
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
    _, simulation = _load_simulation_request(Path(simulation_request_path).resolve())
    scene = load_compiled_acoustic_scene(
        package_manifest_path, allow_nonpassing_research_qa=True
    )
    hrtf = Path(hrtf_file_path).resolve()
    sequence = render_research_review_binaural_rir_sequence(
        scene, simulation, grid=grid, hrtf_file_path=str(hrtf)
    )

    stems, mixture = render_research_review_binaural_audio(
        dry_buses, sequence, grid=grid
    )
    if int(grid.episode_sample_count) != expected_sample_count:
        raise CurrentMP3DDynamicAudioError(
            "dynamic acoustic grid sample boundary differs from the visual "
            "AudioProgram clock"
        )
    for source_id in source_ids:
        stem = stems.get(source_id)
        if stem is None:
            raise CurrentMP3DDynamicAudioError(
                f"dynamic renderer omitted source stem {source_id!r}"
            )
        _require_exact_episode_samples(
            stem.episode,
            expected=expected_sample_count,
            owner=f"binaural stem {source_id!r}",
            channel_major=True,
        )
    mixture = _require_exact_episode_samples(
        mixture,
        expected=expected_sample_count,
        owner="binaural mixture",
        channel_major=True,
    )

    audio_root = output / "audio"
    outputs: dict[str, str] = {}

    def _write(path: Path, samples: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_float32_wav(path, samples, AUDIO_SAMPLE_RATE_HZ)
        outputs[str(path.relative_to(output))] = sha256_file(path)

    for source_id in source_ids:
        dry = np.asarray(dry_buses[source_id], dtype=np.float64)[None, :]
        _write(audio_root / "dry" / f"{source_id}.wav", dry)
        _write(
            audio_root / "binaural" / f"{source_id}_stem.wav",
            stems[source_id].episode,
        )
    _write(audio_root / "binaural" / "mixture.wav", np.asarray(mixture))

    materialized = assembly.materialized_program
    inputs: dict[str, Any] = {
        "simulation_request": _input_record(simulation_request_path),
        "package_manifest": _input_record(package_manifest_path),
        "source_endpoint_registry": _input_record(endpoint_registry_path),
        "sound_asset_registry": _input_record(sound_registry_path),
        "hrtf": {
            **_input_record(hrtf),
            "license_path": (
                str(Path(hrtf_license_path).resolve())
                if hrtf_license_path is not None
                else None
            ),
        },
        "dry_assets": bindings,
    }
    if extra_inputs:
        inputs.update({key: value for key, value in extra_inputs.items()})

    receipt: dict[str, Any] = {
        "schema": CURRENT_MP3D_DYNAMIC_AUDIO_SCHEMA,
        "status": "pass",
        "claim_boundary": CLAIM_BOUNDARY,
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification_claim": False,
        "audio": {
            "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
            "sample_count": clock["sample_count"],
            "layout_type": "binaural",
            "channel_labels": ["left", "right"],
        },
        "audio_program": {
            "path": str(program_path),
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
        "rir": {
            "stride_frames": rir_stride_frames,
            "keyframe_count": len(sequence.keyframe_samples),
            "keyframe_samples": list(sequence.keyframe_samples),
            "trajectory_sha256": sequence.trajectory_sha256,
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    if execution_label is not None:
        receipt["execution_variant"] = execution_label
    (output / "research_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def render_current_mp3d_dynamic_audio(
    *,
    visual_capture_dir: str | Path,
    m1_request_path: str | Path,
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    audio_program_path: str | Path,
    source_endpoint_registry_path: str | Path,
    sound_asset_registry_path: str | Path,
    external_sound_asset_paths: Mapping[str, str | Path],
    hrtf_file_path: str | Path,
    output_path: str | Path,
    rir_stride_frames: int = 3,
    variant_id: str = "A",
    execution_variant: str | None = None,
    hrtf_license_path: str | Path | None = None,
    frame_count: int | None = None,
    frame_rate_hz: int | float | None = None,
    ticks_per_frame: int | None = None,
) -> dict[str, Any]:
    """Render one motion-following binaural research episode for MP3D."""

    program = json.loads(
        Path(audio_program_path).resolve().read_text(encoding="utf-8")
    )
    if not isinstance(program, Mapping):
        raise CurrentMP3DDynamicAudioError("audio program must be a JSON object")
    source_ids = tuple(
        str(value) for value in program.get("candidate_source_endpoint_ids") or ()
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
        audio_program_path=audio_program_path,
        source_endpoint_registry_path=source_endpoint_registry_path,
        sound_asset_registry_path=sound_asset_registry_path,
        external_sound_asset_paths=external_sound_asset_paths,
        hrtf_file_path=hrtf_file_path,
        output_path=output_path,
        position_authority=(
            "current-visual frame_records per-frame source_positions_m"
        ),
        listener_authority=(
            "research M1 request primary_camera_rig composed with "
            "rig_from_listener"
        ),
        rir_stride_frames=rir_stride_frames,
        variant_id=variant_id,
        execution_variant=execution_variant,
        hrtf_license_path=hrtf_license_path,
        visual_frame_count=int(clock["frame_count"]),
        visual_frame_rate_hz=clock["frame_rate_hz"],
        timeline_tick_rate_hz=int(clock["time_base_hz"]),
        ticks_per_frame=int(clock["ticks_per_frame"]),
        extra_inputs={
            "visual_capture_frame_records": _input_record(
                Path(visual_capture_dir).resolve() / "frame_records.json"
            ),
            "m1_request": _input_record(m1_request_path),
        },
    )
