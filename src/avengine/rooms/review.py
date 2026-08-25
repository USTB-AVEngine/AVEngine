"""Build the M6 six-case human-review video without promoting room claims.

The review has six *cases*, not six distinct houses: four real visual room
lineages, two MP3D acoustic-representation cases sharing one visual clip, and
one deliberately corrupted non-room fixture.  The builder normalizes every
segment through FFmpeg, always emits a stereo track, and visibly labels a
synthetic silent track as ``AUDIO UNAVAILABLE``.

This is review media.  In particular, available MP3D-derived audio may be
labelled ``research_only`` or ``unqualified`` but can never be represented as
qualified by this contract.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import file_record, load_json, write_json
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


REQUEST_SCHEMA = "avengine_m6_six_case_review_request_v1"
MANIFEST_SCHEMA = "avengine_m6_six_case_review_manifest_v1"
REQUEST_SCHEMA_FILE = "m6_six_case_review_request_v1.schema.json"
MANIFEST_SCHEMA_FILE = "m6_six_case_review_manifest_v1.schema.json"

CANONICAL_ROLES = (
    "controlled_blender_room",
    "replicacad_room",
    "legacy_ue_apartment_room",
    "mp3d_raw_representation",
    "mp3d_derived_representation",
    "corrupted_fixture_negative",
)
_ROLE_KIND = {
    "controlled_blender_room": "room_review",
    "replicacad_room": "room_review",
    "legacy_ue_apartment_room": "room_review",
    "mp3d_raw_representation": "acoustic_representation_review",
    "mp3d_derived_representation": "acoustic_representation_review",
    "corrupted_fixture_negative": "diagnostic_negative",
}
_ROOM_ROLES = frozenset(CANONICAL_ROLES[:-1])


class M6ReviewError(ValueError):
    """The review request, media, tool execution, or readback is invalid."""

    def __init__(self, errors: str | Sequence[str]):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _schema(filename: str) -> Mapping[str, Any]:
    source = _repository_root() / "schemas" / filename
    if source.is_file():
        return load_json(source)
    installed = Path(os.sys.prefix) / "share" / "avengine" / "schemas" / filename
    if installed.is_file():
        return load_json(installed)
    raise M6ReviewError(f"review schema is unavailable: {filename}")


def _schema_errors(value: Any, filename: str) -> list[str]:
    validator = Draft202012Validator(_schema(filename))
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def _safe_repository_path(value: Any, *, owner: str) -> str | None:
    if not isinstance(value, str) or not value:
        return f"{owner} must be a nonempty repository-relative path"
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return f"{owner} must be a normalized repository-relative path"
    if "\\" in value:
        return f"{owner} must use POSIX separators"
    return None


def validate_six_case_review_request(value: Any) -> list[str]:
    """Return schema and cross-case semantic errors for one review request."""

    errors = _schema_errors(value, REQUEST_SCHEMA_FILE)
    if errors or not isinstance(value, Mapping):
        return errors

    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        return errors
    roles = tuple(case.get("role") for case in cases if isinstance(case, Mapping))
    if roles != CANONICAL_ROLES:
        errors.append(
            "cases must use the canonical six roles in review order: "
            + ", ".join(CANONICAL_ROLES)
        )
    case_ids = [case.get("case_id") for case in cases if isinstance(case, Mapping)]
    if len(set(case_ids)) != len(case_ids):
        errors.append("case_id values must be unique")

    by_role = {
        case.get("role"): case for case in cases if isinstance(case, Mapping)
    }
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            continue
        role = case.get("role")
        if role in _ROLE_KIND and case.get("case_kind") != _ROLE_KIND[role]:
            errors.append(f"cases.{index}.case_kind does not match role {role}")
        expected_room = role in _ROOM_ROLES
        if case.get("is_room") is not expected_room:
            errors.append(f"cases.{index}.is_room must be {expected_room} for {role}")
        has_room_fields = isinstance(case.get("room_id"), str) and isinstance(
            case.get("room_lineage_id"), str
        )
        if expected_room and not has_room_fields:
            errors.append(f"cases.{index} must bind room_id and room_lineage_id")
        if not expected_room and (
            "room_id" in case or "room_lineage_id" in case
        ):
            errors.append(
                f"cases.{index} is a non-room fixture and must not bind room fields"
            )
        for media_key in ("visual", "audio"):
            media = case.get(media_key)
            if isinstance(media, Mapping) and "path" in media:
                path_error = _safe_repository_path(
                    media.get("path"), owner=f"cases.{index}.{media_key}.path"
                )
                if path_error:
                    errors.append(path_error)

        status = case.get("status")
        audio = case.get("audio")
        if isinstance(status, Mapping) and isinstance(audio, Mapping):
            if (
                status.get("qualification") != "qualified"
                and audio.get("evidence_tier") == "qualified"
            ):
                errors.append(
                    f"cases.{index} cannot bind qualified audio to an unqualified case"
                )

    if all(role in by_role for role in CANONICAL_ROLES):
        room_cases = [by_role[role] for role in CANONICAL_ROLES[:-1]]
        lineages = [case.get("room_lineage_id") for case in room_cases]
        if len(set(lineages)) != 4:
            errors.append(
                "the five room cases must represent exactly four room lineages"
            )

        raw = by_role["mp3d_raw_representation"]
        derived = by_role["mp3d_derived_representation"]
        if raw.get("room_id") != derived.get("room_id") or raw.get(
            "room_lineage_id"
        ) != derived.get("room_lineage_id"):
            errors.append("MP3D raw and derived must bind the same room lineage")
        derived_visual = derived.get("visual")
        if not isinstance(derived_visual, Mapping) or (
            derived_visual.get("availability") != "shared"
            or derived_visual.get("reuse_from_case_id") != raw.get("case_id")
        ):
            errors.append(
                "MP3D derived must explicitly reuse the MP3D raw visual case"
            )
        derived_status = derived.get("status")
        derived_audio = derived.get("audio")
        if isinstance(derived_status, Mapping):
            if derived_status.get("qualification") == "qualified":
                errors.append("MP3D derived is research-only and cannot be qualified")
            if derived_status.get("value") == "pass":
                errors.append(
                    "MP3D derived subject status cannot be pass before qualification"
                )
        if isinstance(derived_audio, Mapping) and derived_audio.get(
            "evidence_tier"
        ) == "qualified":
            errors.append("MP3D derived audio cannot be labelled qualified")

        corrupted = by_role["corrupted_fixture_negative"]
        corrupted_audio = corrupted.get("audio")
        if isinstance(corrupted_audio, Mapping) and corrupted_audio.get(
            "availability"
        ) == "available":
            errors.append("the corrupted non-room fixture cannot provide room audio")
    return errors


def load_six_case_review_request(path: str | Path) -> dict[str, Any]:
    try:
        value = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise M6ReviewError(f"could not load review request {path}: {exc}") from exc
    errors = validate_six_case_review_request(value)
    if errors:
        raise M6ReviewError(errors)
    return value


def _tool(value: str | Path, *, owner: str) -> str:
    executable = shutil.which(os.fspath(value))
    if executable is None:
        raise M6ReviewError(f"{owner} executable is unavailable: {value}")
    return executable


def _run(command: Sequence[str], *, owner: str, timeout: float = 180.0) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise M6ReviewError(f"could not execute {owner}: {exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise M6ReviewError(
            f"{owner} returned {completed.returncode}: {message}"
        )
    return completed.stdout


def _version(executable: str, *, owner: str) -> str:
    output = _run([executable, "-version"], owner=f"{owner} version", timeout=30.0)
    return output.splitlines()[0].strip()


def _fraction(value: Any, *, owner: str) -> float:
    try:
        parsed = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise M6ReviewError(f"{owner} is not a valid fraction: {value!r}") from exc
    if parsed <= 0:
        raise M6ReviewError(f"{owner} must be positive")
    return float(parsed)


def _probe(path: Path, *, ffprobe: str) -> dict[str, Any]:
    output = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,width,height,avg_frame_rate,"
                "sample_rate,channels:format=duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        owner=f"ffprobe {path.name}",
        timeout=60.0,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise M6ReviewError(f"ffprobe returned malformed JSON for {path}") from exc
    streams = value.get("streams") if isinstance(value, Mapping) else None
    if not isinstance(streams, list):
        raise M6ReviewError(f"ffprobe returned no stream list for {path}")
    video = next(
        (item for item in streams if item.get("codec_type") == "video"), None
    )
    audio = next(
        (item for item in streams if item.get("codec_type") == "audio"), None
    )
    duration_raw = value.get("format", {}).get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise M6ReviewError(f"ffprobe returned no finite duration for {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise M6ReviewError(f"ffprobe returned invalid duration for {path}")
    return {
        "duration_seconds": duration,
        "video": video,
        "audio": audio,
    }


def _resolve_media_path(
    repository_root: Path,
    raw_path: str,
    *,
    owner: str,
    policy: WorkspacePathPolicy,
) -> Path:
    error = _safe_repository_path(raw_path, owner=owner)
    if error:
        raise M6ReviewError(error)
    return policy.resolve_input(
        repository_root / PurePosixPath(raw_path), owner=owner, kind="file"
    )


def _escape_filter_path(path: Path) -> str:
    # FFmpeg's filter parser consumes these escapes even though no shell is used.
    value = str(path)
    for source, replacement in (
        ("\\", "\\\\"),
        (":", "\\:"),
        ("'", "\\'"),
        (",", "\\,"),
        ("[", "\\["),
        ("]", "\\]"),
    ):
        value = value.replace(source, replacement)
    return value


def _font_path() -> Path:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise M6ReviewError("no supported review title font is installed")


def _title_lines(case: Mapping[str, Any], *, index: int) -> list[str]:
    status = case["status"]
    audio = case["audio"]
    first = f"{index:02d}/06  {case['title']}"
    second = f"SUBJECT={str(status['value']).upper()} ({status['scope']})"
    third = f"QUALIFICATION={str(status['qualification']).upper()}"
    availability = audio["availability"]
    if availability == "available":
        tier = str(audio["evidence_tier"]).upper().replace("_", " ")
        presentation = str(audio["presentation_format"]).upper()
        if case["role"] == "mp3d_derived_representation":
            fourth = f"AUDIO={presentation} / RESEARCH ONLY (NOT QUALIFIED)"
        else:
            fourth = f"AUDIO={presentation} / {tier}"
    else:
        fourth = (
            f"AUDIO UNAVAILABLE ({str(availability).upper()}) - SILENCE ADDED"
        )
    lines = [first, second, third, fourth]
    if case["role"] == "corrupted_fixture_negative":
        lines.append("NON-ROOM DIAGNOSTIC FIXTURE")
    return lines


def _visual_source(
    case: Mapping[str, Any],
    *,
    by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str | None]:
    visual = case["visual"]
    if visual["availability"] != "shared":
        return visual, None
    reused_id = visual["reuse_from_case_id"]
    reused = by_id.get(reused_id)
    if reused is None:
        raise M6ReviewError(
            f"{case['case_id']} reuses unknown visual case {reused_id}"
        )
    source = reused["visual"]
    if source["availability"] == "shared":
        raise M6ReviewError("nested shared visual references are not supported")
    return source, reused_id


def _segment_command(
    *,
    case: Mapping[str, Any],
    effective_visual: Mapping[str, Any],
    resolved_visual: Path | None,
    resolved_audio: Path | None,
    title_path: Path,
    title_lines: Sequence[str],
    destination: Path,
    profile: Mapping[str, Any],
    ffmpeg: str,
    font: Path,
) -> list[str]:
    width = int(profile["width"])
    height = int(profile["height"])
    fps = int(profile["frame_rate_hz"])
    duration = float(profile["segment_duration_seconds"])
    sample_rate = int(profile["audio_sample_rate_hz"])
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n"]
    if resolved_visual is None:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"color=c=0x18212b:s={width}x{height}:r={fps}:d={duration:.9f}",
            ]
        )
    else:
        command.extend(
            [
                "-ss",
                f"{float(effective_visual['start_seconds']):.9f}",
                "-i",
                str(resolved_visual),
            ]
        )
    audio = case["audio"]
    if resolved_audio is None:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}",
            ]
        )
    else:
        command.extend(
            [
                "-ss",
                f"{float(audio['start_seconds']):.9f}",
                "-i",
                str(resolved_audio),
            ]
        )

    escaped_font = _escape_filter_path(font)
    escaped_title = _escape_filter_path(title_path)
    longest_line = max(len(line) for line in title_lines)
    width_limited_size = int((width - 48) / max(1.0, longest_line * 0.62))
    # Keep the review label inside the 120 px letterbox used by the 480 px
    # diagnostic sources on a 720 px canvas.  This preserves the source's own
    # top-of-frame event/flag annotations instead of covering them.
    font_size = max(8, min(24, height // 26, width_limited_size))
    line_spacing = 3
    band_height = min(
        height - 1,
        max(
            60,
            12
            + len(title_lines) * font_size
            + (len(title_lines) - 1) * line_spacing,
        ),
    )
    video_filter = (
        f"[0:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps},tpad=stop_mode=clone:stop_duration={duration:.9f},"
        f"trim=duration={duration:.9f},setpts=PTS-STARTPTS,"
        f"drawbox=x=0:y=0:w=iw:h={band_height}:color=black@0.78:t=fill,"
        f"drawtext=fontfile='{escaped_font}':textfile='{escaped_title}':"
        f"fontcolor=white:fontsize={font_size}:line_spacing={line_spacing}:x=18:y=6:"
        "expansion=none[v]"
    )
    if resolved_audio is not None and int(audio["channel_count"]) == 1:
        channel_filter = "pan=stereo|c0=c0|c1=c0,"
    else:
        channel_filter = "aformat=channel_layouts=stereo,"
    audio_filter = (
        f"[1:a:0]aresample={sample_rate},{channel_filter}"
        f"atrim=duration={duration:.9f},asetpts=PTS-STARTPTS,"
        f"apad=pad_dur={duration:.9f},atrim=duration={duration:.9f}[a]"
    )
    command.extend(
        [
            "-filter_complex",
            f"{video_filter};{audio_filter}",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{duration:.9f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(fps),
            "-keyint_min",
            str(fps),
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-threads",
            "1",
            "-video_track_timescale",
            "48000",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    return command


def _concat_command(
    *, ffmpeg: str, concat_file: Path, destination: Path
) -> list[str]:
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def plan_six_case_review(
    request: Mapping[str, Any],
    *,
    repository_root: str | Path,
    staging_directory: str | Path,
    ffmpeg: str | Path = "ffmpeg",
    check_media: bool = False,
) -> dict[str, Any]:
    """Return exact FFmpeg commands without executing or publishing them.

    ``check_media=False`` intentionally supports hermetic command/semantic
    tests and planning before the optional ReplicaCAD clip exists.
    """

    errors = validate_six_case_review_request(request)
    if errors:
        raise M6ReviewError(errors)
    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise M6ReviewError(f"repository root is not a directory: {root}")
    staging = Path(staging_directory).resolve()
    executable = (
        _tool(ffmpeg, owner="ffmpeg") if check_media else os.fspath(ffmpeg)
    )
    policy = WorkspacePathPolicy.from_roots([root])
    by_id = {case["case_id"]: case for case in request["cases"]}
    font = _font_path()
    segments: list[dict[str, Any]] = []
    for index, case in enumerate(request["cases"], start=1):
        effective_visual, reused_id = _visual_source(case, by_id=by_id)
        visual_path: Path | None = None
        if effective_visual["availability"] == "available":
            if check_media:
                visual_path = _resolve_media_path(
                    root,
                    effective_visual["path"],
                    owner=f"{case['case_id']} visual media",
                    policy=policy,
                )
            else:
                visual_path = root / PurePosixPath(effective_visual["path"])
        audio = case["audio"]
        audio_path: Path | None = None
        if audio["availability"] == "available":
            if check_media:
                audio_path = _resolve_media_path(
                    root,
                    audio["path"],
                    owner=f"{case['case_id']} audio media",
                    policy=policy,
                )
            else:
                audio_path = root / PurePosixPath(audio["path"])

        segment_name = f"{index:02d}_{case['case_id']}.mp4"
        title_path = staging / "titles" / f"{index:02d}_{case['case_id']}.txt"
        destination = staging / "segments" / segment_name
        lines = _title_lines(case, index=index)
        command = _segment_command(
            case=case,
            effective_visual=effective_visual,
            resolved_visual=visual_path,
            resolved_audio=audio_path,
            title_path=title_path,
            title_lines=lines,
            destination=destination,
            profile=request["output_profile"],
            ffmpeg=executable,
            font=font,
        )
        segments.append(
            {
                "index": index,
                "case": case,
                "effective_visual": effective_visual,
                "reused_visual_case_id": reused_id,
                "resolved_visual": visual_path,
                "resolved_audio": audio_path,
                "title_lines": lines,
                "title_path": title_path,
                "destination": destination,
                "command": command,
            }
        )
    concat_file = staging / "concat.txt"
    combined = staging / "m6_six_case_review.mp4"
    return {
        "segments": segments,
        "concat_file": concat_file,
        "combined_video": combined,
        "concat_command": _concat_command(
            ffmpeg=executable, concat_file=concat_file, destination=combined
        ),
    }


def _validate_source_media(
    plan: Mapping[str, Any],
    *,
    ffprobe: str,
) -> None:
    probe_cache: dict[Path, dict[str, Any]] = {}
    for segment in plan["segments"]:
        case = segment["case"]
        visual = segment["resolved_visual"]
        effective_visual = segment["effective_visual"]
        if visual is not None:
            if visual not in probe_cache:
                probe_cache[visual] = _probe(visual, ffprobe=ffprobe)
            report = probe_cache[visual]
            if report["video"] is None:
                raise M6ReviewError(f"{case['case_id']} visual media has no video stream")
            if report["duration_seconds"] <= float(effective_visual["start_seconds"]):
                raise M6ReviewError(
                    f"{case['case_id']} visual start lies beyond the source duration"
                )
        audio = segment["resolved_audio"]
        if audio is not None:
            if audio not in probe_cache:
                probe_cache[audio] = _probe(audio, ffprobe=ffprobe)
            report = probe_cache[audio]
            stream = report["audio"]
            if stream is None:
                raise M6ReviewError(f"{case['case_id']} audio media has no audio stream")
            declared_channels = int(case["audio"]["channel_count"])
            if int(stream.get("channels", 0)) != declared_channels:
                raise M6ReviewError(
                    f"{case['case_id']} audio channel declaration differs from media"
                )
            if report["duration_seconds"] <= float(case["audio"]["start_seconds"]):
                raise M6ReviewError(
                    f"{case['case_id']} audio start lies beyond the source duration"
                )


def _output_probe(
    path: Path,
    *,
    ffprobe: str,
    profile: Mapping[str, Any],
    expected_duration: float,
) -> dict[str, Any]:
    report = _probe(path, ffprobe=ffprobe)
    video = report["video"]
    audio = report["audio"]
    if not isinstance(video, Mapping) or video.get("codec_name") != "h264":
        raise M6ReviewError(f"review output is not H.264: {path}")
    if int(video.get("width", 0)) != int(profile["width"]) or int(
        video.get("height", 0)
    ) != int(profile["height"]):
        raise M6ReviewError(f"review output dimensions differ: {path}")
    fps = _fraction(video.get("avg_frame_rate"), owner=f"{path.name} frame rate")
    if abs(fps - float(profile["frame_rate_hz"])) > 1.0e-9:
        raise M6ReviewError(f"review output frame rate differs: {path}")
    if not isinstance(audio, Mapping) or audio.get("codec_name") != "aac":
        raise M6ReviewError(f"review output is missing AAC audio: {path}")
    if int(audio.get("channels", 0)) != 2 or int(
        audio.get("sample_rate", 0)
    ) != int(profile["audio_sample_rate_hz"]):
        raise M6ReviewError(f"review output audio layout differs: {path}")
    tolerance = max(0.08, 1.5 / float(profile["frame_rate_hz"]))
    if abs(report["duration_seconds"] - expected_duration) > tolerance:
        raise M6ReviewError(
            f"review output duration differs for {path}: "
            f"expected {expected_duration}, got {report['duration_seconds']}"
        )
    return {
        "duration_seconds": report["duration_seconds"],
        "video_codec": "h264",
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate_hz": fps,
        "audio_codec": "aac",
        "audio_sample_rate_hz": int(audio["sample_rate"]),
        "audio_channel_count": int(audio["channels"]),
    }


def _input_record(
    *,
    media: Mapping[str, Any],
    resolved: Path | None,
    repository_root: Path,
    render_mode: str,
    reuse_from_case_id: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "availability": media["availability"],
        "render_mode": render_mode,
    }
    if resolved is not None:
        source = file_record(resolved, relative_to=repository_root)
        record.update(source)
    if reuse_from_case_id is not None:
        record["reuse_from_case_id"] = reuse_from_case_id
    if "reason" in media:
        record["reason"] = media["reason"]
    if "evidence_tier" in media:
        record["evidence_tier"] = media["evidence_tier"]
    return record


def build_six_case_review(
    *,
    request_path: str | Path,
    output_directory: str | Path,
    repository_root: str | Path | None = None,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> tuple[Path, Path]:
    """Encode and atomically publish one immutable six-case review package."""

    root = Path(repository_root or _repository_root()).resolve(strict=True)
    if not root.is_dir():
        raise M6ReviewError(f"repository root is not a directory: {root}")
    policy = WorkspacePathPolicy.from_roots([root])
    request_source = policy.resolve_input(
        request_path, owner="six-case review request", kind="file"
    )
    request = load_six_case_review_request(request_source)
    destination = policy.resolve_output(
        output_directory, owner="six-case review package", create_parent=True
    )
    ffmpeg_executable = _tool(ffmpeg, owner="ffmpeg")
    ffprobe_executable = _tool(ffprobe, owner="ffprobe")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    try:
        (staging / "segments").mkdir()
        (staging / "titles").mkdir()
        request_copy = staging / "request.json"
        shutil.copyfile(request_source, request_copy)
        plan = plan_six_case_review(
            request,
            repository_root=root,
            staging_directory=staging,
            ffmpeg=ffmpeg_executable,
            check_media=True,
        )
        _validate_source_media(plan, ffprobe=ffprobe_executable)

        segment_records: list[dict[str, Any]] = []
        duration = float(request["output_profile"]["segment_duration_seconds"])
        for segment in plan["segments"]:
            segment["title_path"].write_text(
                "\n".join(segment["title_lines"]) + "\n", encoding="utf-8"
            )
            _run(
                segment["command"],
                owner=f"encode review segment {segment['case']['case_id']}",
            )
            probe = _output_probe(
                segment["destination"],
                ffprobe=ffprobe_executable,
                profile=request["output_profile"],
                expected_duration=duration,
            )
            case = segment["case"]
            visual_render_mode = (
                "placeholder"
                if segment["resolved_visual"] is None
                else (
                    "shared_source_media"
                    if segment["reused_visual_case_id"] is not None
                    else "source_media"
                )
            )
            audio_render_mode = (
                "source_audio"
                if segment["resolved_audio"] is not None
                else "silent_unavailable"
            )
            rendered = file_record(segment["destination"], relative_to=staging)
            rendered["probe"] = probe
            record: dict[str, Any] = {
                "index": segment["index"],
                "case_id": case["case_id"],
                "role": case["role"],
                "case_kind": case["case_kind"],
                "title": case["title"],
                "is_room": case["is_room"],
                "status": deepcopy(case["status"]),
                "visual_input": _input_record(
                    media=(
                        case["visual"]
                        if segment["reused_visual_case_id"] is None
                        else {
                            **segment["effective_visual"],
                            "availability": "shared",
                        }
                    ),
                    resolved=segment["resolved_visual"],
                    repository_root=root,
                    render_mode=visual_render_mode,
                    reuse_from_case_id=segment["reused_visual_case_id"],
                ),
                "audio_input": _input_record(
                    media=case["audio"],
                    resolved=segment["resolved_audio"],
                    repository_root=root,
                    render_mode=audio_render_mode,
                ),
                "title_lines": segment["title_lines"],
                "rendered_segment": rendered,
            }
            if case["is_room"]:
                record["room_id"] = case["room_id"]
                record["room_lineage_id"] = case["room_lineage_id"]
            segment_records.append(record)

        concat_lines = [
            "file '" + str(segment["destination"]).replace("'", "'\\''") + "'"
            for segment in plan["segments"]
        ]
        plan["concat_file"].write_text(
            "\n".join(concat_lines) + "\n", encoding="utf-8"
        )
        _run(plan["concat_command"], owner="concatenate six review segments")
        combined_probe = _output_probe(
            plan["combined_video"],
            ffprobe=ffprobe_executable,
            profile=request["output_profile"],
            expected_duration=duration * 6,
        )
        combined = file_record(plan["combined_video"], relative_to=staging)
        combined["probe"] = combined_probe
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "review_id": request["review_id"],
            "request": file_record(request_copy, relative_to=staging),
            "output_profile": deepcopy(request["output_profile"]),
            "case_semantics": {
                "case_count": 6,
                "real_room_lineage_count": 4,
                "mp3d_raw_derived_share_visual": True,
                "corrupted_fixture_is_room": False,
            },
            "segments": segment_records,
            "combined_video": combined,
            "tool_versions": {
                "ffmpeg": _version(ffmpeg_executable, owner="ffmpeg"),
                "ffprobe": _version(ffprobe_executable, owner="ffprobe"),
            },
        }
        manifest_errors = _schema_errors(manifest, MANIFEST_SCHEMA_FILE)
        if manifest_errors:
            raise M6ReviewError(
                [f"generated review manifest: {error}" for error in manifest_errors]
            )
        write_json(staging / "review_manifest.json", manifest)
        shutil.rmtree(staging / "titles")
        plan["concat_file"].unlink()
        published = atomic_publish_directory(policy, staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return published / "review_manifest.json", published / "m6_six_case_review.mp4"


def _review_check(
    checks: list[dict[str, Any]], check_id: str, errors: Sequence[str]
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "pass" if not errors else "fail",
            "errors": list(errors),
        }
    )


def _review_record_errors(
    record: Mapping[str, Any],
    *,
    root: Path,
    owner: str,
) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    raw_path = record.get("path")
    path_error = _safe_repository_path(raw_path, owner=f"{owner}.path")
    if path_error:
        return None, [path_error]
    candidate = (root / PurePosixPath(raw_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, [f"{owner}.path escapes its declared root"]
    if not candidate.is_file():
        return None, [f"{owner} is missing: {raw_path}"]
    if record.get("byte_size") != candidate.stat().st_size:
        errors.append(f"{owner} byte size differs")
    actual = file_record(candidate, relative_to=root)
    if record.get("sha256") != actual["sha256"]:
        errors.append(f"{owner} SHA-256 differs")
    return candidate, errors


def verify_six_case_review(
    manifest_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    ffprobe: str | Path = "ffprobe",
) -> dict[str, Any]:
    """Re-open every declared input/output and verify one published review.

    A review manifest is useful release evidence only if its six segment files,
    combined media, copied request, source-media bindings and room-lineage
    semantics still agree.  This verifier performs that check without
    rebuilding or replacing the package.
    """

    checks: list[dict[str, Any]] = []
    try:
        root = Path(repository_root or _repository_root()).resolve(strict=True)
        source = Path(manifest_path)
        if not source.is_absolute():
            source = root / source
        source = source.resolve(strict=True)
        source.relative_to(root)
        package = source.parent
        manifest = load_json(source)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _review_check(checks, "manifest_json", [f"could not load manifest: {exc}"])
        return {
            "schema": "avengine_m6_six_case_review_verification_v1",
            "status": "fail",
            "checks": checks,
        }
    _review_check(checks, "manifest_json", [])
    schema_errors = _schema_errors(manifest, MANIFEST_SCHEMA_FILE)
    _review_check(checks, "manifest_schema", schema_errors)
    if schema_errors:
        return {
            "schema": "avengine_m6_six_case_review_verification_v1",
            "status": "fail",
            "checks": checks,
        }

    request_path, request_record_errors = _review_record_errors(
        manifest["request"], root=package, owner="request"
    )
    request: dict[str, Any] | None = None
    if request_path is not None:
        try:
            request = load_six_case_review_request(request_path)
        except M6ReviewError as exc:
            request_record_errors.extend(exc.errors)
    _review_check(checks, "request_binding", request_record_errors)

    semantic_errors: list[str] = []
    segments = manifest["segments"]
    if request is not None:
        if manifest.get("review_id") != request.get("review_id"):
            semantic_errors.append("manifest review_id differs from request")
        if manifest.get("output_profile") != request.get("output_profile"):
            semantic_errors.append("manifest output_profile differs from request")
        for index, (case, segment) in enumerate(
            zip(request["cases"], segments, strict=True), start=1
        ):
            expected = {
                key: case[key]
                for key in ("case_id", "role", "case_kind", "title", "is_room")
            }
            expected["index"] = index
            for key, value in expected.items():
                if segment.get(key) != value:
                    semantic_errors.append(
                        f"segment {index} {key} differs from request"
                    )
            if segment.get("status") != case.get("status"):
                semantic_errors.append(f"segment {index} status differs from request")
            if segment.get("title_lines") != _title_lines(case, index=index):
                semantic_errors.append(
                    f"segment {index} title lines differ from request"
                )
            if case["is_room"]:
                for key in ("room_id", "room_lineage_id"):
                    if segment.get(key) != case.get(key):
                        semantic_errors.append(
                            f"segment {index} {key} differs from request"
                        )
            elif "room_id" in segment or "room_lineage_id" in segment:
                semantic_errors.append(
                    f"segment {index} non-room fixture binds room identity"
                )
    _review_check(checks, "case_semantics", semantic_errors)

    source_errors: list[str] = []
    source_media: list[
        tuple[Mapping[str, Any], Mapping[str, Any], Path | None, Path | None]
    ] = []
    if request is not None:
        policy = WorkspacePathPolicy.from_roots([root])
        by_id = {case["case_id"]: case for case in request["cases"]}
        for index, (case, segment) in enumerate(
            zip(request["cases"], segments, strict=True), start=1
        ):
            try:
                effective_visual, reused_id = _visual_source(case, by_id=by_id)
                visual_path = (
                    _resolve_media_path(
                        root,
                        effective_visual["path"],
                        owner=f"{case['case_id']} visual media",
                        policy=policy,
                    )
                    if effective_visual["availability"] == "available"
                    else None
                )
                audio = case["audio"]
                audio_path = (
                    _resolve_media_path(
                        root,
                        audio["path"],
                        owner=f"{case['case_id']} audio media",
                        policy=policy,
                    )
                    if audio["availability"] == "available"
                    else None
                )
                expected_visual = _input_record(
                    media=(
                        case["visual"]
                        if reused_id is None
                        else {**effective_visual, "availability": "shared"}
                    ),
                    resolved=visual_path,
                    repository_root=root,
                    render_mode=(
                        "placeholder"
                        if visual_path is None
                        else "shared_source_media" if reused_id else "source_media"
                    ),
                    reuse_from_case_id=reused_id,
                )
                expected_audio = _input_record(
                    media=audio,
                    resolved=audio_path,
                    repository_root=root,
                    render_mode=(
                        "source_audio" if audio_path is not None else "silent_unavailable"
                    ),
                )
                if segment.get("visual_input") != expected_visual:
                    source_errors.append(
                        f"segment {index} visual_input differs from request/source bytes"
                    )
                if segment.get("audio_input") != expected_audio:
                    source_errors.append(
                        f"segment {index} audio_input differs from request/source bytes"
                    )
                source_media.append(
                    (case, effective_visual, visual_path, audio_path)
                )
            except (M6ReviewError, OSError, ValueError) as exc:
                source_errors.append(f"segment {index} source binding failed: {exc}")
    raw_visual = segments[3]["visual_input"]
    derived_visual = segments[4]["visual_input"]
    for field in ("path", "byte_size", "sha256"):
        if raw_visual.get(field) != derived_visual.get(field):
            source_errors.append(
                f"MP3D raw/derived visual {field} differs"
            )
    if derived_visual.get("reuse_from_case_id") != segments[3].get("case_id"):
        source_errors.append("MP3D derived visual lacks exact raw-case reuse binding")

    try:
        ffprobe_executable = _tool(ffprobe, owner="ffprobe")
    except M6ReviewError as exc:
        source_errors.extend(exc.errors)
        _review_check(checks, "source_media", source_errors)
        _review_check(checks, "segment_media", exc.errors)
        _review_check(checks, "combined_media", ["ffprobe is unavailable"])
        _review_check(checks, "package_closure", ["media checks did not complete"])
        return {
            "schema": "avengine_m6_six_case_review_verification_v1",
            "status": "fail",
            "checks": checks,
        }

    source_probe_cache: dict[Path, dict[str, Any]] = {}
    for index, (case, effective_visual, visual_path, audio_path) in enumerate(
        source_media, start=1
    ):
        if visual_path is not None:
            try:
                report = source_probe_cache.setdefault(
                    visual_path, _probe(visual_path, ffprobe=ffprobe_executable)
                )
                if report["video"] is None:
                    source_errors.append(f"segment {index} visual source has no video")
                if report["duration_seconds"] <= float(
                    effective_visual["start_seconds"]
                ):
                    source_errors.append(
                        f"segment {index} visual start exceeds source duration"
                    )
            except M6ReviewError as exc:
                source_errors.extend(
                    f"segment {index} visual source: {error}" for error in exc.errors
                )
        if audio_path is not None:
            try:
                report = source_probe_cache.setdefault(
                    audio_path, _probe(audio_path, ffprobe=ffprobe_executable)
                )
                stream = report["audio"]
                if stream is None:
                    source_errors.append(f"segment {index} audio source has no audio")
                elif int(stream.get("channels", 0)) != int(
                    case["audio"]["channel_count"]
                ):
                    source_errors.append(
                        f"segment {index} audio channel declaration differs"
                    )
                if report["duration_seconds"] <= float(
                    case["audio"]["start_seconds"]
                ):
                    source_errors.append(
                        f"segment {index} audio start exceeds source duration"
                    )
            except M6ReviewError as exc:
                source_errors.extend(
                    f"segment {index} audio source: {error}" for error in exc.errors
                )
    _review_check(checks, "source_media", source_errors)

    profile = manifest["output_profile"]
    duration = float(profile["segment_duration_seconds"])
    segment_errors: list[str] = []
    for index, segment in enumerate(segments, start=1):
        media_path, errors = _review_record_errors(
            segment["rendered_segment"],
            root=package,
            owner=f"segments[{index - 1}].rendered_segment",
        )
        segment_errors.extend(errors)
        if media_path is None or errors:
            continue
        try:
            probe = _output_probe(
                media_path,
                ffprobe=ffprobe_executable,
                profile=profile,
                expected_duration=duration,
            )
            if probe != segment["rendered_segment"].get("probe"):
                segment_errors.append(f"segment {index} probe differs")
        except M6ReviewError as exc:
            segment_errors.extend(
                f"segment {index}: {error}" for error in exc.errors
            )
    _review_check(checks, "segment_media", segment_errors)

    combined_errors: list[str] = []
    combined_path, errors = _review_record_errors(
        manifest["combined_video"], root=package, owner="combined_video"
    )
    combined_errors.extend(errors)
    if combined_path is not None and not errors:
        try:
            probe = _output_probe(
                combined_path,
                ffprobe=ffprobe_executable,
                profile=profile,
                expected_duration=duration * 6,
            )
            if probe != manifest["combined_video"].get("probe"):
                combined_errors.append("combined video probe differs")
        except M6ReviewError as exc:
            combined_errors.extend(exc.errors)
    _review_check(checks, "combined_media", combined_errors)

    closure_errors: list[str] = []
    expected_files = {
        source,
        package / manifest["request"]["path"],
        package / manifest["combined_video"]["path"],
        *(
            package / segment["rendered_segment"]["path"]
            for segment in segments
        ),
    }
    expected_files = {path.resolve() for path in expected_files}
    expected_directories = {(package / "segments").resolve()}
    for entry in package.rglob("*"):
        if entry.is_symlink():
            closure_errors.append(
                f"review package contains symlink: {entry.relative_to(package)}"
            )
            continue
        resolved = entry.resolve()
        if entry.is_dir():
            if resolved not in expected_directories:
                closure_errors.append(
                    f"review package contains undeclared directory: "
                    f"{entry.relative_to(package)}"
                )
        elif entry.is_file():
            if resolved not in expected_files:
                closure_errors.append(
                    f"review package contains undeclared file: "
                    f"{entry.relative_to(package)}"
                )
        else:
            closure_errors.append(
                f"review package contains non-regular entry: {entry.relative_to(package)}"
            )
    for expected in expected_files:
        if not expected.is_file():
            closure_errors.append(f"declared review file is missing: {expected}")
    _review_check(checks, "package_closure", closure_errors)

    return {
        "schema": "avengine_m6_six_case_review_verification_v1",
        "status": (
            "pass" if all(check["status"] == "pass" for check in checks) else "fail"
        ),
        "checks": checks,
    }


__all__ = [
    "CANONICAL_ROLES",
    "M6ReviewError",
    "build_six_case_review",
    "load_six_case_review_request",
    "plan_six_case_review",
    "validate_six_case_review_request",
    "verify_six_case_review",
]
