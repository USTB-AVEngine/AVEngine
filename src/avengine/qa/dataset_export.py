"""Export verified QA-v3 media as model-neutral public and private records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import subprocess
from typing import Any


class DatasetExportError(ValueError):
    """A released item cannot be exported without changing its meaning."""


_REQUIRED_TEXT = (
    "question_id",
    "scene_id",
    "episode_id",
    "point_id",
    "profile_id",
    "form",
    "task_type",
)
_SCORING_FIELDS = (
    "truth_interval_deg",
    "convention",
    "azimuth_convention",
    "certification_policy",
    "refusal_truth",
)
_FORBIDDEN_PUBLIC_KEYS = {
    "answer",
    "answer_key",
    "fact",
    "facts",
    "gold",
    "truth",
}


def _read_json(path: Path, *, owner: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetExportError(f"cannot read {owner} {path}: {exc}") from exc


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetExportError(f"released item has no non-empty {key}")
    return value.strip()


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatasetExportError(f"{owner} must be a positive integer")
    return int(value)


def _positive_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetExportError(f"{owner} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DatasetExportError(f"{owner} must be a positive finite number")
    return result


def _normalise_clock(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetExportError("released item media_clock must be an object")
    required = ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count")
    missing = [key for key in required if key not in value]
    if missing:
        raise DatasetExportError(f"released item media_clock missing {missing}")
    clock = {
        "frame_count": _positive_int(value["frame_count"], owner="media_clock.frame_count"),
        "frame_rate_hz": _positive_number(
            value["frame_rate_hz"], owner="media_clock.frame_rate_hz"
        ),
        "sample_rate_hz": _positive_int(
            value["sample_rate_hz"], owner="media_clock.sample_rate_hz"
        ),
        "sample_count": _positive_int(
            value["sample_count"], owner="media_clock.sample_count"
        ),
    }
    expected_seconds = clock["frame_count"] / clock["frame_rate_hz"]
    declared_seconds = value.get("clip_seconds", expected_seconds)
    clock["clip_seconds"] = _positive_number(
        declared_seconds, owner="media_clock.clip_seconds"
    )
    if abs(clock["clip_seconds"] - expected_seconds) > 1.0e-6:
        raise DatasetExportError(
            "media_clock.clip_seconds disagrees with frame_count/frame_rate_hz"
        )
    return clock


def _normalise_layouts(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = [str(values)]
    result: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            raise DatasetExportError("layout names must be strings")
        for part in raw.split(","):
            name = part.strip()
            if not name:
                raise DatasetExportError("layout names must be non-empty")
            if name in result:
                raise DatasetExportError(f"duplicate requested layout: {name}")
            result.append(name)
    if not result:
        raise DatasetExportError("at least one requested layout is required")
    return tuple(result)


def _find_audio_receipt(audio_path: Path, *, pipeline_root: Path) -> Path:
    pipeline_root = pipeline_root.resolve()
    for parent in (audio_path.parent, *audio_path.parents):
        candidate = parent / "research_receipt.json"
        if candidate.is_file():
            try:
                candidate.resolve().relative_to(pipeline_root)
            except ValueError:
                raise DatasetExportError(
                    f"audio receipt is outside pipeline root: {candidate.resolve()}"
                )
            return candidate.resolve()
        if parent == pipeline_root:
            break
    raise DatasetExportError(f"cannot find point audio receipt for {audio_path}")


def _audio_info(path: Path) -> dict[str, int]:
    try:
        import soundfile as sf

        info = sf.info(path)
    except (ImportError, OSError, RuntimeError) as exc:
        raise DatasetExportError(f"cannot inspect WAV {path}: {exc}") from exc
    if info.channels < 1 or info.samplerate < 1 or info.frames < 1:
        raise DatasetExportError(f"WAV has invalid header values: {path}")
    return {
        "channel_count": int(info.channels),
        "sample_rate_hz": int(info.samplerate),
        "sample_count": int(info.frames),
    }


def _parse_rate(value: Any) -> float:
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0.0:
            raise DatasetExportError(f"invalid video frame rate {value!r}")
        return float(numerator) / denominator_value
    return float(text)


def probe_video(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DatasetExportError(f"cannot run ffprobe for {path}: {exc}") from exc
    if completed.returncode != 0:
        raise DatasetExportError(
            f"ffprobe failed for {path}: {completed.stderr.strip()[-400:]}"
        )
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        raw_frames = stream.get("nb_read_frames", stream.get("nb_frames"))
        frames = int(raw_frames)
        rate = _parse_rate(stream["r_frame_rate"])
        duration = float(stream["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetExportError(f"ffprobe returned invalid video metadata for {path}") from exc
    if frames < 1 or not math.isfinite(rate) or rate <= 0.0:
        raise DatasetExportError(f"video has invalid frame metadata: {path}")
    if not math.isfinite(duration) or duration <= 0.0:
        raise DatasetExportError(f"video has invalid duration: {path}")
    return {"frame_count": frames, "frame_rate_hz": rate, "duration_seconds": duration}


def _validate_video(
    path: Path,
    clock: Mapping[str, Any],
    *,
    video_probe: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetExportError(f"released video is missing: {path}")
    actual = dict(video_probe(path))
    frames = _positive_int(actual.get("frame_count"), owner="video.frame_count")
    rate = _positive_number(actual.get("frame_rate_hz"), owner="video.frame_rate_hz")
    duration = _positive_number(
        actual.get("duration_seconds"), owner="video.duration_seconds"
    )
    if frames != clock["frame_count"]:
        raise DatasetExportError(
            f"video frame count {frames} differs from media clock {clock['frame_count']}"
        )
    if abs(rate - float(clock["frame_rate_hz"])) > 1.0e-6:
        raise DatasetExportError(
            f"video frame rate {rate} differs from media clock {clock['frame_rate_hz']}"
        )
    tolerance = max(1.0 / float(clock["frame_rate_hz"]), 0.05)
    if abs(duration - float(clock["clip_seconds"])) > tolerance:
        raise DatasetExportError(
            f"video duration {duration} differs from media clock {clock['clip_seconds']}"
        )
    return {
        "path": str(path.resolve()),
        "frame_count": frames,
        "frame_rate_hz": rate,
        "duration_seconds": duration,
    }


def _layout_path(point_root: Path, value: Mapping[str, Any]) -> Path:
    declared = value.get("mixture_path", value.get("path"))
    if declared is not None:
        path = Path(str(declared)).expanduser()
        if not path.is_absolute():
            path = point_root / path
        return path.resolve()
    directory = value.get("output_directory")
    if not isinstance(directory, str) or not directory.strip():
        raise DatasetExportError("audio layout has no path or output_directory")
    return (point_root / "audio" / directory / "mixture.wav").resolve()


def _layout_records(
    receipt_path: Path,
    *,
    requested_layouts: tuple[str, ...] | None,
    clock: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipt = _read_json(receipt_path, owner="audio receipt")
    if not isinstance(receipt, Mapping):
        raise DatasetExportError(f"audio receipt must be an object: {receipt_path}")
    if receipt.get("execution_variant") not in (None, "main"):
        raise DatasetExportError(
            f"released audio receipt is not the main execution variant: {receipt_path}"
        )
    audio = receipt.get("audio")
    if not isinstance(audio, Mapping):
        raise DatasetExportError(f"audio receipt has no audio object: {receipt_path}")
    by_layout = audio.get("by_layout")
    if not isinstance(by_layout, Mapping) or not by_layout:
        raise DatasetExportError(f"audio receipt has no by_layout declaration: {receipt_path}")
    declared_order = audio.get("layouts")
    if isinstance(declared_order, list) and declared_order:
        observed_names = tuple(str(name) for name in declared_order)
    else:
        observed_names = tuple(str(name) for name in by_layout)
    selected = requested_layouts or observed_names
    missing = [name for name in selected if name not in by_layout]
    if missing:
        raise DatasetExportError(
            f"audio receipt {receipt_path} is missing requested layouts {missing}; "
            f"declared={list(observed_names)}"
        )
    result: dict[str, dict[str, Any]] = {}
    point_root = receipt_path.parent
    for name in selected:
        value = by_layout[name]
        if not isinstance(value, Mapping):
            raise DatasetExportError(f"audio layout {name!r} must be an object")
        path = _layout_path(point_root, value)
        if not path.is_file():
            raise DatasetExportError(f"audio layout {name!r} mixture is missing: {path}")
        actual = _audio_info(path)
        declared_channels = _positive_int(
            value.get("channel_count"), owner=f"audio.{name}.channel_count"
        )
        declared_rate = _positive_int(
            value.get("sample_rate_hz"), owner=f"audio.{name}.sample_rate_hz"
        )
        declared_count = _positive_int(
            value.get("sample_count"), owner=f"audio.{name}.sample_count"
        )
        labels = value.get("channel_labels")
        if not isinstance(labels, list) or len(labels) != declared_channels or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ):
            raise DatasetExportError(
                f"audio layout {name!r} channel_labels do not match channel_count"
            )
        expected = {
            "channel_count": declared_channels,
            "sample_rate_hz": declared_rate,
            "sample_count": declared_count,
        }
        if actual != expected:
            raise DatasetExportError(
                f"audio layout {name!r} WAV header {actual} differs from receipt {expected}"
            )
        if declared_rate != clock["sample_rate_hz"] or declared_count != clock["sample_count"]:
            raise DatasetExportError(
                f"audio layout {name!r} clock differs from released media clock"
            )
        result[name] = {
            "path": str(path),
            "layout_type": str(value.get("layout_type", name)),
            "channel_count": declared_channels,
            "channel_labels": [label.strip() for label in labels],
            "sample_rate_hz": declared_rate,
            "sample_count": declared_count,
        }
    return result


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _assert_public_safe(records: Sequence[Mapping[str, Any]]) -> None:
    for index, record in enumerate(records):
        leaked = sorted(set(_walk_keys(record)) & _FORBIDDEN_PUBLIC_KEYS)
        if leaked:
            raise DatasetExportError(
                f"public record {index} contains private answer keys {leaked}"
            )


def build_dataset_records(
    released_items: Sequence[Mapping[str, Any]],
    *,
    pipeline_root: Path,
    layouts: Sequence[str] | None = None,
    video_probe: Callable[[Path], Mapping[str, Any]] = probe_video,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(released_items, (str, bytes)) or not isinstance(released_items, Sequence):
        raise DatasetExportError("released items must be a JSON list")
    requested_layouts = _normalise_layouts(layouts)
    public_records: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(released_items):
        if not isinstance(raw, Mapping):
            raise DatasetExportError(f"released item {index} must be an object")
        row = dict(raw)
        common = {key: _required_text(row, key) for key in _REQUIRED_TEXT}
        question_id = common["question_id"]
        if question_id in seen:
            raise DatasetExportError(f"duplicate question_id: {question_id!r}")
        seen.add(question_id)
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise DatasetExportError(f"released item {question_id!r} has no question")
        options = row.get("options", [])
        if not isinstance(options, list) or any(
            not isinstance(option, str) or not option.strip() for option in options
        ):
            raise DatasetExportError(f"released item {question_id!r} has invalid options")
        clock = _normalise_clock(row.get("media_clock"))
        audio_path = Path(_required_text(row, "audio")).expanduser().resolve()
        if not audio_path.is_file():
            raise DatasetExportError(f"released audio is missing: {audio_path}")
        receipt_path = _find_audio_receipt(audio_path, pipeline_root=pipeline_root)
        audio_by_layout = _layout_records(
            receipt_path, requested_layouts=requested_layouts, clock=clock
        )
        video_path = Path(_required_text(row, "video")).expanduser().resolve()
        video = _validate_video(video_path, clock, video_probe=video_probe)
        optional_ids = {
            key: row[key]
            for key in ("group_id", "pilot_id")
            if isinstance(row.get(key), str) and row[key].strip()
        }
        extra_variants = row.get("audio_by_variant")
        audio_by_variant = None
        if extra_variants is not None:
            if not isinstance(extra_variants, Mapping) or not extra_variants:
                raise DatasetExportError(
                    f"released item {question_id!r} audio_by_variant must be an object"
                )
            audio_by_variant = {}
            for variant, extra_path in extra_variants.items():
                if not isinstance(variant, str) or not variant.strip():
                    raise DatasetExportError(
                        f"released item {question_id!r} has a blank audio variant"
                    )
                extra = Path(str(extra_path)).expanduser().resolve()
                if not extra.is_file():
                    raise DatasetExportError(
                        f"released item {question_id!r} audio variant "
                        f"{variant!r} is missing: {extra}"
                    )
                audio_by_variant[variant] = str(extra)
            if "main" not in audio_by_variant:
                raise DatasetExportError(
                    f"released item {question_id!r} audio_by_variant must include main"
                )
        public = {
            **common,
            **optional_ids,
            "question": question.strip(),
            "options": [option.strip() for option in options],
            "video": video,
            "media_clock": clock,
            "audio_by_layout": audio_by_layout,
        }
        if audio_by_variant is not None:
            public["audio_variants"] = list(audio_by_variant)
            public["audio_by_variant"] = audio_by_variant
        if "truth" not in row:
            raise DatasetExportError(f"released item {question_id!r} has no truth")
        scoring = {
            key: row[key]
            for key in _SCORING_FIELDS
            if key in row
        }
        private = {
            **common,
            **optional_ids,
            "truth": row["truth"],
            "options": [option.strip() for option in options],
            "scoring": scoring,
        }
        public_records.append(public)
        private_records.append(private)
    _assert_public_safe(public_records)
    public_ids = [row["question_id"] for row in public_records]
    private_ids = [row["question_id"] for row in private_records]
    if public_ids != private_ids:
        raise DatasetExportError("public/private question ID order differs")
    return public_records, private_records


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def export_dataset(
    *,
    pipeline_root: Path,
    released_items_path: Path,
    output_root: Path,
    layouts: Sequence[str] | None = None,
    video_probe: Callable[[Path], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    pipeline_root = pipeline_root.expanduser().resolve()
    released_items_path = released_items_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise DatasetExportError(f"refusing to overwrite output root: {output_root}")
    raw = _read_json(released_items_path, owner="released items")
    if not isinstance(raw, list):
        raise DatasetExportError("released items JSON must contain a list")
    public, private = build_dataset_records(
        raw, pipeline_root=pipeline_root, layouts=layouts, video_probe=video_probe
    )
    pipeline_manifest_path = pipeline_root / "pipeline_manifest.json"
    pipeline_manifest = (
        _read_json(pipeline_manifest_path, owner="pipeline manifest")
        if pipeline_manifest_path.is_file()
        else {}
    )
    if pipeline_manifest and not isinstance(pipeline_manifest, Mapping):
        raise DatasetExportError("pipeline manifest must be an object")
    observed_layouts = sorted(
        {name for row in public for name in row["audio_by_layout"]}
    )
    requested = _normalise_layouts(layouts)
    manifest = {
        "kind": "qa_v3_model_neutral_dataset_export",
        "status": "complete",
        "qualification_claim": False,
        "inputs": {
            "pipeline_root": str(pipeline_root),
            "released_items": str(released_items_path),
            "pipeline_source": pipeline_manifest.get("source"),
        },
        "counts": {
            "questions": len(public),
            "private_answers": len(private),
            "scenes": len({row["scene_id"] for row in public}),
            "episodes": len({row["episode_id"] for row in public}),
            "profiles": dict(sorted(Counter(row["profile_id"] for row in public).items())),
            "forms": dict(sorted(Counter(row["form"] for row in public).items())),
        },
        "layouts": {
            "requested": list(requested) if requested is not None else None,
            "observed": observed_layouts,
        },
        "failures": list(pipeline_manifest.get("failures") or []),
        "shortfalls": list(pipeline_manifest.get("shortfalls") or []),
        "outputs": {
            "public_questions": str(output_root / "public/questions.jsonl"),
            "private_answers": str(output_root / "private/answers.jsonl"),
        },
    }
    output_root.mkdir(parents=True)
    (output_root / "public").mkdir()
    (output_root / "private").mkdir()
    _write_jsonl(output_root / "public/questions.jsonl", public)
    _write_jsonl(output_root / "private/answers.jsonl", private)
    with (output_root / "dataset_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest
