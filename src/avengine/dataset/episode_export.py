"""Export shared-room Episodes and model-neutral QA records.

The exporter is intentionally a reference index. It does not copy videos,
audio, room packages, or evidence arrays. A room is recorded once, each
Episode points at that room and at separate plan/actual/media records, and
each QA item points at the Episode. Answers and model outcomes live in
private sidecars so a model-input JSONL cannot accidentally contain a gold
answer.

The input is an explicit avengine_episode_export_request_v1 JSON object.
Paths may be absolute (the normal form for external data roots) or relative to
the request file. Numeric RIR caches are accepted only through an explicit
role=numeric_rir cache entry; geometry files such as rir_*.obj are never
inferred to be numeric cache data.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import wave
from typing import Any


EPISODE_EXPORT_REQUEST_SCHEMA = "avengine_episode_export_request_v1"
EPISODE_EXPORT_SCHEMA = "avengine_episode_export_v1"
EPISODE_EXPORT_ERROR = "Episode export cannot preserve the requested lineage"


class EpisodeExportError(ValueError):
    """An Episode export request or referenced artifact is invalid."""


def _fail(message: str) -> EpisodeExportError:
    return EpisodeExportError(f"{EPISODE_EXPORT_ERROR}: {message}")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _read_json(path: Path, *, owner: str) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail(f"cannot read {owner} {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _fail(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _resolve_path(value: Any, *, base_dir: Path, owner: str) -> Path:
    raw = value.get("path") if isinstance(value, Mapping) else value
    if isinstance(raw, os.PathLike):
        raw = os.fspath(raw)
    if not isinstance(raw, str) or not raw.strip():
        raise _fail(f"{owner} path must be a non-empty string")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"{owner} path does not resolve: {candidate}: {exc}") from exc
    if not resolved.exists():
        raise _fail(f"{owner} path is missing: {resolved}")
    return resolved


def _path_record(
    path: Path,
    *,
    role: str,
    kind: str = "file",
    include_hash: bool = False,
) -> dict[str, Any]:
    if not isinstance(role, str) or not role.strip():
        raise _fail("reference role must be a non-empty string")
    if path.is_file():
        try:
            size = int(path.stat().st_size)
        except OSError as exc:
            raise _fail(f"cannot stat {path}: {exc}") from exc
        record: dict[str, Any] = {
            "role": role.strip(),
            "kind": kind,
            "path": str(path),
            "byte_size": size,
        }
        if include_hash:
            record["sha256"] = _sha256(path)
        return record
    if not path.is_dir():
        raise _fail(f"reference is neither a file nor directory: {path}")
    entries = []
    total = 0
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        try:
            total += int(child.stat().st_size)
        except OSError as exc:
            raise _fail(f"cannot stat {child}: {exc}") from exc
        entries.append(child)
    return {
        "role": role.strip(),
        "kind": "directory",
        "path": str(path),
        "byte_size": total,
        "entry_count": len(entries),
        "hash_scope": "directory_size_only",
    }


def _ref_entries(
    value: Any,
    *,
    base_dir: Path,
    owner: str,
    default_role: str,
    include_hash: bool = True,
) -> list[tuple[dict[str, Any], Path, Mapping[str, Any] | None]]:
    """Normalize a path, list, or role-to-path mapping into references."""

    if value is None:
        return []
    if isinstance(value, Mapping) and "path" not in value:
        raw_entries: list[Any] = [
            {"role": str(role), "path": child}
            for role, child in value.items()
        ]
    elif _is_sequence(value):
        raw_entries = list(value)
    else:
        raw_entries = [value]
    result: list[tuple[dict[str, Any], Path, Mapping[str, Any] | None]] = []
    for index, raw in enumerate(raw_entries):
        metadata: Mapping[str, Any] | None = raw if isinstance(raw, Mapping) else None
        role = (
            metadata.get("role", default_role)
            if metadata is not None
            else default_role
        )
        path = _resolve_path(raw, base_dir=base_dir, owner=f"{owner}[{index}]")
        explicit_hash_bound = bool(
            metadata.get("hash_bound", False)
            if metadata is not None
            else False
        )
        record = _path_record(
            path,
            role=str(role),
            include_hash=include_hash and explicit_hash_bound and path.is_file(),
        )
        if explicit_hash_bound:
            record["hash_bound"] = True
        result.append((record, path, metadata))
    return result


def _document_ref(
    value: Any,
    *,
    base_dir: Path,
    owner: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    entries = _ref_entries(
        value,
        base_dir=base_dir,
        owner=owner,
        default_role=owner,
    )
    if len(entries) != 1:
        raise _fail(f"{owner} must name exactly one JSON document")
    record, path, _ = entries[0]
    if not path.is_file():
        raise _fail(f"{owner} must reference a JSON file: {path}")
    payload = _read_json(path, owner=owner)
    if not isinstance(payload, Mapping):
        raise _fail(f"{owner} JSON must contain an object: {path}")
    summary = {
        "reference": record,
        "schema": payload.get("schema") or payload.get("kind"),
        "status": payload.get("status"),
        "top_level_keys": sorted(str(key) for key in payload),
    }
    return summary, payload


def _require_text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{owner} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _fail(f"{owner} must be a positive integer")
    return int(value)


def _positive_float(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{owner} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise _fail(f"{owner} must be a positive finite number")
    return result


def _normalise_retention(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("evidence_retention must be an explicit object")
    mode = value.get("mode")
    if mode not in {"minimal", "extended"}:
        raise _fail("evidence_retention.mode must be minimal or extended")
    include_extended = value.get("include_extended")
    if not isinstance(include_extended, bool):
        raise _fail("evidence_retention.include_extended must be boolean")
    preserve_required = value.get("preserve_required", True)
    if not isinstance(preserve_required, bool):
        raise _fail("evidence_retention.preserve_required must be boolean")
    if mode == "extended" and not include_extended:
        raise _fail("extended retention mode requires include_extended=true")
    return {
        "mode": mode,
        "include_extended": include_extended,
        "preserve_required": preserve_required,
        "configured": True,
    }


def _load_wav(path: Path) -> dict[str, Any]:
    """Read enough WAV metadata to prove a lossless stereo master."""

    try:
        import soundfile as sf

        info = sf.info(path)
        channels = int(info.channels)
        rate = int(info.samplerate)
        frames = int(info.frames)
        subtype = str(info.subtype or "").upper()
        lossless = subtype.startswith("PCM_") or subtype == "FLOAT"
    except (ImportError, OSError, RuntimeError):
        try:
            with wave.open(str(path), "rb") as handle:
                channels = int(handle.getnchannels())
                rate = int(handle.getframerate())
                frames = int(handle.getnframes())
            subtype = "PCM"
            lossless = True
        except (OSError, wave.Error) as exc:
            raise _fail(f"cannot inspect WAV {path}: {exc}") from exc
    if channels != 2:
        raise _fail(f"audio master must be stereo (2 channels): {path}")
    if rate < 1 or frames < 1:
        raise _fail(f"audio master has invalid WAV clock: {path}")
    if not lossless:
        raise _fail(f"audio master is not a lossless PCM/FLOAT WAV: {path}")
    return {
        "channel_count": channels,
        "channel_labels": ["left", "right"],
        "sample_rate_hz": rate,
        "sample_count": frames,
        "layout": "stereo",
        "lossless": True,
        "lossless_subtype": subtype,
    }


def probe_video(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Probe the actual video master with ffprobe."""

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
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _fail(f"cannot run ffprobe for {path}: {exc}") from exc
    if completed.returncode != 0:
        raise _fail(f"ffprobe failed for {path}: {completed.stderr.strip()[-400:]}")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        raw_frames = stream.get("nb_read_frames", stream.get("nb_frames"))
        frames = int(raw_frames)
        raw_rate = str(stream["r_frame_rate"])
        if "/" in raw_rate:
            numerator, denominator = raw_rate.split("/", 1)
            rate = float(numerator) / float(denominator)
        else:
            rate = float(raw_rate)
        duration = float(stream["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _fail(f"ffprobe returned invalid video metadata for {path}") from exc
    if frames < 1 or not math.isfinite(rate) or rate <= 0.0:
        raise _fail(f"video has invalid frame metadata: {path}")
    if not math.isfinite(duration) or duration <= 0.0:
        raise _fail(f"video has invalid duration: {path}")
    return {
        "frame_count": frames,
        "frame_rate_hz": rate,
        "duration_seconds": duration,
    }


def _declared_clock(
    episode: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    actual: Mapping[str, Any],
    audio: Mapping[str, Any],
    video: Mapping[str, Any],
) -> dict[str, Any]:
    raw = episode.get("media_clock")
    if raw is None:
        raw = actual.get("clock") or plan.get("clock")
    declared = dict(raw) if isinstance(raw, Mapping) else {}
    frame_count = int(video["frame_count"])
    frame_rate = float(video["frame_rate_hz"])
    sample_rate = int(audio["sample_rate_hz"])
    sample_count = int(audio["sample_count"])
    if "frame_count" in declared and _positive_int(
        declared["frame_count"], owner="media_clock.frame_count"
    ) != frame_count:
        raise _fail("declared media clock disagrees with video frame count")
    if "frame_rate_hz" in declared and abs(
        _positive_float(declared["frame_rate_hz"], owner="media_clock.frame_rate_hz")
        - frame_rate
    ) > 1.0e-6:
        raise _fail("declared media clock disagrees with video frame rate")
    if "sample_rate_hz" in declared and _positive_int(
        declared["sample_rate_hz"], owner="media_clock.sample_rate_hz"
    ) != sample_rate:
        raise _fail("declared media clock disagrees with WAV sample rate")
    if "sample_count" in declared and _positive_int(
        declared["sample_count"], owner="media_clock.sample_count"
    ) != sample_count:
        raise _fail("declared media clock disagrees with WAV sample count")
    clip_seconds = frame_count / frame_rate
    if "clip_seconds" in declared and abs(
        _positive_float(declared["clip_seconds"], owner="media_clock.clip_seconds")
        - clip_seconds
    ) > max(1.0 / frame_rate, 0.05):
        raise _fail("declared media clock disagrees with video duration")
    if abs(float(video["duration_seconds"]) - clip_seconds) > max(
        1.0 / frame_rate, 0.05
    ):
        raise _fail("video probe duration disagrees with its frame clock")
    return {
        "frame_count": frame_count,
        "frame_rate_hz": frame_rate,
        "clip_seconds": clip_seconds,
        "sample_rate_hz": sample_rate,
        "sample_count": sample_count,
    }


def _media_value(media: Mapping[str, Any], names: Sequence[str], *, owner: str) -> Any:
    for name in names:
        if name in media:
            return media[name]
    raise _fail(f"{owner} is missing; expected one of {list(names)}")


def _qa_source_rows(
    path: Path, *, owner: str
) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
    # Unified question sets keep concrete requirements in a sibling coverage
    # list. Attach the matching entry so private answers retain that metadata.
    payload = _read_json(path, owner=owner)
    coverage_by_key: dict[str, Mapping[str, Any]] = {}
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for key in ("samples", "questions", "records", "items"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if rows is None:
            raise _fail(f"{owner} has no samples/questions/records list")
        if isinstance(payload.get("deferred"), list):
            rows = [*rows, *payload["deferred"]]
        coverage = payload.get("coverage")
        if isinstance(coverage, list):
            for metadata in coverage:
                if not isinstance(metadata, Mapping):
                    continue
                for key in (
                    metadata.get("question_id"),
                    metadata.get("qa_id"),
                    metadata.get("catalog_id"),
                ):
                    if isinstance(key, str) and key.strip():
                        coverage_by_key.setdefault(key.strip(), metadata)
    else:
        raise _fail(f"{owner} must contain a JSON list or object")
    if not rows:
        raise _fail(f"{owner} contains no QA rows")
    result: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise _fail(f"{owner}[{index}] must be an object")
        metadata = None
        for key in (
            row.get("question_id"),
            row.get("qa_id"),
            row.get("catalog_id"),
        ):
            if isinstance(key, str) and key.strip():
                metadata = coverage_by_key.get(key.strip())
                if metadata is not None:
                    break
        result.append((row, metadata))
    return result


def _qa_rows(path: Path, *, owner: str) -> list[Mapping[str, Any]]:
    return [row for row, _metadata in _qa_source_rows(path, owner=owner)]

def _qa_answer(evaluation: Mapping[str, Any], row: Mapping[str, Any]) -> Any:
    answer = evaluation.get("answer")
    if isinstance(answer, Mapping) and "value" in answer:
        return answer["value"]
    if "truth" in evaluation:
        truth = evaluation["truth"]
        if isinstance(truth, Mapping) and "value" in truth:
            return truth["value"]
        return truth
    if "truth" in row:
        return row["truth"]
    if "answer" in row and not isinstance(row["answer"], Mapping):
        return row["answer"]
    return None


def _qa_status(evaluation: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    status = evaluation.get("status", row.get("status"))
    if status in {"pass", "qualified", "valid"}:
        return "pass"
    if status in {"deferred", "not_run", "unsupported", "rejected", "fail"}:
        return "not_run"
    if status is None:
        return "not_run"
    return str(status)


def _qa_question_text(
    evaluation: Mapping[str, Any], row: Mapping[str, Any]
) -> str | None:
    value = evaluation.get("question", row.get("question"))
    if isinstance(value, Mapping):
        value = value.get("en") or value.get("question_en") or value.get("zh")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _qa_options(
    evaluation: Mapping[str, Any], row: Mapping[str, Any]
) -> list[str]:
    value = evaluation.get("options", row.get("options"))
    if value is None and isinstance(row.get("model_input"), Mapping):
        mcq = row["model_input"].get("mcq")
        if isinstance(mcq, Mapping):
            value = mcq.get("options")
    if value is None:
        return []
    if not _is_sequence(value):
        raise _fail("QA options must be a list")
    result = []
    for option in value:
        if isinstance(option, Mapping):
            option = (
                option.get("label_en")
                or option.get("value")
                or option.get("label_zh")
            )
        if not isinstance(option, str) or not option.strip():
            raise _fail("QA options must contain non-empty text")
        result.append(option.strip())
    return result


def _safe_model_value(value: Any, *, key: str = "") -> Any:
    secret = re.compile(r"(?:api[_-]?key|token|secret|password|authorization)", re.I)
    if secret.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): _safe_model_value(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if _is_sequence(value):
        return [_safe_model_value(child, key=key) for child in value]
    return value


def _model_answer_value(value: Any) -> Any:
    """Keep a raw answer scalar without embedding arbitrary numeric arrays."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _model_scalar_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "question_id",
        "qa_id",
        "pair_id",
        "prediction",
        "model_answer",
        "answer",
        "raw_answer",
        "score",
    }
    return {
        str(key): _safe_model_value(child, key=str(key))
        for key, child in value.items()
        if str(key) not in excluded
        and (child is None or isinstance(child, (str, int, float, bool)))
    }


def _question_id(
    row: Mapping[str, Any],
    *,
    evaluation: Mapping[str, Any],
    episode_id: str,
) -> tuple[str, str]:
    local = row.get("question_id") or evaluation.get("question_id")
    if isinstance(local, str) and local.strip():
        return local.strip(), local.strip()
    label = (
        row.get("catalog_id")
        or evaluation.get("catalog_id")
        or row.get("qa_id")
        or evaluation.get("qa_id")
        or row.get("card")
        or evaluation.get("question_type")
        or "qa"
    )
    components = [
        label,
        row.get("card"),
        row.get("target_actor_id") or evaluation.get("target_actor_id"),
        evaluation.get("question_type"),
    ]
    slug_parts = []
    for value in components:
        if value is None:
            continue
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-")
        if slug and slug not in slug_parts:
            slug_parts.append(slug[:80])
    local = "-".join(slug_parts) or "qa"
    return f"{episode_id}::{local}", local


_SOURCE_ACTOR_ID = re.compile(r"(?<![A-Za-z0-9])source[0-9]+(?![A-Za-z0-9])", re.IGNORECASE)

_PUBLIC_MODEL_INPUT_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answer_index",
        "answer_value",
        "truth",
        "gold",
        "evidence",
        "fact",
        "facts",
        "forms",
        "form_status",
        "source_question_id",
        "episode_id",
        "selection_bucket",
    }
)


def _assert_public_model_input(value: Any, *, owner: str) -> None:
    if isinstance(value, Mapping):
        leaked = sorted(
            _PUBLIC_MODEL_INPUT_FORBIDDEN_KEYS & {str(key) for key in value}
        )
        if leaked:
            raise _fail(
                f"{owner} contains private model-input fields {leaked}"
            )
        for key, child in value.items():
            _assert_public_model_input(key, owner=owner)
            _assert_public_model_input(child, owner=owner)
    elif _is_sequence(value):
        for child in value:
            _assert_public_model_input(child, owner=owner)
    elif isinstance(value, str) and _SOURCE_ACTOR_ID.search(value):
        raise _fail(
            f"{owner} exposes an internal actor identifier in model-facing text"
        )


_OBSERVATION_KEYS = frozenset(
    {
        "observation_window",
        "observation_windows",
        "window",
        "window_frames",
        "window_seconds",
        "time_window",
        "time_windows",
        "statistics_window",
        "query_frame",
        "query_time",
        "query_time_s",
        "frame",
        "start_frame",
        "end_frame",
        "start_s",
        "end_s",
        "post_sound",
    }
)


def _public_question_id(
    source_question_id: str,
    *,
    episode_id: str,
    catalog_id: Any,
    ordinal: int,
    used: set[str],
) -> str:
    # Keep ordinary IDs stable, while replacing generated sourceN actor IDs in
    # the model-facing sidecar with an opaque Episode/catalog key.
    if not _SOURCE_ACTOR_ID.search(source_question_id):
        candidate = source_question_id
    else:
        components = [episode_id, catalog_id or "qa", f"item-{ordinal:04d}"]
        safe = [
            re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-")
            for value in components
        ]
        candidate = "__".join(value for value in safe if value) or f"qa-{ordinal:04d}"
    base = candidate
    suffix = 2
    while candidate in used:
        candidate = f"{base}__{suffix}"
        suffix += 1
    return candidate


def _observation_fields(*sources: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in _OBSERVATION_KEYS:
            if key in source:
                result[key] = copy.deepcopy(source[key])
    return result


def _first_modalities(*sources: Any) -> list[str]:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        value = source.get("required_modalities")
        if not _is_sequence(value):
            continue
        values = [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
        if values:
            return values
    return []


def _qa_records(
    *,
    episode_id: str,
    qa_refs: Sequence[dict[str, Any]],
    media: Mapping[str, Any],
    clock: Mapping[str, Any],
    evidence_refs: Sequence[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    source_seen: set[str] = set()
    public_seen: set[str] = set()
    ordinal = 0
    for qa_ref in qa_refs:
        path = Path(str(qa_ref["path"]))
        for row, coverage in _qa_source_rows(path, owner=f"QA source {path}"):
            ordinal += 1
            evaluation = row.get("evaluation")
            if not isinstance(evaluation, Mapping):
                evaluation = row
            source_question_id, _ = _question_id(
                row, evaluation=evaluation, episode_id=episode_id
            )
            if source_question_id in source_seen:
                raise _fail(
                    f"Episode {episode_id} repeats question ID {source_question_id!r}"
                )
            source_seen.add(source_question_id)
            status = _qa_status(evaluation, row)
            question = _qa_question_text(evaluation, row)
            if question is None and status != "not_run":
                raise _fail(f"QA item {source_question_id!r} has no question")
            options = _qa_options(evaluation, row)
            if (
                question is not None
                and _SOURCE_ACTOR_ID.search(question)
            ) or any(_SOURCE_ACTOR_ID.search(option) for option in options):
                raise _fail(
                    f"QA item {source_question_id!r} exposes an internal actor "
                    "identifier in model-facing question text"
                )
            answer = _qa_answer(evaluation, row)
            catalog_id = (
                row.get("catalog_id")
                or evaluation.get("catalog_id")
                or row.get("qa_id")
            )
            question_type = (
                evaluation.get("question_type")
                or row.get("question_type")
                or row.get("card")
                or row.get("qa_id")
            )
            public_question_id = _public_question_id(
                source_question_id,
                episode_id=episode_id,
                catalog_id=catalog_id,
                ordinal=ordinal,
                used=public_seen,
            )
            public_seen.add(public_question_id)
            evidence = evaluation.get("evidence", row.get("evidence", {}))
            scoring = evaluation.get("scoring", row.get("scoring", {}))
            forms = evaluation.get("forms", row.get("forms"))
            if not isinstance(forms, Mapping):
                forms = {}
            form_status = evaluation.get("form_status", row.get("form_status"))
            if not isinstance(form_status, Mapping):
                form_status = {}
            model_input = evaluation.get("model_input", row.get("model_input"))
            if not isinstance(model_input, Mapping):
                model_input = {}
            _assert_public_model_input(
                model_input, owner=f"QA item {source_question_id!r} model_input"
            )
            requirements = evaluation.get("requirements", row.get("requirements"))
            if not isinstance(requirements, Mapping) and isinstance(coverage, Mapping):
                requirements = coverage.get("requirements")
                if not isinstance(requirements, Mapping):
                    requirements = coverage.get("evidence_requirements")
            if not isinstance(requirements, Mapping):
                requirements = {}
            modality_sources = (
                evaluation,
                row,
                coverage,
                requirements,
                coverage.get("evidence_requirements")
                if isinstance(coverage, Mapping)
                else None,
            )
            raw_modalities = None
            for source in modality_sources:
                if isinstance(source, Mapping) and "required_modalities" in source:
                    raw_modalities = source["required_modalities"]
                    break
            required_modalities = (
                [] if raw_modalities is None else raw_modalities
            )
            if not _is_sequence(required_modalities) or any(
                not isinstance(value, str) or not value.strip()
                for value in required_modalities
            ):
                raise _fail(
                    f"QA item {source_question_id!r} has invalid required_modalities"
                )
            required_modalities = [
                value.strip() for value in required_modalities
            ]
            truth_meta = row.get("truth")
            if not isinstance(truth_meta, Mapping):
                truth_meta = evaluation.get("truth")
            observation = _observation_fields(
                row, evaluation, evidence, forms, requirements, coverage
            )
            if status == "pass":
                if question is None:
                    raise _fail(f"QA item {source_question_id!r} has no question")
                input_record = {
                    "question_id": public_question_id,
                    "episode_id": episode_id,
                    "pair_id": public_question_id,
                    "scene_id": episode_id,
                    "task_name": question_type,
                    "catalog_id": catalog_id,
                    "question_type": question_type,
                    "question": question,
                    "question_en": question,
                    "prompt": question,
                    "options": options,
                    "model_input": copy.deepcopy(dict(model_input)),
                    "input": {
                        "video_path": media["video_master"]["path"],
                        "audio_path": media["stereo_wav"]["path"],
                        "audio_layout": "stereo",
                        "audio_channel_count": 2,
                        "media_clock": dict(clock),
                    },
                    "video_path": media["video_master"]["path"],
                    "audio_path": media["stereo_wav"]["path"],
                }
                public.append(input_record)
            private.append(
                {
                    "question_id": public_question_id,
                    "source_question_id": source_question_id,
                    "episode_id": episode_id,
                    "catalog_id": catalog_id,
                    "question_type": question_type,
                    "status": status,
                    "truth": answer,
                    "truth_label": (
                        truth_meta.get("label")
                        if isinstance(truth_meta, Mapping)
                        else None
                    ),
                    "truth_mcq_value": (
                        truth_meta.get("mcq_value")
                        if isinstance(truth_meta, Mapping)
                        else None
                    ),
                    "truth_source": (
                        truth_meta.get("source")
                        if isinstance(truth_meta, Mapping)
                        else None
                    ),
                    "answer_type": (
                        truth_meta.get("answer_type")
                        if isinstance(truth_meta, Mapping)
                        else None
                    ),
                    "question": question,
                    "options": options,
                    "forms": copy.deepcopy(dict(forms)),
                    "form_status": copy.deepcopy(dict(form_status)),
                    "model_input": copy.deepcopy(dict(model_input)),
                    "reason": evaluation.get("reason", row.get("reason")),
                    "required_modalities": required_modalities,
                    "requirements": copy.deepcopy(dict(requirements)),
                    "coverage": (
                        copy.deepcopy(dict(coverage))
                        if isinstance(coverage, Mapping)
                        else None
                    ),
                    "observation": observation,
                    "scoring": (
                        copy.deepcopy(dict(scoring))
                        if isinstance(scoring, Mapping)
                        else scoring
                    ),
                    "evidence": (
                        copy.deepcopy(dict(evidence))
                        if isinstance(evidence, Mapping)
                        else evidence
                    ),
                    "evidence_resources": [dict(ref) for ref in evidence_refs],
                    "source": qa_ref,
                }
            )
    valid = {
        row["question_id"]
        for row in private
        if row["status"] == "pass" and row["truth"] is not None
    }
    return public, private, len(valid)

def _cache_refs(
    value: Any,
    *,
    base_dir: Path,
    owner: str,
) -> tuple[list[dict[str, Any]], int, int]:
    if value is None:
        return [], 0, 0
    entries = value if _is_sequence(value) else [value]
    refs: list[dict[str, Any]] = []
    current = 0
    peak = 0
    seen: set[tuple[int, int]] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise _fail(f"{owner}[{index}] must be an object")
        role = raw.get("role")
        if role != "numeric_rir":
            raise _fail(
                f"{owner}[{index}] must explicitly use role=numeric_rir; "
                "geometry RIR files are not numeric caches"
            )
        paths = raw.get("paths", raw.get("path"))
        path_entries = _ref_entries(
            paths,
            base_dir=base_dir,
            owner=f"{owner}[{index}].paths",
            default_role="numeric_rir",
            include_hash=False,
        )
        if not path_entries:
            raise _fail(f"{owner}[{index}] has no cache paths")
        local_current = 0
        entry_inodes: set[tuple[int, int]] = set()
        for record, path, _ in path_entries:
            if path.suffix.lower() in {".obj", ".glb", ".gltf"}:
                raise _fail(
                    f"{owner}[{index}] appears to name geometry, not numeric RIR: {path}"
                )
            refs.append({**record, "cache_role": "numeric_rir"})
            try:
                stat = path.stat()
            except OSError as exc:
                raise _fail(f"cannot stat numeric RIR cache {path}: {exc}") from exc
            inode = (int(stat.st_dev), int(stat.st_ino))
            if inode not in seen:
                current += int(stat.st_size)
                seen.add(inode)
            if inode not in entry_inodes:
                local_current += int(stat.st_size)
                entry_inodes.add(inode)
        declared_peak = raw.get("peak_bytes", local_current)
        peak_value = _positive_int(
            declared_peak, owner=f"{owner}[{index}].peak_bytes"
        )
        peak = max(peak, peak_value, local_current)
    return refs, current, peak


def _record_payload_files(record: Mapping[str, Any]) -> list[Path]:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise _fail("reference record has no path")
    path = Path(raw).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"reference payload is missing: {path}: {exc}") from exc
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise _fail(f"reference payload is neither file nor directory: {resolved}")
    files: list[Path] = []
    for child in resolved.rglob("*"):
        try:
            candidate = child.resolve(strict=True)
        except OSError as exc:
            raise _fail(f"reference payload child is missing: {child}: {exc}") from exc
        if candidate.is_file():
            files.append(candidate)
    return files


def _reference_inode_keys(records: Sequence[Mapping[str, Any]]) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for record in records:
        for path in _record_payload_files(record):
            try:
                stat = path.stat()
            except OSError as exc:
                raise _fail(f"cannot stat reference payload {path}: {exc}") from exc
            keys.add((int(stat.st_dev), int(stat.st_ino)))
    return keys


def _reference_size(
    records: Sequence[Mapping[str, Any]],
    *,
    exclude_inodes: Sequence[tuple[int, int]] = (),
) -> int:
    # Count physical payload once. This handles repeated paths, directory/file
    # overlap, symlinks, and hardlinks without charging the same bytes twice.
    seen: set[tuple[int, int]] = set(exclude_inodes)
    total = 0
    for record in records:
        for path in _record_payload_files(record):
            try:
                stat = path.stat()
            except OSError as exc:
                raise _fail(f"cannot stat reference payload {path}: {exc}") from exc
            key = (int(stat.st_dev), int(stat.st_ino))
            if key in seen:
                continue
            seen.add(key)
            total += int(stat.st_size)
    return total



def _inode_size(paths: Sequence[Path]) -> tuple[int, set[tuple[int, int]]]:
    total = 0
    keys: set[tuple[int, int]] = set()
    for path in paths:
        try:
            stat = path.stat()
        except OSError as exc:
            raise _fail(f"cannot stat lifecycle payload {path}: {exc}") from exc
        key = (int(stat.st_dev), int(stat.st_ino))
        if key in keys:
            continue
        keys.add(key)
        total += int(stat.st_size)
    return total, keys


def _resolve_optional_lifecycle_path(
    value: Any, *, base_dir: Path, owner: str
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        return path.resolve(strict=False)
    except OSError as exc:
        raise _fail(f"cannot resolve lifecycle path {path}: {exc}") from exc


def _cache_lifecycle_records(
    value: Any,
    *,
    base_dir: Path,
    owner: str,
) -> dict[str, Any]:
    if value is None:
        return {
            "status": "not_run",
            "records": [],
            "cache_peak_bytes": 0,
            "numeric_payload_bytes_cleared": 0,
            "numeric_payload_bytes_current": 0,
            "metadata_bytes_current": 0,
            "cache_bytes_current": 0,
            "lifecycle_record_bytes": 0,
            "retained_failure_evidence_bytes": 0,
        }
    entries = value if _is_sequence(value) else [value]
    record_refs: list[dict[str, Any]] = []
    payloads: list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    seen_record_paths: set[str] = set()
    global_peak: int | None = None
    global_cleared: int | None = None
    for index, raw in enumerate(entries):
        if isinstance(raw, Mapping):
            raw_path = raw.get("path")
            metadata = raw
        else:
            raw_path = raw
            metadata = {}
        path = _resolve_optional_lifecycle_path(
            raw_path, base_dir=base_dir, owner=f"{owner}[{index}]"
        )
        if path is None or not path.is_file():
            raise _fail(f"{owner}[{index}] lifecycle record is missing: {raw_path}")
        record_ref = _path_record(path, role=str(metadata.get("role", "cache_lifecycle")))
        record_refs.append(record_ref)
        key = str(path)
        if key in seen_record_paths:
            continue
        seen_record_paths.add(key)
        payload = _read_json(path, owner=f"{owner}[{index}]")
        if not isinstance(payload, Mapping):
            raise _fail(f"{owner}[{index}] lifecycle JSON must contain an object")
        # A batch cleanup summary points at the individual cleanup records.
        subrecords = payload.get("records")
        if isinstance(subrecords, list) and not payload.get("cleanup_record"):
            raw_peak = payload.get("cache_peak_bytes")
            if isinstance(raw_peak, int) and not isinstance(raw_peak, bool):
                global_peak = (
                    int(raw_peak)
                    if global_peak is None
                    else max(global_peak, int(raw_peak))
                )
            raw_cleared = payload.get("cleared_numeric_payload_bytes")
            if isinstance(raw_cleared, int) and not isinstance(raw_cleared, bool):
                global_cleared = (
                    int(raw_cleared)
                    if global_cleared is None
                    else max(global_cleared, int(raw_cleared))
                )
            failed_roots = payload.get("failed_capture_roots_preserved", [])
            if not _is_sequence(failed_roots):
                failed_roots = []
            for subindex, subrecord in enumerate(subrecords):
                subpath = _resolve_optional_lifecycle_path(
                    subrecord,
                    base_dir=base_dir,
                    owner=f"{owner}[{index}].records[{subindex}]",
                )
                if subpath is None or not subpath.is_file():
                    raise _fail(
                        f"{owner}[{index}] records[{subindex}] is missing: {subrecord}"
                    )
                subkey = str(subpath)
                if subkey in seen_record_paths:
                    continue
                seen_record_paths.add(subkey)
                subpayload = _read_json(subpath, owner=f"{owner}[{index}].records[{subindex}]")
                if not isinstance(subpayload, Mapping):
                    raise _fail(
                        f"{owner}[{index}] records[{subindex}] must contain an object"
                    )
                submetadata = {
                    "role": "numeric_rir_lifecycle",
                    "failed_capture_roots_preserved": failed_roots,
                }
                payloads.append((_path_record(subpath, role="cache_lifecycle"), subpayload, submetadata))
        else:
            payloads.append((record_ref, payload, metadata))

    current_numeric_paths: list[Path] = []
    current_metadata_paths: list[Path] = []
    current_cache_paths: list[Path] = []
    lifecycle_rows: list[dict[str, Any]] = []
    peak_values: list[int] = []
    cleared_values: list[int] = []
    retained_failure_bytes = 0
    statuses: list[str] = []
    for ref, payload, metadata in payloads:
        root_raw = payload.get("cache_root") or payload.get("historical_cache_path")
        root = _resolve_optional_lifecycle_path(
            root_raw, base_dir=base_dir, owner=f"{ref['path']}.cache_root"
        )
        root_files: list[Path] = []
        if root is not None and root.is_dir():
            root_files = _record_payload_files({"path": str(root)})
            current_cache_paths.extend(root_files)
        declared_numeric = payload.get("numeric_payload_files", [])
        if not _is_sequence(declared_numeric):
            declared_numeric = []
        numeric_paths: list[Path] = []
        for raw_path in declared_numeric:
            path = _resolve_optional_lifecycle_path(
                raw_path, base_dir=base_dir, owner=f"{ref['path']}.numeric_payload_files"
            )
            # Cleaned payloads are allowed to be absent. Existing paths are
            # counted as numeric payload only when they still exist as files.
            if path is not None and path.is_file():
                numeric_paths.append(path)
        current_numeric_paths.extend(numeric_paths)
        numeric_keys = {
            (int(path.stat().st_dev), int(path.stat().st_ino))
            for path in numeric_paths
        }
        metadata_paths = []
        for path in root_files:
            stat = path.stat()
            if (int(stat.st_dev), int(stat.st_ino)) not in numeric_keys:
                metadata_paths.append(path)
        current_metadata_paths.extend(metadata_paths)
        storage = payload.get("storage")
        if not isinstance(storage, Mapping):
            storage = {}
        peak_raw = storage.get("peak_bytes", payload.get("peak_bytes", 0))
        cleared_raw = storage.get(
            "numeric_payload_bytes_cleared",
            payload.get("payload_bytes_cleared", 0),
        )
        before_raw = storage.get(
            "numeric_payload_bytes_before",
            payload.get("numeric_payload_bytes_before", 0),
        )
        peak = _positive_int(peak_raw, owner=f"{ref['path']}.peak_bytes") if peak_raw else 0
        cleared = _positive_int(cleared_raw, owner=f"{ref['path']}.cleared_bytes") if cleared_raw else 0
        before = _positive_int(before_raw, owner=f"{ref['path']}.before_bytes") if before_raw else 0
        peak_values.append(peak)
        cleared_values.append(cleared)
        lifecycle = str(payload.get("numeric_payload_lifecycle") or "")
        status = str(payload.get("status") or "not_run")
        failed_roots = metadata.get("failed_capture_roots_preserved", [])
        retained_failure = bool(metadata.get("retained_failure_evidence", False))
        if lifecycle in {"planned", "retained_failure_evidence"}:
            retained_failure = True
        if retained_failure:
            lifecycle = "retained_failure_evidence"
        elif lifecycle:
            lifecycle = lifecycle
        elif status == "pass":
            lifecycle = "numeric_payload_cleaned"
        else:
            lifecycle = "not_run"
        statuses.append(status)
        current_cache_size, _ = _inode_size(root_files) if root_files else (0, set())
        current_numeric_size, _ = _inode_size(numeric_paths)
        current_metadata_size = max(0, current_cache_size - current_numeric_size)
        if retained_failure:
            retained_failure_bytes += current_cache_size
        lifecycle_rows.append(
            {
                "cleanup_record": ref,
                "cache_root": str(root) if root is not None else root_raw,
                "status": status,
                "numeric_payload_lifecycle": lifecycle,
                "peak_bytes": peak,
                "numeric_payload_bytes_before": before,
                "numeric_payload_bytes_cleared": cleared,
                "current_cache_bytes": current_cache_size,
                "current_numeric_payload_bytes": current_numeric_size,
                "current_metadata_bytes": current_metadata_size,
                "numeric_payload_files_declared": len(declared_numeric),
                "numeric_payload_files_present": len(numeric_paths),
                "retained_failure_evidence": retained_failure,
                "failed_capture_roots_preserved": list(failed_roots) if _is_sequence(failed_roots) else [],
            }
        )
    _, numeric_inodes = _inode_size(current_numeric_paths)
    _, metadata_inodes = _inode_size(current_metadata_paths)
    current_cache_size, _ = _inode_size(current_cache_paths)
    if global_peak is None:
        aggregate_peak = sum(peak_values)
    else:
        aggregate_peak = global_peak + sum(
            row["peak_bytes"]
            for row in lifecycle_rows
            if row["retained_failure_evidence"]
        )
    if global_cleared is None:
        aggregate_cleared = sum(cleared_values)
    else:
        aggregate_cleared = global_cleared
    # Keep the numeric and metadata buckets disjoint even when a cleanup
    # summary repeats a cache root.
    numeric_current = 0
    for path in current_numeric_paths:
        stat = path.stat()
        key = (int(stat.st_dev), int(stat.st_ino))
        if key in numeric_inodes:
            numeric_current += int(stat.st_size)
            numeric_inodes.remove(key)
    metadata_current = 0
    for path in current_metadata_paths:
        stat = path.stat()
        key = (int(stat.st_dev), int(stat.st_ino))
        if key in metadata_inodes:
            metadata_current += int(stat.st_size)
            metadata_inodes.remove(key)
    lifecycle_record_bytes = _reference_size(record_refs)
    overall = "pass" if lifecycle_rows and all(
        status in {"pass", "planned", "not_run"} for status in statuses
    ) else "not_run"
    if not lifecycle_rows:
        overall = "not_run"
    return {
        "status": overall,
        "records": lifecycle_rows,
        "cache_peak_bytes": aggregate_peak,
        "numeric_payload_bytes_cleared": aggregate_cleared,
        "numeric_payload_bytes_current": numeric_current,
        "metadata_bytes_current": metadata_current,
        "cache_bytes_current": current_cache_size,
        "lifecycle_record_bytes": lifecycle_record_bytes,
        "retained_failure_evidence_bytes": retained_failure_bytes,
        "numeric_payload_paths_current": [str(path) for path in current_numeric_paths],
        "numeric_payload_lifecycle": sorted(
            {row["numeric_payload_lifecycle"] for row in lifecycle_rows}
        ),
    }


def _episode_specs(room: Mapping[str, Any], *, owner: str) -> list[Mapping[str, Any]]:
    episodes = room.get("episodes")
    if isinstance(episodes, Mapping):
        episodes = [episodes]
    if not _is_sequence(episodes) or not episodes:
        raise _fail(f"{owner}.episodes must be a non-empty list")
    result = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise _fail(f"{owner}.episodes[{index}] must be an object")
        result.append(episode)
    return result


def _build_records(
    request: Mapping[str, Any],
    *,
    request_dir: Path,
    video_probe: Callable[[Path], Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if request.get("schema") != EPISODE_EXPORT_REQUEST_SCHEMA:
        raise _fail(f"request schema must be {EPISODE_EXPORT_REQUEST_SCHEMA}")
    retention = _normalise_retention(request.get("evidence_retention"))
    rooms = request.get("rooms")
    if not _is_sequence(rooms) or not rooms:
        raise _fail("request.rooms must be a non-empty list")
    room_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    model_specs = request.get("model_evaluations")
    seen_rooms: set[str] = set()
    seen_episodes: set[str] = set()
    seen_questions: set[str] = set()
    all_shared_records: list[dict[str, Any]] = []
    all_permanent_records: list[dict[str, Any]] = []
    all_cache_records: list[dict[str, Any]] = []
    cache_current = 0
    cache_peak = 0
    omitted_extended_bytes = 0
    cache_lifecycle_value = request.get(
        "cache_lifecycle_records", request.get("cache_lifecycle")
    )
    for room_index, room in enumerate(rooms):
        if not isinstance(room, Mapping):
            raise _fail(f"rooms[{room_index}] must be an object")
        room_id = _require_text(
            room.get("room_id"), owner=f"rooms[{room_index}].room_id"
        )
        if room_id in seen_rooms:
            raise _fail(f"duplicate room_id {room_id!r}")
        seen_rooms.add(room_id)
        room_family_id = str(room.get("room_family_id") or room_id)
        shared_value = room.get(
            "shared_resources", room.get("shared", room.get("resources"))
        )
        shared_entries = _ref_entries(
            shared_value,
            base_dir=request_dir,
            owner=f"room {room_id} shared_resources",
            default_role="room_shared",
        )
        if not shared_entries:
            raise _fail(f"room {room_id} has no shared room references")
        shared_refs = [entry[0] for entry in shared_entries]
        all_shared_records.extend(shared_refs)
        room_rows.append(
            {
                "room_id": room_id,
                "room_family_id": room_family_id,
                "status": str(room.get("status") or "research_only"),
                "qualification_claim": bool(room.get("qualification_claim", False)),
                "shared_resources": shared_refs,
            }
        )
        for episode_index, episode in enumerate(
            _episode_specs(room, owner=f"room {room_id}")
        ):
            episode_id = _require_text(
                episode.get("episode_id"),
                owner=f"room {room_id}.episodes[{episode_index}].episode_id",
            )
            if episode_id in seen_episodes:
                raise _fail(f"duplicate episode_id {episode_id!r}")
            seen_episodes.add(episode_id)
            plan, plan_payload = _document_ref(
                episode.get("plan"),
                base_dir=request_dir,
                owner=f"Episode {episode_id} plan",
            )
            actual, actual_payload = _document_ref(
                episode.get("actual"),
                base_dir=request_dir,
                owner=f"Episode {episode_id} actual",
            )
            media_value = episode.get("media")
            if not isinstance(media_value, Mapping):
                raise _fail(f"Episode {episode_id} media must be an object")
            video_value = _media_value(
                media_value,
                ("video_master", "video"),
                owner=f"Episode {episode_id} video_master",
            )
            audio_value = _media_value(
                media_value,
                ("stereo_wav", "binaural_wav", "audio_wav", "audio"),
                owner=f"Episode {episode_id} stereo_wav",
            )
            video_path = _resolve_path(
                video_value, base_dir=request_dir, owner=f"Episode {episode_id} video"
            )
            audio_path = _resolve_path(
                audio_value, base_dir=request_dir, owner=f"Episode {episode_id} audio"
            )
            if not video_path.is_file() or not audio_path.is_file():
                raise _fail(f"Episode {episode_id} media masters must be files")
            video_info = dict(video_probe(video_path))
            for key in ("frame_count", "frame_rate_hz", "duration_seconds"):
                if key not in video_info:
                    raise _fail(f"video probe omitted {key}: {video_path}")
            frame_count = _positive_int(
                video_info["frame_count"], owner="video.frame_count"
            )
            frame_rate = _positive_float(
                video_info["frame_rate_hz"], owner="video.frame_rate_hz"
            )
            duration = _positive_float(
                video_info["duration_seconds"], owner="video.duration_seconds"
            )
            video_record = _path_record(video_path, role="video_master")
            video_record.update(
                {
                    "frame_count": frame_count,
                    "frame_rate_hz": frame_rate,
                    "duration_seconds": duration,
                    "master": True,
                }
            )
            audio_info = _load_wav(audio_path)
            audio_record = _path_record(audio_path, role="stereo_wav")
            audio_record.update({**audio_info, "master": True})
            media = {"video_master": video_record, "stereo_wav": audio_record}
            clock = _declared_clock(
                episode,
                plan=plan_payload,
                actual=actual_payload,
                audio=audio_info,
                video=video_info,
            )
            evidence_entries = _ref_entries(
                episode.get("evidence", episode.get("evidence_resources")),
                base_dir=request_dir,
                owner=f"Episode {episode_id} evidence",
                default_role="episode_evidence",
                include_hash=True,
            )
            included_evidence: list[dict[str, Any]] = []
            omitted_evidence: list[dict[str, Any]] = []
            for record, _path, metadata in evidence_entries:
                retention_kind = (
                    metadata.get("retention", "base")
                    if metadata is not None
                    else "base"
                )
                if retention_kind not in {"base", "extended"}:
                    raise _fail(
                        f"Episode {episode_id} evidence retention must be base or extended"
                    )
                required = metadata.get("required", False) if metadata else False
                if not isinstance(required, bool):
                    raise _fail(
                        f"Episode {episode_id} evidence required flag must be boolean"
                    )
                must_preserve = required and retention["preserve_required"]
                if (
                    retention_kind == "extended"
                    and not retention["include_extended"]
                    and not must_preserve
                ):
                    omitted_evidence.append(
                        {
                            **record,
                            "status": "omitted_by_policy",
                            "retention": "extended",
                            "required": required,
                        }
                    )
                    omitted_extended_bytes += int(record["byte_size"])
                else:
                    included_evidence.append(
                        {
                            **record,
                            "retention": retention_kind,
                            "required": required,
                        }
                    )
            qa_entries = _ref_entries(
                episode.get("qa", episode.get("qa_sources")),
                base_dir=request_dir,
                owner=f"Episode {episode_id} QA",
                default_role="qa_source",
                include_hash=True,
            )
            if not qa_entries:
                raise _fail(f"Episode {episode_id} has no QA source")
            qa_refs = [entry[0] for entry in qa_entries]
            public, private, _ = _qa_records(
                episode_id=episode_id,
                qa_refs=qa_refs,
                media=media,
                clock=clock,
                evidence_refs=included_evidence,
            )
            for row in public:
                if row["question_id"] in seen_questions:
                    raise _fail(f"duplicate question_id {row['question_id']!r}")
                seen_questions.add(row["question_id"])
            question_rows.extend(public)
            answer_rows.extend(private)
            cache_refs, current_bytes, peak_bytes = _cache_refs(
                episode.get("cache", episode.get("caches")),
                base_dir=request_dir,
                owner=f"Episode {episode_id} cache",
            )
            all_cache_records.extend(cache_refs)
            cache_current += current_bytes
            cache_peak = max(cache_peak, peak_bytes)
            episode_permanent = [
                plan["reference"],
                actual["reference"],
                *qa_refs,
                *included_evidence,
                *media.values(),
            ]
            all_permanent_records.extend(episode_permanent)
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "room_id": room_id,
                    "status": str(episode.get("status") or "research_only"),
                    "qualification_claim": bool(
                        episode.get("qualification_claim", False)
                    ),
                    "lineage": {"plan": plan, "actual": actual},
                    "native_readback": {
                        "status": actual.get("status") or "not_run",
                        "readback_verified": actual.get("status") == "pass",
                        "reference": actual["reference"],
                    },
                    "media": media,
                    "media_clock": clock,
                    "qa_question_ids": [row["question_id"] for row in public],
                    "qa_sources": qa_refs,
                    "evidence": {
                        "retention": retention,
                        "included": included_evidence,
                        "omitted": omitted_evidence,
                    },
                    "cache": {
                        "numeric_rir": cache_refs,
                        "policy": "generation_cache_default",
                        "lifecycle_owner": "audio_agent",
                        "status": "generated_cache" if cache_refs else "not_run",
                    },
                }
            )
    cache_lifecycle = _cache_lifecycle_records(
        cache_lifecycle_value,
        base_dir=request_dir,
        owner="request cache_lifecycle_records",
    )
    batch_evidence_entries = _ref_entries(
        request.get("batch_evidence"),
        base_dir=request_dir,
        owner="request batch_evidence",
        default_role="batch_evidence",
        include_hash=False,
    )
    batch_evidence_refs: list[dict[str, Any]] = []
    for record, _path, metadata in batch_evidence_entries:
        required = metadata.get("required", False) if metadata is not None else False
        if not isinstance(required, bool):
            raise _fail("request batch_evidence required flag must be boolean")
        retention_kind = (
            metadata.get("retention", "base")
            if metadata is not None
            else "base"
        )
        if retention_kind not in {"base", "extended"}:
            raise _fail("request batch_evidence retention must be base or extended")
        batch_evidence_refs.append(
            {**record, "required": required, "retention": retention_kind}
        )
    all_permanent_records.extend(batch_evidence_refs)
    valid_ids = {
        row["question_id"]
        for row in answer_rows
        if row["status"] == "pass" and row.get("truth") is not None
    }
    shared_paths = {str(row["path"]) for row in all_shared_records}
    permanent_records = [
        record
        for record in all_permanent_records
        if str(record["path"]) not in shared_paths
    ]
    shared_bytes = _reference_size(all_shared_records)
    shared_inodes = _reference_inode_keys(all_shared_records)
    permanent_bytes = _reference_size(
        permanent_records, exclude_inodes=tuple(sorted(shared_inodes))
    )
    lifecycle_cache_records = [
        {"path": path}
        for path in cache_lifecycle.get("numeric_payload_paths_current", [])
    ]
    cache_bytes = _reference_size([*all_cache_records, *lifecycle_cache_records])
    cache_current = cache_bytes
    cache_peak = max(cache_peak, int(cache_lifecycle["cache_peak_bytes"]))
    total_persistent = permanent_bytes + shared_bytes
    storage = {
        "schema": "avengine_episode_storage_stats_v1",
        "status": "pass",
        "permanent_bytes": permanent_bytes,
        "shared_room_bytes": shared_bytes,
        "cache_bytes": cache_current,
        "cache_peak_bytes": max(cache_peak, cache_current),
        "numeric_rir_cache_policy": "generation_cache_default",
        "numeric_rir_cache_status": (
            "generated_cache"
            if all_cache_records or cache_lifecycle["records"]
            else "not_run"
        ),
        "cache_lifecycle": cache_lifecycle,
        "cache_lifecycle_status": cache_lifecycle["status"],
        "cache_metadata_bytes": cache_lifecycle["metadata_bytes_current"],
        "cache_numeric_payload_bytes_current": cache_lifecycle[
            "numeric_payload_bytes_current"
        ],
        "cache_numeric_payload_bytes_cleared": cache_lifecycle[
            "numeric_payload_bytes_cleared"
        ],
        "cache_lifecycle_record_bytes": cache_lifecycle["lifecycle_record_bytes"],
        "retained_failure_evidence_bytes": cache_lifecycle[
            "retained_failure_evidence_bytes"
        ],
        "cache_lifecycle_owner": "audio_agent",
        "omitted_extended_evidence_bytes": omitted_extended_bytes,
        "persistent_bytes_including_shared_room": total_persistent,
        "verified_unique_question_count": len(valid_ids),
        "amortized_bytes_per_verified_unique_question": (
            total_persistent / len(valid_ids) if valid_ids else None
        ),
        "amortized_permanent_bytes_per_verified_unique_question": (
            permanent_bytes / len(valid_ids) if valid_ids else None
        ),
        "question_count_basis": (
            "unique question IDs whose QA evaluation status is pass and truth is present"
        ),
        "source_paths_are_deduplicated": True,
        "payload_deduplication": "resolved_file_inode_st_dev_st_ino",
        "directory_file_overlap_counted_once": True,
        "hardlink_payload_counted_once": True,
    }
    model_rows = _model_rows(
        model_specs,
        episode_rows=episode_rows,
        question_rows=question_rows,
        answer_rows=answer_rows,
        request_dir=request_dir,
    )
    catalog_counts = dict(
        sorted(
            Counter(
                str(row["catalog_id"])
                for row in answer_rows
                if isinstance(row.get("catalog_id"), str)
                and row["catalog_id"].strip()
            ).items()
        )
    )
    catalog_status_counts = {}
    for row in answer_rows:
        catalog_id = row.get("catalog_id")
        if not isinstance(catalog_id, str) or not catalog_id.strip():
            continue
        status = str(row.get("status") or "not_run")
        by_status = catalog_status_counts.setdefault(catalog_id, {})
        by_status[status] = by_status.get(status, 0) + 1
    catalog_status_counts = {
        key: dict(sorted(value.items()))
        for key, value in sorted(catalog_status_counts.items())
    }
    return room_rows, episode_rows, question_rows, answer_rows, {
        "storage": storage,
        "model_rows": model_rows,
        "retention": retention,
        "question_count": len(question_rows),
        "valid_unique_question_count": len(valid_ids),
        "catalog_counts": catalog_counts,
        "catalog_status_counts": catalog_status_counts,
        "cache_lifecycle": cache_lifecycle,
        "batch_evidence": batch_evidence_refs,
    }


def build_episode_export_records(
    request: Mapping[str, Any],
    *,
    request_dir: str | Path | None = None,
    video_probe_fn: Callable[[Path], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Build validated reference records without publishing an output tree.

    This is the integration seam for a producer such as the Studio QA
    Episode runner.  It is useful when that producer already has an in-memory
    request or when a test supplies a bounded video probe.
    """

    base_dir = (
        Path.cwd()
        if request_dir is None
        else Path(request_dir).expanduser().resolve()
    )
    if not base_dir.is_dir():
        raise _fail(f"request_dir is not a directory: {base_dir}")
    if not isinstance(request, Mapping):
        raise _fail("episode export request must contain an object")
    rooms, episodes, questions, answers, extra = _build_records(
        request,
        request_dir=base_dir,
        video_probe=video_probe_fn,
    )
    return {
        "rooms": rooms,
        "episodes": episodes,
        "model_inputs": questions,
        "answers": answers,
        "model_results": extra["model_rows"],
        "storage": extra["storage"],
        "retention": extra["retention"],
        "cache_lifecycle": extra["cache_lifecycle"],
        "batch_evidence": extra["batch_evidence"],
        "qa_coverage": {
            "catalog_status_counts": extra["catalog_status_counts"],
            "basis": "all emitted private QA rows, including deferred not_run rows",
        },
        "counts": {
            "rooms": len(rooms),
            "episodes": len(episodes),
            "questions": len(questions),
            "answers": len(answers),
            "verified_unique_questions": extra["valid_unique_question_count"],
            "catalog_ids": extra["catalog_counts"],
        },
    }


def _model_payload(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
        rows: list[Mapping[str, Any]] = []
        try:
            if path.suffix == ".gz":
                handle = gzip.open(path, "rt", encoding="utf-8")
            else:
                handle = path.open("r", encoding="utf-8")
            with handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if isinstance(row, Mapping):
                            rows.append(row)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _fail(f"cannot read model result JSONL {path}: {exc}") from exc
        return rows
    payload = _read_json(path, owner="model results")
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("records", "predictions", "results"):
            if _is_sequence(payload.get(key)):
                return [row for row in payload[key] if isinstance(row, Mapping)]
    raise _fail(f"model result file has no records list: {path}")


def _model_rows(
    specs: Any,
    *,
    episode_rows: Sequence[Mapping[str, Any]],
    question_rows: Sequence[Mapping[str, Any]],
    answer_rows: Sequence[Mapping[str, Any]],
    request_dir: Path,
) -> list[dict[str, Any]]:
    if specs is None:
        specs = [
            {
                "model_id": "unspecified",
                "status": "not_run",
                "reason": "no model evaluation supplied",
            }
        ]
    elif isinstance(specs, Mapping):
        specs = [specs]
    if not _is_sequence(specs):
        raise _fail("model_evaluations must be a list or object")
    questions_by_id = {str(row["question_id"]): row for row in question_rows}
    question_aliases = {
        str(answer["source_question_id"]): str(answer["question_id"])
        for answer in answer_rows
        if isinstance(answer.get("source_question_id"), str)
        and isinstance(answer.get("question_id"), str)
        and answer["source_question_id"]
        and answer["source_question_id"] != answer["question_id"]
    }
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(specs):
        if not isinstance(raw, Mapping):
            raise _fail(f"model_evaluations[{index}] must be an object")
        model_id = str(raw.get("model_id") or raw.get("model_name") or "unspecified")
        request_config = _safe_model_value(
            {
                key: value
                for key, value in raw.items()
                if key not in {"results", "results_path", "path"}
            }
        )
        result_rows: list[Mapping[str, Any]] = []
        result_ref = None
        result_value = raw.get("results")
        if result_value is not None:
            if not _is_sequence(result_value):
                raise _fail(f"model_evaluations[{index}].results must be a list")
            result_rows = [row for row in result_value if isinstance(row, Mapping)]
        else:
            result_path_value = raw.get("results_path", raw.get("path"))
            if result_path_value is not None:
                result_path = _resolve_path(
                    result_path_value,
                    base_dir=request_dir,
                    owner=f"model_evaluations[{index}].results_path",
                )
                result_ref = _path_record(result_path, role="model_results")
                result_rows = _model_payload(result_path)
        by_key: dict[tuple[str, str | None], Mapping[str, Any]] = {}
        by_question: dict[str, list[Mapping[str, Any]]] = {}
        for result in result_rows:
            rid = result.get("question_id") or result.get("qa_id") or result.get("pair_id")
            if isinstance(rid, str) and rid:
                rid = question_aliases.get(rid, rid)
                raw_condition = result.get("condition", result.get("setting"))
                condition = (
                    str(raw_condition).strip()
                    if isinstance(raw_condition, str) and raw_condition.strip()
                    else None
                )
                key = (rid, condition)
                if key in by_key:
                    raise _fail(
                        f"model results repeat question/condition {rid!r}/{condition!r}"
                    )
                by_key[key] = result
                by_question.setdefault(rid, []).append(result)
        declared_conditions = raw.get("conditions")
        if declared_conditions is not None:
            if not _is_sequence(declared_conditions) or not declared_conditions:
                raise _fail(
                    f"model_evaluations[{index}].conditions must be a non-empty list"
                )
            conditions = [
                str(value).strip()
                for value in declared_conditions
                if isinstance(value, str) and value.strip()
            ]
            if len(conditions) != len(declared_conditions) or len(
                set(conditions)
            ) != len(conditions):
                raise _fail(
                    f"model_evaluations[{index}].conditions must be unique strings"
                )
        else:
            observed_conditions = sorted(
                {
                    condition
                    for _, condition in by_key
                    if condition is not None
                }
            )
            conditions = observed_conditions or [None]
        requested_status = raw.get("status")
        for question_id, question in questions_by_id.items():
            candidates = conditions
            if conditions == [None] and question_id in by_question:
                candidates = [None]
            for condition in candidates:
                result = by_key.get((question_id, condition))
                if result is None and condition is None:
                    only = by_question.get(question_id, [])
                    if len(only) == 1:
                        result = only[0]
                output: dict[str, Any]
                if result is None:
                    status = (
                        "not_run"
                        if requested_status in (None, "not_run")
                        else str(requested_status)
                    )
                    reason = raw.get("reason") or (
                        "no result row for this question/condition"
                        if result_rows
                        else "model evaluation was not run"
                    )
                    output = {
                        "model_id": model_id,
                        "question_id": question_id,
                        "episode_id": question["episode_id"],
                        "status": status,
                        "reason": str(reason),
                        "raw_answer": None,
                        "score": None,
                        "request": request_config,
                    }
                else:
                    raw_answer = _model_answer_value(
                        result.get(
                            "raw_answer",
                            result.get(
                                "prediction",
                                result.get("model_answer", result.get("answer")),
                            ),
                        )
                    )
                    score = result.get("score")
                    if score is None:
                        score = result.get(
                            "cleaned_exact_match", result.get("exact_match")
                        )
                    status = result.get("status")
                    if status is None:
                        status = "scored" if score is not None else "answered"
                    output = {
                        "model_id": model_id,
                        "question_id": question_id,
                        "episode_id": question["episode_id"],
                        "status": str(status),
                        "reason": result.get("reason"),
                        "raw_answer": raw_answer,
                        "score": score,
                        "request": request_config,
                        "result_fields": _model_scalar_fields(result),
                    }
                if condition is not None:
                    output["condition"] = condition
                if result_ref is not None:
                    output["result_source"] = result_ref
                rows.append(output)
    if not rows:
        rows.append(
            {
                "model_id": "unspecified",
                "question_id": None,
                "episode_id": None,
                "status": "not_run",
                "reason": "no model evaluation was supplied",
                "raw_answer": None,
                "score": None,
                "request": {},
            }
        )
    return rows


def _write_json(path: Path, value: Any, *, gzip_output: bool) -> None:
    if gzip_output:
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        return
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, gzip_output: bool
) -> None:
    if gzip_output:
        handle = gzip.open(path, "wt", encoding="utf-8", newline="\n")
    else:
        handle = path.open("w", encoding="utf-8", newline="\n")
    with handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")


def _safe_publish(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise _fail(f"refusing to overwrite output root: {output}")
    try:
        os.rename(staging, output)
    except FileExistsError as exc:
        raise _fail(f"refusing to overwrite output root: {output}") from exc


def export_episode_bundle(
    *,
    request_path: str | Path,
    output_root: str | Path,
    gzip_json: bool = False,
    ffprobe: str = "ffprobe",
    video_probe_fn: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate and publish a shared-room Episode export.

    The output directory is published from a sibling temporary directory and
    is never replaced. gzip_json compresses JSON and JSONL sidecars; the
    top-level manifest remains plain JSON so a reviewer can discover the
    bundle without a decompressor.
    """

    request = _resolve_path(request_path, base_dir=Path.cwd(), owner="request")
    if not request.is_file():
        raise _fail(f"request must be a file: {request}")
    raw = _read_json(request, owner="episode export request")
    if not isinstance(raw, Mapping):
        raise _fail("episode export request must contain an object")
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise _fail(f"refusing to overwrite output root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".gz" if gzip_json else ""
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    try:
        records = build_episode_export_records(
            raw,
            request_dir=request.parent,
            video_probe_fn=(
                video_probe_fn
                if video_probe_fn is not None
                else lambda path: probe_video(path, ffprobe=ffprobe)
            ),
        )
        rooms = records["rooms"]
        episodes = records["episodes"]
        questions = records["model_inputs"]
        answers = records["answers"]
        extra = {
            "storage": records["storage"],
            "model_rows": records["model_results"],
            "retention": records["retention"],
            "valid_unique_question_count": records["counts"][
                "verified_unique_questions"
            ],
            "catalog_counts": records["counts"]["catalog_ids"],
            "catalog_status_counts": records["qa_coverage"]["catalog_status_counts"],
            "batch_evidence": records["batch_evidence"],
        }
        (temp / "qa").mkdir(parents=True)
        (temp / "evaluation").mkdir(parents=True)
        _write_jsonl(temp / f"rooms.jsonl{suffix}", rooms, gzip_output=gzip_json)
        _write_jsonl(temp / f"episodes.jsonl{suffix}", episodes, gzip_output=gzip_json)
        _write_jsonl(
            temp / f"qa/questions.jsonl{suffix}",
            questions,
            gzip_output=gzip_json,
        )
        _write_jsonl(
            temp / f"qa/answers.jsonl{suffix}",
            answers,
            gzip_output=gzip_json,
        )
        _write_jsonl(
            temp / f"evaluation/model_results.jsonl{suffix}",
            extra["model_rows"],
            gzip_output=gzip_json,
        )
        _write_json(
            temp / f"storage.json{suffix}",
            extra["storage"],
            gzip_output=gzip_json,
        )
        manifest = {
            "schema": EPISODE_EXPORT_SCHEMA,
            "status": "research_only",
            "qualification_claim": False,
            "lineage": {
                "request": str(request),
                "request_schema": raw.get("schema"),
                "evidence_retention": extra["retention"],
            },
            "counts": {
                "rooms": len(rooms),
                "episodes": len(episodes),
                "questions": len(questions),
                "answers": len(answers),
                "verified_unique_questions": extra["valid_unique_question_count"],
                "catalog_ids": extra["catalog_counts"],
                "models": len({row["model_id"] for row in extra["model_rows"]}),
            },
            "batch_evidence": extra["batch_evidence"],
            "qa_coverage": {
                "catalog_status_counts": extra["catalog_status_counts"],
                "basis": "all emitted private QA rows, including deferred not_run rows",
            },
            "outputs": {
                "rooms": f"rooms.jsonl{suffix}",
                "episodes": f"episodes.jsonl{suffix}",
                "model_inputs": f"qa/questions.jsonl{suffix}",
                "answers": f"qa/answers.jsonl{suffix}",
                "model_results": f"evaluation/model_results.jsonl{suffix}",
                "storage": f"storage.json{suffix}",
            },
            "storage": extra["storage"],
            "model_evaluation": {
                "status": (
                    "pass"
                    if any(
                        row["status"] in {"scored", "answered"}
                        for row in extra["model_rows"]
                    )
                    else "not_run"
                ),
                "results_path": f"evaluation/model_results.jsonl{suffix}",
            },
        }
        _write_json(temp / "manifest.json", manifest, gzip_output=False)
        _safe_publish(temp, output)
    except Exception:
        # Keep the caller's output root untouched. The temporary sibling is
        # deliberately left for failure diagnosis rather than deleting a
        # potentially useful failed-evidence tree.
        raise
    return manifest


__all__ = [
    "EPISODE_EXPORT_REQUEST_SCHEMA",
    "EPISODE_EXPORT_SCHEMA",
    "EpisodeExportError",
    "build_episode_export_records",
    "export_episode_bundle",
    "probe_video",
]
