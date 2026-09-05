"""Resolve declarative QA runtime artifacts for one candidate.

The design producers deliberately keep runtime execution data next to the
candidate fact rather than teaching the pipeline about every profile.  This
module is the small compatibility boundary between those descriptions and the
fixed QA runners.  It accepts the current list form as well as the historical
mapping form, ignores unknown fields, and supplies the main visual input for
older candidates that predate the declarations.

Only input paths are read from a candidate description.  Commands, Python
interpreters and output roots are owned by the pipeline runtime configuration;
there is no command field in the description format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


class RuntimeArtifactError(ValueError):
    """A candidate runtime description is malformed or unsafe."""


# These names are deliberately small registries.  A caller may record a new
# kind before a consumer is implemented; the pipeline will report it as
# pending instead of treating a description as executable shell input.
VISUAL_CAPTURE_KINDS = frozenset({
    "qa_v3_current_apartment_visual",
    "capture_current_apartment_visual",
})
PIXEL_PRODUCER_KINDS = frozenset({"qa_v3_timeline_native_pixel"})
PIXEL_CONSUMER_KINDS = frozenset({"qa_v3_extended_pixel"})
MEDIA_CONSUMER_KINDS = frozenset({"qa_v3_review_clip"})


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeArtifactError(f"{owner} must be an object")
    return dict(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeArtifactError(f"cannot read candidate fact {path}: {exc}") from exc


def _safe_id(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeArtifactError(f"{owner} must be a non-empty string")
    text = value.strip()
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise RuntimeArtifactError(f"{owner} must be a single identifier: {text!r}")
    return text


def _path(
    value: Any,
    *,
    base: Path,
    owner: str,
    must_exist: bool = True,
    within_base: bool = False,
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise RuntimeArtifactError(f"{owner} must be a non-empty path")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = base / result
    result = result.resolve()
    if within_base and result != base and base not in result.parents:
        raise RuntimeArtifactError(f"{owner} escapes candidate root: {result}")
    if must_exist and not result.is_file():
        raise RuntimeArtifactError(f"{owner} is missing: {result}")
    return result


def _entries(value: Any, *, owner: str) -> list[dict[str, Any]]:
    """Turn list/map descriptor forms into entries without dropping extras."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        # A single descriptor is accepted for convenience.  Otherwise each
        # mapping key is an identifier and its value is the descriptor body.
        if any(key in value for key in ("id", "name", "variant", "segment_id", "segment")):
            return [dict(value)]
        result = []
        for key, item in value.items():
            if not isinstance(item, Mapping):
                raise RuntimeArtifactError(f"{owner}[{key!r}] must be an object")
            row = dict(item)
            row.setdefault("id", key)
            result.append(row)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise RuntimeArtifactError(f"{owner}[{index}] must be an object")
            result.append(dict(item))
        return result
    raise RuntimeArtifactError(f"{owner} must be a list or object")


def _entry_id(row: Mapping[str, Any], *, owner: str, default: str | None = None) -> str:
    for key in ("id", "name", "variant", "segment_id", "segment"):
        if row.get(key) is not None:
            return _safe_id(row[key], owner=f"{owner}.{key}")
    if default is not None:
        return default
    raise RuntimeArtifactError(f"{owner} has no id/name")


def _unique(rows: Sequence[dict[str, Any]], *, owner: str) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for row in rows:
        ident = row["id"]
        if ident in seen:
            raise RuntimeArtifactError(f"{owner} has duplicate id {ident!r}")
        seen.add(ident)
        result.append(row)
    return result


def _kind(row: Mapping[str, Any], *, owner: str, default: str) -> str:
    value = row.get("kind", default)
    return _safe_id(value, owner=f"{owner}.kind")


def _visual_variants(point_dir: Path, fact: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    raw = fact.get("visual_variants")
    legacy = raw is None
    rows = _entries(raw, owner="visual_variants")
    if legacy:
        rows = [{
            "id": "main",
            "kind": "qa_v3_current_apartment_visual",
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
            "release": True,
            "legacy_compatibility": True,
        }]
    normalized = []
    for index, source in enumerate(rows):
        row = dict(source)
        ident = _entry_id(row, owner=f"visual_variants[{index}]", default=f"variant_{index + 1}")
        row["id"] = ident
        row["kind"] = _kind(row, owner=f"visual_variants[{index}]", default="qa_v3_current_apartment_visual")
        for field in ("actor_selection", "timeline"):
            if field not in row:
                raise RuntimeArtifactError(f"visual_variants[{ident}] has no {field}")
            row[field] = _path(
                row[field], base=point_dir, owner=f"visual_variants[{ident}].{field}",
                within_base=True)
        row["release"] = bool(row.get("release", False))
        normalized.append(row)
    return _unique(normalized, owner="visual_variants"), legacy


def _segments(
    point_dir: Path,
    fact: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw = fact.get("segments")
    rows = _entries(raw, owner="segments")
    by_variant = {str(row["id"]): row for row in variants}
    if raw is None:
        main = by_variant.get("main")
        if main is None:
            raise RuntimeArtifactError("legacy candidate has no main visual variant")
        rows = [{
            "id": "segment1",
            "variant": "main",
            "kind": main.get("kind"),
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
            "release": bool(main.get("release")),
            "legacy_compatibility": True,
        }]
    normalized = []
    for index, source in enumerate(rows):
        row = dict(source)
        ident = _entry_id(row, owner=f"segments[{index}]", default=f"segment{index + 1}")
        row["id"] = ident
        reference = row.get("variant") or row.get("visual_variant")
        if reference is not None:
            reference = _safe_id(reference, owner=f"segments[{ident}].variant")
            row["variant"] = reference
            source_variant = by_variant.get(reference)
            if source_variant is None:
                raise RuntimeArtifactError(f"segments[{ident}] references unknown visual variant {reference!r}")
        else:
            source_variant = by_variant.get("main")
            row["variant"] = "main"
        if source_variant is None:
            raise RuntimeArtifactError(
                f"segments[{ident}] has no main visual variant to inherit"
            )
        row["kind"] = _kind(
            row, owner=f"segments[{ident}]",
            default=str(source_variant.get("kind")),
        )
        for field in ("actor_selection", "timeline"):
            if field not in row:
                row[field] = source_variant[field]
            else:
                row[field] = _path(
                    row[field], base=point_dir, owner=f"segments[{ident}].{field}",
                    within_base=True)
        row["actor_selection"] = Path(row["actor_selection"]).resolve()
        row["timeline"] = Path(row["timeline"]).resolve()
        row["release"] = bool(row.get("release", False))
        normalized.append(row)
    return _unique(normalized, owner="segments")


def _external_path(value: Any, *, point_dir: Path, owner: str, must_exist: bool = True) -> Path:
    # Pixel evidence commonly lives in a retained external output root.  It is
    # still resolved as a path, but no executable field is ever accepted.
    return _path(value, base=point_dir, owner=owner, must_exist=must_exist)


def _pixel_evidence(point_dir: Path, fact: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _entries(fact.get("pixel_evidence"), owner="pixel_evidence")
    normalized = []
    for index, source in enumerate(rows):
        row = dict(source)
        ident = _entry_id(row, owner=f"pixel_evidence[{index}]", default=f"pixel_{index + 1}")
        row["id"] = ident
        row["kind"] = _kind(row, owner=f"pixel_evidence[{ident}]", default="qa_v3_extended_pixel")
        # These paths are allowed to be absent until a native pixel producer
        # runs.  A missing input therefore becomes pipeline pending, not a
        # fake complete join.
        if row.get("fact") is not None:
            row["fact"] = _external_path(row["fact"], point_dir=point_dir, owner=f"pixel_evidence[{ident}].fact")
        else:
            row["fact"] = point_dir / "fact_record.json"
        if row.get("pixel_truth") is not None:
            row["pixel_truth"] = _external_path(row["pixel_truth"], point_dir=point_dir, owner=f"pixel_evidence[{ident}].pixel_truth")
        if row.get("pixel_arrays") is not None:
            row["pixel_arrays"] = _external_path(row["pixel_arrays"], point_dir=point_dir, owner=f"pixel_evidence[{ident}].pixel_arrays")
        if row.get("params") is not None:
            row["params"] = _external_path(row["params"], point_dir=point_dir, owner=f"pixel_evidence[{ident}].params")
        if row.get("output") is not None:
            row["output"] = _external_path(row["output"], point_dir=point_dir, owner=f"pixel_evidence[{ident}].output", must_exist=False)
        row["status"] = str(row.get("status", "pending"))
        normalized.append(row)
    return _unique(normalized, owner="pixel_evidence")


def _timeline_frame_count(path: Path, *, owner: str) -> int:
    """Read a live frame_count from a candidate timeline without hashing it."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeArtifactError(f"{owner} cannot read timeline {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise RuntimeArtifactError(f"{owner} timeline must be a JSON object: {path}")
    render = document.get("render")
    render = render if isinstance(render, Mapping) else {}
    frames = document.get("frames")
    if isinstance(frames, list) and frames:
        count: object = len(frames)
    elif "frame_count" in render:
        count = render.get("frame_count")
    elif "frame_count" in document:
        count = document.get("frame_count")
    else:
        raise RuntimeArtifactError(
            f"{owner} timeline has no frame_count: {path}"
        )
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise RuntimeArtifactError(
            f"{owner} timeline frame_count must be a positive integer: {path}"
        )
    return count


def _pixel_producers(point_dir: Path, fact: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _entries(fact.get("pixel_producers"), owner="pixel_producers")
    normalized = []
    for index, source in enumerate(rows):
        row = dict(source)
        ident = _entry_id(
            row, owner=f"pixel_producers[{index}]", default=f"pixel_{index + 1}"
        )
        row["id"] = ident
        row["kind"] = _kind(
            row,
            owner=f"pixel_producers[{ident}]",
            default="qa_v3_timeline_native_pixel",
        )
        for field in ("actor_selection", "timeline"):
            if row.get(field) is None:
                row[field] = (
                    "actor_selection.json"
                    if field == "actor_selection"
                    else "timeline.json"
                )
            row[field] = _path(
                row[field],
                base=point_dir,
                owner=f"pixel_producers[{ident}].{field}",
                within_base=True,
            )
        frames = row.get("binding_frames") or []
        if frames and (
            not isinstance(frames, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in frames
            )
        ):
            raise RuntimeArtifactError(
                f"pixel_producers[{ident}].binding_frames must be integers"
            )
        frame_count = _timeline_frame_count(
            Path(row["timeline"]),
            owner=f"pixel_producers[{ident}]",
        )
        normalized_frames = []
        for value in list(frames):
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeArtifactError(
                    f"pixel_producers[{ident}].binding_frames must be integers"
                )
            if not 0 <= value < frame_count:
                raise RuntimeArtifactError(
                    f"pixel_producers[{ident}].binding_frames must satisfy "
                    f"0 <= frame < frame_count={frame_count}"
                )
            normalized_frames.append(value)
        row["binding_frames"] = normalized_frames
        row["frame_count"] = frame_count
        row["status"] = str(row.get("status", "pending"))
        normalized.append(row)
    return _unique(normalized, owner="pixel_producers")


def _release_media(point_dir: Path, fact: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    raw = fact.get("release_media")
    legacy = raw is None
    rows = _entries(raw, owner="release_media")
    if legacy:
        rows = [{
            "id": "main",
            "variant": "main",
            "segment": "segment1",
            "kind": "qa_v3_review_clip",
            "release": True,
            "legacy_compatibility": True,
        }]
    normalized = []
    for index, source in enumerate(rows):
        row = dict(source)
        ident = _entry_id(row, owner=f"release_media[{index}]", default=f"media_{index + 1}")
        row["id"] = ident
        row["kind"] = _kind(row, owner=f"release_media[{ident}]", default="qa_v3_review_clip")
        row["variant"] = _safe_id(row.get("variant", "main"), owner=f"release_media[{ident}].variant")
        audio_variant = row.get("audio_variant")
        row["audio_variant"] = (
            None
            if audio_variant is None
            else _safe_id(
                audio_variant,
                owner=f"release_media[{ident}].audio_variant",
            )
        )
        if row.get("segment") is not None:
            row["segment"] = _safe_id(row["segment"], owner=f"release_media[{ident}].segment")
        row["release"] = bool(row.get("release", False))
        row["status"] = str(row.get("status", "pending"))
        normalized.append(row)
    return _unique(normalized, owner="release_media"), legacy


def declared_audio_variants(plan: Mapping[str, Any]) -> list[str]:
    """Unique audio variants from normalized release_media, in declaration order.

    Callers must pass the mapping returned by ``load_runtime_artifacts`` so
    list and mapping source forms yield the same variants.
    """

    result: list[str] = []
    for release in plan.get("release_media") or []:
        if not isinstance(release, Mapping):
            continue
        value = release.get("audio_variant")
        if not isinstance(value, str) or not value.strip():
            continue
        if value not in result:
            result.append(value)
    return result


def declared_audio_variants_from_fact(
    fact: Any,
    *,
    point_dir: str | Path,
) -> list[str]:
    """Declared audio variants from one candidate fact.

    ``fact`` must be an object. Missing or JSON-null ``release_media`` is a
    question-only fact: return no variants and do not load visual files.
    Declared list or mapping forms go through ``load_runtime_artifacts``.
    """

    if not isinstance(fact, Mapping):
        raise RuntimeArtifactError("fact_record must be an object")
    if fact.get("release_media") is None:
        return []
    return declared_audio_variants(load_runtime_artifacts(point_dir, fact))


def load_runtime_artifacts(point_dir: str | Path, fact: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load and normalize one candidate's declarative runtime descriptions.

    Returned path values are resolved pathlib.Path objects for callers that
    execute fixed runners.  Unknown descriptor keys are retained so future
    fields can be added without breaking older readers.
    """
    root = Path(point_dir).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeArtifactError(f"candidate directory is missing: {root}")
    if fact is None:
        fact_path = root / "fact_record.json"
        if not fact_path.is_file():
            raise RuntimeArtifactError(f"candidate fact is missing: {fact_path}")
        fact = _mapping(_read_json(fact_path), owner="fact_record")
    else:
        fact = _mapping(fact, owner="fact_record")
    variants, legacy_visual = _visual_variants(root, fact)
    segments = _segments(root, fact, variants)
    pixels = _pixel_evidence(root, fact)
    producers = _pixel_producers(root, fact)
    releases, legacy_release = _release_media(root, fact)
    variant_ids = {row["id"] for row in variants}
    segment_ids = {row["id"] for row in segments}
    if legacy_release and releases and segments:
        releases[0]["segment"] = segments[0]["id"]
        releases[0]["variant"] = segments[0]["variant"]
        releases[0]["audio_variant"] = "main"
    for row in releases:
        if row["variant"] not in variant_ids:
            raise RuntimeArtifactError(
                f"release_media[{row['id']}] references unknown variant "
                f"{row['variant']!r}"
            )
        if row.get("segment") is not None and row["segment"] not in segment_ids:
            raise RuntimeArtifactError(
                f"release_media[{row['id']}] references unknown segment "
                f"{row['segment']!r}"
            )
    consumer_status = fact.get(
        "runtime_consumer_status", "legacy_or_declared"
    )
    if not isinstance(consumer_status, str) or not consumer_status.strip():
        raise RuntimeArtifactError(
            "runtime_consumer_status must be non-empty text when declared"
        )
    return {
        "runtime_consumer_status": consumer_status.strip(),
        "visual_variants": variants,
        "segments": segments,
        "pixel_evidence": pixels,
        "pixel_producers": producers,
        "release_media": releases,
        "legacy_visual_compatibility": legacy_visual,
        "legacy_release_compatibility": legacy_release,
    }


def resolve_runtime_artifacts(
    point_dir: str | Path, fact: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Alias used by tools that call the layer a resolver."""
    return load_runtime_artifacts(point_dir, fact)


def registered_pixel_consumer(kind: str) -> Path | None:
    """Return the fixed in-repository pixel consumer for a registered kind."""
    if kind not in PIXEL_CONSUMER_KINDS:
        return None
    return (
        Path(__file__).resolve().parents[3]
        / "tools" / "qa" / "join_qa_v3_extended_pixel.py"
    )


def registered_pixel_producer(kind: str) -> Path | None:
    """Return the fixed in-repository extra-video/pixel producer for a kind."""
    if kind not in PIXEL_PRODUCER_KINDS:
        return None
    return (
        Path(__file__).resolve().parents[3]
        / "tools" / "qa" / "capture_qa_v3_timeline_pixel.py"
    )
