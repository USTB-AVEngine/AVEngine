"""Portable reader for a self-contained V1 AV QA dataset export.

The reader takes one export root and nothing else.  It resolves every media,
question and calibration reference relative to that root, refuses absolute or
escaping references, and never consults the producing staging tree.  Historical
producer paths stay readable as provenance inside the delivered documents, but
they are never a run dependency of this reader.

Two interfaces are deliberately separated:

* :class:`QaDatasetReader` serves model input only -- the selected audio
  layout, video, question text and public calibration.  It carries no gold
  answer, no engine/world/group identifier and no intervention record.
* :class:`QaDatasetPrivateReader` serves training labels, private grouping and
  scoring.  A caller has to ask for it explicitly.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import struct
from typing import Any


DATASET_INDEX_SCHEMA = "avengine_qa_dataset_index_v1"
PRIVATE_INDEX_SCHEMA = "avengine_qa_dataset_private_index_v1"

PUBLIC_INDEX_NAME = "public/dataset_index.json"
PRIVATE_INDEX_NAME = "private/gold_index.json"

FORMS = ("mcq", "open")
LANGUAGES = ("en", "zh")
QUESTION_KINDS = ("main", "angle_followup")

# Reader behaviour comes from configuration, never from a literal at a call site.
DEFAULT_READER_CONFIG: dict[str, Any] = {
    "form": "mcq",
    "language": "en",
    "audio_layout": "binaural",
    "video_view": "preview",
    "split": None,
    "question_kinds": list(QUESTION_KINDS),
    "observation_protocol": None,
    "probe_media": True,
}

# Any of these appearing in a public projection is a truth or engine-identity leak.
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "truth",
        "gold",
        "correct_index",
        "classes",
        "answer_type",
        "evidence",
        "actor_id",
        "source_id",
        "world_id",
        "world_key",
        "group_id",
        "member_id",
        "episode_id",
        "facts_path",
        "questions_path",
        "source_paths",
        "deferred",
        "input_facts",
        "allow_value",
        "requirements",
        "intervention",
        "interventions",
    }
)


class QaDatasetReadError(ValueError):
    """A dataset export cannot be read as a self-contained V1 dataset."""


def _load(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise QaDatasetReadError(f"{label} is missing from the export: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise QaDatasetReadError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise QaDatasetReadError(f"{label} must be a JSON object: {path}")
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _scan_public(value: Any, *, where: str) -> None:
    """Fail closed when a private field reached a public projection."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_PUBLIC_KEYS:
                raise QaDatasetReadError(
                    f"private field {key!r} reached the public projection at {where}"
                )
            _scan_public(item, where=f"{where}.{key}")
    elif _is_sequence(value):
        for index, item in enumerate(value):
            _scan_public(item, where=f"{where}[{index}]")


def riff_probe(path: str | Path) -> dict[str, Any]:
    """Read a RIFF/WAVE header with the standard library only.

    The delivered mixtures are IEEE-float WAV (format tag 3), which the
    stdlib wave module refuses, so the reader walks the chunks itself rather
    than requiring ffmpeg or soundfile beside the export.
    """

    path = Path(path)
    if not path.is_file():
        raise QaDatasetReadError(f"delivered audio is absent: {path}")
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise QaDatasetReadError(f"delivered audio is not a RIFF/WAVE file: {path}")
        fmt: dict[str, Any] | None = None
        data_bytes: int | None = None
        while True:
            chunk = handle.read(8)
            if len(chunk) < 8:
                break
            name, size = struct.unpack("<4sI", chunk)
            if name == b"fmt ":
                body = handle.read(size + (size % 2))
                if len(body) < 16:
                    raise QaDatasetReadError(f"truncated WAVE fmt chunk: {path}")
                tag, channels, rate, _byte_rate, _align, bits = struct.unpack(
                    "<HHIIHH", body[:16]
                )
                fmt = {
                    "format_tag": int(tag),
                    "channel_count": int(channels),
                    "sample_rate_hz": int(rate),
                    "bits_per_sample": int(bits),
                }
                continue
            if name == b"data":
                data_bytes = int(size)
            handle.seek(size + (size % 2), 1)
        if fmt is None or data_bytes is None:
            raise QaDatasetReadError(f"WAVE file has no fmt/data chunk: {path}")
    frame_bytes = fmt["channel_count"] * max(fmt["bits_per_sample"] // 8, 1)
    fmt["frame_count"] = data_bytes // frame_bytes if frame_bytes else 0
    fmt["data_bytes"] = data_bytes
    fmt["encoding"] = {1: "pcm_int", 3: "pcm_float"}.get(fmt["format_tag"], "other")
    fmt["duration_s"] = (
        fmt["frame_count"] / fmt["sample_rate_hz"] if fmt["sample_rate_hz"] else None
    )
    fmt["probe"] = "stdlib_riff_header"
    return fmt


def _mp4_probe(path: Path) -> dict[str, Any]:
    """Read an ISO-BMFF mvhd duration with the standard library only."""

    if not path.is_file():
        raise QaDatasetReadError(f"delivered video is absent: {path}")
    data = path.read_bytes()
    if len(data) < 8 or data[4:8] not in (b"ftyp", b"moov", b"mdat", b"free", b"skip"):
        raise QaDatasetReadError(f"delivered video is not an ISO-BMFF file: {path}")
    offset = data.find(b"mvhd")
    result: dict[str, Any] = {
        "container": "iso_bmff",
        "byte_size": len(data),
        "probe": "stdlib_iso_bmff_header",
        "duration_s": None,
        "timescale": None,
    }
    if offset > 0:
        # mvhd body: version(1) flags(3) creation modification timescale duration.
        # Version 0 uses 32-bit creation/modification/duration, version 1 uses 64-bit.
        version = data[offset + 4]
        timescale = duration = 0
        if version == 0 and len(data) >= offset + 24:
            timescale, duration = struct.unpack(">II", data[offset + 16 : offset + 24])
        elif version == 1 and len(data) >= offset + 36:
            timescale, duration = struct.unpack(">IQ", data[offset + 24 : offset + 36])
        if timescale:
            result["timescale"] = int(timescale)
            result["duration_s"] = duration / timescale
    return result


def probe_media(path: str | Path) -> dict[str, Any]:
    """Read one delivered media file's real header, without external tools."""

    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix == ".wav":
        return riff_probe(path)
    if suffix in {".mp4", ".m4v", ".mov"}:
        return _mp4_probe(path)
    if not path.is_file():
        raise QaDatasetReadError(f"delivered media is absent: {path}")
    return {"container": "unknown", "byte_size": path.stat().st_size, "probe": "size_only"}


def _reader_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_READER_CONFIG)
    for key, value in dict(config or {}).items():
        if key not in merged:
            raise QaDatasetReadError(f"unknown reader configuration key: {key!r}")
        merged[key] = value
    if merged["form"] not in FORMS:
        raise QaDatasetReadError(f"unsupported answer form: {merged['form']!r}")
    if merged["language"] not in LANGUAGES:
        raise QaDatasetReadError(f"unsupported language: {merged['language']!r}")
    kinds = tuple(str(value) for value in merged["question_kinds"])
    if not kinds or any(kind not in QUESTION_KINDS for kind in kinds):
        raise QaDatasetReadError(f"unsupported question kinds: {kinds!r}")
    merged["question_kinds"] = list(kinds)
    return merged


class QaDatasetReader:
    """Read one export root as a self-contained public V1 QA dataset."""

    def __init__(self, root: str | Path, *, config: Mapping[str, Any] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise QaDatasetReadError(f"dataset root is not a directory: {self.root}")
        self.config = _reader_config(config)
        self.index = _load(self.root / PUBLIC_INDEX_NAME, label="public dataset index")
        if self.index.get("schema") != DATASET_INDEX_SCHEMA:
            raise QaDatasetReadError(
                f"public index schema is not {DATASET_INDEX_SCHEMA}: {self.index.get('schema')!r}"
            )
        samples = self.index.get("samples")
        if not _is_sequence(samples) or not samples:
            raise QaDatasetReadError("public dataset index has no samples")
        self._samples: dict[str, Mapping[str, Any]] = {}
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise QaDatasetReadError("public sample entry is not an object")
            sample_id = sample.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise QaDatasetReadError("public sample entry has no sample_id")
            if sample_id in self._samples:
                raise QaDatasetReadError(f"duplicate public sample_id: {sample_id}")
            self._samples[sample_id] = sample
        # The public index is itself a public projection: check it once, here,
        # so a leaking export fails at open time rather than at model-input time.
        _scan_public(self.index, where="public_index")

    # -- declarations ----------------------------------------------------

    @property
    def dataset_version(self) -> str:
        return str(self.index.get("dataset_version", "unknown"))

    @property
    def counts(self) -> dict[str, Any]:
        return deepcopy(dict(self.index.get("counts") or {}))

    def audio_layouts(self) -> tuple[str, ...]:
        declarations = self.index.get("audio_layouts")
        if not isinstance(declarations, Mapping) or not declarations:
            raise QaDatasetReadError("public index declares no audio layout")
        return tuple(sorted(str(key) for key in declarations))

    def layout_declaration(self, layout: str | None = None) -> dict[str, Any]:
        """Return the delivered layout declaration; never assume a default.

        A first-order ambisonics consumer must read channel order,
        normalization and coordinate frame from this declaration.  The export
        refuses to write an ambisonics track without one, so a missing
        declaration here is an error rather than an assumed SN3D /
        camera-relative track.
        """

        layout = str(layout or self.config["audio_layout"])
        declarations = self.index.get("audio_layouts")
        if not isinstance(declarations, Mapping) or layout not in declarations:
            raise QaDatasetReadError(
                f"audio layout {layout!r} is not declared by this export; "
                f"declared layouts: {self.audio_layouts()}"
            )
        declaration = declarations[layout]
        if not isinstance(declaration, Mapping):
            raise QaDatasetReadError(f"audio layout declaration is not an object: {layout!r}")
        required = ("layout_id", "channel_count", "channel_labels", "channel_order",
                    "normalization", "coordinate_frame", "sample_rate_hz")
        missing = [key for key in required if declaration.get(key) in (None, "")]
        if missing:
            raise QaDatasetReadError(
                f"audio layout {layout!r} declaration is incomplete: missing {missing!r}"
            )
        return deepcopy(dict(declaration))

    def observation_protocols(self) -> dict[str, Any]:
        return deepcopy(dict(self.index.get("observation_protocols") or {}))

    def splits(self) -> dict[str, list[str]]:
        value = self.index.get("splits")
        if not isinstance(value, Mapping):
            return {}
        return {str(key): [str(item) for item in items] for key, items in value.items()}

    # -- path resolution -------------------------------------------------

    def resolve(self, relative: str) -> Path:
        """Resolve one delivered reference; refuse absolute or escaping paths."""

        if not isinstance(relative, str) or not relative.strip():
            raise QaDatasetReadError("delivered reference is empty")
        candidate = Path(relative)
        if candidate.is_absolute():
            raise QaDatasetReadError(
                f"delivered reference must be relative to the export root: {relative!r}"
            )
        if any(part in {"", ".", ".."} for part in candidate.parts):
            raise QaDatasetReadError(f"delivered reference is not confined: {relative!r}")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise QaDatasetReadError(
                f"delivered reference escapes the export root: {relative!r}"
            ) from error
        if not resolved.exists():
            raise QaDatasetReadError(f"delivered reference does not exist: {relative!r}")
        return resolved

    # -- listing ---------------------------------------------------------

    def list_samples(
        self,
        *,
        split: str | None = None,
        room_family: str | None = None,
        qa_id: str | None = None,
        record_kind: str | None = None,
        audio_layout: str | None = None,
        form: str | None = None,
    ) -> list[str]:
        """List public sample IDs matching the selected filters."""

        split = self.config["split"] if split is None else split
        selected = set(self._samples)
        if split is not None:
            splits = self.splits()
            if split not in splits:
                raise QaDatasetReadError(
                    f"split {split!r} is not declared; declared splits: {sorted(splits)}"
                )
            selected &= set(splits[split])
        if audio_layout is not None:
            self.layout_declaration(audio_layout)
        result = []
        for sample_id in sorted(selected):
            sample = self._samples[sample_id]
            if room_family is not None and sample.get("room_family") != room_family:
                continue
            if record_kind is not None and sample.get("record_kind") != record_kind:
                continue
            if audio_layout is not None:
                tracks = (sample.get("media") or {}).get("audio") or {}
                if audio_layout not in tracks:
                    continue
            questions = self.questions(sample_id, form=form)
            if qa_id is not None and not any(row["qa_id"] == qa_id for row in questions):
                continue
            if (qa_id is not None or form is not None) and not questions:
                continue
            result.append(sample_id)
        return result

    def sample(self, sample_id: str) -> dict[str, Any]:
        if sample_id not in self._samples:
            raise QaDatasetReadError(f"unknown sample_id: {sample_id!r}")
        return deepcopy(dict(self._samples[sample_id]))

    def questions(
        self,
        sample_id: str,
        *,
        form: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List this sample's public question rows for the selected form.

        Only questions the export marked valid are listed.  A question that
        exists only as a deferred requirement is not a valid question and is
        never counted here.
        """

        if sample_id not in self._samples:
            raise QaDatasetReadError(f"unknown sample_id: {sample_id!r}")
        wanted_kinds = tuple(kinds) if kinds is not None else tuple(self.config["question_kinds"])
        rows = self._samples[sample_id].get("questions") or []
        result = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise QaDatasetReadError(f"public question row is not an object: {sample_id}")
            if row.get("kind") not in wanted_kinds:
                continue
            forms = row.get("forms") or []
            if form is not None and form not in forms:
                continue
            result.append(deepcopy(dict(row)))
        return result

    # -- media -----------------------------------------------------------

    def media(
        self,
        sample_id: str,
        *,
        audio_layout: str | None = None,
        video_view: str | None = None,
        probe: bool | None = None,
    ) -> dict[str, Any]:
        """Resolve one sample's selected audio layout and video view."""

        sample = self.sample(sample_id)
        layout = str(audio_layout or self.config["audio_layout"])
        view = str(video_view or self.config["video_view"])
        declaration = self.layout_declaration(layout)
        media = sample.get("media") or {}
        tracks = media.get("audio") or {}
        if layout not in tracks:
            raise QaDatasetReadError(
                f"sample {sample_id} has no {layout!r} audio track; "
                f"available: {sorted(tracks)}"
            )
        views = media.get("video") or {}
        if view not in views:
            raise QaDatasetReadError(
                f"sample {sample_id} has no {view!r} video view; available: {sorted(views)}"
            )
        track = tracks[layout]
        audio_relative = track["path"] if isinstance(track, Mapping) else track
        audio_path = self.resolve(str(audio_relative))
        video_path = self.resolve(str(views[view]))
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "audio_layout": layout,
            "audio_path": audio_path,
            "audio_relative_path": str(audio_relative),
            "audio_declaration": declaration,
            "video_view": view,
            "video_path": video_path,
            "video_relative_path": str(views[view]),
        }
        if isinstance(track, Mapping):
            if track.get("attached_view_of"):
                result["attached_view_of"] = str(track["attached_view_of"])
            if track.get("attachment_evidence"):
                result["attachment_evidence"] = deepcopy(dict(track["attachment_evidence"]))
        probe_media_flag = self.config["probe_media"] if probe is None else bool(probe)
        if probe_media_flag:
            audio_probe = probe_media(audio_path)
            if audio_probe.get("channel_count") != declaration["channel_count"]:
                raise QaDatasetReadError(
                    f"sample {sample_id} {layout!r} track has "
                    f"{audio_probe.get('channel_count')} channels but the declaration says "
                    f"{declaration['channel_count']}"
                )
            if audio_probe.get("sample_rate_hz") != declaration["sample_rate_hz"]:
                raise QaDatasetReadError(
                    f"sample {sample_id} {layout!r} track sample rate "
                    f"{audio_probe.get('sample_rate_hz')} differs from the declared "
                    f"{declaration['sample_rate_hz']}"
                )
            result["audio_probe"] = audio_probe
            result["video_probe"] = probe_media(video_path)
        return result

    def read_audio_bytes(self, sample_id: str, *, audio_layout: str | None = None) -> bytes:
        """Read the selected audio track's bytes from the export root."""

        return self.media(sample_id, audio_layout=audio_layout, probe=False)["audio_path"].read_bytes()

    # -- model input -----------------------------------------------------

    def model_input(
        self,
        sample_id: str,
        *,
        form: str | None = None,
        language: str | None = None,
        audio_layout: str | None = None,
        video_view: str | None = None,
        include_media: bool = True,
        question_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Build one sample's public model input.

        The payload is assembled field by field from the public index rather
        than filtered out of a private document, and it is scanned again
        before it is returned.
        """

        form = str(form or self.config["form"])
        language = str(language or self.config["language"])
        if form not in FORMS:
            raise QaDatasetReadError(f"unsupported answer form: {form!r}")
        if language not in LANGUAGES:
            raise QaDatasetReadError(f"unsupported language: {language!r}")
        sample = self.sample(sample_id)
        wanted = None if question_ids is None else {str(value) for value in question_ids}
        items = []
        for row in self.questions(sample_id, form=form):
            if wanted is not None and row["question_id"] not in wanted:
                continue
            prompt = (row.get("prompt") or {}).get(form)
            if not isinstance(prompt, Mapping):
                raise QaDatasetReadError(
                    f"public question {row['question_id']} has no {form!r} prompt"
                )
            text = prompt.get(f"question_{language}")
            if not isinstance(text, str) or not text.strip():
                raise QaDatasetReadError(
                    f"public question {row['question_id']} has no {language!r} {form!r} text"
                )
            item: dict[str, Any] = {
                "question_id": row["question_id"],
                "qa_id": row["qa_id"],
                "kind": row["kind"],
                "form": form,
                "question": text,
            }
            if row.get("required_modalities") is not None:
                item["required_modalities"] = deepcopy(row["required_modalities"])
            if form == "mcq":
                options = prompt.get("options")
                if not _is_sequence(options) or not options:
                    raise QaDatasetReadError(
                        f"public MCQ question {row['question_id']} has no options"
                    )
                item["options"] = [
                    {
                        "option": str(option["option"]),
                        "label": str(option[f"label_{language}"]),
                    }
                    for option in options
                ]
            items.append(item)
        payload: dict[str, Any] = {
            "schema": "avengine_qa_model_input_v1",
            "sample_id": sample_id,
            "form": form,
            "language": language,
            "room_family": sample.get("room_family"),
            "record_kind": sample.get("record_kind"),
            "calibration": deepcopy(dict(sample.get("calibration") or {})),
            "items": items,
        }
        if include_media:
            media = self.media(
                sample_id,
                audio_layout=audio_layout,
                video_view=video_view,
                probe=False,
            )
            payload["media"] = {
                "audio_layout": media["audio_layout"],
                "audio_path": str(media["audio_path"]),
                "audio_relative_path": media["audio_relative_path"],
                "audio_declaration": media["audio_declaration"],
                "video_view": media["video_view"],
                "video_path": str(media["video_path"]),
                "video_relative_path": media["video_relative_path"],
            }
            if "attached_view_of" in media:
                payload["media"]["attached_view_of"] = media["attached_view_of"]
        _scan_public(payload, where=f"model_input[{sample_id}]")
        return payload

    def iter_model_inputs(
        self,
        *,
        split: str | None = None,
        form: str | None = None,
        language: str | None = None,
        audio_layout: str | None = None,
        room_family: str | None = None,
        qa_id: str | None = None,
        include_media: bool = True,
    ) -> Iterator[dict[str, Any]]:
        form = str(form or self.config["form"])
        for sample_id in self.list_samples(
            split=split, room_family=room_family, qa_id=qa_id, form=form,
            audio_layout=audio_layout,
        ):
            yield self.model_input(
                sample_id,
                form=form,
                language=language,
                audio_layout=audio_layout,
                include_media=include_media,
                question_ids=None if qa_id is None else [
                    row["question_id"]
                    for row in self.questions(sample_id, form=form)
                    if row["qa_id"] == qa_id
                ],
            )

    # -- coverage and accounting ----------------------------------------

    def _private_world_keys_for_accounting(self) -> dict[str, str] | None:
        """World joins are administrative metadata, never public sample inputs."""
        if not (self.root / PRIVATE_INDEX_NAME).is_file():
            return None
        private = self.private()
        keys = {}
        for sample_id in self._samples:
            value = private.grouping(sample_id).get("world_key")
            if not isinstance(value, str) or not value.strip():
                raise QaDatasetReadError(
                    f"private world identity is missing for sample {sample_id!r}"
                )
            keys[sample_id] = value
        return keys

    def coverage(self, *, form: str | None = None, split: str | None = None) -> dict[str, Any]:
        """Count actually valid public questions, never requested quota."""

        form = None if form is None else str(form)
        sample_ids = self.list_samples(split=split)
        world_keys = self._private_world_keys_for_accounting()
        per_qa: dict[str, dict[str, Any]] = {}
        form_counts: dict[str, int] = {name: 0 for name in FORMS}
        kind_counts = {kind: 0 for kind in QUESTION_KINDS}
        for sample_id in sample_ids:
            world_key = None if world_keys is None else world_keys[sample_id]
            for row in self.questions(sample_id, form=form, kinds=QUESTION_KINDS):
                bucket = per_qa.setdefault(
                    row["qa_id"],
                    {"main": 0, "angle_followup": 0, "main_world_keys": set(), "forms": {}},
                )
                bucket[row["kind"]] += 1
                kind_counts[row["kind"]] += 1
                if row["kind"] == "main" and world_key is not None:
                    bucket["main_world_keys"].add(world_key)
                for name in row.get("forms") or []:
                    if name in form_counts:
                        form_counts[name] += 1
                    bucket["forms"][name] = bucket["forms"].get(name, 0) + 1
        rows = {}
        for qa_id, bucket in sorted(per_qa.items()):
            rows[qa_id] = {
                "valid_main_questions": bucket["main"],
                "valid_angle_followups": bucket["angle_followup"],
                "distinct_worlds_with_main": (
                    None if world_keys is None else len(bucket["main_world_keys"])
                ),
                "form_counts": dict(sorted(bucket["forms"].items())),
            }
        declared = self.index.get("requested_qa_ids") or []
        missing = [qa_id for qa_id in declared if rows.get(qa_id, {}).get("valid_main_questions", 0) == 0]
        return {
            "schema": "avengine_qa_dataset_coverage_v1",
            "status": "pass",
            "split": split if split is not None else self.config["split"],
            "form": form,
            "sample_count": len(sample_ids),
            "distinct_world_keys": (
                None if world_keys is None else len({world_keys[value] for value in sample_ids})
            ),
            "world_count_authority": (
                "private_index" if world_keys is not None else "unavailable_in_public_only_export"
            ),
            "question_kind_counts": kind_counts,
            "form_counts": form_counts,
            "by_qa_id": rows,
            "qa_ids_without_valid_main_question": missing,
            "counting_note": (
                "One question item with both mcq and open forms counts once; the form "
                "counts are a breakdown of the same items. Angle follow-ups are listed "
                "separately and never counted as main questions. Only questions the "
                "export marked valid are counted; deferred requirements are not."
            ),
        }

    def verify_world_accounting(self) -> dict[str, Any]:
        """Check every world is exported and counted exactly once."""

        world_keys = self._private_world_keys_for_accounting()
        buckets: dict[str, list[str]] = {}
        for sample_id, world_key in (world_keys or {}).items():
            buckets.setdefault(world_key, []).append(sample_id)
        declared = self.counts.get("world_count")
        if world_keys is not None and declared is not None and int(declared) != len(buckets):
            raise QaDatasetReadError(
                f"index declares world_count={declared} but carries {len(buckets)} world keys"
            )
        # A duplicated media payload would mean one world exported twice.
        by_size: dict[int, list[tuple[str, Path]]] = {}
        for sample_id in sorted(self._samples):
            media = self._samples[sample_id].get("media") or {}
            references = [
                str(value) for value in (media.get("video") or {}).values()
            ] + [
                str(track["path"] if isinstance(track, Mapping) else track)
                for track in (media.get("audio") or {}).values()
            ]
            for relative in references:
                path = self.resolve(relative)
                by_size.setdefault(path.stat().st_size, []).append((relative, path))
        duplicates = []
        for size, rows in by_size.items():
            distinct = {relative: path for relative, path in rows}
            if len(distinct) < 2:
                continue
            payloads: dict[bytes, str] = {}
            for relative, path in sorted(distinct.items()):
                payload = path.read_bytes()
                if payload in payloads:
                    duplicates.append(
                        {"byte_size": size, "paths": sorted([payloads[payload], relative])}
                    )
                else:
                    payloads[payload] = relative
        if duplicates:
            raise QaDatasetReadError(
                f"the export stores the same media payload more than once: {duplicates!r}"
            )
        return {
            "schema": "avengine_qa_dataset_world_accounting_v1",
            "status": "pass" if world_keys is not None else "not_run",
            "reason": None if world_keys is not None else "private world metadata is not distributed with public-only inputs",
            "world_count_authority": "private_index" if world_keys is not None else None,
            "world_key_count": None if world_keys is None else len(buckets),
            "declared_world_count": declared,
            "samples_per_world_key": {key: len(value) for key, value in sorted(buckets.items())},
            "distinct_media_payloads": sum(len({row[0] for row in rows}) for rows in by_size.values()),
            "duplicate_media_payloads": 0,
        }

    def verify_self_contained(self) -> dict[str, Any]:
        """Check every reference this reader needs resolves inside the root."""

        checked = 0
        absolute: list[str] = []
        for sample_id in sorted(self._samples):
            sample = self._samples[sample_id]
            media = sample.get("media") or {}
            references = [str(value) for value in (media.get("video") or {}).values()]
            references += [
                str(track["path"] if isinstance(track, Mapping) else track)
                for track in (media.get("audio") or {}).values()
            ]
            for relative in references:
                if Path(relative).is_absolute():
                    absolute.append(relative)
                    continue
                path = self.resolve(relative)
                if path.is_symlink():
                    raise QaDatasetReadError(
                        f"delivered media is a symlink, not a portable payload: {relative}"
                    )
                checked += 1
        if absolute:
            raise QaDatasetReadError(
                f"public index carries absolute media references: {sorted(set(absolute))!r}"
            )
        return {
            "schema": "avengine_qa_dataset_portability_v1",
            "status": "pass",
            "root": str(self.root),
            "checked_media_references": checked,
            "absolute_references": 0,
            "symlink_references": 0,
            "claim_boundary": (
                "Public read/score references resolve inside this export root only. "
                "Historical producer paths remain inside delivered provenance documents "
                "as history and are not read by this reader."
            ),
        }

    def public_payload_paths(self) -> list[str]:
        """List the relative paths a public-only distribution needs.

        The delivered package keeps the private question sets and the provenance
        tree beside the public projection, so a publisher needs the explicit
        list rather than a whole-directory copy.
        """

        payload = self.index.get("public_payload")
        if not isinstance(payload, Mapping) or not _is_sequence(payload.get("paths")):
            raise QaDatasetReadError("public index declares no public_payload path list")
        return [str(value) for value in payload["paths"]]

    def verify_private_isolation(self) -> dict[str, Any]:
        """Check the declared public payload resolves and carries no gold.

        Every JSON file in the payload is scanned for private fields, and every
        media reference must be covered by the list, so a public-only copy
        cannot silently omit a file the reader needs or include a gold file.
        """

        paths = self.public_payload_paths()
        referenced = set()
        for sample_id in sorted(self._samples):
            media = self._samples[sample_id].get("media") or {}
            referenced |= {str(value) for value in (media.get("video") or {}).values()}
            referenced |= {
                str(track["path"] if isinstance(track, Mapping) else track)
                for track in (media.get("audio") or {}).values()
            }
        uncovered = sorted(referenced - set(paths))
        if uncovered:
            raise QaDatasetReadError(
                f"public payload list omits media the reader needs: {uncovered!r}"
            )
        scanned = 0
        for relative in paths:
            path = self.resolve(relative)
            if path.suffix.casefold() != ".json":
                continue
            _scan_public(json.loads(path.read_text(encoding="utf-8")), where=relative)
            scanned += 1
        return {
            "schema": "avengine_qa_dataset_private_isolation_v1",
            "status": "pass",
            "public_payload_file_count": len(paths),
            "public_json_files_scanned": scanned,
            "media_references_covered": len(referenced),
            "claim_boundary": (
                "The declared public payload resolves and carries no gold, engine "
                "identity or intervention field. The rest of the package is private."
            ),
        }

    # -- private hand-off ------------------------------------------------

    def private(self) -> "QaDatasetPrivateReader":
        """Return the separate private-label interface for this export."""

        return QaDatasetPrivateReader(self.root, public=self)


class QaDatasetPrivateReader:
    """Training labels, private grouping and scoring for one export root.

    This interface is separate on purpose.  Nothing here is reachable from
    :meth:`QaDatasetReader.model_input`.
    """

    def __init__(self, root: str | Path, *, public: QaDatasetReader | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.public = public if public is not None else QaDatasetReader(self.root)
        self.index = _load(self.root / PRIVATE_INDEX_NAME, label="private gold index")
        if self.index.get("schema") != PRIVATE_INDEX_SCHEMA:
            raise QaDatasetReadError(
                f"private index schema is not {PRIVATE_INDEX_SCHEMA}: {self.index.get('schema')!r}"
            )
        self._records: dict[str, Mapping[str, Any]] = {}
        for record in self.index.get("records") or []:
            if not isinstance(record, Mapping):
                raise QaDatasetReadError("private record is not an object")
            self._records[str(record.get("sample_id"))] = record
        missing = sorted(set(self.public.list_samples()) - set(self._records))
        if missing:
            raise QaDatasetReadError(f"private index has no record for samples: {missing!r}")
        self._question_sets: dict[str, Mapping[str, Any]] = {}

    def question_set(self, sample_id: str) -> Mapping[str, Any]:
        """Load one sample's private question set from the export root."""

        if sample_id not in self._records:
            raise QaDatasetReadError(f"unknown sample_id: {sample_id!r}")
        if sample_id not in self._question_sets:
            relative = str(self._records[sample_id]["questions_path"])
            self._question_sets[sample_id] = _load(
                self.public.resolve(relative), label="private question set"
            )
        return self._question_sets[sample_id]

    def grouping(self, sample_id: str) -> dict[str, Any]:
        """Return one sample's private grouping and engine identity."""

        if sample_id not in self._records:
            raise QaDatasetReadError(f"unknown sample_id: {sample_id!r}")
        record = self._records[sample_id]
        return {
            "sample_id": sample_id,
            "record_kind": record.get("record_kind"),
            "group_id": record.get("group_id"),
            "member_id": record.get("member_id"),
            "world_id": record.get("world_id"),
            "episode_id": record.get("episode_id"),
            "world_key": record.get("world_key"),
            "core_task": record.get("core_task"),
        }

    def groups(self) -> dict[str, list[str]]:
        """Map each private group_id to its member sample IDs."""

        result: dict[str, list[str]] = {}
        for sample_id, record in sorted(self._records.items()):
            group_id = record.get("group_id")
            if group_id is None:
                continue
            result.setdefault(str(group_id), []).append(sample_id)
        return result

    def question_id_map(self, sample_id: str) -> dict[str, str]:
        """Map this sample's public question IDs to their private IDs."""

        if sample_id not in self._records:
            raise QaDatasetReadError(f"unknown sample_id: {sample_id!r}")
        mapping = self._records[sample_id].get("question_id_map")
        if not isinstance(mapping, Mapping) or not mapping:
            raise QaDatasetReadError(f"private record has no question_id_map: {sample_id}")
        return {str(key): str(value) for key, value in mapping.items()}

    def gold(self, sample_id: str, *, form: str | None = None) -> list[dict[str, Any]]:
        """Return this sample's gold answers keyed by public question ID."""

        form = str(form or self.public.config["form"])
        if form not in FORMS:
            raise QaDatasetReadError(f"unsupported answer form: {form!r}")
        question_set = self.question_set(sample_id)
        by_private = {}
        for item in _iter_items(question_set):
            by_private[str(item.get("question_id"))] = item
        rows = []
        for public_id, private_id in sorted(self.question_id_map(sample_id).items()):
            item = by_private.get(private_id)
            if item is None:
                raise QaDatasetReadError(
                    f"private question {private_id!r} is absent from the delivered set"
                )
            forms = item.get("forms") or {}
            if form not in forms:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "public_question_id": public_id,
                    "private_question_id": private_id,
                    "qa_id": item.get("qa_id"),
                    "form": form,
                    "gold_answer": _gold_answer(forms[form], form),
                }
            )
        return rows

    def gold_answers(self, *, split: str | None = None, form: str | None = None) -> dict[str, Any]:
        """Collect gold answers for every selected sample, for a replay smoke."""

        form = str(form or self.public.config["form"])
        answers: dict[str, dict[str, Any]] = {}
        for sample_id in self.public.list_samples(split=split, form=form):
            answers[sample_id] = {
                row["public_question_id"]: row["gold_answer"]
                for row in self.gold(sample_id, form=form)
            }
        return answers

    def score(
        self,
        predictions: Mapping[str, Mapping[str, Any]],
        *,
        form: str | None = None,
        split: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score public-ID predictions against the delivered private gold.

        Every valid question keeps the denominator.  A missing or unparsable
        answer scores zero rather than dropping out of the count.
        """

        from avengine.qa.unified_scoring import score_unified_item

        form = str(form or self.public.config["form"])
        if form not in FORMS:
            raise QaDatasetReadError(f"unsupported answer form: {form!r}")
        sample_ids = self.public.list_samples(split=split, form=form)
        unknown = sorted(set(predictions) - set(sample_ids))
        if unknown:
            raise QaDatasetReadError(f"predictions name samples outside the selection: {unknown!r}")
        records: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            question_set = self.question_set(sample_id)
            by_private = {str(item.get("question_id")): item for item in _iter_items(question_set)}
            given = dict(predictions.get(sample_id) or {})
            for public_id, private_id in sorted(self.question_id_map(sample_id).items()):
                item = by_private.get(private_id)
                if item is None or form not in (item.get("forms") or {}):
                    continue
                answered = public_id in given
                answer = given.get(public_id, "")
                result = score_unified_item(item, answer, form=form, params=params)
                records.append(
                    {
                        "sample_id": sample_id,
                        "public_question_id": public_id,
                        "qa_id": item.get("qa_id"),
                        "kind": "angle_followup"
                        if _is_angle_followup(question_set, private_id)
                        else "main",
                        "form": form,
                        "answered": answered,
                        "score": 0.0 if not answered else float(result.get("score") or 0.0),
                        "status": result.get("status") if answered else "missing_answer",
                        "correct": bool(answered and float(result.get("score") or 0.0) >= 1.0),
                    }
                )
        return _aggregate(records, form=form, split=split)

    def score_group_relations(
        self,
        predictions: Mapping[str, Any] | None = None,
        *,
        form: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score the four-member core-group answer relations for this export.

        Predictions are keyed by public sample_id and translated to the private
        core sample IDs the group document uses. With no predictions the
        delivered gold is replayed, which tests the relation scorer rather than
        a model. An Episode-only export has no group relation and says so.
        """

        from avengine.qa.binding_group_scoring import (
            BindingGroupScoreError,
            score_binding_groups,
        )

        form = str(form or self.public.config["form"])
        if form not in FORMS:
            raise QaDatasetReadError(f"unsupported answer form: {form!r}")
        relative = self.index.get("core_bundle")
        if not relative:
            return {
                "schema": "avengine_qa_dataset_group_relation_score_v1",
                "status": "not_applicable",
                "reason": "this export carries no core group bundle",
                "form": form,
            }
        document = _load(self.public.resolve(str(relative)), label="delivered core bundle")
        core_by_public = {
            sample_id: str(record["core_sample_id"])
            for sample_id, record in self._records.items()
            if record.get("core_sample_id")
        }
        answers: dict[str, Any] = {}
        replayed_gold = predictions is None
        if replayed_gold:
            for group in document.get("groups") or []:
                for member in group.get("members") or []:
                    question = member.get("question")
                    if isinstance(question, Mapping) and form in (question.get("forms") or {}):
                        answers[str(member["sample_id"])] = _group_gold_answer(question, form)
        else:
            unknown = sorted(set(predictions) - set(core_by_public))
            if unknown:
                raise QaDatasetReadError(
                    f"group predictions name samples without a core member: {unknown!r}"
                )
            for sample_id, answer in predictions.items():
                answers[core_by_public[sample_id]] = answer
        try:
            result = score_binding_groups(document, answers, form=form, params=params)
        except (BindingGroupScoreError, ValueError) as error:
            raise QaDatasetReadError(f"group relation scoring failed: {error}") from error
        return {
            "schema": "avengine_qa_dataset_group_relation_score_v1",
            "status": "pass",
            "form": form,
            "replayed_delivered_gold": replayed_gold,
            "counts": deepcopy(dict(result.get("counts") or {})),
            "group_metrics": deepcopy(dict(result.get("group_metrics") or {})),
            "relation_metrics": deepcopy(dict(result.get("relation_metrics") or {})),
            "denominator_note": (
                "members_total keeps every delivered member in the denominator; a member "
                "without the selected form is reported, not dropped."
            ),
        }

    def gold_replay_smoke(
        self,
        *,
        form: str | None = None,
        split: str | None = None,
        wrong_answer: str = "definitely-not-the-answer",
    ) -> dict[str, Any]:
        """Score gold answers and a deliberately wrong answer set.

        A scorer that accepts anything is not a scorer, so the smoke reports
        both directions instead of only the gold pass.
        """

        form = str(form or self.public.config["form"])
        gold = self.gold_answers(split=split, form=form)
        wrong = {
            sample_id: {question_id: wrong_answer for question_id in rows}
            for sample_id, rows in gold.items()
        }
        gold_result = self.score(gold, form=form, split=split)
        wrong_result = self.score(wrong, form=form, split=split)
        return {
            "schema": "avengine_qa_dataset_gold_replay_v1",
            "status": "pass",
            "form": form,
            "split": split if split is not None else self.public.config["split"],
            "gold": gold_result,
            "wrong": wrong_result,
            "separates_correct_from_wrong": (
                gold_result["metrics"]["accuracy"] > wrong_result["metrics"]["accuracy"]
            ),
            "claim_boundary": (
                "Gold replay and a wrong-answer control for the delivered scorer only. "
                "No model is run and no human answerability is claimed."
            ),
        }


def _iter_items(question_set: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    seen: set[str] = set()
    for key in ("items", "angle_followups"):
        for item in question_set.get(key) or []:
            if not isinstance(item, Mapping):
                continue
            question_id = str(item.get("question_id"))
            if question_id in seen:
                continue
            seen.add(question_id)
            yield item


def _is_angle_followup(question_set: Mapping[str, Any], private_id: str) -> bool:
    for item in question_set.get("angle_followups") or []:
        if isinstance(item, Mapping) and str(item.get("question_id")) == private_id:
            return True
    return False


def _group_gold_answer(question: Mapping[str, Any], form: str) -> Any:
    """Extract the gold answer a core-group member question declares."""

    spec = (question.get("forms") or {})[form]
    if not isinstance(spec, Mapping):
        raise QaDatasetReadError(f"core member question has no {form!r} form")
    return _gold_answer(spec, form)


def _gold_answer(form_spec: Mapping[str, Any], form: str) -> Any:
    """Extract the answer a perfect responder would give for one form."""

    if form == "mcq":
        gold = form_spec.get("gold")
        if isinstance(gold, Mapping):
            index = gold.get("correct_index")
            options = form_spec.get("options")
            if isinstance(index, int) and _is_sequence(options) and 0 <= index < len(options):
                option = options[index]
                if isinstance(option, Mapping) and option.get("option"):
                    return str(option["option"])
                letter = chr(ord("A") + index)
                return letter
            if gold.get("value") is not None:
                return str(gold["value"])
        raise QaDatasetReadError("delivered MCQ form carries no gold")
    # These answer_type names are the ones avengine.qa.unified_scoring dispatches
    # on, so a replay produces the exact text that scorer parses.
    answer_type = form_spec.get("answer_type")
    truth = form_spec.get("truth")
    if answer_type == "closed_set":
        classes = form_spec.get("classes")
        if isinstance(classes, Mapping):
            labels = classes.get(truth)
            if _is_sequence(labels) and labels:
                return str(labels[0])
        return str(truth)
    if answer_type in {"angle_deg", "time_s"}:
        if isinstance(truth, Mapping):
            truth = truth.get("azimuth_deg", truth.get("value"))
        return str(truth)
    if answer_type in {"count_pair", "count_single"}:
        values = truth if _is_sequence(truth) else [truth]
        return " ".join(str(int(value)) for value in values)
    if answer_type == "time_range_s":
        if isinstance(truth, Mapping):
            start = truth.get("start_s", truth.get("start"))
            end = truth.get("end_s", truth.get("end"))
        elif _is_sequence(truth) and len(truth) == 2:
            start, end = truth
        else:
            raise QaDatasetReadError(f"time_range_s truth is not an interval: {truth!r}")
        return f"{start} to {end}"
    if answer_type == "transcript_wer":
        return str(truth)
    if isinstance(truth, (str, int, float)):
        return str(truth)
    raise QaDatasetReadError(
        f"delivered open form carries no replayable truth (answer_type={answer_type!r})"
    )


def _aggregate(records: Sequence[Mapping[str, Any]], *, form: str, split: str | None) -> dict[str, Any]:
    total = len(records)
    answered = sum(1 for row in records if row["answered"])
    correct = sum(1 for row in records if row["correct"])
    score_sum = sum(float(row["score"]) for row in records)
    by_qa: dict[str, dict[str, Any]] = {}
    for row in records:
        bucket = by_qa.setdefault(
            str(row["qa_id"]), {"questions": 0, "answered": 0, "correct": 0, "score_sum": 0.0}
        )
        bucket["questions"] += 1
        bucket["answered"] += int(row["answered"])
        bucket["correct"] += int(row["correct"])
        bucket["score_sum"] += float(row["score"])
    for bucket in by_qa.values():
        bucket["accuracy"] = bucket["correct"] / bucket["questions"] if bucket["questions"] else 0.0
        bucket["mean_score"] = (
            bucket["score_sum"] / bucket["questions"] if bucket["questions"] else 0.0
        )
    kinds: dict[str, int] = {}
    for row in records:
        kinds[str(row["kind"])] = kinds.get(str(row["kind"]), 0) + 1
    return {
        "schema": "avengine_qa_dataset_score_v1",
        "status": "pass",
        "form": form,
        "split": split,
        "metrics": {
            "question_count": total,
            "answered_count": answered,
            "missing_answer_count": total - answered,
            "correct_count": correct,
            "accuracy": correct / total if total else 0.0,
            "mean_score": score_sum / total if total else 0.0,
        },
        "question_kind_counts": kinds,
        "by_qa_id": dict(sorted(by_qa.items())),
        "records": list(records),
        "denominator_note": (
            "Every valid question of the selected form stays in the denominator. "
            "A missing or unparsable answer scores zero; it is not dropped."
        ),
    }


def open_qa_dataset(
    root: str | Path, *, config: Mapping[str, Any] | None = None
) -> QaDatasetReader:
    """Open one export root as a public V1 QA dataset."""

    return QaDatasetReader(root, config=config)


__all__ = [
    "DATASET_INDEX_SCHEMA",
    "DEFAULT_READER_CONFIG",
    "FORMS",
    "LANGUAGES",
    "PRIVATE_INDEX_SCHEMA",
    "PRIVATE_INDEX_NAME",
    "PUBLIC_INDEX_NAME",
    "QUESTION_KINDS",
    "QaDatasetPrivateReader",
    "QaDatasetReadError",
    "QaDatasetReader",
    "open_qa_dataset",
    "probe_media",
    "riff_probe",
]
