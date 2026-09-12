"""Build and validate an intermediate portable delivery for binding groups.

The delivery copies the public media/questions and private group/catalog golds
needed for the current accepted snapshot.  Native runtime dependencies remain
explicit external inputs; this package is not a portable UE, Habitat, or RLR
renderer.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

from avengine.qa.binding_catalog import whole_degree_display
from avengine.qa.binding_catalog_scoring import (
    BindingCatalogScoreError,
    score_binding_catalog,
)
from avengine.qa.binding_group_scoring import (
    BindingGroupScoreError,
    score_binding_groups,
)
from avengine.qa.unified_catalog import (
    QA_IDS,
    generate_unified_questions,
    iter_unified_items,
)


DELIVERY_SCHEMA = "avengine_binding_delivery_v1"
_EXTERNAL_SCHEMA = "avengine_binding_delivery_external_dependencies_v1"
_PATH_MAP_SCHEMA = "avengine_binding_delivery_path_map_v1"
_JOIN_SCHEMA = "avengine_binding_delivery_join_index_v1"
_VALIDATION_SCHEMA = "avengine_binding_delivery_validation_v1"
_FORMS = ("mcq", "open")
_JSON_READBACK_KEYS = {
    "plan",
    "neutral_readback",
    "frame_readbacks",
    "pixel_visibility_truth",
    "audio_program",
    "audio_readback",
    "research_report",
    "appearance_review",
    "occluder_evidence",
    "occluder_registry",
    "sound_registry",
    "derived_from_facts",
}
_PATH_KEYS = {
    "sound_pool",
    "source_registry",
    "room_catalog",
    "prepared_manifest",
    "hrtf",
    "uproject",
    "unreal_editor",
    "runtime_prefix",
    "rlr_sdk_root",
    "spear_ext_dir",
    "magnum_python_site",
    "mp3d_root",
    "ddc_directory",
}


class BindingDeliveryError(ValueError):
    """A portable binding delivery input or output is malformed."""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _load(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise BindingDeliveryError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BindingDeliveryError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise BindingDeliveryError(f"{label} must be a JSON object: {path}")
    return value


def _write(path: Path, value: Any, *, refuse_existing: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise BindingDeliveryError(f"refusing to overwrite delivery file: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BindingDeliveryError(f"expected a non-empty path under {base}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _portable_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BindingDeliveryError(f"delivery path is not relative and confined: {value!r}")
    return path.as_posix()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise BindingDeliveryError(f"delivery source is absent: {source}")
    if destination.exists() or destination.is_symlink():
        raise BindingDeliveryError(f"refusing to overwrite delivery source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _path_like_key(key: Any) -> bool:
    text = str(key).casefold()
    return (
        text in _PATH_KEYS
        or text.endswith("_path")
        or text.endswith("_root")
        or text.endswith("_dir")
    )


def _source_path(value: Any, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _resolve(value, base)


def _add_reference(
    table: dict[str, dict[str, Any]],
    path: Path,
    *,
    kind: str,
    reference: str,
    delivered_path: str | None = None,
) -> None:
    key = str(path.resolve(strict=False))
    row = table.setdefault(
        key,
        {
            "source_path": key,
            "delivered_path": delivered_path,
            "kind": kind,
            "references": [],
        },
    )
    if delivered_path is not None:
        previous = row.get("delivered_path")
        if previous is not None and previous != delivered_path:
            raise BindingDeliveryError(
                f"one source file maps to two delivery paths: {key}: "
                f"{previous!r} and {delivered_path!r}"
            )
        row["delivered_path"] = delivered_path
    if kind != row.get("kind") and row.get("kind") != "facts":
        row["kind"] = row.get("kind") or kind
    if reference not in row["references"]:
        row["references"].append(reference)


def _rewrite_paths(
    value: Any,
    path_map: Mapping[str, Mapping[str, Any]],
    *,
    destination_base: Path,
    source_base: Path,
) -> Any:
    # Rewrites copied links relative to the copied JSON file's directory.
    if isinstance(value, Mapping):
        return {
            key: _rewrite_paths(
                child,
                path_map,
                destination_base=destination_base,
                source_base=source_base,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_paths(
                child,
                path_map,
                destination_base=destination_base,
                source_base=source_base,
            )
            for child in value
        ]
    if not isinstance(value, str):
        return value
    row = path_map.get(value)
    if row is None:
        try:
            resolved = (
                (source_base / value).resolve(strict=False)
                if not Path(value).is_absolute()
                else Path(value).expanduser().resolve(strict=False)
            )
            row = path_map.get(str(resolved))
        except (OSError, RuntimeError, ValueError):
            row = None
    if row is None or not row.get("delivered_path"):
        return value
    relative = Path(row["delivered_path"])
    try:
        return Path(
            __import__("os").path.relpath(relative, start=destination_base)
        ).as_posix()
    except ValueError as error:
        raise BindingDeliveryError(
            f"copied path cannot be made relative: {row['source_path']}"
        ) from error


def _collect_path_values(
    value: Any,
    *,
    reference: str,
    path_map: Mapping[str, Mapping[str, Any]],
    external: dict[str, dict[str, Any]],
    key_hint: str | None = None,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_ref = f"{reference}.{key}"
            if isinstance(child, str) and Path(child).is_absolute():
                try:
                    resolved = str(Path(child).expanduser().resolve(strict=False))
                except (OSError, RuntimeError, ValueError):
                    resolved = child
                if resolved not in path_map:
                    row = external.setdefault(
                        resolved,
                        {
                            "path": resolved,
                            "kind": "external_generation_dependency",
                            "copied": False,
                            "references": [],
                        },
                    )
                    if child_ref not in row["references"]:
                        row["references"].append(child_ref)
            elif isinstance(child, (Mapping, list)):
                _collect_path_values(
                    child,
                    reference=child_ref,
                    path_map=path_map,
                    external=external,
                    key_hint=str(key),
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_path_values(
                child,
                reference=f"{reference}[{index}]",
                path_map=path_map,
                external=external,
                key_hint=key_hint,
            )


def _gold_prediction(item: Mapping[str, Any], form: str) -> str:
    specification = item.get("forms", {}).get(form)
    if not isinstance(specification, Mapping):
        raise BindingDeliveryError(f"cannot derive GT answer: missing {form} form")
    if form == "mcq":
        gold = specification.get("gold")
        options = specification.get("options")
        if (
            not isinstance(gold, Mapping)
            or not isinstance(options, Sequence)
            or not isinstance(gold.get("correct_index"), int)
        ):
            raise BindingDeliveryError("MCQ gold is malformed")
        return chr(ord("A") + int(gold["correct_index"]))

    answer_type = specification.get("answer_type")
    truth = specification.get("truth")
    if answer_type == "closed_set":
        classes = specification.get("classes")
        if isinstance(classes, Mapping):
            aliases = classes.get(str(truth))
            if isinstance(aliases, Sequence) and aliases:
                return str(aliases[0])
        return str(truth)
    if answer_type == "transcript_wer":
        return str(truth)
    if answer_type == "angle_deg":
        return f"{truth} degrees"
    if answer_type == "time_s":
        return f"{truth} seconds"
    if answer_type == "time_range_s":
        return f"{truth[0]}-{truth[1]}"
    if answer_type in {"count_pair", "count_single"}:
        values = truth if isinstance(truth, Sequence) and not isinstance(truth, (str, bytes)) else [truth]
        return " ".join(str(value) for value in values)
    raise BindingDeliveryError(f"unsupported open answer type: {answer_type!r}")


def _media_probe(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    if not path.is_file():
        raise BindingDeliveryError(f"delivered media is absent: {path}")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise BindingDeliveryError(
            f"ffprobe failed for delivered media {path}: {result.stderr.strip()}"
        )
    try:
        value = json.loads(result.stdout)
    except ValueError as error:
        raise BindingDeliveryError(f"ffprobe returned invalid JSON: {path}") from error
    streams = value.get("streams")
    if not isinstance(streams, list) or not streams:
        raise BindingDeliveryError(f"delivered media has no streams: {path}")
    duration = value.get("format", {}).get("duration")
    if duration is not None and float(duration) <= 0:
        raise BindingDeliveryError(f"delivered media has non-positive duration: {path}")
    cache[key] = {
        "path": key,
        "stream_count": len(streams),
        "duration_s": float(duration) if duration is not None else None,
        "streams": streams,
    }
    return cache[key]


def _join_inputs(
    core_path: Path | None,
    catalog_path: Path,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any], list[dict[str, Any]]]:
    """Join an accepted core bundle with its catalog, or take the catalog alone.

    An accepted four-member core group is one route into a delivery, not the
    only one. With ``core_path`` omitted, the catalog's ordinary Episode
    records are delivered on their own and no group relation is claimed.
    """

    catalog = _load(catalog_path, label="full catalog index")
    if catalog.get("status") != "research_candidate":
        raise BindingDeliveryError("catalog index is not research_candidate")
    records = catalog.get("records")
    if not _is_sequence(records) or not records:
        raise BindingDeliveryError("catalog index has no records")
    sample_ids = [record.get("sample_id") for record in records if isinstance(record, Mapping)]
    if (len(sample_ids) != len(records) or any(not isinstance(value, str) or not value for value in sample_ids)
            or len(set(sample_ids)) != len(sample_ids)):
        raise BindingDeliveryError("catalog sample IDs must be unique nonempty strings")
    if core_path is None:
        joined = []
        for record in sorted(records, key=lambda row: str(row.get("sample_id"))):
            if not isinstance(record, Mapping):
                raise BindingDeliveryError("catalog record is not an object")
            if record.get("record_kind") != "episode":
                raise BindingDeliveryError(
                    "a core-free delivery accepts only record_kind 'episode' records; "
                    f"sample {record.get('sample_id')!r} is "
                    f"{record.get('record_kind')!r}"
                )
            joined.append(
                {
                    "group_id": None,
                    "member_id": None,
                    "episode_id": record.get("episode_id"),
                    "core_sample_id": None,
                    "catalog_sample_id": record.get("sample_id"),
                    "core_facts_path": None,
                    "catalog_facts_path": record.get("facts_path"),
                    "catalog_questions_path": record.get("questions_path"),
                    "catalog_public_question_ids": list(record.get("public_question_ids", [])),
                    "core_group": None,
                    "core_member": None,
                    "catalog_record": record,
                }
            )
        return None, catalog, joined
    core = _load(core_path, label="core binding bundle")
    if core.get("schema") != "avengine_binding_groups_v1":
        raise BindingDeliveryError("core bundle schema is not avengine_binding_groups_v1")
    if core.get("status") != "research_candidate":
        raise BindingDeliveryError("core bundle is not research_candidate")
    groups = core.get("groups")
    if not _is_sequence(groups) or not groups:
        raise BindingDeliveryError("core bundle has no groups")

    core_by_key: dict[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise BindingDeliveryError("core group is not an object")
        group_id = str(group.get("group_id"))
        for member in group.get("members", []):
            if not isinstance(member, Mapping):
                raise BindingDeliveryError("core member is not an object")
            key = (group_id, str(member.get("member_id")))
            if key in core_by_key:
                raise BindingDeliveryError(f"duplicate core join key: {key!r}")
            core_by_key[key] = (group, member)

    catalog_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    ordinary_records = []
    ordinary_ids = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise BindingDeliveryError("catalog record is not an object")
        if record.get("record_kind") == "episode":
            episode_id = record.get("episode_id")
            if (not isinstance(episode_id, str) or not episode_id or episode_id in ordinary_ids
                    or record.get("group_id") is not None or record.get("member_id") is not None):
                raise BindingDeliveryError("ordinary catalog records need unique episode identity and no core join key")
            ordinary_ids.add(episode_id)
            ordinary_records.append(record)
            continue
        key = (str(record.get("group_id")), str(record.get("member_id")))
        if key in catalog_by_key:
            raise BindingDeliveryError(f"duplicate catalog join key: {key!r}")
        catalog_by_key[key] = record
    if set(core_by_key) != set(catalog_by_key):
        raise BindingDeliveryError(
            "core/catalog join keys differ: "
            f"core_only={sorted(set(core_by_key) - set(catalog_by_key))!r}, "
            f"catalog_only={sorted(set(catalog_by_key) - set(core_by_key))!r}"
        )

    joined = []
    for key in sorted(core_by_key):
        group, member = core_by_key[key]
        record = catalog_by_key[key]
        joined.append(
            {
                "group_id": key[0],
                "member_id": key[1],
                "core_sample_id": member.get("sample_id"),
                "catalog_sample_id": record.get("sample_id"),
                "core_facts_path": member.get("facts_path"),
                "catalog_facts_path": record.get("facts_path"),
                "catalog_questions_path": record.get("questions_path"),
                "catalog_public_question_ids": list(record.get("public_question_ids", [])),
                "core_group": group,
                "core_member": member,
                "catalog_record": record,
            }
        )
    for record in sorted(ordinary_records, key=lambda row: str(row["sample_id"])):
        joined.append({
            "group_id": None, "member_id": None, "episode_id": record["episode_id"],
            "core_sample_id": None, "catalog_sample_id": record["sample_id"],
            "core_facts_path": None, "catalog_facts_path": record.get("facts_path"),
            "catalog_questions_path": record.get("questions_path"),
            "catalog_public_question_ids": list(record.get("public_question_ids", [])),
            "core_group": None, "core_member": None, "catalog_record": record,
        })
    return core, catalog, joined


def _validate_core_shape(core: Mapping[str, Any], package_root: Path) -> None:
    groups = core.get("groups")
    if not _is_sequence(groups) or not groups:
        raise BindingDeliveryError("delivered core bundle has no groups")
    seen = set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise BindingDeliveryError("delivered core group is malformed")
        for member in group.get("members", []):
            if not isinstance(member, Mapping):
                raise BindingDeliveryError("delivered core member is malformed")
            key = (str(group.get("group_id")), str(member.get("member_id")))
            if key in seen:
                raise BindingDeliveryError(f"duplicate delivered core key: {key!r}")
            seen.add(key)
            facts = member.get("facts_path")
            if not isinstance(facts, str) or Path(facts).is_absolute():
                raise BindingDeliveryError(f"delivered core facts path is not relative: {facts!r}")
            facts_path = (package_root / "core" / facts).resolve()
            if not _inside(facts_path, package_root) or not facts_path.is_file():
                raise BindingDeliveryError(f"delivered core facts path escapes/missing: {facts!r}")
            for kind in ("video_path", "audio_path"):
                media = member.get("media", {}).get(kind)
                if not isinstance(media, str) or Path(media).is_absolute():
                    raise BindingDeliveryError(f"delivered core {kind} path is not relative: {media!r}")


def _validate_catalog_shape(catalog: Mapping[str, Any], root: Path) -> None:
    catalog_root = root / "catalog"
    for record in catalog.get("records", []):
        for key in ("facts_path", "questions_path"):
            value = record.get(key)
            if not isinstance(value, str) or Path(value).is_absolute():
                raise BindingDeliveryError(f"delivered catalog {key} is not relative: {value!r}")
            resolved = (catalog_root / value).resolve()
            if not _inside(resolved, root) or not resolved.is_file():
                raise BindingDeliveryError(f"delivered catalog {key} escapes/missing: {value}")
        ids = record.get("public_question_ids")
        if not _is_sequence(ids):
            raise BindingDeliveryError("delivered catalog public IDs are not a list")


def _validate_media_links(
    core: Mapping[str, Any] | None,
    core_root: Path,
    catalog_public: Mapping[str, Any],
    catalog_root: Path,
) -> dict[str, Any]:
    cache: dict[str, dict[str, Any]] = {}
    count = 0
    for group in (core or {}).get("groups", []):
        for member in group.get("members", []):
            for kind in ("video_path", "audio_path"):
                path = (core_root / member["media"][kind]).resolve()
                if not _inside(path, core_root) or not path.is_file():
                    raise BindingDeliveryError(f"delivered core media escapes/missing: {path}")
                _media_probe(path, cache)
                count += 1
    public_count = 0
    for sample in catalog_public.get("samples", []):
        media = sample.get("media", {})
        for kind in ("video_path", "audio_path"):
            value = media.get(kind)
            if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
                raise BindingDeliveryError(f"delivered public media path is not relative: {value!r}")
            path = (catalog_root / value).resolve()
            if not _inside(path, catalog_root) or not path.is_file():
                raise BindingDeliveryError(f"delivered public media escapes/missing: {path}")
            _media_probe(path, cache)
            public_count += 1
    return {
        "status": "pass",
        "core_media_references": count if core is not None else "not_applicable",
        "catalog_public_media_references": public_count,
        "unique_media_files": len(cache),
        "ffprobe": list(cache.values()),
    }


def _apply_catalog_sampling(
    facts: Mapping[str, Any],
    request_config: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(facts))
    sampling = deepcopy(result.get("sampling") or {})
    override = request_config.get("qa_sampling") or {}
    if not isinstance(override, Mapping):
        raise BindingDeliveryError("catalog request_config.qa_sampling is malformed")
    sampling.update(override)
    sampling["qa_sampling"] = {
        **(sampling.get("qa_sampling") or {}),
        **dict(override),
    }
    result["sampling"] = sampling
    return result


def _validate_qa_regeneration(root: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    config = _load(root / "catalog/request_config.json", label="delivered catalog request_config")
    seed = str(config.get("seed"))
    items_per_type = int(config.get("items_per_type", 1))
    regenerated = 0
    item_count = 0
    metadata_mismatch = 0
    qa_ids_seen: set[str] = set()
    for record in catalog.get("records", []):
        record_config = record.get("generation_config") or config
        if not isinstance(record_config, Mapping):
            raise BindingDeliveryError("record generation_config must be an object")
        seed = str(record_config.get("seed"))
        items_per_type = int(record_config.get("items_per_type", 1))
        # A core-group member is seeded by group/member; an ordinary Episode has
        # neither, so it is seeded by its own episode_id. The label follows the
        # same rule so a failure names the record that actually failed.
        if record.get("record_kind") == "episode":
            record_seed = f"{seed}:{record['episode_id']}"
            label = f"episode {record['episode_id']}"
        else:
            record_seed = f"{seed}:{record['group_id']}:{record['member_id']}"
            label = f"{record['group_id']}/{record['member_id']}"
        catalog_root = root / "catalog"
        facts_path = (catalog_root / record["facts_path"]).resolve()
        questions_path = (catalog_root / record["questions_path"]).resolve()
        facts = _load(facts_path, label="delivered facts")
        expected = _load(questions_path, label="delivered private questions")
        prepared = _apply_catalog_sampling(facts, record_config)
        actual = whole_degree_display(
            generate_unified_questions(
                prepared,
                qa_ids=QA_IDS,
                items_per_type=items_per_type,
                seed=record_seed,
            )
        )
        # Compare the persisted JSON representation: Python tuples in measured
        # windows are JSON arrays on disk. All keys and values still participate.
        actual_payload = json.loads(json.dumps(actual))
        expected_payload = dict(expected)
        actual_payload.pop("input_facts", None)
        expected_payload.pop("input_facts", None)
        if actual_payload != expected_payload:
            raise BindingDeliveryError(f"25-QA regeneration differs for {label}")
        if actual.get("input_facts") != expected.get("input_facts"):
            metadata_mismatch += 1
        items = list(iter_unified_items(expected))
        public_ids = record.get("public_question_ids")
        if len(items) != len(public_ids):
            raise BindingDeliveryError(
                f"public/private question alignment differs for {label}"
            )
        qa_ids_seen.update(str(item.get("qa_id")) for item in items)
        item_count += len(items)
        regenerated += 1
    return {
        "status": "pass",
        "records_regenerated": regenerated,
        "items_regenerated": item_count,
        "qa_ids_seen": sorted(qa_ids_seen),
        "input_facts_metadata_mismatch": metadata_mismatch,
        "input_facts_note": (
            "The current normalizer may represent visibility frame keys as integers; "
            "question items, IDs, forms, deferred rows and angle followups were compared exactly."
        ),
        "source": "copied_facts_only",
        "original_staging_required": False,
    }


def _validate_catalog_scoring(root: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    rows_by_form: dict[str, list[dict[str, Any]]] = {form: [] for form in _FORMS}
    for record in catalog.get("records", []):
        questions = _load(root / "catalog" / record["questions_path"], label="delivered private questions")
        items = list(iter_unified_items(questions))
        for public_id, item in zip(record["public_question_ids"], items):
            for form in _FORMS:
                if form in item.get("forms", {}):
                    rows_by_form[form].append(
                        {
                            "question_id": public_id,
                            "prediction": _gold_prediction(item, form),
                        }
                    )
    results = {}
    for form, predictions in rows_by_form.items():
        result = score_binding_catalog(
            root / "catalog/catalog_index.json",
            predictions,
            form=form,
        )
        expected = result["counts"]["form_available"]
        if result["counts"]["correct"] != expected or result["counts"]["invalid"] or result["counts"]["missing"]:
            raise BindingDeliveryError(f"delivered catalog {form} GT scorer check failed")
        results[form] = {
            "status": "pass",
            "form_available": expected,
            "scored": result["counts"]["scored"],
            "correct": result["counts"]["correct"],
            "invalid": result["counts"]["invalid"],
            "missing": result["counts"]["missing"],
        }
    return {
        "status": "pass",
        "forms": results,
        "claim_boundary": "scorer test using private gold; not model evaluation",
    }


def _validate_core_scoring(root: Path, core: Mapping[str, Any]) -> dict[str, Any]:
    forms = {}
    for form in _FORMS:
        answers = {}
        for group in core.get("groups", []):
            for member in group.get("members", []):
                question = member.get("question")
                if isinstance(question, Mapping) and form in question.get("forms", {}):
                    answers[str(member["sample_id"])] = _gold_prediction(question, form)
        try:
            result = score_binding_groups(core, answers, form=form)
        except (BindingGroupScoreError, ValueError) as error:
            raise BindingDeliveryError(f"delivered core {form} scorer check failed: {error}") from error
        forms[form] = {
            "status": "pass",
            "members_total": result["counts"]["members_total"],
            "members_form_available": result["counts"]["members_form_available"],
            "members_scored": result["counts"]["members_scored"],
            "members_correct": result["counts"]["members_correct"],
            "members_missing": result["counts"]["members_missing"],
            "members_invalid": result["counts"]["members_invalid"],
        }
    return {
        "status": "pass",
        "forms": forms,
        "claim_boundary": "scorer test using private gold; not model evaluation",
    }


def _validate_private_path_links(root: Path, catalog: Mapping[str, Any], core: Mapping[str, Any]) -> None:
    _validate_core_shape(core, root)
    _validate_catalog_shape(catalog, root)
    path_map = _load(root / "provenance/path_map.json", label="delivery path map")
    entries = path_map.get("entries")
    if not isinstance(entries, list):
        raise BindingDeliveryError("delivery path map has no entries")
    for entry in entries:
        value = entry.get("delivered_path")
        if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
            raise BindingDeliveryError(f"path map delivered path is not relative: {value!r}")
        if not (root / value).resolve().is_file():
            raise BindingDeliveryError(f"path map delivered path is missing: {value}")
    external = _load(
        root / "provenance/external_dependencies.json",
        label="external dependency map",
    )
    if not isinstance(external.get("dependencies"), list):
        raise BindingDeliveryError("external dependency map has no dependencies")


def validate_binding_delivery(
    output: str | Path,
    *,
    check_media: bool = True,
) -> dict[str, Any]:
    """Validate a delivered package without consulting source staging."""
    root = Path(output).expanduser().resolve()
    manifest = _load(root / "manifest.json", label="delivery manifest")
    if manifest.get("schema") != DELIVERY_SCHEMA:
        raise BindingDeliveryError("delivery manifest schema is invalid")
    if manifest.get("status") != "intermediate_delivery":
        raise BindingDeliveryError("delivery manifest status is not intermediate_delivery")
    core_file = root / "core/binding_groups.json"
    core = _load(core_file, label="delivered core bundle") if core_file.is_file() else None
    catalog = _load(root / "catalog/catalog_index.json", label="delivered catalog index")
    catalog_public = _load(root / "catalog/model_inputs.json", label="delivered catalog model_inputs")
    if core is not None:
        _validate_private_path_links(root, catalog, core)
    media = (
        _validate_media_links(core, root / "core", catalog_public, root / "catalog")
        if check_media
        else {"status": "not_run"}
    )
    regeneration = _validate_qa_regeneration(root, catalog)
    catalog_scoring = _validate_catalog_scoring(root, catalog)
    core_scoring = (
        _validate_core_scoring(root, core)
        if core is not None
        else {"status": "not_applicable", "reason": "episode delivery carries no core group"}
    )
    result = {
        "schema": _VALIDATION_SCHEMA,
        "status": "pass",
        "delivery_status": manifest.get("status"),
        "delivery_kind": manifest.get("delivery_kind"),
        "join_key": manifest.get("join_key"),
        "catalog_regeneration": regeneration,
        "catalog_scoring": catalog_scoring,
        "core_scoring": core_scoring,
        "media_playback": media,
        "claim_boundary": (
            "Portable-package scorer and regeneration checks only. This does not "
            "run UE, Habitat, RLR, a model, human calibration, or formal admission."
        ),
    }
    _write(root / "validation/readback.json", result, refuse_existing=False)
    return result


def export_binding_delivery(
    core_bundle: str | Path | None,
    catalog_index: str | Path,
    output: str | Path,
    *,
    validate: bool = True,
    check_media: bool = True,
) -> dict[str, Any]:
    """Copy a fresh delivery and optionally validate it.

    ``core_bundle`` may be ``None``, which delivers ordinary Episode catalog
    records without requiring an accepted four-member core group first.
    """
    core_path = None if core_bundle is None else Path(core_bundle).expanduser().resolve()
    catalog_path = Path(catalog_index).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise BindingDeliveryError(f"refusing existing delivery output: {destination}")
    core, catalog, joined = _join_inputs(core_path, catalog_path)

    destination.mkdir(parents=True)
    if core is not None:
        (destination / "core/media").mkdir(parents=True)
        (destination / "core/groups").mkdir(parents=True)
    (destination / "catalog/media").mkdir(parents=True)
    (destination / "catalog/questions").mkdir(parents=True)
    (destination / "native/facts").mkdir(parents=True)
    (destination / "native/readbacks").mkdir(parents=True)
    (destination / "provenance").mkdir(parents=True)
    (destination / "validation").mkdir(parents=True)

    path_entries: dict[str, dict[str, Any]] = {}
    fact_sources: set[Path] = set()
    readback_sources: set[Path] = set()
    facts_cache: dict[str, Mapping[str, Any]] = {}
    external: dict[str, dict[str, Any]] = {}

    for index, row in enumerate(joined):
        for role in ("core_facts_path", "catalog_facts_path"):
            if row[role] is None:
                continue
            base = catalog_path.parent
            if role.startswith("core"):
                assert core_path is not None
                base = core_path.parent
            source = _resolve(row[role], base)
            fact_sources.add(source)
            row[f"{role}_resolved"] = str(source)
    for source in sorted(fact_sources):
        facts = _load(source, label="source facts")
        facts_cache[str(source)] = facts
        for key, value in (facts.get("source_paths") or {}).items():
            source_path = _source_path(value, source.parent)
            if source_path is None:
                continue
            if source_path.suffix.casefold() == ".json":
                readback_sources.add(source_path)
                if key == "plan":
                    plan = _load(source_path, label="source episode plan")
                    request = plan.get("request")
                    identity = request.get("binding_identity") if isinstance(request, Mapping) else None
                    if isinstance(identity, Mapping):
                        query = _source_path(identity.get("native_polyline_query_path"), source_path.parent)
                        if query is not None:
                            if query.suffix.casefold() != ".json" or not query.is_file():
                                raise BindingDeliveryError(f"native motion query receipt is missing: {query}")
                            readback_sources.add(query)
        audio = facts.get("audio")
        if isinstance(audio, Mapping):
            for key in ("path", "actual_path"):
                source_path = _source_path(audio.get(key), source.parent)
                if source_path is not None and source_path.suffix.casefold() == ".json":
                    readback_sources.add(source_path)

    sorted_facts = sorted(fact_sources)
    for index, source in enumerate(sorted_facts, start=1):
        _add_reference(
            path_entries,
            source,
            kind="facts",
            reference=f"facts:{index}",
            delivered_path=f"native/facts/facts_{index:04d}.json",
        )
    fact_keys = set(path_entries)
    sorted_readbacks = sorted(source for source in readback_sources if str(source) not in fact_keys)
    for index, source in enumerate(sorted_readbacks, start=1):
        _add_reference(
            path_entries,
            source,
            kind="native_json_readback",
            reference=f"readback:{index}",
            delivered_path=f"native/readbacks/readback_{index:04d}_{source.name}",
        )

    core_media_map: dict[str, str] = {}
    for media in sorted((core_path.parent / "media").glob("*")) if core_path else []:
        if media.is_file():
            _add_reference(path_entries, media.resolve(), kind="core_final_media", reference="core.media", delivered_path=f"core/media/{media.name}")
            core_media_map[str(media.resolve())] = path_entries[str(media.resolve())]["delivered_path"]
    catalog_media_map: dict[str, str] = {}
    for media in sorted((catalog_path.parent / "media").glob("*")):
        if media.is_file():
            _add_reference(path_entries, media.resolve(), kind="catalog_public_media", reference="catalog.media", delivered_path=f"catalog/media/{media.name}")
            catalog_media_map[str(media.resolve())] = path_entries[str(media.resolve())]["delivered_path"]

    question_map: dict[str, str] = {}
    for record in catalog.get("records", []):
        source = _resolve(record["questions_path"], catalog_path.parent)
        destination_rel = f"catalog/questions/{source.name}"
        _add_reference(path_entries, source, kind="private_questions", reference=f"catalog.questions:{record['group_id']}/{record['member_id']}", delivered_path=destination_rel)
        question_map[str(source)] = destination_rel

    # Copy file payloads after all destinations are known.
    for source, row in path_entries.items():
        destination_file = destination / row["delivered_path"]
        _copy(Path(source), destination_file)

    # Build path-rewritten private facts.
    for source in sorted_facts:
        # Keep facts byte/structure exact so regeneration preserves input_facts,
        # question IDs, forms and display text. Their source_paths remain
        # provenance/external dependency fields and are listed separately.
        _write(
            destination / path_entries[str(source)]["delivered_path"],
            deepcopy(dict(facts_cache[str(source)])),
            refuse_existing=False,
        )

    # Core bundle and its per-group private convenience files.
    def rewrite_core_document(value: Mapping[str, Any]) -> dict[str, Any]:
        assert core_path is not None
        return _rewrite_paths(
            deepcopy(dict(value)),
            path_entries,
            destination_base=Path("core"),
            source_base=core_path.parent,
        )

    if core is not None and core_path is not None:
        rewritten_core = rewrite_core_document(core)
        _write(destination / "core/binding_groups.json", rewritten_core)
        core_public = _load(
            core_path.parent / "model_inputs.json", label="core public model_inputs"
        )
        core_public = _rewrite_paths(
            core_public,
            path_entries,
            destination_base=Path("core"),
            source_base=core_path.parent,
        )
        _write(destination / "core/model_inputs.json", core_public)
        for source_file in sorted((core_path.parent / "groups").glob("*.json")):
            group_doc = _load(source_file, label="core per-group private JSON")
            _write(
                destination / "core/groups" / source_file.name,
                rewrite_core_document(group_doc),
            )

    # Catalog index and public model inputs.
    rewritten_catalog = _rewrite_paths(
        deepcopy(dict(catalog)),
        path_entries,
        destination_base=Path("catalog"),
        source_base=catalog_path.parent,
    )
    _write(destination / "catalog/catalog_index.json", rewritten_catalog)
    public_catalog = _load(catalog_path.parent / "model_inputs.json", label="catalog public model_inputs")
    public_catalog = _rewrite_paths(
        public_catalog,
        path_entries,
        destination_base=Path("catalog"),
        source_base=catalog_path.parent,
    )
    _write(destination / "catalog/model_inputs.json", public_catalog)
    request_config = _load(catalog_path.parent / "request_config.json", label="catalog request_config")
    _write(destination / "catalog/request_config.json", request_config)

    # Rewritten direct readback JSONs, retaining external strings as provenance.
    for source in sorted_readbacks:
        value = _load(source, label="source JSON readback")
        _write(
            destination / path_entries[str(source)]["delivered_path"],
            _rewrite_paths(
                value,
                path_entries,
                destination_base=Path(path_entries[str(source)]["delivered_path"]).parent,
                source_base=source.parent,
            ),
            refuse_existing=False,
        )

    if core is not None:
        _collect_path_values(
            core,
            reference="core_bundle",
            path_map=path_entries,
            external=external,
        )
    _collect_path_values(
        catalog,
        reference="catalog_index",
        path_map=path_entries,
        external=external,
    )
    for source in sorted_facts:
        _collect_path_values(
            facts_cache[str(source)],
            reference=f"facts:{source}",
            path_map=path_entries,
            external=external,
        )
    for source in sorted_readbacks:
        value = _load(source, label="source JSON readback")
        _collect_path_values(
            value,
            reference=f"readback:{source}",
            path_map=path_entries,
            external=external,
        )
    for row in path_entries.values():
        row["references"] = sorted(row["references"])
    for row in external.values():
        row["references"] = sorted(row["references"])
    path_map_doc = {
        "schema": _PATH_MAP_SCHEMA,
        "status": "pass",
        "entries": sorted(path_entries.values(), key=lambda row: row["source_path"]),
        "claim_boundary": "ordinary source-to-delivered path mapping; no content hash",
    }
    external_doc = {
        "schema": _EXTERNAL_SCHEMA,
        "status": "external_dependencies_required",
        "dependencies": sorted(external.values(), key=lambda row: row["path"]),
        "claim_boundary": "external generation/runtime inputs are listed, not copied or claimed portable",
    }
    join_key = ["group_id", "member_id"] if core is not None else ["sample_id"]

    def delivered_questions_path(row: Mapping[str, Any]) -> str:
        for record in rewritten_catalog["records"]:
            if core is not None:
                if (
                    record["group_id"] == row["group_id"]
                    and record["member_id"] == row["member_id"]
                ):
                    return str(record["questions_path"])
            elif record["sample_id"] == row["catalog_sample_id"]:
                return str(record["questions_path"])
        raise BindingDeliveryError(
            f"delivered catalog has no record for join row {row['catalog_sample_id']!r}"
        )

    join_doc = {
        "schema": _JOIN_SCHEMA,
        "status": "pass",
        "join_key": join_key,
        "records": [
            {
                "group_id": row["group_id"],
                "member_id": row["member_id"],
                "episode_id": row.get("episode_id"),
                "core_sample_id": row["core_sample_id"],
                "catalog_sample_id": row["catalog_sample_id"],
                "catalog_questions_path": delivered_questions_path(row),
                "public_question_ids": row["catalog_public_question_ids"],
            }
            for row in joined
        ],
    }
    provenance_doc = {
        "schema": "avengine_binding_delivery_provenance_v1",
        "status": "pass",
        "source_core_bundle": None if core_path is None else str(core_path),
        "source_catalog_index": str(catalog_path),
        "join_key": join_key,
        "delivery_kind": "binding_group_intermediate" if core is not None else "episode_catalog",
        "accepted_group_count": len(core.get("groups", [])) if core is not None else 0,
        "accepted_core_sample_count": (
            sum(len(group.get("members", [])) for group in core.get("groups", []))
            if core is not None
            else 0
        ),
        "catalog_question_count": catalog.get("catalog_question_count"),
        "public_id_policy": "catalog public_question_ids preserved exactly; core sample IDs preserved exactly",
        "path_map": "provenance/path_map.json",
        "external_dependencies": "provenance/external_dependencies.json",
        "join_index": "provenance/join_index.json",
    }
    _write(destination / "provenance/path_map.json", path_map_doc)
    _write(destination / "provenance/external_dependencies.json", external_doc)
    _write(destination / "provenance/join_index.json", join_doc)
    _write(destination / "provenance/provenance.json", provenance_doc)

    manifest = {
        "schema": DELIVERY_SCHEMA,
        "status": "intermediate_delivery",
        "qualification_claim": False,
        "delivery_kind": "binding_group_intermediate" if core is not None else "episode_catalog",
        "join_key": join_key,
        "counts": {
            "group_count": core.get("group_count") if core is not None else 0,
            "world_count": (
                core.get("world_count") if core is not None else catalog.get("world_count")
            ),
            "core_sample_count": core.get("sample_count") if core is not None else 0,
            "catalog_sample_count": catalog.get("av_sample_count"),
            "catalog_question_count": catalog.get("catalog_question_count"),
            "catalog_form_counts": catalog.get("form_counts"),
        },
        "package": {
            "core_binding_groups": "core/binding_groups.json" if core is not None else None,
            "core_model_inputs": "core/model_inputs.json" if core is not None else None,
            "catalog_index": "catalog/catalog_index.json",
            "catalog_model_inputs": "catalog/model_inputs.json",
            "catalog_request_config": "catalog/request_config.json",
            "facts_root": "native/facts",
            "readbacks_root": "native/readbacks",
            "path_map": "provenance/path_map.json",
            "external_dependencies": "provenance/external_dependencies.json",
            "join_index": "provenance/join_index.json",
            "validation": "validation/readback.json",
        },
        "external_runtime_boundary": {
            "ue": "external_dependency",
            "habitat": "external_dependency",
            "rlr": "external_dependency",
            "hrtf": "external_dependency",
            "shared_sound_assets": "external_dependency",
            "portable_full_renderer": False,
        },
        "model_evaluation": "not_run",
        "human_answerability": "not_run",
        "claim_boundary": (
            "Intermediate portable delivery of accepted research-candidate groups. "
            "Copied facts support QA regeneration and scorer checks; external runtime "
            "and generation dependencies remain required for native rerender."
        ),
    }
    _write(destination / "manifest.json", manifest)
    validation = None
    if validate:
        validation = validate_binding_delivery(destination, check_media=check_media)
    return {
        "status": manifest["status"],
        "output": str(destination),
        "joined_members": len(joined),
        "copied_files": len(path_entries),
        "external_dependencies": len(external),
        "validation": validation,
    }


__all__ = [
    "BindingDeliveryError",
    "DELIVERY_SCHEMA",
    "export_binding_delivery",
    "validate_binding_delivery",
]



# ---------------------------------------------------------------------------
# Self-contained V1 dataset index (public projection + private gold index)
# ---------------------------------------------------------------------------

DATASET_INDEX_SCHEMA = "avengine_qa_dataset_index_v1"
PRIVATE_INDEX_SCHEMA = "avengine_qa_dataset_private_index_v1"

# Every knob here is configuration, not an algorithm constant. A caller that
# needs a different split, layout set or observation protocol passes it in.
DEFAULT_DATASET_INDEX_CONFIG: dict[str, Any] = {
    "dataset_version": "v1",
    "audio_layouts": ["binaural"],
    "video_views": {"preview": "video"},
    "split_policy": {
        "mode": "by_world_key",
        "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "seed": "avengine-v1-split",
    },
    "observation_protocols": {
        "full_episode_av": {
            "audio": "full_episode",
            "video": "full_episode",
            "query_time_display": "whole_seconds",
            "angle_display": "whole_degrees",
        }
    },
    "world_key_prefix": "world_",
    "world_key_seed": "avengine-v1-world-key",
    "static_index": True,
}


def _dataset_index_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_DATASET_INDEX_CONFIG)
    for key, value in dict(config or {}).items():
        if key not in merged:
            raise BindingDeliveryError(f"unknown dataset index configuration key: {key!r}")
        merged[key] = deepcopy(value)
    layouts = [str(value) for value in merged["audio_layouts"]]
    if not layouts or len(set(layouts)) != len(layouts):
        raise BindingDeliveryError(f"audio_layouts must be a non-empty unique list: {layouts!r}")
    merged["audio_layouts"] = layouts
    policy = merged["split_policy"]
    if not isinstance(policy, Mapping) or policy.get("mode") not in {"by_world_key", "single"}:
        raise BindingDeliveryError("split_policy.mode must be by_world_key or single")
    if policy["mode"] == "by_world_key":
        fractions = policy.get("fractions")
        if not isinstance(fractions, Mapping) or not fractions:
            raise BindingDeliveryError("split_policy.fractions is required for by_world_key")
        total = sum(float(value) for value in fractions.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise BindingDeliveryError(f"split_policy.fractions must sum to 1.0, got {total}")
    return merged


class _MediaPayloadIndex:
    """Collapse byte-identical delivered media onto one canonical reference.

    A visual episode that the producer copied once per audio variant arrives in
    the package as two files with different names and identical bytes. Pointing
    the dataset index at both would count and ship one world twice, so the
    index references the canonical copy and reports the redundant files.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._canonical: dict[str, str] = {}
        self._by_size: dict[int, list[tuple[str, bytes]]] = {}
        self.redundant: dict[str, str] = {}

    def canonical(self, relative: str) -> str:
        if relative in self._canonical:
            return self._canonical[relative]
        path = self._root / relative
        if not path.is_file():
            raise BindingDeliveryError(f"delivered media is absent: {relative}")
        size = path.stat().st_size
        payload = path.read_bytes()
        for known, known_payload in self._by_size.setdefault(size, []):
            if known_payload == payload:
                self._canonical[relative] = known
                if known != relative:
                    self.redundant[relative] = known
                return known
        self._by_size[size].append((relative, payload))
        self._canonical[relative] = relative
        return relative

    def report(self) -> dict[str, Any]:
        return {
            "distinct_payloads": sum(len(rows) for rows in self._by_size.values()),
            "referenced_files": len(self._canonical),
            "redundant_delivered_files": len(self.redundant),
            "redundant_to_canonical": dict(sorted(self.redundant.items())),
            "note": (
                "Redundant files stay on disk as retained package content; the index "
                "references one canonical payload per distinct world/media so a world "
                "is exported and counted once."
            ),
        }


def _root_relative(value: Any, *, base: str) -> str:
    """Rebase one delivered reference from a package subdirectory to the root.

    The catalog projection stores its media relative to ``catalog/``, so the
    dataset index has to carry the root-relative form a portable reader can
    resolve from the export root alone.
    """

    relative = _portable_relative(str(value))
    return (Path(base) / relative).as_posix()


def _layout_declaration(layout: str) -> dict[str, Any]:
    from avengine.timeline.current_mp3d_dynamic_audio import layout_output_contract

    return dict(layout_output_contract(layout))


def _delivered_readback(root: Path, source_absolute: str) -> Path | None:
    """Map one recorded producer path to its delivered copy inside the export."""

    path_map = _load(root / "provenance/path_map.json", label="delivery path map")
    wanted = str(Path(source_absolute))
    wanted_resolved = str(Path(source_absolute).resolve(strict=False))
    for entry in path_map.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        source = str(entry.get("source_path"))
        if source in {wanted, wanted_resolved}:
            return root / str(entry["delivered_path"])
    return None


def _member_audio_identity(root: Path, facts: Mapping[str, Any]) -> dict[str, Any]:
    """Read this member's own delivered audio-program identity.

    The identity comes from the member's delivered research report, which
    already records program_id, revision, program_content_sha256 and the
    timeline. Two audio variants of the same episode share program_id but not
    program_content_sha256, so this is what distinguishes them.
    """

    report_source = (facts.get("source_paths") or {}).get("research_report")
    if not isinstance(report_source, str) or not report_source:
        raise BindingDeliveryError("member facts declare no research_report source path")
    delivered = _delivered_readback(root, report_source)
    if delivered is None or not delivered.is_file():
        raise BindingDeliveryError(
            f"member research report is not delivered inside the export: {report_source}"
        )
    report = _load(delivered, label="delivered member research report")
    metadata = report.get("audio_program_metadata") or report.get("audio_program")
    if not isinstance(metadata, Mapping):
        raise BindingDeliveryError("delivered research report has no audio_program metadata")
    audio = facts.get("audio") or {}
    return {
        "program_id": metadata.get("program_id"),
        "revision": metadata.get("revision"),
        "program_content_sha256": metadata.get("program_content_sha256"),
        "timeline": deepcopy(dict(metadata.get("timeline") or {})),
        "episode_id": facts.get("episode_id"),
        "sample_count": audio.get("sample_count"),
        "sample_rate_hz": audio.get("sample_rate_hz"),
        "delivered_research_report": str(delivered.relative_to(root)),
    }


def attach_audio_layout(
    output: str | Path,
    *,
    sample_id: str,
    layout: str,
    receipt: str | Path,
    mixture: str | Path | None = None,
) -> dict[str, Any]:
    """Attach one extra audio layout view to an already delivered sample.

    An attached view is another audio track of the *same* sample. It is not a
    new world, a new episode or a new question, and it is accepted only when
    the rendering receipt's own audio-program identity matches this member's
    delivered identity exactly. A receipt from a sibling audio variant of the
    same episode is refused, because program_content_sha256 differs.
    """

    root = Path(output).expanduser().resolve()
    manifest = _load(root / "manifest.json", label="delivery manifest")
    if manifest.get("schema") != DELIVERY_SCHEMA:
        raise BindingDeliveryError("delivery manifest schema is invalid")
    catalog = _load(root / "catalog/catalog_index.json", label="delivered catalog index")
    records = [
        record
        for record in catalog.get("records") or []
        if isinstance(record, Mapping) and str(record.get("sample_id")) == str(sample_id)
    ]
    if len(records) != 1:
        raise BindingDeliveryError(f"sample_id {sample_id!r} is not a unique delivered record")
    record = records[0]
    facts = _load(root / "catalog" / str(record["facts_path"]), label="delivered member facts")
    expected = _member_audio_identity(root, facts)

    receipt_path = Path(receipt).expanduser().resolve()
    receipt_doc = _load(receipt_path, label="audio layout receipt")
    program = receipt_doc.get("audio_program_record") or receipt_doc.get("audio_program_metadata")
    if not isinstance(program, Mapping):
        raise BindingDeliveryError("audio layout receipt has no audio_program_record")
    mismatches = []
    for key in ("program_id", "revision", "program_content_sha256"):
        if program.get(key) != expected[key]:
            mismatches.append({"field": key, "member": expected[key], "receipt": program.get(key)})
    if dict(program.get("timeline") or {}) != expected["timeline"]:
        mismatches.append(
            {"field": "timeline", "member": expected["timeline"], "receipt": program.get("timeline")}
        )
    if mismatches:
        raise BindingDeliveryError(
            "audio layout receipt is not this member's own render; refusing to attach it as "
            f"{layout!r} for {sample_id}: {mismatches!r}"
        )

    delivery = ((receipt_doc.get("audio") or {}).get("layout_delivery") or {}).get(layout)
    if not isinstance(delivery, Mapping):
        raise BindingDeliveryError(
            f"receipt declares no layout_delivery for {layout!r}; a consumer must read the "
            "declared channel order, normalization and coordinate frame, so an undeclared "
            "layout is refused rather than assumed"
        )
    declared = {
        "layout_type": layout,
        "layout_id": delivery.get("layout_id"),
        "channel_count": delivery.get("channel_count"),
        "channel_labels": list(delivery.get("channel_labels") or []),
        "channel_order": delivery.get("channel_order"),
        "normalization": delivery.get("normalization"),
        "coordinate_frame": delivery.get("coordinate_frame"),
        "sample_rate_hz": (delivery.get("mixture") or {}).get("sample_rate_hz")
        or expected["sample_rate_hz"],
    }
    missing = [key for key, value in declared.items() if value in (None, "", [])]
    if missing:
        raise BindingDeliveryError(
            f"receipt layout declaration for {layout!r} is incomplete: missing {missing!r}"
        )
    runtime = _layout_declaration(layout)
    conflicts = {
        key: {"receipt": declared[key], "runtime_contract": runtime[key]}
        for key in ("layout_id", "channel_count", "channel_order", "normalization",
                    "coordinate_frame")
        if declared[key] != runtime[key]
    }
    if conflicts:
        raise BindingDeliveryError(
            f"receipt layout declaration disagrees with the runtime layout contract: {conflicts!r}"
        )

    if mixture is None:
        mixture_declared = (delivery.get("mixture") or {}).get("path")
        if not isinstance(mixture_declared, str) or not mixture_declared:
            raise BindingDeliveryError("receipt declares no mixture path for this layout")
        mixture_path = Path(mixture_declared)
    else:
        mixture_path = Path(mixture).expanduser()
    mixture_path = mixture_path.resolve()
    if not mixture_path.is_file():
        raise BindingDeliveryError(f"declared layout mixture is absent: {mixture_path}")

    relative = f"attachments/{sample_id}/{layout}/mixture.wav"
    destination = root / relative
    _copy(mixture_path, destination)
    attachment = {
        "schema": "avengine_binding_delivery_audio_attachment_v1",
        "status": "pass",
        "sample_id": str(sample_id),
        "layout": layout,
        "attached_view_of": "binaural",
        "delivered_path": relative,
        "declaration": declared,
        "member_identity": expected,
        "receipt_identity": {
            "program_id": program.get("program_id"),
            "revision": program.get("revision"),
            "program_content_sha256": program.get("program_content_sha256"),
            "timeline": deepcopy(dict(program.get("timeline") or {})),
        },
        "source_receipt": str(receipt_path),
        "source_mixture": str(mixture_path),
        "claim_boundary": (
            "One extra audio layout view of the same delivered sample. It is not a "
            "separate world, episode or question and is never counted as one."
        ),
    }
    _write(root / f"attachments/{sample_id}/{layout}/attachment.json", attachment)
    return attachment


def _world_keys(world_ids: Sequence[str], *, prefix: str, seed: str) -> dict[str, str]:
    """Assign one opaque public key per engine world id."""

    ordered = sorted({str(value) for value in world_ids})
    keys = [f"{prefix}{index + 1:04d}" for index in range(len(ordered))]
    random.Random(f"{seed}:world-keys").shuffle(keys)
    return dict(zip(ordered, keys, strict=True))


def _assign_splits(world_keys: Sequence[str], policy: Mapping[str, Any]) -> dict[str, str]:
    """Assign a split per world key so one world never straddles two splits."""

    ordered = sorted({str(value) for value in world_keys})
    if policy["mode"] == "single":
        name = str(policy.get("name", "all"))
        return {key: name for key in ordered}
    shuffled = list(ordered)
    random.Random(f"{policy.get('seed', 'avengine-v1-split')}:splits").shuffle(shuffled)
    fractions = {str(key): float(value) for key, value in policy["fractions"].items()}
    names = sorted(fractions)
    result: dict[str, str] = {}
    total = len(shuffled)
    start = 0
    for index, name in enumerate(names):
        count = total - start if index == len(names) - 1 else int(round(fractions[name] * total))
        for key in shuffled[start : start + count]:
            result[key] = name
        start += count
    for key in shuffled[start:]:
        result[key] = names[-1]
    return result


def build_dataset_index(
    output: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the self-contained public dataset index and private gold index.

    The public index carries only what a model may see: media per selected
    layout, question text per form, public calibration, an opaque world key and
    a split. Engine world/group/member identity, private question IDs and gold
    answers go to the private index instead.
    """

    root = Path(output).expanduser().resolve()
    settings = _dataset_index_config(config)
    manifest = _load(root / "manifest.json", label="delivery manifest")
    if manifest.get("schema") != DELIVERY_SCHEMA:
        raise BindingDeliveryError("delivery manifest schema is invalid")
    catalog = _load(root / "catalog/catalog_index.json", label="delivered catalog index")
    public_catalog = _load(root / "catalog/model_inputs.json", label="delivered catalog projection")
    public_by_sample = {
        str(sample.get("sample_id")): sample
        for sample in public_catalog.get("samples") or []
        if isinstance(sample, Mapping)
    }
    records = [record for record in catalog.get("records") or [] if isinstance(record, Mapping)]
    if not records:
        raise BindingDeliveryError("delivered catalog index has no records")

    layout_declarations: dict[str, dict[str, Any]] = {}
    for layout in settings["audio_layouts"]:
        layout_declarations[layout] = _layout_declaration(layout)

    attachments: dict[tuple[str, str], Mapping[str, Any]] = {}
    attachments_root = root / "attachments"
    if attachments_root.is_dir():
        for path in sorted(attachments_root.glob("*/*/attachment.json")):
            attachment = _load(path, label="delivered audio attachment")
            key = (str(attachment["sample_id"]), str(attachment["layout"]))
            attachments[key] = attachment
            declared = dict(attachment["declaration"])
            existing = layout_declarations.get(declared["layout_type"])
            if existing is None:
                layout_declarations[declared["layout_type"]] = declared
            elif {key: existing[key] for key in declared if key in existing} != {
                key: declared[key] for key in declared if key in existing
            }:
                raise BindingDeliveryError(
                    f"attached {declared['layout_type']!r} declaration conflicts with the "
                    f"index declaration for the same layout"
                )

    media_index = _MediaPayloadIndex(root)
    world_key_by_id = _world_keys(
        [str(record.get("world_id")) for record in records],
        prefix=str(settings["world_key_prefix"]),
        seed=str(settings["world_key_seed"]),
    )
    split_by_world_key = _assign_splits(list(world_key_by_id.values()), settings["split_policy"])

    samples: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []
    splits: dict[str, list[str]] = {}
    form_counts: dict[str, int] = {form: 0 for form in _FORMS}
    kind_counts = {"main": 0, "angle_followup": 0}
    qa_counts: dict[str, int] = {}

    for record in sorted(records, key=lambda row: str(row.get("sample_id"))):
        sample_id = str(record["sample_id"])
        projection = public_by_sample.get(sample_id)
        if projection is None:
            raise BindingDeliveryError(f"delivered projection has no sample {sample_id!r}")
        question_set = _load(
            root / "catalog" / str(record["questions_path"]), label="delivered question set"
        )
        private_items = list(iter_unified_items(question_set))
        public_items = list(projection.get("items") or [])
        if len(private_items) != len(public_items):
            raise BindingDeliveryError(
                f"sample {sample_id} projects {len(public_items)} public questions for "
                f"{len(private_items)} private questions"
            )
        angle_ids = {
            str(item.get("question_id"))
            for item in question_set.get("angle_followups") or []
            if isinstance(item, Mapping)
        }
        question_rows: list[dict[str, Any]] = []
        question_id_map: dict[str, str] = {}
        for private_item, public_item in zip(private_items, public_items, strict=True):
            if str(private_item.get("qa_id")) != str(public_item.get("qa_id")):
                raise BindingDeliveryError(
                    f"sample {sample_id} public/private question order disagrees"
                )
            if private_item.get("status") != "pass":
                continue
            public_id = str(public_item["question_id"])
            private_id = str(private_item["question_id"])
            forms = sorted(
                form
                for form in _FORMS
                if form in (private_item.get("forms") or {})
                and (private_item.get("form_status") or {}).get(form, {}).get("status") == "pass"
            )
            if not forms:
                continue
            prompt = {
                form: deepcopy(dict((private_item.get("model_input") or {})[form]))
                for form in forms
                if form in (private_item.get("model_input") or {})
            }
            if sorted(prompt) != forms:
                raise BindingDeliveryError(
                    f"sample {sample_id} question {public_id} lacks a public prompt for "
                    f"{sorted(set(forms) - set(prompt))!r}"
                )
            for form in prompt.values():
                for option in form.get("options") or []:
                    option.pop("allow_value", None)
                    option.pop("value", None)
            kind = "angle_followup" if private_id in angle_ids else "main"
            question_rows.append(
                {
                    "question_id": public_id,
                    "qa_id": str(private_item["qa_id"]),
                    "kind": kind,
                    "forms": forms,
                    "prompt": prompt,
                    "required_modalities": deepcopy(public_item.get("required_modalities")),
                }
            )
            question_id_map[public_id] = private_id
            kind_counts[kind] += 1
            for form in forms:
                form_counts[form] += 1
            if kind == "main":
                qa_counts[str(private_item["qa_id"])] = qa_counts.get(
                    str(private_item["qa_id"]), 0
                ) + 1

        facts = _load(root / "catalog" / str(record["facts_path"]), label="delivered member facts")
        calibration = dict(facts.get("camera_calibration") or {})
        if calibration and calibration.get("public") is not True:
            raise BindingDeliveryError(
                f"sample {sample_id} camera calibration is not marked public; refusing to "
                "project it into model input"
            )
        # An absent calibration is a reportable gap in the source facts, not a
        # privacy failure. Keep the sample readable and count the gap instead of
        # silently emitting an empty calibration that reads as complete.
        calibration_status = "public" if calibration else "absent_in_source_facts"
        media_public = dict(projection.get("media") or {})
        audio_tracks: dict[str, Any] = {}
        for layout in settings["audio_layouts"]:
            if layout == "binaural":
                audio_tracks[layout] = {
                    "path": media_index.canonical(
                        _root_relative(media_public["audio_path"], base="catalog")
                    )
                }
            else:
                attachment = attachments.get((sample_id, layout))
                if attachment is None:
                    continue
                audio_tracks[layout] = {
                    "path": media_index.canonical(
                        str(attachment["delivered_path"])
                    ),
                    "attached_view_of": str(attachment["attached_view_of"]),
                    "attachment_evidence": {
                        "declaration": deepcopy(dict(attachment["declaration"])),
                        "verified_member_identity": True,
                    },
                }
        for (attached_sample, layout), attachment in attachments.items():
            if attached_sample == sample_id and layout not in audio_tracks:
                audio_tracks[layout] = {
                    "path": media_index.canonical(
                        str(attachment["delivered_path"])
                    ),
                    "attached_view_of": str(attachment["attached_view_of"]),
                    "attachment_evidence": {
                        "declaration": deepcopy(dict(attachment["declaration"])),
                        "verified_member_identity": True,
                    },
                }
        world_key = world_key_by_id[str(record["world_id"])]
        split = split_by_world_key[world_key]
        samples.append(
            {
                "sample_id": sample_id,
                "record_kind": str(record.get("record_kind", "core_group_member")),
                "room_family": str(record.get("room_family")),
                "split": split,
                "calibration": {
                    **{key: value for key, value in calibration.items() if key != "public"},
                    "status": calibration_status,
                    "time": deepcopy(dict(facts.get("time") or {})),
                },
                "media": {
                    "video": {
                        view: media_index.canonical(
                            _root_relative(media_public[f"{field}_path"], base="catalog")
                        )
                        for view, field in settings["video_views"].items()
                        if f"{field}_path" in media_public
                    },
                    "audio": audio_tracks,
                },
                "questions": question_rows,
            }
        )
        splits.setdefault(split, []).append(sample_id)
        private_records.append(
            {
                "sample_id": sample_id,
                "record_kind": str(record.get("record_kind", "core_group_member")),
                "group_id": record.get("group_id"),
                "member_id": record.get("member_id"),
                "world_id": record.get("world_id"),
                "world_key": world_key,
                "episode_id": facts.get("episode_id"),
                "core_sample_id": record.get("core_sample_id"),
                "core_task": record.get("core_task"),
                "questions_path": f"catalog/{record['questions_path']}",
                "facts_path": str((root / "catalog" / str(record["facts_path"])).resolve().relative_to(root)),
                "question_id_map": question_id_map,
                "deferred_count": len(record.get("deferred") or []),
            }
        )
    return _finalize_dataset_index(
        root,
        settings=settings,
        catalog=catalog,
        samples=samples,
        private_records=private_records,
        splits=splits,
        layout_declarations=layout_declarations,
        form_counts=form_counts,
        kind_counts=kind_counts,
        qa_counts=qa_counts,
        attachments=attachments,
        media_index=media_index,
    )


def _finalize_dataset_index(
    root: Path,
    *,
    settings: Mapping[str, Any],
    catalog: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    private_records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[str]],
    layout_declarations: Mapping[str, Mapping[str, Any]],
    form_counts: Mapping[str, int],
    kind_counts: Mapping[str, int],
    qa_counts: Mapping[str, int],
    attachments: Mapping[tuple[str, str], Mapping[str, Any]],
    media_index: "_MediaPayloadIndex",
) -> dict[str, Any]:
    world_keys = {str(record["world_key"]) for record in private_records}
    record_kinds: dict[str, int] = {}
    for sample in samples:
        kind = str(sample["record_kind"])
        record_kinds[kind] = record_kinds.get(kind, 0) + 1
    index = {
        "schema": DATASET_INDEX_SCHEMA,
        "status": "pass",
        "dataset_version": str(settings["dataset_version"]),
        "qualification_claim": False,
        "requested_qa_ids": list(QA_IDS),
        "audio_layouts": {
            layout: dict(declaration) for layout, declaration in sorted(layout_declarations.items())
        },
        "observation_protocols": deepcopy(dict(settings["observation_protocols"])),
        "counts": {
            "sample_count": len(samples),
            "world_count": len(world_keys),
            "record_kind_counts": dict(sorted(record_kinds.items())),
            "valid_main_question_count": int(kind_counts.get("main", 0)),
            "valid_angle_followup_count": int(kind_counts.get("angle_followup", 0)),
            "form_counts": dict(sorted(form_counts.items())),
            "audio_attachment_count": len(attachments),
            "samples_without_source_calibration": sum(
                1
                for sample in samples
                if (sample.get("calibration") or {}).get("status") != "public"
            ),
        },
        "media_payload_accounting": media_index.report(),
        "public_payload": {
            "note": (
                "A public-only distribution needs exactly these paths. Everything "
                "else in the package -- catalog/questions, catalog/catalog_index.json, "
                "native, provenance, validation and private -- carries gold, engine "
                "identity or provenance and must not be shipped to a model."
            ),
            "paths": sorted(
                {"manifest.json", "public/dataset_index.json"}
                | ({"public/index.html"} if (root / "public/index.html").is_file() else set())
                | {
                    str(value)
                    for sample in samples
                    for value in (sample.get("media") or {}).get("video", {}).values()
                }
                | {
                    str(track["path"] if isinstance(track, Mapping) else track)
                    for sample in samples
                    for track in ((sample.get("media") or {}).get("audio") or {}).values()
                }
            ),
        },
        "splits": {name: sorted(ids) for name, ids in sorted(splits.items())},
        "samples": list(samples),
        "counting_note": (
            "One question item with both mcq and open forms is one question; form_counts "
            "is a breakdown of the same items, not an extra total. Angle follow-ups are "
            "counted separately from main questions. An attached audio layout view is "
            "another track of the same sample, never another world or question."
        ),
        "model_evaluation": "not_run",
        "human_answerability": "not_run",
        "claim_boundary": (
            "Public projection of a self-contained V1 export: selected audio layout, "
            "video, question text and public calibration only. No gold answer, engine "
            "identifier or intervention record is present."
        ),
    }
    private_index = {
        "schema": PRIVATE_INDEX_SCHEMA,
        "status": "pass",
        "dataset_version": str(settings["dataset_version"]),
        "world_key_policy": {
            "prefix": str(settings["world_key_prefix"]),
            "seed": str(settings["world_key_seed"]),
            "note": "opaque private key per engine world id; sample grouping stays in this private index",
        },
        "split_policy": deepcopy(dict(settings["split_policy"])),
        "core_bundle": (
            "core/binding_groups.json" if (root / "core/binding_groups.json").is_file() else None
        ),
        "valid_main_question_count_by_qa": dict(sorted(qa_counts.items())),
        "catalog_counting_note": catalog.get("counting_note"),
        "records": list(private_records),
        "claim_boundary": (
            "Training labels and private grouping. Never reachable from the public "
            "model-input projection."
        ),
    }
    _write(root / "public/dataset_index.json", index, refuse_existing=False)
    _write(root / "private/gold_index.json", private_index, refuse_existing=False)
    static = None
    if settings["static_index"]:
        static = str(write_static_index(root).relative_to(root))
    return {
        "status": "pass",
        "output": str(root),
        "public_index": "public/dataset_index.json",
        "private_index": "private/gold_index.json",
        "static_index": static,
        "counts": index["counts"],
        "media_payload_accounting": {
            key: value
            for key, value in media_index.report().items()
            if key != "redundant_to_canonical"
        },
        "splits": {name: len(ids) for name, ids in index["splits"].items()},
        "audio_layouts": sorted(index["audio_layouts"]),
    }


def write_static_index(output: str | Path) -> Path:
    """Write a small dependency-free HTML index beside the public index.

    This reuses the delivered public projection. It starts no server and
    publishes nothing; it is a local file a reviewer can open from the export
    root itself.
    """

    root = Path(output).expanduser().resolve()
    index = _load(root / "public/dataset_index.json", label="public dataset index")
    rows = []
    for sample in index.get("samples") or []:
        media = sample.get("media") or {}
        videos = media.get("video") or {}
        audios = media.get("audio") or {}
        video_cells = " ".join(
            f'<a href="../{value}">{view}</a>' for view, value in sorted(videos.items())
        )
        audio_cells = " ".join(
            f'<a href="../{track["path"] if isinstance(track, Mapping) else track}">{layout}</a>'
            for layout, track in sorted(audios.items())
        )
        main = sum(1 for row in sample.get("questions") or [] if row.get("kind") == "main")
        angle = sum(
            1 for row in sample.get("questions") or [] if row.get("kind") == "angle_followup"
        )
        rows.append(
            "<tr>"
            f"<td>{sample.get('sample_id')}</td>"
            f"<td>{sample.get('record_kind')}</td>"
            f"<td>{sample.get('room_family')}</td>"
            f"<td>{sample.get('split')}</td>"
            f"<td>{main}</td><td>{angle}</td>"
            f"<td>{video_cells}</td><td>{audio_cells}</td>"
            "</tr>"
        )
    counts = index.get("counts") or {}
    layouts = ", ".join(
        f"{layout} ({declaration.get('channel_count')}ch, {declaration.get('channel_order')}, "
        f"{declaration.get('normalization')}, {declaration.get('coordinate_frame')})"
        for layout, declaration in sorted((index.get("audio_layouts") or {}).items())
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>AVEngine V1 QA dataset index</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
th {{ background: #f4f4f4; }}
</style>
<h1>AVEngine V1 QA dataset index</h1>
<p>Version {index.get('dataset_version')} &middot;
{counts.get('sample_count')} samples &middot;
{counts.get('world_count')} worlds &middot;
{counts.get('valid_main_question_count')} valid main questions &middot;
{counts.get('valid_angle_followup_count')} angle follow-ups</p>
<p>Audio layouts: {layouts}</p>
<p>{index.get('counting_note')}</p>
<table>
<tr><th>sample</th><th>kind</th><th>room</th><th>split</th>
<th>main</th><th>angle</th><th>video</th><th>audio</th></tr>
{chr(10).join(rows)}
</table>
<p>{index.get('claim_boundary')}</p>
"""
    path = root / "public/index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


__all__ += [
    "DATASET_INDEX_SCHEMA",
    "DEFAULT_DATASET_INDEX_CONFIG",
    "PRIVATE_INDEX_SCHEMA",
    "attach_audio_layout",
    "build_dataset_index",
    "write_static_index",
]
