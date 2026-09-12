"""Derive the full QA catalog for every member of validated binding groups.

The ordinary catalog and the paired core questions have separate counts.
Existing native facts and shared media are reused; no engine render is needed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import random
import shutil

from avengine.qa.binding_groups import _load, _path, _write, BindingGroupError
from avengine.qa.unified_catalog import (
    QA_IDS, generate_unified_questions, iter_unified_items, model_input_questions,
)


def whole_degree_display(value):
    """Change display strings only; numeric evidence keeps its full precision."""
    if isinstance(value, dict):
        return {key: whole_degree_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [whole_degree_display(item) for item in value]
    if not isinstance(value, str):
        return value

    def rounded(match):
        number = float(match.group(1))
        result = math.floor(number + 0.5) if number >= 0 else math.ceil(number - 0.5)
        return str(result)

    return re.sub(r"(-?\d+\.\d+)(?=\s*(?:°|degrees?\b|deg\b|度))", rounded, value)


EPISODE_SPEC_SCHEMA = "avengine_qa_episode_catalog_request_v1"


class _MediaCopier:
    """Copy public media into a catalog, storing one copy per distinct payload.

    Keying only on the producer path exports the same visual episode twice when
    the producer wrote one copy per audio variant. Two byte-identical inputs are
    one payload, so they get one delivered file and one world.
    """

    def __init__(self, output, media_ids):
        self._output = Path(output)
        self._media_ids = list(media_ids)
        self._by_path = {}
        self._by_payload = {}
        self.reused_by_payload = {}

    def copy(self, original, kind):
        original = Path(original).resolve()
        key = (kind, str(original))
        if key in self._by_path:
            return self._by_path[key]
        if not original.is_file():
            raise BindingGroupError(f"public {kind} media is absent: {original}")
        payload = original.read_bytes()
        payload_key = (kind, len(payload), payload)
        existing = self._by_payload.get(payload_key)
        if existing is not None:
            self._by_path[key] = existing
            self.reused_by_payload[str(original)] = existing
            return existing
        index = len(self._by_payload)
        if index >= len(self._media_ids):
            raise BindingGroupError("ran out of media identifiers for this catalog")
        relative = Path("media") / f"{kind}_{self._media_ids[index]:06d}{original.suffix}"
        destination = self._output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise BindingGroupError(f"refusing to overwrite catalog media: {destination}")
        shutil.copyfile(original, destination)
        self._by_payload[payload_key] = str(relative)
        self._by_path[key] = str(relative)
        return str(relative)

    def report(self):
        return {
            "distinct_payloads": len(self._by_payload),
            "input_references": len(self._by_path),
            "inputs_reused_by_payload": len(self.reused_by_payload),
            "note": (
                "One delivered file per distinct media payload. A producer copy of the "
                "same visual episode per audio variant is one payload, not two worlds."
            ),
        }


def derive_binding_catalog(bundle_paths, *, output, qa_sampling, items_per_type=1, seed="binding-catalog"):
    """Generate every catalog type, recording inapplicability instead of padding."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing catalog output: {output}")
    if int(items_per_type) < 1:
        raise ValueError("items_per_type must be positive")
    sampling_override = deepcopy(dict(qa_sampling))
    if sampling_override.get("time_display_precision", 0) != 0:
        raise ValueError("this catalog export requires whole-second public query times")
    sampling_override["time_display_precision"] = 0
    bundles = [(Path(path).resolve(), _load(Path(path).resolve())) for path in bundle_paths]
    if not bundles or any(bundle.get("validation") != "media_checked" for _, bundle in bundles):
        raise BindingGroupError("full catalog requires completed media-checked binding groups")
    keys = [(group["group_id"], member["member_id"])
            for _, bundle in bundles for group in bundle["groups"] for member in group["members"]]
    if len(set(keys)) != len(keys):
        raise BindingGroupError("duplicate group/member supplied for full catalog")
    sample_ids = [f"sample_{index + 1:06d}" for index in range(len(keys))]
    random.Random(seed+":samples").shuffle(sample_ids)
    media_ids = list(range(len(keys)*2))
    random.Random(seed+":media").shuffle(media_ids)
    output.mkdir(parents=True)
    _write(output/"request_config.json", {"qa_ids": list(QA_IDS), "qa_sampling": sampling_override,
            "items_per_type": int(items_per_type), "seed": seed})
    copier = _MediaCopier(output, media_ids)
    public, records = [], []
    counts, forms = Counter(), Counter()
    next_question = 0
    for source, bundle in bundles:
        for group in bundle["groups"]:
            for member in group["members"]:
                exported_clock = member.get("media_clock", {}).get("exported", {})
                if exported_clock.get("status") != "pass" or exported_clock.get("observation_cutoff_s") is not None:
                    raise BindingGroupError("full catalog requires full-duration media, not a core question prefix")
                index = len(records)
                sample_id = sample_ids[index]
                facts_path = _path(member["facts_path"], source.parent)
                facts = _load(facts_path)
                sampling = deepcopy(facts.get("sampling") or {})
                sampling.update(sampling_override)
                sampling["qa_sampling"] = {
                    **sampling.get("qa_sampling", {}), **sampling_override}
                facts["sampling"] = sampling
                questions = generate_unified_questions(
                    facts, qa_ids=QA_IDS, items_per_type=int(items_per_type),
                    seed=seed+":"+group["group_id"]+":"+member["member_id"])
                questions = whole_degree_display(questions)
                items = list(iter_unified_items(questions))
                private_path = output/"questions"/f"{sample_id}.json"
                _write(private_path, questions)
                media = {
                    kind + "_path": copier.copy(
                        _path(member["media"][kind + "_path"], source.parent), kind
                    )
                    for kind in ("video", "audio")
                }
                projection = model_input_questions(questions)
                for item in projection["items"]:
                    next_question += 1
                    item["question_id"] = f"question_{next_question:06d}"
                public.append({"sample_id": sample_id, "media": media, "items": projection["items"]})
                local_counts = Counter(item["qa_id"] for item in items)
                local_forms = Counter(form for item in items for form in item["forms"])
                counts.update(local_counts)
                forms.update(local_forms)
                records.append({"sample_id": sample_id, "record_kind": "core_group_member",
                    "group_id": group["group_id"], "core_task": group["task_family"],
                    "world_id": group["world_id"], "core_sample_id": member["sample_id"],
                    "member_id": member["member_id"], "room_family": group["room_family"],
                    "facts_path": str(facts_path), "questions_path": str(private_path.relative_to(output)),
                    "catalog_question_count": len(items), "core_question_count": 1,
                    "generated_by_qa": dict(local_counts), "forms": dict(local_forms),
                    "deferred": deepcopy(questions.get("deferred", []))})
    question_ids = [f"question_{index + 1:06d}" for index in range(next_question)]
    random.Random(seed+":questions").shuffle(question_ids)
    offset = 0
    for record, sample in zip(records, public, strict=True):
        for item in sample["items"]:
            item["question_id"] = question_ids[offset]
            offset += 1
        record["public_question_ids"] = [item["question_id"] for item in sample["items"]]
    random.Random(seed+":public-order").shuffle(public)
    result = {"status": "research_candidate", "av_sample_count": len(records),
        "group_count": len({record["group_id"] for record in records}),
        "world_count": len({record["world_id"] for record in records}),
        "catalog_question_count": sum(counts.values()), "core_question_count": len(records),
        "generated_by_qa": {qa_id: counts[qa_id] for qa_id in QA_IDS},
        "form_counts": dict(forms), "requested_qa_ids": list(QA_IDS), "records": records,
        "media_payload_accounting": copier.report(),
        "counting_note": "Core paired questions and ordinary catalog questions are separate; do not add their counts as semantically distinct questions.",
        "model_evaluation": "not_run", "human_answerability": "not_run",
        "media": "copied from validated public group media without rerendering"}
    _write(output/"catalog_index.json", result)
    _write(output/"model_inputs.json", {"schema": "avengine_qa_public_questions_v1",
                                      "samples": public, "question_count": sum(counts.values())})
    return result


def _episode_specs(specs):
    """Normalise the ordinary-Episode catalog request."""

    if isinstance(specs, (str, Path)):
        document = _load(Path(specs).resolve())
        if document.get("schema") != EPISODE_SPEC_SCHEMA:
            raise BindingGroupError(
                f"episode catalog request schema must be {EPISODE_SPEC_SCHEMA}"
            )
        base = Path(specs).resolve().parent
        rows = document.get("episodes")
    else:
        base = Path.cwd()
        rows = list(specs)
    if not isinstance(rows, list) or not rows:
        raise BindingGroupError("episode catalog request has no episodes")
    # Shape and identity are checked before any path is resolved, so a request
    # with a duplicate episode_id reports that rather than a missing file.
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise BindingGroupError("episode entry is not an object")
        episode_id = str(row.get("episode_id") or "")
        if not episode_id:
            raise BindingGroupError("episode entry is missing ['episode_id']")
        if episode_id in seen:
            raise BindingGroupError(f"duplicate episode_id in request: {episode_id!r}")
        seen.add(episode_id)
    normalised = []
    for row in rows:
        missing = [
            key
            for key in ("episode_id", "facts_path", "room_family", "world_id", "media")
            if not row.get(key)
        ]
        if missing:
            raise BindingGroupError(f"episode entry is missing {missing!r}")
        media = row["media"]
        if not isinstance(media, dict) or not media.get("video_path") or not media.get("audio_path"):
            raise BindingGroupError(
                f"episode {row['episode_id']!r} needs media.video_path and media.audio_path"
            )
        normalised.append(
            {
                "episode_id": str(row["episode_id"]),
                "facts_path": _path(row["facts_path"], base),
                "room_family": str(row["room_family"]),
                "world_id": str(row["world_id"]),
                "video_path": _path(media["video_path"], base),
                "audio_path": _path(media["audio_path"], base),
            }
        )
    return normalised


def derive_episode_catalog(
    episode_specs,
    *,
    output,
    qa_sampling,
    items_per_type=1,
    seed="episode-catalog",
):
    """Derive the full QA catalog for ordinary Episodes, without binding groups.

    An accepted four-member core group is one route to a catalog, not the only
    one. Any Episode whose facts pass the unified generator can carry the whole
    QA-01..25 catalog, so an ordinary Episode does not have to be promoted into
    a core group before its questions exist. Core-group relation scoring stays
    a separate, additional capability for the groups that do have one.
    """

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing catalog output: {output}")
    if int(items_per_type) < 1:
        raise ValueError("items_per_type must be positive")
    sampling_override = deepcopy(dict(qa_sampling))
    if sampling_override.get("time_display_precision", 0) != 0:
        raise ValueError("this catalog export requires whole-second public query times")
    sampling_override["time_display_precision"] = 0
    episodes = _episode_specs(episode_specs)

    sample_ids = [f"sample_{index + 1:06d}" for index in range(len(episodes))]
    random.Random(seed + ":samples").shuffle(sample_ids)
    media_ids = list(range(len(episodes) * 2))
    random.Random(seed + ":media").shuffle(media_ids)
    output.mkdir(parents=True)
    _write(
        output / "request_config.json",
        {
            "qa_ids": list(QA_IDS),
            "qa_sampling": sampling_override,
            "items_per_type": int(items_per_type),
            "seed": seed,
            "record_kind": "episode",
        },
    )
    copier = _MediaCopier(output, media_ids)
    public, records = [], []
    counts, forms = Counter(), Counter()
    next_question = 0
    for index, episode in enumerate(episodes):
        sample_id = sample_ids[index]
        facts_path = episode["facts_path"]
        facts = _load(facts_path)
        if facts.get("schema") != "avengine_qa_unified_episode_facts_v1":
            raise BindingGroupError(
                f"episode {episode['episode_id']!r} facts schema is not "
                "avengine_qa_unified_episode_facts_v1"
            )
        if facts.get("status") != "pass":
            raise BindingGroupError(
                f"episode {episode['episode_id']!r} facts status is "
                f"{facts.get('status')!r}, not pass"
            )
        sampling = deepcopy(facts.get("sampling") or {})
        sampling.update(sampling_override)
        sampling["qa_sampling"] = {**sampling.get("qa_sampling", {}), **sampling_override}
        facts["sampling"] = sampling
        questions = generate_unified_questions(
            facts,
            qa_ids=QA_IDS,
            items_per_type=int(items_per_type),
            seed=seed + ":" + episode["episode_id"],
        )
        questions = whole_degree_display(questions)
        items = list(iter_unified_items(questions))
        private_path = output / "questions" / f"{sample_id}.json"
        _write(private_path, questions)
        media = {
            "video_path": copier.copy(episode["video_path"], "video"),
            "audio_path": copier.copy(episode["audio_path"], "audio"),
        }
        projection = model_input_questions(questions)
        for item in projection["items"]:
            next_question += 1
            item["question_id"] = f"question_{next_question:06d}"
        public.append({"sample_id": sample_id, "media": media, "items": projection["items"]})
        local_counts = Counter(item["qa_id"] for item in items)
        local_forms = Counter(form for item in items for form in item["forms"])
        counts.update(local_counts)
        forms.update(local_forms)
        records.append(
            {
                "sample_id": sample_id,
                "record_kind": "episode",
                "group_id": None,
                "member_id": None,
                "episode_id": episode["episode_id"],
                "world_id": episode["world_id"],
                "core_sample_id": None,
                "room_family": episode["room_family"],
                "facts_path": str(facts_path),
                "questions_path": str(private_path.relative_to(output)),
                "catalog_question_count": len(items),
                "core_question_count": 0,
                "generated_by_qa": dict(local_counts),
                "forms": dict(local_forms),
                "deferred": deepcopy(questions.get("deferred", [])),
            }
        )
    question_ids = [f"question_{index + 1:06d}" for index in range(next_question)]
    random.Random(seed + ":questions").shuffle(question_ids)
    offset = 0
    for record, sample in zip(records, public, strict=True):
        for item in sample["items"]:
            item["question_id"] = question_ids[offset]
            offset += 1
        record["public_question_ids"] = [item["question_id"] for item in sample["items"]]
    random.Random(seed + ":public-order").shuffle(public)
    result = {
        "status": "research_candidate",
        "record_kind": "episode",
        "av_sample_count": len(records),
        "group_count": 0,
        "world_count": len({record["world_id"] for record in records}),
        "catalog_question_count": sum(counts.values()),
        "core_question_count": 0,
        "generated_by_qa": {qa_id: counts[qa_id] for qa_id in QA_IDS},
        "form_counts": dict(forms),
        "requested_qa_ids": list(QA_IDS),
        "records": records,
        "media_payload_accounting": copier.report(),
        "counting_note": (
            "Ordinary Episode catalog. These samples carry no core-group relation "
            "questions, so core_question_count is zero rather than one per sample."
        ),
        "model_evaluation": "not_run",
        "human_answerability": "not_run",
        "media": "copied from the selected Episode public media without rerendering",
    }
    _write(output / "catalog_index.json", result)
    _write(
        output / "model_inputs.json",
        {
            "schema": "avengine_qa_public_questions_v1",
            "samples": public,
            "question_count": sum(counts.values()),
        },
    )
    return result


__all__ = [
    "EPISODE_SPEC_SCHEMA",
    "derive_binding_catalog",
    "derive_episode_catalog",
    "whole_degree_display",
]


def merge_binding_catalogs(catalog_index_paths, *, output, seed="merged-catalog"):
    """Merge existing core/ordinary catalogs without changing private questions.

    Only public sample/question/media identifiers are reassigned. Each record
    keeps its actual source generation config, so the normal export validator
    still regenerates its questions exactly from the copied facts.
    """
    if isinstance(catalog_index_paths, (str, Path)):
        raise BindingGroupError("catalog_index_paths must be a sequence of catalog paths")
    catalog_index_paths = tuple(catalog_index_paths)
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing merged catalog: {destination}")
    inputs = []
    seen_records, seen_facts = set(), set()
    for value in catalog_index_paths:
        path = Path(value).expanduser().resolve()
        catalog = _load(path)
        if catalog.get("status") != "research_candidate":
            raise BindingGroupError(f"input catalog is not research_candidate: {path}")
        config = _load(path.parent / "request_config.json")
        public = _load(path.parent / "model_inputs.json")
        samples = public.get("samples") or []
        by_sample = {row["sample_id"]: row for row in samples}
        if len(by_sample) != len(samples):
            raise BindingGroupError(f"duplicate public sample IDs in {path}")
        records = catalog.get("records") or []
        if not records or len({row["sample_id"] for row in records}) != len(records):
            raise BindingGroupError(f"empty or duplicate catalog records in {path}")
        if {row["sample_id"] for row in records} != set(by_sample):
            raise BindingGroupError(f"public/catalog sample IDs differ in {path}")
        for row in records:
            kind = row.get("record_kind", "core_group_member")
            if kind not in {"core_group_member", "episode"}:
                raise BindingGroupError(f"unsupported record kind: {kind}")
            key = ((kind, row.get("episode_id")) if kind == "episode"
                   else (kind, row.get("group_id"), row.get("member_id")))
            facts = _path(row["facts_path"], path.parent).resolve()
            if key in seen_records or str(facts) in seen_facts:
                raise BindingGroupError(f"duplicate source record/facts across catalogs: {key}")
            seen_records.add(key); seen_facts.add(str(facts))
            if not isinstance(row.get("world_id"), str) or not row["world_id"].strip():
                raise BindingGroupError(f"record {key} lacks its world identity")
            sample = by_sample[row["sample_id"]]
            ids = [item["question_id"] for item in sample.get("items") or []]
            if ids != list(row.get("public_question_ids") or []):
                raise BindingGroupError(f"public question order differs for {key}")
            questions = _load(_path(row["questions_path"], path.parent))
            if len(list(iter_unified_items(questions))) != len(ids):
                raise BindingGroupError(f"public/private question count differs for {key}")
            actual_config = row.get("generation_config") or config
            if not isinstance(actual_config, dict) or not isinstance(actual_config.get("seed"), str):
                raise BindingGroupError(f"record {key} lacks its actual generation seed")
            inputs.append((path, row, sample, questions, facts, deepcopy(actual_config)))
    if not inputs:
        raise BindingGroupError("merged catalog needs source records")
    destination.mkdir(parents=True)
    sample_ids = [f"sample_{i+1:06d}" for i in range(len(inputs))]
    random.Random(str(seed) + ":samples").shuffle(sample_ids)
    total_questions = sum(len(item[2].get("items") or []) for item in inputs)
    question_ids = [f"question_{i+1:06d}" for i in range(total_questions)]
    random.Random(str(seed) + ":questions").shuffle(question_ids)
    media_ids = list(range(2 * len(inputs)))
    random.Random(str(seed) + ":media").shuffle(media_ids)
    copier = _MediaCopier(destination, media_ids)
    records, public, counts, forms = [], [], Counter(), Counter()
    offset = 0
    for sample_id, (path, row, sample, questions, facts, config) in zip(sample_ids, inputs):
        record = deepcopy(dict(row)); record["sample_id"] = sample_id
        record["facts_path"] = str(facts)
        record["questions_path"] = f"questions/{sample_id}.json"
        record["generation_config"] = config
        _write(destination / record["questions_path"], questions)
        items = deepcopy(list(sample.get("items") or []))
        for item in items:
            item["question_id"] = question_ids[offset]; offset += 1
        record["public_question_ids"] = [item["question_id"] for item in items]
        media = {kind + "_path": copier.copy(
            _path(sample["media"][kind + "_path"], path.parent), kind
        ) for kind in ("video", "audio")}
        public.append({"sample_id": sample_id, "media": media, "items": items})
        actual_items = list(iter_unified_items(questions))
        local_counts = Counter(item["qa_id"] for item in actual_items)
        local_forms = Counter(form for item in actual_items for form in item["forms"])
        record["catalog_question_count"] = len(actual_items)
        record["generated_by_qa"] = dict(local_counts)
        record["forms"] = dict(local_forms)
        counts.update(local_counts); forms.update(local_forms); records.append(record)
    random.Random(str(seed) + ":public-order").shuffle(public)
    config = {"qa_ids": list(QA_IDS), "qa_sampling": {}, "items_per_type": 1,
              "seed": str(seed), "per_record_generation_config": True,
              "source_catalogs": [str(Path(value).expanduser().resolve()) for value in catalog_index_paths]}
    result = {"status": "research_candidate", "records": records,
              "av_sample_count": len(records),
              "group_count": len({r["group_id"] for r in records if r.get("group_id")}),
              "world_count": len({r["world_id"] for r in records}),
              "catalog_question_count": sum(counts.values()),
              "core_question_count": sum(int(r.get("core_question_count", 0)) for r in records),
              "generated_by_qa": {q: counts[q] for q in QA_IDS}, "form_counts": dict(forms),
              "requested_qa_ids": list(QA_IDS), "media_payload_accounting": copier.report(),
              "counting_note": "Core/ordinary records share one index; forms and angle follow-ups are not extra main questions; original private question content is unchanged.",
              "model_evaluation": "not_run", "human_answerability": "not_run"}
    _write(destination / "request_config.json", config)
    _write(destination / "catalog_index.json", result)
    _write(destination / "model_inputs.json", {"schema": "avengine_qa_public_questions_v1",
           "samples": public, "question_count": total_questions})
    return result
