"""Export corrected forms and optional audio variants without changing retained banks."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil

from avengine.qa.binding_bank_merge import _publish_media
from avengine.qa.choice_support import apply_choice_support
from avengine.qa.prior_audit import read_jsonl, write_prior_receipt


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _rows(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def curate_bank(bank_in, output, *, audio_summary=None, split_reference=None, seed=0):
    """Retain all valid questions; priors are reported rather than hidden by pruning."""
    bank, output = Path(bank_in).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("public", "private", "media"):
        (output/folder).mkdir()
    public = read_jsonl(bank/"public/questions.jsonl")
    answers = read_jsonl(bank/"private/answers.jsonl")
    sources = read_jsonl(bank/"private/sources.jsonl")
    by_id = {s["question_id"]: s for s in sources}
    public_by_id = {p["question_id"]: p for p in public}
    if len(by_id) != len(sources) or set(by_id) != set(public_by_id) or set(by_id) != {a["question_id"] for a in answers}:
        raise ValueError("bank question/source/answer identities differ")
    for path in (bank/"media").iterdir():
        if path.is_file():
            # Copy immutable media; metadata is always independently written.
            shutil.copyfile(path, output/"media"/path.name)
    for path in (bank/"private").iterdir():
        if path.is_file() and path.name not in {"answers.jsonl", "sources.jsonl", "answer_priors.json", "splits.jsonl", "strata.jsonl"}:
            shutil.copyfile(path, output/"private"/path.name)
    facts_cache = {}
    for answer in answers:
        if "forms" not in answer:
            raise ValueError("curation needs full generator forms; remerge compact binding exports first")
        item = {**deepcopy(answer), "model_input": deepcopy(public_by_id[answer["question_id"]]["forms"])}
        if answer["qa_id"] == "QA-21" and "observed_sound_classes" not in item.get("evidence", {}):
            path = by_id[answer["question_id"]]["facts_path"]
            if path not in facts_cache:
                facts_cache[path] = json.loads(Path(path).read_text())
            item.setdefault("evidence", {})["observed_sound_classes"] = sorted({e["sound_class"] for e in facts_cache[path]["events"]})
        item = apply_choice_support(item)
        answer.update({k: item[k] for k in ("truth", "forms", "evidence") if k in item})
        public_by_id[answer["question_id"]]["forms"] = item["model_input"]
    worlds_by_facts = {s["facts_path"]: s.get("world_id") or s["episode_id"] for s in sources}
    world_splits = {}
    reference = Path(split_reference).resolve() if split_reference else bank
    split_file = reference/"private/splits.jsonl"
    if split_file.is_file():
        ref_sources = {s["question_id"]: s for s in read_jsonl(reference/"private/sources.jsonl")}
        for row in read_jsonl(split_file):
            source = ref_sources[row["question_id"]]
            world = source.get("world_id") or source["episode_id"]
            if source["facts_path"] in worlds_by_facts and worlds_by_facts[source["facts_path"]] != world:
                raise ValueError("reference and current bank disagree on visual world identity")
            worlds_by_facts[source["facts_path"]] = world
            split = row["split"]
            if world in world_splits and world_splits[world] != split:
                raise ValueError("reference splits put one visual world in multiple splits")
            if split not in {"train", "valid", "test"}:
                raise ValueError("invalid reference split")
            world_splits[world] = split
    added = 0
    if audio_summary:
        summary = json.loads(Path(audio_summary).read_text())
        if summary.get("status") not in {"completed", "completed_with_deficits"}:
            raise ValueError("cannot curate an incomplete audio plan")
        for record in summary["jobs"]:
            root = Path(record["output"])
            job = record["job"]
            facts_path = root/"delivery/facts.json"
            facts = json.loads(facts_path.read_text())
            source_facts = str(Path(job["source"])/"delivery/facts.json")
            world = worlds_by_facts.get(source_facts, job["world_id"])
            if job.get("split"):
                if world in world_splits and world_splits[world] != job["split"]:
                    raise ValueError("audio variant split differs from retained visual world")
                world_splits[world] = job["split"]
            paths = facts["source_paths"]
            media = {"video": _publish_media(Path(paths["video"]), output/"media"),
                     "audio": _publish_media(Path(paths.get("mixture_audio") or paths["audio_readback"]), output/"media")}
            for item in json.loads((root/"targeted_questions.json").read_text())["items"]:
                qid = f"audio_question_{added+1:06d}"
                if qid in by_id:
                    raise ValueError("audio question ID already exists; use the unaugmented base bank")
                public.append({"question_id": qid, "qa_id": item["qa_id"], "forms": item["model_input"],
                               "required_modalities": item.get("required_modalities"), "media": media})
                answers.append({"question_id": qid, "source_question_id": item["question_id"], "qa_id": item["qa_id"],
                                "truth": item["truth"], "forms": item["forms"], "evidence": item["evidence"], "research_only": True})
                original = next((s for s in sources if s["facts_path"] == source_facts), {})
                original = {k: v for k, v in original.items() if not k.startswith("binding_")}
                sources.append({**deepcopy(original), "question_id": qid, "facts_path": str(facts_path),
                                "episode_id": facts["episode_id"], "world_id": world,
                                "audio_variant_root": str(root), "retained_source_root": job["source"],
                                "task_family": "audio_answer_balance",
                                "sound_asset_ids": sorted({e["sound_asset_id"] for e in facts["events"]}),
                                "audio_path": paths.get("mixture_audio") or paths["audio_readback"], "video_path": paths["video"],
                                "questions_path": str(root/"targeted_questions.json")})
                added += 1
    # New worlds get one split each. A reference bank preserves prior assignments
    # when extending an existing dataset; never split audio variants separately.
    missing = sorted({s.get("world_id") or s["episode_id"] for s in sources} - set(world_splits))
    rng = random.Random(seed)
    for world in missing:
        draw = rng.random()
        world_splits[world] = "train" if draw < .8 else "valid" if draw < .9 else "test"
    splits = [{"question_id": s["question_id"], "world_id": s.get("world_id") or s["episode_id"],
               "facts_path": s["facts_path"], "split": world_splits[s.get("world_id") or s["episode_id"]]} for s in sources]
    for path, rows in (("public/questions.jsonl", public), ("private/answers.jsonl", answers),
                       ("private/sources.jsonl", sources), ("private/splits.jsonl", splits)):
        _rows(output/path, rows)
    # Form changes invalidate old MCQ candidate counts. Recompute private strata
    # from current forms and facts, including appended audio and binding rows.
    from avengine.dataset.question_strata import question_strata
    source_index = {s["question_id"]: s for s in sources}
    strata, strata_errors = [], 0
    for item in answers:
        source = source_index[item["question_id"]]
        path = source["facts_path"]
        try:
            if path not in facts_cache:
                facts_cache[path] = json.loads(Path(path).read_text())
            strata.append(question_strata(item["question_id"], item, facts_cache[path], source))
        except (ValueError, KeyError, TypeError) as error:
            strata_errors += 1
            strata.append({"question_id": item["question_id"], "qa_id": item["qa_id"],
                           "status": "not_derived", "reason": f"{type(error).__name__}: {error}"})
    _rows(output/"private/strata.jsonl", strata)
    prior = write_prior_receipt(output)
    report = {"status": "completed_with_prior_flags" if prior["status"] != "pass" else "completed",
              "base_bank": str(bank), "exported_question_count": len(public), "added_audio_questions": added,
              "qa_counts": dict(sorted(Counter(p["qa_id"] for p in public).items())),
              "forms": dict(Counter(f for p in public for f in p["forms"])),
              "world_count": len({s.get("world_id") or s["episode_id"] for s in sources}),
              "split_counts": dict(Counter(s["split"] for s in splits)),
              "answer_prior_audit": {"path": "private/answer_priors.json", "status": prior["status"]},
              "strata_rows": len(strata), "strata_errors": strata_errors,
              "new_visual_renders": 0, "scoring_policy": {"form_denominator": "offered_forms"},
              "claim_boundary": "Corrected research bank. Historical and new variants are not all balanced; audit flags and independent-world quotas remain visible."}
    _write(output/"report.json", report)
    (output/"README.md").write_text("# Corrected QA-01 through QA-25 bank\n\nPublic inputs: public/questions.jsonl. Authoritative answers, source provenance, world-level splits and answer-prior receipt are under private/. Use each form's declared scorer and offered_forms denominator. Report-only flags do not certify balance or human/model admission.\n")
    return report
