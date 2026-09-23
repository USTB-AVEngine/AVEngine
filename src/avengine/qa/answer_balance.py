"""Thin the commonest answers of a question bank, split by split.

A scene recipe decides most answers: sequential voices make every overlap
question "no", still speakers make every motion question "still", a camera
that hides the speaker makes every visibility question "out of view". Wording
cannot change that, and a model that has learnt the recipe scores it. This
drops the surplus of each over-represented answer inside each split, until
no answer of a type holds more than ``max_share`` of it, or exactly half
when the type has only two answers.

It never drops a grouped member (a group is scored whole) or the last
question of a world, so a splitter that assigns worlds finds the same worlds
and puts them where it put them before. The bank that is read is not written.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def answer_key(answer_row: Mapping[str, Any]) -> str | None:
    truth = ((answer_row.get("forms") or {}).get("open") or {}).get("truth",
             (answer_row.get("truth") or {}).get("value"))
    return None if truth is None else json.dumps(truth, sort_keys=True, ensure_ascii=False)


def allowed_counts(counts: Mapping[str, int], *, max_share: float, domain_size: int) -> dict[str, int]:
    """Largest counts no answer of which exceeds its share limit.

    The limit is ``max_share``, or 1/domain_size when that is higher: a type
    with two answers cannot go below one half and is held to exactly that.
    """
    kept = dict(counts)
    if len(kept) < 2:
        return {key: 0 for key in kept}  # one answer only: nothing to balance against
    # With m answers present no share can go below 1/m. Thinning further would only
    # empty the cell; balance among the present answers and leave the rest to the gate.
    limit = max(float(max_share), 1.0 / max(1, domain_size), 1.0 / len(kept))
    for _ in range(4 * len(kept) + 8):
        total = sum(kept.values())
        top = max(kept, key=lambda k: (kept[k], k))
        if kept[top] <= limit * total + 1e-9:
            break
        others = total - kept[top]
        # largest n with n / (n + others) <= limit
        kept[top] = int((limit * others) / (1.0 - limit) + 1e-9) if limit < 1 else kept[top]
    return kept


def _drop_surplus(ids_by_answer, target, *, sources, per_world, rng, drop):
    dropped = {}
    for key, ids in sorted(ids_by_answer.items()):
        ids = [i for i in ids if i not in drop]
        surplus = len(ids) - target.get(key, len(ids))
        if surplus <= 0:
            continue
        # Take from the worlds that hold the most questions first.
        order = sorted(ids, key=lambda i: (-per_world[sources[i].get("world_id")], rng.random()))
        taken = 0
        for qid in order:
            if taken == surplus:
                break
            world = sources[qid].get("world_id")
            if per_world[world] <= 1:
                continue  # the world's last question keeps the world and its split
            drop.add(qid)
            per_world[world] -= 1
            taken += 1
        dropped[key] = taken
    return dropped


def balance_bank(bank_in: str | Path, bank_out: str | Path, splits: Mapping[str, str], *,
                 max_share: float = 0.45, seed: int = 20260924,
                 min_template_cell: int = 4) -> dict[str, Any]:
    source, out = Path(bank_in).resolve(), Path(bank_out).resolve()
    if out.exists():
        raise FileExistsError(out)
    public = _read(source / "public/questions.jsonl")
    answers = {r["question_id"]: r for r in _read(source / "private/answers.jsonl")}
    sources = {r["question_id"]: r for r in _read(source / "private/sources.jsonl")}
    per_world = Counter(sources[q["question_id"]].get("world_id") for q in public)
    # The same option count the release gate reads: multiple-choice options, or for a
    # type offered only as open answers, the candidates each question names.
    mcq_size, open_size = defaultdict(int), defaultdict(int)
    for q in public:
        options = ((q.get("forms") or {}).get("mcq") or {}).get("options") or []
        mcq_size[q["qa_id"]] = max(mcq_size[q["qa_id"]], len(options))
        classes = (((answers[q["question_id"]].get("forms") or {}).get("open") or {}).get("classes")) or {}
        if isinstance(classes, Mapping):
            open_size[q["qa_id"]] = max(open_size[q["qa_id"]], len(classes))
    domain = {qa: mcq_size[qa] or open_size[qa] for qa in set(mcq_size) | set(open_size)}
    cells: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for q in public:
        qid = q["question_id"]
        if sources[qid].get("binding_group_id") or splits.get(qid) is None:
            continue
        key = answer_key(answers[qid])
        if key is not None:
            cells[(splits[qid], q["qa_id"])][key].append(qid)
    rng = random.Random(seed)
    drop, report = set(), {}
    # A stem can carry its answer while the type is balanced: "was the television moving"
    # is always no. Stems are the audit's number-masked templates. Whether the words decide
    # a stem's answer is read over all of its questions, before anything else is thinned.
    from avengine.qa.prior_audit import _template
    prompt_of = {q["question_id"]: ((q.get("forms") or {}).get("open") or {}).get("question_en") for q in public}
    stem_of = {}
    stems: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for (split, qa), by_answer in cells.items():
        for key, ids in by_answer.items():
            for qid in ids:
                prompt = (((answers[qid].get("forms") or {}).get("open") or {}).get("question_en")
                          or prompt_of.get(qid))
                stem_of[qid] = _template(prompt)
                stems[(qa, stem_of[qid])][key].append(qid)
    text_determined = 0
    for (qa, stem), by_answer in sorted(stems.items()):
        if len(by_answer) == 1 and sum(len(v) for v in by_answer.values()) >= min_template_cell:
            dropped = _drop_surplus(by_answer, {k: 0 for k in by_answer}, sources=sources,
                                    per_world=per_world, rng=rng, drop=drop)
            text_determined += sum(dropped.values())
            report[f"stem:{qa}:{stem[:80]}"] = {"text_determined": True, "dropped": dropped}

    def type_pass():
        changed = 0
        for (split, qa), by_answer in sorted(cells.items()):
            live = {k: [i for i in v if i not in drop] for k, v in by_answer.items()}
            live = {k: v for k, v in live.items() if v}
            counts = {k: len(v) for k, v in live.items()}
            size = domain.get(qa) or len(counts)
            target = allowed_counts(counts, max_share=max_share, domain_size=size)
            dropped = _drop_surplus(live, target, sources=sources, per_world=per_world, rng=rng, drop=drop)
            if dropped:
                changed += sum(dropped.values())
                cell = report.setdefault(f"type:{split}:{qa}", {"before": counts, "dropped": {},
                                                                 "limit": max(max_share, 1.0 / max(1, size))})
                for k, n in dropped.items():
                    cell["dropped"][k] = cell["dropped"].get(k, 0) + n
        return changed

    def stem_pass():
        changed = 0
        per_split: dict[tuple[str, str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for (split, qa), by_answer in cells.items():
            for key, ids in by_answer.items():
                for qid in ids:
                    if qid not in drop:
                        per_split[(split, qa, stem_of[qid])][key].append(qid)
        for (split, qa, stem), by_answer in sorted(per_split.items()):
            if sum(len(v) for v in by_answer.values()) < min_template_cell or len(by_answer) < 2:
                continue
            counts = {k: len(v) for k, v in by_answer.items()}
            target = allowed_counts(counts, max_share=max_share, domain_size=domain.get(qa) or len(counts))
            dropped = _drop_surplus(by_answer, target, sources=sources, per_world=per_world, rng=rng, drop=drop)
            if dropped:
                changed += sum(dropped.values())
                cell = report.setdefault(f"stem:{split}:{qa}:{stem[:80]}", {"before": counts, "dropped": {}})
                for k, n in dropped.items():
                    cell["dropped"][k] = cell["dropped"].get(k, 0) + n
        return changed

    # Thinning a stem can unbalance its type and the reverse; both only remove, so
    # alternating ends. The type pass runs last, so every type ends balanced.
    rounds = 0
    while rounds < 12:
        rounds += 1
        if not (type_pass() + stem_pass()):
            break
    type_pass()
    out.mkdir(parents=True)
    kept_public = [q for q in public if q["question_id"] not in drop]
    _write(out / "public/questions.jsonl", kept_public)
    kept = {q["question_id"] for q in kept_public}
    for rel in ("private/answers.jsonl", "private/sources.jsonl", "private/strata.jsonl"):
        rows = _read(source / rel)
        if rows:
            _write(out / rel, [r for r in rows if r.get("question_id") in kept])
    for rel in ("private/binding_groups.json", "report.json", "README.md", "run_config.json"):
        if (source / rel).is_file():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, out / rel)
    if (source / "media").is_dir():
        # Hard links, not a link to the directory: a later step writes derived media
        # (mono downmixes) into this bank, and must not write into the source bank.
        (out / "media").mkdir()
        for item in (source / "media").iterdir():
            if item.is_file():
                try:
                    os.link(item, out / "media" / item.name)
                except OSError:
                    shutil.copy2(item, out / "media" / item.name)
    summary = {"schema": "avengine_bank_answer_balance_v1", "source_bank": str(source), "bank_run": str(out),
               "max_share": max_share, "seed": seed, "questions_before": len(public),
               "questions_after": len(kept_public), "dropped": len(drop),
               "by_qa_after": dict(sorted(Counter(q["qa_id"] for q in kept_public).items())),
               "cells": report,
               "text_determined_dropped": text_determined, "min_template_cell": min_template_cell,
               "balance_rounds": rounds,
               "rule": "per split and type, then per split and number-masked stem, drop surplus of the commonest "
                       "answers until none exceeds max(max_share, 1/option count); a stem with one answer over "
                       "the whole bank and at least min_template_cell questions is dropped as text-determined; "
                       "grouped members and each world's last question stay"}
    (out / "answer_balance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
