"""Surplus answers are thinned per split; groups and each world's last question stay."""
import json
from pathlib import Path

from avengine.qa.answer_balance import allowed_counts, balance_bank


def test_two_answers_are_brought_to_exact_balance():
    assert allowed_counts({"no": 180, "yes": 24}, max_share=0.45, domain_size=2) == {"no": 24, "yes": 24}


def test_many_answers_stop_at_the_share_limit():
    kept = allowed_counts({"out": 140, "clear": 30, "occluded": 20}, max_share=0.45, domain_size=3)
    assert max(kept.values()) / sum(kept.values()) <= 0.45 and kept["clear"] == 30


def _bank(tmp_path, rows):
    for rel, key in (("public/questions.jsonl", "public"), ("private/answers.jsonl", "answer"),
                     ("private/sources.jsonl", "source")):
        path = tmp_path / "in" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r[key]) + "\n" for r in rows))
    return tmp_path / "in"


def _row(i, answer, world, group=None):
    qid = f"q{i}"
    opts = [{"value": "yes"}, {"value": "no"}]
    return {"public": {"question_id": qid, "qa_id": "QA-05", "forms": {"mcq": {"options": opts}}},
            "answer": {"question_id": qid, "qa_id": "QA-05", "forms": {"open": {"truth": answer}}},
            "source": {"question_id": qid, "world_id": world, **({"binding_group_id": group} if group else {})}}


def test_bank_balance_keeps_groups_and_every_world(tmp_path):
    rows = [_row(i, "no", f"w{i % 3}") for i in range(12)] + [_row(12, "yes", "w0"), _row(13, "yes", "w1")]
    rows.append(_row(14, "no", "w9"))           # a world with one question
    rows.append(_row(15, "no", "w0", group="g"))  # a grouped member
    source = _bank(tmp_path, rows)
    summary = balance_bank(source, tmp_path / "out", {r["public"]["question_id"]: "train" for r in rows})
    kept = [json.loads(l) for l in (tmp_path / "out/public/questions.jsonl").read_text().splitlines()]
    kept_ids = {q["question_id"] for q in kept}
    assert {"q14", "q15"} <= kept_ids
    worlds = {json.loads(l)["world_id"] for l in (tmp_path / "out/private/sources.jsonl").read_text().splitlines()}
    assert worlds == {"w0", "w1", "w2", "w9"}
    answers = [json.loads(l) for l in (tmp_path / "out/private/answers.jsonl").read_text().splitlines()]
    sources = {json.loads(l)["question_id"]: json.loads(l) for l in (tmp_path / "out/private/sources.jsonl").read_text().splitlines()}
    free = [a["forms"]["open"]["truth"] for a in answers if not sources[a["question_id"]].get("binding_group_id")]
    # two yes survive; no is cut to match, except the one a single-question world must keep
    assert free.count("yes") == 2 and 2 <= free.count("no") <= 3


def test_a_stem_whose_every_question_has_one_answer_is_dropped(tmp_path):
    rows = []
    for i in range(6):   # "is the television moving" - always still, whatever the scene
        r = _row(i, "no", f"w{i}")
        r["answer"]["forms"]["open"]["question_en"] = "Was the television moving?"
        rows.append(r)
    for i in range(6, 12):
        r = _row(i, "yes" if i % 2 else "no", f"w{i}")
        r["answer"]["forms"]["open"]["question_en"] = "Was the person moving?"
        rows.append(r)
    for i in range(12):
        extra = _row(100 + i, "yes" if i % 2 else "no", f"w{i}")
        extra["public"]["qa_id"] = extra["answer"]["qa_id"] = "QA-11"
        rows.append(extra)  # every world keeps other questions
    source = _bank(tmp_path, rows)
    summary = balance_bank(source, tmp_path / "out", {r["public"]["question_id"]: "train" for r in rows})
    kept = {json.loads(l)["question_id"] for l in (tmp_path / "out/public/questions.jsonl").read_text().splitlines()}
    assert not kept & {f"q{i}" for i in range(6)}
    assert summary["text_determined_dropped"] == 6


def test_two_present_answers_of_a_larger_domain_are_balanced_not_emptied():
    """Four options, two of them present: one half is the best reachable share."""
    assert allowed_counts({"a": 4, "b": 4}, max_share=0.45, domain_size=4) == {"a": 4, "b": 4}
    assert allowed_counts({"a": 9, "b": 4}, max_share=0.45, domain_size=4) == {"a": 4, "b": 4}
