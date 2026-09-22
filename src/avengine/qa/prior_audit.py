"""Report answer shortcuts and sample support without changing questions or scores."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.qa.unified_scoring import score_open_form

# The owner-requested first stage is diagnostic, not an admission gate. These
# thresholds expose dominant answers, weak MCQs and small validation samples.
DEFAULT_THRESHOLDS = {
    "max_majority_share": 0.60,
    "min_mcq_options": 4,
    "min_valid_questions": 24,
    "max_position_deviation": 0.10,
}
BINARY_QAS = frozenset({"QA-01", "QA-04", "QA-05", "QA-06", "QA-07", "QA-09", "QA-11", "QA-15", "QA-16", "QA-17"})
WORDING_QAS = frozenset({"QA-04", "QA-07", "QA-15", "QA-16"})


def _index(rows, name):
    result = {}
    for row in rows:
        key = row.get("question_id")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} row has no question_id")
        if key in result:
            raise ValueError(f"duplicate {name} question_id: {key}")
        result[key] = row
    return result


def _binding_form(answer, public):
    """Recover the supported compact binding export; never guess index zero.

    The retained merged bank has 66 compact core rows. Their public labels and
    private truth are sufficient, but silently skipping them biases the audit.
    """
    truth = answer.get("truth") or {}
    value, label = truth.get("value"), truth.get("label")
    result = {}
    forms = public.get("forms") or {}
    answer_type = truth.get("answer_type")
    if "open" in forms:
        form = {"answer_type": answer_type, "truth": value}
        if answer_type == "closed_set":
            if label is None:
                raise ValueError("compact binding truth has no public answer label")
            form["truth"] = str(label)
            classes = {}
            for option in forms.get("mcq", {}).get("options", []):
                text = str(option["label_en"])
                classes[text] = [text, str(option.get("label_zh", text))]
            if str(label) not in classes:
                classes[str(label)] = [str(label)]
            form["classes"] = classes
        elif answer_type != "time_range_s":
            raise ValueError(f"unsupported compact binding answer type: {answer_type}")
        result["open"] = form
    if "mcq" in forms:
        options = deepcopy(forms["mcq"].get("options", []))
        if answer_type == "time_range_s":
            # Compact binding labels say "s", public labels say "seconds".
            # Match the actual endpoints using the existing numeric scorer.
            numeric = {"answer_type": "time_range_s", "truth": value}
            matches = [i for i,o in enumerate(options)
                       if score_open_form(numeric, str(o.get("label_en", ""))).get("score") == 1.0]
        else:
            matches = [i for i, o in enumerate(options) if label in (o.get("label_en"), o.get("label_zh"))]
        if len(matches) != 1:
            raise ValueError("compact binding gold does not identify exactly one public option")
        result["mcq"] = {"options": options, "gold": {"correct_index": matches[0]}}
    return result


def _answer_text(form):
    value = form.get("truth")
    kind = form.get("answer_type")
    if kind == "closed_set":
        aliases = (form.get("classes") or {}).get(str(value))
        if not aliases:
            raise ValueError(f"closed truth {value!r} has no public alias")
        return str(aliases[0])
    if kind in {"count_pair", "count_single"}:
        values = value if isinstance(value, (list, tuple)) else [value]
        return " ".join(str(v) for v in values)
    if kind == "time_range_s":
        return f"[{value[0]:g}, {value[1]:g}) seconds"
    if kind == "angle_deg":
        return f"{float(value):g} degrees"
    if kind == "time_s":
        return f"{float(value):g} seconds"
    if kind == "transcript_wer":
        return str(value)
    raise ValueError(f"unsupported answer type: {kind}")


def _distribution(values):
    counts = Counter(values)
    n = sum(counts.values())
    mode = sorted(counts, key=lambda k: (-counts[k], k))[0] if counts else None
    return {"n": n, "distinct": len(counts), "counts": dict(sorted(counts.items())),
            "most_common_answer": mode, "majority_share": counts[mode] / n if n else None,
            "entropy_bits": -sum(v/n * math.log2(v/n) for v in counts.values()) if n else None}


def _profile(rows):
    opened = [r for r in rows if "open" in r["forms"]]
    mcq = [r for r in rows if "mcq" in r["forms"]]
    distribution = _distribution(r["answer"] for r in opened)
    by_k = defaultdict(Counter)
    for r in mcq:
        form = r["forms"]["mcq"]
        n = len(form.get("options", []))
        index = form.get("gold", {}).get("correct_index")
        if n < 2 or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n:
            raise ValueError(f"invalid MCQ domain/gold: {r['question_id']}")
        by_k[n][index] += 1
    position = {}
    for n, counts in sorted(by_k.items()):
        total = sum(counts.values())
        shares = {str(i): counts[i]/total for i in range(n)}
        position[str(n)] = {"n": total, "counts": {str(i): counts[i] for i in range(n)},
                            "shares": shares, "max_deviation": max(abs(v-1/n) for v in shares.values())}
    return {"questions": len(rows), "open_questions": len(opened), "mcq_questions": len(mcq),
            "known_worlds": len({r["world_id"] for r in rows if r["world_id"]}),
            "missing_world_ids": sum(not r["world_id"] for r in rows),
            "open_answers": distribution,
            "open_scoring_modes": dict(Counter(str(r["forms"]["open"].get("scoring_mode", r["forms"]["open"].get("answer_type"))) for r in opened)),
            "mcq_option_count_histogram": {str(k): sum(v.values()) for k,v in sorted(by_k.items())},
            "mcq_random_baseline": sum(1/len(r["forms"]["mcq"]["options"]) for r in mcq)/len(mcq) if mcq else None,
            "correct_option_positions_by_option_count": position,
            "wording_orders": dict(Counter(" / ".join(r["wording_order"]) for r in rows if r["wording_order"])),
            "wording_order_missing": sum(not r["wording_order"] for r in rows)}


def audit_priors(public_rows, answer_rows, *, source_rows=(), splits=None, thresholds=None):
    """Audit one bank. `splits` maps question IDs, never presentation IDs, to splits."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if set(limits) != set(DEFAULT_THRESHOLDS):
        raise ValueError("unknown prior-audit threshold")
    for name in ("max_majority_share", "max_position_deviation"):
        if not isinstance(limits[name], (int, float)) or isinstance(limits[name], bool) or not 0 <= limits[name] <= 1:
            raise ValueError(f"invalid threshold {name}")
    for name in ("min_mcq_options", "min_valid_questions"):
        if isinstance(limits[name], bool) or not isinstance(limits[name], int) or limits[name] < 1:
            raise ValueError(f"invalid threshold {name}")
    public, answers, sources = _index(public_rows, "public"), _index(answer_rows, "answer"), _index(source_rows, "source")
    if set(public) != set(answers):
        raise ValueError("public/answer question IDs do not match")
    if set(sources) - set(public):
        raise ValueError("source rows reference unknown questions")
    splits = dict(splits or {})
    if set(splits) - set(public):
        raise ValueError("split rows reference unknown questions")
    if any(v not in {"train", "valid", "test"} for v in splits.values()):
        raise ValueError("split must be train, valid or test")
    grouped = defaultdict(list)
    for key, a in answers.items():
        q = public[key]
        if q.get("qa_id") != a.get("qa_id"):
            raise ValueError(f"public/answer type mismatch: {key}")
        forms = a.get("forms") if isinstance(a.get("forms"), Mapping) else _binding_form(a, q)
        if set(forms) != set(q.get("forms", {})):
            raise ValueError(f"public/private form mismatch: {key}")
        evidence = a.get("evidence") or (a.get("truth") or {}).get("evidence") or {}
        grouped[a["qa_id"]].append({"question_id": key, "forms": forms,
            "answer": _answer_text(forms["open"]) if "open" in forms else None,
            "wording_order": evidence.get("wording_order"),
            "world_id": sources.get(key, {}).get("world_id"), "split": splits.get(key),
            "prompt": forms.get("open", {}).get("question_en") or q.get("forms", {}).get("open", {}).get("question_en")})
    results = {}
    for qa in sorted(set(grouped) | {f"QA-{i:02d}" for i in range(1, 26)}):
        rows = grouped[qa]
        result = _profile(rows)
        split_rows = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
        result["by_split"] = {s: _profile(rr) for s, rr in split_rows.items()}
        flags = []
        if not rows:
            flags.append("no_questions")
        for scope, profile in [("bank", result), *result["by_split"].items()]:
            if (profile["open_answers"]["majority_share"] or 0) > limits["max_majority_share"]:
                flags.append(f"{scope}:dominant_open_answer")
            if any(x["max_deviation"] > limits["max_position_deviation"] for x in profile["correct_option_positions_by_option_count"].values()):
                flags.append(f"{scope}:correct_option_position_skew")
        if qa not in BINARY_QAS and any(int(n) < limits["min_mcq_options"] for n in result["mcq_option_count_histogram"]):
            flags.append("mcq_domain_too_small")
        result["binary_mcq_warning"] = qa in BINARY_QAS and result["mcq_questions"] > 0
        if result["binary_mcq_warning"]:
            flags.append("binary_mcq_has_50_percent_guessing_baseline")
        if qa in WORDING_QAS and rows:
            if result["wording_order_missing"]:
                flags.append("wording_randomization_unrecorded")
            elif len(result["wording_orders"]) < 2:
                flags.append("wording_order_constant")
        if splits and len(split_rows["valid"]) < limits["min_valid_questions"]:
            flags.append("validation_under_24" if limits["min_valid_questions"] == 24 else "validation_under_minimum")
        result["validation_quota_status"] = "not_checked" if not splits else ("pass" if len(split_rows["valid"]) >= limits["min_valid_questions"] else "under_powered")
        train = [r for r in split_rows["train"] if "open" in r["forms"]]
        valid = [r for r in split_rows["valid"] if "open" in r["forms"]]
        if train:
            guess = _distribution(r["answer"] for r in train)["most_common_answer"]
            lookup = defaultdict(list)
            for r in train:
                lookup[r["prompt"]].append(r["answer"])
            result["train_fitted_constant_answer"] = guess
            result["train_fitted_prompt_lookup"] = {"fit_questions": len(train)}
            for split, rr in split_rows.items():
                opened = [r for r in rr if "open" in r["forms"]]
                scores = [float(score_open_form(r["forms"]["open"], guess)["score"]) for r in opened]
                result["by_split"][split]["train_constant_score"] = sum(scores)/len(scores) if scores else None
                if scores and sum(scores)/len(scores) > limits["max_majority_share"]:
                    flags.append(f"{split}:high_constant_answer_score")
                lookup_scores = [float(score_open_form(r["forms"]["open"], _distribution(lookup[r["prompt"]])["most_common_answer"] if r["prompt"] in lookup else guess)["score"]) for r in opened]
                result["train_fitted_prompt_lookup"][split] = {"n": len(opened), "mean_score": sum(lookup_scores)/len(lookup_scores) if lookup_scores else None,
                    "prompts_seen_in_train": sum(r["prompt"] in lookup for r in opened)}
        if train and valid:
            t, v = Counter(r["answer"] for r in train), Counter(r["answer"] for r in valid)
            result["train_valid_answer_priors"] = {
                "total_variation": sum(abs(t[k]/len(train)-v[k]/len(valid)) for k in t.keys() | v.keys())/2,
                "same_most_common_answer": result["by_split"]["train"]["open_answers"]["most_common_answer"] == result["by_split"]["valid"]["open_answers"]["most_common_answer"],
                "note": "Similarity itself is not a failure; inspect dominance and held-out constant scores."}
        result["flags"] = flags
        result["status"] = "under_powered" if flags else "pass"
        result["eligible_for_supported_main_average"] = bool(splits) and result["validation_quota_status"] == "pass"
        results[qa] = result
    aggregate = {}
    for split in ("train", "valid", "test"):
        supported = [qa for qa,r in results.items() if r["eligible_for_supported_main_average"]]
        total = sum(r["by_split"][split]["open_questions"] for r in results.values())
        fitted = [r for r in results.values() if r["by_split"][split].get("train_constant_score") is not None]
        scored_n = sum(r["by_split"][split]["open_questions"] for r in fitted)
        aggregate[split] = {"open_questions": total, "constant_baseline_scored_questions": scored_n,
            "train_constant_micro_score": sum(r["by_split"][split]["open_questions"] * r["by_split"][split]["train_constant_score"] for r in fitted)/scored_n if scored_n else None,
            "train_constant_macro_score": sum(r["by_split"][split]["train_constant_score"] for r in fitted)/len(fitted) if fitted else None,
            "train_constant_supported_macro_score": (
                sum(results[qa]["by_split"][split]["train_constant_score"] for qa in supported
                    if results[qa]["by_split"][split].get("train_constant_score") is not None)
                / len([qa for qa in supported if results[qa]["by_split"][split].get("train_constant_score") is not None])
                if any(results[qa]["by_split"][split].get("train_constant_score") is not None for qa in supported) else None),
            "main_average_eligible_qa_ids": supported,
            "main_average_exclusion_reason": "validation sample quota only; report all other prior flags alongside scores"}
    return {"schema": "avengine_answer_prior_audit_v1", "mode": "report_only", "thresholds": limits,
        "threshold_rationale": "Owner-requested 60% dominance, four options, 24 validation questions and 10 percentage-point position deviation; flags do not reject media or change scores.",
        "question_count": len(public), "split_assigned_questions": len(splits),
        "split_status": "complete" if len(splits) == len(public) and public else "partial" if splits else "not_provided",
        "status": "under_powered" if any(r["flags"] for r in results.values()) else "pass",
        "by_qa": results, "aggregate": aggregate,
        "counting": "One question_id per bank; forms and presentations do not increase sample size. World counts use supplied world_id, never episode_key.",
        "scoring": "Train-only constant and prompt lookup evaluated with each retained form's declared scorer; exact answer frequency is reported separately. No validation label is used to fit a predictor."}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_map_from_views(directory):
    mapping = {}
    for split in ("train", "valid", "test"):
        path = Path(directory) / f"{split}.jsonl"
        for row in read_jsonl(path):
            key = row["question_id"]
            if key in mapping and mapping[key] != split:
                raise ValueError(f"question belongs to multiple splits: {key}")
            mapping[key] = split
    return mapping


def audit_bank(bank, *, split_views=None, thresholds=None):
    bank = Path(bank)
    sources = bank / "private/sources.jsonl"
    split_file = bank / "private/splits.jsonl"
    splits = split_map_from_views(split_views) if split_views else (
        {key: row["split"] for key, row in _index(read_jsonl(split_file), "split").items()}
        if split_file.is_file() else None)
    return audit_priors(read_jsonl(bank/"public/questions.jsonl"), read_jsonl(bank/"private/answers.jsonl"),
        source_rows=read_jsonl(sources) if sources.exists() else (),
        splits=splits, thresholds=thresholds)


def write_prior_receipt(bank, *, output=None, split_views=None, thresholds=None):
    try:
        result = audit_bank(bank, split_views=split_views, thresholds=thresholds)
    except (ValueError, KeyError, TypeError) as error:
        # Reporting must not turn a legacy partial schema into a production
        # failure or a false PASS. The owner can opt into a failing CLI exit.
        result = {"schema": "avengine_answer_prior_audit_v1", "mode": "report_only",
                  "status": "invalid_input", "error": f"{type(error).__name__}: {error}",
                  "by_qa": {}, "split_status": "not_checked"}
    output = Path(output) if output is not None else Path(bank)/"private/answer_priors.json"
    # A diagnostic on a historical bank must use a new explicit output path.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result
