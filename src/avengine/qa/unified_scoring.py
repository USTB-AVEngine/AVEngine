"""Deterministic scoring for the unified QA-01..QA-25 question forms.

The scorer accepts one generated question item and a model answer. It never
reads the hidden engine evidence from the input bundle. MCQ answers are exact
option matches; open answers use explicit numeric tolerances, closed-set
normalization, integer count equality, or word-error rate for transcripts.
Ambiguous answers are invalid and do not silently become wrong answers.
Refusal terms are tracked as abstained with zero score.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


UNIFIED_SCORE_SCHEMA = "avengine_qa_unified_score_v1"
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_UNSIGNED_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
_TIME_RANGE_SEPARATOR = re.compile(
    rf"(?<![\w.+-])(?P<start>{_UNSIGNED_NUMBER})\s*(?:-|–|—|−|~|～|到|至|\bto\b)\s*"
    rf"(?P<end>{_UNSIGNED_NUMBER})(?![\w.])",
    re.IGNORECASE,
)
_UNSIGNED_NUMBER_TOKEN = re.compile(
    rf"(?<![A-Za-z0-9_.]){_UNSIGNED_NUMBER}(?![A-Za-z0-9_.])"
)
_ANGLE_MARK = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*(?:°|度|deg(?:ree)?s?)",
    re.IGNORECASE,
)
_TIME_MARK = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*(?:秒|s\b|sec(?:ond)?s?)",
    re.IGNORECASE,
)
_ABSTAIN = (
    "无法判断",
    "无法确定",
    "不知道",
    "说不准",
    "不确定",
    "cannot tell",
    "can't tell",
    "not sure",
    "unable to determine",
)


class UnifiedScoreError(ValueError):
    """A scoring request is malformed."""


def circular_distance_deg(first: float, second: float) -> float:
    distance = abs(float(first) - float(second)) % 360.0
    return min(distance, 360.0 - distance)


def _number(
    text: str,
    *,
    marked: re.Pattern[str],
) -> tuple[float | None, str | None]:
    marked_values = [float(match.group(1)) for match in marked.finditer(text)]
    if len(marked_values) == 1:
        return marked_values[0], None
    if len(marked_values) > 1:
        return None, f"multiple marked numbers: {marked_values}"
    bare = [float(match.group(0)) for match in _NUMBER.finditer(text)]
    if len(bare) == 1:
        return bare[0], None
    if not bare:
        return None, "no number found"
    return None, f"multiple numbers: {bare}"


def _finite_tolerance(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnifiedScoreError(f"{name} must be finite and non-negative")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise UnifiedScoreError(f"{name} must be finite and non-negative")
    return number


def _has_abstention(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).casefold()
    return any(term.casefold() in lowered for term in _ABSTAIN)


def _canonical_text(text: Any) -> str:
    if not isinstance(text, str):
        raise UnifiedScoreError("model answer must be text")
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _term_is_negated(text: str, term: str, start: int) -> bool:
    """Avoid turning a negated short alias into a positive direction."""

    if term not in {
        "远", "更远", "近", "更近", "far", "farther", "further",
        "near", "nearer", "closer",
    }:
        return False
    prefix = text[:start].rstrip()
    return any(
        prefix.endswith(marker)
        for marker in ("不", "并不", "不是", "没有", "没", "无", "not", "no")
    )


def _closed_match(
    answer: str,
    classes: Mapping[str, Sequence[str]],
) -> tuple[str | None, str | None]:
    text = _canonical_text(answer)
    hits: dict[str, str] = {}
    for label, terms in classes.items():
        if not isinstance(label, str) or not isinstance(terms, Sequence):
            continue
        matches: list[str] = []
        for term in terms:
            if not isinstance(term, str):
                continue
            normalized = _canonical_text(term)
            if not normalized:
                continue
            start = text.find(normalized)
            if start >= 0 and not _term_is_negated(text, normalized, start):
                matches.append(normalized)
        best = max(matches, key=len, default="")
        if best:
            hits[label] = best
    if not hits:
        return None, "no vocabulary hit"
    if len(hits) == 1:
        return next(iter(hits)), None
    ordered = sorted(hits, key=lambda label: (-len(hits[label]), label))
    longest_label = ordered[0]
    longest_term = hits[longest_label]
    survivors = [
        label for label in ordered
        if not (hits[label] != longest_term and hits[label] in longest_term)
    ]
    if len(survivors) == 1:
        return survivors[0], None
    return None, f"conflicting vocabulary hits: { {label: hits[label] for label in survivors} }"


def _closed_match_v2(answer: str, classes: Mapping[str, Sequence[str]]) -> tuple[str | None, str | None]:
    """Bound Latin aliases and resolve only explicit binary negation.

    The old substring parser accepts 'bright' as right and '不是' as yes.
    Retained forms keep that parser unless they declare this corrected policy.
    This is deliberately a small answer parser, not a general language judge.
    """
    text = _canonical_text(answer)
    opposites = {}
    for first, second in (("yes", "no"), ("left", "right"), ("moving", "still"), ("nearer", "farther")):
        if first in classes and second in classes:
            opposites[first], opposites[second] = second, first
    hits = {}
    original_labels = set()
    for label, terms in classes.items():
        if not isinstance(label, str) or not isinstance(terms, Sequence):
            continue
        for term in terms:
            if not isinstance(term, str) or not term.strip():
                continue
            normalized = _canonical_text(term)
            pattern = re.escape(normalized)
            if re.search(r"[a-z0-9]", normalized):
                pattern = r"(?<![a-z0-9_])" + pattern + r"(?![a-z0-9_])"
            for match in re.finditer(pattern, text):
                original_labels.add(label)
                prefix = text[:match.start()].rstrip()
                negated = bool(re.search(r"(?:不是|并非|没有|并不|不|没|未|非)$", prefix) or
                    re.search(r"\b(?:not|never|no)\s+(?:(?:on|the|to|at|is|was|be|being|did|do|does|a|an)\s+){0,4}$", text[:match.start()]))
                parsed = opposites.get(label) if negated else label
                if parsed is None:
                    continue
                if len(normalized) > len(hits.get(parsed, "")):
                    hits[parsed] = normalized
    if len(original_labels) > 1 and re.search(r"\bor\b|或者|或是", text):
        return None, "answer offers alternative classes"
    if not hits:
        return None, "no unambiguous vocabulary hit"
    survivors = [label for label, term in hits.items() if not any(
        other != term and term in other for other in hits.values())]
    if len(survivors) == 1:
        return survivors[0], None
    return None, f"conflicting vocabulary hits: {hits}"


def score_closed(
    answer: str,
    truth: str,
    classes: Mapping[str, Sequence[str]],
    *,
    refusal_allowed: bool = False,
    policy: str = "legacy",
) -> dict[str, Any]:
    if policy not in {"legacy", "token_negation_v2"}:
        raise UnifiedScoreError(f"unknown closed-set parsing policy: {policy}")
    if _has_abstention(answer):
        return {
            "status": "abstained",
            "score": 0.0,
            "abstention": True,
            "refusal_allowed": bool(refusal_allowed),
        }
    label, reason = (_closed_match_v2(answer, classes) if policy == "token_negation_v2"
                     else _closed_match(answer, classes))
    if label is None:
        return {"status": "invalid", "reason": reason, "score": 0.0}
    return {
        "status": "scored",
        "parsed": label,
        "score": 1.0 if label == str(truth) else 0.0,
        "abstention": False,
    }


def _count_tokens(answer: str) -> tuple[list[int] | None, str | None]:
    tokens = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        answer,
    )
    values: list[int] = []
    for token in tokens:
        if "e" in token.casefold():
            return None, "counts require plain decimal integers"
        signless = token[1:] if token[:1] in "+-" else token
        if "." in signless and signless.rstrip("0").rstrip(".") != signless.split(".", 1)[0]:
            return None, "counts require integer values"
        try:
            value = int(token)
        except ValueError:
            return None, "invalid count value"
        if value < 0:
            return None, "counts cannot be negative"
        values.append(value)
    return values, None


def score_counts(answer: str, truth: Sequence[int]) -> dict[str, Any]:
    if not truth or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in truth
    ):
        return {"status": "invalid", "reason": "count truth must be nonnegative integers", "score": 0.0}
    values, reason = _count_tokens(answer)
    if values is None:
        return {"status": "invalid", "reason": reason, "score": 0.0}
    if len(values) != len(truth):
        return {
            "status": "invalid",
            "reason": f"expected {len(truth)} number(s), found {len(values)}",
            "score": 0.0,
        }
    return {
        "status": "scored",
        "parsed": values,
        "score": 1.0 if list(values) == [int(value) for value in truth] else 0.0,
    }


def _normalization(form: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    value = form.get("normalization") or params.get("TRANSCRIPT_NORMALIZATION")
    if value is None:
        value = {
            "unicode_form": "NFKC",
            "casefold": True,
            "punctuation": "space",
        }
    if not isinstance(value, Mapping):
        raise UnifiedScoreError("transcript normalization must be an object")
    required = {"unicode_form", "casefold", "punctuation"}
    if set(value) != required:
        raise UnifiedScoreError(
            f"transcript normalization requires exactly {sorted(required)}"
        )
    if value["unicode_form"] not in {"NFC", "NFKC", "none"}:
        raise UnifiedScoreError("transcript unicode_form must be NFC, NFKC, or none")
    if not isinstance(value["casefold"], bool):
        raise UnifiedScoreError("transcript casefold must be boolean")
    if value["punctuation"] not in {"keep", "space", "remove"}:
        raise UnifiedScoreError("transcript punctuation must be keep, space, or remove")
    return dict(value)


def _words(text: str, policy: Mapping[str, Any]) -> list[str]:
    if policy["unicode_form"] != "none":
        text = unicodedata.normalize(policy["unicode_form"], text)
    if policy["casefold"]:
        text = text.casefold()
    punctuation = policy["punctuation"]
    if punctuation != "keep":
        replacement = " " if punctuation == "space" else ""
        text = "".join(
            replacement if unicodedata.category(char).startswith("P") else char
            for char in text
        )
    return text.split()


def score_transcript(
    answer: str,
    truth: Any,
    *,
    form: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(truth, str):
        return {"status": "invalid", "reason": "transcript truth must be text", "score": 0.0}
    if form.get("reject_multiple_statements", False):
        normalized_answer = _canonical_text(answer)
        classes = form.get("classes")
        matched_candidates = 0
        if isinstance(classes, Mapping):
            for terms in classes.values():
                if isinstance(terms, Sequence) and any(
                    isinstance(term, str)
                    and _canonical_text(term)
                    and _canonical_text(term) in normalized_answer
                    for term in terms
                ):
                    matched_candidates += 1
        if matched_candidates > 1 or re.search(r"[;；\n\r]", answer):
            return {
                "status": "invalid",
                "reason": "multiple candidate statements in transcript answer",
                "score": 0.0,
            }
    policy = _normalization(form, params)
    reference = _words(truth, policy)
    hypothesis = _words(answer, policy)
    if not reference:
        return {"status": "invalid", "reason": "transcript truth is empty", "score": 0.0}
    previous = list(range(len(hypothesis) + 1))
    for row, reference_word in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    edits = previous[-1]
    wer = edits / len(reference)
    return {
        "status": "scored",
        "metric": "word_error_rate",
        "word_edits": edits,
        "reference_word_count": len(reference),
        "hypothesis_word_count": len(hypothesis),
        "wer": wer,
        "score": max(0.0, 1.0 - wer),
        "exact_match": edits == 0,
        "normalization": policy,
    }


def score_angle(
    answer: str,
    truth: Any,
    *,
    full_tolerance_deg: float,
    half_tolerance_deg: float,
    convention: str = "right_positive",
    strict: bool = False,
) -> dict[str, Any]:
    value, reason = _number(answer, marked=_ANGLE_MARK)
    if value is None:
        return {"status": "invalid", "reason": reason, "score": 0.0}
    lowered = answer.casefold()
    has_left = "左" in lowered or "left" in lowered
    has_right = "右" in lowered or "right" in lowered
    if has_left and has_right:
        return {"status": "invalid", "reason": "both left and right are present", "score": 0.0}
    if convention not in {"left_positive", "right_positive"}:
        return {
            "status": "invalid",
            "reason": f"unknown angle convention {convention!r}",
            "score": 0.0,
        }
    if has_left or has_right:
        expected_sign = (
            -1.0
            if (has_left and convention == "right_positive")
            or (has_right and convention == "left_positive")
            else 1.0
        )
        if value < 0.0 and expected_sign > 0.0:
            return {
                "status": "invalid",
                "reason": "direction word conflicts with negative angle",
                "score": 0.0,
            }
        parsed = value if value < 0.0 else expected_sign * value
    else:
        parsed = value
    if isinstance(truth, Mapping):
        truth = truth.get("azimuth_deg", truth.get("value"))
    try:
        target = float(truth)
    except (TypeError, ValueError):
        return {"status": "invalid", "reason": "angle truth is not numeric", "score": 0.0}
    full = _finite_tolerance(full_tolerance_deg, name="full_tolerance_deg")
    half = _finite_tolerance(half_tolerance_deg, name="half_tolerance_deg")
    if full > half:
        raise UnifiedScoreError("full angle tolerance cannot exceed half tolerance")
    error = circular_distance_deg(parsed, target)
    diagnostic = 1.0 if error <= full else 0.5 if error <= half else 0.0
    return {
        "status": "scored",
        "parsed": parsed,
        "circular_error_deg": error,
        "score": 1.0 if error <= full else 0.0 if strict else diagnostic,
        "diagnostic_two_tier_score": diagnostic,
        "angle_convention": convention,
    }


def score_time(
    answer: str,
    truth: Any,
    *,
    full_tolerance_s: float,
    half_tolerance_s: float,
    strict: bool = False,
) -> dict[str, Any]:
    value, reason = _number(answer, marked=_TIME_MARK)
    if value is None:
        return {"status": "invalid", "reason": reason, "score": 0.0}
    try:
        target = float(truth)
    except (TypeError, ValueError):
        return {"status": "invalid", "reason": "time truth is not numeric", "score": 0.0}
    full = _finite_tolerance(full_tolerance_s, name="full_tolerance_s")
    half = _finite_tolerance(half_tolerance_s, name="half_tolerance_s")
    if full > half:
        raise UnifiedScoreError("full time tolerance cannot exceed half tolerance")
    error = abs(value - target)
    diagnostic = 1.0 if error <= full else 0.5 if error <= half else 0.0
    return {
        "status": "scored",
        "parsed": value,
        "absolute_error_s": error,
        "score": 1.0 if error <= full else 0.0 if strict else diagnostic,
        "diagnostic_two_tier_score": diagnostic,
    }


def score_mcq(form: Mapping[str, Any], answer: str) -> dict[str, Any]:
    options = form.get("options")
    gold = form.get("gold")
    if not isinstance(options, Sequence) or not options or not isinstance(gold, Mapping):
        return {"status": "invalid", "reason": "MCQ form is missing options or gold", "score": 0.0}
    correct_index = gold.get("correct_index")
    if isinstance(correct_index, bool) or not isinstance(correct_index, int):
        return {"status": "invalid", "reason": "MCQ correct index is invalid", "score": 0.0}
    raw = _canonical_text(answer)
    parsed_indices: set[int] = set()
    if re.fullmatch(r"[a-z]", raw):
        parsed_indices.add(ord(raw) - ord("a"))
    elif re.fullmatch(r"[1-9]\d*", raw):
        number = int(raw)
        if 1 <= number <= len(options):
            parsed_indices.add(number - 1)
    elif raw == "0":
        parsed_indices.add(0)
    for index, option in enumerate(options):
        if not isinstance(option, Mapping):
            continue
        terms = {
            _canonical_text(option.get("label_en", "")),
            _canonical_text(option.get("label_zh", "")),
        }
        if bool(option.get("allow_value", True)):
            terms.add(_canonical_text(option.get("value", "")))
        if raw in terms and raw:
            parsed_indices.add(index)
    if len(parsed_indices) != 1:
        return {
            "status": "invalid",
            "reason": "MCQ answer does not identify exactly one option",
            "score": 0.0,
        }
    parsed = next(iter(parsed_indices))
    return {
        "status": "scored",
        "parsed_index": parsed,
        "correct_index": correct_index,
        "score": 1.0 if parsed == correct_index else 0.0,
    }


def _finite_time_range(value: Any) -> tuple[float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        return None
    start, end = float(value[0]), float(value[1])
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return start, end


def _time_range_numbers(answer: str) -> list[float]:
    """Parse two endpoints without treating a range separator as a sign."""

    text = unicodedata.normalize("NFKC", answer)
    range_match = _TIME_RANGE_SEPARATOR.search(text)
    if range_match is not None:
        # Accept the separator form only when the answer contains exactly its
        # two endpoints; extra numbers must remain invalid.
        tokens = list(_UNSIGNED_NUMBER_TOKEN.finditer(text))
        group_spans = {
            (range_match.start("start"), range_match.end("start")),
            (range_match.start("end"), range_match.end("end")),
        }
        if len(tokens) == 2 and {
            (token.start(), token.end()) for token in tokens
        } == group_spans:
            return [float(range_match.group("start")), float(range_match.group("end"))]
    marked = [float(match.group(1)) for match in _TIME_MARK.finditer(text)]
    if len(marked) == 2:
        return marked
    return [float(match.group(0)) for match in _NUMBER.finditer(text)]


def score_time_range(
    answer: str,
    truth: Any,
    *,
    form: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Score an answer against one declared half-open time interval."""
    target = _finite_time_range(truth)
    if target is None:
        return {"status": "invalid", "reason": "time range truth is invalid", "score": 0.0}
    form = form or {}
    classes = form.get("classes")
    if isinstance(classes, Mapping):
        label, _reason = _closed_match(answer, classes)
        if label is not None:
            ranges = form.get("time_ranges_s")
            expected_index = form.get("time_range_index")
            if isinstance(expected_index, bool) or not isinstance(expected_index, int):
                expected_index = None
            if expected_index is None and isinstance(ranges, Sequence):
                for index, candidate in enumerate(ranges):
                    candidate_range = _finite_time_range(candidate)
                    if candidate_range is not None and all(
                        math.isclose(candidate_range[pos], target[pos], rel_tol=0.0, abs_tol=1.0e-9)
                        for pos in (0, 1)
                    ):
                        expected_index = index
                        break
            parsed_index = None
            match = re.fullmatch(r"band_(\d+)", str(label))
            if match:
                parsed_index = int(match.group(1))
            if expected_index is not None and parsed_index is not None:
                return {
                    "status": "scored",
                    "parsed": list(target if parsed_index == expected_index else (
                        _finite_time_range(ranges[parsed_index]) if isinstance(ranges, Sequence) and parsed_index < len(ranges) else target
                    )),
                    "expected_range_s": list(target),
                    "parsed_range_index": parsed_index,
                    "expected_range_index": expected_index,
                    "score": 1.0 if parsed_index == expected_index else 0.0,
                }
    if _has_abstention(answer):
        return {
            "status": "abstained",
            "score": 0.0,
            "abstention": True,
            "refusal_allowed": False,
        }
    numbers = _time_range_numbers(answer)
    if len(numbers) != 2:
        return {
            "status": "invalid",
            "reason": "time range answer must identify exactly two endpoints",
            "score": 0.0,
        }
    parsed = _finite_time_range(numbers)
    if parsed is None:
        return {"status": "invalid", "reason": "time range answer is invalid", "score": 0.0}
    error = max(abs(parsed[0] - target[0]), abs(parsed[1] - target[1]))
    exact = error <= 1.0e-6
    return {
        "status": "scored",
        "parsed": list(parsed),
        "expected_range_s": list(target),
        "range_error_s": error,
        "score": 1.0 if exact else 0.0,
        "strict": bool(strict),
    }


def score_open_form(
    form: Mapping[str, Any],
    answer: str,
    *,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    answer_type = form.get("answer_type")
    truth = form.get("truth")
    if answer_type == "closed_set":
        classes = form.get("classes")
        if not isinstance(classes, Mapping):
            return {"status": "invalid", "reason": "closed-set form has no classes", "score": 0.0}
        return score_closed(
            answer,
            str(truth),
            classes,
            refusal_allowed=bool(form.get("refusal_truth", False)),
            policy=str(params.get("closed_set_policy", form.get("closed_set_policy", "legacy"))),
        )
    if answer_type == "transcript_wer":
        result = score_transcript(answer, truth, form=form, params=params)
        classes = form.get("classes")
        attribution_matches: list[str] = []
        if isinstance(classes, Mapping):
            answer_words = _words(answer, _normalization(form, params))
            for label in classes:
                if not isinstance(label, str):
                    continue
                if answer_words == _words(label, _normalization(form, params)):
                    attribution_matches.append(label)
        result["attribution_match_count"] = len(attribution_matches)
        result["attribution_matches"] = attribution_matches
        result["attribution_match"] = len(attribution_matches) == 1
        if len(attribution_matches) > 1:
            result["status"] = "invalid"
            result["reason"] = "transcript attribution is ambiguous"
            result["score"] = 0.0
        return result
    if answer_type == "angle_deg":
        full = form.get("theta_full_deg", params.get("THETA_FULL", 15.0))
        half = form.get("theta_half_deg", params.get("THETA_HALF", 30.0))
        result = score_angle(
            answer,
            truth,
            full_tolerance_deg=full,
            half_tolerance_deg=half,
            convention=str(form.get("convention", "right_positive")),
            strict=bool(form.get("strict_certification", False)),
        )
        mode = str(form.get("scoring_mode") or "two_tier")
        if mode == "continuous" and result.get("status") == "scored":
            # 1 - error/180 decays so slowly that it is not a pass rate at all. Measured on the
            # 2026-09-22 valid split, over the 46 QA-25 rows: a uniformly random angle scores
            # 0.51, answering "0 degrees" every time scores 0.84, and the best constant answer
            # scores 0.84 - above both models measured that day, which reached 0.80 with full
            # audio and video and 0.84 with mono audio. Averaging it into an overall figure
            # therefore imports a 50% floor. It stays a useful per-row diagnostic, and this
            # branch stays so a bank that declared it keeps reproducing its own numbers, but
            # banks generated after 2026-09-22 declare threshold_graded instead.
            result["score"] = 1.0 - result["circular_error_deg"] / 180.0
            result["score_definition"] = (
                "1 - circular_error_deg / 180; a constant answer scores about 0.84 on a "
                "front-loaded angle distribution, so use angle_metrics as primary"
            )
            result.pop("diagnostic_two_tier_score", None)
        elif mode == "threshold_graded" and result.get("status") == "scored":
            result["score_definition"] = (
                f"1.0 within {full:g} deg, 0.5 within {half:g} deg, else 0.0"
            )
        return result
    if answer_type == "time_range_s":
        return score_time_range(answer, truth, form=form)
    if answer_type == "time_s":
        full = form.get("t_full_s", params.get("T_FULL", 0.3))
        half = form.get("t_half_s", params.get("T_HALF", 1.0))
        return score_time(
            answer,
            truth,
            full_tolerance_s=full,
            half_tolerance_s=half,
            strict=bool(form.get("strict_certification", False)),
        )
    if answer_type in {"count_pair", "count_single"}:
        values = truth if isinstance(truth, Sequence) and not isinstance(truth, (str, bytes)) else [truth]
        return score_counts(answer, values)
    return {"status": "invalid", "reason": f"unknown open answer type {answer_type!r}", "score": 0.0}


def score_unified_item(
    item: Mapping[str, Any],
    answer: Any,
    *,
    form: str = "open",
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(item, Mapping) or item.get("status") != "pass":
        return {
            "status": "invalid",
            "reason": "question item is not a valid candidate",
            "score": 0.0,
        }
    forms = item.get("forms")
    if not isinstance(forms, Mapping) or form not in forms:
        return {
            "status": "invalid",
            "reason": f"question item has no {form} form",
            "score": 0.0,
        }
    text = str(answer)
    result = (
        score_mcq(forms[form], text)
        if form == "mcq"
        else score_open_form(forms[form], text, params=params)
    )
    result["question_id"] = item.get("question_id")
    result["qa_id"] = item.get("qa_id")
    result["form"] = form
    return result


def score_unified_question_set(
    question_set: Mapping[str, Any],
    answers: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    form: str = "open",
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    items = question_set.get("items") if isinstance(question_set, Mapping) else None
    if not isinstance(items, Sequence):
        raise UnifiedScoreError("question set has no items")
    answer_map: dict[str, Any] = {}
    if isinstance(answers, Mapping):
        for key, value in answers.items():
            answer_map[str(key)] = value
    elif isinstance(answers, Sequence) and not isinstance(answers, (str, bytes)):
        for record in answers:
            if not isinstance(record, Mapping) or not isinstance(record.get("question_id"), str):
                raise UnifiedScoreError("answer records need question_id")
            answer_map[record["question_id"]] = record.get(
                "answer", record.get("model_answer", "")
            )
    else:
        raise UnifiedScoreError("answers must be a mapping or records")
    from avengine.qa.unified_catalog import iter_unified_items
    all_items = list(iter_unified_items(question_set))
    # Public exports use ordinal transport IDs because legacy private IDs carry
    # target/event/frame evidence. Keep legacy answer IDs accepted as well.
    for index, item in enumerate(all_items):
        public_id = f"question_{index + 1:06d}"
        if public_id in answer_map and item["question_id"] not in answer_map:
            answer_map[item["question_id"]] = answer_map[public_id]
    denominator = (params or {}).get("denominator_policy", (question_set.get("scoring_policy") or {}).get("form_denominator", "legacy_all_items"))
    if denominator not in {"legacy_all_items", "offered_forms"}:
        raise UnifiedScoreError(f"unknown denominator policy: {denominator}")
    items = [item for item in iter_unified_items(question_set, include_angle_followups=form == "open")
             if (form in item.get("forms", {}) if denominator == "offered_forms" else
                 not (form not in item.get("forms", {}) and
                      item.get("form_status", {}).get(form, {}).get("code") == "continuous_numeric_only"))]
    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        question_id = item.get("question_id")
        if not isinstance(question_id, str):
            continue
        if question_id not in answer_map:
            records.append(
                {
                    "status": "invalid",
                    "reason": "model answer is missing",
                    "score": 0.0,
                    "question_id": question_id,
                    "qa_id": item.get("qa_id"),
                    "form": form,
                }
            )
            continue
        records.append(
            score_unified_item(item, answer_map[question_id], form=form, params=params)
        )
    scored = [record for record in records if record.get("status") == "scored"]
    return {
        "schema": UNIFIED_SCORE_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "form": form,
        "counts": {
            "total": len(records),
            "scored": len(scored),
            "invalid": sum(record.get("status") == "invalid" for record in records),
            "abstained": sum(record.get("status") == "abstained" for record in records),
        },
        "denominator_policy": denominator,
        "mean_score_over_all": (
            sum(float(record.get("score", 0.0)) for record in records) / len(records)
            if records
            else None
        ),
        "records": records,
        "angle_metrics": angular_metrics(items, records) if form == "open" else None,
        "claim_boundary": (
            "Scores describe supplied model answers against research candidate "
            "golds; they do not certify data validity or modality necessity."
        ),
    }


__all__ = [
    "UNIFIED_SCORE_SCHEMA",
    "UnifiedScoreError",
    "circular_distance_deg",
    "score_angle",
    "score_closed",
    "score_counts",
    "score_mcq",
    "score_open_form",
    "score_time",
    "score_time_range",
    "score_transcript",
    "score_unified_item",
    "score_unified_question_set",
]


def angular_metrics(items: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
                    thresholds_deg: Sequence[float] = (1.0, 3.0, 5.0, 10.0)) -> dict[str, Any]:
    """Circular errors on parsed answers; accuracy denominators include missing/invalid."""
    thresholds = [float(value) for value in thresholds_deg]
    if not thresholds or any(not math.isfinite(value) or value < 0 or value > 180 for value in thresholds):
        raise UnifiedScoreError("angle thresholds must be finite degrees between 0 and 180")
    by_id = {row.get("question_id"): row for row in records}
    angle_items = [item for item in items if item.get("forms", {}).get("open", {}).get("answer_type") == "angle_deg"]

    def summarize(selected):
        errors = [float(by_id[item["question_id"]]["circular_error_deg"]) for item in selected
                  if by_id.get(item["question_id"], {}).get("status") == "scored"
                  and isinstance(by_id[item["question_id"]].get("circular_error_deg"), (int, float))
                  and math.isfinite(by_id[item["question_id"]]["circular_error_deg"])]
        return {"total": len(selected), "parsed": len(errors), "unparsed": len(selected) - len(errors),
                "mae_deg": statistics.fmean(errors) if errors else None,
                "median_deg": statistics.median(errors) if errors else None,
                "accuracy_at_deg": {f"{threshold:g}": sum(error <= threshold for error in errors) / len(selected)
                                    if selected else None for threshold in thresholds}}

    paired = [item for item in angle_items if item.get("parent_question_id")]
    binding = [item for item in paired if item.get("binding_parent")]
    def joint(selected):
        correct = [item for item in selected if by_id.get(item["parent_question_id"], {}).get("status") == "scored"
                   and by_id[item["parent_question_id"]].get("score") == 1.0]
        return {"total": len(selected), "parent_accuracy": len(correct) / len(selected) if selected else None,
                "joint_accuracy_at_deg": {f"{threshold:g}": sum(
                    by_id.get(item["question_id"], {}).get("status") == "scored"
                    and by_id[item["question_id"]].get("circular_error_deg", math.inf) <= threshold
                    for item in correct) / len(selected) if selected else None for threshold in thresholds}}
    def constant_answer_baseline(selected):
        """What a model that perceives nothing scores on these very rows.

        Any claim about angle accuracy has to clear this line first. It is reported next to
        the model's own numbers so an inflated metric cannot pass unnoticed.
        """
        truths = []
        for item in selected:
            truth = (item.get("forms", {}).get("open", {}) or {}).get("truth")
            if isinstance(truth, Mapping):
                truth = truth.get("azimuth_deg", truth.get("value"))
            try:
                value = float(truth)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                truths.append(value)
        if not truths:
            return None

        def linear(guess):
            return statistics.fmean(
                1.0 - circular_distance_deg(guess, truth) / 180.0 for truth in truths
            )

        def hit_rate(guess, threshold):
            return sum(circular_distance_deg(guess, truth) <= threshold for truth in truths) / len(truths)

        best = max(range(-180, 180), key=linear)
        return {
            "rows": len(truths),
            "answering_zero_degrees": {
                "linear_score": linear(0.0),
                "accuracy_at_deg": {f"{threshold:g}": hit_rate(0.0, threshold) for threshold in thresholds},
            },
            "best_constant_answer": {
                "degrees": best,
                "linear_score": linear(best),
                "accuracy_at_deg": {f"{threshold:g}": hit_rate(float(best), threshold) for threshold in thresholds},
            },
            "claim_boundary": "a model that perceives nothing should not reach these numbers",
        }

    return {**summarize(angle_items),
            "constant_answer_baseline": constant_answer_baseline(angle_items),
            "constant_answer_baseline_by_qa": {
                qa_id: constant_answer_baseline([item for item in angle_items if item.get("qa_id") == qa_id])
                for qa_id in sorted({item["qa_id"] for item in angle_items})},
            "error_denominator": "finite parsed angle answers", "accuracy_denominator": "all angle questions including missing, invalid and abstained",
            "thresholds_are_reporting_dimensions": True,
            "by_qa": {qa_id: summarize([item for item in angle_items if item.get("qa_id") == qa_id])
                      for qa_id in sorted({item["qa_id"] for item in angle_items})},
            "qa25_by_subset": {subset: summarize([item for item in angle_items if item.get("qa_id") == "QA-25" and item.get("angle_subset") == subset])
                               for subset in ("A", "V", "AV")},
            "followup_parent_joint": joint(paired), "instance_binding_joint": joint(binding)}

__all__.append("angular_metrics")
