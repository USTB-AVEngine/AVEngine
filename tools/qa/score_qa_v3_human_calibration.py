#!/usr/bin/env python3
"""Score QA-v3 human calibration responses without mixing binding errors.

Numeric tolerance candidates are computed only on binding-correct trials.
All trials still contribute to the separately reported full-AV binding rate.
The P75/P95 outputs are decision material, not automatic parameter updates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _circular(a, b):
    value = abs(float(a) - float(b)) % 360.0
    return min(value, 360.0 - value)


def _quantiles(values):
    if not values:
        return {"n": 0, "p75": None, "p95": None, "median": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": int(data.size),
        "median": float(np.quantile(data, 0.5)),
        "p75": float(np.quantile(data, 0.75)),
        "p95": float(np.quantile(data, 0.95)),
    }


def _response_convention(answer_key, response_documents, assumed):
    """Refuse to score answers whose azimuth convention may differ from the key.

    A response produced under "right is positive" scored against a key under
    "positive to the left" is wrong by twice the angle and nothing else would
    notice.  A document that declares nothing is the ambiguous case that
    actually bit us on 2026-09-03, so it needs an explicit assumption that is
    echoed into the output rather than a silent default.
    """

    expected = answer_key.get("azimuth_convention")
    used = None
    for document in response_documents:
        declared = document.get("azimuth_convention")
        if declared is None:
            declared = assumed
            if expected is not None and declared is None:
                raise ValueError(
                    "a response document declares no azimuth_convention while "
                    f"the answer key declares {expected!r}; pass "
                    "--assume-response-convention to state the assumption "
                    "explicitly, it is recorded in the output")
        if expected is not None and declared is not None and declared != expected:
            raise ValueError(
                f"response azimuth convention {declared!r} disagrees with the "
                f"answer key {expected!r}; the angular errors would be wrong "
                "by roughly twice the angle")
        used = declared or used
    return expected, used


def score(answer_key, response_documents, *, assume_response_convention=None):
    key_convention, response_convention = _response_convention(
        answer_key, response_documents, assume_response_convention)
    keys = {item["item_id"]: item for item in answer_key["items"]}
    rows = []
    seen = set()
    for document in response_documents:
        for response in document.get("responses", []):
            participant = str(response["participant_id"])
            item_id = str(response["item_id"])
            identity = (participant, item_id)
            if identity in seen:
                raise ValueError(f"duplicate participant/item response {identity}")
            seen.add(identity)
            key = keys.get(item_id)
            if key is None:
                raise ValueError(f"unknown calibration item {item_id}")
            binding_correct = (
                str(response["binding_answer"]) == str(key["binding_truth"]))
            numeric = float(response["numeric_answer"])
            if not math.isfinite(numeric):
                raise ValueError(f"non-finite numeric answer for {identity}")
            error = (_circular(numeric, key["numeric_truth"])
                     if key["error_kind"] == "circular_angle_deg"
                     else abs(numeric - float(key["numeric_truth"])))
            rows.append({
                "participant_id": participant,
                "item_id": item_id,
                "profile_id": key["profile_id"],
                "error_kind": key["error_kind"],
                "binding_correct": binding_correct,
                "numeric_error": error,
            })
    by_kind = defaultdict(list)
    for row in rows:
        if row["binding_correct"]:
            by_kind[row["error_kind"]].append(row["numeric_error"])
    participants = sorted({row["participant_id"] for row in rows})
    items = sorted({row["item_id"] for row in rows})
    return {
        "schema": "qa_v3_human_calibration_scores_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "azimuth_convention": key_convention,
        "response_azimuth_convention": response_convention,
        "assumed_response_convention": assume_response_convention,
        "participant_count": len(participants),
        "item_count": len(items),
        "response_count": len(rows),
        "full_av_binding_accuracy": (
            sum(row["binding_correct"] for row in rows) / len(rows)
            if rows else None),
        "numeric_error_on_binding_correct_trials": {
            kind: _quantiles(by_kind.get(kind, []))
            for kind in ("circular_angle_deg", "absolute_time_s")
        },
        "proposed_parameter_mapping": {
            "THETA_FULL": "circular_angle_deg.p75",
            "THETA_HALF": "circular_angle_deg.p95",
            "T_FULL": "absolute_time_s.p75",
            "T_HALF": "absolute_time_s.p95_diagnostic_only_for_card8",
        },
        "boundary": (
            "Decision material only. Numeric errors from binding-wrong trials "
            "are excluded from tolerance quantiles but retained in full-AV "
            "answerability reporting."),
        "records": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assume-response-convention",
                        choices=("right_positive", "left_positive"),
                        help=("state the convention of response files that do "
                              "not declare one; it is echoed into the output"))
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    result = score(
        _read(args.answer_key), [_read(path) for path in args.responses],
        assume_response_convention=args.assume_response_convention)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "participant_count": result["participant_count"],
        "binding_accuracy": result["full_av_binding_accuracy"],
        "quantiles": result["numeric_error_on_binding_correct_trials"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
