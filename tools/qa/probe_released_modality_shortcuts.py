#!/usr/bin/env python3
"""Probe text-, audio-, or video-only shortcuts from final released media.

This is an empirical lower-bound attack, not a proof that an untested optimal
unimodal strategy fails.  Audio features are the existing publication-WAV-only
physical probe.  Video features are computed only from decoded RGB frames.
Text features use question/options character n-grams.  Classification and
numeric Open forms are reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_physical_features import FEATURE_NAMES, extract_features, probe  # noqa: E402
from score_open_answers import (  # noqa: E402
    DEFAULT_VOCAB, angle_credit_radius, resolve_angle_policy, score_item,
    score_time,
)


def _text_matrix(items):
    texts = [str(item["question"]) + " " + " ".join(item.get("options", []))
             for item in items]
    grams = sorted({text[i:i + n] for text in texts for n in (1, 2, 3)
                    for i in range(max(0, len(text) - n + 1))})
    index = {gram: i for i, gram in enumerate(grams)}
    matrix = np.zeros((len(texts), len(grams)), dtype=np.float64)
    for row, text in enumerate(texts):
        for n in (1, 2, 3):
            for i in range(max(0, len(text) - n + 1)):
                matrix[row, index[text[i:i + n]]] += 1.0
    return matrix


def _video_vector(path):
    capture = cv2.VideoCapture(str(path))
    rows, previous = [], None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(frame, (64, 36)).astype(np.float64) / 255.0
        rgb = small[:, :, ::-1]
        means = rgb.mean(axis=(0, 1)); stds = rgb.std(axis=(0, 1))
        left = rgb[:, :32].mean(axis=(0, 1)); right = rgb[:, 32:].mean(axis=(0, 1))
        motion = 0.0 if previous is None else float(np.abs(rgb - previous).mean())
        rows.append(np.r_[means, stds, left - right, motion])
        previous = rgb
    capture.release()
    if not rows:
        raise ValueError(f"video has no decodable frames: {path}")
    values = np.asarray(rows)
    return np.r_[values.mean(axis=0), values.std(axis=0), values[0], values[-1]]


def _features(items, modality):
    if modality == "text":
        return _text_matrix(items)
    if modality == "audio":
        return np.asarray([
            [extract_features(item["audio"])[name] for name in FEATURE_NAMES]
            for item in items], dtype=np.float64)
    return np.asarray([_video_vector(item["video"]) for item in items],
                      dtype=np.float64)


def _standardize(train, test):
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (test - mean) / std


def _numeric_predictions(x, y, kind, folds):
    predictions = np.zeros(len(y), dtype=np.float64)
    fold_ids = np.arange(len(y)) % max(2, min(folds, len(y)))
    for fold in sorted(set(fold_ids)):
        test = fold_ids == fold; train = ~test
        xtr, xte = _standardize(x[train], x[test])
        design = np.c_[xtr, np.ones(train.sum())]
        test_design = np.c_[xte, np.ones(test.sum())]
        ridge = 1e-2 * np.eye(design.shape[1])
        if kind == "numeric_angle":
            radians = np.radians(y[train])
            target = np.c_[np.sin(radians), np.cos(radians)]
            weights = np.linalg.pinv(design.T @ design + ridge) @ design.T @ target
            raw = test_design @ weights
            predictions[test] = np.degrees(np.arctan2(raw[:, 0], raw[:, 1]))
        else:
            weights = np.linalg.pinv(design.T @ design + ridge) @ design.T @ y[train]
            predictions[test] = test_design @ weights
    return predictions


def _constant_baseline(y, kind):
    if kind == "numeric_angle":
        radians = np.radians(y)
        value = math.degrees(math.atan2(np.sin(radians).mean(),
                                        np.cos(radians).mean()))
    else:
        value = float(np.median(y))
    return np.full(len(y), value, dtype=np.float64)


def _numeric_score(predictions, truths, kind, params, items=None):
    scores = []
    metadata = [{} for _ in truths] if items is None else items
    for prediction, truth, item in zip(predictions, truths, metadata, strict=True):
        if kind == "numeric_angle":
            scored_item = {
                "answer_type": "angle_deg", "truth": float(truth),
                "model_answer": f"{prediction} deg",
                **{key: item[key] for key in (
                    "convention", "truth_interval_deg", "certification_policy")
                   if item.get(key) is not None},
            }
            result = score_item(scored_item, params, DEFAULT_VOCAB)
        elif kind == "numeric_time":
            result = score_time(
                f"{prediction} s", float(truth),
                params["T_FULL"], params["T_HALF"],
                strict_certification=True)
        else:
            raise ValueError(f"unsupported numeric task type {kind!r}")
        if result["status"] != "scored":
            raise ValueError(f"numeric probe item could not be scored: {result}")
        scores.append(float(result["score"]))
    return float(np.mean(scores)), scores



SCORING_KEYS = ("THETA_FULL", "THETA_HALF", "T_FULL", "T_HALF")


def scoring_snapshot(params):
    """Parameters the probe's numeric scorers actually execute (fail closed)."""
    missing = [key for key in SCORING_KEYS if key not in params]
    if missing:
        raise ValueError(f"params missing explicit scoring keys {missing}")
    # Validate the angle tolerance relationship even for a classification-only
    # probe.  The snapshot is the public record of the parameters the probe
    # accepts, so malformed values must not survive merely because this run has
    # no numeric-angle group.
    angle_credit_radius(params)
    snapshot = {key: float(params[key]) for key in SCORING_KEYS}
    snapshot["T_FULL_status"] = str(params.get(
        "T_FULL_status", "unspecified_treat_as_placeholder"))
    snapshot["time_certification_policy"] = "strict_full_credit_only"
    snapshot["time_wide_tolerance_role"] = "diagnostic_only"
    snapshot["angle_policy"] = resolve_angle_policy(params)
    snapshot["angle_policy_scope"] = "parameter default; explicit item policies are retained per group"
    return snapshot


def run(items, modality, params, folds):
    scoring = scoring_snapshot(params)
    groups = {}
    for item in items:
        key = (item["profile_id"], item["form"], item["task_type"])
        groups.setdefault(key, []).append(item)
    records = []
    for (profile, form, task), rows in sorted(groups.items()):
        if len(rows) < 4:
            records.append({"profile_id": profile, "form": form,
                            "task_type": task, "status": "insufficient_n",
                            "n": len(rows)})
            continue
        x = _features(rows, modality)
        truths = [row["truth"] for row in rows]
        if task == "classification":
            accuracy, predictions = probe(
                x, [str(value) for value in truths], folds=folds,
                seed=f"{modality}|{profile}|{form}")
            majority = Counter(str(value) for value in truths).most_common(1)[0][1] / len(rows)
            metric = {"accuracy": accuracy,
                      "empirical_majority_baseline": majority}
        else:
            y = np.asarray(truths, dtype=np.float64)
            predictions = _numeric_predictions(x, y, task, folds)
            baseline = _constant_baseline(y, task)
            score, per_item = _numeric_score(predictions, y, task, params, rows)
            baseline_score, _ = _numeric_score(baseline, y, task, params, rows)
            metric = {"mean_scorer_score": score,
                      "empirical_constant_baseline": baseline_score}
            if task == "numeric_angle":
                metric["angle_policies"] = sorted({
                    resolve_angle_policy(params, row.get("certification_policy"))
                    for row in rows})
        records.append({
            "profile_id": profile, "form": form, "task_type": task,
            "status": "research_probe_complete", "n": len(rows),
            **metric,
            "predictions": [
                {"question_id": row["question_id"], "truth": row["truth"],
                 **{key: row[key] for key in (
                     "truth_interval_deg", "convention", "certification_policy")
                    if row.get(key) is not None},
                 "prediction": (None if predictions[index] is None
                                else str(predictions[index]))}
                for index, row in enumerate(rows)],
        })
    return {
        "schema": "qa_v3_released_modality_shortcut_probe_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "modality": modality,
        "scoring_params": scoring,
        "records": records,
        "boundary": (
            "Released-media empirical shortcut probe only. Failure to exceed "
            "the empirical baseline does not prove an untested unimodal "
            "strategy cannot succeed."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True)
    parser.add_argument("--modality", choices=("text", "audio", "video"),
                        required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    try:
        result = run(json.load(open(args.items)), args.modality,
                     json.load(open(args.params)), args.folds)
    except ValueError as exc:
        print(f"probe refused: {exc}", file=sys.stderr)
        return 2
    result["params_source"] = {"path": os.path.abspath(args.params)}
    with open(args.output, "w") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": os.path.abspath(args.output),
                      "modality": args.modality,
                      "groups": len(result["records"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
