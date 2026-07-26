#!/usr/bin/env python3
"""Compare same-checkpoint same-room and cross-room evaluation reports."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping

import avengine_v43.evaluation as evaluation_module
import avengine_v43.publication as publication_module
from avengine_v43.evaluation import compare_evaluation_reports
from avengine_v43.publication import atomic_publish_directory


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation report must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _producer_identity() -> Mapping[str, Any]:
    paths = {
        "compare_evaluations.py": Path(__file__).resolve(),
        "avengine_v43/evaluation.py": Path(evaluation_module.__file__).resolve(),
        "avengine_v43/publication.py": Path(
            publication_module.__file__
        ).resolve(),
    }
    return {
        "schema": "avengine_v43_comparison_producer_v1",
        "code": {
            name: {"sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
    }


def _metric_table(metrics: Mapping[str, Mapping[str, float]]) -> str:
    labels = {
        "mean_absolute_error_deg": "Mean MAE (deg)",
        "median_absolute_error_deg": "Median MAE (deg)",
        "p90_absolute_error_deg": "P90 MAE (deg)",
        "error_over_45deg_rate": "Frames >45 (rate)",
        "error_over_90deg_rate": "Frames >90 (rate)",
        "uniform_target_region_macro_mean_absolute_error_deg": "Region-macro MAE (deg)",
    }
    rows = []
    for key, value in metrics.items():
        rows.append(
            "<tr>"
            f"<td>{escape(labels.get(key, key))}</td>"
            f"<td>{float(value['baseline']):.4f}</td>"
            f"<td>{float(value['candidate']):.4f}</td>"
            f"<td>{float(value['delta']):+.4f}</td></tr>"
        )
    return "".join(rows)


def _write_html(path: Path, comparison: Mapping[str, Any]) -> None:
    training_room_id = escape(str(comparison["training_room_id"]))
    baseline_room_id = escape(str(comparison["baseline_room_id"]))
    candidate_room_id = escape(str(comparison["candidate_room_id"]))
    path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>AVEngine v4.3 room generalization comparison</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;background:#f7f8fa;color:#20242a}}
table{{border-collapse:collapse;background:white}}th,td{{border:1px solid #ddd;padding:8px}}
th{{background:#eef2f6}}</style></head><body>
<h1>Same-room → cross-room zero-shot comparison</h1>
<p>Training room: {training_room_id}; same-room baseline: {baseline_room_id};
cross-room candidate: {candidate_room_id}. Research-only; both reports use the
same model and CLAP checkpoints. Positive deltas mean larger error.</p>
<table><thead><tr><th>Metric</th><th>{baseline_room_id}</th>
<th>{candidate_room_id}</th><th>Delta</th></tr>
</thead><tbody>{_metric_table(comparison['metrics'])}</tbody></table>
<p><a href="comparison.json">Machine-readable comparison</a></p>
</body></html>""",
        encoding="utf-8",
    )


def main() -> int:
    args = _arguments()
    baseline_path = args.baseline_evaluation.resolve()
    candidate_path = args.candidate_evaluation.resolve()
    output = args.output_root.resolve()
    for path in (baseline_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    comparison = compare_evaluation_reports(
        _load(baseline_path), _load(candidate_path)
    )
    comparison["inputs"] = {
        "baseline_evaluation": str(baseline_path),
        "baseline_evaluation_sha256": _sha256_file(baseline_path),
        "candidate_evaluation": str(candidate_path),
        "candidate_evaluation_sha256": _sha256_file(candidate_path),
    }
    comparison["comparison_producer"] = _producer_identity()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging output: {staging}")
    try:
        staging.mkdir()
        with (staging / "comparison.json").open("x", encoding="utf-8") as handle:
            json.dump(comparison, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _write_html(staging / "REVIEW_INDEX.html", comparison)
        atomic_publish_directory(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"ROOM_EVALUATION_COMPARISON_OK output={output} "
        f"mean_delta={comparison['metrics']['mean_absolute_error_deg']['delta']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
