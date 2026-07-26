#!/usr/bin/env python3
"""Compare EDT/DRR/late-energy between two retained RIR caches on matched jobs.

Both caches must cover the same RIR job plan. Jobs are matched by exact
``job_id``; a deterministic seeded sample keeps the comparison affordable on
full production caches. Metrics come from the tested M3 ``analyze_ir``
measurement path; this tool adds no new acoustic claims and its output is a
research diagnostic, not a calibration result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.m3.metrics import AcousticMetricError, analyze_ir  # noqa: E402


def _load_cache_index(cache: Path) -> tuple[dict[str, tuple[Path, int]], float]:
    """Map job_id -> (shard path, row) without loading payloads."""

    shards = sorted((cache / "shards").glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards found under {cache}")
    mapping: dict[str, tuple[Path, int]] = {}
    sample_rate = None
    for shard in shards:
        with np.load(shard, allow_pickle=False) as value:
            job_ids = np.asarray(value["job_ids"])
            rate = float(np.asarray(value["sample_rate_hz"]).reshape(-1)[0])
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise SystemExit(f"inconsistent sample rate inside {cache}")
        for row, job_id in enumerate(job_ids):
            mapping[str(job_id)] = (shard, row)
    assert sample_rate is not None
    return mapping, sample_rate


def _load_ir(entry: tuple[Path, int]) -> np.ndarray:
    shard, row = entry
    with np.load(shard, allow_pickle=False) as value:
        length = int(np.asarray(value["lengths"])[row])
        samples = np.asarray(value["samples"])[row, :, :length]
    return np.asarray(samples, dtype=np.float64)


def _stable_sample(job_ids: list[str], count: int, seed: int) -> list[str]:
    def key(job_id: str) -> str:
        return hashlib.sha256(f"{seed}\0{job_id}".encode("utf-8")).hexdigest()

    return sorted(job_ids, key=key)[: max(0, count)]


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--label-a", default="cache_a")
    parser.add_argument("--label-b", default="cache_b")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index_a, rate_a = _load_cache_index(args.cache_a.resolve())
    index_b, rate_b = _load_cache_index(args.cache_b.resolve())
    shared = sorted(set(index_a) & set(index_b))
    if not shared:
        raise SystemExit("caches share no job_ids; nothing to compare")
    selected = _stable_sample(shared, args.sample_count, args.seed)

    metrics_a: dict[str, list[float]] = {"edt_seconds": [], "drr_db": [], "late_energy": []}
    metrics_b: dict[str, list[float]] = {"edt_seconds": [], "drr_db": [], "late_energy": []}
    deltas: dict[str, list[float]] = {"edt_seconds": [], "drr_db": [], "late_energy": []}
    skipped: list[dict[str, str]] = []
    for job_id in selected:
        try:
            result_a = analyze_ir(_load_ir(index_a[job_id]), rate_a)
            result_b = analyze_ir(_load_ir(index_b[job_id]), rate_b)
        except AcousticMetricError as error:
            skipped.append({"job_id": job_id, "reason": str(error)})
            continue
        for name, value_a, value_b in (
            ("edt_seconds", result_a.edt_seconds, result_b.edt_seconds),
            ("drr_db", result_a.drr_db, result_b.drr_db),
            ("late_energy", result_a.late_energy, result_b.late_energy),
        ):
            metrics_a[name].append(float(value_a))
            metrics_b[name].append(float(value_b))
            deltas[name].append(float(value_b) - float(value_a))

    if not metrics_a["edt_seconds"]:
        raise SystemExit("no matched jobs produced finite metrics")

    report = {
        "schema": "avengine_m7_rir_cache_metric_comparison_v1",
        "qualification_claim": False,
        "claim_boundary": (
            "Matched-job EDT/DRR/late-energy diagnostic between two retained "
            "RIR caches; no physical room-acoustic truth or calibration claim"
        ),
        "cache_a": {"label": args.label_a, "path": str(args.cache_a.resolve()), "sample_rate_hz": rate_a},
        "cache_b": {"label": args.label_b, "path": str(args.cache_b.resolve()), "sample_rate_hz": rate_b},
        "shared_job_count": len(shared),
        "requested_sample_count": args.sample_count,
        "measured_pair_count": len(metrics_a["edt_seconds"]),
        "skipped": skipped,
        "seed": args.seed,
        "metrics": {
            name: {
                args.label_a: _summary(metrics_a[name]),
                args.label_b: _summary(metrics_b[name]),
                "delta_b_minus_a": _summary(deltas[name]),
            }
            for name in metrics_a
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    concise = {
        name: {
            "mean_a": report["metrics"][name][args.label_a]["mean"],
            "mean_b": report["metrics"][name][args.label_b]["mean"],
            "mean_delta": report["metrics"][name]["delta_b_minus_a"]["mean"],
        }
        for name in metrics_a
    }
    print("RIR_CACHE_COMPARE_OK", json.dumps(concise))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
