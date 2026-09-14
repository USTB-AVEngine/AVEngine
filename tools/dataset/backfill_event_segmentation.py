#!/usr/bin/env python3
"""Measure how many sound events each pool recording holds and record it.

QA-23 counts how many independent sound events begin in a clip, and it
refuses to ask the question until every non-speech recording the episode
plays carries a segmentation record.  No sound pool wrote one before
2026-09-14, so every episode with a non-speech event deferred QA-23 with
``event_segmentation_not_reviewed``.

This is a thin entry point over ``avengine.dataset.event_segmentation``.  It
reads one pool, measures each eligible recording read-only through the one
uniform detector policy, and writes two new files: a full copy of the pool
with an ``event_segmentation`` field on each measured row and an
``event_segmentation_backfill`` block at the top, and a sidecar indexed by
``sound_asset_id`` for overlaying the same records onto episodes that were
rendered before the pool had them.  The input pool is opened read-only and is
never modified, and neither output is overwritten unless ``--force`` says so.

A recording whose sounding stretches are separated by more than
``--gap-max-s`` is reported ``multi_event_candidate``, not certified.  That is
not a failure to fix by raising the threshold: an episode that plays such a
recording keeps deferring QA-23 until a person decides what it is.

    python tools/dataset/backfill_event_segmentation.py \
        --pool T06_SOUND_POOL.json \
        --out-pool T06_SOUND_POOL_segmented_v1.json \
        --sidecar event_segmentation_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.event_segmentation import (  # noqa: E402
    DEFAULT_GAP_MAX_S,
    MIN_PULSE_TRAIN_ONSETS,
    backfill_pool,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", required=True, type=Path,
                        help="sound pool JSON to read (never modified)")
    parser.add_argument("--out-pool", required=True, type=Path,
                        help="where to write the pool copy with the records")
    parser.add_argument("--sidecar", required=True, type=Path,
                        help="where to write the by-sound_asset_id sidecar")
    parser.add_argument("--report", type=Path, default=None,
                        help="optional statistics-only JSON (no records)")
    parser.add_argument("--gap-max-s", type=float, default=DEFAULT_GAP_MAX_S,
                        help="silence longer than this starts a new event "
                             f"(default {DEFAULT_GAP_MAX_S})")
    parser.add_argument("--min-pulse-train-onsets", type=int,
                        default=MIN_PULSE_TRAIN_ONSETS,
                        help="onsets needed before regular spacing may count "
                             "as one repeating sound")
    parser.add_argument("--relative-peak-db", type=float, default=None,
                        help="override the gate's dB under the clip peak")
    parser.add_argument("--window-s", type=float, default=None,
                        help="override the analysis window")
    parser.add_argument("--hop-s", type=float, default=None,
                        help="override the analysis hop")
    parser.add_argument("--gap-merge-s", type=float, default=None,
                        help="override the dip-merge that absorbs micro-gaps")
    parser.add_argument("--include-speech", action="store_true",
                        help="also measure rows QA-23 exempts as speech")
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="re-measure rows that already carry a record")
    parser.add_argument("--force", action="store_true",
                        help="allow replacing existing output files")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="print a line every N measured rows (0 = silent)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for target in (args.out_pool, args.sidecar, args.report):
        if target is not None and target.exists() and not args.force:
            raise SystemExit(f"{target} already exists; pass --force to replace it")
    pool_path = args.pool.expanduser().resolve()
    payload = json.loads(pool_path.read_text(encoding="utf-8"))

    overrides = {
        key: getattr(args, key)
        for key in ("relative_peak_db", "window_s", "hop_s", "gap_merge_s")
        if getattr(args, key) is not None
    }
    seen = {"count": 0}
    started = time.monotonic()

    def on_row(_index: int, sound_id: str, record: dict) -> None:
        seen["count"] += 1
        if args.progress_every and seen["count"] % args.progress_every == 0:
            print(f"  measured {seen['count']}  last {sound_id} "
                  f"{record['status']}", flush=True)

    out_payload, sidecar = backfill_pool(
        payload,
        pool_path=pool_path,
        gap_max_s=args.gap_max_s,
        min_pulse_train_onsets=args.min_pulse_train_onsets,
        include_speech=args.include_speech,
        overwrite_existing=args.overwrite_existing,
        policy_overrides=overrides,
        on_row=on_row,
    )
    elapsed = time.monotonic() - started
    statistics = sidecar["statistics"]
    statistics["elapsed_s"] = elapsed

    for target, value in ((args.out_pool, out_payload), (args.sidecar, sidecar)):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"source_pool": str(pool_path),
                        "parameters": sidecar["parameters"],
                        "statistics": statistics}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")

    counts = statistics["counts"]
    print(f"rows {counts['rows']}  measured {counts['measured']}  "
          f"skipped_speech {counts['skipped_speech']}  "
          f"skipped_existing {counts['skipped_existing']}  "
          f"failed {counts['failed']}  in {elapsed:.1f}s")
    print(f"certified automatically {statistics['certified']}  "
          f"not certified {statistics['not_certified']}  "
          f"human reviewed {statistics['human_reviewed']}")
    print(f"by rule {statistics['by_rule']}")
    for name in sorted(statistics["by_sound_class"]):
        row = statistics["by_sound_class"][name]
        print(f"  {name:30s} certified {row['certified']:4d} / {row['measured']:4d}")
    for failure in statistics["failures"]:
        print(f"  FAILED {failure['sound_asset_id']}: {failure['error']}")
    print(f"pool    {args.out_pool}")
    print(f"sidecar {args.sidecar}")
    return 0 if statistics["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
