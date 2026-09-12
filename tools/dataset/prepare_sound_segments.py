#!/usr/bin/env python3
"""Select the sounding part of dry recordings and cut it to a time budget.

A thin entry point over ``avengine.dataset.sound_segments``: everything
reusable lives in the module, so a sampler calls the functions directly and
this script exists for censuses, spot checks and one-off preparation.

Two modes:

* ``--plan-only`` indexes recordings and writes coordinates and measurements
  only.  Nothing is cut.  Run this over a whole library to see, per class, how
  many recordings can yield a segment for a given budget and why the rest
  cannot.
* the default also cuts each selected region into fresh PCM under
  ``--output-root`` and reads every file back.

Original recordings are opened read-only and are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.sound_segments import (  # noqa: E402
    DEFAULT_CROP_GUARD_S,
    DEFAULT_EDGE_FADE_S,
    SegmentBudget,
    SoundSegmentError,
    iter_library_clips,
    plan_segments,
    prepare_segments,
)
from avengine.assets.sound_prepare import TARGET_RATE_HZ  # noqa: E402

def _policy_override(pairs: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--policy-override needs KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        raw = raw.strip()
        if key == "activity_family":
            overrides[key] = raw
            continue
        try:
            overrides[key] = float(raw)
        except ValueError as error:
            raise SystemExit(
                f"--policy-override {key} needs a number, got {raw!r}"
            ) from error
    return overrides


def _sources(args: argparse.Namespace) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for path in args.source or []:
        sources.append({
            "source_path": str(path),
            "sound_class": args.sound_class,
            "source_asset_id": Path(path).parent.name,
        })
    if args.library_root is not None:
        classes = None
        if args.classes:
            classes = [name for item in args.classes for name in item.split(",")
                       if name]
        seen: dict[str, int] = {}
        for clip in iter_library_clips(args.library_root, classes=classes):
            bucket = str(clip["sound_class"])
            if args.limit_per_class and seen.get(bucket, 0) >= args.limit_per_class:
                continue
            seen[bucket] = seen.get(bucket, 0) + 1
            sources.append({
                "source_path": clip["source_path"],
                "sound_class": args.sound_class or clip["sound_class"],
                "source_asset_id": clip["source_asset_id"],
                "metadata": {
                    "clip_json": clip.get("clip_json"),
                    "clip_qc_json": clip.get("clip_qc_json"),
                    "clip_json_path": clip.get("clip_json_path"),
                    "clip_qc_json_path": clip.get("clip_qc_json_path"),
                },
            })
    if not sources:
        raise SystemExit("give --library-root or at least one --source")
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library-root", type=Path,
                        help="sound library laid out as <class>/<clip_id>/clip.wav")
    parser.add_argument("--source", type=Path, action="append",
                        help="one original WAV; repeatable")
    parser.add_argument("--sound-class",
                        help="override the class for every source")
    parser.add_argument("--classes", action="append",
                        help="restrict the library walk to these classes")
    parser.add_argument("--limit-per-class", type=int, default=0,
                        help="stop after N recordings per class (0 = no limit)")
    parser.add_argument("--max-duration-s", type=float, required=True,
                        help="the caller's time budget for one segment")
    parser.add_argument("--min-duration-s", type=float)
    parser.add_argument("--crop-guard-s", type=float, default=DEFAULT_CROP_GUARD_S)
    parser.add_argument("--edge-fade-s", type=float, default=DEFAULT_EDGE_FADE_S)
    parser.add_argument("--target-rate-hz", type=int, default=TARGET_RATE_HZ,
                        help="0 keeps each recording's own rate")
    parser.add_argument("--selection", default="longest_active",
                        choices=("longest_active", "earliest", "highest_coverage"))
    parser.add_argument("--temporal-policy", default="declared",
                        choices=("declared", "measured"),
                        help="declared: the class's own coverage and silence "
                             "thresholds. measured: the thresholds matching "
                             "the shape each recording actually has, which "
                             "qualifies more recordings and is therefore "
                             "opt-in and recorded on every plan")
    parser.add_argument("--remove-dc", action="store_true")
    parser.add_argument("--normalize-peak-dbfs", type=float,
                        help="uniform peak normalisation; off by default")
    parser.add_argument("--policy-override", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="applies to every class in the run")
    parser.add_argument("--class-policy-file", type=Path,
                        help="JSON mapping sound class -> policy overrides, so "
                             "one class can be treated as a different temporal "
                             "form without renaming it or changing the others")
    parser.add_argument("--output-root", type=Path,
                        help="where fresh segment PCM is written")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--include-frame-levels", action="store_true",
                        help="write every window level into the report")
    args = parser.parse_args(argv)

    budget = SegmentBudget(
        max_duration_s=args.max_duration_s,
        min_duration_s=args.min_duration_s,
        crop_guard_s=args.crop_guard_s,
        edge_fade_s=args.edge_fade_s,
        target_rate_hz=args.target_rate_hz or None,
        remove_dc=args.remove_dc,
        normalize_peak=args.normalize_peak_dbfs is not None,
        target_peak_dbfs=args.normalize_peak_dbfs,
        selection=args.selection,
        temporal_policy=args.temporal_policy,
    )
    overrides = _policy_override(args.policy_override)
    class_overrides: dict[str, dict[str, object]] = {}
    if args.class_policy_file is not None:
        raw = json.loads(args.class_policy_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SystemExit("--class-policy-file must hold a JSON object")
        for name, value in raw.items():
            if not isinstance(value, dict):
                raise SystemExit(
                    f"--class-policy-file entry {name!r} must be an object")
            class_overrides[str(name)] = _policy_override(
                [f"{key}={item}" for key, item in value.items()])
    sources = _sources(args)

    try:
        if args.plan_only:
            report = plan_segments(
                sources, budget=budget, policy_overrides=overrides,
                class_policy_overrides=class_overrides,
                include_frame_levels=args.include_frame_levels,
            )
        else:
            if args.output_root is None:
                raise SystemExit("--output-root is required unless --plan-only")
            report = prepare_segments(
                sources, args.output_root, budget=budget,
                policy_overrides=overrides,
                class_policy_overrides=class_overrides,
                verify=not args.no_verify,
                include_frame_levels=args.include_frame_levels,
            )
    except SoundSegmentError as error:
        print(f"sound segment error: {error}", file=sys.stderr)
        return 2

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = report["counts"]
    print(f"considered {counts['considered']}  selected {counts['selected']}  "
          f"rejected {counts['rejected']}")
    if not args.plan_only:
        print(f"materialized {counts['materialized']}  "
              f"verification {counts['verification_by_status']}  "
              f"failures {counts['failures']}")
    index = report if args.plan_only else report["index"]
    for name in sorted(index["by_sound_class"]):
        row = index["by_sound_class"][name]
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(row["reasons"].items())
        )
        forms = index["measured_temporal_form_by_class"].get(name, {})
        shape = ", ".join(f"{form}x{count}" for form, count
                          in sorted(forms.items(), key=lambda kv: -kv[1]))
        print(f"  {name:28s} selected {row['selected']:4d} / {row['considered']:4d}"
              + (f"   {reasons}" if reasons else "")
              + (f"   [{shape}]" if shape else ""))
    print(f"report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
