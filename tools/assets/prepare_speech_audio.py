#!/usr/bin/env python3
"""Build the P7 speech-band prepared set and review manifests.

This entry point writes only a new derived output tree. It does not modify the
source PCM, the event registry, or historical Episode metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.assets.sound_prepare import (  # noqa: E402
    activity_profile_for_class,
    build_nonverbal_source_inventory,
    listening_sample_records,
    prepare_speech_registry,
)


def _write_json_no_clobber(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_listening_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    lines = [
        "# P7 prepared speech listening log",
        "",
        "Human review status: pending_human",
        "No human or model listening result is recorded by this file.",
        "Fill reviewer, heard, consonant_preserved, and notes after real listening.",
        "",
        "| sample | prepared_audio_id | gender | speaker | utterance | source_activity_s | source_span_s | reviewer | heard | consonant_preserved | notes |",
        "|---:|---|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        def cell(value: Any) -> str:
            return str(value if value is not None else "").replace("|", "\\|")
        lines.append(
            "| {sample_index} | {prepared_audio_id} | {gender} | {speaker_id} | "
            "{utterance_id} | {source_activity_duration_s} | "
            "{source_audible_span_s} | {reviewer} | {heard} | "
            "{consonant_preserved} | {notes} |".format(
                **{key: cell(value) for key, value in row.items()}
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-library-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--nonverbal-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--include-non-vctk", action="store_true")
    parser.add_argument("--prepared-set-id", default="speech_band_prepared_20260906_v1")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260906)
    args = parser.parse_args(argv)

    manifest = prepare_speech_registry(
        args.registry.resolve(),
        args.output_root.resolve(),
        source_library_root=args.source_library_root.resolve(),
        prepared_set_id=args.prepared_set_id,
        vctk_only=not args.include_non_vctk,
    )
    output = args.output_root.resolve()
    inventory = build_nonverbal_source_inventory(args.nonverbal_csv.resolve())
    _write_json_no_clobber(output / "nonverbal_source_inventory.json", inventory)
    sample_rows = listening_sample_records(
        manifest, count=args.sample_count, seed=args.sample_seed
    )
    _write_json_no_clobber(output / "listening_samples_pending.json", sample_rows)
    _write_listening_log(output / "listening_log_pending.md", sample_rows)

    payload = json.loads(args.registry.read_text(encoding="utf-8"))
    classes = sorted(
        {
            str(row.get("semantic_sound_class"))
            for row in payload.get("sound_assets", [])
            if isinstance(row, dict) and row.get("semantic_sound_class")
        }
    )
    _write_json_no_clobber(
        output / "activity_profiles.json",
        {
            "schema": "avengine_sound_activity_profiles_v1",
            "calibration": "placeholder",
            "profiles": [activity_profile_for_class(name) for name in classes],
        },
    )
    print(
        json.dumps(
            {
                "prepared_set_id": manifest["prepared_set_id"],
                "output_root": str(output),
                "prepared": manifest["counts"].get("prepared", 0),
                "counts_by_gender": manifest["counts_by_gender"],
                "human_review_status": "pending_human",
                "nonverbal_independent_source_counts": inventory.get(
                    "independent_source_counts", {}
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
