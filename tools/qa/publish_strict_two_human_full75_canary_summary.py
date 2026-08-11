#!/usr/bin/env python3
"""Publish a reviewable four-row summary for strict two-human full75 canaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

CANARY_COUNT = 4
REVIEW_FRAME_INDICES = (0, 37, 74)
THUMBNAIL_SIZE = (320, 180)
LABEL_HEIGHT = 24


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _review_frame(output_root: Path, frame_index: int) -> Path:
    path = output_root / "rgb_frames" / f"frame_{frame_index:06d}.png"
    _require(path.is_file(), f"review RGB frame missing: {path}")
    return path


def _contact_sheet(rows: Sequence[dict[str, Any]], output: Path) -> None:
    width = THUMBNAIL_SIZE[0] * len(REVIEW_FRAME_INDICES)
    row_height = THUMBNAIL_SIZE[1] + LABEL_HEIGHT
    sheet = Image.new("RGB", (width, row_height * len(rows)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y = row_index * row_height
        for column, frame_index in enumerate(REVIEW_FRAME_INDICES):
            with Image.open(row["review_frames"][str(frame_index)]) as source:
                thumbnail = ImageOps.fit(
                    source.convert("RGB"),
                    THUMBNAIL_SIZE,
                    method=Image.Resampling.LANCZOS,
                )
            x = column * THUMBNAIL_SIZE[0]
            sheet.paste(thumbnail, (x, y + LABEL_HEIGHT))
            label = (
                f"C{row['canary_index']} {row['target_identity_key']} target "
                f"{row['target_side'].upper()} | f{frame_index:02d}"
            )
            draw.rectangle(
                (x, y, x + THUMBNAIL_SIZE[0], y + LABEL_HEIGHT), fill=(0, 0, 0)
            )
            draw.text((x + 7, y + 6), label, fill=(255, 255, 255), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)


def _review_markdown(rows: Sequence[dict[str, Any]], contact_sheet: Path) -> str:
    lines = [
        "# Strict two-human full75 canary review",
        "",
        "Machine gate: 4/4 PASS. These are research canaries, not formal dataset rows.",
        "",
        f"![Four-row contact sheet]({contact_sheet.name})",
        "",
        "| Canary | Target / distractor | Side | Target visible | Distractor visible | Video |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {canary_index} | {target_identity_key} / {distractor_identity_key} "
            "| {target_side} | {target_fraction:.3f} | {distractor_fraction:.3f} "
            "| [5 s binaural]({video}) |".format(
                canary_index=row["canary_index"],
                target_identity_key=row["target_identity_key"],
                distractor_identity_key=row["distractor_identity_key"],
                target_side=row["target_side"],
                target_fraction=row["minimum_target_visible_fraction_during_speech"],
                distractor_fraction=row["minimum_distractor_visible_fraction"],
                video=row["binaural_video"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Boundary: the four-row gate authorizes only the first 20 single-room "
                "mechanism pilots. The final 100-row multi-room batch remains blocked "
                "until at least three real rooms pass visual and exact-acoustic closure."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def publish(canary_plan_path: Path, finalization_root: Path, output: Path) -> Path:
    plan = _load(canary_plan_path)
    canaries = plan.get("canaries")
    _require(isinstance(canaries, list), "canary plan rows are missing")
    _require(
        len(canaries) == CANARY_COUNT, "canary plan must contain exactly four rows"
    )
    by_index = {int(item["canary_index"]): item for item in canaries}
    _require(
        set(by_index) == set(range(1, CANARY_COUNT + 1)), "canary index closure failed"
    )

    rows: list[dict[str, Any]] = []
    for canary_index in range(1, CANARY_COUNT + 1):
        canary = by_index[canary_index]
        finalization_path = (
            finalization_root / f"canary_{canary_index:02d}" / "finalization.json"
        )
        finalization = _load(finalization_path)
        _require(
            finalization.get("schema")
            == "avengine_native_strict_two_human_full75_canary_finalization_v1"
            and finalization.get("status") == "pass"
            and finalization.get("full75_canary_pass") is True,
            f"canary {canary_index} finalization is not PASS",
        )
        _require(
            finalization.get("canary_index") == canary_index
            and finalization.get("episode_id") == canary["episode_id"],
            f"canary {canary_index} identity closure failed",
        )
        _require(
            finalization.get("captured_frame_count") == 75
            and finalization.get("duration_seconds") == 5,
            f"canary {canary_index} is not a full75 five-second Episode",
        )
        output_root = Path(canary["output_root"])
        review_frames = {
            str(frame_index): str(_review_frame(output_root, frame_index))
            for frame_index in REVIEW_FRAME_INDICES
        }
        pixels = finalization["pixels"]
        artifacts = finalization["artifacts"]
        row = {
            "canary_index": canary_index,
            "episode_id": canary["episode_id"],
            "target_identity_key": canary["target_identity_key"],
            "distractor_identity_key": canary["distractor_identity_key"],
            "target_side": canary["target_side"],
            "captured_frame_count": 75,
            "duration_seconds": 5,
            "normal_rgb_frame_count": finalization["native_arrays"][
                "normal_rgb_frame_count"
            ],
            "metric_depth_frame_count": finalization["native_arrays"][
                "metric_depth_frame_count"
            ],
            "target_only_frame_count": (
                finalization["native_arrays"]["source1_target_only_frame_count"]
                + finalization["native_arrays"]["source2_target_only_frame_count"]
            ),
            "minimum_target_visible_fraction_during_speech": pixels[
                "minimum_target_visible_fraction_during_speech"
            ],
            "minimum_distractor_visible_fraction": pixels[
                "minimum_distractor_visible_fraction"
            ],
            "minimum_target_visible_pixels_during_speech": pixels[
                "minimum_target_visible_pixels_during_speech"
            ],
            "minimum_distractor_visible_pixels": pixels[
                "minimum_distractor_visible_pixels"
            ],
            "physical_gpu_index": finalization["gpu"]["physical_index"],
            "binaural_video": artifacts["binaural_video"],
            "binaural_wav": artifacts["binaural_wav"],
            "pixel_visibility_truth": artifacts["pixel_visibility_truth"],
            "runtime_asset_readbacks": artifacts["runtime_asset_readbacks"],
            "finalization": str(finalization_path.resolve()),
            "review_frames": review_frames,
            "formal": False,
            "qualification_claim": False,
            "status": "pass",
        }
        rows.append(row)

    _require(
        len({row["episode_id"] for row in rows}) == CANARY_COUNT, "duplicate Episode"
    )
    _require(
        Counter(row["target_side"] for row in rows) == {"left": 2, "right": 2},
        "target-side canary balance drift",
    )
    _require(
        Counter(
            (row["target_identity_key"], row["distractor_identity_key"]) for row in rows
        )
        == {("M", "F"): 2, ("F", "M"): 2},
        "identity-order canary balance drift",
    )
    _require(
        all(row["physical_gpu_index"] == 1 for row in rows),
        "canary used a forbidden physical GPU",
    )

    output.mkdir(parents=True, exist_ok=True)
    contact_sheet = output / "contact_sheet_f00_f37_f74.png"
    _contact_sheet(rows, contact_sheet)
    rows_path = output / "rows.jsonl"
    rows_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    (output / "REVIEW.md").write_text(
        _review_markdown(rows, contact_sheet), encoding="utf-8"
    )
    summary = {
        "schema": "avengine_native_strict_two_human_full75_canary_summary_v1",
        "status": "pass",
        "canary_count": CANARY_COUNT,
        "machine_pass_count": CANARY_COUNT,
        "captured_episode_seconds": sum(row["duration_seconds"] for row in rows),
        "normal_rgb_frame_count": sum(row["normal_rgb_frame_count"] for row in rows),
        "metric_depth_frame_count": sum(
            row["metric_depth_frame_count"] for row in rows
        ),
        "target_only_frame_count": sum(row["target_only_frame_count"] for row in rows),
        "native_render_frame_count": sum(
            row["normal_rgb_frame_count"] + row["target_only_frame_count"]
            for row in rows
        ),
        "target_side_counts": dict(Counter(row["target_side"] for row in rows)),
        "identity_order_counts": {
            f"{target}/{distractor}": count
            for (target, distractor), count in Counter(
                (row["target_identity_key"], row["distractor_identity_key"])
                for row in rows
            ).items()
        },
        "minimum_target_visible_fraction_during_speech": min(
            row["minimum_target_visible_fraction_during_speech"] for row in rows
        ),
        "minimum_distractor_visible_fraction": min(
            row["minimum_distractor_visible_fraction"] for row in rows
        ),
        "minimum_target_visible_pixels_during_speech": min(
            row["minimum_target_visible_pixels_during_speech"] for row in rows
        ),
        "minimum_distractor_visible_pixels": min(
            row["minimum_distractor_visible_pixels"] for row in rows
        ),
        "single_room_mechanism_pilot_authorized": True,
        "single_room_mechanism_pilot_target": 20,
        "final_multi_room_100_authorized": False,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "rows": rows,
        "artifacts": {
            "rows_jsonl": str(rows_path.resolve()),
            "contact_sheet": str(contact_sheet.resolve()),
            "review_markdown": str((output / "REVIEW.md").resolve()),
        },
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    return summary_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-plan", required=True, type=Path)
    parser.add_argument("--finalization-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = publish(args.canary_plan, args.finalization_root, args.output)
    print(f"STRICT_TWO_HUMAN_FULL75_CANARY_SUMMARY_OK summary={path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
