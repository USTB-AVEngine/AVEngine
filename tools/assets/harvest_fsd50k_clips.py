#!/usr/bin/env python3
"""Fill the sound-effect classes from the FSD50K copy already on this machine.

Nothing is downloaded and nothing is chosen by hand: the mapping file says
which FSD50K label serves which of our event classes, and this script pulls
up to the target number of clips per class out of the single-label subset -
the one where each recording carries exactly one label, so a dog bark does
not arrive with a conversation underneath it.

A recording already in the library is never copied a second time. If it
turns out to serve another class too, the extra class is added to its
sidecar, because two files holding the same audio would let a question ask
which of two sources is sounding when both are playing the same waveform.

Run the QC and preparation tools afterwards; this script only collects.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.assets.sound_harvest import (  # noqa: E402
    plan_fsd50k_harvest,
    sidecar_for_effect,
)


def _existing(library_root: Path) -> tuple[dict[str, list[str]], dict[str, Path]]:
    """FSD50K ids already in the library, and where their sidecars live."""

    classes: dict[str, list[str]] = {}
    sidecars: dict[str, Path] = {}
    for sidecar in sorted(library_root.glob("*/fsd50k_*/clip.json")):
        fsd_id = sidecar.parent.name.removeprefix("fsd50k_")
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            continue
        classes.setdefault(fsd_id, []).extend(
            str(name) for name in record.get("event_classes") or []
        )
        sidecars.setdefault(fsd_id, sidecar)
    return classes, sidecars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=REPOSITORY / "examples/assets/sound_harvest_map_v1.json",
    )
    parser.add_argument(
        "--pool-csv",
        type=Path,
        default=Path("/data/datasets/omniaudio/tse_data/single_label_output.csv"),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("/data/datasets/omniaudio/source_data/FSD50K"),
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path("/data/avengine_external/assets/sound_library_v1"),
    )
    parser.add_argument("--target-per-class", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if mapping.get("review_status") == "draft_pending_review":
        print(
            "注意:对照表还标着 draft_pending_review(等人核对)。"
            "先跑也可以,核对后重跑会自动补齐差额。\n"
        )
    with args.pool_csv.open(encoding="utf-8") as handle:
        pool_rows = list(csv.DictReader(handle))

    present, sidecars = _existing(args.library_root)
    plan = plan_fsd50k_harvest(
        mapping,
        pool_rows,
        already_present=present,
        target_per_class=args.target_per_class,
    )

    copied = missing = 0
    for pick in plan["picks"]:
        source = args.audio_root / f"{pick['fsd50k_id']}.wav"
        if not source.is_file():
            missing += 1
            continue
        target_dir = args.library_root / pick["relative_dir"]
        if args.dry_run:
            copied += 1
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / "clip.wav")
        (target_dir / "clip.json").write_text(
            json.dumps(
                sidecar_for_effect(pick, [pick["event_class"]]),
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        copied += 1

    extended = 0
    for fsd_id, classes in plan["extra_classes_for_existing"].items():
        sidecar = sidecars.get(fsd_id)
        if sidecar is None or args.dry_run:
            continue
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        names = list(record.get("event_classes") or [])
        for name in classes:
            if name not in names:
                names.append(name)
        record["event_classes"] = names
        sidecar.write_text(
            json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        extended += 1

    if plan["shortfalls"]:
        print("下面这些类在单标签池里凑不够,需要放宽目标或去多标签池补:")
        for row in plan["shortfalls"]:
            print(
                f"  {row['event_class']:<28} 还差 {row['wanted'] - row['found']} 条"
                f"(想要 {row['wanted']},只找到 {row['found']})"
            )
    print(
        f"\n{'（演习，未写盘）' if args.dry_run else ''}"
        f"新增 {copied} 条 · 给已有素材补标事件类 {extended} 条"
        f"{f' · 源文件缺失 {missing} 条' if missing else ''}\n"
        f"接下来跑质检和格式处理:\n"
        f"  python3 tools/assets/qc_sound_library.py\n"
        f"  python3 tools/assets/prepare_sound_library.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
