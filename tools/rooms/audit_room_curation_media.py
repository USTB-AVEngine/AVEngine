#!/usr/bin/env python3
"""Export a reproducible QA inventory for room-curation videos and verdicts.

The script is read-only with respect to Studio and the media files.  It joins
the current room-curation API inventory with the V2 prescreen measurements and
writes one JSON record and one CSV row per expected room.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.rooms.runtime_config import (RUNTIME_PREFIX, MAGNUM_SITE, RLR_SDK_ROOT, MP3D_ROOT, TASKS_ROOT, MEDIA_ROOT, ROOM_PYTHON)

import argparse
import csv
import json
import urllib.request
from collections import Counter
from pathlib import Path


DEFAULT_MEDIA_ROOT = Path(str(MEDIA_ROOT))


def load_jsonp(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    prefix = "window.PRESCREEN ="
    if not text.lstrip().startswith(prefix):
        raise ValueError(f"{path} is not a PRESCREEN JSONP file")
    return json.loads(text.lstrip()[len(prefix):].strip().removesuffix(";"))


def video_quality(metrics: dict | None) -> tuple[str, str]:
    if metrics is None:
        return "missing", "巡房视频缺失"
    if metrics.get("read_error"):
        return "decode_error", "ffmpeg 抽帧解码失败"
    black = float(metrics.get("black_ratio") or 0.0)
    if black >= 0.35:
        return "bad_severe_black", f"6个方向抽帧的黑像素比例为{black:.1%}（>=35%）"
    if black >= 0.20:
        return "warning_black", f"6个方向抽帧的黑像素比例为{black:.1%}（20%-35%）"
    return "good", f"视频可解码，6个方向抽帧黑像素比例为{black:.1%}（<20%）"


def verdict_value(value: object) -> tuple[str, str, str]:
    if not isinstance(value, dict):
        return "pending", "", ""
    return (
        str(value.get("verdict") or "pending"),
        str(value.get("author") or ""),
        str(value.get("note") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765/api/room-curation")
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--prescreen", type=Path)
    parser.add_argument("--json-output", type=Path, default=Path("logs/room_media_qa.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("logs/room_media_qa.csv"))
    args = parser.parse_args()
    prescreen_path = args.prescreen or args.media_root / "prescreen.js"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.api_url, timeout=120) as response:
        inventory = json.load(response)
    suggestions = {
        (row["house"], row["label"]): row for row in load_jsonp(prescreen_path)
    }

    records: list[dict] = []
    expected_paths: set[Path] = set()
    for house_record in inventory["houses"]:
        house = house_record["house"]
        for room in house_record["rooms"]:
            label = room["label"]
            video_path = args.media_root / house / f"{label}.mp4"
            expected_paths.add(video_path.resolve())
            suggestion = suggestions.get((house, label), {})
            measurements = suggestion.get("metrics") or {}
            quality, quality_reason = video_quality(measurements.get("video"))
            verdict, author, note = verdict_value(room.get("verdict"))
            camera = measurements.get("camera") or {}
            records.append({
                "house": house,
                "room_label": label,
                "room_type": room.get("room_type") or "未识别",
                "floor_area_m2": room.get("floor_area_m2"),
                "video_path": str(video_path),
                "video_exists": video_path.is_file() and video_path.stat().st_size > 0,
                "video_size_bytes": video_path.stat().st_size if video_path.is_file() else 0,
                "video_quality": quality,
                "video_quality_reason": quality_reason,
                "black_ratio": (measurements.get("video") or {}).get("black_ratio"),
                "dark_ratio": (measurements.get("video") or {}).get("dark_ratio"),
                "camera_strategy": camera.get("placement_strategy") or "legacy_no_metadata",
                "camera_centre_offset_xz_m": camera.get("centre_offset_xz_m"),
                "prescreen_suggestion": suggestion.get("suggestion") or "missing",
                "prescreen_reason": suggestion.get("reason") or "",
                "human_verdict": verdict,
                "human_author": author,
                "human_note": note,
            })

    stale = sorted(
        str(path.resolve())
        for path in args.media_root.glob("*/*.mp4")
        if path.resolve() not in expected_paths
    )
    summary = {
        "schema": "avengine_room_media_qa_v1",
        "expected_rooms": len(records),
        "video_quality": dict(Counter(row["video_quality"] for row in records)),
        "prescreen": dict(Counter(row["prescreen_suggestion"] for row in records)),
        "human_verdict": dict(Counter(row["human_verdict"] for row in records)),
        "camera_strategy": dict(Counter(row["camera_strategy"] for row in records)),
        "stale_media": stale,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps({"summary": summary, "rooms": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
