#!/usr/bin/env python3
"""Build a per-house visual review sheet from room-tour videos.

This is a reviewer aid only: it never writes room verdicts. Each row contains
six evenly spaced views from one room's 360-degree clip plus the room metadata
reported by the Studio API. Original top-down images are copied beside the
sheet so every visual verdict remains traceable to its source material.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TASKS_ROOT = Path("/data/avengine_external/studio/tasks")
MEDIA_ROOT = Path("/data/avengine_external/studio/room_curation_media")
DEFAULT_API = "http://127.0.0.1:8765/api/room-curation"


def fetch_json(url: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=120) as response:
        return json.load(response)


def font(size: int):
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def extract_strip(video: Path, destination: Path, frames: int = 6) -> None:
    # Every tour is three seconds. fps=frames/3 samples the full circle at
    # uniform angles; tile preserves all frames in one auditable image row.
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video),
            "-vf", f"fps={frames}/3,scale=300:169,tile={frames}x1:padding=2:margin=2",
            "-frames:v", "1", str(destination),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default=DEFAULT_API)
    args = parser.parse_args()

    data = fetch_json(args.api_url)
    record = next((h for h in data["houses"] if h["house"] == args.house), None)
    if record is None:
        raise SystemExit(f"house not found: {args.house}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    title_font, body_font, small_font = font(26), font(20), font(16)
    label_width, strip_width, row_height = 380, 1804, 205
    width = label_width + strip_width
    rows: list[tuple[dict, Image.Image | None]] = []

    with tempfile.TemporaryDirectory(prefix="room-review-frames-") as tmp:
        tmp_dir = Path(tmp)
        for room in record["rooms"]:
            label = room["label"]
            video = MEDIA_ROOT / args.house / f"{label}.mp4"
            strip = None
            if video.is_file() and video.stat().st_size > 0:
                frame_path = tmp_dir / f"{label}.jpg"
                extract_strip(video, frame_path)
                strip = Image.open(frame_path).convert("RGB").copy()
            rows.append((room, strip))

    canvas = Image.new("RGB", (width, 58 + row_height * len(rows)), "#f3f5f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), args.house, fill="#152233", font=title_font)
    for index, (room, strip) in enumerate(rows):
        y = 58 + index * row_height
        draw.rectangle((0, y, width, y + row_height - 2), fill="#ffffff")
        label = room["label"]
        area = float(room.get("floor_area_m2") or 0)
        extent = room.get("extent_m") or []
        categories = ", ".join(room.get("top_categories") or [])
        verdict = room.get("verdict")
        if isinstance(verdict, dict):
            verdict = verdict.get("verdict")
        draw.text((14, y + 12), f"{label}  {room.get('room_type') or 'unknown'}", fill="#152233", font=title_font)
        draw.text((14, y + 52), f"area {area:.1f} m2  extent {extent}", fill="#33465c", font=body_font)
        draw.multiline_text((14, y + 86), f"objects: {categories}\nverdict: {verdict or 'pending'}", fill="#617085", font=small_font, spacing=4)
        if strip is None:
            draw.rectangle((label_width, y + 2, width - 2, y + row_height - 4), fill="#202832")
            draw.text((label_width + 24, y + 76), "VIDEO MISSING", fill="#ffffff", font=title_font)
        else:
            canvas.paste(strip, (label_width, y + 16))

    sheet = args.output_dir / f"{args.house}__rooms.jpg"
    canvas.save(sheet, quality=92)

    topdown_dir = TASKS_ROOT / record["task_id"] / "output" / "render"
    copied = []
    for index, rel in enumerate(record.get("topdowns") or []):
        source = topdown_dir / rel
        if not source.is_file():
            continue
        suffix = source.suffix.lower() or ".png"
        destination = args.output_dir / f"{args.house}__topdown_{index}{suffix}"
        destination.write_bytes(source.read_bytes())
        copied.append(str(destination))

    manifest = {
        "schema": "avengine_room_visual_review_sheet_v1",
        "house": args.house,
        "task_id": record["task_id"],
        "sheet": str(sheet),
        "topdowns": copied,
        "rooms": [room["label"] for room, _ in rows],
    }
    (args.output_dir / f"{args.house}__manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
