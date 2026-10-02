#!/usr/bin/env python3
"""Generate conservative, auditable room-curation suggestions.

This helper never writes authoritative verdict files. It combines room
metadata, footprint geometry, semantic categories, rendered-video quality,
and recorded camera placement into a suggestion for a human reviewer.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.rooms.runtime_config import (RUNTIME_PREFIX, MAGNUM_SITE, RLR_SDK_ROOT, MP3D_ROOT, TASKS_ROOT, MEDIA_ROOT, ROOM_PYTHON)

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

MEDIA_ROOT = Path(str(MEDIA_ROOT))
DEFAULT_QA_JSON = Path("/data/smy/projects/AVEngine/logs/room_media_qa.json")

_FURNITURE = frozenset({
    "bed", "pillow", "wardrobe", "nightstand", "couch", "sofa", "tv",
    "coffee table", "fireplace", "shelving", "fridge", "stove", "sink",
    "oven", "cabinet", "microwave", "dishwasher", "table", "desk", "chair",
    "toilet", "shower", "bathtub", "bookshelf", "book rack", "counter",
    "dresser", "rug", "mattress", "dining chair", "display cabinet",
})
_USE_MARKERS = frozenset({
    "bed", "pillow", "wardrobe", "nightstand", "mattress",
    "couch", "sofa", "tv", "coffee table", "fireplace",
    "fridge", "stove", "oven", "microwave", "dishwasher", "sink",
})
_TRANSIT_MARKERS = frozenset({
    "stairs", "stair", "step", "balustrade", "railing", "gutter",
})
_SKIP_ROOM_TYPES = ("走廊", "过道", "楼梯", "车库", "设备间")
_USE_ROOM_TYPES = ("卧室", "客厅", "厨房", "餐厅", "书房", "办公室")


def video_black_metrics(video: Path) -> dict | None:
    """Measure exact/near-black pixels in six uniformly sampled frames."""
    if not video.is_file() or video.stat().st_size <= 0:
        return None
    width, height, samples = 160, 90, 6
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vf", f"fps={samples}/3,scale={width}:{height},format=gray",
            "-frames:v", str(samples), "-f", "rawvideo", "pipe:1",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    pixels = result.stdout
    if result.returncode or not pixels:
        return {"read_error": True}
    total = len(pixels)
    return {
        "sample_frames": total // (width * height),
        "black_ratio": round(sum(value <= 12 for value in pixels) / total, 4),
        "dark_ratio": round(sum(value <= 28 for value in pixels) / total, 4),
    }


def camera_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"read_error": True}
    return {
        "placement_strategy": data.get("placement_strategy"),
        "centre_offset_xz_m": data.get("centre_offset_xz_m"),
    }


def suggest(room: dict, video: dict | None, camera: dict | None) -> tuple[str, str, dict]:
    categories = {str(value).lower() for value in room.get("top_categories") or []}
    extent = [float(value) for value in (room.get("extent_m") or [0.0, 0.0])]
    longer, shorter = max(extent), min(extent)
    aspect = shorter / longer if longer > 0 else 1.0
    area = float(room.get("floor_area_m2") or 0.0)
    room_type = str(room.get("room_type") or "未识别")
    furniture = categories & _FURNITURE
    use_markers = categories & _USE_MARKERS
    transit = categories & _TRANSIT_MARKERS
    cautions: list[str] = []
    blockers: list[str] = []
    signals: list[str] = []

    # Visual integrity outranks semantic labels. Large exact-black surfaces in
    # these HM3D clips normally mean unreconstructed geometry rather than shade.
    if video is None:
        return "unsure", "巡房视频尚缺失，不能完成视觉质量确认", {"confidence": "low", "review_priority": "high", "signals": ["video_missing"]}
    if video.get("read_error"):
        return "unsure", "巡房视频无法解码，需要人工检查", {"confidence": "low", "review_priority": "high", "signals": ["video_decode_error"]}
    black = float(video.get("black_ratio") or 0.0)
    if black >= 0.35:
        blockers.append(f"视频黑色缺失区域严重（黑像素约{black:.0%}）")
        signals.append("severe_black")
    if black >= 0.20:
        cautions.append(f"视频存在明显黑色缺失区域（黑像素约{black:.0%}）")
        signals.append("black_warning")

    if any(name in room_type for name in _SKIP_ROOM_TYPES):
        blockers.append(f"系统房型为{room_type}，属于交通或非居住空间")
        signals.append("non_residential_type")
    if transit and aspect < 0.55:
        blockers.append(
            f"含通行结构（{', '.join(sorted(transit))}）且空间细长"
            f"（短/长={aspect:.2f}），疑似楼梯或走廊"
        )
        signals.append("transit_shape")
    if not furniture and aspect < 0.42:
        blockers.append(f"无居家家具且空间细长（短/长={aspect:.2f}），疑似走廊/边缘区域")
        signals.append("empty_narrow_shape")
    if blockers:
        return "skip", "；".join(dict.fromkeys(blockers)), {"confidence": "medium", "review_priority": "medium", "signals": signals}

    offset = None if not camera else camera.get("centre_offset_xz_m")
    if isinstance(offset, (int, float)) and offset >= 2.0:
        cautions.append(f"相机离几何中心较远（{offset:.2f}m），需确认仍在本房间")
        signals.append("camera_far_from_centre")

    if categories & {"toilet", "shower", "bathtub"}:
        cautions.append("卫生间类空间是否纳入任务需人工确认")
        signals.append("bathroom_policy")
        return "unsure", "；".join(cautions), {"confidence": "medium", "review_priority": "high", "signals": signals}
    if cautions:
        if use_markers or any(name in room_type for name in _USE_ROOM_TYPES):
            cautions.append("其余房型/家具信号正常")
        return "unsure", "；".join(cautions), {"confidence": "medium", "review_priority": "high", "signals": signals}

    if use_markers:
        signals.extend(["usable_video", "strong_furniture"])
        return "use", f"扫描画面正常，含典型居家家具（{', '.join(sorted(use_markers))}）", {"confidence": "high", "review_priority": "low", "signals": signals}
    if any(name in room_type for name in _USE_ROOM_TYPES) and furniture:
        signals.extend(["usable_video", "room_type_and_furniture"])
        return "use", f"房型为{room_type}且有居家家具（{', '.join(sorted(furniture))}），画面质量正常", {"confidence": "high", "review_priority": "low", "signals": signals}
    if furniture:
        signals.extend(["usable_video", "furniture_only"])
        return "unsure", f"画面质量正常且有家具（{', '.join(sorted(furniture))}），但房间功能不够明确", {"confidence": "medium", "review_priority": "high", "signals": signals}
    signals.append("weak_semantic_evidence")
    return "unsure", f"无明确家具或房型证据（{area:.1f}m²，短/长={aspect:.2f}），需人工判断", {"confidence": "low", "review_priority": "high", "signals": signals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765/api/room-curation")
    parser.add_argument("--media-root", type=Path, default=MEDIA_ROOT)
    parser.add_argument("--house", help="only analyse one house (for testing)")
    parser.add_argument("--skip-video-analysis", action="store_true")
    parser.add_argument(
        "--qa-json", type=Path, default=DEFAULT_QA_JSON,
        help="reuse the read-only media QA result instead of starting ffmpeg per room",
    )
    parser.add_argument("--output", type=Path, default=MEDIA_ROOT / "prescreen.js")
    args = parser.parse_args()

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.api_url, timeout=120) as response:
        data = json.load(response)

    results: list[dict] = []
    counts = {"use": 0, "skip": 0, "unsure": 0}
    qa_index: dict[tuple[str, str], dict] = {}
    if not args.skip_video_analysis and args.qa_json.is_file():
        try:
            qa_data = json.loads(args.qa_json.read_text(encoding="utf-8"))
            for item in qa_data.get("rooms", []):
                qa_index[(str(item.get("house")), str(item.get("room_label")))] = item
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: cannot read QA cache {args.qa_json}: {exc}", file=sys.stderr)
    for house_record in data["houses"]:
        house = house_record["house"]
        if args.house and house != args.house:
            continue
        for room in house_record["rooms"]:
            label = room["label"]
            qa = qa_index.get((house, label))
            if qa is not None:
                video = None if not qa.get("video_exists") else {
                    "sample_frames": 6,
                    "black_ratio": qa.get("black_ratio"),
                    "dark_ratio": qa.get("dark_ratio"),
                }
            else:
                video = None if args.skip_video_analysis else video_black_metrics(
                    args.media_root / house / f"{label}.mp4"
                )
            camera = camera_metrics(args.media_root / house / f"{label}.camera.json")
            suggestion, reason, assessment = suggest(room, video, camera)
            results.append({
                "schema": "avengine_room_prescreen_v2",
                "house": house, "label": label,
                "suggestion": suggestion, "reason": reason,
                "assessment": assessment,
                "room_type": room.get("room_type"),
                "categories": room.get("top_categories") or [],
                "metrics": {"video": video, "camera": camera},
            })
            counts[suggestion] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "window.PRESCREEN = " + json.dumps(results, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {len(results)} rooms "
        f"(suggest-use {counts['use']}, suggest-skip {counts['skip']}, suggest-unsure {counts['unsure']}); "
        "authoritative verdicts unchanged"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
