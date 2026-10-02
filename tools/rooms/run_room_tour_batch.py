#!/usr/bin/env python3
"""Batch driver for render_room_tour.py: render every curated room a tour clip.

Pulls the room list from the studio server's /api/room-curation endpoint, finds
each house's rooms.json through the task_id the API reports (the same source
the server itself uses), resolves the HM3D glb path by convention, and calls
render_room_tour.py once per room, strictly serially - the GPU is shared and
the handoff doc says one room at a time.

Idempotent: rooms whose clip already exists (and is non-empty) are skipped, so
re-running after new houses arrive continues instead of redoing everything.

Exit codes from the renderer are honoured: 3 means the room centre is not
navigable, which is recorded in the log and skipped, not fatal.
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
import time
import urllib.request
import urllib.error
from pathlib import Path

HM3D_ROOT = Path("/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d")
TASKS_ROOT = Path(str(TASKS_ROOT))
DEFAULT_MEDIA_ROOT = Path(str(MEDIA_ROOT))
DEFAULT_PYTHON = Path(
    ROOM_PYTHON
)
# studio_config_48g.json "hm3d_episode" 段里的现成环境值。
DEFAULT_RUNTIME_PREFIX = (
    RUNTIME_PREFIX
)
DEFAULT_MAGNUM_SITE = (
    MAGNUM_SITE
)
DEFAULT_RLR_SDK_ROOT = RLR_SDK_ROOT

RENDER_SCRIPT = Path(__file__).resolve().parent / "render_room_tour.py"


def fetch_room_curation(api_url: str, attempts: int = 6) -> dict:
    """Read curation inventory, tolerating transient Studio 502 responses."""
    last_error: Exception | None = None
    # Studio is a loopback service. Do not let global HTTP_PROXY route this
    # request through the user's outbound proxy (which returns intermittent 502).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(1, attempts + 1):
        try:
            with opener.open(api_url, timeout=120) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(2 ** (attempt - 1), 15))
    raise RuntimeError(
        f"room-curation API unavailable after {attempts} attempts: {last_error}"
    ) from last_error


def resolve_rooms_json(house: str, task_id: str) -> Path | None:
    """rooms.json 与 server 同源：<tasks_root>/<task_id>/output/render/rooms/<house>/。"""
    output_dir = TASKS_ROOT / task_id / "output" / "render"
    for candidate in (
        output_dir / "rooms" / house / "rooms.json",
        output_dir / house / "rooms.json",  # hm3d_room_prepare 模板的布局
    ):
        if candidate.is_file():
            return candidate
    return None


def resolve_glb(house: str) -> Path:
    parts = house.split("_")
    if len(parts) < 4 or parts[0] != "hm3d":
        raise ValueError(f"cannot parse house id {house!r}")
    split, index, scene_id = parts[1], parts[2], parts[3]
    return HM3D_ROOT / split / f"{index}-{scene_id}" / f"{scene_id}.glb"


def room_label(room: dict) -> str:
    return f"R{room['region_id']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765/api/room-curation")
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--limit", type=int, default=0, help="max rooms to render (0=all)")
    parser.add_argument("--house", help="render only this house")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, render nothing")
    parser.add_argument("--log", type=Path, default=Path("logs/curation_render.log"))
    args = parser.parse_args()

    data = fetch_room_curation(args.api_url)
    planned: list[tuple[str, str, dict, Path]] = []
    for house_record in data["houses"]:
        house = house_record["house"]
        if args.house and house != args.house:
            continue
        rooms_json = resolve_rooms_json(house, house_record["task_id"])
        if rooms_json is None:
            print(f"[{house}] skip: rooms.json not found "
                  f"(task {house_record['task_id']})")
            continue
        inventory = json.loads(rooms_json.read_text(encoding="utf-8"))
        by_region = {r["region_id"]: r for r in inventory.get("rooms", [])}
        glb = resolve_glb(house)
        for room in house_record["rooms"]:
            label = room["label"]
            region = by_region.get(int(label[1:]))
            if region is None:
                print(f"[{house}] {label}: no region {label[1:]} in rooms.json, skip")
                continue
            clip = args.media_root / house / f"{label}.mp4"
            if clip.is_file() and clip.stat().st_size > 0:
                continue  # 已有产物，增量续跑
            planned.append((house, label, region, glb))

    print(f"plan: {len(planned)} rooms to render "
          f"(of {sum(len(h['rooms']) for h in data['houses'])} registered)")
    if args.dry_run:
        for house, label, region, glb in planned:
            centre = region["bbox_xz_m"]
            cx = (centre[0][0] + centre[1][0]) / 2
            cz = (centre[0][1] + centre[1][1]) / 2
            print(f"  {house} {label} area={region['floor_area_m2']:.1f}m2 "
                  f"centre=({cx:.2f},{cz:.2f}) glb={glb.name}")
        return 0

    args.log.parent.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    with args.log.open("a", encoding="utf-8") as log:
        log.write(f"\n=== batch start {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                  f"({len(planned)} rooms) ===\n")
        for position, (house, label, region, glb) in enumerate(planned, 1):
            if args.limit and position > args.limit:
                break
            centre = region["bbox_xz_m"]
            cx = (centre[0][0] + centre[1][0]) / 2
            cz = (centre[0][1] + centre[1][1]) / 2
            centre_arg = f"{cx:.3f},{cz:.3f}"
            bbox_arg = (
                f"{centre[0][0]:.3f},{centre[0][1]:.3f},"
                f"{centre[1][0]:.3f},{centre[1][1]:.3f}"
            )
            started = time.time()
            line = (f"[{position}/{len(planned)}] {house} {label} "
                    f"({region['floor_area_m2']:.1f}m2)")
            print(line, flush=True)
            log.write(f"{line} glb={glb}\n")
            result = subprocess.run(
                [
                    str(args.python), str(RENDER_SCRIPT),
                    "--glb", str(glb),
                    f"--center-xz={centre_arg}",  # 等号传参：负值坐标不会被 argparse 当选项
                    f"--bbox-xz={bbox_arg}",
                    "--floor-y", str(region["floor_y_m"]),
                    "--label", label,
                    "--house", house,
                    "--output-dir", str(args.media_root),
                    "--runtime-prefix", DEFAULT_RUNTIME_PREFIX,
                    "--magnum-site", DEFAULT_MAGNUM_SITE,
                    "--rlr-sdk-root", DEFAULT_RLR_SDK_ROOT,
                ],
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            elapsed = time.time() - started
            if result.returncode == 0:
                log.write(f"ok in {elapsed:.0f}s\n")
            elif result.returncode == 3:
                # 中心/snap/bbox 网格全找不到可走点才 skip（rc=3），记录并跳过。
                log.write(f"skip (no navigable point) after {elapsed:.0f}s\n")
                print(f"  skip: no navigable point in room bbox", flush=True)
            else:
                failed.append(f"{house} {label}")
                log.write(f"FAILED rc={result.returncode} after {elapsed:.0f}s\n")
                print(f"  FAILED rc={result.returncode}", flush=True)
        log.write(f"=== batch end: {len(planned) - len(failed)} ok, "
                  f"{len(failed)} failed ===\n")
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
