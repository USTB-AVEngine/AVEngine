#!/usr/bin/env python3
"""Retire raw rgb.npy arrays for a finished QA v2 batch (owner policy
20260823: once a batch's questions are generated and every point has a
retained mp4, the npy is deleted and the mp4 is the kept visual).

Per point, fail-closed before deleting anything:
  1. a retained mp4 must exist: the review-page AV clip, or a silent clip
     muxed here from the npy for points without audio;
  2. ffprobe must report all 75 frames in that mp4;
  3. frame_records.json and research_receipt.json stay untouched.
Only then arrays/rgb.npy is removed. A deletion manifest (point, bytes,
mp4 path, mp4 sha256) is written next to the captures root.

Audio remixing stays possible without npy (stream-copy video + new audio).
What is permanently given up: pixel-exact truth (e.g. a future P1 pass) for
the retired batch - accepted by owner decision 20260823.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hashlib import sha256

REPOSITORY = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures-root", required=True, type=Path)
    p.add_argument("--clips-dir", required=True, type=Path,
                   help="review-page clips directory (AV mp4s)")
    p.add_argument("--questions", required=True, type=Path,
                   help="questions.json proving the batch's questions exist")
    p.add_argument("--silent-clips-dir", type=Path,
                   help="output for silent mp4s of points without an AV clip "
                        "(default <clips-dir>)")
    p.add_argument("--execute", action="store_true",
                   help="actually delete; default is a dry run")
    return p.parse_args()


def frame_count(mp4: Path) -> int:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True,
    )
    try:
        return int(proc.stdout.strip().split(",")[0])
    except ValueError:
        return -1


def mux_silent(capture_dir: Path, out_mp4: Path) -> bool:
    """Silent 15fps clip straight from the npy (BGR-corrected)."""
    script = (
        "import numpy as np, subprocess, sys\n"
        f"r = np.load(r'{capture_dir}/arrays/rgb.npy', mmap_mode='r')\n"
        "p = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo',"
        "'-pix_fmt','rgb24','-s','1280x720','-r','15','-i','-','-c:v','libx264',"
        f"'-pix_fmt','yuv420p','-crf','18',r'{out_mp4}'], stdin=subprocess.PIPE)\n"
        "for i in range(r.shape[0]):\n"
        "    p.stdin.write(np.asarray(r[i])[:, :, :3][:, :, ::-1].tobytes())\n"
        "p.stdin.close()\n"
        "sys.exit(p.wait())\n"
    )
    return subprocess.run([sys.executable, "-c", script]).returncode == 0


def file_sha256(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not questions.get("questions"):
        print(json.dumps({"error": "questions file has no questions; refusing"}))
        return 2
    silent_dir = args.silent_clips_dir or args.clips_dir
    silent_dir.mkdir(parents=True, exist_ok=True)

    rows, freed = [], 0
    for capture_dir in sorted(args.captures_root.iterdir()):
        npy = capture_dir / "arrays/rgb.npy"
        if not npy.is_file():
            continue
        pid = capture_dir.name
        if not (capture_dir / "research_receipt.json").is_file():
            rows.append({"point": pid, "action": "skip_no_receipt"})
            continue
        mp4 = args.clips_dir / f"{pid}.mp4"
        if not mp4.is_file():
            mp4 = silent_dir / f"{pid}.silent.mp4"
            if not mp4.is_file() and not mux_silent(capture_dir, mp4):
                rows.append({"point": pid, "action": "skip_silent_mux_failed"})
                continue
        frames = frame_count(mp4)
        if frames != 75:
            rows.append({"point": pid, "action": "skip_bad_mp4", "frames": frames,
                         "mp4": str(mp4)})
            continue
        size = npy.stat().st_size
        entry = {"point": pid, "action": "retire", "npy_bytes": size,
                 "retained_mp4": str(mp4), "mp4_sha256": file_sha256(mp4)}
        if args.execute:
            npy.unlink()
            arrays_dir = capture_dir / "arrays"
            if not any(arrays_dir.iterdir()):
                arrays_dir.rmdir()
            freed += size
        rows.append(entry)

    manifest = {
        "schema": "avengine_qa_v2_npy_retirement_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": ("owner decision 20260823: after question generation, retain "
                   "mp4 only; pixel-exact reuse for retired batches is waived"),
        "captures_root": str(args.captures_root),
        "executed": bool(args.execute),
        "freed_bytes": freed,
        "rows": rows,
    }
    out = args.captures_root / (
        "npy_retirement_manifest.json" if args.execute
        else "npy_retirement_dryrun.json"
    )
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({
        "executed": bool(args.execute),
        "retire": sum(1 for r in rows if r["action"] == "retire"),
        "skipped": sum(1 for r in rows if r["action"] != "retire"),
        "freed_gib": round(freed / 2**30, 2),
        "manifest": str(out),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
