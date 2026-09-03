#!/usr/bin/env python3
"""Raise a calibration pack to a listenable level without touching the render.

The rendered episodes are deliberately never normalised: the binaural contract
forbids normalisation and limiting because both destroy the interaural level
difference and the distance cues the questions are about.  The consequence is
that a pack straight out of the renderer peaks around -32 dBTP and a 0.3 s bark
is inaudible -- on 2026-09-03 owner could not answer the first item because he
heard nothing.

This tool writes a separate listening copy.  One scalar for the whole pack, so
every level difference between items and every ILD inside an item survives
sample for sample.  The gain is a rule, not a constant, because a future pack
may contain a louder item and a hard-coded value would clip it:

    listening_gain_db = TARGET_TRUE_PEAK_DBTP - max(true peak of every item)

Never edit the source pack.  Never apply per-item normalisation here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA = "avengine_qa_v3_listening_copy_v1"
TARGET_TRUE_PEAK_DBTP = -3.0
# A true peak above this would clip on playback and clipping silently destroys
# the ILD, which is the cue under test.  The rule cannot produce it; the check
# is a positive control on the output, not a clamp on the input.
CEILING_TRUE_PEAK_DBTP = -1.0
GAIN_RULE = (
    "listening_gain_db = -3.0 - max(input_tp of every clip in the pack); one "
    "scalar for the whole pack so inter-item level differences and per-item "
    "ILD are unchanged")


class ListeningCopyError(RuntimeError):
    """The pack cannot be levelled without guessing something."""


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def listening_gain_db(true_peaks_dbtp, *, target=TARGET_TRUE_PEAK_DBTP):
    """The one scalar this pack needs.  Loudest item lands on the target."""

    peaks = [float(value) for value in true_peaks_dbtp]
    if not peaks:
        raise ListeningCopyError("no measured true peaks; nothing to level")
    gain = target - max(peaks)
    if max(peaks) + gain > CEILING_TRUE_PEAK_DBTP + 1e-9:
        raise ListeningCopyError(
            f"gain {gain:+.2f} dB would leave a true peak above "
            f"{CEILING_TRUE_PEAK_DBTP} dBTP")
    return round(gain, 2)


def _ffmpeg(*args):
    try:
        done = subprocess.run(("ffmpeg", "-hide_banner", *args),
                              capture_output=True, text=True)
    except FileNotFoundError as error:
        raise ListeningCopyError(
            "ffmpeg is required to measure and apply the listening gain") from error
    return done


def measure_true_peak_dbtp(path):
    """Integrated loudness and true peak from ffmpeg loudnorm, never estimated."""

    done = _ffmpeg("-i", str(path), "-af", "loudnorm=print_format=json",
                   "-f", "null", "-")
    match = re.search(r"\{[^{}]*input_tp[^{}]*\}", done.stderr, re.S)
    if match is None:
        raise ListeningCopyError(
            f"{path}: loudnorm printed no measurement; refusing to guess a level")
    payload = json.loads(match.group(0))
    return float(payload["input_tp"]), float(payload["input_i"])


def _apply_gain(source, target, gain_db):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    done = _ffmpeg("-v", "error", "-i", str(source), "-c:v", "copy",
                   "-af", f"volume={gain_db}dB", "-c:a", "aac", "-b:a", "128k",
                   str(target))
    if done.returncode != 0 or not target.is_file():
        raise ListeningCopyError(f"{source}: ffmpeg failed: {done.stderr[-400:]}")


def build_listening_copy(pack_root, output_root):
    pack_root = Path(pack_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == pack_root:
        raise ListeningCopyError("refusing to write the listening copy over the pack")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refuse to write into non-empty {output_root}")
    study_path = pack_root / "public" / "study_items.json"
    media_root = pack_root / "public" / "media"
    if not study_path.is_file() or not media_root.is_dir():
        raise ListeningCopyError(f"{pack_root} is not a built calibration pack")

    clips = sorted(media_root.glob("*.mp4"))
    if not clips:
        raise ListeningCopyError(f"{media_root} holds no media")
    measured = {}
    for clip in clips:
        peak, loudness = measure_true_peak_dbtp(clip)
        measured[clip.name] = {
            "source_sha256": _sha256(clip),
            "input_true_peak_dbtp": peak,
            "input_loudness_lufs": loudness,
        }
    gain = listening_gain_db(row["input_true_peak_dbtp"]
                             for row in measured.values())

    out_public = output_root / "public"
    (out_public / "media").mkdir(parents=True)
    for clip in clips:
        _apply_gain(clip, out_public / "media" / clip.name, gain)
    for name, row in measured.items():
        peak, loudness = measure_true_peak_dbtp(out_public / "media" / name)
        row["output_true_peak_dbtp"] = peak
        row["output_loudness_lufs"] = loudness
        row["output_sha256"] = _sha256(out_public / "media" / name)
        if peak > CEILING_TRUE_PEAK_DBTP + 1e-9:
            raise ListeningCopyError(
                f"{name}: output true peak {peak} dBTP is above "
                f"{CEILING_TRUE_PEAK_DBTP}; clipping would destroy the ILD")

    level = {
        "listening_gain_db": gain,
        "listening_gain_rule": GAIN_RULE,
        "listening_target_true_peak_dbtp": TARGET_TRUE_PEAK_DBTP,
        "listening_source_pack": str(pack_root),
        "listening_note_zh": (
            f"整包统一抬升 {gain:+.2f} dB，只改听音副本，不动渲染产物；"
            "条目之间的电平差和每条的左右耳差逐样本不变。"),
    }
    for name in ("study_items.json", "practice_items.json"):
        source = pack_root / "public" / name
        if not source.is_file():
            continue
        document = _read(source)
        document.update(level)
        _write(out_public / name, document)
    shutil.copy2(pack_root / "public" / "index.html", out_public / "index.html")

    key_path = pack_root / "private" / "answer_key.json"
    if key_path.is_file():
        key = _read(key_path)
        key.update(level)
        (output_root / "private").mkdir(parents=True, exist_ok=True)
        _write(output_root / "private" / "answer_key.json", key)

    manifest = {
        "schema": SCHEMA,
        "source_pack": str(pack_root),
        "output_root": str(output_root),
        **level,
        "clips": measured,
    }
    _write(output_root / "listening_copy_manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_listening_copy(args.pack_root, args.output_root)
    print(json.dumps({
        "output": manifest["output_root"],
        "listening_gain_db": manifest["listening_gain_db"],
        "clips": len(manifest["clips"]),
        "max_output_true_peak_dbtp": max(
            row["output_true_peak_dbtp"] for row in manifest["clips"].values()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
