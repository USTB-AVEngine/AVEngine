#!/usr/bin/env python3
"""Split a prepared sound library into one wav per sounding event.

Writes a new tree. Never overwrites the prepared library. Pulse classes
become several short clips; continuous classes become one trimmed clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from avengine.assets.sound_events import (  # noqa: E402
    SoundEventError,
    extract_sound_events,
    slice_event,
)

SCHEMA = "avengine_sound_event_library_v1"
TARGET_PEAK_DBFS = -3.0


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(frames)
    if width != 2:
        raise SoundEventError(f"unhandled sample width {width}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        usable = len(samples) // channels * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples, rate


def _write_wav_mono(path: Path, samples: np.ndarray, rate: int) -> str:
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 0:
        gain = (10 ** (TARGET_PEAK_DBFS / 20)) / peak
        samples = samples * gain
    ints = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class_from_relative(relative: str) -> str:
    return relative.split("/", 1)[0]


def split_library(library_root: Path, output_root: Path) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refuse to write into non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    counts = {"pulse_events": 0, "continuous_events": 0, "failed": 0, "sources": 0}
    for wav_path in sorted(library_root.rglob("*.wav")):
        relative = wav_path.relative_to(library_root).as_posix()
        event_class = _class_from_relative(relative)
        counts["sources"] += 1
        try:
            samples, rate = _read_wav_mono(wav_path)
            events = extract_sound_events(
                samples, rate, event_class=event_class
            )
        except (SoundEventError, wave.Error, OSError, ValueError) as error:
            counts["failed"] += 1
            records.append(
                {
                    "source": relative,
                    "status": "failed",
                    "reason_zh": f"切事件失败:{error}",
                }
            )
            continue

        stem = Path(relative).with_suffix("")
        for index, event in enumerate(events):
            name = f"{stem}_e{index:03d}.wav"
            target = output_root / name
            sha = _write_wav_mono(target, slice_event(samples, event), rate)
            counts[f"{event.purpose}_events"] = (
                counts.get(f"{event.purpose}_events", 0) + 1
            )
            records.append(
                {
                    "source": relative,
                    "event_index": index,
                    "event_count": len(events),
                    "prepared": name,
                    "status": "event",
                    "purpose": event.purpose,
                    "event_class": event_class,
                    "start_sample": event.start_sample,
                    "end_sample_exclusive": event.end_sample_exclusive,
                    "duration_s": round(event.duration_s(rate), 4),
                    "prepared_sha256": sha,
                }
            )

    manifest = {
        "schema": SCHEMA,
        "library_root": str(library_root),
        "output_root": str(output_root),
        "counts": counts,
        "clips": records,
    }
    manifest_path = output_root / "event_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    library_root = args.library_root.resolve()
    output_root = args.output_root.resolve()
    if not library_root.is_dir():
        raise SystemExit(f"library root missing: {library_root}")
    manifest = split_library(library_root, output_root)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
