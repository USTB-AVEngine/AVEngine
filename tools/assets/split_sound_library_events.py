#!/usr/bin/env python3
"""Split a prepared sound library into one wav per sounding event.

Writes a new tree. Never overwrites the prepared library. Pulse classes
become several short clips; continuous classes become one trimmed clip.

Layout (same three-level shape as the 3D source assets):

    <root>/index.json
    <root>/<category>/<type>/<variant>/event.wav

<type> is the event class. <variant> is the first eight hex digits of the
cut clip's sha256. The asset id is sound_<class>_<sha8>_v1.

Source numbering, occurrence index, split family, gain, truncation and
purpose stay in event_manifest.json. They are not the identity of the
event: a family reclassification must not rename the files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
    split_family,
)

SCHEMA = "avengine_sound_event_library_v1"
INDEX_SCHEMA = "avengine_sound_event_library_index_v1"
LAYOUT = "<category>/<type>/<variant>"
TARGET_PEAK_DBFS = -3.0

# Closed table: every prepared-library class maps to one domain. A missing
# class is an error, not a guess. The 2026-09-03 ring/bell/beep/alarm token
# heuristic is what put doorbells in the wrong bucket.
#
# Category is what the sound is to a listener, not which device produced it.
#   alert      a cue that demands attention
#   ambience   ongoing background, not pointing at any event
#   appliance  a machine's own noise while it runs
#   animal     an animal vocalization
#   water      water in motion
#   speech     someone talking (kept as its own category because
#              downstream treats it specially: separate scheduler,
#              question types, 10 s clips). Do not fold speech_playback
#              into playback.
#   playback   a recording reproduced by a loudspeaker, not a
#              program-generated waveform
#
# microwave_beep is alert and microwave_hum is appliance because of this
# rule, not because they share a device. alarm_clock stays in alert
# (the ring is a cue); clock_tick is ambience (mechanical ticking is
# background, not appliance self-noise).
CLASS_CATEGORY: dict[str, str] = {
    "dog_bark": "animal",
    "cat_meow": "animal",
    "speech_playback": "speech",
    "alarm_beep": "alert",
    "alarm_bell": "alert",
    "alarm_clock": "alert",
    "busy_signal": "alert",
    "buzzer": "alert",
    "cellphone_vibration_alert": "alert",
    "chime": "alert",
    "ding_dong": "alert",
    "doorbell": "alert",
    "doorbell_chime": "alert",
    "fire_alarm": "alert",
    "microwave_beep": "alert",
    "phone_ring": "alert",
    "ringtone": "alert",
    "smoke_alarm": "alert",
    "telephone": "alert",
    "telephone_bell_ringing": "alert",
    "telephone_dialing_dtmf": "alert",
    "air_conditioning": "appliance",
    "blender": "appliance",
    "clock_tick": "ambience",
    "microwave_hum": "appliance",
    "printer": "appliance",
    "bathtub_filling_washing": "water",
    "drip": "water",
    "gurgling": "water",
    "sink_filling_washing": "water",
    "toilet_flush": "water",
    "water_tap_faucet": "water",
    "crackle": "ambience",
    "fire": "ambience",
    "any_audioset_class_playback": "playback",
    "music_playback": "playback",
}


def category_for_class(event_class: str) -> str:
    if event_class not in CLASS_CATEGORY:
        raise SoundEventError(
            f"event class {event_class!r} has no explicit category; "
            "CLASS_CATEGORY is a closed table, not a heuristic"
        )
    return CLASS_CATEGORY[event_class]


def sound_asset_id_for(event_class: str, sha8: str) -> str:
    return f"sound_{event_class}_{sha8}_v1"


def relative_event_wav(category: str, event_class: str, sha8: str) -> str:
    return f"{category}/{event_class}/{sha8}/event.wav"


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


def _encode_wav_mono(samples: np.ndarray, rate: int, *,
                     peak_normalize: bool) -> tuple[bytes, float]:
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    applied_gain_db = 0.0
    if peak_normalize and peak > 0:
        gain = (10 ** (TARGET_PEAK_DBFS / 20)) / peak
        samples = samples * gain
        applied_gain_db = float(20.0 * np.log10(gain))
    ints = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())
    return buf.getvalue(), applied_gain_db


def _class_from_relative(relative: str) -> str:
    return relative.split("/", 1)[0]


def _source_library_sha256(library_root: Path) -> str | None:
    manifest = library_root / "prepared_manifest.json"
    if not manifest.is_file():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def split_library(library_root: Path, output_root: Path, *,
                  peak_normalize: bool = False) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refuse to write into non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    index_assets: list[dict] = []
    seen_sha8: dict[tuple[str, str], str] = {}
    counts = {"pulse_events": 0, "continuous_events": 0, "failed": 0, "sources": 0}
    source_library_sha256 = _source_library_sha256(library_root)
    for wav_path in sorted(library_root.rglob("*.wav")):
        relative = wav_path.relative_to(library_root).as_posix()
        event_class = _class_from_relative(relative)
        category = category_for_class(event_class)
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

        source_sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        family = split_family(event_class)
        for index, event in enumerate(events):
            payload, applied_gain_db = _encode_wav_mono(
                slice_event(samples, event), rate,
                peak_normalize=peak_normalize)
            sha256 = hashlib.sha256(payload).hexdigest()
            sha8 = sha256[:8]
            collision_key = (event_class, sha8)
            if collision_key in seen_sha8:
                raise FileExistsError(
                    f"sha8 collision for class {event_class!r} sha8={sha8}: "
                    f"{seen_sha8[collision_key]} and {relative}"
                )
            name = relative_event_wav(category, event_class, sha8)
            target = output_root / name
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            seen_sha8[collision_key] = relative
            counts[f"{event.purpose}_events"] = (
                counts.get(f"{event.purpose}_events", 0) + 1
            )
            untruncated_end = event.untruncated_end_sample_exclusive
            asset_id = sound_asset_id_for(event_class, sha8)
            with wave.open(io.BytesIO(payload), "rb") as handle:
                sample_count = handle.getnframes()
            records.append(
                {
                    "source": relative,
                    "source_sha256": source_sha256,
                    "occurrence_index": index,
                    "event_index": index,
                    "event_count": len(events),
                    "prepared": name,
                    "status": "event",
                    "purpose": event.purpose,
                    "split_family": family,
                    "event_class": event_class,
                    "category": category,
                    "sound_asset_id": asset_id,
                    "variant": sha8,
                    "start_sample": event.start_sample,
                    "end_sample_exclusive": event.end_sample_exclusive,
                    "duration_s": round(event.duration_s(rate), 4),
                    "sample_rate_hz": rate,
                    "channel_count": 1,
                    "sample_count": sample_count,
                    "truncated": bool(event.truncated),
                    "untruncated_end_sample_exclusive": untruncated_end,
                    "untruncated_duration_s": (
                        None if untruncated_end is None
                        else round(
                            (untruncated_end - event.start_sample) / rate, 4)),
                    "applied_gain_db": round(applied_gain_db, 4),
                    "prepared_sha256": sha256,
                }
            )
            index_assets.append(
                {
                    "asset_id": asset_id,
                    "path": f"{category}/{event_class}/{sha8}",
                    "category": category,
                    "event_class": event_class,
                    "variant": sha8,
                    "wav": name,
                    "sha256": sha256,
                    "sample_rate_hz": rate,
                    "channel_count": 1,
                    "sample_count": sample_count,
                }
            )

    index_assets.sort(key=lambda item: item["asset_id"])
    index = {
        "schema": INDEX_SCHEMA,
        "layout": LAYOUT,
        "layout_note": (
            "category is the domain an engine asks for first; type is the "
            "event class; variant is the content sha256 prefix of this cut "
            "clip. Source path, occurrence index, split family, applied "
            "gain, truncation and purpose stay in event_manifest.json "
            "because they are not the identity of the event."
        ),
        "library_root": str(library_root),
        "source_library_sha256": source_library_sha256,
        "assets": index_assets,
    }
    manifest = {
        "schema": SCHEMA,
        "layout": LAYOUT,
        "library_root": str(library_root),
        "output_root": str(output_root),
        "source_library_sha256": source_library_sha256,
        "splitter": "tools/assets/split_sound_library_events.py",
        "counts": counts,
        "clips": records,
    }
    manifest_path = output_root / "event_manifest.json"
    index_path = output_root / "index.json"
    if manifest_path.exists() or index_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--peak-normalize", action="store_true",
        help="Peak-normalize each cut event to -3 dBFS. Off by default "
             "because prepared clips are already peak-normalized.")
    args = parser.parse_args(argv)
    library_root = args.library_root.resolve()
    output_root = args.output_root.resolve()
    if not library_root.is_dir():
        raise SystemExit(f"library root missing: {library_root}")
    prepared = library_root / "prepared_manifest.json"
    if output_root == library_root or (
            prepared.is_file() and output_root == prepared.parent):
        raise SystemExit(
            f"refuse to write into the prepared library: {library_root}")
    manifest = split_library(
        library_root, output_root, peak_normalize=args.peak_normalize)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
