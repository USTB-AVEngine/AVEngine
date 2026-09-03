#!/usr/bin/env python3
"""Build an avengine_sound_event_pool_v1 catalog from event_manifest.json.

Each split wav is already one pulse. The catalog row therefore uses the
whole file as the source window (start 0, end = frame count).

sound_asset_id is the prepared relative path without '.wav', with '/'
replaced by '__'. Example: dog_bark/clip_e000.wav -> dog_bark__clip_e000.
This tool does not write the sound asset registry. The render chain will
refuse any id that is not registered; add each clip to
examples/registry/registries/sound_assets_v1.json (or the live registry)
before SOUND_SOURCE_MODE=event_pool.

A pulse event_class with purpose=continuous is the hysteresis fallback
that kept a whole long file as one event. Those rows are refused so a
9 s "bark" cannot enter the pool.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from avengine.assets.sound_events import event_policy_for_class  # noqa: E402
from avengine.assets.sound_pool import SCHEMA as POOL_SCHEMA  # noqa: E402

MANIFEST_SCHEMA = "avengine_sound_event_library_v1"


class PoolBuildError(ValueError):
    """The event manifest cannot become a pool catalog."""


def sound_asset_id_for_prepared(prepared: str) -> str:
    stem = prepared[:-4] if prepared.endswith(".wav") else prepared
    return stem.replace("/", "__")


def _wav_rate_and_frames(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnframes()


def build_pool_catalog(manifest_path: Path, output_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise PoolBuildError(
            f"{manifest_path} schema {payload.get('schema')!r} is not "
            f"{MANIFEST_SCHEMA}")
    output_root = Path(payload["output_root"])
    clips = []
    for index, row in enumerate(payload.get("clips") or []):
        if row.get("status") != "event":
            continue
        owner = f"clips[{index}]"
        event_class = str(row.get("event_class") or "")
        purpose = str(row.get("purpose") or "")
        prepared = str(row.get("prepared") or "")
        if not event_class or not prepared:
            raise PoolBuildError(f"{owner} needs event_class and prepared")
        if (event_policy_for_class(event_class) == "pulse"
                and purpose == "continuous"):
            raise PoolBuildError(
                f"{owner} pulse class {event_class!r} has purpose=continuous; "
                "refusing to put a hysteresis-fallback span into the pool")
        wav_path = output_root / prepared
        if not wav_path.is_file():
            raise PoolBuildError(f"{owner} wav missing: {wav_path}")
        rate, frames = _wav_rate_and_frames(wav_path)
        if frames <= 0 or rate <= 0:
            raise PoolBuildError(f"{owner} empty wav {wav_path}")
        clips.append({
            "sound_asset_id": sound_asset_id_for_prepared(prepared),
            "event_class": event_class,
            "sample_rate_hz": rate,
            "duration_samples": frames,
            "source_start_sample": 0,
            "source_end_sample_exclusive": frames,
            "prepared": prepared,
            "purpose": purpose,
        })
    if not clips:
        raise PoolBuildError(f"{manifest_path} has no event rows")
    catalog = {
        "schema": POOL_SCHEMA,
        "source_manifest": str(manifest_path),
        "clips": clips,
        "registration_note": (
            "sound_asset_id is not registered by this tool. Add each id to "
            "the sound asset registry before render."
        ),
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    catalog = build_pool_catalog(args.manifest.resolve(), args.output.resolve())
    print(json.dumps({"clips": len(catalog["clips"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
