"""Studio-authored turn-taking AudioPrograms bound through the engine.

The Studio lets the user pick a registered sound per source; this module
turns that choice into a validated, hash-bound M6 AudioProgram using the
engine's own authoring APIs. Event slices are chosen by an RMS energy scan
of the registered dry asset so a picked sound never plays its silence.
Research-only authoring: admission_state stays "research".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from avengine.timeline.audio_program import bind_audio_program_hash, validate_audio_program
from avengine.m6.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)

SAMPLE_RATE_HZ = 16000
SAMPLE_COUNT = 80000
TICKS_PER_SAMPLE = 3

_TIMELINE = {
    "time_base_hz": 48000,
    "ticks_per_frame": 3200,
    "video_fps": 15,
    "frame_count": 75,
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "ticks_per_sample": TICKS_PER_SAMPLE,
    "sample_count": SAMPLE_COUNT,
}


class StudioProgramError(ValueError):
    """Raised when a program request cannot be authored."""


def _sound_records(registry: dict) -> dict[str, dict]:
    return {
        record["sound_asset_id"]: record for record in registry["sound_assets"]
    }


def resolve_dry_audio_path(
    record: dict,
    repository_root: Path,
    external_paths: dict[str, str] | None = None,
) -> Path:
    uri = record["dry_audio"]["uri"]
    if uri.startswith("repo://"):
        path = repository_root / uri.removeprefix("repo://")
    elif uri.startswith("artifact://"):
        # artifact-scheme assets resolve through the deployment's declared
        # external staging (same mapping the render chain receives)
        mapped = (external_paths or {}).get(record["sound_asset_id"])
        if not mapped:
            raise StudioProgramError(
                f"no external path configured for artifact asset "
                f"{record['sound_asset_id']}"
            )
        path = Path(mapped)
    else:
        path = Path(uri)
    path = path.resolve()
    if not path.is_file():
        raise StudioProgramError(f"dry audio missing for {record['sound_asset_id']}: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["dry_audio"]["sha256"]:
        raise StudioProgramError(
            f"dry audio sha256 mismatch for {record['sound_asset_id']}"
        )
    return path


def _read_mono_wav(path: Path) -> np.ndarray:
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    if rate != SAMPLE_RATE_HZ:
        raise StudioProgramError(f"{path} must be {SAMPLE_RATE_HZ} Hz, got {rate}")
    if data.ndim != 1:
        raise StudioProgramError(f"{path} must be mono")
    if data.dtype != np.float32:
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    return np.asarray(data, dtype=np.float64)


def pick_energetic_slice(samples: np.ndarray, length: int) -> tuple[int, int]:
    """Highest-RMS window of the dry asset, so events never land on silence."""

    total = int(samples.shape[0])
    length = min(length, total)
    if total <= length:
        return 0, total
    energy = samples.astype(np.float64) ** 2
    window = np.convolve(energy, np.ones(length), mode="valid")
    start = int(np.argmax(window))
    return start, start + length


def build_turn_taking_program(
    *,
    program_id: str,
    candidate_source_endpoint_ids: list[str],
    sound_by_endpoint: dict[str, str],
    source_endpoint_registry_path: str | Path,
    sound_asset_registry_path: str | Path,
    repository_root: str | Path,
    external_sound_asset_paths: dict[str, str] | None = None,
    event_count: int = 6,
    event_samples: int = 8000,
    linear_gain: float = 0.2,
) -> dict:
    if len(candidate_source_endpoint_ids) < 1:
        raise StudioProgramError("at least one candidate source endpoint required")
    missing = sorted(set(candidate_source_endpoint_ids) - set(sound_by_endpoint))
    if missing:
        raise StudioProgramError(f"no sound selected for endpoints: {missing}")
    if not 800 <= event_samples <= 20000:
        raise StudioProgramError("event_samples must be within [800, 20000]")
    if not 2 <= event_count <= 10:
        raise StudioProgramError("event_count must be within [2, 10]")

    repository_root = Path(repository_root)
    endpoints = load_source_endpoint_registry(source_endpoint_registry_path)
    sounds = load_sound_asset_registry(sound_asset_registry_path)
    records = _sound_records(sounds)

    slices: dict[str, tuple[int, int]] = {}
    for endpoint_id, sound_id in sound_by_endpoint.items():
        record = records.get(sound_id)
        if record is None:
            raise StudioProgramError(f"unknown sound asset: {sound_id}")
        if "sequential_sources" not in record.get("permitted_event_usage", ()):
            raise StudioProgramError(
                f"sound {sound_id} does not permit sequential_sources usage"
            )
        samples = _read_mono_wav(
            resolve_dry_audio_path(
                record, repository_root, external_paths=external_sound_asset_paths
            )
        )
        slices[endpoint_id] = pick_energetic_slice(samples, event_samples)

    margin = 4000
    usable = SAMPLE_COUNT - 2 * margin
    stride = usable // event_count
    if stride <= event_samples:
        raise StudioProgramError("events would overlap; reduce count or length")

    events = []
    for index in range(event_count):
        endpoint_id = candidate_source_endpoint_ids[
            index % len(candidate_source_endpoint_ids)
        ]
        sound_id = sound_by_endpoint[endpoint_id]
        start_sample = margin + index * stride
        end_sample = start_sample + slices[endpoint_id][1] - slices[endpoint_id][0]
        events.append(
            {
                "event_id": f"studio_event_{index + 1}",
                "source_endpoint_id": endpoint_id,
                "sound_asset_id": sound_id,
                "start_tick": start_sample * TICKS_PER_SAMPLE,
                "end_tick_exclusive": end_sample * TICKS_PER_SAMPLE,
                "start_sample": start_sample,
                "end_sample_exclusive": end_sample,
                "source_start_sample": slices[endpoint_id][0],
                "source_end_sample_exclusive": slices[endpoint_id][1],
                "linear_gain": linear_gain,
                "fade_samples": 80,
                "normalization_policy": "use_sound_asset_policy",
                "render_source_stem": True,
            }
        )

    program = {
        "schema": "avengine_m6_audio_program_v1",
        "program_id": program_id,
        "revision": "v1",
        "mode": "sequential_sources",
        "timeline": dict(_TIMELINE),
        # the schema requires the candidate list in canonical (sorted) order;
        # event alternation above still follows the caller's order
        "candidate_source_endpoint_ids": sorted(
            candidate_source_endpoint_ids, key=lambda item: item.encode("utf-8")
        ),
        "events": events,
        "source_specific_stems": True,
        "admission_state": "research",
    }
    program = bind_audio_program_hash(program)
    errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    if errors:
        raise StudioProgramError("authored program failed validation: " + "; ".join(errors))
    return program


def persist_program(program: dict, programs_root: str | Path) -> Path:
    root = Path(programs_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{program['program_id']}.json"
    if path.exists():
        raise StudioProgramError(f"program already exists (fresh/no-clobber): {path}")
    path.write_text(
        json.dumps(program, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
