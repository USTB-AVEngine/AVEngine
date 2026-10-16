#!/usr/bin/env python3
"""Build controlled dog/speech AudioProgram contracts for the A native canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import write_json  # noqa: E402
from avengine.timeline.audio_program import (  # noqa: E402
    bind_audio_program_hash,
    validate_audio_program,
)
from avengine.registry.registry import bind_content_hash  # noqa: E402
from avengine.registry.sources import (  # noqa: E402
    validate_sound_asset_registry,
    validate_source_endpoint_registry,
)


DOG_SOUND_ID = "dog_freesound_125791_v1"
SPEECH_SOUND_ID = "speech_cremad_1001_ieo_neu_v1"
SOURCE1_ASSET_ID = "generated_border_collie_black_white_medium_standard_adult_research_v1"
SOURCE2_ASSET_ID = "rocketbox_human_male_adult_01_m5_1_candidate"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _controlled_asset(registry: Mapping[str, Any], sound_id: str) -> Mapping[str, Any]:
    matches = [item for item in registry["assets"] if item["sound_asset_id"] == sound_id]
    _require(len(matches) == 1, f"controlled sound {sound_id!r} must resolve uniquely")
    return matches[0]


def _runtime_asset(registry: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    records = registry.get("assets", registry.get("profiles"))
    _require(isinstance(records, list), "runtime asset registry has no asset list")
    matches = [item for item in records if item.get("asset_id") == asset_id]
    _require(len(matches) == 1, f"runtime asset {asset_id!r} must resolve uniquely")
    return matches[0]


def _validate_wave(path: Path, audio: Mapping[str, Any]) -> None:
    _require(path.is_file(), f"controlled media is missing: {path}")
    with wave.open(str(path), "rb") as stream:
        observed = {
            "sample_rate_hz": stream.getframerate(),
            "channel_count": stream.getnchannels(),
            "sample_count": stream.getnframes(),
            "sample_width": stream.getsampwidth(),
        }
    _require(observed["sample_width"] == 2, f"{path}: expected PCM16")
    for field in ["sample_rate_hz", "channel_count", "sample_count"]:
        _require(observed[field] == audio[field], f"{path}: {field} drift")
    _require(_sha256(path) == audio["sha256"], f"{path}: SHA-256 drift")


def _event(
    *,
    event_id: str,
    endpoint_id: str,
    sound_id: str,
    start_sample: int,
    end_sample: int,
    source_start: int,
    gain: float,
) -> dict[str, Any]:
    duration = end_sample - start_sample
    return {
        "event_id": event_id,
        "source_endpoint_id": endpoint_id,
        "sound_asset_id": sound_id,
        "start_tick": start_sample * 3,
        "end_tick_exclusive": end_sample * 3,
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "source_start_sample": source_start,
        "source_end_sample_exclusive": source_start + duration,
        "linear_gain": gain,
        "fade_samples": 80,
        "normalization_policy": "use_sound_asset_policy",
        "render_source_stem": True,
    }


def build_contracts(
    *,
    controlled_registry: Mapping[str, Any],
    controlled_registry_path: Path,
    runtime_registry: Mapping[str, Any],
    runtime_registry_path: Path,
) -> dict[str, Any]:
    dog = _controlled_asset(controlled_registry, DOG_SOUND_ID)
    speech = _controlled_asset(controlled_registry, SPEECH_SOUND_ID)
    source1 = _runtime_asset(runtime_registry, SOURCE1_ASSET_ID)
    source2 = _runtime_asset(runtime_registry, SOURCE2_ASSET_ID)
    media_root = controlled_registry_path.parent / "media"
    media = {
        DOG_SOUND_ID: media_root / f"{DOG_SOUND_ID}.wav",
        SPEECH_SOUND_ID: media_root / f"{SPEECH_SOUND_ID}.wav",
    }
    for record in [dog, speech]:
        _validate_wave(media[record["sound_asset_id"]], record["audio"])

    runtime_sha = _sha256(runtime_registry_path)
    endpoints = bind_content_hash(
        {
            "schema": "avengine_m6_source_endpoint_registry_v1",
            "registry_id": "lead_a_native_controlled_endpoints_v1",
            "revision": "v1",
            "source_endpoints": [
                {
                    "source_endpoint_id": "lead_a_source1_muzzle",
                    "revision": "v1",
                    "binding": {
                        "kind": "entity_anchor",
                        "entity_instance_id": "source1",
                        "entity_asset_id": SOURCE1_ASSET_ID,
                        "entity_asset_revision": source1["revision"],
                        "emitter_anchor_id": "muzzle",
                    },
                    "source_visibility_mode": "visible_entity",
                    "allowed_sound_class_ids": ["animal_vocalization"],
                    "directivity_profile_id": "point_emitter_v1",
                    "persistent_when_silent": True,
                    "admission_state": "research",
                    "evidence_sha256": runtime_sha,
                },
                {
                    "source_endpoint_id": "lead_a_source2_mouth",
                    "revision": "v1",
                    "binding": {
                        "kind": "entity_anchor",
                        "entity_instance_id": "source2",
                        "entity_asset_id": SOURCE2_ASSET_ID,
                        "entity_asset_revision": source2["revision"],
                        "emitter_anchor_id": "mouth",
                    },
                    "source_visibility_mode": "visible_entity",
                    "allowed_sound_class_ids": ["human_speech"],
                    "directivity_profile_id": "point_emitter_v1",
                    "persistent_when_silent": True,
                    "admission_state": "research",
                    "evidence_sha256": runtime_sha,
                },
            ],
        }
    )
    rights_evidence = _sha256(controlled_registry_path)
    sound_assets = []
    for record, sound_class, taxonomy in [
        (dog, "animal_vocalization", ["animal", "canine", "bark"]),
        (speech, "human_speech", ["human", "voice", "speech"]),
    ]:
        content = record["content"]
        sound_assets.append(
            {
                "sound_asset_id": record["sound_asset_id"],
                "revision": "v1",
                "semantic_sound_class": sound_class,
                "taxonomy_path": taxonomy,
                "instance_lineage_id": content.get("speaker_id") or record["sound_asset_id"],
                "dry_audio": {
                    "uri": record["audio"]["uri"],
                    "sha256": record["audio"]["sha256"],
                    "sample_rate_hz": record["audio"]["sample_rate_hz"],
                    "channel_count": record["audio"]["channel_count"],
                    "sample_count": record["audio"]["sample_count"],
                },
                "normalization_policy": {"mode": "preserve", "target_dbfs": None},
                "allowed_transforms": ["crop", "gain", "zero_pad"],
                "permitted_event_usage": ["intermittent_events"],
                "tags": sorted(set(content["content_tags"] + [content["species"]])),
                "provenance": {
                    "origin": "lead_b_controlled_sound_content_registry_v1",
                    "license": None,
                    "rights_status": "review_required",
                    "rights_evidence_sha256": rights_evidence,
                },
                "admissibility": "research",
            }
        )
    sounds = bind_content_hash(
        {
            "schema": "avengine_m6_sound_asset_registry_v1",
            "registry_id": "lead_a_native_controlled_sounds_v1",
            "revision": "v1",
            "sound_assets": sorted(sound_assets, key=lambda item: item["sound_asset_id"]),
        }
    )
    speech_samples = speech["audio"]["sample_count"]
    program = bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": "lead_a_native_full_occlusion_controlled_audio_v1",
            "revision": "v1",
            "mode": "intermittent_events",
            "timeline": {
                "time_base_hz": 48000,
                "ticks_per_frame": 3200,
                "video_fps": 15,
                "frame_count": 75,
                "sample_rate_hz": 16000,
                "ticks_per_sample": 3,
                "sample_count": 80000,
            },
            "candidate_source_endpoint_ids": [
                "lead_a_source1_muzzle",
                "lead_a_source2_mouth",
            ],
            "events": [
                _event(
                    event_id="source1_bark_000",
                    endpoint_id="lead_a_source1_muzzle",
                    sound_id=DOG_SOUND_ID,
                    start_sample=0,
                    end_sample=12000,
                    source_start=0,
                    gain=0.1,
                ),
                _event(
                    event_id="source2_speech_000",
                    endpoint_id="lead_a_source2_mouth",
                    sound_id=SPEECH_SOUND_ID,
                    start_sample=24000,
                    end_sample=24000 + speech_samples,
                    source_start=0,
                    gain=0.22,
                ),
                _event(
                    event_id="source1_bark_001",
                    endpoint_id="lead_a_source1_muzzle",
                    sound_id=DOG_SOUND_ID,
                    start_sample=60000,
                    end_sample=72000,
                    source_start=12000,
                    gain=0.1,
                ),
            ],
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )
    endpoint_errors = validate_source_endpoint_registry(endpoints)
    sound_errors = validate_sound_asset_registry(sounds)
    program_errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    _require(not endpoint_errors, "; ".join(endpoint_errors))
    _require(not sound_errors, "; ".join(sound_errors))
    _require(not program_errors, "; ".join(program_errors))
    return {
        "source_endpoint_registry": endpoints,
        "sound_asset_registry": sounds,
        "audio_program": program,
        "sound_audio_paths": {key: str(value.resolve()) for key, value in media.items()},
        "controlled_content": {
            "source1": {
                "sound_asset_id": DOG_SOUND_ID,
                "statement_id": None,
                "transcript": None,
                "language": "und",
            },
            "source2": {
                "sound_asset_id": SPEECH_SOUND_ID,
                "statement_id": speech["content"]["statement_id"],
                "transcript": speech["content"]["transcript"],
                "language": speech["content"]["language"],
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-registry", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    result = build_contracts(
        controlled_registry=_load(args.controlled_registry.resolve()),
        controlled_registry_path=args.controlled_registry.resolve(),
        runtime_registry=_load(args.runtime_registry.resolve()),
        runtime_registry_path=args.runtime_registry.resolve(),
    )
    for key in ["source_endpoint_registry", "sound_asset_registry", "audio_program"]:
        write_json(output / f"{key}.json", result.pop(key))
    write_json(output / "controlled_audio_binding.json", result)
    print(f"NATIVE_CONTROLLED_AUDIO_PROGRAM_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
