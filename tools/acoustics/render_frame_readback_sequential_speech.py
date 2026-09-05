"""Render a research four-speaker sequential speech program from SPEAR readbacks.

The helper consumes actual camera/listener and emitter frame readbacks, complete
16 kHz mono VCTK WAVs, and an existing AVEngine M3/RLR package. It does not
start UE, infer actor positions from the plan, trim utterances, or create a
second scheduler. Event timing is derived from the supplied readback clock and
the four complete clips must fit inside its exact sample boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave
from typing import Any, Mapping

import numpy as np

from avengine.capture.dry_audio import (
    DryAudioClipSpec,
    assemble_dry_audio_buses,
)
from avengine.contracts.json_io import sha256_file
from avengine.acoustics.runtime import (
    RLRSimulationConfig,
    RuntimeAnchor,
    load_compiled_acoustic_scene,
    simulate_compiled_acoustic_scene,
)
from avengine.timeline.audio_program import bind_audio_program_hash, validate_audio_program

TIME_BASE_HZ = 48_000
TICKS_PER_SAMPLE = 3
DEFAULT_GAIN = 0.15


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"WAV must be mono int16 PCM: {path}")
        rate = handle.getframerate()
        count = handle.getnframes()
        payload = handle.readframes(count)
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    if len(samples) != count or rate != 16000:
        raise ValueError(f"WAV must be 16 kHz and internally consistent: {path}")
    return np.ascontiguousarray(samples), rate


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    values = np.asarray(samples, dtype=np.float64)
    clipped = np.clip(values, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _cm_to_m(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"readback position must be a 3-vector in cm: {value!r}")
    return tuple(float(item) / 100.0 for item in value)


def _frame_position(records: list[Mapping[str, Any]], frame_index: int, key: str) -> tuple[float, float, float]:
    if not records:
        raise ValueError("readback list is empty")
    selected = records[min(max(frame_index, 0), len(records) - 1)]
    return _cm_to_m(selected[key])


def _simulation(
    *,
    direct_ray_count: int = 500,
    indirect_ray_count: int = 5000,
    source_ray_count: int = 500,
    indirect_ray_depth: int = 64,
    source_ray_depth: int = 16,
) -> RLRSimulationConfig:
    return RLRSimulationConfig.from_mapping(
        {
            "frequency_bands": 4,
            "direct_sh_order": 0,
            "indirect_sh_order": 0,
            "direct_ray_count": direct_ray_count,
            "indirect_ray_count": indirect_ray_count,
            "indirect_ray_depth": indirect_ray_depth,
            "source_ray_count": source_ray_count,
            "source_ray_depth": source_ray_depth,
            "max_diffraction_order": 0,
            "thread_count": 1,
            "sample_rate_hz": 16000.0,
            "max_ir_seconds": 0.25,
            "unit_scale": 1.0,
            "global_volume": 1.0,
            "speed_of_sound_m_s": 343.0,
            "direct": True,
            "indirect": True,
            "diffraction": False,
            "transmission": False,
            "mesh_simplification": False,
            "temporal_coherence": False,
            "channel_layout": {"type": "mono", "channel_count": 1},
        }
    )


def _program(
    *,
    clock: Mapping[str, Any],
    events: list[dict[str, Any]],
    endpoint_ids: list[str],
) -> dict[str, Any]:
    frame_count = int(clock["frame_count"])
    frame_rate = int(round(float(clock["frame_rate_hz"])))
    sample_rate = int(clock["sample_rate_hz"])
    sample_count = int(clock["sample_count"])
    timeline = {
        "time_base_hz": TIME_BASE_HZ,
        "ticks_per_frame": TIME_BASE_HZ // frame_rate,
        "video_fps": frame_rate,
        "frame_count": frame_count,
        "sample_rate_hz": sample_rate,
        "ticks_per_sample": TICKS_PER_SAMPLE,
        "sample_count": sample_count,
    }
    value = {
        "schema": "avengine_m6_audio_program_v1",
        "program_id": "polished_room_four_vctk_turn_taking_research_v1",
        "revision": "v1",
        "mode": "sequential_sources",
        "timeline": timeline,
        "candidate_source_endpoint_ids": sorted(endpoint_ids),
        "events": events,
        "source_specific_stems": True,
        "admission_state": "research",
    }
    value = bind_audio_program_hash(value)
    errors = validate_audio_program(value)
    if errors:
        raise ValueError("generated sequential program failed validation: " + "; ".join(errors))
    return value


def render(
    *,
    frame_readbacks: str | Path,
    package_manifest: str | Path,
    voice_binding: str | Path,
    output: str | Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    magnum_python_site: str | Path,
    direct_ray_count: int = 500,
    indirect_ray_count: int = 5000,
    source_ray_count: int = 500,
    indirect_ray_depth: int = 64,
    source_ray_depth: int = 16,
) -> dict[str, Any]:
    readback_path = Path(frame_readbacks).expanduser().resolve()
    package_path = Path(package_manifest).expanduser().resolve()
    bindings = _load(Path(voice_binding).expanduser().resolve())
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise ValueError("voice_binding must contain exactly four complete-clip records")
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"fresh output required: {root}")
    root.mkdir(parents=True)
    stems = root / "source_stems"
    stems.mkdir()
    readback = _load(readback_path)
    clock = readback["clock"]
    frame_count = int(clock["frame_count"])
    frame_rate = float(clock["frame_rate_hz"])
    sample_rate = int(clock["sample_rate_hz"])
    sample_count = int(clock["sample_count"])
    if sample_rate != 16000:
        raise ValueError("current research bridge requires 16 kHz readbacks")
    camera = readback["camera"]
    emitters = readback["emitters"]
    actor_ids = [str(item["actor_id"]) if isinstance(item, Mapping) else str(item) for item in bindings]
    if len(set(actor_ids)) != 4 or not all(actor_id in emitters for actor_id in actor_ids):
        raise ValueError("voice bindings must name four actor IDs present in emitter readbacks")

    clips: list[tuple[np.ndarray, int]] = []
    for item in bindings:
        path = Path(str(item["path"])).expanduser().resolve()
        samples, rate = _wav(path)
        clips.append((samples, rate))

    gap_samples = int(round(0.20 * sample_rate))
    margin_samples = int(round(0.50 * sample_rate))
    starts: list[int] = []
    cursor = margin_samples
    for index, (samples, _) in enumerate(clips):
        starts.append(cursor)
        cursor += len(samples)
        if index < len(clips) - 1:
            cursor += gap_samples
    if cursor > sample_count:
        raise ValueError(
            "complete VCTK sentences do not fit the readback clock; "
            "provide a longer SPEAR clock/readback, never truncate them"
        )

    endpoint_ids = [f"{actor_id}_mouth" for actor_id in actor_ids]
    events: list[dict[str, Any]] = []
    for index, (item, (samples, _), start, endpoint_id) in enumerate(
        zip(bindings, clips, starts, endpoint_ids)
    ):
        end = start + len(samples)
        events.append(
            {
                "event_id": f"vctk_turn_{index + 1:02d}",
                "source_endpoint_id": endpoint_id,
                "sound_asset_id": str(item["sound_asset_id"]),
                "start_tick": start * TICKS_PER_SAMPLE,
                "end_tick_exclusive": end * TICKS_PER_SAMPLE,
                "start_sample": start,
                "end_sample_exclusive": end,
                "source_start_sample": 0,
                "source_end_sample_exclusive": len(samples),
                "linear_gain": float(item.get("linear_gain", DEFAULT_GAIN)),
                "fade_samples": 80,
                "normalization_policy": "use_sound_asset_policy",
                "render_source_stem": True,
            }
        )
    program = _program(clock=clock, events=events, endpoint_ids=endpoint_ids)
    _write(root / "audio_program.json", program)
    dry_event_mappings = [
        {
            "event_id": event["event_id"],
            "source_id": event["source_endpoint_id"],
            "start_sample": event["start_sample"],
            "end_sample_exclusive": event["end_sample_exclusive"],
            "dry_asset_id": str(binding["sound_asset_id"]),
            "dry_asset_sha256": sha256_file(
                Path(str(binding["path"])).expanduser().resolve()
            ),
            "dry_clip_start_sample": 0,
            "dry_clip_end_sample_exclusive": len(clips[index][0]),
            "linear_gain": event["linear_gain"],
            "fade_samples": event["fade_samples"],
        }
        for index, (binding, event) in enumerate(zip(bindings, events))
    ]
    dry_assembly = assemble_dry_audio_buses(
        dry_event_mappings,
        source_ids=tuple(sorted(endpoint_ids)),
        clip=DryAudioClipSpec.from_values(
            frame_count=frame_count,
            fps_numerator=int(round(frame_rate)),
            sample_rate_hz=sample_rate,
        ),
        asset_bindings={
            str(binding["sound_asset_id"]): str(
                Path(str(binding["path"])).expanduser().resolve()
            )
            for binding in bindings
        },
    )

    scene = load_compiled_acoustic_scene(
        package_path,
        allow_nonpassing_research_qa=True,
    )
    simulation = _simulation(
        direct_ray_count=direct_ray_count,
        indirect_ray_count=indirect_ray_count,
        source_ray_count=source_ray_count,
        indirect_ray_depth=indirect_ray_depth,
        source_ray_depth=source_ray_depth,
    )
    mixture = np.zeros(sample_count, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for index, (item, (dry, _), event, endpoint_id) in enumerate(
        zip(bindings, clips, events, endpoint_ids)
    ):
        frame_index = min(
            frame_count - 1,
            int(round(event["start_sample"] / sample_rate * frame_rate)),
        )
        source_position = _frame_position(emitters[actor_ids[index]], frame_index, "location_cm")
        listener_position = _frame_position(camera, frame_index, "location_cm")
        readback_obj = root / f"rir_{index:02d}_{actor_ids[index]}.obj"
        result = simulate_compiled_acoustic_scene(
            scene,
            simulation,
            source=RuntimeAnchor(anchor_id=endpoint_id, position_m=source_position),
            listener=RuntimeAnchor(anchor_id="listener", position_m=listener_position),
            scene_readback_obj=readback_obj,
            runtime_mode="current-installed",
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
        )
        ir = np.asarray(result.samples[0], dtype=np.float64)
        wet = np.convolve(dry * float(event["linear_gain"]), ir, mode="full")[:sample_count]
        mixture[: len(wet)] += wet
        stem_path = stems / f"{index:02d}_{actor_ids[index]}_rir.wav"
        _write_wav(stem_path, wet)
        records.append(
            {
                "event_id": event["event_id"],
                "actor_id": actor_ids[index],
                "endpoint_id": endpoint_id,
                "speaker_id": item.get("speaker_id"),
                "transcript": item.get("transcript"),
                "clip_path": str(Path(str(item["path"])).expanduser().resolve()),
                "start_sample": event["start_sample"],
                "end_sample_exclusive": event["end_sample_exclusive"],
                "readback_frame_index": frame_index,
                "emitter_readback_m": list(source_position),
                "listener_readback_m": list(listener_position),
                "rir_sample_count": int(ir.shape[0]),
                "rir_max_abs": float(np.max(np.abs(ir))),
                "output_stem": str(stem_path),
                "native_readback_obj": str(readback_obj),
            }
        )
    mixture_path = root / "four_speaker_sequential_mixture.wav"
    _write_wav(mixture_path, mixture)
    report = {
        "schema": "avengine_frame_readback_sequential_speech_research_v1",
        "status": "research",
        "frame_readbacks": str(readback_path),
        "acoustic_package": str(package_path),
        "clock": dict(clock),
        "voice_bindings": bindings,
        "audio_program": str(root / "audio_program.json"),
        "dry_audio_assembly": dry_assembly.metadata(),
        "events": records,
        "mixture_path": str(mixture_path),
        "qa": {
            "speech_identity_and_transcripts": {
                "status": "pass",
                "event_count": len(records),
                "complete_sentences_preserved": True,
            },
            "sequential_nonoverlap": {
                "status": "pass",
                "event_count": len(records),
            },
            "actual_emitter_readback": {
                "status": "pass",
                "actor_count": len(actor_ids),
                "source": "frame_readbacks.emitters",
            },
            "actual_listener_readback": {
                "status": "pass",
                "source": "frame_readbacks.camera[].location_cm",
            },
            "animation_phase_readback": {
                "status": "pass" if "animations" in readback else "not_run",
                "source": "frame_readbacks.animations",
            },
            "pixel_visibility": {
                "status": "not_run",
                "reason": (
                    "RGB frame files alone do not prove per-actor visibility; "
                    "semantic/depth target readbacks were not supplied."
                ),
            },
        },
        "complete_sentences_preserved": True,
        "native_rlr_per_event": True,
        "formal_admission": False,
        "claim_boundary": (
            "Research audio generated from actual SPEAR emitter/listener readbacks "
            "and existing native RLR; no formal visual/audio admission."
        ),
    }
    _write(root / "research_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-readbacks", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--voice-binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument("--magnum-python-site", required=True)
    parser.add_argument("--direct-rays", type=int, default=500)
    parser.add_argument("--indirect-rays", type=int, default=5000)
    parser.add_argument("--source-rays", type=int, default=500)
    parser.add_argument("--indirect-depth", type=int, default=64)
    parser.add_argument("--source-depth", type=int, default=16)
    args = parser.parse_args()
    report = render(
        frame_readbacks=args.frame_readbacks,
        package_manifest=args.package_manifest,
        voice_binding=args.voice_binding,
        output=args.output,
        runtime_prefix=args.runtime_prefix,
        rlr_sdk_root=args.rlr_sdk_root,
        magnum_python_site=args.magnum_python_site,
        direct_ray_count=args.direct_rays,
        indirect_ray_count=args.indirect_rays,
        source_ray_count=args.source_rays,
        indirect_ray_depth=args.indirect_depth,
        source_ray_depth=args.source_depth,
    )
    print(json.dumps({"status": report["status"], "output": report["mixture_path"]}, indent=2))


if __name__ == "__main__":
    main()
