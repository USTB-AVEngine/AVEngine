"""Plan four distinct complete VCTK sentences on an AVEngine clock.

This is an audio-program/duration planner only. It never truncates a source
sentence and does not claim listener/emitter readback or render RLR audio.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import wave
from typing import Any, Mapping

from avengine.timeline.audio_program import bind_audio_program_hash, validate_audio_program

TIME_BASE_HZ = 48_000
TICKS_PER_SAMPLE = 3


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clip_metadata(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"VCTK clip must be mono int16 PCM: {path}")
        rate = handle.getframerate()
        count = handle.getnframes()
    if rate != 16000 or count < 1:
        raise ValueError(f"VCTK clip must be nonempty 16 kHz PCM: {path}")
    sidecar = path.with_name("clip.json")
    if not sidecar.is_file():
        raise ValueError(f"VCTK provenance sidecar is missing: {sidecar}")
    return count, rate


def plan(binding_path: str | Path, output: str | Path, *, gap_seconds: float = 0.20, margin_seconds: float = 0.50) -> dict[str, Any]:
    binding_file = Path(binding_path).expanduser().resolve()
    bindings = _load(binding_file)
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise ValueError("context voice binding must contain exactly four records")
    output_root = Path(output).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output required: {output_root}")
    output_root.mkdir(parents=True)
    fps = 15
    rate = 16000
    gap = int(round(gap_seconds * rate))
    margin = int(round(margin_seconds * rate))
    events: list[dict[str, Any]] = []
    durations: list[dict[str, Any]] = []
    cursor = margin
    endpoint_ids = []
    for index, item in enumerate(bindings):
        path = Path(str(item["path"])).expanduser().resolve()
        count, clip_rate = _clip_metadata(path)
        sidecar = _load(path.with_name("clip.json"))
        transcript = sidecar.get("transcript")
        if transcript != item.get("transcript"):
            raise ValueError(
                f"binding transcript differs from source sidecar for {path}"
            )
        actor_id = str(item["actor_id"])
        endpoint_id = f"{actor_id}_mouth"
        endpoint_ids.append(endpoint_id)
        start = cursor
        end = start + count
        events.append(
            {
                "event_id": f"context_vctk_turn_{index + 1:02d}",
                "source_endpoint_id": endpoint_id,
                "sound_asset_id": str(item["sound_asset_id"]),
                "start_tick": start * TICKS_PER_SAMPLE,
                "end_tick_exclusive": end * TICKS_PER_SAMPLE,
                "start_sample": start,
                "end_sample_exclusive": end,
                "source_start_sample": 0,
                "source_end_sample_exclusive": count,
                "linear_gain": float(item.get("linear_gain", 0.15)),
                "fade_samples": 80,
                "normalization_policy": "use_sound_asset_policy",
                "render_source_stem": True,
            }
        )
        durations.append(
            {
                "actor_id": actor_id,
                "endpoint_id": endpoint_id,
                "speaker_id": item.get("speaker_id"),
                "gender": item.get("gender"),
                "split": sidecar.get("split"),
                "transcript": transcript,
                "clip_path": str(path),
                "clip_sample_count": count,
                "clip_seconds": count / clip_rate,
                "start_sample": start,
                "end_sample_exclusive": end,
            }
        )
        cursor = end + (gap if index < len(bindings) - 1 else 0)
    required_seconds = math.ceil(cursor / rate)
    frame_count = required_seconds * fps
    sample_count = frame_count * rate // fps
    timeline = {
        "time_base_hz": TIME_BASE_HZ,
        "ticks_per_frame": TIME_BASE_HZ // fps,
        "video_fps": fps,
        "frame_count": frame_count,
        "sample_rate_hz": rate,
        "ticks_per_sample": TICKS_PER_SAMPLE,
        "sample_count": sample_count,
    }
    program = bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": "polished_room_four_distinct_vctk_turn_taking_research_v1",
            "revision": "v1",
            "mode": "sequential_sources",
            "timeline": timeline,
            "candidate_source_endpoint_ids": sorted(endpoint_ids),
            "events": events,
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )
    errors = validate_audio_program(program)
    if errors:
        raise ValueError("generated program failed validation: " + "; ".join(errors))
    _write(output_root / "audio_program.json", program)
    report = {
        "schema": "avengine_four_distinct_vctk_duration_plan_v1",
        "status": "research",
        "clock": timeline,
        "required_content_seconds": cursor / rate,
        "clock_seconds": sample_count / rate,
        "slack_seconds": (sample_count - cursor) / rate,
        "gap_seconds": gap / rate,
        "margin_seconds": margin / rate,
        "events": durations,
        "voice_binding": str(binding_file),
        "future_frame_readbacks_required": True,
        "listener_emitter_render_status": "not_run",
        "complete_sentences_preserved": True,
        "claim_boundary": (
            "Four distinct complete VCTK sentence plan only; no RLR audio, "
            "listener readback, emitter readback, pixel or formal admission claim."
        ),
    }
    _write(output_root / "duration_plan.json", report)
    _write(output_root / "voice_binding.json", {"bindings": bindings})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plan(args.voice_binding, args.output)
    print(json.dumps({"status": result["status"], "clock": result["clock"], "slack_seconds": result["slack_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
