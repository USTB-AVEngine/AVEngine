#!/usr/bin/env python3
"""Transcribe declared review audio with an installed Whisper model.

This diagnostic tests speech content after rendering. Mono ASR input cannot
establish spatial answerability or audio/visual modality necessity. Original
multichannel media are read only; model, decoding settings, inputs and windows
are supplied by one JSON request, and results use a fresh output directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def run(request_path: Path, output: Path) -> None:
    import whisper

    request_path = request_path.resolve()
    request = json.loads(request_path.read_text())
    def resolve(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else request_path.parent / path).resolve()
    model_path = resolve(request["model_path"])
    if not model_path.is_file():
        raise ValueError(f"model weights are missing: {model_path}")
    items = request["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("request items must be a nonempty list")
    ids = [item["id"] for item in items]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("review item IDs must be unique nonempty strings")
    decoding = dict(request["decoding"])
    decoding.setdefault("verbose", False)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    try:
        model = whisper.load_model(str(model_path), device=request["device"])
        records = []
        for item in items:
            audio_path = resolve(item["audio_path"])
            # The installed SDK performs its documented mono/16 kHz conversion.
            # This is an ASR copy in memory, never a replacement of dataset audio.
            samples = whisper.load_audio(str(audio_path))
            rate = whisper.audio.SAMPLE_RATE
            duration = len(samples) / rate
            start, end = item.get("window_seconds", [0.0, duration])
            start, end = float(start), float(end)
            if not (math.isfinite(start) and math.isfinite(end)
                    and 0 <= start < end <= duration):
                raise ValueError(f"invalid review window for {item['id']}")
            chunk = samples[round(start * rate):round(end * rate)]
            if not len(chunk):
                raise ValueError(f"empty review window for {item['id']}")
            result = model.transcribe(chunk, **decoding)
            records.append({"id": item["id"], "audio_path": str(audio_path),
                            "window_seconds": [start, end],
                            "text": result["text"], "segments": result["segments"],
                            "segment_time_origin_seconds": start})
            print(f"transcribed {item['id']}: {result['text']}", flush=True)
        payload = {"status": "complete", "research_only": True,
                   "model_path": str(model_path), "runtime_module": whisper.__file__,
                   "device": request["device"], "decoding": decoding,
                   "asr_input": "Whisper SDK mono 16 kHz conversion; source media unchanged",
                   "boundary": "speech-content diagnostic; no spatial or modality certification",
                   "records": records}
        (output / "transcripts.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({
            "status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.request, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
