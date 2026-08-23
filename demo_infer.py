#!/usr/bin/env python3
"""Single-GPU smoke inference for released Spatial-Omni SO-7B weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from scripts.batch_bench_so_qa import (  # noqa: E402
    SpatialBeatsEvalCollator,
    clean_generated_answer,
    instantiate_model_for_checkpoint,
    to_generation_inputs,
)
from train_so_qa import SAMPLE_RATE  # noqa: E402


def binaural_to_foa(stereo: np.ndarray) -> np.ndarray:
    """Crude mid/side packing. Not physically correct FOA."""
    if stereo.ndim == 1:
        stereo = np.stack([stereo, stereo], axis=1)
    if stereo.shape[1] == 1:
        stereo = np.repeat(stereo, 2, axis=1)
    left, right = stereo[:, 0], stereo[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    foa = np.stack([mid, side, np.zeros_like(mid), np.zeros_like(mid)], axis=1)
    return foa.astype(np.float32)


def load_records(qa_jsonl: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in qa_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/data/models/spatial-omni/SO-7B_finetuned.pt")
    parser.add_argument("--qa-jsonl", default="/data/datasets/spatial-omni/demo/smoke.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    runtime = argparse.Namespace(
        device=args.device,
        device_map=None,
        dtype="bfloat16",
    )
    print(f"loading {args.ckpt}", flush=True)
    model, processor, train_args, _ckpt, load_result = instantiate_model_for_checkpoint(
        runtime, args.ckpt
    )
    print(
        "load missing",
        len(load_result.missing_keys),
        "unexpected",
        len(load_result.unexpected_keys),
        flush=True,
    )
    model.to(args.device)
    model.eval()

    collator = SpatialBeatsEvalCollator(processor=processor, sample_rate=SAMPLE_RATE)
    records = load_records(Path(args.qa_jsonl), args.limit)
    results = []
    for rec in records:
        batch = collator([rec])
        generation_inputs = to_generation_inputs(batch, args.device)
        with torch.no_grad():
            generated = model.generate(
                **generation_inputs,
                return_audio=False,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        prompt_len = generation_inputs["input_ids"].shape[1]
        text = processor.tokenizer.decode(
            generated[0, prompt_len:], skip_special_tokens=True
        ).strip()
        cleaned = clean_generated_answer(text)
        item = {
            "task": rec.get("task_name"),
            "question": rec.get("question"),
            "gold": rec.get("canonical_answer") or rec.get("answer"),
            "pred": cleaned,
            "raw": text,
            "audio_path": rec.get("audio_path"),
        }
        results.append(item)
        print("=" * 72)
        print("TASK", item["task"])
        print("Q   ", item["question"][:240])
        print("GOLD", item["gold"])
        print("PRED", item["pred"])

    out = REPO / "demo_data/smoke_predictions.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
