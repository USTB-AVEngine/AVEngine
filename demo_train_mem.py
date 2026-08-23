#!/usr/bin/env python3
"""One-step training VRAM probe for Spatial-Omni SO-7B."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train_so_qa import (  # noqa: E402
    SAMPLE_RATE,
    SpatialBeatsQACollator,
    apply_llm_lora,
    build_model,
    build_optimizer,
    build_processor,
    compute_batch_loss,
    configure_beats_lora_training,
    configure_encoder_lora_training,
    freeze_all_but_projector,
)


def gb(n: int) -> float:
    return n / (1024**3)


def snapshot(tag: str) -> dict:
    torch.cuda.synchronize()
    rec = {
        "tag": tag,
        "allocated_gb": round(gb(torch.cuda.memory_allocated()), 2),
        "reserved_gb": round(gb(torch.cuda.memory_reserved()), 2),
        "max_allocated_gb": round(gb(torch.cuda.max_memory_allocated()), 2),
    }
    print(
        f"{tag:32s}  alloc={rec['allocated_gb']:6.2f}  "
        f"reserved={rec['reserved_gb']:6.2f}  peak={rec['max_allocated_gb']:6.2f}",
        flush=True,
    )
    return rec


def make_args(mode: str, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        model_id="/data/models/Qwen2.5-Omni-7B",
        beats_checkpoint="/data/models/spatial-omni/SO-Encoder_finetuned.pt",
        beats_repo="",
        so_repo=str(REPO),
        train_mode=mode,
        device=device,
        dtype="bfloat16",
        attn_impl="sdpa",
        gradient_checkpointing=False,
        projector_fp32=False,
        projector_type="pixel_shuffle",
        projector_shuffle_factor=4,
        encoder_token_rate=10.0,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_target_prefixes=["thinker.model"],
        lr=3e-5,
        projector_lr=1e-6,
        lora_lr=3e-5,
        beats_lr=1e-6,
        spatial_null_lr=None,
        weight_decay=0.01,
        projector_weight_decay=None,
        device_map=None,
    )


def one_mode(mode: str, records: list, batch_sizes: list[int], device: str) -> list[dict]:
    print(f"\n===== {mode} =====", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    args = make_args(mode, device)
    processor = build_processor(args.model_id, args.so_repo)
    model = build_model(args, processor)
    if mode == "projector_only":
        freeze_all_but_projector(model)
    elif mode == "encoder_lora":
        model, _ = apply_llm_lora(model, args)
        configure_encoder_lora_training(model, args)
    elif mode == "beats_lora":
        model, _ = apply_llm_lora(model, args)
        configure_beats_lora_training(model, args)
    model.to(device)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"params trainable={trainable/1e6:.1f}M  total={total/1e6:.1f}M", flush=True)
    rows = [snapshot(f"{mode}/after_load")]
    collator = SpatialBeatsQACollator(processor=processor, sample_rate=SAMPLE_RATE)
    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        batch_recs = (records * ((bs + len(records) - 1) // len(records)))[:bs]
        batch = collator(batch_recs)
        opt = build_optimizer(model, args)
        opt.zero_grad(set_to_none=True)
        loss, stats = compute_batch_loss(model, batch, device)
        snapshot(f"{mode}/bs{bs}_forward")
        loss.backward()
        opt.step()
        rec = snapshot(f"{mode}/bs{bs}_backward")
        rec["loss"] = stats["loss"]
        rec["trainable_m"] = round(trainable / 1e6, 1)
        rec["total_m"] = round(total / 1e6, 1)
        rec["mode"] = mode
        rec["batch_size"] = bs
        rows.append(rec)
        opt.zero_grad(set_to_none=True)
        del batch, opt, loss
        torch.cuda.empty_cache()
    del model, processor
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--modes",
        default="projector_only,encoder_lora,beats_lora",
    )
    parser.add_argument("--batch-sizes", default="1,2")
    args = parser.parse_args()
    rec_path = Path("/data/datasets/spatial-omni/demo/smoke.jsonl")
    records = []
    for line in rec_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "foa_44e5c481f2533e830170" in row.get("audio_path", ""):
            row["prompt"] = row.get("question")
            records.append(row)
            break
    if not records:
        raise SystemExit("need the 20s FOA clip in smoke.jsonl")
    print("probe clip", records[0]["audio_path"], flush=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    all_rows = []
    for mode in modes:
        all_rows.extend(one_mode(mode, records, batches, args.device))
    out = Path("/data/datasets/spatial-omni/demo/train_mem.json")
    out.write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
