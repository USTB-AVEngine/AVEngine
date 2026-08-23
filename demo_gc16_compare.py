#!/usr/bin/env python3
"""16-frame train step: gradient checkpointing on vs off, time and VRAM."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from demo_av_mem import (  # noqa: E402
    ANSWER,
    AUDIO_PATH,
    QUESTION,
    build_batch,
    gb,
    load_video,
    pack_2ch_to_4ch,
)
from train_so_qa import (  # noqa: E402
    SAMPLE_RATE,
    apply_llm_lora,
    build_model,
    build_optimizer,
    build_processor,
    compute_batch_loss,
    configure_beats_lora_training,
    enable_gradient_checkpointing,
)


def make_args(device: str, gc: bool) -> argparse.Namespace:
    return argparse.Namespace(
        model_id="/data/models/Qwen2.5-Omni-7B",
        beats_checkpoint="/data/models/spatial-omni/SO-Encoder_finetuned.pt",
        beats_repo="",
        so_repo=str(REPO),
        train_mode="beats_lora",
        device=device,
        dtype="bfloat16",
        attn_impl="sdpa",
        gradient_checkpointing=gc,
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


def token_stats(processor, batch: dict) -> dict:
    ids = batch["input_ids"][0]
    tok = processor.tokenizer
    return {
        "seq_len": int(ids.numel()),
        "n_video_tokens": int((ids == tok.convert_tokens_to_ids(processor.video_token)).sum()),
        "n_audio_tokens": int((ids == tok.convert_tokens_to_ids(processor.audio_token)).sum()),
        "n_spatial_tokens": int((ids == tok.convert_tokens_to_ids(processor.spatial_token)).sum()),
        "video_grid_thw": batch["video_grid_thw"].tolist() if "video_grid_thw" in batch else None,
    }


def run_one(device: str, processor, wav4, video, max_pixels: int, gc: bool, repeats: int) -> dict:
    tag = f"{'compact' if max_pixels <= 256 * 28 * 28 else 'default'}_16f_gc{'on' if gc else 'off'}"
    print(f"\n===== {tag} =====", flush=True)
    targs = make_args(device, gc)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(targs, processor)
    model, _ = apply_llm_lora(model, targs)
    if gc:
        enable_gradient_checkpointing(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
        model.thinker.config.use_cache = False
    configure_beats_lora_training(model, targs)
    model.to(device)
    model.train()
    rec = {
        "tag": tag,
        "gradient_checkpointing": gc,
        "max_pixels": max_pixels,
        "n_frames": int(video.shape[0]),
        "after_load_gb": round(gb(torch.cuda.memory_allocated()), 2),
    }
    try:
        batch = build_batch(processor, wav4, video, pad_20s=False, max_pixels=max_pixels)
        rec.update(token_stats(processor, batch))
        print(
            f"{tag}: seq={rec['seq_len']} video={rec['n_video_tokens']} "
            f"audio={rec['n_audio_tokens']} spatial={rec['n_spatial_tokens']}",
            flush=True,
        )
    except Exception as exc:
        rec["build_error"] = f"{type(exc).__name__}: {exc}"
        del model
        torch.cuda.empty_cache()
        return rec

    opt = build_optimizer(model, targs)
    step_times = []
    try:
        for i in range(repeats + 1):
            torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss, stats = compute_batch_loss(model, batch, device)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            row = {
                "i": i,
                "warmup": i == 0,
                "fwd_s": round(t1 - t0, 3),
                "bwd_s": round(t2 - t1, 3),
                "step_s": round(t2 - t0, 3),
                "loss": stats["loss"],
                "peak_allocated_gb": round(gb(torch.cuda.max_memory_allocated()), 2),
                "peak_reserved_gb": round(gb(torch.cuda.max_memory_reserved()), 2),
                "allocated_gb": round(gb(torch.cuda.memory_allocated()), 2),
            }
            print(
                f"{tag} step{i}{' (warmup)' if i == 0 else ''}: "
                f"fwd={row['fwd_s']:.3f}s bwd={row['bwd_s']:.3f}s "
                f"step={row['step_s']:.3f}s peak={row['peak_allocated_gb']:.2f}/"
                f"{row['peak_reserved_gb']:.2f} GB",
                flush=True,
            )
            if i > 0:
                step_times.append(row)
            del loss
        rec["oom"] = False
        rec["steps"] = step_times
        rec["mean_fwd_s"] = round(sum(s["fwd_s"] for s in step_times) / len(step_times), 3)
        rec["mean_bwd_s"] = round(sum(s["bwd_s"] for s in step_times) / len(step_times), 3)
        rec["mean_step_s"] = round(sum(s["step_s"] for s in step_times) / len(step_times), 3)
        rec["peak_allocated_gb"] = max(s["peak_allocated_gb"] for s in step_times)
        rec["peak_reserved_gb"] = max(s["peak_reserved_gb"] for s in step_times)
    except torch.cuda.OutOfMemoryError as exc:
        rec["oom"] = True
        rec["oom_msg"] = str(exc).split("\n")[0]
        rec["peak_allocated_gb"] = round(gb(torch.cuda.max_memory_allocated()), 2)
        rec["peak_reserved_gb"] = round(gb(torch.cuda.max_memory_reserved()), 2)
        print(f"{tag} OOM after peak {rec['peak_allocated_gb']}/{rec['peak_reserved_gb']} GB", flush=True)
        torch.cuda.empty_cache()
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-2500:]
        print(f"{tag} FAIL {rec['error']}", flush=True)
        torch.cuda.empty_cache()

    del model, opt, batch
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    wav2, sr = sf.read(str(AUDIO_PATH), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"expected {SAMPLE_RATE} Hz, got {sr}")
    wav4 = pack_2ch_to_4ch(wav2)
    video = load_video(16)
    print(f"audio {wav2.shape} packed {wav4.shape} video {video.shape}", flush=True)

    torch.cuda.set_device(args.device)
    processor = build_processor("/data/models/Qwen2.5-Omni-7B", str(REPO))

    cases = [
        (256 * 28 * 28, False),
        (256 * 28 * 28, True),
        (768 * 28 * 28, False),
        (768 * 28 * 28, True),
    ]
    rows = []
    out = Path("/data/datasets/spatial-omni/demo/gc16_compare.json")
    for max_pixels, gc in cases:
        rec = run_one(args.device, processor, wav4, video, max_pixels, gc, args.repeats)
        rows.append(rec)
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
