#!/usr/bin/env python3
"""VRAM probe: 2ch binaural (packed to 4ch) ± video through Spatial-Omni SO-7B.

This is a memory / wiring hack, not a correct FOA or AV-finetune.
2ch L/R is stuffed into the SO-Encoder as [L, R, 0, 0].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train_so_qa import (  # noqa: E402
    SAMPLE_RATE,
    apply_llm_lora,
    build_model,
    build_optimizer,
    build_processor,
    compute_batch_loss,
    configure_beats_lora_training,
    to_device,
)

AUDIO_PATH = Path("/data/datasets/spatial-omni/demo/avengine_0089.wav")
FRAME_DIR = Path(
    "/data/jzy/code/AVEngine-lead-a/tmp/lead_a_native_scenarios_v1/"
    "captures/occlusion_reappearance_0089/rgb_frames"
)
QUESTION = (
    "Listen to the binaural audio and watch the video. "
    "Is the main sound source more on the left or the right of the listener?"
)
ANSWER = "The main sound source is more on the left."


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
        f"{tag:36s}  alloc={rec['allocated_gb']:6.2f}  "
        f"reserved={rec['reserved_gb']:6.2f}  peak={rec['max_allocated_gb']:6.2f}",
        flush=True,
    )
    return rec


def pack_2ch_to_4ch(wav_tc: np.ndarray) -> np.ndarray:
    """wav_tc: [T, 2] -> [4, T] as [L, R, 0, 0]."""
    if wav_tc.ndim != 2 or wav_tc.shape[1] != 2:
        raise ValueError(f"expected [T, 2], got {wav_tc.shape}")
    left, right = wav_tc[:, 0], wav_tc[:, 1]
    zeros = np.zeros_like(left)
    return np.stack([left, right, zeros, zeros], axis=0).astype(np.float32)


def load_video(n_frames: int) -> np.ndarray:
    paths = sorted(FRAME_DIR.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(FRAME_DIR)
    n_frames = min(n_frames, len(paths))
    if n_frames % 2:
        n_frames -= 1
    if n_frames < 2:
        raise ValueError("need at least 2 frames")
    idxs = np.linspace(0, len(paths) - 1, n_frames).round().astype(int)
    frames = [np.array(Image.open(paths[i]).convert("RGB")) for i in idxs]
    return np.stack(frames, axis=0)


def make_args(device: str) -> argparse.Namespace:
    return argparse.Namespace(
        model_id="/data/models/Qwen2.5-Omni-7B",
        beats_checkpoint="/data/models/spatial-omni/SO-Encoder_finetuned.pt",
        beats_repo="",
        so_repo=str(REPO),
        train_mode="beats_lora",
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


def count_tokens(processor, batch: dict) -> dict:
    ids = batch["input_ids"][0]
    tok = processor.tokenizer
    video_id = tok.convert_tokens_to_ids(processor.video_token)
    audio_id = tok.convert_tokens_to_ids(processor.audio_token)
    spatial_id = tok.convert_tokens_to_ids(processor.spatial_token)
    return {
        "seq_len": int(ids.numel()),
        "n_video_tokens": int((ids == video_id).sum()),
        "n_audio_tokens": int((ids == audio_id).sum()),
        "n_spatial_tokens": int((ids == spatial_id).sum()),
        "video_grid_thw": (
            batch["video_grid_thw"].tolist() if "video_grid_thw" in batch else None
        ),
        "pixel_values_videos": (
            list(batch["pixel_values_videos"].shape)
            if "pixel_values_videos" in batch
            else None
        ),
        "spatial_audio": list(batch["spatial_audio"].shape),
        "audio_samples": int(batch["spatial_audio_lengths"][0].item()),
    }


def build_batch(
    processor,
    wav4: np.ndarray,
    video: np.ndarray | None,
    pad_20s: bool,
    max_pixels: int | None = None,
):
    wav = wav4
    if pad_20s and wav.shape[1] < SAMPLE_RATE * 20:
        pad = SAMPLE_RATE * 20 - wav.shape[1]
        wav = np.pad(wav, ((0, 0), (0, pad)))
    T = int(wav.shape[1])
    eos = getattr(processor.tokenizer, "eos_token", None) or ""
    if video is None:
        prefix = processor.audio_token + processor.spatial_token + f"\n{QUESTION}\n"
    else:
        prefix = (
            processor.video_token
            + processor.audio_token
            + processor.spatial_token
            + f"\n{QUESTION}\n"
        )
    ans_sfx = ANSWER + eos
    full = prefix + ans_sfx
    tok = processor.tokenizer
    prev = getattr(tok, "padding_side", "left")
    tok.padding_side = "right"
    kwargs = dict(
        text=[full],
        audio=[wav],
        padding=True,
        padding_side="right",
        return_tensors="pt",
        use_audio_in_video=False,
    )
    if video is not None:
        kwargs["videos"] = [video]
        kwargs["fps"] = float(video.shape[0]) / 5.0
        if max_pixels is not None:
            kwargs["max_pixels"] = int(max_pixels)
    try:
        batch = processor(**kwargs)
    finally:
        tok.padding_side = prev

    sa = torch.from_numpy(wav.T).unsqueeze(0).float()  # [1, T, 4]
    batch["spatial_audio"] = sa
    batch["spatial_audio_attention_mask"] = torch.ones(1, T, dtype=torch.bool)
    batch["spatial_audio_lengths"] = torch.tensor([T], dtype=torch.long)
    if "video_second_per_grid" in batch and not torch.is_tensor(batch["video_second_per_grid"]):
        batch["video_second_per_grid"] = torch.tensor(
            batch["video_second_per_grid"], dtype=torch.float32
        )

    labels = batch["input_ids"].clone()
    if "attention_mask" in batch:
        labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
    ab = processor.tokenizer(ans_sfx, padding=False, return_tensors="pt", add_special_tokens=False)
    al = int(ab["attention_mask"].sum().item())
    vl = int(batch["attention_mask"].sum().item())
    pl = vl - al
    if pl < 0:
        raise ValueError(f"negative prefix {pl} vl={vl} al={al}")
    labels[0, :pl] = -100
    batch["labels"] = labels
    batch["prefix_lengths"] = torch.tensor([pl], dtype=torch.long)
    return batch


def run_case(
    model,
    processor,
    device,
    wav4,
    video,
    pad_20s,
    tag,
    do_train,
    do_infer,
    max_pixels=None,
):
    rec = {
        "tag": tag,
        "n_frames": 0 if video is None else int(video.shape[0]),
        "frame_hw": None if video is None else [int(video.shape[1]), int(video.shape[2])],
        "pad_audio_20s": pad_20s,
        "max_pixels": max_pixels,
        "oom": False,
    }
    try:
        batch = build_batch(processor, wav4, video, pad_20s, max_pixels=max_pixels)
        rec.update(count_tokens(processor, batch))
        print(
            f"{tag}: seq={rec['seq_len']} video={rec['n_video_tokens']} "
            f"audio={rec['n_audio_tokens']} spatial={rec['n_spatial_tokens']} "
            f"grid={rec['video_grid_thw']}",
            flush=True,
        )
    except Exception as exc:
        rec["build_error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-2000:]
        print(f"{tag} BUILD FAIL {rec['build_error']}", flush=True)
        return rec

    if do_train:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model.train()
            opt = build_optimizer(model, make_args(device))
            opt.zero_grad(set_to_none=True)
            loss, stats = compute_batch_loss(model, batch, device)
            snapshot(f"{tag}/fwd")
            loss.backward()
            opt.step()
            mem = snapshot(f"{tag}/bwd")
            rec["train"] = {
                **mem,
                "loss": stats["loss"],
                "supervised_tokens": stats["supervised_tokens"],
            }
            opt.zero_grad(set_to_none=True)
            del opt, loss
        except torch.cuda.OutOfMemoryError as exc:
            rec["oom"] = True
            rec["train_oom"] = str(exc).split("\n")[0]
            print(f"{tag} TRAIN OOM", flush=True)
            torch.cuda.empty_cache()
        except Exception as exc:
            rec["train_error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-2500:]
            print(f"{tag} TRAIN FAIL {rec['train_error']}", flush=True)
            torch.cuda.empty_cache()

    if do_infer and not rec.get("oom"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model.eval()
            gen_batch = {
                k: v
                for k, v in batch.items()
                if k not in {"labels", "prefix_lengths", "meta"}
            }
            # generation prompt = drop answer tokens
            pl = int(batch["prefix_lengths"][0].item())
            gen_batch["input_ids"] = batch["input_ids"][:, :pl]
            gen_batch["attention_mask"] = batch["attention_mask"][:, :pl]
            with torch.no_grad():
                out = model.generate(
                    **to_device(gen_batch, device),
                    return_audio=False,
                    max_new_tokens=16,
                    num_beams=1,
                    do_sample=False,
                )
            mem = snapshot(f"{tag}/infer")
            rec["infer"] = {
                **mem,
                "gen_tokens": int(out.shape[1] - pl),
                "text": processor.tokenizer.decode(out[0, pl:], skip_special_tokens=True),
            }
            del out
        except torch.cuda.OutOfMemoryError as exc:
            rec["oom"] = True
            rec["infer_oom"] = str(exc).split("\n")[0]
            print(f"{tag} INFER OOM", flush=True)
            torch.cuda.empty_cache()
        except Exception as exc:
            rec["infer_error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-2500:]
            print(f"{tag} INFER FAIL {rec['infer_error']}", flush=True)
            torch.cuda.empty_cache()

    del batch
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", default="0,2,8,16,32,74")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--out", default="/data/datasets/spatial-omni/demo/av_mem.json")
    args = parser.parse_args()

    wav2, sr = sf.read(str(AUDIO_PATH), dtype="float32", always_2d=True)
    print(f"audio {AUDIO_PATH} shape={wav2.shape} sr={sr} ch={wav2.shape[1]}", flush=True)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"expected {SAMPLE_RATE} Hz, got {sr}")
    wav4 = pack_2ch_to_4ch(wav2)
    print(f"packed 2ch->4ch {wav4.shape} (L,R,0,0)", flush=True)

    frame_counts = [int(x) for x in args.frames.split(",") if x.strip()]

    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    targs = make_args(args.device)
    processor = build_processor(targs.model_id, targs.so_repo)
    model = build_model(targs, processor)
    model, _ = apply_llm_lora(model, targs)
    configure_beats_lora_training(model, targs)
    model.to(args.device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"params trainable={trainable/1e6:.1f}M  total={total/1e6:.1f}M", flush=True)
    rows = [snapshot("after_load")]
    rows[0]["trainable_m"] = round(trainable / 1e6, 1)
    rows[0]["total_m"] = round(total / 1e6, 1)
    out = Path(args.out)

    def _flush() -> None:
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    default_pixels = 768 * 28 * 28  # Qwen2.5-Omni processor default
    compact_pixels = 256 * 28 * 28  # more practical video token budget

    def _infer_ok(n: int) -> bool:
        if args.skip_infer:
            return False
        if args.skip_train:
            return True
        return n in {0, 8, 16}

    def _run(**kwargs):
        rec = run_case(model, processor, args.device, wav4, **kwargs)
        rows.append(rec)
        _flush()
        return rec

    for n in frame_counts:
        video = None if n <= 0 else load_video(n)
        tag = "audio2ch" if video is None else f"audio2ch+video{video.shape[0]}f"
        _run(
            video=video,
            pad_20s=False,
            tag=tag,
            do_train=not args.skip_train,
            do_infer=_infer_ok(n if video is None else video.shape[0]),
            max_pixels=None if video is None else default_pixels,
        )

    for n in [n for n in frame_counts if n >= 2]:
        video = load_video(n)
        _run(
            video=video,
            pad_20s=False,
            tag=f"audio2ch+video{video.shape[0]}f_compact",
            do_train=not args.skip_train,
            do_infer=_infer_ok(video.shape[0]),
            max_pixels=compact_pixels,
        )

    if 16 in frame_counts:
        _run(
            video=load_video(16),
            pad_20s=True,
            tag="audio2ch_20s+video16f_compact",
            do_train=not args.skip_train,
            do_infer=not args.skip_infer,
            max_pixels=compact_pixels,
        )

    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
