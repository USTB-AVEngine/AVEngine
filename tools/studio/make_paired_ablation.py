#!/usr/bin/env python3
"""Paired ablation variants for a rendered dynamic-audio bundle.

Takes one rendered audio directory (binaural mixture + per-source stems)
and derives end-of-chain ablation variants that share the parent's seed,
trajectories, and RIRs byte-for-byte up to the final transform:

  left_zeroed   — left channel set to zero (Left-bias ablation)
  right_zeroed  — right channel set to zero
  mono_folded   — both channels replaced by their mean (spatial cue removal)
  muted:<stem>  — the named stem removed from the mixture (per --mute-stem)

Every variant is written as a fresh wav next to a pair manifest binding
parent and variants under one pair_id. Research-only: the manifest carries
research_only/episode_counted=false and no dataset admission is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PAIR_SCHEMA = "avengine_studio_paired_ablation_v1"


class PairedAblationError(ValueError):
    """Raised on malformed inputs."""


def read_float32_stereo_wav(path: Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise PairedAblationError(f"{path} is not a stereo wav")
    if data.dtype != np.float32:
        raise PairedAblationError(f"{path} must be float32 PCM, got {data.dtype}")
    return int(rate), data


def write_float32_wav(path: Path, rate: int, data: np.ndarray) -> str:
    from scipy.io import wavfile

    if path.exists():
        raise PairedAblationError(f"output exists (fresh/no-clobber): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, np.ascontiguousarray(data, dtype=np.float32))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_paired_ablation(
    audio_dir: str | Path,
    output_dir: str | Path,
    *,
    pair_id: str,
    mute_stems: tuple[str, ...] = (),
) -> dict:
    audio_root = Path(audio_dir).resolve()
    mixture_path = audio_root / "audio" / "binaural" / "mixture.wav"
    if not mixture_path.is_file():
        raise PairedAblationError(f"mixture not found: {mixture_path}")
    rate, mixture = read_float32_stereo_wav(mixture_path)

    stems: dict[str, np.ndarray] = {}
    for stem_path in sorted((audio_root / "audio" / "binaural").glob("*_stem.wav")):
        stem_rate, stem = read_float32_stereo_wav(stem_path)
        if stem_rate != rate or stem.shape != mixture.shape:
            raise PairedAblationError(f"stem shape/rate mismatch: {stem_path}")
        stems[stem_path.stem.removesuffix("_stem")] = stem
    unknown = sorted(set(mute_stems) - set(stems))
    if unknown:
        raise PairedAblationError(
            f"unknown stems {unknown}; available: {sorted(stems)}"
        )

    output_root = Path(output_dir).resolve()
    if output_root.exists():
        raise PairedAblationError(f"output exists (fresh/no-clobber): {output_root}")
    output_root.mkdir(parents=True)

    parent_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()
    variants: dict[str, dict] = {}

    def add_variant(name: str, transform: str, data: np.ndarray) -> None:
        wav_path = output_root / f"{name}.wav"
        sha = write_float32_wav(wav_path, rate, data)
        variants[name] = {"path": str(wav_path), "sha256": sha, "transform": transform}

    left = mixture.copy(); left[:, 0] = 0.0
    add_variant("left_zeroed", "left channel set to zero", left)
    right = mixture.copy(); right[:, 1] = 0.0
    add_variant("right_zeroed", "right channel set to zero", right)
    mono = np.repeat(mixture.mean(axis=1, keepdims=True), 2, axis=1)
    add_variant("mono_folded", "channels replaced by their mean", mono)
    for stem_name in mute_stems:
        add_variant(
            f"muted_{stem_name}",
            f"stem {stem_name} subtracted from the mixture",
            mixture - stems[stem_name],
        )

    manifest = {
        "schema": PAIR_SCHEMA,
        "pair_id": pair_id,
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "claim_boundary": (
            "end-of-chain ablation variants sharing the parent render's seed, "
            "trajectories, and RIRs; no dataset admission is claimed"
        ),
        "parent": {
            "audio_dir": str(audio_root),
            "mixture_path": str(mixture_path),
            "mixture_sha256": parent_sha,
            "sample_rate_hz": rate,
            "sample_count": int(mixture.shape[0]),
            "stems": sorted(stems),
        },
        "variants": variants,
    }
    manifest_path = output_root / "pair_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path,
                        help="rendered dynamic-audio output directory")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--mute-stem", action="append", default=[],
                        help="also derive a variant with this stem removed; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_paired_ablation(
        args.audio_dir,
        args.output,
        pair_id=args.pair_id,
        mute_stems=tuple(args.mute_stem),
    )
    print(json.dumps(
        {
            "status": "pass",
            "pair_id": manifest["pair_id"],
            "variants": sorted(manifest["variants"]),
            "manifest": str(Path(args.output).resolve() / "pair_manifest.json"),
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
