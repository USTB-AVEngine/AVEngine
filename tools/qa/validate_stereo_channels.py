#!/usr/bin/env python3
"""Stereo-channel integrity validator (pilot work order item 1.1).

评测入口的前置检查:进模型的音频必须保留两条**互不相同**的声道。历史
事故(20260823):评测管线把双耳折成单声道(预处理后形状 [[76800]]),
导致一切"空间能力"分数作废——本工具是那次事故的制度化防线,挂在所有
评测与探针入口,不过即停。

判定(PASS 须同时满足):
  - 声道数恰为 2;
  - 逐样本 |L−R| > amp_epsilon 的样本占比 >= min_diff_ratio(默认 1%)。
其余一律 FAIL:单声道、双声道但 L==R 复制、空文件、双声道静音。

用法:
  validate_stereo_channels.py PATH [PATH ...] \
      [--min-diff-ratio 0.01] [--amp-epsilon 1e-6] [--manifest OUT.json]

  PATH 是 wav 文件或目录(目录递归收 *.wav)。任一文件 FAIL 时以非零码
  退出(失败即停);--manifest 写逐文件 JSON 记录,目标已存在即拒绝
  (fresh/no-clobber)。库用法:
  from validate_stereo_channels import validate_wav

阈值是显式参数并写入 manifest;不做任何重采样或响度归一(原样校验)。
research_candidate;不构成 dataset admission。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def _read_wav(path: str):
    """读 wav 为 (frames, channels) float64;优先 soundfile,回退 stdlib wave。"""
    try:
        import soundfile as sf  # noqa: PLC0415

        data, rate = sf.read(path, always_2d=True)
        return np.asarray(data, dtype=np.float64), int(rate)
    except ImportError:
        import wave  # noqa: PLC0415

        with wave.open(path, "rb") as handle:
            frames = handle.getnframes()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(frames)
        if width == 2:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif width == 4:
            arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(
                f"unsupported sample width {width} bytes; install soundfile for float wavs"
            )
        if channels <= 0:
            raise ValueError("wav reports zero channels")
        return arr.reshape(-1, channels), int(rate)


def validate_wav(path: str, *, min_diff_ratio: float = 0.01, amp_epsilon: float = 1e-6) -> dict:
    """校验单个 wav;返回逐文件记录 dict(status = pass|fail,含原因与实测占比)。"""
    record: dict = {"file": str(path)}
    data, rate = _read_wav(path)
    frames, channels = data.shape
    record.update(sample_rate=rate, channels=channels, frames=frames)
    if channels != 2:
        record.update(status="fail", diff_ratio=0.0,
                      reason=f"expected exactly 2 channels, got {channels}")
        return record
    if frames == 0:
        record.update(status="fail", diff_ratio=0.0, reason="empty audio")
        return record
    diff_ratio = float(np.mean(np.abs(data[:, 0] - data[:, 1]) > amp_epsilon))
    record["diff_ratio"] = round(diff_ratio, 6)
    if diff_ratio < min_diff_ratio:
        record.update(status="fail",
                      reason=(f"channels nearly identical: diff_ratio {diff_ratio:.6f} "
                              f"< min_diff_ratio {min_diff_ratio}"))
    else:
        record["status"] = "pass"
    return record


def _collect(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.wav"), recursive=True)))
        else:
            files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--min-diff-ratio", type=float, default=0.01)
    parser.add_argument("--amp-epsilon", type=float, default=1e-6)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args(argv)

    if args.manifest and os.path.exists(args.manifest):
        print(f"refusing to overwrite existing manifest: {args.manifest}", file=sys.stderr)
        return 2

    files = _collect(args.paths)
    if not files:
        print("no wav files found under the given paths", file=sys.stderr)
        return 2

    records = []
    failures = 0
    for f in files:
        try:
            rec = validate_wav(f, min_diff_ratio=args.min_diff_ratio,
                               amp_epsilon=args.amp_epsilon)
        except Exception as exc:  # 读不动也是 FAIL,不静默
            rec = {"file": str(f), "status": "fail", "reason": f"unreadable: {exc!r}"}
        records.append(rec)
        if rec["status"] != "pass":
            failures += 1
            print(f"FAIL {f}: {rec.get('reason')}")
    print(f"checked={len(records)} pass={len(records) - failures} fail={failures} "
          f"min_diff_ratio={args.min_diff_ratio} amp_epsilon={args.amp_epsilon}")

    if args.manifest:
        payload = {
            "schema": "avengine_stereo_channel_validation_v1",
            "status": "research_candidate",
            "qualification_claim": False,
            "parameters": {"min_diff_ratio": args.min_diff_ratio,
                           "amp_epsilon": args.amp_epsilon},
            "checked": len(records),
            "failures": failures,
            "records": records,
        }
        with open(args.manifest, "w") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=1)
        print(f"manifest={args.manifest}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
