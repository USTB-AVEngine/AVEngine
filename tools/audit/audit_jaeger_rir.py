#!/usr/bin/env python3
"""JAEGER SpatialSceneQA 公开包 RIR 混响审计脚本(可重跑版)。

用途
  对解包后的 val/ 目录下全部 ir*.npy 计算能量衰减类指标,输出逐文件 CSV
  与按场景(scene)分层的汇总表,供审计报告引用。只读分析,不改任何输入。

用法
  python3 audit_jaeger_rir.py --input <解包根目录(含 val/)> \
      --out-csv rir_audit_val_full.csv --out-summary rir_audit_summary.md

算法与参数声明(与报告口径一致,改动须同步报告)
  采样率      : RIR 无显式采样率;主口径按 16 kHz(官方卷积脚本
                conv_ir_speaker_foa.py 不重采样,干声为 LibriSpeech 16 kHz,
                故 RIR 隐含 16 kHz);汇总同时给 44.1 kHz 口径换算。
  能量        : 四通道(FOA)逐样本平方和;一维数组按单通道处理。
  能量峰(直达): 能量序列的最大值位置(索引)。
  直达窗      : [峰-DIRECT_PRE, 峰+DIRECT_POST) 采样,默认 16/40,
                即 16 kHz 口径下峰前 1.0 ms、峰后 2.5 ms。
  DRR         : 10*log10(直达窗能量 / 窗后剩余能量),分母下限 1e-300
                防除零;窗前能量单独报告占比(pre_pct)。
  衰减曲线    : Schroeder 逆积分(能量从后往前累加),相对总能量取 dB;
                本批 RIR 尾部为数值零(1e-50 量级以下),不做底噪补偿——
                若移植到真实录音需先加噪声地板估计,超出本审计范围。
  交叉点 dNN  : 衰减曲线首次 <= -NN dB 的样本位;不存在记 -1。
  T20         : 取 -5 与 -25 dB 交叉点之间做线性拟合,斜率外推 60 dB;
                区间为空或非降(本批普遍如此)记 NaN——"T20 不可计算"
                本身即"无衰减坡"的证据。
  峰后间距    : d60 - 峰位,单位采样;衡量响应在直达之后延续多久。
  shape 检查  : 严格要求二维且含 4 通道轴((4,N) 或 (N,4)),否则记错误;
                不接受一维或其他形状。
  异常处理    : 无法加载/空数组/全零/形状不符 → 记入 errors 列表并写进
                汇总,**且脚本以非零退出码结束**(失败即停,不静默)。
  输出保护    : 输出文件已存在即拒绝执行(no-clobber)。
"""

import argparse
import csv
import glob
import json
import os
import statistics as st

import numpy as np

SR_MAIN = 16000.0        # 主口径采样率(推断依据见 docstring)
SR_ALT = 44100.0         # 备选口径
DIRECT_PRE = 16          # 直达窗峰前采样数(16k 口径 1.0 ms)
DIRECT_POST = 40         # 直达窗峰后采样数(16k 口径 2.5 ms)
EPS = 1e-300


def analyze_one(path):
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array, got ndim={arr.ndim} shape={arr.shape}")
    if arr.shape[0] == 4:
        pass
    elif arr.shape[1] == 4:
        arr = arr.T
    else:
        raise ValueError(f"no 4-channel axis in shape {arr.shape}")
    energy = (arr.astype(np.float64) ** 2).sum(axis=0)
    total = float(energy.sum())
    n = int(energy.size)
    if n == 0 or total <= 0.0:
        raise ValueError("empty or all-zero RIR")
    peak = int(np.argmax(energy))
    d0, d1 = max(0, peak - DIRECT_PRE), min(n, peak + DIRECT_POST)
    direct = float(energy[d0:d1].sum())
    rest = float(energy[d1:].sum())
    pre = float(energy[:d0].sum())
    sch = np.cumsum(energy[::-1])[::-1]
    sch_db = 10.0 * np.log10(sch / sch[0] + EPS)

    def cross(th):
        idx = np.where(sch_db <= th)[0]
        return int(idx[0]) if idx.size else -1

    d5, d20, d25 = cross(-5.0), cross(-20.0), cross(-25.0)
    d40, d60 = cross(-40.0), cross(-60.0)
    t20 = float("nan")
    if 0 <= d5 < d25:
        xs = np.arange(d5, d25)
        slope = np.polyfit(xs, sch_db[d5:d25], 1)[0] if xs.size >= 2 else 0.0
        if slope < 0:
            t20 = -60.0 / slope
    parts = path.split(os.sep)
    scene = next((p for p in parts if "-" in p and p[0].isdigit()), "unknown")
    return dict(
        file=os.sep.join(parts[-3:]),
        scene=scene,
        task="task2" if ("male" in parts[-1] or "female" in parts[-1]) else "task1",
        n=n,
        peak=peak,
        gap60=(d60 - peak) if d60 >= 0 else -1,
        d20=d20, d40=d40, d60=d60,
        t20_samples=round(t20, 1) if t20 == t20 else "",
        drr_db=round(10.0 * np.log10(direct / max(rest, EPS)), 1),
        pre_pct=round(pre / total * 100.0, 4),
        tail50_pct=f"{energy[n // 2:].sum() / total * 100.0:.2e}",
    )


def q(vals, p):
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    for p in (args.out_csv, args.out_summary):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite existing output: {p}")

    files = sorted(glob.glob(os.path.join(args.input, "**", "ir*.npy"), recursive=True))
    rows, errors = [], []
    for f in files:
        try:
            rows.append(analyze_one(f))
        except Exception as exc:  # 显式记录,不静默
            errors.append({"file": f, "error": repr(exc)})
    if not rows:
        raise SystemExit("no RIR analyzed; check --input")

    with open(args.out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scenes = sorted({r["scene"] for r in rows})
    lines = [
        "# JAEGER val RIR 审计汇总(脚本 audit_jaeger_rir.py 生成)",
        "",
        f"- 文件总数 {len(rows)};场景数 {len(scenes)};加载失败 {len(errors)}",
        f"- 参数:直达窗 峰前{DIRECT_PRE}/峰后{DIRECT_POST} 采样;主口径 {SR_MAIN:.0f} Hz",
        "",
        "| scene | 条数 | 长度中位(采样) | 峰后→-60dB 中位/最大(采样) | 同(ms@16k) | DRR 中位(dB) | T20 可算条数 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sc in scenes:
        sub = [r for r in rows if r["scene"] == sc]
        gaps = [r["gap60"] for r in sub if r["gap60"] >= 0]
        t20n = sum(1 for r in sub if r["t20_samples"] != "")
        lines.append(
            f"| {sc} | {len(sub)} | {st.median([r['n'] for r in sub]):.0f} "
            f"| {st.median(gaps):.0f} / {max(gaps):.0f} "
            f"| {st.median(gaps)/SR_MAIN*1000:.2f} / {max(gaps)/SR_MAIN*1000:.2f} "
            f"| {st.median([r['drr_db'] for r in sub]):.1f} | {t20n}/{len(sub)} |"
        )
    gaps_all = [r["gap60"] for r in rows if r["gap60"] >= 0]
    drr_all = [r["drr_db"] for r in rows]
    lines += [
        "",
        f"**全体**:峰后→-60dB 中位 {st.median(gaps_all):.0f} 采样"
        f"(={st.median(gaps_all)/SR_MAIN*1000:.2f} ms@16k,"
        f"{st.median(gaps_all)/SR_ALT*1000:.2f} ms@44.1k);"
        f"P95 {q(gaps_all,95):.0f};最大 {max(gaps_all):.0f}。"
        f"DRR 中位 {st.median(drr_all):.1f} dB,min {min(drr_all):.1f} dB。"
        f"T20 可计算 {sum(1 for r in rows if r['t20_samples'] != '')}/{len(rows)} 条。",
    ]
    if errors:
        lines += ["", "## 加载失败清单"] + [f"- {e['file']}: {e['error']}" for e in errors]
    with open(args.out_summary, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"rows={len(rows)} scenes={len(scenes)} errors={len(errors)}")
    print(f"csv={args.out_csv} summary={args.out_summary}")
    if errors:
        print(f"FAIL: {len(errors)} file(s) failed analysis; see summary for the list")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
