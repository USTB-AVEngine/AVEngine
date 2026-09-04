#!/usr/bin/env python3
"""Physical-feature classifier probe (pilot work order item 1.6).

批级认证的第一层探针:只用**最终发布的双耳 wav** 计算物理特征,训练最
简单的分类器,测"只听音频的物理侧信道能把答案猜到多准"。它最便宜、
最能揭穿响度/双耳差/事件密度这类侧信道——探针语义是"捷径强度的下界
估计":简单分类器都能超基线,说明侧信道真实存在;探针没超不代表安全
(还有模型探针两层)。

**oracle 条款(硬边界)**:特征只准从最终发布给模型的音频文件计算。
本模块除标准库与 numpy/soundfile(经同目录 validate_stereo_channels
的读取器)外不 import 任何东西,绝不读取干声素材、真实脉冲响应、
timeline、audio program 或任何引擎侧真值——单测里有对源码的静态自检。

特征(50ms 窗 / 25ms 跳;逐条 wav 一个特征向量;定义与局限如下):
  rms_slope        双声道合能量逐帧 RMS 的线性斜率(渐强/渐弱 → 靠近/远离)
  rms_end_start_db 末 1/4 与首 1/4 能量比(dB)
  ild_mean/ild_slope 逐帧左右能量比(dB)的均值与斜率(方向及其变化)
  itd_samples      GCC-PHAT 全局互相关延迟(采样;正=左超前)
  itd_drift        前半段与后半段延迟差(声源横向移动)
  event_density    相对峰值 −25dB 门限的能量上升沿计数(事件数)
  centroid_mean_hz 频谱质心均值(粗"声纹/音色"代理;**不是**真声纹嵌入)
  decay_proxy      峰值帧后 50ms 能量与峰值帧能量比(混响/衰减的发布端
                   近似;发布混音无法分离直达,**不是**真 DRR)

分类器:标准化后的最近质心(任意类数)与 L2 逻辑回归(二类),分层
k 折交叉验证,**取两者较高者**作为该特征集的探针准确率(探针要尽力
作弊)。输出逐题预测(供认证脚本做聚类 bootstrap)、全特征与逐特征
消融准确率(指认哪条侧信道在泄)、多数类基线。输出 no-clobber。
research_candidate;探针结论只作批级证据,不冒充逐题证书。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_stereo_channels import _read_wav  # noqa: E402  (只读发布 wav)

FRAME_S = 0.050
HOP_S = 0.025
FEATURE_NAMES = ["rms_slope", "rms_end_start_db", "ild_mean", "ild_slope",
                 "itd_samples", "itd_drift", "event_density",
                 "centroid_mean_hz", "decay_proxy"]


def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        return x[None, :] if len(x) else np.zeros((1, 1))
    n = 1 + (len(x) - frame) // hop
    return np.stack([x[i * hop:i * hop + frame] for i in range(n)])


def _gcc_phat_delay(left: np.ndarray, right: np.ndarray, max_lag: int) -> int:
    n = 1
    while n < 2 * len(left):
        n <<= 1
    lf = np.fft.rfft(left, n)
    rf = np.fft.rfft(right, n)
    cross = lf * np.conj(rf)
    cross /= np.maximum(np.abs(cross), 1e-12)
    cc = np.fft.irfft(cross, n)
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    return int(np.argmax(cc)) - max_lag


def extract_features(wav_path: str) -> dict:
    data, rate = _read_wav(wav_path)
    if data.shape[1] != 2:
        raise ValueError(f"{wav_path}: probe requires the released 2-channel audio")
    left, right = data[:, 0], data[:, 1]
    mono = (left + right) / 2.0
    frame, hop = max(1, int(FRAME_S * rate)), max(1, int(HOP_S * rate))
    fl, fr = _frames(left, frame, hop), _frames(right, frame, hop)
    e_l = (fl ** 2).mean(axis=1) + 1e-12
    e_r = (fr ** 2).mean(axis=1) + 1e-12
    e_all = (e_l + e_r) / 2.0
    rms = np.sqrt(e_all)
    t = np.arange(len(rms))
    rms_slope = float(np.polyfit(t, rms, 1)[0]) if len(rms) >= 2 else 0.0
    q = max(1, len(rms) // 4)
    rms_end_start_db = float(10 * np.log10(e_all[-q:].mean() / e_all[:q].mean()))
    ild = 10 * np.log10(e_l / e_r)
    ild_slope = float(np.polyfit(t, ild, 1)[0]) if len(ild) >= 2 else 0.0
    max_lag = max(2, int(0.001 * rate))  # ±1ms,覆盖人头尺度
    half = len(mono) // 2
    itd_a = _gcc_phat_delay(left[:half], right[:half], max_lag)
    itd_b = _gcc_phat_delay(left[half:], right[half:], max_lag)
    itd_full = _gcc_phat_delay(left, right, max_lag)
    peak_db = 10 * np.log10(e_all.max())
    gate = e_all > 10 ** ((peak_db - 25.0) / 10.0)
    rising = int(np.sum(gate[1:] & ~gate[:-1]) + int(gate[0]))
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / rate)
    centroid = float((spec * freqs).sum() / max(spec.sum(), 1e-12))
    pk = int(np.argmax(e_all))
    after = slice(pk + 1, min(len(e_all), pk + 1 + max(1, int(0.050 / HOP_S))))
    decay_proxy = float(e_all[after].mean() / e_all[pk]) if e_all[after].size else 0.0
    return {"rms_slope": rms_slope, "rms_end_start_db": rms_end_start_db,
            "ild_mean": float(ild.mean()), "ild_slope": ild_slope,
            "itd_samples": float(itd_full), "itd_drift": float(itd_b - itd_a),
            "event_density": float(rising), "centroid_mean_hz": centroid,
            "decay_proxy": decay_proxy}


# ---------------- 分类器(numpy 手写,零外部依赖) ----------------

def _standardize(train: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def _nearest_centroid(train_x, train_y, test_x):
    labels = sorted(set(train_y))
    cents = {l: train_x[np.array(train_y) == l].mean(axis=0) for l in labels}
    out = []
    for row in test_x:
        out.append(min(labels, key=lambda l: float(np.linalg.norm(row - cents[l]))))
    return out


def _logreg_binary(train_x, train_y, test_x, iters=300, lr=0.5, l2=1e-2):
    labels = sorted(set(train_y))
    if len(labels) != 2:
        return None
    y = np.array([1.0 if v == labels[1] else 0.0 for v in train_y])
    x = np.hstack([train_x, np.ones((len(train_x), 1))])
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30.0, 30.0)))
        w -= lr * ((x.T @ (p - y)) / len(y) + l2 * w)
    xt = np.hstack([test_x, np.ones((len(test_x), 1))])
    pt = 1.0 / (1.0 + np.exp(-np.clip(xt @ w, -30.0, 30.0)))
    return [labels[1] if v >= 0.5 else labels[0] for v in pt]


def _stratified_folds(labels: list, folds: int, seed: str):
    # 种子必须跨进程可复现:禁用内置 hash(受 PYTHONHASHSEED 随机化影响)
    digest = hashlib.sha256(f"{seed}\0folds".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16))
    assign = np.zeros(len(labels), dtype=int)
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        rng.shuffle(idx)
        for rank, i in enumerate(idx):
            assign[i] = rank % folds
    return assign


def probe(features: np.ndarray, labels: list, *, folds: int = 5, seed: str = "probe"):
    """返回 (cv 准确率, 逐样本预测)。取最近质心与逻辑回归的较高者。"""
    folds = max(2, min(folds, len(labels)))
    fold_of = _stratified_folds(labels, folds, seed)
    preds_nc, preds_lr = [None] * len(labels), [None] * len(labels)
    for k in range(folds):
        tr, te = fold_of != k, fold_of == k
        if te.sum() == 0 or tr.sum() == 0 or len(set(np.array(labels)[tr])) < 2:
            continue
        xtr, xte = _standardize(features[tr], features[te])
        ytr = list(np.array(labels)[tr])
        nc = _nearest_centroid(xtr, ytr, xte)
        lr = _logreg_binary(xtr, ytr, xte)
        for j, i in enumerate(np.where(te)[0]):
            preds_nc[i] = nc[j]
            preds_lr[i] = lr[j] if lr else None
    y = np.array(labels)
    acc_nc = float(np.mean([p == t for p, t in zip(preds_nc, y) if p is not None] or [0.0]))
    lr_pairs = [(p, t) for p, t in zip(preds_lr, y) if p is not None]
    acc_lr = float(np.mean([p == t for p, t in lr_pairs])) if lr_pairs else 0.0
    if acc_lr >= acc_nc:
        return acc_lr, preds_lr
    return acc_nc, preds_nc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True,
                        help="JSON 列表:{question_id, wav, label, group_id?}")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", default="probe_v1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if os.path.exists(args.out):
        print(f"refusing to overwrite existing output: {args.out}", file=sys.stderr)
        return 2
    items = json.load(open(args.items))
    if len(items) < 4:
        print("need at least 4 items to probe", file=sys.stderr)
        return 2

    feats, labels, errors = [], [], []
    for it in items:
        try:
            f = extract_features(it["wav"])
            feats.append([f[k] for k in FEATURE_NAMES])
            labels.append(str(it["label"]))
        except Exception as exc:
            errors.append({"question_id": it.get("question_id"), "error": repr(exc)})
    if errors:
        print(f"FAIL: {len(errors)} item(s) unreadable; first: {errors[0]}", file=sys.stderr)
        return 1
    x = np.array(feats)
    majority = Counter(labels).most_common(1)[0][1] / len(labels)
    acc_all, preds = probe(x, labels, folds=args.folds, seed=args.seed)
    ablation = {}
    for i, name in enumerate(FEATURE_NAMES):
        acc_i, _ = probe(x[:, [i]], labels, folds=args.folds, seed=args.seed)
        ablation[name] = round(acc_i, 4)

    payload = {
        "schema": "avengine_physical_probe_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "seed": args.seed, "folds": args.folds,
        "n": len(labels),
        "label_counts": dict(Counter(labels)),
        "majority_baseline": round(majority, 4),
        "accuracy_all_features": round(acc_all, 4),
        "accuracy_per_feature": ablation,
        "feature_names": FEATURE_NAMES,
        "predictions": [
            {"question_id": it.get("question_id"), "group_id": it.get("group_id"),
             "label": str(it["label"]), "pred": preds[i]}
            for i, it in enumerate(items)
        ],
    }
    with open(args.out, "w") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=1)
    top = max(ablation, key=ablation.get)
    print(f"n={len(labels)} majority={majority:.3f} probe_all={acc_all:.3f} "
          f"strongest_single_feature={top}({ablation[top]:.3f}) out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
